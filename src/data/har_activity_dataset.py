from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .har_fall_cache import PreprocessCache
from .har_fall_dataset import HARWindowDataset, NAME_RE, TrialMeta, _build_windows, _normalize_train_only, _split_trials, _iter_subject_zips


@dataclass
class ActivityLabelEncoding:
    classes: List[str]
    to_index: Dict[str, int]


def discover_activity_trials(
    root_dir: Path,
    activity_ids: Optional[Sequence[int]] = None,
    label_mode: str = "activity",
    max_files: Optional[int] = None,
) -> Tuple[List[TrialMeta], ActivityLabelEncoding, Dict[Tuple[int, int], int]]:
    selected = None if activity_ids is None else set(int(v) for v in activity_ids)

    raw: List[Tuple[Path, str, int, int, int, int, int]] = []
    found_activities: set[int] = set()

    for zip_path in _iter_subject_zips(root_dir):
        with ZipFile(zip_path, "r") as zf:
            for member_name in zf.namelist():
                base = Path(member_name).name
                match = NAME_RE.match(base)
                if match is None:
                    continue

                env = int(match.group("env"))
                subject = int(match.group("subject"))
                experiment = int(match.group("exp"))
                activity = int(match.group("activity"))
                trial = int(match.group("trial"))

                if selected is not None and activity not in selected:
                    continue

                raw.append((zip_path, member_name, env, subject, experiment, activity, trial))
                found_activities.add(activity)

                if max_files is not None and len(raw) >= max_files:
                    break
            if max_files is not None and len(raw) >= max_files:
                break

    if not raw:
        raise RuntimeError(f"No matching activity trials found under: {root_dir}")

    mode = str(label_mode).lower()
    if mode == "activity":
        label_keys = sorted((0, act) for act in found_activities)
        classes = [f"A{act}" for _, act in label_keys]
    elif mode in {"experiment_activity", "exp_activity", "composite"}:
        label_keys = sorted({(exp, activity) for _, _, _, _, exp, activity, _ in raw})
        classes = [f"C{exp}_A{activity}" for exp, activity in label_keys]
    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    label_key_to_index = {key: i for i, key in enumerate(label_keys)}

    trials: List[TrialMeta] = []
    for zip_path, member_name, env, subject, experiment, activity, trial in raw:
        trials.append(
            TrialMeta(
                zip_path=zip_path,
                member_name=member_name,
                env=env,
                subject=subject,
                experiment=experiment,
                activity=activity,
                trial=trial,
                label=int(label_key_to_index[(0, activity)] if mode == "activity" else label_key_to_index[(experiment, activity)]),
            )
        )

    return trials, ActivityLabelEncoding(classes=classes, to_index={c: i for i, c in enumerate(classes)}), label_key_to_index


def build_har_activity_dataloaders(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    train_cfg = config.get("train", {})

    root_dir = Path(data_cfg.get("root_dir", "data/v38wjmz6f6-1"))
    activity_ids = data_cfg.get("activity_ids")
    max_files = data_cfg.get("max_files")
    window_size = int(data_cfg.get("window_size", 256))
    stride = int(data_cfg.get("stride", 128))
    min_packets = int(data_cfg.get("min_packets", 64))
    feature_mode = str(data_cfg.get("feature_mode", "amp_log_sincos")).lower()
    motion_detrend = bool(data_cfg.get("motion_detrend", False))
    temporal_diff = bool(data_cfg.get("temporal_diff", False))
    packet_step = int(data_cfg.get("packet_step", 1))
    max_windows_per_trial = int(data_cfg.get("max_windows_per_trial", 0))
    max_windows_sampling_mode = str(data_cfg.get("max_windows_sampling_mode", "linspace"))
    sampling_seed = int(split_cfg.get("seed", config.get("seed", 42)))
    phase_unwrap = bool(data_cfg.get("phase_unwrap", True))
    phase_centering = bool(data_cfg.get("phase_centering", True))
    subcarrier_count = int(data_cfg.get("subcarrier_count", 30))
    short_sequence_pad_mode = str(data_cfg.get("short_sequence_pad_mode", "reflect"))
    preprocess_workers = int(data_cfg.get("preprocess_workers", 8))
    label_mode = str(data_cfg.get("label_mode", "experiment_activity")).lower()
    use_cache = bool(data_cfg.get("use_cache", True))
    balanced_sampling = bool(data_cfg.get("balanced_sampling", False))

    cache = PreprocessCache(cache_dir=data_cfg.get("cache_dir", "data/.har_cache")) if use_cache else None

    trials, label_encoding, label_key_to_index = discover_activity_trials(
        root_dir=root_dir,
        activity_ids=activity_ids,
        label_mode=label_mode,
        max_files=max_files,
    )

    if len(trials) == 0:
        raise RuntimeError(f"No trials discovered under: {root_dir}")
    print(f"[data] discovered {len(trials)} activity trials")

    train_trials, val_trials, test_trials = _split_trials(trials, split_cfg=split_cfg)
    if len(train_trials) == 0 or len(val_trials) == 0 or len(test_trials) == 0:
        raise RuntimeError("Split produced an empty train/val/test partition")
    print(
        "[data] split trials "
        f"train={len(train_trials)} val={len(val_trials)} test={len(test_trials)}"
    )

    if cache and cache.contains(config, "train"):
        x_train, y_train = cache.load(config, "train")
    else:
        x_train, y_train = _build_windows(
            train_trials,
            window_size=window_size,
            stride=stride,
            min_packets=min_packets,
            feature_mode=feature_mode,
            motion_detrend=motion_detrend,
            temporal_diff=temporal_diff,
            packet_step=packet_step,
            max_windows_per_trial=max_windows_per_trial,
            max_windows_sampling_mode=max_windows_sampling_mode,
            sampling_seed=sampling_seed,
            phase_unwrap=phase_unwrap,
            phase_centering=phase_centering,
            subcarrier_count=subcarrier_count,
            short_sequence_pad_mode=short_sequence_pad_mode,
            preprocess_workers=preprocess_workers,
        )
        if cache:
            cache.save(config, "train", x_train, y_train)

    if cache and cache.contains(config, "val"):
        x_val, y_val = cache.load(config, "val")
    else:
        x_val, y_val = _build_windows(
            val_trials,
            window_size=window_size,
            stride=stride,
            min_packets=min_packets,
            feature_mode=feature_mode,
            motion_detrend=motion_detrend,
            temporal_diff=temporal_diff,
            packet_step=packet_step,
            max_windows_per_trial=max_windows_per_trial,
            max_windows_sampling_mode=max_windows_sampling_mode,
            sampling_seed=sampling_seed,
            phase_unwrap=phase_unwrap,
            phase_centering=phase_centering,
            subcarrier_count=subcarrier_count,
            short_sequence_pad_mode=short_sequence_pad_mode,
            preprocess_workers=preprocess_workers,
        )
        if cache:
            cache.save(config, "val", x_val, y_val)

    if cache and cache.contains(config, "test"):
        x_test, y_test = cache.load(config, "test")
    else:
        x_test, y_test = _build_windows(
            test_trials,
            window_size=window_size,
            stride=stride,
            min_packets=min_packets,
            feature_mode=feature_mode,
            motion_detrend=motion_detrend,
            temporal_diff=temporal_diff,
            packet_step=packet_step,
            max_windows_per_trial=max_windows_per_trial,
            max_windows_sampling_mode=max_windows_sampling_mode,
            sampling_seed=sampling_seed,
            phase_unwrap=phase_unwrap,
            phase_centering=phase_centering,
            subcarrier_count=subcarrier_count,
            short_sequence_pad_mode=short_sequence_pad_mode,
            preprocess_workers=preprocess_workers,
        )
        if cache:
            cache.save(config, "test", x_test, y_test)

    x_train, x_val, x_test, norm = _normalize_train_only(x_train, x_val, x_test)

    num_classes = len(label_encoding.classes)
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool, use_balanced_sampler: bool = False) -> DataLoader:
        ds = HARWindowDataset(x, y)

        if use_balanced_sampler:
            counts = np.bincount(y, minlength=num_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            class_weight = 1.0 / counts
            sample_weight = class_weight[y]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weight).double(),
                num_samples=len(y),
                replacement=True,
            )
            return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def count_by_class(y: np.ndarray) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for class_name, idx in label_encoding.to_index.items():
            out[class_name] = int((y == idx).sum())
        return out

    index_to_key = {idx: key for key, idx in label_key_to_index.items()}

    metadata = {
        "split_mode": str(split_cfg.get("mode", "environment")).lower(),
        "feature_mode": feature_mode,
        "label_mode": label_mode,
        "motion_detrend": motion_detrend,
        "temporal_diff": temporal_diff,
        "packet_step": packet_step,
        "max_windows_per_trial": max_windows_per_trial,
        "max_windows_sampling_mode": max_windows_sampling_mode,
        "phase_unwrap": phase_unwrap,
        "phase_centering": phase_centering,
        "subcarrier_count": subcarrier_count,
        "short_sequence_pad_mode": short_sequence_pad_mode,
        "num_classes": num_classes,
        "classes": label_encoding.classes,
        "class_to_activity": {name: int(index_to_key[idx][1]) for name, idx in label_encoding.to_index.items()},
        "class_to_experiment": {
            name: (int(index_to_key[idx][0]) if index_to_key[idx][0] != 0 else None)
            for name, idx in label_encoding.to_index.items()
        },
        "num_trials_total": len(trials),
        "num_trials_train": len(train_trials),
        "num_trials_val": len(val_trials),
        "num_trials_test": len(test_trials),
        "num_windows_train": int(len(y_train)),
        "num_windows_val": int(len(y_val)),
        "num_windows_test": int(len(y_test)),
        "label_counts_train": count_by_class(y_train),
        "label_counts_val": count_by_class(y_val),
        "label_counts_test": count_by_class(y_test),
    }

    return {
        "train": make_loader(x_train, y_train, shuffle=not balanced_sampling, use_balanced_sampler=balanced_sampling),
        "val": make_loader(x_val, y_val, shuffle=False),
        "test": make_loader(x_test, y_test, shuffle=False),
        "num_classes": num_classes,
        "label_encoding": label_encoding,
        "input_shape": {
            "channels": int(x_train.shape[1]),
            "seq_len": int(x_train.shape[2]),
        },
        "targets": {
            "train": y_train,
            "val": y_val,
            "test": y_test,
        },
        "normalization": norm,
        "metadata": metadata,
    }
