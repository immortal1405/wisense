from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile

from .har_fall_cache import PreprocessCache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


NAME_RE = re.compile(r"^E(?P<env>\d+)_S(?P<subject>\d+)_C(?P<exp>\d+)_A(?P<activity>\d+)_T(?P<trial>\d+)\.csv$")
CSI_RE = re.compile(r"^csi_(?P<tx>\d+)_(?P<rx>\d+)_(?P<sc>\d+)$")


@dataclass
class FallLabelEncoding:
    classes: List[str]
    to_index: Dict[str, int]


@dataclass
class TrialMeta:
    zip_path: Path
    member_name: str
    env: int
    subject: int
    experiment: int
    activity: int
    trial: int
    label: int


class HARWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError("Feature and label arrays must have the same length")
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _iter_subject_zips(root_dir: Path) -> Iterable[Path]:
    for env_dir in sorted(root_dir.glob("Environment *")):
        if not env_dir.is_dir():
            continue
        for zip_path in sorted(env_dir.glob("Subject *.zip")):
            yield zip_path


def discover_trials(root_dir: Path, positive_activities: Sequence[int], max_files: Optional[int] = None) -> List[TrialMeta]:
    pos_set = set(int(v) for v in positive_activities)
    trials: List[TrialMeta] = []

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
                label = 1 if activity in pos_set else 0

                trials.append(
                    TrialMeta(
                        zip_path=zip_path,
                        member_name=member_name,
                        env=env,
                        subject=subject,
                        experiment=experiment,
                        activity=activity,
                        trial=trial,
                        label=label,
                    )
                )

                if max_files is not None and len(trials) >= max_files:
                    return trials

    return trials


def _sorted_csi_columns(columns: Sequence[str]) -> List[str]:
    parsed = []
    for col in columns:
        m = CSI_RE.match(col)
        if m is None:
            continue
        parsed.append((int(m.group("tx")), int(m.group("rx")), int(m.group("sc")), col))

    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in parsed]


def _complex_from_strings(values: np.ndarray) -> np.ndarray:
    # Dataset strings are formatted like "15+15i" and "23+-7i".
    txt = np.char.replace(values.astype(str), "i", "j")
    txt = np.char.replace(txt, "+-", "-")
    return txt.astype(np.complex64)


def _read_trial_csi(trial: TrialMeta) -> np.ndarray:
    with ZipFile(trial.zip_path, "r") as zf:
        with zf.open(trial.member_name, "r") as fh:
            raw = fh.read()

    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c.startswith("csi_"))
    csi_cols = _sorted_csi_columns(df.columns.tolist())
    if len(csi_cols) == 0:
        raise ValueError(f"No CSI columns found in {trial.member_name}")

    csi_complex = _complex_from_strings(df[csi_cols].to_numpy())
    return csi_complex


def _calibrate_phase(
    phase: np.ndarray,
    phase_unwrap: bool,
    phase_centering: bool,
    subcarrier_count: int,
) -> np.ndarray:
    if phase.shape[1] % subcarrier_count != 0:
        return phase

    grouped = phase.reshape(phase.shape[0], phase.shape[1] // subcarrier_count, subcarrier_count)
    if phase_unwrap:
        grouped = np.unwrap(grouped, axis=2)
    if phase_centering:
        grouped = grouped - grouped.mean(axis=2, keepdims=True)

    return grouped.reshape(phase.shape)


def _transform_csi_features(
    csi_complex: np.ndarray,
    feature_mode: str,
    phase_unwrap: bool,
    phase_centering: bool,
    subcarrier_count: int,
) -> np.ndarray:
    amp = np.abs(csi_complex)
    phase = np.angle(csi_complex)
    phase = _calibrate_phase(
        phase=phase,
        phase_unwrap=phase_unwrap,
        phase_centering=phase_centering,
        subcarrier_count=subcarrier_count,
    )

    if feature_mode == "amp_phase":
        return np.concatenate([amp, phase], axis=1).astype(np.float32)

    if feature_mode == "amp_sincos":
        return np.concatenate([amp, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)

    if feature_mode == "amp_log_sincos":
        amp_log = np.log1p(amp)
        return np.concatenate([amp_log, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def _apply_motion_preprocess(sequence: np.ndarray, motion_detrend: bool, temporal_diff: bool) -> np.ndarray:
    seq = sequence

    if motion_detrend:
        # Remove trial-level static bias so model focuses on motion-induced variation.
        seq = seq - seq.mean(axis=0, keepdims=True)

    if temporal_diff:
        # High-pass in time while keeping the original sequence length.
        seq = np.diff(seq, axis=0, prepend=seq[:1, :])

    return seq.astype(np.float32)


def _windowize(sequence: np.ndarray, window_size: int, stride: int, min_packets: int, short_sequence_pad_mode: str) -> np.ndarray:
    num_packets, num_feats = sequence.shape
    if num_packets < min_packets:
        return np.empty((0, num_feats, window_size), dtype=np.float32)

    if num_packets < window_size:
        pad_count = window_size - num_packets
        pad_mode = short_sequence_pad_mode.lower()
        if pad_mode == "edge":
            tail = np.repeat(sequence[-1:, :], pad_count, axis=0)
            padded = np.concatenate([sequence, tail], axis=0)
        elif pad_mode == "reflect":
            padded = np.pad(sequence, ((0, pad_count), (0, 0)), mode="reflect")
        elif pad_mode == "zero":
            padded = np.pad(sequence, ((0, pad_count), (0, 0)), mode="constant", constant_values=0.0)
        else:
            raise ValueError(f"Unsupported short_sequence_pad_mode: {short_sequence_pad_mode}")
        return np.transpose(padded[None, :, :], (0, 2, 1)).astype(np.float32)

    windows = []
    for start in range(0, num_packets - window_size + 1, stride):
        chunk = sequence[start : start + window_size, :]
        windows.append(np.transpose(chunk, (1, 0)))

    if not windows:
        chunk = sequence[-window_size:, :]
        windows = [np.transpose(chunk, (1, 0))]

    return np.asarray(windows, dtype=np.float32)


def _split_trials(
    trials: Sequence[TrialMeta],
    split_cfg: Dict[str, Any],
) -> Tuple[List[TrialMeta], List[TrialMeta], List[TrialMeta]]:
    mode = str(split_cfg.get("mode", "environment")).lower()

    if mode == "environment":
        test_envs = set(int(x) for x in split_cfg.get("test_envs", [3]))
        val_envs = set(int(x) for x in split_cfg.get("val_envs", []))
        train_envs = set(int(x) for x in split_cfg.get("train_envs", [1, 2]))

        # Never allow overlap between train/val/test environments.
        train_envs = train_envs - test_envs - val_envs

        train = [t for t in trials if t.env in train_envs]
        test = [t for t in trials if t.env in test_envs]

        if val_envs:
            val = [t for t in trials if t.env in val_envs]
            return train, val, test

        # If val_envs is not specified, split validation by subject from train envs.
        val_subject_fraction = float(split_cfg.get("val_subject_fraction", 0.2))
        train_subjects = sorted({t.subject for t in train})
        if not train_subjects:
            return [], [], test

        num_val_subjects = max(1, int(round(len(train_subjects) * val_subject_fraction)))
        explicit_val_subjects = split_cfg.get("val_subjects", [])
        if explicit_val_subjects:
            val_subjects = set(int(s) for s in explicit_val_subjects)
        else:
            split_seed = int(split_cfg.get("seed", 42))
            rng = np.random.default_rng(split_seed)
            sampled = rng.choice(np.asarray(train_subjects, dtype=np.int64), size=num_val_subjects, replace=False)
            val_subjects = set(int(v) for v in sampled.tolist())

        val = [t for t in train if t.subject in val_subjects]
        train = [t for t in train if t.subject not in val_subjects]
        return train, val, test

    if mode == "subject":
        test_subjects = set(int(x) for x in split_cfg.get("test_subjects", [25, 26, 27, 28, 29, 30]))
        val_subjects = set(int(x) for x in split_cfg.get("val_subjects", [21, 22, 23, 24]))
        train = [t for t in trials if t.subject not in test_subjects and t.subject not in val_subjects]
        val = [t for t in trials if t.subject in val_subjects]
        test = [t for t in trials if t.subject in test_subjects]
        return train, val, test

    raise ValueError(f"Unsupported split.mode: {mode}")


def _build_windows(
    trials: Sequence[TrialMeta],
    window_size: int,
    stride: int,
    min_packets: int,
    feature_mode: str,
    motion_detrend: bool,
    temporal_diff: bool,
    packet_step: int,
    max_windows_per_trial: int,
    max_windows_sampling_mode: str,
    sampling_seed: int,
    phase_unwrap: bool,
    phase_centering: bool,
    subcarrier_count: int,
    short_sequence_pad_mode: str,
    preprocess_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    def process_one(trial: TrialMeta) -> Tuple[np.ndarray, np.ndarray]:
        csi_complex = _read_trial_csi(trial)
        seq = _transform_csi_features(
            csi_complex,
            feature_mode=feature_mode,
            phase_unwrap=phase_unwrap,
            phase_centering=phase_centering,
            subcarrier_count=subcarrier_count,
        )
        seq = _apply_motion_preprocess(seq, motion_detrend=motion_detrend, temporal_diff=temporal_diff)
        if packet_step > 1:
            seq = seq[::packet_step]

        windows = _windowize(
            seq,
            window_size=window_size,
            stride=stride,
            min_packets=min_packets,
            short_sequence_pad_mode=short_sequence_pad_mode,
        )
        if max_windows_per_trial > 0 and len(windows) > max_windows_per_trial:
            sampling_mode = max_windows_sampling_mode.lower()
            if sampling_mode == "linspace":
                idx = np.linspace(0, len(windows) - 1, num=max_windows_per_trial, dtype=np.int64)
            elif sampling_mode == "random":
                trial_seed = (
                    int(sampling_seed)
                    + int(trial.env) * 1_000_003
                    + int(trial.subject) * 100_003
                    + int(trial.experiment) * 10_007
                    + int(trial.activity) * 1_009
                    + int(trial.trial) * 101
                )
                rng = np.random.default_rng(trial_seed)
                idx = np.sort(rng.choice(len(windows), size=max_windows_per_trial, replace=False).astype(np.int64))
            else:
                raise ValueError(f"Unsupported max_windows_sampling_mode: {max_windows_sampling_mode}")
            windows = windows[idx]

        labels = np.full((len(windows),), int(trial.label), dtype=np.int64)
        return windows, labels

    total_trials = len(trials)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, preprocess_workers)) as ex:
        futures = [ex.submit(process_one, t) for t in trials]
        for fut in as_completed(futures):
            windows, labels = fut.result()
            done += 1

            if len(windows) > 0:
                x_list.append(windows)
                y_list.append(labels)

            if done == 1 or done % 200 == 0 or done == total_trials:
                print(f"[preprocess] processed trials: {done}/{total_trials}")

    if not x_list:
        raise RuntimeError("No windows were built from the selected trials")

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y


def _normalize_train_only(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    # Per-feature normalization over all training windows and time steps.
    # x shape: (N, C, L)
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_n = ((x_train - mean) / std).astype(np.float32)
    x_val_n = ((x_val - mean) / std).astype(np.float32)
    x_test_n = ((x_test - mean) / std).astype(np.float32)

    norm = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }
    return x_train_n, x_val_n, x_test_n, norm


def build_har_fall_dataloaders(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    train_cfg = config.get("train", {})

    root_dir = Path(data_cfg.get("root_dir", "data/v38wjmz6f6-1"))
    positive_activities = data_cfg.get("positive_activities", [2, 5])
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
    use_cache = bool(data_cfg.get("use_cache", True))
    balanced_sampling = bool(data_cfg.get("balanced_sampling", False))

    cache = PreprocessCache(cache_dir=data_cfg.get("cache_dir", "data/.har_cache")) if use_cache else None

    trials = discover_trials(root_dir, positive_activities=positive_activities, max_files=max_files)
    if len(trials) == 0:
        raise RuntimeError(f"No trials discovered under: {root_dir}")
    print(f"[data] discovered {len(trials)} trial files")

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

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool, use_balanced_sampler: bool = False) -> DataLoader:
        ds = HARWindowDataset(x, y)
        if use_balanced_sampler:
            counts = np.bincount(y, minlength=2).astype(np.float64)
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

    label_encoding = FallLabelEncoding(classes=["non_fall", "fall"], to_index={"non_fall": 0, "fall": 1})

    split_mode = str(split_cfg.get("mode", "environment")).lower()
    metadata = {
        "split_mode": split_mode,
        "feature_mode": feature_mode,
        "motion_detrend": motion_detrend,
        "temporal_diff": temporal_diff,
        "packet_step": packet_step,
        "max_windows_per_trial": max_windows_per_trial,
        "max_windows_sampling_mode": max_windows_sampling_mode,
        "phase_unwrap": phase_unwrap,
        "phase_centering": phase_centering,
        "subcarrier_count": subcarrier_count,
        "short_sequence_pad_mode": short_sequence_pad_mode,
        "num_trials_total": len(trials),
        "num_trials_train": len(train_trials),
        "num_trials_val": len(val_trials),
        "num_trials_test": len(test_trials),
        "num_windows_train": int(len(y_train)),
        "num_windows_val": int(len(y_val)),
        "num_windows_test": int(len(y_test)),
        "label_counts_train": {"non_fall": int((y_train == 0).sum()), "fall": int((y_train == 1).sum())},
        "label_counts_val": {"non_fall": int((y_val == 0).sum()), "fall": int((y_val == 1).sum())},
        "label_counts_test": {"non_fall": int((y_test == 0).sum()), "fall": int((y_test == 1).sum())},
    }

    return {
        "train": make_loader(x_train, y_train, shuffle=not balanced_sampling, use_balanced_sampler=balanced_sampling),
        "val": make_loader(x_val, y_val, shuffle=False),
        "test": make_loader(x_test, y_test, shuffle=False),
        "num_classes": 2,
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