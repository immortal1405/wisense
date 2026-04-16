from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _make_config_hash(config: Dict[str, Any]) -> str:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    positive_activities = data_cfg.get("positive_activities") or []
    activity_ids = data_cfg.get("activity_ids") or []
    key = {
        "task": str(data_cfg.get("task", "har_fall_binary")),
        "root_dir": str(data_cfg.get("root_dir", "")),
        "positive_activities": sorted(positive_activities),
        "activity_ids": sorted(activity_ids),
        "window_size": int(data_cfg.get("window_size", 256)),
        "stride": int(data_cfg.get("stride", 128)),
        "min_packets": int(data_cfg.get("min_packets", 64)),
        "feature_mode": str(data_cfg.get("feature_mode", "")),
        "label_mode": str(data_cfg.get("label_mode", "activity")),
        "phase_unwrap": bool(data_cfg.get("phase_unwrap", True)),
        "phase_centering": bool(data_cfg.get("phase_centering", True)),
        "subcarrier_count": int(data_cfg.get("subcarrier_count", 30)),
        "motion_detrend": bool(data_cfg.get("motion_detrend", False)),
        "temporal_diff": bool(data_cfg.get("temporal_diff", False)),
        "packet_step": int(data_cfg.get("packet_step", 1)),
        "max_windows_per_trial": int(data_cfg.get("max_windows_per_trial", 0)),
        "max_windows_sampling_mode": str(data_cfg.get("max_windows_sampling_mode", "linspace")),
        "short_sequence_pad_mode": str(data_cfg.get("short_sequence_pad_mode", "reflect")),
        "split_mode": str(split_cfg.get("mode", "")),
        "val_subject_fraction": float(split_cfg.get("val_subject_fraction", 0.2)),
        "split_seed": int(split_cfg.get("seed", 42)),
        "val_subjects": sorted(split_cfg.get("val_subjects", [])),
        "train_subjects": sorted(split_cfg.get("train_subjects", [])),
        "test_subjects": sorted(split_cfg.get("test_subjects", [])),
        "train_envs": sorted(split_cfg.get("train_envs", [])),
        "val_envs": sorted(split_cfg.get("val_envs", [])),
        "test_envs": sorted(split_cfg.get("test_envs", [])),
    }
    text = json.dumps(key, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class PreprocessCache:
    def __init__(self, cache_dir: Path | str = "data/.har_cache"):
        requested = Path(cache_dir)
        try:
            requested.mkdir(parents=True, exist_ok=True)
            self.cache_dir = requested
        except OSError:
            fallback = Path("data/.har_cache")
            fallback.mkdir(parents=True, exist_ok=True)
            self.cache_dir = fallback
            print(f"[cache] falling back to writable cache dir: {self.cache_dir}")

    def _split_path(self, config: Dict[str, Any], split_name: str) -> Path:
        config_hash = _make_config_hash(config)
        return self.cache_dir / f"har_{config_hash}_{split_name}.npz"

    def contains(self, config: Dict[str, Any], split_name: str) -> bool:
        path = self._split_path(config, split_name)
        return path.exists()

    def save(self, config: Dict[str, Any], split_name: str, x: np.ndarray, y: np.ndarray) -> None:
        path = self._split_path(config, split_name)
        np.savez_compressed(path, x=x, y=y)
        print(f"[cache] saved {split_name} to {path}")

    def load(self, config: Dict[str, Any], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self._split_path(config, split_name)
        data = np.load(path)
        print(f"[cache] loaded {split_name} from {path}: x.shape={data['x'].shape} y.shape={data['y'].shape}")
        return data["x"], data["y"]

    def clear(self, config: Dict[str, Any], split_name: Optional[str] = None) -> None:
        if split_name:
            path = self._split_path(config, split_name)
            if path.exists():
                path.unlink()
                print(f"[cache] cleared {path}")
        else:
            config_hash = _make_config_hash(config)
            for path in self.cache_dir.glob(f"har_{config_hash}_*.npz"):
                path.unlink()
            print(f"[cache] cleared all runs with config_hash={config_hash}")
