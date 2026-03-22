from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .splits import build_day_split, build_standard_split


@dataclass
class LabelEncoding:
    name: str
    classes: List[str]
    to_index: Dict[str, int]


def load_csi_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    if dataset_path.suffix == ".pkl":
        df = pd.read_pickle(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    return df.copy()


def _sorted_feature_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No feature columns found for prefix '{prefix}'")
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def _build_label_encoding(series: pd.Series, ordered_classes: Optional[List[str]] = None) -> LabelEncoding:
    values = series.astype(str)
    if ordered_classes is None:
        classes = sorted(values.unique().tolist())
    else:
        present = set(values.unique().tolist())
        classes = [c for c in ordered_classes if c in present]
    to_index = {label: idx for idx, label in enumerate(classes)}
    return LabelEncoding(name=series.name, classes=classes, to_index=to_index)


def _encode_series(series: pd.Series, encoding: LabelEncoding) -> np.ndarray:
    values = series.astype(str).map(encoding.to_index)
    if values.isnull().any():
        missing = series.astype(str)[values.isnull()].unique().tolist()
        raise ValueError(f"Found unseen labels for {series.name}: {missing}")
    return values.to_numpy(dtype=np.int64)


def _prepare_type_target(df: pd.DataFrame, drop_empty: bool = True) -> pd.DataFrame:
    type_values = df["type"].astype(str).str.lower()
    normalized = type_values.replace({"metallic": "metalic"})
    work = df.copy()
    work["type"] = normalized

    if drop_empty:
        work = work[work["type"].isin(["metalic", "organic"])].reset_index(drop=True)

    return work


def _build_input_array(df: pd.DataFrame, amp_cols: Iterable[str], phase_cols: Iterable[str]) -> np.ndarray:
    amp = df[list(amp_cols)].to_numpy(dtype=np.float32)
    phase = df[list(phase_cols)].to_numpy(dtype=np.float32)
    # CNN input shape: (batch, channels=2, sequence_length=52)
    x = np.stack([amp, phase], axis=1)
    return x


class CSIDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError("Feature and label arrays must have the same length")
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def build_dataloaders(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    train_cfg = config.get("train", {})

    dataset_path = data_cfg.get("dataset_path", "data/dataset_normalized.pkl")
    target = data_cfg.get("target", "type")
    drop_empty_for_type = bool(data_cfg.get("drop_empty_for_type", True))

    df = load_csi_dataframe(dataset_path)
    if target == "type":
        df = _prepare_type_target(df, drop_empty=drop_empty_for_type)

    amp_cols = _sorted_feature_columns(df, "amp_")
    phase_cols = _sorted_feature_columns(df, "phase_")
    if len(amp_cols) != len(phase_cols):
        raise ValueError("amp_* and phase_* columns have mismatched lengths")

    x_all = _build_input_array(df, amp_cols, phase_cols)

    if target == "type":
        label_encoding = _build_label_encoding(df[target], ordered_classes=["organic", "metalic"])
    else:
        label_encoding = _build_label_encoding(df[target])
    y_all = _encode_series(df[target], label_encoding)

    mode = split_cfg.get("mode", "standard")
    if mode == "standard":
        split_indices = build_standard_split(
            df=df,
            test_size=float(split_cfg.get("test_size", 0.15)),
            val_size=float(split_cfg.get("val_size", 0.15)),
            random_state=int(split_cfg.get("random_state", 42)),
            stratify_cols=split_cfg.get("stratify_cols", [target]),
        )
    elif mode == "day":
        split_indices = build_day_split(
            df=df,
            train_day=str(split_cfg.get("train_day", "1")),
            test_day=str(split_cfg.get("test_day", "2")),
            val_size=float(split_cfg.get("val_size", 0.15)),
            random_state=int(split_cfg.get("random_state", 42)),
            stratify_cols=split_cfg.get("stratify_cols", [target]),
        )
    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        subset_x = x_all[indices]
        subset_y = y_all[indices]
        ds = CSIDataset(subset_x, subset_y)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(split_indices["train"], shuffle=True)
    val_loader = make_loader(split_indices["val"], shuffle=False) if len(split_indices["val"]) else None
    test_loader = make_loader(split_indices["test"], shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": len(label_encoding.classes),
        "label_encoding": label_encoding,
        "feature_info": {
            "amp_cols": amp_cols,
            "phase_cols": phase_cols,
            "sequence_length": len(amp_cols),
            "channels": 2,
        },
        "split_sizes": {k: int(len(v)) for k, v in split_indices.items()},
    }
