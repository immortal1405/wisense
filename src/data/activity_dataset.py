from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class NotebookActivityEncoding:
    classes: List[str]
    to_index: Dict[str, int]


class ActivityDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError("Feature and label arrays must have the same length")
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def file_open(path: str | Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as read_file:
        data = json.load(read_file)
    return np.asarray(data)


def load_data_set(path: str | Path) -> np.ndarray:
    data = file_open(path)
    data_to_analyze = []
    for i in range(len(data)):
        line = [data[i][j] for j in range(9)]
        data_to_analyze.append(line)
    return np.asarray(data_to_analyze).T


def extract_data(data: np.ndarray, from_pos: int, to_pos: int, step: int) -> np.ndarray:
    data = data.T
    return np.asarray([data[i : i + step] for i in range(from_pos, to_pos, step)])


def build_train_data_set(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_len = min(len(p1), len(p2), len(p3))
    x: List[np.ndarray] = []
    y: List[int] = []
    for i in range(max_len):
        x.extend([p1[i], p2[i], p3[i]])
        y.extend([0, 1, 2])
    return np.asarray(x), np.asarray(y, dtype=np.int64)


def _reshape_and_scale(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
    test_scaled = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

    # Keras notebook uses (batch, time, features); for PyTorch Conv1d we flip to (batch, features, time).
    train_scaled = train_scaled.reshape(train_scaled.shape[0], train_scaled.shape[1], -1)
    test_scaled = test_scaled.reshape(test_scaled.shape[0], test_scaled.shape[1], -1)

    train_scaled = np.transpose(train_scaled, (0, 2, 1))
    test_scaled = np.transpose(test_scaled, (0, 2, 1))

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def build_activity_dataloaders(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    activity_cfg = config.get("activity", {})

    train_path = data_cfg.get("train_path", "data/data_apml.json")
    test_path = data_cfg.get("test_path", "data/data_apml_min.json")
    window_size = int(data_cfg.get("window_size", 180))

    train_ranges = data_cfg.get(
        "train_ranges",
        [[3100, 10000], [13300, 20000], [23000, 30200]],
    )
    test_ranges = data_cfg.get(
        "test_ranges",
        [[3100, 6300], [10000, 13300], [16100, 19200]],
    )

    class_names = activity_cfg.get("class_order", ["squat", "arm_jerk", "stationary_walk"])
    if len(class_names) != 3:
        raise ValueError("This pipeline expects exactly 3 activity classes")

    dataset_full = load_data_set(train_path)
    dataset_min = load_data_set(test_path)

    class1_full = extract_data(dataset_full, int(train_ranges[0][0]), int(train_ranges[0][1]), window_size)
    class2_full = extract_data(dataset_full, int(train_ranges[1][0]), int(train_ranges[1][1]), window_size)
    class3_full = extract_data(dataset_full, int(train_ranges[2][0]), int(train_ranges[2][1]), window_size)
    train_x, train_y = build_train_data_set(class1_full, class2_full, class3_full)

    class1_min = extract_data(dataset_min, int(test_ranges[0][0]), int(test_ranges[0][1]), window_size)
    class2_min = extract_data(dataset_min, int(test_ranges[1][0]), int(test_ranges[1][1]), window_size)
    class3_min = extract_data(dataset_min, int(test_ranges[2][0]), int(test_ranges[2][1]), window_size)
    test_x, test_y = build_train_data_set(class1_min, class2_min, class3_min)

    train_x, test_x, norm = _reshape_and_scale(train_x, test_x)

    val_size = float(config.get("split", {}).get("val_size", 0.2))
    random_state = int(config.get("split", {}).get("random_state", 42))
    x_train, x_val, y_train, y_val = train_test_split(
        train_x,
        train_y,
        test_size=val_size,
        random_state=random_state,
        stratify=train_y,
    )

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))

    def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = ActivityDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    label_encoding = NotebookActivityEncoding(classes=class_names, to_index={name: idx for idx, name in enumerate(class_names)})

    return {
        "train": make_loader(x_train, y_train, shuffle=True),
        "val": make_loader(x_val, y_val, shuffle=False),
        "test": make_loader(test_x, test_y, shuffle=False),
        "test_loader": make_loader(test_x, test_y, shuffle=False),
        "num_classes": 3,
        "label_encoding": label_encoding,
        "input_shape": {
            "channels": int(train_x.shape[1]),
            "seq_len": int(train_x.shape[2]),
        },
        "normalization": norm,
        "targets": {
            "train": y_train,
            "val": y_val,
            "test": test_y,
        },
        "raw_slices": {
            "train": train_ranges,
            "test": test_ranges,
            "window_size": window_size,
        },
    }
