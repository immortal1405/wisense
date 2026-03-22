from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IndexSplit = Dict[str, np.ndarray]


def _safe_stratify_labels(df: pd.DataFrame, columns: Iterable[str]) -> Optional[np.ndarray]:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return None

    strat_labels = df[cols].astype(str).agg("__".join, axis=1)
    counts = strat_labels.value_counts()
    # train_test_split with stratify requires at least 2 samples in each class.
    if (counts < 2).any():
        return None
    return strat_labels.to_numpy()


def build_standard_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_cols: Optional[Iterable[str]] = None,
) -> IndexSplit:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    indices = np.arange(len(df))
    stratify = _safe_stratify_labels(df, stratify_cols or [])

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if val_size == 0:
        return {
            "train": np.asarray(train_val_idx, dtype=np.int64),
            "val": np.asarray([], dtype=np.int64),
            "test": np.asarray(test_idx, dtype=np.int64),
        }

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    stratify_train_val = _safe_stratify_labels(train_val_df, stratify_cols or [])

    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_rel, val_rel = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    train_idx = np.asarray(train_val_idx[train_rel], dtype=np.int64)
    val_idx = np.asarray(train_val_idx[val_rel], dtype=np.int64)

    return {
        "train": train_idx,
        "val": val_idx,
        "test": np.asarray(test_idx, dtype=np.int64),
    }


def build_day_split(
    df: pd.DataFrame,
    train_day: str,
    test_day: str,
    val_size: float,
    random_state: int,
    stratify_cols: Optional[Iterable[str]] = None,
) -> IndexSplit:
    if "day" not in df.columns:
        raise ValueError("day split requested but 'day' column is missing")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")

    day_values = df["day"].astype(str)
    train_val_idx = np.where(day_values == str(train_day))[0]
    test_idx = np.where(day_values == str(test_day))[0]

    if len(train_val_idx) == 0:
        raise ValueError(f"No rows found for train_day={train_day}")
    if len(test_idx) == 0:
        raise ValueError(f"No rows found for test_day={test_day}")

    if val_size == 0:
        return {
            "train": np.asarray(train_val_idx, dtype=np.int64),
            "val": np.asarray([], dtype=np.int64),
            "test": np.asarray(test_idx, dtype=np.int64),
        }

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    stratify = _safe_stratify_labels(train_val_df, stratify_cols or [])

    train_rel, val_rel = train_test_split(
        np.arange(len(train_val_idx)),
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "train": np.asarray(train_val_idx[train_rel], dtype=np.int64),
        "val": np.asarray(train_val_idx[val_rel], dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }
