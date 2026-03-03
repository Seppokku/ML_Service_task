from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_split_ratios(
    train_ratio: float, valid_ratio: float, test_ratio: float
) -> Tuple[float, float, float]:
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0):
        raise ValueError("Split ratios must be positive values.")
    ratio_sum = ratios.sum()
    if not np.isfinite(ratio_sum) or ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    if not np.isclose(ratio_sum, 1.0):
        ratios = ratios / ratio_sum
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def stratified_train_valid_test_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Dataset must contain {label_col} column.")

    train_ratio, valid_ratio, test_ratio = _normalize_split_ratios(
        train_ratio, valid_ratio, test_ratio
    )

    labels = df[label_col].astype(str)
    class_counts = labels.value_counts(dropna=False)
    if class_counts.min() < 3:
        raise ValueError(
            "Not enough samples per class to build train/valid/test stratified splits."
        )

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    valid_share = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_share,
        random_state=seed,
        stratify=temp_df[label_col].astype(str),
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
