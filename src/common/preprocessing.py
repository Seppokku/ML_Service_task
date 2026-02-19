from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

RAW_FEATURES: list[str] = [
    "meetings_count",
    "meetings_minutes",
    "after_hours_ratio",
    "commits_count",
    "active_days",
    "tasks_completed",
    "tasks_reopened",
    "messages_count",
    "context_switches",
    "deep_work_minutes",
]

DERIVED_FEATURES: list[str] = [
    "reopen_ratio",
    "meeting_load",
    "commit_intensity",
    "messages_per_day",
    "context_switches_per_day",
    "deep_work_ratio",
]

NUMERIC_FEATURES: list[str] = RAW_FEATURES + DERIVED_FEATURES
CATEGORICAL_FEATURES: list[str] = [
    "weekday",
    "month",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, RAW_FEATURES)

    features = df.copy()
    for col in RAW_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
        features[col] = features[col].clip(lower=0)

    active_days = features["active_days"] + 1.0
    tasks_completed = features["tasks_completed"] + 1.0

    features["reopen_ratio"] = features["tasks_reopened"] / tasks_completed
    features["meeting_load"] = features["meetings_minutes"] / active_days
    features["commit_intensity"] = features["commits_count"] / active_days
    features["messages_per_day"] = features["messages_count"] / active_days
    features["context_switches_per_day"] = features["context_switches"] / active_days
    features["deep_work_ratio"] = features["deep_work_minutes"] / (active_days * 60.0)

    if "period_start" in features.columns:
        dt = pd.to_datetime(features["period_start"], errors="coerce")
        features["weekday"] = dt.dt.day_name()
        month = dt.dt.month.fillna(0).astype(int).astype(str)
        features["month"] = month.replace("0", "unknown")
    else:
        features["weekday"] = "unknown"
        features["month"] = "unknown"

    features["weekday"] = features["weekday"].fillna("unknown")
    features["month"] = features["month"].fillna("unknown").astype(str)

    features.replace([np.inf, -np.inf], 0, inplace=True)
    return features[FEATURE_COLUMNS]


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS.copy()


def get_numeric_features() -> list[str]:
    return NUMERIC_FEATURES.copy()


def get_categorical_features() -> list[str]:
    return CATEGORICAL_FEATURES.copy()
