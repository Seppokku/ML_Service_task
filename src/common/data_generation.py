from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_stratified(
    df: pd.DataFrame,
    label_col: str = "burnout_label",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df[label_col],
        random_state=seed,
    )
    valid_share = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_share),
        stratify=temp_df[label_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _inject_outliers(
    rng: np.random.Generator,
    values: np.ndarray,
    prob: float = 0.03,
    multiplier_range: Tuple[float, float] = (2.0, 4.0),
    max_value: Optional[float] = None,
) -> np.ndarray:
    values = values.astype(float)
    mask = rng.random(values.shape[0]) < prob
    if mask.any():
        factors = rng.uniform(multiplier_range[0], multiplier_range[1], size=values.shape[0])
        values[mask] = values[mask] * factors[mask]
    if max_value is not None:
        values = np.clip(values, 0, max_value)
    return values


def generate_synthetic_data(
    path: Path,
    rows: int = 500,
    seed: int = 42,
    label_positive_rate: float = 0.18,
    label_noise_std: float = 0.12,
    label_sharpness: float = 2.0,
    label_signal_scale: float = 1.15,
) -> Path:
    rng = np.random.default_rng(seed)

    team_ids = [f"team-{i}" for i in range(1, 11)]
    base_date = date.today() - timedelta(days=rows)
    t = np.arange(rows)

    # сезонность + скрытая "нагрузка", чтобы связать фичи между собой
    seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * t / 30) + 0.1 * np.sin(2 * np.pi * t / 7)
    team_biases = rng.normal(0, 0.25, size=len(team_ids))
    team_index = rng.integers(0, len(team_ids), size=rows)
    team_bias = team_biases[team_index]
    workload = rng.normal(0, 1, size=rows) + 0.6 * team_bias + 0.3 * (seasonality - 1)
    intensity = np.clip(1.0 + 0.35 * workload, 0.3, 2.5)

    active_days = rng.integers(1, 8, size=rows)
    low_activity = rng.random(rows) < 0.06
    active_days = np.where(low_activity, rng.integers(1, 3, size=rows), active_days)

    meetings_count = rng.normal(6.5 * intensity + 1.0, 2.5, size=rows)
    meetings_count = _inject_outliers(
        rng, meetings_count, prob=0.02, multiplier_range=(1.5, 2.4), max_value=60
    )
    meetings_count = np.clip(np.round(meetings_count), 0, 60).astype(int)

    meeting_duration = rng.lognormal(mean=3.1, sigma=0.45, size=rows)
    meetings_minutes = meetings_count * meeting_duration
    meetings_minutes = _inject_outliers(
        rng, meetings_minutes, prob=0.03, multiplier_range=(1.8, 3.5), max_value=1800
    )

    after_hours_ratio = rng.beta(2.2, 6.2, size=rows)
    after_hours_ratio = after_hours_ratio + np.clip((intensity - 1.0) * 0.08, -0.1, 0.25)
    high_after_hours = rng.random(rows) < 0.03
    after_hours_ratio[high_after_hours] = rng.uniform(0.7, 0.98, size=high_after_hours.sum())
    after_hours_ratio = np.clip(after_hours_ratio, 0, 1)

    commits_lam = 5 + 1.1 * active_days + 1.8 * intensity
    commits_count = rng.poisson(commits_lam).astype(float)
    zero_commits_prob = 0.08 + 0.06 * (1 - active_days / 7) + low_activity * 0.08
    zero_commits = rng.random(rows) < zero_commits_prob
    commits_count[zero_commits] = 0
    commits_count = _inject_outliers(
        rng, commits_count, prob=0.03, multiplier_range=(2.0, 4.0), max_value=140
    )
    commits_count = np.clip(np.round(commits_count), 0, 140).astype(int)

    tasks_completed = rng.normal(
        9 + 1.3 * active_days + 0.25 * commits_count + 2.5 * intensity, 5.5, size=rows
    )
    tasks_completed = _inject_outliers(
        rng, tasks_completed, prob=0.02, multiplier_range=(1.5, 2.0), max_value=100
    )
    tasks_completed = np.clip(np.round(tasks_completed), 0, 100).astype(int)

    messages_base = rng.lognormal(
        mean=3.6 + 0.03 * meetings_count + 0.15 * intensity, sigma=0.55, size=rows
    )
    messages_count = messages_base + 2.5 * meetings_count + rng.normal(0, 10, size=rows)
    messages_count = _inject_outliers(
        rng, messages_count, prob=0.03, multiplier_range=(1.8, 3.0), max_value=2000
    )
    zero_messages = rng.random(rows) < (0.03 + 0.04 * low_activity)
    messages_count[zero_messages] = 0
    messages_count = np.clip(np.round(messages_count), 0, 2000).astype(int)

    context_switches = rng.gamma(shape=2.0, scale=3.0, size=rows) + 0.05 * messages_count
    context_switches = _inject_outliers(
        rng, context_switches, prob=0.03, multiplier_range=(2.0, 4.0), max_value=120
    )
    context_switches = np.clip(np.round(context_switches), 0, 120).astype(int)

    tasks_reopened_lam = 0.4 + 0.03 * tasks_completed + 1.2 * after_hours_ratio + 0.01 * context_switches
    tasks_reopened = rng.poisson(tasks_reopened_lam)
    reopen_outliers = rng.random(rows) < 0.03
    tasks_reopened = tasks_reopened + reopen_outliers * rng.integers(6, 14, size=rows)
    tasks_reopened = np.clip(tasks_reopened, 0, 30).astype(int)

    deep_work_minutes = rng.normal(
        240 + 12 * active_days - 5.0 * meetings_count - 0.05 * messages_count, 60, size=rows
    )
    deep_work_minutes = np.clip(deep_work_minutes, 30, 600)

    overload_flags = (
        (after_hours_ratio > 0.65).astype(float)
        + (meetings_minutes > 500).astype(float)
        + (context_switches > 30).astype(float)
    )

    raw_score = label_signal_scale * (
        0.0026 * meetings_minutes
        + 2.0 * after_hours_ratio
        + 0.06 * tasks_reopened
        + 0.0075 * messages_count
        + 0.03 * context_switches
        + 0.01 * commits_count
        - 0.0045 * deep_work_minutes
        + 0.35 * overload_flags
    )
    raw_score = raw_score + rng.normal(0, label_noise_std, size=rows)
    target_positive_rate = label_positive_rate
    low, high = raw_score.min() - 5.0, raw_score.max() + 5.0
    for _ in range(40):
        mid = (low + high) / 2
        rate = (1 / (1 + np.exp(-(raw_score - mid) * label_sharpness))).mean()
        if rate > target_positive_rate:
            low = mid
        else:
            high = mid
    shift = (low + high) / 2
    prob = 1 / (1 + np.exp(-(raw_score - shift) * label_sharpness))
    burnout_label = (rng.random(rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "team_id": np.take(team_ids, team_index),
            "period_start": [
                (base_date + timedelta(days=int(i))).isoformat() for i in range(rows)
            ],
            "meetings_count": meetings_count,
            "meetings_minutes": meetings_minutes.round(1),
            "after_hours_ratio": after_hours_ratio.round(3),
            "commits_count": commits_count,
            "active_days": active_days,
            "tasks_completed": tasks_completed,
            "tasks_reopened": tasks_reopened,
            "messages_count": messages_count,
            "context_switches": context_switches,
            "deep_work_minutes": deep_work_minutes.round(1),
            "burnout_label": burnout_label,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _normalize_split_ratios(
    train_ratio: float, valid_ratio: float, test_ratio: float
) -> Tuple[float, float, float]:
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0):
        raise ValueError("Split ratios must be positive values.")
    total = ratios.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    if not np.isclose(total, 1.0):
        ratios = ratios / total
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def split_dataset_stratified(
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
    labels = df[label_col]
    class_counts = labels.value_counts(dropna=False)
    if class_counts.min() < 3:
        raise ValueError(
            "Not enough samples per class to build train/valid/test stratified splits."
        )

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    valid_share = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_share,
        stratify=temp_df[label_col],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def ensure_dataset(
    path: Path,
    rows: int,
    seed: int,
    generate_if_missing: bool,
    label_positive_rate: float = 0.18,
    label_noise_std: float = 0.12,
    label_sharpness: float = 2.0,
    label_signal_scale: float = 1.15,
) -> Path:
    if path.exists():
        return path
    if not generate_if_missing:
        raise FileNotFoundError(f"Dataset not found: {path}")
    return generate_synthetic_data(
        path,
        rows=rows,
        seed=seed,
        label_positive_rate=label_positive_rate,
        label_noise_std=label_noise_std,
        label_sharpness=label_sharpness,
        label_signal_scale=label_signal_scale,
    )


def ensure_split_datasets(
    raw_path: Path,
    processed_dir: Path,
    rows: int,
    seed: int,
    generate_if_missing: bool,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    label_positive_rate: float = 0.18,
    label_noise_std: float = 0.12,
    label_sharpness: float = 2.0,
    label_signal_scale: float = 1.15,
) -> Tuple[Path, Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    valid_path = processed_dir / "valid.csv"
    test_path = processed_dir / "test.csv"

    if train_path.exists() and valid_path.exists() and test_path.exists():
        return train_path, valid_path, test_path

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        if not generate_if_missing:
            raise FileNotFoundError(f"Dataset not found: {raw_path}")
        raw_path = generate_synthetic_data(
            raw_path,
            rows=rows,
            seed=seed,
            label_positive_rate=label_positive_rate,
            label_noise_std=label_noise_std,
            label_sharpness=label_sharpness,
            label_signal_scale=label_signal_scale,
        )
        df = pd.read_csv(raw_path)

    train_df, valid_df, test_df = split_dataset_stratified(
        df,
        label_col="burnout_label",
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, valid_path, test_path
