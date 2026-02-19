from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _to_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class AppConfig:
    data_path: Path
    processed_dir: Path
    registry_path: Path
    model_version: str
    inference_reload_url: Optional[str]
    metrics_min_pr_auc: float
    metrics_min_recall: float
    recall_precision_threshold: float
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    random_seed: int
    generate_data_if_missing: bool
    synthetic_rows: int
    log_level: str
    cb_tuning_trials: int
    cb_early_stopping_rounds: int
    label_positive_rate: float
    label_noise_std: float
    label_sharpness: float
    label_signal_scale: float


_CACHED_CONFIG: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    data_path = _resolve_path(
        _get_env("DATA_PATH", "data/raw/team_activity.csv") or "data/raw/team_activity.csv",
        ROOT_DIR,
    )
    processed_dir = _resolve_path(
        _get_env("PROCESSED_DIR", "data/processed") or "data/processed",
        ROOT_DIR,
    )
    registry_path = _resolve_path(
        _get_env("REGISTRY_PATH", "model_registry") or "model_registry",
        ROOT_DIR,
    )
    train_ratio = _to_float(_get_env("TRAIN_RATIO", None), 0.7)
    valid_ratio = _to_float(_get_env("VALID_RATIO", None), 0.2)
    test_ratio = _to_float(_get_env("TEST_RATIO", None), 0.1)
    legacy_valid_ratio = _get_env("TRAIN_TEST_SPLIT", None)
    if legacy_valid_ratio is not None and _get_env("VALID_RATIO", None) is None:
        valid_ratio = _to_float(legacy_valid_ratio, valid_ratio)
        train_ratio = max(0.0, 1.0 - valid_ratio - test_ratio)

    _CACHED_CONFIG = AppConfig(
        data_path=data_path,
        processed_dir=processed_dir,
        registry_path=registry_path,
        model_version=_get_env("MODEL_VERSION", "latest") or "latest",
        inference_reload_url=_get_env("INFERENCE_RELOAD_URL", None),
        metrics_min_pr_auc=_to_float(_get_env("METRICS_MIN_PR_AUC", None), 0.45),
        metrics_min_recall=_to_float(_get_env("METRICS_MIN_RECALL", None), 0.55),
        recall_precision_threshold=_to_float(_get_env("RECALL_PRECISION_THRESHOLD", None), 0.8),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=_to_int(_get_env("RANDOM_SEED", None), 42),
        generate_data_if_missing=_to_bool(_get_env("GENERATE_DATA_IF_MISSING", None), True),
        synthetic_rows=_to_int(_get_env("SYNTHETIC_ROWS", None), 500),
        log_level=_get_env("LOG_LEVEL", "INFO") or "INFO",
        cb_tuning_trials=_to_int(_get_env("CB_TUNING_TRIALS", None), 6),
        cb_early_stopping_rounds=_to_int(
            _get_env("CB_EARLY_STOPPING_ROUNDS", None), 50
        ),
        label_positive_rate=_to_float(_get_env("LABEL_POS_RATE", None), 0.18),
        label_noise_std=_to_float(_get_env("LABEL_NOISE_STD", None), 0.12),
        label_sharpness=_to_float(_get_env("LABEL_SHARPNESS", None), 2.0),
        label_signal_scale=_to_float(_get_env("LABEL_SIGNAL_SCALE", None), 1.15),
    )
    return _CACHED_CONFIG
