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
    metrics_min_f1_macro: float
    metrics_min_accuracy: float
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    random_seed: int
    text_use_stopwords: bool
    text_use_stem: bool
    text_use_lemma: bool
    max_features: int
    log_level: str


_CACHED_CONFIG: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    data_path = _resolve_path(
        _get_env("DATA_PATH", "data/raw/bbc-text.csv") or "data/raw/bbc-text.csv",
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

    _CACHED_CONFIG = AppConfig(
        data_path=data_path,
        processed_dir=processed_dir,
        registry_path=registry_path,
        model_version=_get_env("MODEL_VERSION", "latest") or "latest",
        inference_reload_url=_get_env("INFERENCE_RELOAD_URL", None),
        metrics_min_f1_macro=_to_float(_get_env("METRICS_MIN_F1_MACRO", None), 0.9),
        metrics_min_accuracy=_to_float(_get_env("METRICS_MIN_ACCURACY", None), 0.9),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_seed=_to_int(_get_env("RANDOM_SEED", None), 42),
        text_use_stopwords=_to_bool(_get_env("TEXT_USE_STOPWORDS", None), True),
        text_use_stem=_to_bool(_get_env("TEXT_USE_STEM", None), False),
        text_use_lemma=_to_bool(_get_env("TEXT_USE_LEMMA", None), False),
        max_features=_to_int(_get_env("MAX_FEATURES", None), 50000),
        log_level=_get_env("LOG_LEVEL", "INFO") or "INFO",
    )
    return _CACHED_CONFIG
