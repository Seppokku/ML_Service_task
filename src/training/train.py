from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from common.config import ROOT_DIR, get_config
from common.dataset import stratified_train_valid_test_split
from common.logging import get_logger, setup_logging
from common.preprocessing import (
    TARGET_COLUMN,
    TEXT_COLUMN,
    TextPreprocessConfig,
    preprocess_texts,
    validate_training_frame,
)
from common.registry import ModelRegistry
from training.metrics import compute_metrics
from training.validator import is_model_acceptable


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    validate_training_frame(df)
    frame = df[[TEXT_COLUMN, TARGET_COLUMN]].copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].fillna("unknown").astype(str)
    frame = frame[frame[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset has no non-empty text rows.")
    return frame


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _build_candidates(max_features: int, seed: int) -> list[tuple[str, Pipeline]]:
    return [
        (
            "count_mnb",
            Pipeline(
                [
                    ("vect", CountVectorizer(max_features=max_features)),
                    ("clf", MultinomialNB()),
                ]
            ),
        ),
        (
            "tfidf_logreg",
            Pipeline(
                [
                    (
                        "vect",
                        TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=(1, 2),
                            min_df=2,
                        ),
                    ),
                    ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
        ),
    ]


def _notify_inference(reload_url: str, version: str, logger) -> None:
    try:
        response = requests.post(reload_url, json={"version": version}, timeout=10)
        response.raise_for_status()
        logger.info("Inference reload triggered. status=%s", response.status_code)
    except requests.RequestException as exc:
        logger.warning("Failed to notify inference: %s", exc)


def run_training(
    data_path_override: Optional[Path] = None,
    notify_inference: bool = True,
) -> Dict[str, Any]:
    config = get_config()
    setup_logging(config.log_level)
    logger = get_logger("training")

    data_path = Path(data_path_override) if data_path_override else config.data_path
    logger.info("Loading dataset: %s", data_path)
    fallback_data_path = ROOT_DIR / "data" / "raw" / "bbc-text.csv"
    try:
        raw_df = _load_dataset(data_path)
    except (FileNotFoundError, ValueError) as exc:
        if data_path != fallback_data_path and fallback_data_path.exists():
            logger.warning(
                "Primary dataset is invalid (%s). Falling back to %s",
                exc,
                fallback_data_path,
            )
            data_path = fallback_data_path
            raw_df = _load_dataset(data_path)
        else:
            raise
    train_df, valid_df, test_df = stratified_train_valid_test_split(
        raw_df,
        label_col=TARGET_COLUMN,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.random_seed,
    )

    text_cfg = TextPreprocessConfig(
        use_stopwords=config.text_use_stopwords,
        use_stem=config.text_use_stem,
        use_lemma=config.text_use_lemma,
    )

    X_train = preprocess_texts(train_df[TEXT_COLUMN].tolist(), text_cfg)
    X_valid = preprocess_texts(valid_df[TEXT_COLUMN].tolist(), text_cfg)
    X_test = preprocess_texts(test_df[TEXT_COLUMN].tolist(), text_cfg)
    y_train = train_df[TARGET_COLUMN].astype(str).to_numpy()
    y_valid = valid_df[TARGET_COLUMN].astype(str).to_numpy()
    y_test = test_df[TARGET_COLUMN].astype(str).to_numpy()

    logger.info(
        "Split sizes: train=%s valid=%s test=%s",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    candidates = _build_candidates(config.max_features, config.random_seed)
    best_score = None
    best_metrics = None
    best_model = None
    best_model_name = None

    for idx, (model_name, model) in enumerate(candidates, start=1):

        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning("Trial %s failed: %s", idx, exc)
            continue

        y_pred = model.predict(X_valid)
        metrics = compute_metrics(y_valid, y_pred)
        logger.info(
            "Trial %s/%s model=%s metrics=%s",
            idx,
            len(candidates),
            model_name,
            json.dumps(metrics, indent=2),
        )

        score = (
            metrics.get("f1_macro", 0.0),
            metrics.get("accuracy", 0.0),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics
            best_model = model
            best_model_name = model_name

    if best_model is None or best_metrics is None:
        logger.warning("No successful training trials.")
        return {
            "status": "failed",
            "message": "No successful training trials.",
            "metrics": {},
        }

    metrics = best_metrics
    model = best_model
    logger.info("Best model: %s", best_model_name)
    logger.info("Validation metrics: %s", json.dumps(metrics, indent=2))

    y_test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    metrics.update(_prefix_metrics(test_metrics, "test_"))
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    if not is_model_acceptable(
        metrics, config.metrics_min_f1_macro, config.metrics_min_accuracy
    ):
        logger.warning(
            "Model rejected. f1_macro=%.4f accuracy=%.4f",
            metrics.get("f1_macro", 0.0),
            metrics.get("accuracy", 0.0),
        )
        return {
            "status": "rejected",
            "message": "Model did not pass validation thresholds.",
            "metrics": metrics,
            "best_model": best_model_name or "",
        }

    registry = ModelRegistry(config.registry_path)
    version = registry.save_model(
        model=model,
        metrics=metrics,
        feature_names=[TEXT_COLUMN],
        extra_metadata={
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "labels": sorted(raw_df[TARGET_COLUMN].astype(str).unique().tolist()),
            "text_preprocessing": {
                "use_stopwords": text_cfg.use_stopwords,
                "use_stem": text_cfg.use_stem,
                "use_lemma": text_cfg.use_lemma,
            },
            "best_model": best_model_name or "",
        },
    )
    logger.info("Model saved to registry. version=%s", version)

    if notify_inference and config.inference_reload_url:
        _notify_inference(config.inference_reload_url, version, logger)

    return {
        "status": "accepted",
        "message": "Model saved to registry.",
        "model_version": version,
        "metrics": metrics,
        "best_model": best_model_name or "",
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
    }


def train() -> int:
    result = run_training()
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(train())
