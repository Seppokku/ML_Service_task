from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterSampler

from common.config import get_config
from common.data_generation import ensure_split_datasets, split_dataset_stratified
from common.logging import get_logger, setup_logging
from common.preprocessing import (
    build_features,
    get_categorical_features,
)
from common.registry import ModelRegistry
from training.metrics import compute_metrics
from training.validator import is_model_acceptable


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "burnout_label" not in df.columns:
        raise ValueError("Dataset must contain burnout_label column.")
    return df


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _load_split_datasets(
    config, logger, data_path_override: Optional[Path]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_path_override is not None:
        data_path = Path(data_path_override)
        logger.info("Using override dataset: %s", data_path)
        df = _load_dataset(data_path)
        train_df, valid_df, test_df = split_dataset_stratified(
            df,
            label_col="burnout_label",
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio,
            test_ratio=config.test_ratio,
            seed=config.random_seed,
        )
        logger.info(
            "Override split sizes: train=%s valid=%s test=%s",
            len(train_df),
            len(valid_df),
            len(test_df),
        )
        return train_df, valid_df, test_df

    train_path, valid_path, test_path = ensure_split_datasets(
        config.data_path,
        processed_dir=config.processed_dir,
        rows=config.synthetic_rows,
        seed=config.random_seed,
        generate_if_missing=config.generate_data_if_missing,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        label_positive_rate=config.label_positive_rate,
        label_noise_std=config.label_noise_std,
        label_sharpness=config.label_sharpness,
        label_signal_scale=config.label_signal_scale,
    )
    logger.info("Using dataset splits: %s | %s | %s", train_path, valid_path, test_path)
    return (
        _load_dataset(train_path),
        _load_dataset(valid_path),
        _load_dataset(test_path),
    )


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

    if data_path_override is not None:
        train_df, valid_df, test_df = _load_split_datasets(
            config, logger, data_path_override
        )
    else:
        train_df, valid_df, test_df = _load_split_datasets(config, logger, None)

    y_train = train_df["burnout_label"].astype(int).to_numpy()
    y_valid = valid_df["burnout_label"].astype(int).to_numpy()
    y_test = test_df["burnout_label"].astype(int).to_numpy()

    if len(set(y_train)) < 2 or len(set(y_valid)) < 2:
        logger.warning("Train/valid splits have a single class. Training skipped.")
        return {
            "status": "skipped",
            "message": "Train/valid splits have a single class.",
            "metrics": {},
        }

    X_train = build_features(train_df)
    X_valid = build_features(valid_df)
    X_test = build_features(test_df)
    categorical_features = get_categorical_features()
    cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features]
    logger.info(
        "Split sizes: train=%s valid=%s test=%s",
        len(X_train),
        len(X_valid),
        len(X_test),
    )

    param_space = {
        "iterations": [300, 500, 800, 1000, 1200],
        "depth": [4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "l2_leaf_reg": [3, 5, 7, 9],
        "subsample": [0.7, 0.85, 1.0],
        "rsm": [0.7, 0.85, 1.0],
    }

    candidates = list(
        ParameterSampler(
            param_space,
            n_iter=max(1, config.cb_tuning_trials),
            random_state=config.random_seed,
        )
    )

    best_score = None
    best_metrics = None
    best_model = None
    best_params = None

    for idx, params in enumerate(candidates, start=1):
        params = params.copy()
        if params.get("subsample", 1.0) < 1.0:
            params["bootstrap_type"] = "Bernoulli"

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=config.random_seed,
            verbose=False,
            auto_class_weights="Balanced",
            od_type="Iter",
            od_wait=config.cb_early_stopping_rounds,
            **params,
        )

        try:
            model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
            )
        except Exception as exc:
            logger.warning("Trial %s failed: %s", idx, exc)
            continue

        y_prob = model.predict_proba(X_valid)[:, 1]
        metrics = compute_metrics(
            y_valid, y_prob, precision_threshold=config.recall_precision_threshold
        )
        logger.info(
            "Trial %s/%s params=%s metrics=%s",
            idx,
            len(candidates),
            params,
            json.dumps(metrics, indent=2),
        )

        score = (
            metrics.get("pr_auc", 0.0),
            metrics.get("recall_at_precision", 0.0),
            metrics.get("recall_at_top_k", 0.0),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics
            best_model = model
            best_params = params

    if best_model is None or best_metrics is None:
        logger.warning("No successful training trials.")
        return {
            "status": "failed",
            "message": "No successful training trials.",
            "metrics": {},
        }

    metrics = best_metrics
    model = best_model
    decision_threshold = metrics.get("best_f2_threshold", 0.5)
    logger.info("Best params: %s", json.dumps(best_params, indent=2))
    logger.info("Validation metrics: %s", json.dumps(metrics, indent=2))

    if len(set(y_test)) < 2:
        logger.warning("Test split has a single class. Test metrics skipped.")
    else:
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(
            y_test, y_test_prob, precision_threshold=config.recall_precision_threshold
        )
        metrics.update(_prefix_metrics(test_metrics, "test_"))
        logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    if not is_model_acceptable(metrics, config.metrics_min_pr_auc, config.metrics_min_recall):
        logger.warning(
            "Model rejected. pr_auc=%.4f recall@p>=%.2f=%.4f",
            metrics.get("pr_auc", 0.0),
            metrics.get("precision_threshold", 0.0),
            metrics.get("recall_at_precision", 0.0),
        )
        return {
            "status": "rejected",
            "message": "Model did not pass validation thresholds.",
            "metrics": metrics,
            "best_params": best_params or {},
            "decision_threshold": float(decision_threshold),
        }

    registry = ModelRegistry(config.registry_path)
    version = registry.save_model(
        model=model,
        metrics=metrics,
        feature_names=list(X_train.columns),
        extra_metadata={
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(y_train.mean()),
            "best_params": best_params or {},
            "decision_threshold": float(decision_threshold),
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
        "best_params": best_params or {},
        "decision_threshold": float(decision_threshold),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y_train.mean()),
    }


def train() -> int:
    result = run_training()
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(train())
