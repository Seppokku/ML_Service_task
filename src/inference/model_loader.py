from __future__ import annotations

from typing import Any, Optional

import numpy as np

from common.registry import ModelRegistry


class ModelService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.model: Optional[Any] = None
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.version: Optional[str] = None

    def load(self, version: Optional[str] = None) -> str:
        model, metadata, metrics = self.registry.load_model(version)
        self.model = model
        self.metadata = metadata
        self.metrics = metrics
        self.version = metadata.get("version") or version or "unknown"
        return self.version

    def predict_labels(self, texts: list[str]) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        preds = self.model.predict(texts)
        return [str(item) for item in preds]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(texts)
            return probas.tolist()
        labels = self.predict_labels(texts)
        known_labels = self.metadata.get("labels", [])
        if known_labels:
            return [
                [1.0 if label == known_label else 0.0 for known_label in known_labels]
                for label in labels
            ]
        return np.ones((len(labels), 1), dtype=float).tolist()
