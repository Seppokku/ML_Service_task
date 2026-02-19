from __future__ import annotations

from typing import Any, Optional

import pandas as pd

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

    def predict(self, features: pd.DataFrame) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)[:, 1]
        else:
            probas = self.model.predict(features)
        return probas.tolist()
