from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from common.registry import ModelRegistry


def test_registry_save_load(tmp_path) -> None:
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    model = GradientBoostingClassifier().fit(X, y)

    registry = ModelRegistry(tmp_path)
    version = registry.save_model(
        model=model,
        metrics={"pr_auc": 0.5, "recall_at_precision": 0.5},
        feature_names=list(X.columns),
    )

    loaded_model, metadata, metrics = registry.load_model(version)
    assert loaded_model is not None
    assert metadata["feature_names"] == list(X.columns)
    assert metrics["pr_auc"] == 0.5
