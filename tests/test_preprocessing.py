from __future__ import annotations

import pandas as pd

from common.preprocessing import FEATURE_COLUMNS, build_features, get_numeric_features


def test_build_features_columns(sample_feature_row) -> None:
    df = pd.DataFrame([sample_feature_row])

    features = build_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    numeric_cols = get_numeric_features()
    assert (features[numeric_cols].values >= 0).all()
    assert features["weekday"].iloc[0] == "unknown"
    assert features["month"].iloc[0] == "unknown"