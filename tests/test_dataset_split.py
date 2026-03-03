from __future__ import annotations

import pandas as pd

from common.dataset import stratified_train_valid_test_split


def test_stratified_train_valid_test_split_preserves_rows_and_labels() -> None:
    rows = 150
    df = pd.DataFrame(
        {
            "text": [f"text sample {idx}" for idx in range(rows)],
            "category": ["sport"] * 50 + ["business"] * 50 + ["tech"] * 50,
        }
    )
    train_df, valid_df, test_df = stratified_train_valid_test_split(
        df,
        label_col="category",
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )

    assert len(train_df) == 105
    assert len(valid_df) == 30
    assert len(test_df) == 15

    train_labels = set(train_df["category"].unique())
    valid_labels = set(valid_df["category"].unique())
    test_labels = set(test_df["category"].unique())
    assert train_labels == valid_labels == test_labels == {"sport", "business", "tech"}
