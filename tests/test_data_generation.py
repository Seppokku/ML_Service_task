from __future__ import annotations

import pandas as pd

from common.data_generation import split_dataset_stratified


def test_split_dataset_stratified_preserves_ratio() -> None:
    df = pd.DataFrame(
        {
            "feature": list(range(100)),
            "burnout_label": [0] * 80 + [1] * 20,
        }
    )
    train_df, valid_df, test_df = split_dataset_stratified(
        df,
        label_col="burnout_label",
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )

    assert len(train_df) == 70
    assert len(valid_df) == 20
    assert len(test_df) == 10

    overall_rate = df["burnout_label"].mean()
    for split in (train_df, valid_df, test_df):
        assert abs(split["burnout_label"].mean() - overall_rate) < 0.01
