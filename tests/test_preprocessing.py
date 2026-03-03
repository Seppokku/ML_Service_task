from __future__ import annotations

import pandas as pd

from common.preprocessing import (
    TARGET_COLUMN,
    TEXT_COLUMN,
    TextPreprocessConfig,
    preprocess_texts,
    validate_training_frame,
)


def test_validate_training_frame(sample_text_row) -> None:
    df = pd.DataFrame([sample_text_row])
    validate_training_frame(df)
    assert TEXT_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns


def test_preprocess_texts_stopwords(sample_text_row) -> None:
    texts = [sample_text_row["text"]]
    processed = preprocess_texts(
        texts, TextPreprocessConfig(use_stopwords=True, use_stem=False, use_lemma=False)
    )
    assert len(processed) == 1
    assert "as" not in processed[0]
    assert "economy" in processed[0]