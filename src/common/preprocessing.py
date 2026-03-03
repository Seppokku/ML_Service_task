from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TEXT_COLUMN = "text"
TARGET_COLUMN = "category"
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class TextPreprocessConfig:
    use_stopwords: bool = True
    use_stem: bool = False
    use_lemma: bool = False


def get_stopwords() -> set[str]:
    return set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def basic_tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [token.lower() for token in _WORD_RE.findall(text)]


def _stem_tokens(tokens: list[str]) -> list[str]:
    try:
        from nltk.stem import PorterStemmer  # type: ignore

        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    except Exception:
        return tokens


def _lemmatize_tokens(tokens: list[str]) -> list[str]:
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore

        lemmatizer = WordNetLemmatizer()
        try:
            return [lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            return tokens
    except Exception:
        return tokens


def preprocess_text(
    text: str,
    stopwords: Optional[set[str]] = None,
    use_stem: bool = False,
    use_lemma: bool = False,
) -> str:
    tokens = basic_tokenize(clean_text(text))
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    if use_stem:
        tokens = _stem_tokens(tokens)
    if use_lemma:
        tokens = _lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_texts(
    texts: Iterable[str],
    config: Optional[TextPreprocessConfig] = None,
) -> list[str]:
    cfg = config or TextPreprocessConfig()
    stopwords = get_stopwords() if cfg.use_stopwords else None
    return [
        preprocess_text(
            text=text,
            stopwords=stopwords,
            use_stem=cfg.use_stem,
            use_lemma=cfg.use_lemma,
        )
        for text in texts
    ]


def validate_training_frame(df: pd.DataFrame) -> None:
    required = [TEXT_COLUMN, TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset must contain columns: {missing}")
