from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def sample_text_row() -> dict:
    return {
        "text": "Britain economy grows as market confidence improves.",
        "category": "business",
    }
