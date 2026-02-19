from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def sample_feature_row() -> dict:
    return {
        "meetings_count": 5,
        "meetings_minutes": 180,
        "after_hours_ratio": 0.2,
        "commits_count": 12,
        "active_days": 4,
        "tasks_completed": 10,
        "tasks_reopened": 2,
        "messages_count": 80,
        "context_switches": 15,
        "deep_work_minutes": 200,
    }
