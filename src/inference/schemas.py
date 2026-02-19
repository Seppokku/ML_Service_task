from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BurnoutFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_id: Optional[str] = None
    period_start: Optional[str] = None
    meetings_count: int = Field(ge=0, le=200)
    meetings_minutes: float = Field(ge=0)
    after_hours_ratio: float = Field(ge=0.0, le=1.0)
    commits_count: int = Field(ge=0)
    active_days: int = Field(ge=0, le=31)
    tasks_completed: int = Field(ge=0)
    tasks_reopened: int = Field(ge=0)
    messages_count: int = Field(ge=0)
    context_switches: int = Field(ge=0)
    deep_work_minutes: float = Field(ge=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[BurnoutFeatures] = Field(min_length=1)


class PredictionItem(BaseModel):
    risk_score: float
    is_high_risk: bool


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictionItem]


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Optional[str] = None


class ReloadResponse(BaseModel):
    model_version: str
    message: str


class RegistryEntry(BaseModel):
    version: str
    created_at: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistryResponse(BaseModel):
    versions: List[RegistryEntry]


class StatsResponse(BaseModel):
    model_version: Optional[str] = None
    predict_requests: int
    predict_errors: int
    reload_count: int
    last_reload_at: Optional[str] = None

