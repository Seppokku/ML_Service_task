from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=5)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[TextPayload] = Field(min_length=1)


class PredictionItem(BaseModel):
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float] = Field(default_factory=dict)


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

