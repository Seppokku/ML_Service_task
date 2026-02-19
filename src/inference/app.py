from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from common.config import get_config
from common.logging import get_logger, setup_logging
from common.preprocessing import build_features
from common.registry import ModelRegistry
from inference.model_loader import ModelService
from inference.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionItem,
    RegistryResponse,
    ReloadRequest,
    ReloadResponse,
    StatsResponse,
)

config = get_config()
setup_logging(config.log_level)
logger = get_logger("inference")

app = FastAPI(title="Burnout Risk Inference", version="0.1.0")
registry = ModelRegistry(config.registry_path)
model_service = ModelService(registry)
static_dir = Path(__file__).resolve().parent / "static"
index_file = static_dir / "index.html"

app.state.stats = {
    "predict_requests": 0,
    "predict_errors": 0,
    "reload_count": 0,
    "last_reload_at": None,
}


def _http_error(status_code: int, error_code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Request %s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.on_event("startup")
def startup() -> None:
    try:
        version = model_service.load(config.model_version)
        logger.info("Model loaded on startup. version=%s", version)
        app.state.stats["last_reload_at"] = datetime.now(timezone.utc).isoformat()
    except FileNotFoundError:
        logger.warning("Model registry is empty. Run training to create a model.")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_version": model_service.version,
    }


@app.get("/")
def ui_root() -> FileResponse:
    return FileResponse(index_file)


@app.get("/ui")
def ui_alias() -> FileResponse:
    return FileResponse(index_file)


@app.get("/registry", response_model=RegistryResponse)
def registry_versions(limit: int = 5) -> RegistryResponse:
    versions = registry.get_versions_info(limit=limit)
    return RegistryResponse(versions=versions)


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    stats_payload = app.state.stats.copy()
    stats_payload["model_version"] = model_service.version
    return StatsResponse(**stats_payload)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    app.state.stats["predict_requests"] += 1
    if model_service.model is None:
        app.state.stats["predict_errors"] += 1
        _http_error(503, "MODEL_NOT_LOADED", "Model is not loaded.")

    payload = [item.model_dump() for item in request.items]
    logger.info("Predict request received. items=%s", len(payload))
    raw_df = pd.DataFrame(payload)

    try:
        features = build_features(raw_df)
    except ValueError as exc:
        logger.warning("Invalid input: %s", exc)
        app.state.stats["predict_errors"] += 1
        _http_error(400, "INVALID_INPUT", str(exc))

    feature_names = model_service.metadata.get("feature_names")
    if feature_names:
        missing = [col for col in feature_names if col not in features.columns]
        if missing:
            app.state.stats["predict_errors"] += 1
            _http_error(400, "FEATURE_MISMATCH", f"Missing features: {missing}")
        features = features[feature_names]

    scores = model_service.predict(features)
    threshold = model_service.metadata.get("decision_threshold", 0.5)
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.5
    predictions = [
        PredictionItem(risk_score=score, is_high_risk=score >= threshold_value)
        for score in scores
    ]
    return PredictResponse(model_version=model_service.version or "unknown", predictions=predictions)


@app.post("/reload", response_model=ReloadResponse)
def reload_model(request: Optional[ReloadRequest] = None) -> ReloadResponse:
    version = request.version if request else None
    try:
        new_version = model_service.load(version or "latest")
    except FileNotFoundError as exc:
        _http_error(404, "MODEL_NOT_FOUND", str(exc))

    app.state.stats["reload_count"] += 1
    app.state.stats["last_reload_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Model reloaded. version=%s", new_version)
    return ReloadResponse(model_version=new_version, message="Model reloaded")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("Validation error for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content={"error_code": "VALIDATION_ERROR", "message": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error_code": "INTERNAL_ERROR", "message": "Internal server error"},
    )
