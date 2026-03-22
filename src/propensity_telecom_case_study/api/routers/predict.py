"""Inference endpoints — health check and batch scoring."""

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from propensity_telecom_case_study.api.schemas import (
    CustomerPrediction,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from propensity_telecom_case_study.config import TrainConfig

router = APIRouter(tags=["inference"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness check — confirms the model is loaded and the server is ready."""
    model_loaded = (
        hasattr(request.app.state, "pipeline")
        and request.app.state.pipeline is not None
    )
    model_name = request.app.state.cfg.mlflow.model_name if model_loaded else "unknown"
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
    )


@router.post("/predict", response_model=PredictionResponse, name="predict")
def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    """Score a batch of customers and return their propensity scores."""
    if not hasattr(request.app.state, "pipeline") or request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cfg: TrainConfig = request.app.state.cfg
    pipeline: Any = request.app.state.pipeline

    all_features = cfg.features.numeric + cfg.features.categorical + cfg.features.binary
    records = [customer.model_dump() for customer in body.customers]
    df = pd.DataFrame(records)[all_features]

    scores: list[float] = pipeline.predict_proba(df)[:, 1].tolist()
    logger.info(
        f"Scored {len(scores)} | mean propensity: {sum(scores) / len(scores):.3f}"
    )

    return PredictionResponse(
        predictions=[CustomerPrediction(propensity_score=s) for s in scores],
        model_name=cfg.mlflow.model_name,
        count=len(scores),
    )
