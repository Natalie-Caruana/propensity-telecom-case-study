"""FastAPI inference server — loads the registered model once and scores customers."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from propensity_telecom_case_study.api.schemas import (
    CustomerPrediction,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from propensity_telecom_case_study.config import TrainConfig, load_config


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load the MLflow model once at startup; release on shutdown."""
    cfg: TrainConfig = load_config(cli_overrides=False)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model_uri = f"models:/{cfg.mlflow.model_name}/latest"
    logger.info(f"Loading model from {model_uri}")
    app.state.pipeline = mlflow.sklearn.load_model(model_uri)
    app.state.cfg = cfg
    logger.info("Model loaded — server ready.")
    yield
    logger.info("Server shutting down.")


_STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Propensity Telecom Inference API",
    description="Score telecom customers for upsell propensity using the latest model.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the scoring UI."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check — confirms the model is loaded and the server is ready."""
    model_loaded = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    model_name = app.state.cfg.mlflow.model_name if model_loaded else "unknown"
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score a batch of customers and return their propensity scores."""
    if not hasattr(app.state, "pipeline") or app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cfg: TrainConfig = app.state.cfg
    pipeline: Any = app.state.pipeline

    all_features = cfg.features.numeric + cfg.features.categorical + cfg.features.binary
    records = [customer.model_dump() for customer in request.customers]
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
