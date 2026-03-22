"""FastAPI inference server — loads the registered model once and scores customers."""

from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from propensity_telecom_case_study.api.routers import frontend, predict
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")

app.include_router(predict.router)
app.include_router(frontend.router)
