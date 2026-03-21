"""Inference pipeline — loads a registered model and scores new customers."""

from pathlib import Path

import mlflow.sklearn
import pandas as pd
from loguru import logger

from propensity_telecom_case_study.config import TrainConfig


def run_inference(cfg: TrainConfig, input_path: str | Path) -> pd.DataFrame:
    """Score a batch of customers using the latest registered model.

    Args:
        cfg: Validated TrainConfig (uses mlflow config for registry lookup).
        input_path: Path to a CSV file of customers to score.

    Returns:
        Input DataFrame with an added `propensity_score` column.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} customers to score from {input_path}")

    all_features = cfg.features.numeric + cfg.features.categorical + cfg.features.binary
    X = df[all_features]

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model_uri = f"models:/{cfg.mlflow.model_name}/latest"
    logger.info(f"Loading model from {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    df["propensity_score"] = pipeline.predict_proba(X)[:, 1]
    logger.info(f"Scored {len(df):,} customers  |  mean propensity: {df['propensity_score'].mean():.3f}")

    return df
