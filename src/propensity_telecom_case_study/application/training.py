"""Training pipeline — wires domain and I/O together."""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from propensity_telecom_case_study.config import TrainConfig
from propensity_telecom_case_study.domain.metrics import compute_metrics
from propensity_telecom_case_study.domain.model import build_pipeline
from propensity_telecom_case_study.io.datasets import DatasetLoader
from propensity_telecom_case_study.io.registries import ModelRegistry


def run_training(cfg: TrainConfig) -> dict[str, float]:
    """Execute the full training pipeline end-to-end.

    Steps:
        1. Load raw data
        2. Split into train / test (stratified)
        3. Build and fit the sklearn pipeline
        4. Evaluate on the held-out test set
        5. Log params, metrics, and model to MLflow

    Args:
        cfg: Validated TrainConfig from YAML + CLI overrides.

    Returns:
        Dictionary of test-set evaluation metrics.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    loader = DatasetLoader(cfg.data.raw_path)
    df = loader.load()

    all_features = cfg.features.numeric + cfg.features.categorical + cfg.features.binary
    X: pd.DataFrame = df[all_features]
    y: pd.Series = df[cfg.features.target]

    # ── 2. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    logger.info(f"Upgrade rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    # ── 3. Build & fit pipeline ───────────────────────────────────────────────
    pipeline = build_pipeline(
        numeric_features=cfg.features.numeric,
        categorical_features=cfg.features.categorical,
        binary_features=cfg.features.binary,
        cfg=cfg.model,
    )
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob)
    logger.info(f"Test metrics: {metrics}")

    # ── 5. Log to MLflow ──────────────────────────────────────────────────────
    registry = ModelRegistry(cfg.mlflow)
    registry.log_run(
        params=cfg.model.model_dump(),
        metrics=metrics,
        pipeline=pipeline,
    )

    return metrics
