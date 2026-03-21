"""Training pipeline — wires domain and I/O together."""

import tempfile
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from propensity_telecom_case_study.config import TrainConfig
from propensity_telecom_case_study.domain.drift import (
    build_drift_report,
    save_drift_report,
)
from propensity_telecom_case_study.domain.explainability import (
    compute_shap_values,
    save_shap_summary,
)
from propensity_telecom_case_study.domain.features import get_feature_names
from propensity_telecom_case_study.domain.metrics import compute_metrics
from propensity_telecom_case_study.domain.model import build_pipeline
from propensity_telecom_case_study.domain.reproducibility import (
    get_git_commit,
    set_global_seeds,
)
from propensity_telecom_case_study.io.alerts import alert_on_metric_threshold
from propensity_telecom_case_study.io.datasets import DatasetLoader
from propensity_telecom_case_study.io.registries import ModelRegistry

# Minimum acceptable ROC-AUC before an alert fires
ROC_AUC_THRESHOLD = 0.55


def run_training(cfg: TrainConfig) -> dict[str, float]:
    """Execute the full training pipeline end-to-end.

    Steps:
        1. Fix random seeds + capture git commit hash
        2. Load raw data and log lineage to MLflow
        3. Split into stratified train / test sets
        4. Build and fit the sklearn pipeline
        5. Evaluate on the held-out test set
        6. Generate SHAP explanations and drift report
        7. Log everything to MLflow; alert if metrics degrade

    Args:
        cfg: Validated TrainConfig from YAML + CLI overrides.

    Returns:
        Dictionary of test-set evaluation metrics.
    """
    # ── 1. Reproducibility ────────────────────────────────────────────────────
    set_global_seeds(cfg.data.random_state)
    git_commit = get_git_commit()
    logger.info(f"Git commit: {git_commit}")

    # params + metrics auto-logged; model logged manually via ModelRegistry
    mlflow.sklearn.autolog(log_models=False)
    mlflow.enable_system_metrics_logging()

    # ── 2. Load + data lineage ────────────────────────────────────────────────
    loader = DatasetLoader(cfg.data.raw_path)
    df = loader.load()

    all_features = cfg.features.numeric + cfg.features.categorical + cfg.features.binary
    X: pd.DataFrame = df[all_features]
    y: pd.Series = df[cfg.features.target]

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    logger.info(
        f"Upgrade rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}"
    )  # noqa: E501

    # ── 4. Build & fit pipeline ───────────────────────────────────────────────
    pipeline = build_pipeline(
        numeric_features=cfg.features.numeric,
        categorical_features=cfg.features.categorical,
        binary_features=cfg.features.binary,
        cfg=cfg.model,
    )
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob)
    logger.info(f"Test metrics: {metrics}")

    # ── 6. Explainability + drift ─────────────────────────────────────────────
    feature_names = get_feature_names(
        pipeline.named_steps["preprocessor"],
        cfg.features.numeric,
        cfg.features.categorical,
        cfg.features.binary,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # SHAP summary plot
        shap_values, _ = compute_shap_values(pipeline, X_test)
        shap_plot_path = tmp_path / "shap_summary.png"
        save_shap_summary(shap_values, feature_names, shap_plot_path)
        logger.info("SHAP summary computed")

        # Data drift report (train vs test as proxy for production drift)
        numeric_cols = cfg.features.numeric
        drift_snapshot = build_drift_report(X_train, X_test, numeric_cols)
        drift_path = tmp_path / "drift_report.html"
        save_drift_report(drift_snapshot, drift_path)
        logger.info("Drift report generated")

        # ── 7. Log to MLflow ──────────────────────────────────────────────────
        registry = ModelRegistry(cfg.mlflow)
        registry.log_run(
            params={**cfg.model.model_dump(), "git_commit": git_commit},
            metrics=metrics,
            pipeline=pipeline,
            artifacts={
                "shap_summary": str(shap_plot_path),
                "drift_report": str(drift_path),
            },
            train_df=X_train,
            dataset_name=Path(cfg.data.raw_path).stem,
        )

    # ── Alert if model quality degrades ──────────────────────────────────────
    alert_on_metric_threshold(
        metric_name="roc_auc",
        value=metrics["roc_auc"],
        threshold=ROC_AUC_THRESHOLD,
        direction="below",
    )

    return metrics
