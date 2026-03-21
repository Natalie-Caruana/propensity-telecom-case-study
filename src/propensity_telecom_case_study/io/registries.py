"""MLflow model registry interactions — all MLflow I/O lives here."""

import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.pipeline import Pipeline

from propensity_telecom_case_study.config import MLflowConfig


class ModelRegistry:
    """Handles MLflow experiment tracking and model registration.

    Args:
        cfg: Validated MLflow configuration.
    """

    def __init__(self, cfg: MLflowConfig) -> None:
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

    def log_run(
        self,
        params: dict[str, object],
        metrics: dict[str, float],
        pipeline: Pipeline,
    ) -> str:
        """Log a training run: params, metrics, and the fitted pipeline artifact.

        Args:
            params: Hyperparameters to log.
            metrics: Evaluation metrics to log.
            pipeline: Fitted sklearn Pipeline to register.

        Returns:
            The MLflow run ID.
        """
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=self.cfg.model_name,
            )
            run_id = run.info.run_id

        # Tag latest registered version as @champion — stable deployment pointer
        client = mlflow.tracking.MlflowClient()
        latest = client.get_registered_model(self.cfg.model_name).latest_versions[-1]
        client.set_registered_model_alias(
            self.cfg.model_name, "champion", latest.version
        )

        logger.info(f"MLflow run logged: {run_id}")
        logger.info(f"Model '{self.cfg.model_name}' v{latest.version} tagged @champion")
        logger.info(f"Metrics: {metrics}")
        return run_id
