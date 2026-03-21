"""MLflow model registry interactions — all MLflow I/O lives here."""

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
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
        artifacts: dict[str, str] | None = None,
        train_df: pd.DataFrame | None = None,
        dataset_name: str = "telecom_customers",
    ) -> str:
        """Log a training run: params, metrics, artifacts, lineage, and model.

        Args:
            params: Hyperparameters to log (including git_commit).
            metrics: Evaluation metrics to log.
            pipeline: Fitted sklearn Pipeline to register.
            artifacts: Optional dict of {label: local_file_path} to upload.
            train_df: Optional training DataFrame for MLflow data lineage.
            dataset_name: Name tag for the logged dataset.

        Returns:
            The MLflow run ID.
        """
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Data lineage
            if train_df is not None:
                dataset = mlflow.data.from_pandas(
                    train_df, name=dataset_name, targets=None
                )
                mlflow.log_input(dataset, context="training")

            # Artifacts (SHAP plot, drift report, etc.)
            if artifacts:
                for label, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=label)

            # Register model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=self.cfg.model_name,
            )
            run_id = run.info.run_id

        # Tag latest version as @champion
        client = mlflow.tracking.MlflowClient()
        latest = client.get_registered_model(self.cfg.model_name).latest_versions[-1]
        client.set_registered_model_alias(
            self.cfg.model_name, "champion", latest.version
        )

        logger.info(f"MLflow run logged: {run_id}")
        logger.info(f"Model '{self.cfg.model_name}' v{latest.version} tagged @champion")
        logger.info(f"Metrics: {metrics}")
        return run_id
