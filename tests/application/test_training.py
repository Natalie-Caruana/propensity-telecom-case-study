"""Integration tests for application/training.py."""

from propensity_telecom_case_study.application.training import run_training
from propensity_telecom_case_study.config import TrainConfig


def test_run_training_returns_metrics(train_config: TrainConfig) -> None:
    # Given: a valid TrainConfig pointing at temp data and mlruns
    # When: training runs end-to-end
    metrics = run_training(train_config)

    # Then: expected metric keys are present and values are in range
    assert "roc_auc" in metrics
    assert "avg_precision" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["avg_precision"] <= 1.0


def test_run_training_logs_to_mlflow(train_config: TrainConfig) -> None:
    # Given: training has run
    import mlflow

    mlflow.set_tracking_uri(train_config.mlflow.tracking_uri)
    run_training(train_config)

    # When: querying the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(train_config.mlflow.experiment_name)

    # Then: at least one run was created
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1
