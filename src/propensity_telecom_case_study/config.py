"""Pydantic schemas and OmegaConf loader for training configuration."""

from pathlib import Path

import omegaconf
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_path: str = "data/raw/telecom_customers.csv"
    processed_path: str = "data/processed/features.parquet"
    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42


class FeaturesConfig(BaseModel):
    numeric: list[str]
    categorical: list[str]
    binary: list[str]
    target: str = "upgraded"


class ModelConfig(BaseModel):
    n_estimators: int = Field(200, gt=0)
    max_depth: int = Field(6, gt=0)
    min_samples_leaf: int = Field(5, gt=0)
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


class MLflowConfig(BaseModel):
    experiment_name: str = "propensity-telecom"
    model_name: str = "propensity-rf"
    tracking_uri: str = "mlruns"


class TrainConfig(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    mlflow: MLflowConfig


def load_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    """Load and validate training config from a YAML file.

    Merges YAML with any CLI overrides (e.g. model.max_depth=8).

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated TrainConfig instance.
    """
    file_conf = omegaconf.OmegaConf.load(path)
    cli_conf = omegaconf.OmegaConf.from_cli()
    merged = omegaconf.OmegaConf.merge(file_conf, cli_conf)
    return TrainConfig(**omegaconf.OmegaConf.to_container(merged, resolve=True))  # type: ignore[arg-type]
