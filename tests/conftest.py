"""Shared pytest fixtures."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from propensity_telecom_case_study.config import (
    DataConfig,
    FeaturesConfig,
    MLflowConfig,
    ModelConfig,
    TrainConfig,
)

NUMERIC_FEATURES = [
    "age", "tenure_months", "monthly_charges",
    "data_usage_gb", "call_minutes", "num_products",
    "num_complaints", "customer_service_calls",
]
CATEGORICAL_FEATURES = ["region", "contract_type", "internet_service"]
BINARY_FEATURES = ["has_streaming", "has_device_protection"]


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small deterministic DataFrame matching the raw telecom schema."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": rng.integers(18, 76, n),
        "tenure_months": rng.integers(1, 121, n),
        "monthly_charges": rng.uniform(20, 100, n).round(2),
        "data_usage_gb": rng.uniform(0, 30, n).round(2),
        "call_minutes": rng.integers(0, 1001, n),
        "num_products": rng.integers(1, 5, n),
        "num_complaints": rng.integers(0, 6, n),
        "customer_service_calls": rng.integers(0, 11, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n),
        "internet_service": rng.choice(["Fiber", "DSL", "None"], n),
        "has_streaming": rng.integers(0, 2, n),
        "has_device_protection": rng.integers(0, 2, n),
        "upgraded": rng.integers(0, 2, n),
    })


@pytest.fixture()
def sample_csv(sample_df: pd.DataFrame, tmp_path: pathlib.Path) -> pathlib.Path:
    """Write sample_df to a temporary CSV and return its path."""
    path = tmp_path / "telecom_customers.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def train_config(sample_csv: pathlib.Path, tmp_path: pathlib.Path) -> TrainConfig:
    """Minimal TrainConfig pointing at the temp CSV and a temp mlruns dir."""
    return TrainConfig(
        data=DataConfig(
            raw_path=str(sample_csv),
            processed_path=str(tmp_path / "features.parquet"),
            test_size=0.2,
            random_state=42,
        ),
        features=FeaturesConfig(
            numeric=NUMERIC_FEATURES,
            categorical=CATEGORICAL_FEATURES,
            binary=BINARY_FEATURES,
            target="upgraded",
        ),
        model=ModelConfig(
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        mlflow=MLflowConfig(
            experiment_name="test-experiment",
            model_name="test-model",
            tracking_uri=(tmp_path / "mlruns").as_uri(),
        ),
    )
