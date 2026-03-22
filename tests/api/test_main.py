"""Tests for the FastAPI inference server."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from propensity_telecom_case_study.api.main import app
from propensity_telecom_case_study.config import (
    DataConfig,
    FeaturesConfig,
    MLflowConfig,
    ModelConfig,
    TrainConfig,
)

NUMERIC = [
    "age",
    "tenure_months",
    "monthly_charges",
    "data_usage_gb",
    "call_minutes",
    "num_products",
    "num_complaints",
    "customer_service_calls",
]
CATEGORICAL = ["region", "contract_type", "internet_service"]
BINARY = ["has_streaming", "has_device_protection"]

SAMPLE_CUSTOMER = {
    "age": 35.0,
    "tenure_months": 24.0,
    "monthly_charges": 65.0,
    "data_usage_gb": 10.0,
    "call_minutes": 200.0,
    "num_products": 2,
    "num_complaints": 0,
    "customer_service_calls": 1,
    "region": "North",
    "contract_type": "monthly",
    "internet_service": "fiber",
    "has_streaming": 1,
    "has_device_protection": 0,
}


@pytest.fixture()
def client() -> TestClient:
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.side_effect = lambda X: np.column_stack(
        [np.full(len(X), 0.3), np.full(len(X), 0.7)]
    )

    mock_cfg = TrainConfig(
        data=DataConfig(raw_path="data/raw/telecom_customers.csv"),
        features=FeaturesConfig(
            numeric=NUMERIC, categorical=CATEGORICAL, binary=BINARY
        ),
        model=ModelConfig(),
        mlflow=MLflowConfig(),
    )

    cfg_path = "propensity_telecom_case_study.api.main.load_config"
    model_path = "propensity_telecom_case_study.api.main.mlflow.sklearn.load_model"
    with (
        patch(cfg_path, return_value=mock_cfg),
        patch(model_path, return_value=mock_pipeline),
        TestClient(app) as c,
    ):
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "propensity-rf"


def test_predict_single_customer(client: TestClient) -> None:
    response = client.post("/predict", json={"customers": [SAMPLE_CUSTOMER]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert len(body["predictions"]) == 1
    score = body["predictions"][0]["propensity_score"]
    assert 0.0 <= score <= 1.0


def test_predict_batch(client: TestClient) -> None:
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_empty_list_rejected(client: TestClient) -> None:
    response = client.post("/predict", json={"customers": []})
    assert response.status_code == 422


def test_predict_missing_field_rejected(client: TestClient) -> None:
    bad_customer = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "age"}
    response = client.post("/predict", json={"customers": [bad_customer]})
    assert response.status_code == 422
