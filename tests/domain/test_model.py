"""Tests for domain/model.py."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from propensity_telecom_case_study.config import ModelConfig
from propensity_telecom_case_study.domain.model import build_pipeline

NUMERIC = ["age", "tenure_months", "monthly_charges"]
CATEGORICAL = ["region"]
BINARY = ["has_streaming"]

MODEL_CFG = ModelConfig(
    n_estimators=5,
    max_depth=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=0,
    n_jobs=1,
)


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 45, 60, 30, 55],
        "tenure_months": [12, 36, 60, 6, 48],
        "monthly_charges": [30.0, 50.0, 80.0, 25.0, 70.0],
        "region": ["North", "South", "East", "North", "West"],
        "has_streaming": [0, 1, 0, 1, 1],
        "upgraded": [0, 1, 0, 0, 1],
    })


def test_build_pipeline_returns_sklearn_pipeline() -> None:
    # Given: feature lists and config
    # When: pipeline is built
    pipeline = build_pipeline(NUMERIC, CATEGORICAL, BINARY, MODEL_CFG)

    # Then: returns a Pipeline with expected steps
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_fit_predict(small_df: pd.DataFrame) -> None:
    # Given: a built pipeline and labelled data
    pipeline = build_pipeline(NUMERIC, CATEGORICAL, BINARY, MODEL_CFG)
    X = small_df[NUMERIC + CATEGORICAL + BINARY]
    y = small_df["upgraded"]

    # When: pipeline is fitted and predict_proba called
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)

    # Then: probabilities have correct shape and are valid
    assert probs.shape == (len(X), 2)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_pipeline_uses_config_hyperparams() -> None:
    # Given: a specific n_estimators setting
    cfg = ModelConfig(n_estimators=7, max_depth=2, min_samples_leaf=1,
                      class_weight="balanced", random_state=0, n_jobs=1)
    pipeline = build_pipeline(NUMERIC, CATEGORICAL, BINARY, cfg)

    # Then: classifier reflects that setting
    clf = pipeline.named_steps["classifier"]
    assert clf.n_estimators == 7
