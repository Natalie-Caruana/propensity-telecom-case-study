"""Tests for domain/metrics.py."""

import numpy as np
import pytest

from propensity_telecom_case_study.domain.metrics import compute_metrics


def test_compute_metrics_perfect_model() -> None:
    # Given: perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.1, 0.9, 1.0])

    # When: metrics are computed
    result = compute_metrics(y_true, y_prob)

    # Then: both scores are 1.0
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["avg_precision"] == pytest.approx(1.0)


def test_compute_metrics_random_model() -> None:
    # Given: random (uninformative) predictions
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)

    # When: metrics are computed
    result = compute_metrics(y_true, y_prob)

    # Then: ROC-AUC is near 0.5 for random predictions
    assert 0.3 < result["roc_auc"] < 0.7


def test_compute_metrics_returns_floats() -> None:
    # Given: valid inputs
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])

    # When: metrics computed
    result = compute_metrics(y_true, y_prob)

    # Then: values are Python floats in [0, 1]
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float"
        assert 0.0 <= val <= 1.0, f"{key} out of range"


def test_compute_metrics_keys() -> None:
    # Given / When
    result = compute_metrics(np.array([0, 1]), np.array([0.3, 0.7]))

    # Then: expected keys present
    assert "roc_auc" in result
    assert "avg_precision" in result
