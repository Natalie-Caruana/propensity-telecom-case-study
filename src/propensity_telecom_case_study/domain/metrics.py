"""Pure evaluation metric functions."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics for a binary propensity model.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with roc_auc and avg_precision scores.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }
