"""SHAP-based model explainability — pure computation, no I/O."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_display: int = 15,
) -> tuple[shap.Explanation, np.ndarray]:
    """Compute SHAP values for a fitted sklearn Pipeline.

    Extracts the preprocessed feature matrix and passes it to a
    TreeExplainer backed by the pipeline's classifier.

    Args:
        pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'classifier' steps.
        X: Raw (unprocessed) input DataFrame.
        max_display: Max features to show in summary plot.

    Returns:
        Tuple of (shap.Explanation, transformed feature matrix).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_transformed)
    return shap_values, X_transformed


def save_shap_summary(
    shap_values: shap.Explanation,
    feature_names: list[str],
    path: str | Path,
    max_display: int = 15,
) -> None:
    """Save a SHAP beeswarm summary plot to disk.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: Names for each feature column.
        path: Destination PNG file path.
        max_display: Maximum number of features to display.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values.values[:, :, 1]
        if shap_values.values.ndim == 3
        else shap_values.values,
        features=shap_values.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
