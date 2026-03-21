"""Model construction — pure, no I/O."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from propensity_telecom_case_study.config import ModelConfig
from propensity_telecom_case_study.domain.features import build_preprocessor


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    cfg: ModelConfig,
    cache_dir: str | None = None,
) -> Pipeline:
    """Assemble the full preprocessing + classifier pipeline.

    Args:
        numeric_features: Continuous numeric column names.
        categorical_features: Nominal categorical column names.
        binary_features: Binary 0/1 column names.
        cfg: Validated model hyperparameter config.
        cache_dir: Optional path for joblib pipeline caching.

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, binary_features
    )

    classifier = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)],
        memory=cache_dir,
    )
