"""Pure feature engineering functions — no I/O, no side effects."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric and encodes categorical features.

    Args:
        numeric_features: Names of continuous numeric columns.
        categorical_features: Names of nominal categorical columns.
        binary_features: Names of 0/1 binary columns (treated as numeric).

    Returns:
        Unfitted ColumnTransformer ready to be placed in a Pipeline.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features + binary_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
) -> list[str]:
    """Return feature names after the ColumnTransformer has been fitted.

    Args:
        preprocessor: A fitted ColumnTransformer.
        numeric_features: Original numeric column names.
        categorical_features: Original categorical column names.
        binary_features: Original binary column names.

    Returns:
        Ordered list of output feature names (numeric + binary + one-hot encoded).
    """
    cat_names: list[str] = list(
        preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
            categorical_features
        )
    )
    return numeric_features + binary_features + cat_names
