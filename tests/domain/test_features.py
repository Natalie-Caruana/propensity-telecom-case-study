"""Tests for domain/features.py."""

import numpy as np
import pandas as pd
import pytest

from propensity_telecom_case_study.domain.features import (
    build_preprocessor,
    get_feature_names,
)

NUMERIC = ["age", "tenure_months"]
CATEGORICAL = ["region"]
BINARY = ["has_streaming"]


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 45, 60],
        "tenure_months": [12, 36, 60],
        "region": ["North", "South", "East"],
        "has_streaming": [0, 1, 0],
    })


def test_build_preprocessor_returns_column_transformer(small_df: pd.DataFrame) -> None:
    # Given: valid feature lists
    # When: preprocessor is built and fitted
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL, BINARY)
    preprocessor.fit(small_df)

    # Then: output shape is correct (2 numeric + 1 binary + 3 one-hot = 6)
    out = preprocessor.transform(small_df)
    assert out.shape == (3, 6)


def test_preprocessor_scales_numeric(small_df: pd.DataFrame) -> None:
    # Given: a fitted preprocessor
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL, BINARY)
    preprocessor.fit(small_df)

    # When: transformed on same data
    out = preprocessor.transform(small_df)

    # Then: numeric columns are z-score scaled (mean ~0)
    numeric_out = out[:, :2]
    assert abs(numeric_out.mean()) < 1.0


def test_preprocessor_handles_unknown_category(small_df: pd.DataFrame) -> None:
    # Given: a preprocessor fitted without "West"
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL, BINARY)
    preprocessor.fit(small_df)

    # When: transform is called with an unseen category
    new_df = small_df.copy()
    new_df["region"] = "West"

    # Then: no error is raised (handle_unknown="ignore")
    out = preprocessor.transform(new_df)
    assert out.shape[0] == 3


def test_get_feature_names_length(small_df: pd.DataFrame) -> None:
    # Given: a fitted preprocessor
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL, BINARY)
    preprocessor.fit(small_df)

    # When: feature names are retrieved
    names = get_feature_names(preprocessor, NUMERIC, CATEGORICAL, BINARY)

    # Then: length matches output columns
    out = preprocessor.transform(small_df)
    assert len(names) == out.shape[1]


def test_preprocessor_no_nan_in_output(small_df: pd.DataFrame) -> None:
    # Given: input with a missing value
    small_df.loc[0, "age"] = float("nan")
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL, BINARY)
    preprocessor.fit(small_df)

    # When: transform is called
    out = preprocessor.transform(small_df)

    # Then: no NaN in output (imputer fills them)
    assert not np.isnan(out).any()
