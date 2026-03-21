"""Tests for io/datasets.py."""

import pathlib

import pandas as pd
import pytest

from propensity_telecom_case_study.io.datasets import DatasetLoader


def test_load_returns_dataframe(sample_csv: pathlib.Path) -> None:
    # Given: a valid CSV on disk
    # When: DatasetLoader.load() is called
    loader = DatasetLoader(sample_csv)
    df = loader.load()

    # Then: returns a non-empty DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_load_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    # Given: a path that does not exist
    missing = tmp_path / "ghost.csv"

    # When / Then: FileNotFoundError is raised
    with pytest.raises(FileNotFoundError, match="Raw data not found"):
        DatasetLoader(missing).load()


def test_save_processed_creates_parquet(
    sample_csv: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # Given: a loaded DataFrame
    loader = DatasetLoader(sample_csv)
    df = loader.load()

    # When: saved as parquet
    out = tmp_path / "sub" / "features.parquet"
    loader.save_processed(df, out)

    # Then: file exists and can be read back
    assert out.exists()
    df_back = pd.read_parquet(out)
    assert df_back.shape == df.shape
