"""Data loading and saving — all file I/O lives here."""

from pathlib import Path

import pandas as pd
from loguru import logger


class DatasetLoader:
    """Loads raw telecom customer data from disk.

    Args:
        raw_path: Path to the raw CSV file.
    """

    def __init__(self, raw_path: str | Path) -> None:
        self.raw_path = Path(raw_path)

    def load(self) -> pd.DataFrame:
        """Read the raw dataset from disk.

        Returns:
            DataFrame with all raw columns.

        Raises:
            FileNotFoundError: If the raw file does not exist.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {self.raw_path}")
        logger.info(f"Loading data from {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def save_processed(self, df: pd.DataFrame, path: str | Path) -> None:
        """Persist a processed DataFrame as Parquet.

        Args:
            df: DataFrame to save.
            path: Destination file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        logger.info(f"Saved processed data to {dest}  ({len(df):,} rows)")
