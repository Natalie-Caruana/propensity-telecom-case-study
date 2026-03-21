"""Drift detection using Evidently — pure computation, no I/O."""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
) -> Snapshot:
    """Compute a data drift report comparing reference vs current distributions.

    Args:
        reference: Training data (reference distribution).
        current: Production/scoring data (current distribution).
        columns: Feature columns to include in the drift analysis.

    Returns:
        Evidently Snapshot with drift results.
    """
    definition = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur_dataset = Dataset.from_pandas(current[columns], data_definition=definition)

    report = Report([DataDriftPreset()])
    return report.run(reference_data=ref_dataset, current_data=cur_dataset)


def save_drift_report(snapshot: Snapshot, path: str | Path) -> None:
    """Save an Evidently drift snapshot to an HTML file.

    Args:
        snapshot: Evidently Snapshot returned by Report.run().
        path: Destination HTML file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(dest))
