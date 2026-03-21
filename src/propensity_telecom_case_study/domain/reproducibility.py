"""Reproducibility helpers — seed fixing and run provenance."""

import random
import subprocess  # nosec B404

import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: The integer seed value to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo.

    Returns:
        Short 8-character commit hash, or 'unknown'.
    """
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
