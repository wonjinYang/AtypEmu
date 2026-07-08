"""Parquet export entrypoints reserved for future IDEAL support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_ideal_dataframes(
    ideal_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public dataframe export entrypoint for IDEAL.

    Args:
        ideal_root: Source-owned ``data/ideal`` root.
        refresh: Whether existing dataset outputs would be replaced.

    Raises:
        NotImplementedError: Always, until IDEAL export is implemented.
    """
    raise NotImplementedError(
        "IDEAL dataframe export is not implemented yet. The package scaffold is in place."
    )
