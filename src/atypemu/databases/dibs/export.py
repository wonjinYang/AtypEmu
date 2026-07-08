"""Parquet export entrypoints reserved for future DIBS support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_dibs_dataframes(
    dibs_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public dataframe export entrypoint for DIBS.

    Args:
        dibs_root: Source-owned ``data/dibs`` root.
        refresh: Whether existing dataset outputs would be replaced.

    Raises:
        NotImplementedError: Always, until DIBS export is implemented.
    """
    raise NotImplementedError(
        "DIBS dataframe export is not implemented yet. The package scaffold is in place."
    )
