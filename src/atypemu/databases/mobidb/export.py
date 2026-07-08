"""Parquet export entrypoints reserved for future MobiDB support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_mobidb_dataframes(
    mobidb_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public dataframe export entrypoint for MobiDB.

    Args:
        mobidb_root: Source-owned ``data/mobidb`` root.
        refresh: Whether existing dataset outputs would be replaced.

    Raises:
        NotImplementedError: Always, until MobiDB export is implemented.
    """
    raise NotImplementedError(
        "MobiDB dataframe export is not implemented yet. The package scaffold is in place."
    )
