"""Collection entrypoints for the MobiDB source package."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def crawl_mobidb(
    data_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public crawl entrypoint for future MobiDB support.

    Args:
        data_root: Repository data root containing ``data/mobidb``.
        refresh: Whether existing source-owned artifacts would be replaced.

    Raises:
        NotImplementedError: Always, until MobiDB ingestion is implemented.
    """
    raise NotImplementedError(
        "MobiDB crawl is not implemented yet. The package scaffold is in place."
    )
