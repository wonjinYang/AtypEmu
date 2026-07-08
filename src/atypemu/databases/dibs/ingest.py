"""Collection entrypoints for the DIBS source package."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def crawl_dibs(
    data_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public crawl entrypoint for future DIBS support.

    Args:
        data_root: Repository data root containing ``data/dibs``.
        refresh: Whether existing source-owned artifacts would be replaced.

    Raises:
        NotImplementedError: Always, until DIBS ingestion is implemented.
    """
    raise NotImplementedError(
        "DIBS crawl is not implemented yet. The package scaffold is in place."
    )
