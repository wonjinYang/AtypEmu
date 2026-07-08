"""Collection entrypoints for the IDEAL source package."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def crawl_ideal(
    data_root: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Reserve the public crawl entrypoint for future IDEAL support.

    Args:
        data_root: Repository data root containing ``data/ideal``.
        refresh: Whether existing source-owned artifacts would be replaced.

    Raises:
        NotImplementedError: Always, until IDEAL ingestion is implemented.
    """
    raise NotImplementedError(
        "IDEAL crawl is not implemented yet. The package scaffold is in place."
    )
