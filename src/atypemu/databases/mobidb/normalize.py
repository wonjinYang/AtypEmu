"""Normalization helpers reserved for future MobiDB support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_mobidb(
    mobidb_root: str | Path,
) -> dict[str, Any]:
    """Reserve the public normalization entrypoint for MobiDB.

    Args:
        mobidb_root: Source-owned ``data/mobidb`` root.

    Raises:
        NotImplementedError: Always, until MobiDB normalization is implemented.
    """
    raise NotImplementedError(
        "MobiDB normalization is not implemented yet. The package scaffold is in place."
    )
