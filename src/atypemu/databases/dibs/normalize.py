"""Normalization helpers reserved for future DIBS support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_dibs(
    dibs_root: str | Path,
) -> dict[str, Any]:
    """Reserve the public normalization entrypoint for DIBS.

    Args:
        dibs_root: Source-owned ``data/dibs`` root.

    Raises:
        NotImplementedError: Always, until DIBS normalization is implemented.
    """
    raise NotImplementedError(
        "DIBS normalization is not implemented yet. The package scaffold is in place."
    )
