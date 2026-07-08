"""Normalization helpers reserved for future IDEAL support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_ideal(
    ideal_root: str | Path,
) -> dict[str, Any]:
    """Reserve the public normalization entrypoint for IDEAL.

    Args:
        ideal_root: Source-owned ``data/ideal`` root.

    Raises:
        NotImplementedError: Always, until IDEAL normalization is implemented.
    """
    raise NotImplementedError(
        "IDEAL normalization is not implemented yet. The package scaffold is in place."
    )
