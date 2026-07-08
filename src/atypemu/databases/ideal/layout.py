"""Layout helpers for the IDEAL source root."""

from __future__ import annotations

from pathlib import Path


def ideal_root(data_root: str | Path) -> Path:
    """Return the IDEAL source root under ``data/``."""
    return Path(data_root) / "ideal"


def ideal_downloads_root(data_root: str | Path) -> Path:
    """Return the IDEAL downloads directory."""
    return ideal_root(data_root) / "downloads"


def ideal_tables_root(data_root: str | Path) -> Path:
    """Return the IDEAL curated-table directory."""
    return ideal_root(data_root) / "tables"


def ideal_datasets_root(data_root: str | Path) -> Path:
    """Return the IDEAL dataset root."""
    return ideal_root(data_root) / "datasets"


def ideal_assets_root(data_root: str | Path) -> Path:
    """Return the IDEAL reusable-assets directory."""
    return ideal_root(data_root) / "assets"


def ideal_debug_root(data_root: str | Path) -> Path:
    """Return the IDEAL provenance directory."""
    return ideal_root(data_root) / "_debug"
