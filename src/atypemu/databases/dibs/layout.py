"""Layout helpers for the DIBS source root."""

from __future__ import annotations

from pathlib import Path


def dibs_root(data_root: str | Path) -> Path:
    """Return the DIBS source root under ``data/``."""
    return Path(data_root) / "dibs"


def dibs_downloads_root(data_root: str | Path) -> Path:
    """Return the DIBS downloads directory."""
    return dibs_root(data_root) / "downloads"


def dibs_tables_root(data_root: str | Path) -> Path:
    """Return the DIBS curated-table directory."""
    return dibs_root(data_root) / "tables"


def dibs_datasets_root(data_root: str | Path) -> Path:
    """Return the DIBS dataset root."""
    return dibs_root(data_root) / "datasets"


def dibs_assets_root(data_root: str | Path) -> Path:
    """Return the DIBS reusable-assets directory."""
    return dibs_root(data_root) / "assets"


def dibs_debug_root(data_root: str | Path) -> Path:
    """Return the DIBS provenance directory."""
    return dibs_root(data_root) / "_debug"
