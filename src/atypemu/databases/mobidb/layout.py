"""Layout helpers for the MobiDB source root."""

from __future__ import annotations

from pathlib import Path


def mobidb_root(data_root: str | Path) -> Path:
    """Return the MobiDB source root under ``data/``."""
    return Path(data_root) / "mobidb"


def mobidb_downloads_root(data_root: str | Path) -> Path:
    """Return the MobiDB downloads directory."""
    return mobidb_root(data_root) / "downloads"


def mobidb_tables_root(data_root: str | Path) -> Path:
    """Return the MobiDB curated-table directory."""
    return mobidb_root(data_root) / "tables"


def mobidb_datasets_root(data_root: str | Path) -> Path:
    """Return the MobiDB dataset root."""
    return mobidb_root(data_root) / "datasets"


def mobidb_assets_root(data_root: str | Path) -> Path:
    """Return the MobiDB reusable-assets directory."""
    return mobidb_root(data_root) / "assets"


def mobidb_debug_root(data_root: str | Path) -> Path:
    """Return the MobiDB provenance directory."""
    return mobidb_root(data_root) / "_debug"
