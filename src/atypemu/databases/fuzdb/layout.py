"""Layout helpers for the source-owned FuzDB data root."""

from __future__ import annotations

from pathlib import Path


def fuzdb_root(data_root: str | Path) -> Path:
    """Return the canonical FuzDB source root below ``data/``."""
    root = Path(data_root)
    if root.name == "fuzdb":
        return root
    return root / "fuzdb"


def fuzdb_download_root(root: str | Path) -> Path:
    """Return the FuzDB download root."""
    return Path(root) / "downloads"


def fuzdb_entries_download_path(root: str | Path) -> Path:
    """Return the raw FuzDB entries JSON path."""
    return fuzdb_download_root(root) / "entries.json"


def fuzdb_entries_tsv_download_path(root: str | Path) -> Path:
    """Return the raw FuzDB TSV export path."""
    return fuzdb_download_root(root) / "entries.tsv"


def fuzdb_entries_xml_download_path(root: str | Path) -> Path:
    """Return the raw FuzDB XML export path."""
    return fuzdb_download_root(root) / "entries.xml"


def fuzdb_entries_txt_download_path(root: str | Path) -> Path:
    """Return the raw FuzDB TXT export path."""
    return fuzdb_download_root(root) / "entries.txt"


def fuzdb_tables_root(root: str | Path) -> Path:
    """Return the curated table root."""
    return Path(root) / "tables"


def fuzdb_assets_root(root: str | Path) -> Path:
    """Return the reusable asset root."""
    return Path(root) / "assets"


def fuzdb_dataset_root(root: str | Path) -> Path:
    """Return the source-owned dataset root."""
    return Path(root) / "datasets"


def fuzdb_workspace_dataset_root(
    root: str | Path,
    workspace: str = "default",
) -> Path:
    """Return the dataset workspace root."""
    return fuzdb_dataset_root(root) / workspace


def fuzdb_debug_root(root: str | Path) -> Path:
    """Return the debug/provenance root."""
    return Path(root) / "_debug"


def fuzdb_debug_pages_root(root: str | Path) -> Path:
    """Return the debug HTML snapshot root."""
    return fuzdb_debug_root(root) / "pages"


def fuzdb_browse_debug_root(root: str | Path) -> Path:
    """Return the browse-page debug root."""
    return fuzdb_debug_pages_root(root) / "browse"


def fuzdb_entry_debug_root(root: str | Path) -> Path:
    """Return the entry-page debug root."""
    return fuzdb_debug_pages_root(root) / "entries"


def ensure_fuzdb_workspace(root: str | Path) -> None:
    """Create the canonical FuzDB source subtree."""
    for path in [
        fuzdb_download_root(root),
        fuzdb_tables_root(root),
        fuzdb_assets_root(root),
        fuzdb_dataset_root(root),
        fuzdb_workspace_dataset_root(root),
        fuzdb_debug_root(root),
        fuzdb_debug_pages_root(root),
        fuzdb_browse_debug_root(root),
        fuzdb_entry_debug_root(root),
    ]:
        path.mkdir(parents=True, exist_ok=True)
