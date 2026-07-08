"""Layout helpers for the source-owned MFIB data root."""

from __future__ import annotations

from pathlib import Path


def mfib_root(data_root: str | Path) -> Path:
    """Return the canonical MFIB source root below ``data/``."""
    root = Path(data_root)
    if root.name == "mfib":
        return root
    return root / "mfib"


def mfib_download_root(root: str | Path) -> Path:
    """Return the MFIB download root."""
    return Path(root) / "downloads"


def mfib_tables_root(root: str | Path) -> Path:
    """Return the curated table root."""
    return Path(root) / "tables"


def mfib_assets_root(root: str | Path) -> Path:
    """Return the reusable asset root."""
    return Path(root) / "assets"


def mfib_cif_root(root: str | Path) -> Path:
    """Return the extracted CIF asset root."""
    return mfib_assets_root(root) / "cif"


def mfib_geometry_root(root: str | Path) -> Path:
    """Return the heavy geometry asset root."""
    return mfib_assets_root(root) / "geometry"


def mfib_contact_map_root(root: str | Path) -> Path:
    """Return the contact-map asset root."""
    return mfib_geometry_root(root) / "contact_maps"


def mfib_interface_residue_set_root(root: str | Path) -> Path:
    """Return the interface-residue-set asset root."""
    return mfib_geometry_root(root) / "interface_residue_sets"


def mfib_dataset_root(root: str | Path) -> Path:
    """Return the source-owned dataset root."""
    return Path(root) / "datasets"


def mfib_workspace_dataset_root(root: str | Path, workspace: str = "default") -> Path:
    """Return the dataset workspace root."""
    return mfib_dataset_root(root) / workspace


def mfib_debug_root(root: str | Path) -> Path:
    """Return the debug/provenance root."""
    return Path(root) / "_debug"


def ensure_mfib_workspace(root: str | Path) -> None:
    """Create the canonical MFIB source subtree."""
    for path in [
        mfib_download_root(root),
        mfib_tables_root(root),
        mfib_assets_root(root),
        mfib_cif_root(root),
        mfib_geometry_root(root),
        mfib_contact_map_root(root),
        mfib_interface_residue_set_root(root),
        mfib_dataset_root(root),
        mfib_workspace_dataset_root(root),
        mfib_debug_root(root),
    ]:
        path.mkdir(parents=True, exist_ok=True)
