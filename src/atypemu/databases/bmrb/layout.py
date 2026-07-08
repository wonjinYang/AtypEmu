"""Layout helpers for the source-owned BMRB data root."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def bmrb_download_root(root: str | Path) -> Path:
    """Return the canonical BMRB downloads directory."""
    return Path(root) / "downloads"


def bmrb_raw_nmrstar_root(root: str | Path) -> Path:
    """Return the canonical raw NMR-STAR directory."""
    return bmrb_download_root(root) / "nmrstar"


def bmrb_raw_noe_root(root: str | Path) -> Path:
    """Return the canonical raw NOE directory."""
    return bmrb_download_root(root) / "noe"


def bmrb_tables_root(root: str | Path) -> Path:
    """Return the canonical BMRB public-table directory."""
    return Path(root) / "tables"


def bmrb_assets_root(root: str | Path) -> Path:
    """Return the canonical BMRB asset directory."""
    return Path(root) / "assets"


def bmrb_bundle_root(root: str | Path) -> Path:
    """Return the canonical parsed bundle directory."""
    return bmrb_assets_root(root) / "bundles"


def bmrb_dataset_root(root: str | Path) -> Path:
    """Return the canonical BMRB dataset root."""
    return Path(root) / "datasets"


def bmrb_manifest_root(root: str | Path) -> Path:
    """Return the canonical BMRB human-readable table directory."""
    return bmrb_tables_root(root)


def migrate_legacy_bmrb_root(root: str | Path) -> dict[str, Any]:
    """Migrate a legacy BMRB root into the source-owned directory contract."""
    root_path = Path(root)
    moved_paths: list[str] = []
    legacy_mapping = {
        root_path / "raw" / "nmrstar": bmrb_raw_nmrstar_root(root_path),
        root_path / "raw" / "noe": bmrb_raw_noe_root(root_path),
        root_path / "parsed" / "bundles": bmrb_bundle_root(root_path),
        root_path / "manifests": bmrb_tables_root(root_path),
    }
    for source, destination in legacy_mapping.items():
        if not source.exists():
            continue
        _move_path(source, destination)
        moved_paths.append(str(source))

    for legacy_dir in [root_path / "raw", root_path / "parsed"]:
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    return {
        "bmrb_root": str(root_path),
        "downloads_root": str(bmrb_download_root(root_path)),
        "tables_root": str(bmrb_tables_root(root_path)),
        "assets_root": str(bmrb_assets_root(root_path)),
        "datasets_root": str(bmrb_dataset_root(root_path)),
        "moved_path_count": len(moved_paths),
    }


def _move_path(source: Path, destination: Path) -> None:
    """Move one path, merging directories when needed."""
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists() and destination.is_dir():
            for child in sorted(source.iterdir()):
                _move_path(child, destination / child.name)
            try:
                source.rmdir()
            except OSError:
                pass
            return
        shutil.move(str(source), str(destination))
        return
    if destination.exists():
        destination.unlink()
    shutil.move(str(source), str(destination))
