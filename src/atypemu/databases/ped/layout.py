"""Layout helpers for the source-owned PED data root."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


PED_WORKSPACES = ("benchmark", "catalog")


def normalize_ped_workspace(workspace: str) -> str:
    """Normalize and validate a PED workspace name.

    Args:
        workspace: Requested workspace name.

    Returns:
        Lowercase validated workspace name.

    Raises:
        ValueError: If the workspace is unsupported.
    """
    normalized = workspace.strip().lower()
    if normalized not in PED_WORKSPACES:
        raise ValueError(
            f"Unsupported PED workspace '{workspace}'. "
            f"Expected one of: {', '.join(PED_WORKSPACES)}."
        )
    return normalized


def ped_tables_root(ped_root: str | Path, workspace: str) -> Path:
    """Return the canonical human-facing table root for one PED workspace."""
    return Path(ped_root) / "tables" / normalize_ped_workspace(workspace)


def ped_assets_root(ped_root: str | Path, workspace: str) -> Path:
    """Return the canonical asset root for one PED workspace."""
    return Path(ped_root) / "assets" / normalize_ped_workspace(workspace)


def ped_models_root(ped_root: str | Path, workspace: str) -> Path:
    """Return the canonical model asset root for one PED workspace."""
    return ped_assets_root(ped_root, workspace) / "models"


def ped_debug_root(ped_root: str | Path, workspace: str) -> Path:
    """Return the debug root for one PED workspace."""
    return Path(ped_root) / "_debug" / normalize_ped_workspace(workspace)


def ped_dataset_root(ped_root: str | Path) -> Path:
    """Return the canonical PED dataset root."""
    return Path(ped_root) / "datasets"


def ped_workspace_dataset_root(ped_root: str | Path, workspace: str) -> Path:
    """Return the canonical dataset workspace root for one PED workspace."""
    return ped_dataset_root(ped_root) / normalize_ped_workspace(workspace)


def ensure_ped_workspace(ped_root: str | Path, workspace: str) -> tuple[Path, Path]:
    """Create the canonical table and debug roots for one PED workspace."""
    tables_root = ped_tables_root(ped_root, workspace)
    debug_root = ped_debug_root(ped_root, workspace)
    ped_assets_root(ped_root, workspace).mkdir(parents=True, exist_ok=True)
    ped_workspace_dataset_root(ped_root, workspace).mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    return tables_root, debug_root


def resolve_ped_table(ped_root: str | Path, workspace: str, filename: str) -> Path:
    """Return the canonical path for one PED public table."""
    return ped_tables_root(ped_root, workspace) / filename


def migrate_legacy_ped_source_root(ped_root: str | Path) -> dict[str, Any]:
    """Migrate a legacy PED root into the source-owned directory contract.

    Args:
        ped_root: Existing PED root that may still contain top-level
            ``benchmark`` or ``catalog`` directories.

    Returns:
        Summary dictionary describing the migration.
    """
    root = Path(ped_root)
    moved_paths: list[str] = []

    for workspace in PED_WORKSPACES:
        legacy_workspace_root = root / workspace
        if not legacy_workspace_root.exists():
            continue

        tables_root, _ = ensure_ped_workspace(root, workspace)
        legacy_models_root = legacy_workspace_root / "models"
        if legacy_models_root.exists():
            destination = ped_models_root(root, workspace)
            _move_path(legacy_models_root, destination)
            moved_paths.append(str(legacy_models_root))

        for child in sorted(legacy_workspace_root.iterdir()):
            if child.name == "models":
                continue
            _move_path(child, tables_root / child.name)
            moved_paths.append(str(child))

        try:
            legacy_workspace_root.rmdir()
        except OSError:
            pass

    return {
        "ped_root": str(root),
        "tables_root": str(root / "tables"),
        "assets_root": str(root / "assets"),
        "debug_root": str(root / "_debug"),
        "datasets_root": str(root / "datasets"),
        "moved_path_count": len(moved_paths),
    }


def _move_path(source: Path, destination: Path) -> None:
    """Move one file or directory, merging directories when needed."""
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
