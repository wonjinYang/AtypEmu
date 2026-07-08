"""Common layout helpers for database-oriented source packages."""

from __future__ import annotations

from pathlib import Path

from atypemu.databases.common.contracts import DatabaseId, DatabaseLayout


def build_database_layout(
    root: str | Path,
    source_id: DatabaseId,
    *,
    public_subdir: str | None = None,
    debug_subdir: str = "_debug",
    raw_subdir: str | None = "raw",
) -> DatabaseLayout:
    """Build a normalized layout object for one database source."""
    root_path = Path(root)
    public_root = root_path if public_subdir is None else root_path / public_subdir
    debug_root = root_path / debug_subdir
    raw_root = None if raw_subdir is None else root_path / raw_subdir
    return DatabaseLayout(
        source_id=source_id,
        root=root_path,
        public_root=public_root,
        debug_root=debug_root,
        raw_root=raw_root,
    )
