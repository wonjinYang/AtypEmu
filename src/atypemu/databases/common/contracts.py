"""Shared database contracts for source-specific AtypEmu packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


DatabaseId = Literal[
    "bmrb",
    "ped",
    "mobidb",
    "sasbdb",
    "dibs",
    "mfib",
    "ideal",
    "fuzdb",
]


@dataclass(slots=True)
class DatabaseLayout:
    """Resolved public, debug, and raw roots for one database source."""

    source_id: DatabaseId
    root: Path
    public_root: Path
    debug_root: Path
    raw_root: Path | None = None


@dataclass(slots=True)
class DatabaseTableSpec:
    """Schema metadata for one curated database table."""

    name: str
    primary_key: list[str]
    join_keys: list[str]
    level: Literal["entry", "model", "tag", "derived"]
    columns: list[str]


@dataclass(slots=True)
class DatabaseWorkspaceMeta:
    """Workspace-level metadata for one database source."""

    source_id: DatabaseId
    workspace: str | None
    available_tables: dict[str, DatabaseTableSpec] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


class DatabaseCollector(Protocol):
    """Protocol for raw collection or crawl entrypoints."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Collect source data and return a summary payload."""


class DatabaseNormalizer(Protocol):
    """Protocol for raw-to-curated normalization entrypoints."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Normalize raw source data into curated public tables."""


class DatabaseExporter(Protocol):
    """Protocol for optional source-specific downstream exports."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Export curated data into derived downstream artifacts."""
