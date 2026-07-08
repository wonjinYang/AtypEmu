"""Typed record definitions for future DIBS normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DibsEntryRecord:
    """Canonical entry-level record for future DIBS ingestion."""

    native_id: str
    entry_uid: str
    source_id: str = "dibs"
