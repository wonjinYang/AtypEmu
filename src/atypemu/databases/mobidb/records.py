"""Typed record definitions for future MobiDB normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MobidbEntryRecord:
    """Canonical entry-level record for future MobiDB ingestion."""

    native_id: str
    entry_uid: str
    source_id: str = "mobidb"
