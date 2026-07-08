"""Typed record definitions for future IDEAL normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IdealEntryRecord:
    """Canonical entry-level record for future IDEAL ingestion."""

    native_id: str
    entry_uid: str
    source_id: str = "ideal"
