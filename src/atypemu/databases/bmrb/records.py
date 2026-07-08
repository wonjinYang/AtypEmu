"""Typed records for curated BMRB manifests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BmrbManifestRow:
    """One normalized BMRB manifest row."""

    bmrb_id: str
    status: str
    chemical_shift_count: int
    j_coupling_count: int
    noe_count: int
