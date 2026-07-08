"""Typed records for curated PED tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PedWorkspaceRecord:
    """Minimal metadata record for one PED workspace."""

    workspace: str
    public_root: str
    debug_root: str
