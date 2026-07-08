"""Bundle helpers for BMRB-derived target payloads."""

from __future__ import annotations

from pathlib import Path

from atypemu.types import NMRTargetBundle


def load_bundle(path: str | Path) -> NMRTargetBundle:
    """Load one normalized BMRB bundle from JSON."""
    return NMRTargetBundle.from_json(path)


def write_bundle(bundle: NMRTargetBundle, path: str | Path) -> None:
    """Write one normalized BMRB bundle to JSON."""
    bundle.to_json(path)
