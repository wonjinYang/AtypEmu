"""Frozen source-fold label eligibility shared by corrected K=8 validation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def eligible_source_atom_ids(
    atom_ids: Sequence[str],
    target_values: Sequence[float],
    inventory: Iterable[str],
) -> tuple[str, ...]:
    """Return inventory labels with >=2 finite, nonconstant source targets."""

    if len(atom_ids) != len(target_values):
        raise ValueError("source atom/target length mismatch")
    ordered = tuple(inventory)
    if len(set(ordered)) != len(ordered):
        raise ValueError("source inventory contains duplicate labels")
    grouped = {atom_id: [] for atom_id in ordered}
    for atom_id, target in zip(atom_ids, target_values, strict=True):
        value = float(target)
        if atom_id in grouped and math.isfinite(value):
            grouped[atom_id].append(value)
    return tuple(
        atom_id
        for atom_id in ordered
        if len(grouped[atom_id]) >= 2
        and min(grouped[atom_id]) != max(grouped[atom_id])
    )


def source_eligible_row_mask(
    atom_ids: Sequence[str],
    target_values: Sequence[float],
    inventory: Iterable[str],
) -> tuple[bool, ...]:
    eligible = set(eligible_source_atom_ids(atom_ids, target_values, inventory))
    return tuple(
        atom_id in eligible and math.isfinite(float(target))
        for atom_id, target in zip(atom_ids, target_values, strict=True)
    )
