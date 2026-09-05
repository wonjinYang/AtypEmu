"""Frozen source-subset Atom_ID eligibility for corrected K=8 validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


ELIGIBILITY_CONTRACT = "corrected_k8_source_subset_eligibility_v1"
IDENTITY_COLUMNS = ("entity_uid", "target_id", "atom_id", "target_value")


def canonical_atom_id(value: object) -> str:
    return str(value).strip().upper()


def row_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash exact entity-qualified target rows without prediction columns."""

    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing source row identity columns: {sorted(missing)}")
    rows: list[tuple[str, str, str, float]] = []
    for entity_uid, target_id, atom_id, target_value in frame[
        list(IDENTITY_COLUMNS)
    ].itertuples(index=False, name=None):
        value = float(target_value)
        if not math.isfinite(value):
            raise ValueError("source row identity contains nonfinite target")
        rows.append(
            (
                str(entity_uid),
                str(target_id),
                canonical_atom_id(atom_id),
                value,
            )
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3].hex()))
    identities = [(row[0], row[1]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate entity-qualified source row identity")
    digest = hashlib.sha256()
    for entity_uid, target_id, atom_id, target_value in rows:
        payload = json.dumps(
            [entity_uid, target_id, atom_id, target_value.hex()],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8") + b"\n")
    return digest.hexdigest()


def eligible_source_atom_ids(
    atom_ids: Sequence[str],
    target_values: Sequence[float],
    inventory: Iterable[str],
    *,
    variance_epsilon: float = 1.0e-15,
) -> tuple[str, ...]:
    """Return inventory labels with >=2 finite, variable subset targets."""

    if len(atom_ids) != len(target_values):
        raise ValueError("source atom/target length mismatch")
    ordered = tuple(canonical_atom_id(value) for value in inventory)
    if len(set(ordered)) != len(ordered):
        raise ValueError("source inventory contains duplicate labels")
    grouped = {atom_id: [] for atom_id in ordered}
    for atom_id, target in zip(atom_ids, target_values, strict=True):
        value = float(target)
        atom_id = canonical_atom_id(atom_id)
        if atom_id in grouped and math.isfinite(value):
            grouped[atom_id].append(value)
    return tuple(
        atom_id
        for atom_id in ordered
        if len(grouped[atom_id]) >= 2
        and float(np.var(grouped[atom_id])) > variance_epsilon
    )


def source_eligible_row_mask(
    atom_ids: Sequence[str],
    target_values: Sequence[float],
    inventory: Iterable[str],
) -> tuple[bool, ...]:
    eligible = set(eligible_source_atom_ids(atom_ids, target_values, inventory))
    return tuple(
        canonical_atom_id(atom_id) in eligible and math.isfinite(float(target))
        for atom_id, target in zip(atom_ids, target_values, strict=True)
    )


def build_eligibility_receipt(
    input_frame: pd.DataFrame,
    selected_frame: pd.DataFrame,
    *,
    frozen_atom_ids: Iterable[str],
    fold: str,
    held_half: int,
    role: str,
) -> dict[str, Any]:
    """Bind the exact rows admitted to one fitting or assimilation subset."""

    frozen = tuple(canonical_atom_id(value) for value in frozen_atom_ids)
    atom_ids = input_frame["atom_id"].astype(str).tolist()
    targets = input_frame["target_value"].astype(float).tolist()
    eligible = eligible_source_atom_ids(atom_ids, targets, frozen)
    expected = np.asarray(
        source_eligible_row_mask(atom_ids, targets, frozen), dtype=bool
    )
    expected_frame = input_frame.loc[expected].reset_index(drop=True)
    if row_identity_sha256(selected_frame) != row_identity_sha256(expected_frame):
        raise ValueError("selected source rows do not equal target-only eligibility")
    selected_atoms = set(selected_frame["atom_id"].map(canonical_atom_id))
    if selected_atoms != set(eligible):
        raise ValueError("selected source labels do not equal eligible labels")
    excluded_frame = input_frame.loc[~expected].reset_index(drop=True)
    return {
        "contract": ELIGIBILITY_CONTRACT,
        "fold": str(fold),
        "held_half": int(held_half),
        "role": str(role),
        "rule": "frozen_inventory_and_finite_count_ge_2_and_population_variance_gt_1e-15",
        "frozen_atom_ids": list(frozen),
        "eligible_atom_ids": list(eligible),
        "input_row_count": int(len(input_frame)),
        "input_row_identity_sha256": row_identity_sha256(input_frame),
        "selected_row_count": int(len(selected_frame)),
        "selected_row_identity_sha256": row_identity_sha256(selected_frame),
        "excluded_row_count": int(len(excluded_frame)),
        "excluded_row_identity_sha256": row_identity_sha256(excluded_frame),
    }


def validate_eligibility_receipt(
    frame: pd.DataFrame, receipt: dict[str, Any]
) -> None:
    """Fail closed unless a model phase sees exactly its receipted rows."""

    if receipt.get("contract") != ELIGIBILITY_CONTRACT:
        raise ValueError("source eligibility receipt contract mismatch")
    if int(receipt.get("selected_row_count", -1)) != len(frame):
        raise ValueError("source eligibility selected-row count mismatch")
    if receipt.get("selected_row_identity_sha256") != row_identity_sha256(frame):
        raise ValueError("source eligibility selected-row identity mismatch")
    frozen = tuple(receipt.get("frozen_atom_ids", ()))
    eligible = tuple(receipt.get("eligible_atom_ids", ()))
    if eligible_source_atom_ids(
        frame["atom_id"].astype(str).tolist(),
        frame["target_value"].astype(float).tolist(),
        frozen,
    ) != eligible:
        raise ValueError("source eligibility labels are not defined on model rows")
    atoms = set(frame["atom_id"].map(canonical_atom_id))
    if atoms != set(eligible):
        raise ValueError("model phase contains rows outside source eligibility")
    if len(frame) == 0 or not np.isfinite(
        pd.to_numeric(frame["target_value"], errors="coerce").to_numpy(float)
    ).all():
        raise ValueError("model phase has no finite eligible source rows")
