#!/usr/bin/env python3
"""Independent arithmetic/schema checker for matched K32 source-gate outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTS = tuple(
    f"BioEmu_{value}"
    for value in (
        1, 32, 63, 94, 126, 157, 188, 221, 251, 281, 312, 344, 376,
        407, 438, 469, 501, 533, 565, 595, 626, 656, 687, 719, 751, 781,
        811, 843, 876, 906, 937, 968,
    )
)
NESTED = tuple(SUPPORTS[index] for index in (0, 4, 8, 12, 16, 20, 24, 28))
MODES = (
    "full_k32",
    "full_nested_k8",
    "no_coordinate",
    "uniform_q",
    "anchor_only",
)
PREDICTION_COLUMNS = (
    "fold", "held_half", "entity_uid", "target_id", "seq_id", "comp_id",
    "atom_id", "target_value", "split", "observer_fold",
    "observer_state_sha256", *MODES,
)
Q_COLUMNS = (
    "fold", "held_half", "entity_uid", "mode", "support_id", "q",
    "observer_state_sha256",
)
AUDIT_COLUMNS = (
    "entity_uid", "mode", "support_id", "maximum_displacement_angstrom",
    "minimum_distinct_atom_distance_angstrom", "coordinate_gradient_norm",
)
STATE_KEYS = {
    "entity_uids", "modes", "residue_offsets", "residue_ids", "residue_delta",
    "coordinate_gradient_norm",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_identity_sha256(frame: pd.DataFrame) -> str:
    rows = []
    for entity_uid, target_id, atom_id, target_value in frame[
        ["entity_uid", "target_id", "atom_id", "target_value"]
    ].itertuples(index=False, name=None):
        value = float(target_value)
        if not math.isfinite(value):
            raise ValueError("nonfinite source row identity")
        rows.append((str(entity_uid), str(target_id), str(atom_id).strip().upper(), value))
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3].hex()))
    if len({(row[0], row[1]) for row in rows}) != len(rows):
        raise ValueError("duplicate source row identity")
    digest = hashlib.sha256()
    for entity_uid, target_id, atom_id, target_value in rows:
        payload = json.dumps(
            [entity_uid, target_id, atom_id, target_value.hex()],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode() + b"\n")
    return digest.hexdigest()


def check_cell_output(path: Path) -> dict[str, Any]:
    receipt = json.loads((path / "receipt.json").read_text())
    required_receipt = {
        "contract", "coordinate_audit_rows", "files", "fold", "held_half",
        "observer_final_loss", "observer_initial_loss", "observer_state_sha256",
        "prediction_row_identity_sha256", "prediction_rows", "q_rows",
    }
    if set(receipt) != required_receipt or receipt["contract"] != "k32_nested_k8_source_gate_cell_output_v1":
        raise ValueError("source-cell receipt schema mismatch")
    expected_files = {
        "predictions.parquet", "q.parquet", "coordinate_audit.parquet",
        "coordinate_states.npz",
    }
    if set(receipt["files"]) != expected_files:
        raise ValueError("source-cell file inventory mismatch")
    for name, expected in receipt["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"source-cell file hash mismatch: {name}")
    prediction = pd.read_parquet(path / "predictions.parquet")
    q = pd.read_parquet(path / "q.parquet")
    audit = pd.read_parquet(path / "coordinate_audit.parquet")
    if tuple(prediction.columns) != PREDICTION_COLUMNS:
        raise ValueError("source-cell prediction schema mismatch")
    if tuple(q.columns) != Q_COLUMNS or tuple(audit.columns) != AUDIT_COLUMNS:
        raise ValueError("source-cell q or audit schema mismatch")
    fold, held_half = str(receipt["fold"]), int(receipt["held_half"])
    if (
        set(prediction["fold"].astype(str)) != {fold}
        or set(prediction["held_half"].astype(int)) != {held_half}
        or set(prediction["observer_fold"].astype(str)) != {fold}
        or set(prediction["split"].astype(str)) != {"train"}
        or len(prediction) != int(receipt["prediction_rows"])
        or row_identity_sha256(prediction) != receipt["prediction_row_identity_sha256"]
    ):
        raise ValueError("source-cell prediction identity mismatch")
    numeric = prediction[["target_value", *MODES]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("source-cell prediction is nonfinite")
    observer_hash = str(receipt["observer_state_sha256"])
    if (
        len(observer_hash) != 64
        or set(prediction["observer_state_sha256"].astype(str)) != {observer_hash}
        or set(q["observer_state_sha256"].astype(str)) != {observer_hash}
        or len(q) != int(receipt["q_rows"])
    ):
        raise ValueError("source-cell observer identity mismatch")
    entities = set(prediction["entity_uid"].astype(str))
    if set(q["entity_uid"].astype(str)) != entities or set(q["mode"].astype(str)) != set(MODES):
        raise ValueError("source-cell q identity mismatch")
    for (entity_uid, mode), rows in q.groupby(["entity_uid", "mode"], sort=True):
        expected = NESTED if mode == "full_nested_k8" else SUPPORTS
        if (
            tuple(rows["support_id"].astype(str)) != expected
            or not np.isfinite(rows["q"].to_numpy(float)).all()
            or np.any(rows["q"].to_numpy(float) < 0)
            or not math.isclose(float(rows["q"].sum()), 1.0, abs_tol=1.0e-6)
        ):
            raise ValueError(f"invalid source-cell q: {entity_uid}/{mode}")
        if mode in {"uniform_q", "anchor_only"} and not np.allclose(
            rows["q"].to_numpy(float), 1.0 / 32, atol=1.0e-7, rtol=0
        ):
            raise ValueError("uniform source-cell control has nonuniform q")
    with np.load(path / "coordinate_states.npz", allow_pickle=False) as values:
        if set(values.files) != STATE_KEYS:
            raise ValueError("source-cell coordinate-state schema mismatch")
        state_entities = values["entity_uids"].astype(str)
        state_modes = values["modes"].astype(str)
        offsets = np.array(values["residue_offsets"], copy=True)
        residue_ids = np.array(values["residue_ids"], copy=True)
        residue_delta = np.array(values["residue_delta"], copy=True)
        gradients = np.array(values["coordinate_gradient_norm"], copy=True)
    state_ids = list(zip(state_entities, state_modes, strict=True))
    expected_states = {(entity, mode) for entity in entities for mode in ("full_k32", "uniform_q")}
    if (
        set(state_ids) != expected_states
        or len(state_ids) != len(expected_states)
        or offsets.dtype != np.int64
        or offsets.tolist()[0] != 0
        or offsets.tolist()[-1] != len(residue_ids)
        or np.any(np.diff(offsets) <= 0)
        or residue_ids.dtype != np.int32
        or residue_delta.dtype != np.float32
        or residue_delta.shape != (len(residue_ids), 32, 4)
        or gradients.shape != (len(state_ids),)
        or not np.isfinite(residue_delta).all()
        or not np.isfinite(gradients).all()
        or np.any(gradients <= 0)
        or np.any(np.abs(residue_delta) > 0.05 + 1.0e-8)
    ):
        raise ValueError("source-cell coordinate-state invariant mismatch")
    expected_audit = {
        (entity, mode, support) for entity, mode in state_ids for support in SUPPORTS
    }
    observed_audit = set(
        audit[["entity_uid", "mode", "support_id"]].astype(str).itertuples(index=False, name=None)
    )
    if (
        observed_audit != expected_audit
        or len(audit) != int(receipt["coordinate_audit_rows"])
        or not np.isfinite(audit[list(AUDIT_COLUMNS[3:])].to_numpy(float)).all()
        or audit["maximum_displacement_angstrom"].max() > 1.0 + 1.0e-6
        or audit["maximum_displacement_angstrom"].max() <= 1.0e-8
        or audit["minimum_distinct_atom_distance_angstrom"].min() < 0.5 - 1.0e-6
    ):
        raise ValueError("source-cell coordinate audit mismatch")
    return {
        "fold": fold,
        "held_half": held_half,
        "prediction_rows": len(prediction),
        "q_rows": len(q),
        "coordinate_audit_rows": len(audit),
        "passed": True,
    }


def concordance(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (
        np.mean(target_centered**2) + np.mean(prediction_centered**2)
        + float(target.mean() - prediction.mean()) ** 2
    )
    if len(target) < 2 or denominator <= 1.0e-15:
        raise ValueError("undefined source CCC")
    return float(2 * np.mean(target_centered * prediction_centered) / denominator)


def aggregate_source_scores(
    frames: tuple[pd.DataFrame, ...], atom_inventory: tuple[str, ...]
) -> dict[str, Any]:
    frame = pd.concat(frames, ignore_index=True)
    if frame["target_id"].astype(str).duplicated().any():
        raise ValueError("source OOF predictions duplicate target identities")
    if set(frame["fold"].astype(str)) != {"A", "B"}:
        raise ValueError("source OOF predictions omit a fold")
    for fold in ("A", "B"):
        if set(frame.loc[frame["fold"].eq(fold), "held_half"].astype(int)) != {0, 1}:
            raise ValueError("source OOF predictions omit a held half")
    scores: dict[str, dict[str, float]] = {}
    for scope in ("A", "B", "OOF"):
        selected = frame if scope == "OOF" else frame.loc[frame["fold"].eq(scope)]
        scope_scores = {}
        for mode in MODES:
            per_atom = []
            for atom_id in atom_inventory:
                rows = selected["atom_id"].astype(str).eq(atom_id)
                if rows.sum() < 2:
                    raise ValueError(f"source scope lacks frozen Atom_ID: {scope}/{atom_id}")
                per_atom.append(concordance(
                    selected.loc[rows, "target_value"].to_numpy(float),
                    selected.loc[rows, mode].to_numpy(float),
                ))
            scope_scores[mode] = float(np.mean(per_atom))
        scores[scope] = scope_scores
    gains = {
        scope: {
            "over_nested_k8": values["full_k32"] - values["full_nested_k8"],
            "over_no_coordinate": values["full_k32"] - values["no_coordinate"],
        }
        for scope, values in scores.items()
    }
    passed = all(gain > 0.0005 for values in gains.values() for gain in values.values())
    return {"scores": scores, "gains": gains, "threshold": 0.0005, "passed": passed}
