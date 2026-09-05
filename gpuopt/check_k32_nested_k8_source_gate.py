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
SOURCE_COMMITMENT = ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
SOURCE_COMMITMENT_SHA256 = "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8"
SIDECHAIN_CHI_BONDS = {
    "ARG": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")),
    "ASN": (("CA", "CB"), ("CB", "CG")),
    "ASP": (("CA", "CB"), ("CB", "CG")),
    "CYS": (("CA", "CB"),),
    "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
    "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
    "HIS": (("CA", "CB"), ("CB", "CG")),
    "ILE": (("CA", "CB"), ("CB", "CG1")),
    "LEU": (("CA", "CB"), ("CB", "CG")),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD")),
    "PHE": (("CA", "CB"), ("CB", "CG")),
    "SER": (("CA", "CB"),),
    "THR": (("CA", "CB"),),
    "TRP": (("CA", "CB"), ("CB", "CG")),
    "TYR": (("CA", "CB"), ("CB", "CG")),
    "VAL": (("CA", "CB"),),
}
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


def rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    shifted = points - origin
    return (
        shifted * math.cos(angle)
        + np.cross(unit, shifted) * math.sin(angle)
        + np.outer(shifted @ unit, unit) * (1 - math.cos(angle))
        + origin
    )


def replay_pdb(
    path: Path, residue_ids: np.ndarray, residue_delta: np.ndarray
) -> tuple[float, float]:
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or next(
            (character.upper() for character in name if character.isalpha()), ""
        )
        records.append((
            name, element, int(line[22:26]), line[17:20].strip(), line[21:22],
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        ))
    base = np.asarray([record[5] for record in records], dtype=np.float64)
    if not records or not np.isfinite(base).all():
        raise ValueError("coordinate replay PDB is empty or nonfinite")
    moved = base.copy()
    for seq_id, angles in zip(residue_ids, residue_delta, strict=True):
        indices = [index for index, record in enumerate(records) if record[2] == seq_id]
        if not indices:
            continue
        chain = records[indices[0]][4]
        indices = [index for index in indices if records[index][4] == chain]
        adjacency = {index: set() for index in indices}
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                hydrogen = records[first][1] == "H" or records[second][1] == "H"
                cutoff = 1.25 if hydrogen else 1.95
                if np.linalg.norm(moved[first] - moved[second]) <= cutoff:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        bonds = SIDECHAIN_CHI_BONDS.get(records[indices[0]][3], ())
        for angle, (proximal_name, distal_name) in zip(angles, bonds, strict=False):
            if abs(float(angle)) <= 1.0e-12:
                continue
            proximal = next(
                (index for index in indices if records[index][0] == proximal_name), None
            )
            distal_axis = next(
                (index for index in indices if records[index][0] == distal_name), None
            )
            if proximal is None or distal_axis is None:
                continue
            distal, stack = {distal_axis}, [distal_axis]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {proximal, distal_axis}:
                        continue
                    if neighbor not in distal:
                        distal.add(neighbor)
                        stack.append(neighbor)
            if any(records[index][0] in {"N", "CA", "C", "O", "OXT"} for index in distal):
                continue
            selected = sorted(distal)
            moved[selected] = rotate(
                moved[selected], moved[proximal], moved[distal_axis] - moved[proximal],
                float(angle),
            )
    base = base.astype(np.float32)
    moved = moved.astype(np.float32)
    maximum = float(np.linalg.norm(moved - base, axis=1).max(initial=0))
    minimum = math.inf
    for row in range(len(moved) - 1):
        minimum = min(
            minimum,
            float(np.linalg.norm(moved[row + 1 :] - moved[row], axis=1).min(initial=math.inf)),
        )
    return maximum, minimum


def check_cell_output(path: Path, *, root: Path) -> dict[str, Any]:
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
    root = root.resolve()
    commitment_path = root / SOURCE_COMMITMENT
    if sha256(commitment_path) != SOURCE_COMMITMENT_SHA256:
        raise ValueError("coordinate source commitment hash mismatch")
    commitment = json.loads(commitment_path.read_text())
    pdb_rows = {
        (str(row["entity_uid"]), str(row["support_id"])): row
        for row in commitment["pdb_files"]
    }
    audit_rows = audit.set_index(["entity_uid", "mode", "support_id"])
    replay_rows = 0
    for state_number, (entity_uid, mode) in enumerate(state_ids):
        start, stop = int(offsets[state_number]), int(offsets[state_number + 1])
        for support_number, support_id in enumerate(SUPPORTS):
            pdb_row = pdb_rows.get((entity_uid, support_id))
            if pdb_row is None:
                raise ValueError("coordinate replay source is absent")
            pdb_path = (root / str(pdb_row["relative_path"])).resolve()
            if root not in pdb_path.parents or sha256(pdb_path) != pdb_row["sha256"]:
                raise ValueError("coordinate replay source binding mismatch")
            maximum, minimum = replay_pdb(
                pdb_path,
                residue_ids[start:stop],
                residue_delta[start:stop, support_number],
            )
            reported = audit_rows.loc[(entity_uid, mode, support_id)]
            if (
                not math.isclose(maximum, float(reported["maximum_displacement_angstrom"]), abs_tol=1.0e-6)
                or not math.isclose(minimum, float(reported["minimum_distinct_atom_distance_angstrom"]), abs_tol=1.0e-6)
                or not math.isclose(float(gradients[state_number]), float(reported["coordinate_gradient_norm"]), abs_tol=1.0e-12)
            ):
                raise ValueError(
                    "independent coordinate replay disagrees with audit: "
                    f"{entity_uid}/{mode}/{support_id} "
                    f"maximum={maximum}/{reported['maximum_displacement_angstrom']} "
                    f"minimum={minimum}/{reported['minimum_distinct_atom_distance_angstrom']}"
                )
            replay_rows += 1
    return {
        "fold": fold,
        "held_half": held_half,
        "prediction_rows": len(prediction),
        "q_rows": len(q),
        "coordinate_audit_rows": len(audit),
        "coordinate_replay_rows": replay_rows,
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
