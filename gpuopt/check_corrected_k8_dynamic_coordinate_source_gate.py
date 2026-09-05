#!/usr/bin/env python3
"""Independent arithmetic and loss-contract check for the dynamic source gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


MINIMUM_GAIN_EACH_FOLD = 0.0005
LOSS_FUNCTIONS = {"train_observer", "optimize_assimilation"}
ELIGIBILITY_GUARDS = {
    "crossfit_anchor_selection": 2,
    "normalization": 2,
    "calibrate_reference_bounds": 1,
    "train_observer": 1,
    "optimize_assimilation": 1,
    "predict_surface": 1,
    "coordinate_audit": 1,
}
COMMITMENT_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_commitment_v1"
CONSUMED_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_consumed_v1"
EXTERNAL_CLAIM_CONTRACT = (
    "corrected_k8_dynamic_coordinate_source_gate_external_claim_v1"
)
RESULT_CONTRACT = "corrected_k8_dynamic_coordinate_source_oof_gate_v1"
DECISION_CONTRACT = "corrected_k8_dynamic_coordinate_source_oof_check_v1"
ELIGIBILITY_CONTRACT = "corrected_k8_source_subset_eligibility_v1"
AUTHORIZATION_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_authorization_v1"
EXPECTED_EPOCHS = 1024
EXPECTED_STEPS = 100
CANONICAL_INVALIDATION_SHA256 = (
    "5a84b9ea624694976e3ff31548437fae8e0f8856d19698fd77b2997d3d96152d"
)
STATIC_COMMITTED_FILES = {
    ".auto/checks.sh",
    ".auto/measure_body.sh",
    "gpuopt/candidates/__init__.py",
    "gpuopt/candidates/corrected_k8_dynamic_coordinate.py",
    "gpuopt/source_gate_eligibility.py",
    "gpuopt/preflight_all_label_autoresearch.py",
    "gpuopt/freeze_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py",
    "gpuopt/slurm/run_corrected_k8_dynamic_coordinate_source_gate_l40s.sbatch",
    "gpuopt/tests/test_corrected_k8_dynamic_coordinate_source_gate.py",
    ".auto/preunblind/corrected_k8_dynamic_coordinate_plan.json",
    ".auto/frozen/all_label_inventory.json",
    "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/post_handoff_audit.json",
    "data/all_atom_observer_v1/commitment.json",
    "data/all_atom_observer_v1/feature_receipt.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/receipt.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/summary.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/summary.json",
}
EXPECTED_SUPPORT_IDS = tuple(
    f"BioEmu_{index}" for index in (1, 126, 251, 376, 501, 626, 751, 876)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_commitment(root: Path, path: Path) -> str:
    commitment = json.loads(path.read_text())
    if commitment.get("contract") != COMMITMENT_CONTRACT:
        raise ValueError("source commitment contract mismatch")
    expected_metadata = {
        "invalidated_parent_job": "134710",
        "invalidated_parent_disposition": "INVALIDATED",
        "canonical_invalidation_sha256": CANONICAL_INVALIDATION_SHA256,
        "formal_evaluation_authorized": False,
        "target_values_opened": False,
        "support_ids": list(EXPECTED_SUPPORT_IDS),
    }
    for field, expected in expected_metadata.items():
        if commitment.get(field) != expected:
            raise ValueError(f"source commitment metadata mismatch: {field}")
    for relative, expected in commitment["files"].items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents or sha256_file(candidate) != expected:
            raise ValueError(f"source commitment mismatch: {relative}")
    return sha256_file(path)


def independent_expected_source_manifest(
    root: Path, data_commitment: dict
) -> set[str]:
    paths = {root / relative for relative in STATIC_COMMITTED_FILES}
    data_root = root / "data/all_atom_observer_v1"
    shared_root = root / "data/all_atom_shared_q_e2e_v0"
    ucb_root = shared_root / "ucbshift_x_anchor_v0"
    embedding_cache = shared_root / "esm2_cache"
    paths.update(ucb_root.glob("*.npz"))
    for entity in data_commitment["entities"]:
        if entity.get("split") != "train" or entity.get("observer_fold") not in {
            "A",
            "B",
        }:
            continue
        bmrb_id = str(entity["bmrb_id"])
        paths.add(data_root / "features" / f"{bmrb_id}.parquet")
        paths.add(data_root / "targets" / f"{bmrb_id}.parquet")
        sequence_digest = hashlib.sha1(  # noqa: S324 - frozen cache identity
            f"esm2_t30_150M_UR50D|640|{entity['sequence']}".encode()
        ).hexdigest()[:16]
        paths.add(embedding_cache / f"esm2_{sequence_digest}.npz")
        support_ids = tuple(str(value) for value in entity["support_ids"])
        if support_ids != EXPECTED_SUPPORT_IDS:
            raise ValueError("checker source support roster is not frozen K=8")
        for support_id in support_ids:
            paths.add(
                root / "data/BioEmu" / bmrb_id / f"{bmrb_id}_{support_id}.pdb"
            )
    return {str(path.relative_to(root)) for path in paths}


def verify_source_manifest_complete(
    root: Path, commitment_path: Path, data_commitment: dict
) -> None:
    commitment = json.loads(commitment_path.read_text())
    declared = set(commitment["files"])
    expected = independent_expected_source_manifest(root, data_commitment)
    source_entity_count = sum(
        entity.get("split") == "train"
        and entity.get("observer_fold") in {"A", "B"}
        for entity in data_commitment["entities"]
    )
    if commitment.get("source_entity_count") != source_entity_count:
        raise ValueError("checker source commitment entity count mismatch")
    if declared != expected:
        raise ValueError(
            "checker source commitment manifest mismatch: "
            f"missing={sorted(expected - declared)[:3]} "
            f"extra={sorted(declared - expected)[:3]}"
        )


def concordance(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(target) < 2:
        return math.nan
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (
        np.mean(target_centered**2)
        + np.mean(prediction_centered**2)
        + float(target.mean() - prediction.mean()) ** 2
    )
    if denominator <= 1.0e-15:
        return math.nan
    return float(2.0 * np.mean(target_centered * prediction_centered) / denominator)


def macro_score(
    frame: pd.DataFrame, prediction_column: str, eligible_atom_ids: list[str]
) -> tuple[float, int]:
    values = []
    for atom_id in eligible_atom_ids:
        rows = frame["atom_id"].astype(str).eq(atom_id).to_numpy()
        target = frame.loc[rows, "target_value"].to_numpy(float)
        if len(target) < 2 or float(np.var(target)) <= 1.0e-15:
            continue
        value = concordance(
            target,
            frame.loc[rows, prediction_column].to_numpy(float),
        )
        if not math.isfinite(value):
            raise ValueError(f"nonfinite prediction-defined source CCC: {atom_id}")
        values.append(value)
    if not values:
        raise ValueError("no defined Atom_ID source CCC")
    return float(np.mean(values)), len(values)


def atom_family_macro_score(
    frame: pd.DataFrame, prediction_column: str, eligible_atom_ids: list[str]
) -> dict[str, float]:
    diagnostics = {}
    for family in ("C", "H", "N"):
        values = []
        for atom_id in eligible_atom_ids:
            if not atom_id.startswith(family):
                continue
            rows = frame["atom_id"].astype(str).eq(atom_id).to_numpy()
            target = frame.loc[rows, "target_value"].to_numpy(float)
            if len(target) < 2 or float(np.var(target)) <= 1.0e-15:
                continue
            value = concordance(
                target, frame.loc[rows, prediction_column].to_numpy(float)
            )
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {family}-family source CCC: {atom_id}")
            values.append(value)
        if not values:
            raise ValueError(f"source OOF has no defined {family} Atom_ID labels")
        diagnostics[family] = float(np.mean(values))
    return diagnostics


def verify_family_diagnostics(
    reported: object, computed: dict[str, dict[str, float]], scope: str
) -> None:
    if not isinstance(reported, dict) or set(reported) != set(computed):
        raise ValueError(f"atom-family diagnostic control inventory mismatch: {scope}")
    for control, families in computed.items():
        values = reported.get(control)
        if not isinstance(values, dict) or set(values) != {"C", "H", "N"}:
            raise ValueError(f"atom-family diagnostic inventory mismatch: {scope}")
        for family, expected in families.items():
            if abs(float(values[family]) - expected) > 1.0e-12:
                raise ValueError(
                    f"atom-family diagnostic arithmetic mismatch: {scope}/{control}/{family}"
                )


def verify_ccc_absent_from_loss(candidate_path: Path) -> None:
    tree = ast.parse(candidate_path.read_text())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in LOSS_FUNCTIONS:
            continue
        found.add(node.name)
        text = ast.unparse(node).lower()
        if "ccc" in text or "concordance" in text:
            raise ValueError(f"CCC entered loss function {node.name}")
    if found != LOSS_FUNCTIONS:
        raise ValueError(f"missing audited loss functions: {LOSS_FUNCTIONS - found}")


def verify_model_phase_eligibility_guards(candidate_path: Path) -> None:
    """Inspect bound candidate bytes without importing runner/helper functions."""

    tree = ast.parse(candidate_path.read_text())
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in ELIGIBILITY_GUARDS:
            continue
        statements = list(node.body)
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements = statements[1:]
        required = ELIGIBILITY_GUARDS[node.name]
        prefix = statements[:required]
        calls = sum(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "require_source_eligibility"
            for statement in prefix
        )
        found[node.name] = calls
    if set(found) != set(ELIGIBILITY_GUARDS):
        raise ValueError(
            "missing guarded model phases: "
            f"{sorted(set(ELIGIBILITY_GUARDS) - set(found))}"
        )
    for name, required in ELIGIBILITY_GUARDS.items():
        if found[name] != required:
            raise ValueError(f"model phase does not begin with eligibility guard: {name}")


def verify_control_factorial(runner_path: Path) -> None:
    """Verify three separately initialized assimilation calls and fixed controls."""

    tree = ast.parse(runner_path.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "optimize_assimilation"
    ]
    if len(calls) != 3:
        raise ValueError("source factorial must contain three assimilation calls")

    def keyword_literal(call: ast.Call, name: str) -> object:
        for keyword in call.keywords:
            if keyword.arg == name and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        return None

    zero_jacobian = [
        call
        for call in calls
        if keyword_literal(call, "geometry_jacobian_scale") == 0.0
    ]
    uniform_q = [
        call for call in calls if keyword_literal(call, "fixed_uniform_q") is True
    ]
    full = [
        call
        for call in calls
        if keyword_literal(call, "geometry_jacobian_scale") is None
        and keyword_literal(call, "fixed_uniform_q") is None
    ]
    if len(full) != 1 or len(zero_jacobian) != 1 or len(uniform_q) != 1:
        raise ValueError("source factorial control wiring mismatch")


def verify_project_import_guard(runner_path: Path) -> None:
    """Reject project-local science imports during runner module initialization."""

    tree = ast.parse(runner_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gpuopt"):
            raise ValueError("runner imports project science before authorization")
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("gpuopt") for alias in node.names
        ):
            raise ValueError("runner imports project science before authorization")
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    call_lines: dict[str, list[int]] = {}
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {"consume_authorization", "verify_source_commitment", "load_science_api"}
        ):
            call_lines.setdefault(node.func.id, []).append(node.lineno)
    required = ("consume_authorization", "verify_source_commitment", "load_science_api")
    if not all(name in call_lines for name in required):
        raise ValueError("runner lacks project-science guard phase")
    if (
        len(call_lines["consume_authorization"]) != 1
        or len(call_lines["load_science_api"]) != 1
    ):
        raise ValueError("runner project-science guard call multiplicity mismatch")
    if not (
        call_lines["consume_authorization"][0]
        < min(call_lines["verify_source_commitment"])
        < call_lines["load_science_api"][0]
    ):
        raise ValueError("runner imports project science before authorization verification")


def canonical_atom_id(value: object) -> str:
    return str(value).strip().upper()


def independent_row_identity_sha256(frame: pd.DataFrame) -> str:
    columns = ["entity_uid", "target_id", "atom_id", "target_value"]
    if not set(columns).issubset(frame.columns):
        raise ValueError("checker source identity columns are incomplete")
    rows = []
    for entity_uid, target_id, atom_id, target_value in frame[columns].itertuples(
        index=False, name=None
    ):
        value = float(target_value)
        if not math.isfinite(value):
            raise ValueError("checker source identity contains nonfinite target")
        rows.append(
            (str(entity_uid), str(target_id), canonical_atom_id(atom_id), value)
        )
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3].hex()))
    identities = [(row[0], row[1]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("checker source identity is duplicated")
    digest = hashlib.sha256()
    for entity_uid, target_id, atom_id, target_value in rows:
        digest.update(
            json.dumps(
                [entity_uid, target_id, atom_id, target_value.hex()],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def independent_eligible_atom_ids(
    frame: pd.DataFrame, frozen_atom_ids: list[str]
) -> tuple[str, ...]:
    atom = frame["atom_id"].map(canonical_atom_id)
    target = pd.to_numeric(frame["target_value"], errors="coerce")
    eligible = []
    for atom_id in (canonical_atom_id(value) for value in frozen_atom_ids):
        values = target[atom.eq(atom_id) & target.map(math.isfinite)].to_numpy(float)
        if len(values) >= 2 and float(np.var(values)) > 1.0e-15:
            eligible.append(atom_id)
    return tuple(eligible)


def independent_eligibility_receipt(
    frame: pd.DataFrame,
    *,
    frozen_atom_ids: list[str],
    fold: str,
    held_half: int,
    role: str,
) -> tuple[pd.DataFrame, dict]:
    frozen = tuple(canonical_atom_id(value) for value in frozen_atom_ids)
    eligible = independent_eligible_atom_ids(frame, list(frozen))
    atom = frame["atom_id"].map(canonical_atom_id)
    target = pd.to_numeric(frame["target_value"], errors="coerce")
    selected_mask = atom.isin(set(eligible)) & target.map(math.isfinite)
    selected = frame.loc[selected_mask].reset_index(drop=True)
    excluded = frame.loc[~selected_mask].reset_index(drop=True)
    receipt = {
        "contract": ELIGIBILITY_CONTRACT,
        "fold": fold,
        "held_half": held_half,
        "role": role,
        "rule": "frozen_inventory_and_finite_count_ge_2_and_population_variance_gt_1e-15",
        "frozen_atom_ids": list(frozen),
        "eligible_atom_ids": list(eligible),
        "input_row_count": len(frame),
        "input_row_identity_sha256": independent_row_identity_sha256(frame),
        "selected_row_count": len(selected),
        "selected_row_identity_sha256": independent_row_identity_sha256(selected),
        "excluded_row_count": len(excluded),
        "excluded_row_identity_sha256": independent_row_identity_sha256(excluded),
    }
    return selected, receipt


def independent_sequence_cluster_halves(entities: list[dict]) -> dict[str, int]:
    clusters: dict[str, list[dict]] = {}
    for entity in sorted(entities, key=lambda item: str(item["entity_uid"])):
        clusters.setdefault(str(entity["sequence_cluster_id"]), []).append(entity)
    loads = [0, 0]
    assignment: dict[str, int] = {}
    for _cluster_id, members in sorted(
        clusters.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        half = min(range(2), key=lambda index: (loads[index], index))
        for entity in members:
            assignment[str(entity["entity_uid"])] = half
        loads[half] += len(members)
    if not all(loads):
        raise ValueError("checker source fold cannot form two halves")
    return assignment


def committed_source_targets(
    root: Path, commitment: dict, fold: str
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    parts = []
    support_ids: dict[str, list[str]] = {}
    for entity in commitment["entities"]:
        if entity.get("split") != "train" or entity.get("observer_fold") != fold:
            continue
        entity_uid = str(entity["entity_uid"])
        targets = pd.read_parquet(
            root / "data/all_atom_observer_v1/targets" / f"{entity['bmrb_id']}.parquet",
            columns=("entity_uid", "target_id", "target_value"),
        )
        targets = targets[targets["entity_uid"].astype(str).eq(entity_uid)].copy()
        features = pd.read_parquet(
            root / "data/all_atom_observer_v1/features" / f"{entity['bmrb_id']}.parquet",
            columns=("entity_uid", "target_id", "atom_id"),
        )
        features = features[features["entity_uid"].astype(str).eq(entity_uid)].copy()
        identity = features[["entity_uid", "target_id", "atom_id"]].drop_duplicates()
        if identity.duplicated(["entity_uid", "target_id"]).any():
            raise ValueError(f"ambiguous committed Atom_ID identity: {entity_uid}")
        frame = targets.merge(
            identity,
            on=["entity_uid", "target_id"],
            how="left",
            validate="one_to_one",
        )
        if frame["atom_id"].isna().any():
            raise ValueError(f"missing committed Atom_ID identity: {entity_uid}")
        frame["atom_id"] = frame["atom_id"].astype(str).str.strip().str.upper()
        frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
        frame = frame[frame["target_value"].map(math.isfinite)]
        frame["sequence_cluster_id"] = str(entity["sequence_cluster_id"])
        parts.append(frame)
        support_ids[entity_uid] = [str(value) for value in entity["support_ids"]]
        if tuple(support_ids[entity_uid]) != EXPECTED_SUPPORT_IDS:
            raise ValueError(f"committed source support roster is not frozen K=8: {fold}")
    expected = pd.concat(parts, ignore_index=True)
    if expected.duplicated(["entity_uid", "target_id"]).any():
        raise ValueError(f"duplicate committed source identity: {fold}")
    return expected, support_ids


def verify_runner_targets(actual: pd.DataFrame, expected: pd.DataFrame, fold: str) -> None:
    identity = ["entity_uid", "target_id"]
    for column in identity:
        actual[column] = actual[column].astype(str)
        expected[column] = expected[column].astype(str)
    actual = actual.sort_values(identity, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(identity, kind="stable").reset_index(drop=True)
    if len(actual) != len(expected) or not actual[identity].equals(expected[identity]):
        raise ValueError(f"runner OOF identity differs from committed targets: {fold}")
    if not actual["atom_id"].astype(str).equals(expected["atom_id"].astype(str)):
        raise ValueError(f"runner OOF Atom_ID differs from committed targets: {fold}")
    if not actual["sequence_cluster_id"].astype(str).equals(
        expected["sequence_cluster_id"].astype(str)
    ):
        raise ValueError(f"runner OOF cluster differs from commitment: {fold}")
    if not np.array_equal(
        actual["target_value"].to_numpy(float),
        expected["target_value"].to_numpy(float),
    ):
        raise ValueError(f"runner OOF target values differ from commitment: {fold}")


def audit_coordinate_output(
    path: Path,
    *,
    root: Path,
    expected_entities: set[str],
    support_ids: dict[str, list[str]],
    bmrb_ids: dict[str, str],
) -> dict[str, float | int]:
    with np.load(path) as arrays:
        conditioned = np.asarray(arrays["conditioned_coordinates"], dtype=np.float64)
        base = np.asarray(arrays["no_evidence_coordinates"], dtype=np.float64)
        atom_mask = np.asarray(arrays["atom_mask"], dtype=bool)
        atom_state = np.asarray(arrays["atom_state_index"], dtype=np.int64)
        atom_name = np.asarray(arrays["atom_name"], dtype=str)
        atom_element = np.asarray(arrays["atom_element"], dtype=str)
        atom_seq_id = np.asarray(arrays["atom_seq_id"], dtype=np.int64)
        state_entity = np.asarray(arrays["state_entity_uid"], dtype=str)
        state_support = np.asarray(arrays["state_support_id"], dtype=str)
    expected_states = {
        (entity_uid, support_id)
        for entity_uid in expected_entities
        for support_id in support_ids[entity_uid]
    }
    states = list(zip(state_entity, state_support, strict=True))
    if len(states) != len(expected_states) or set(states) != expected_states:
        raise ValueError(f"coordinate state inventory mismatch: {path.name}")
    if conditioned.shape != base.shape or conditioned.ndim != 2 or conditioned.shape[1] != 3:
        raise ValueError(f"coordinate shape mismatch: {path.name}")
    if atom_mask.shape != conditioned.shape[:-1] or atom_state.shape != atom_mask.shape:
        raise ValueError(f"coordinate atom identity mismatch: {path.name}")
    if (
        atom_name.shape != atom_mask.shape
        or atom_element.shape != atom_mask.shape
        or atom_seq_id.shape != atom_mask.shape
    ):
        raise ValueError(f"coordinate atom metadata mismatch: {path.name}")
    if not atom_mask.all() or not np.array_equal(
        np.unique(atom_state), np.arange(len(states), dtype=np.int64)
    ):
        raise ValueError(f"coordinate atom-state coverage mismatch: {path.name}")
    if not np.isfinite(conditioned[atom_mask]).all() or not np.isfinite(base[atom_mask]).all():
        raise ValueError(f"nonfinite coordinates: {path.name}")
    displacement = np.linalg.norm(conditioned[atom_mask] - base[atom_mask], axis=1)
    rmsd = float(np.sqrt(np.mean(np.square(displacement))))
    maximum_displacement = float(displacement.max())
    if not math.isfinite(rmsd) or rmsd <= 1.0e-6 or maximum_displacement > 1.0:
        raise ValueError(f"invalid evidence-conditioned coordinate movement: {path.name}")
    minimum_distance = math.inf
    for state_number in range(len(states)):
        rows = atom_mask & (atom_state == state_number)
        if not rows.any():
            raise ValueError(f"empty coordinate state: {path.name}:{state_number}")
        entity_uid, support_id = states[state_number]
        pdb_path = (
            root
            / "data/BioEmu"
            / bmrb_ids[entity_uid]
            / f"{bmrb_ids[entity_uid]}_{support_id}.pdb"
        )
        records = [
            line
            for line in pdb_path.read_text().splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        ]
        pdb_coordinates = np.asarray(
            [
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in records
            ],
            dtype=np.float32,
        ).astype(np.float64)
        if not np.array_equal(base[rows], pdb_coordinates):
            raise ValueError(f"emitted base differs from committed PDB: {path.name}")
        if not np.array_equal(
            atom_name[rows], np.asarray([line[12:16].strip() for line in records])
        ) or not np.array_equal(
            atom_seq_id[rows], np.asarray([int(line[22:26]) for line in records])
        ):
            raise ValueError(f"emitted atom identity differs from committed PDB: {path.name}")
        expected_elements = np.asarray(
            [
                next(
                    (
                        character
                        for character in line[12:16].strip()
                        if character.isalpha()
                    ),
                    "",
                ).upper()
                for line in records
            ]
        )
        if not np.array_equal(atom_element[rows], expected_elements):
            raise ValueError(
                f"emitted atom element differs from committed PDB: {path.name}"
            )
        for coordinate in (pdb_coordinates, conditioned[rows]):
            distances, _ = cKDTree(coordinate).query(coordinate, k=2)
            state_minimum = float(np.min(distances[:, 1]))
            if not math.isfinite(state_minimum) or state_minimum < 0.5:
                raise ValueError(
                    f"severe distinct-atom clash: {path.name}:{state_number}"
                )
            minimum_distance = min(minimum_distance, state_minimum)
    return {
        "state_count": len(states),
        "atom_count": int(atom_mask.sum()),
        "coordinate_rmsd_angstrom": rmsd,
        "maximum_displacement_angstrom": maximum_displacement,
        "minimum_distinct_atom_distance_angstrom": minimum_distance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--source-commitment", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--consumed-authorization", type=Path, required=True)
    parser.add_argument("--external-authorization-claim", type=Path, required=True)
    parser.add_argument("--authorization-git-dir", type=Path, required=True)
    parser.add_argument("--authorization-git-blob", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--container-image-sha256", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads(args.result.read_text())
    if result.get("contract") != RESULT_CONTRACT:
        raise ValueError("corrected K8 result contract mismatch")
    if result.get("epochs") != EXPECTED_EPOCHS or result.get("steps") != EXPECTED_STEPS:
        raise ValueError("corrected K8 frozen training horizon mismatch")
    source_commitment_sha256 = verify_source_commitment(root, args.source_commitment)
    invalidation_path = root / (
        "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/"
        "post_handoff_audit.json"
    )
    if sha256_file(invalidation_path) != CANONICAL_INVALIDATION_SHA256:
        raise ValueError("canonical Job 134710 invalidation audit mismatch")
    if result.get("source_commitment_sha256") != source_commitment_sha256:
        raise ValueError("runner/checker source commitment mismatch")
    consumed = json.loads(args.consumed_authorization.read_text())
    if consumed.get("contract") != CONSUMED_CONTRACT:
        raise ValueError("consumed authorization contract mismatch")
    authorization_bytes = args.authorization.read_bytes()
    expected_authorization_ref = (
        f"refs/atypemu-authorizations/corrected-k8/job-{args.slurm_job_id}"
    )
    if args.authorization_ref != expected_authorization_ref:
        raise ValueError("authorization ref/Slurm-job mismatch")
    computed_git_blob = hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(authorization_bytes)}\0".encode() + authorization_bytes
    ).hexdigest()
    if computed_git_blob != args.authorization_git_blob:
        raise ValueError("authorization Git-blob identity mismatch")
    external_claim_bytes = args.external_authorization_claim.read_bytes()
    external_claim_git_blob = hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(external_claim_bytes)}\0".encode() + external_claim_bytes
    ).hexdigest()
    resolved_claim_blob = subprocess.run(
        [
            "git",
            f"--git-dir={args.authorization_git_dir}",
            "rev-parse",
            f"{args.authorization_ref}^{{blob}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_claim_blob != external_claim_git_blob:
        raise ValueError("external authorization claim ref/blob mismatch")
    external_authorization_bytes = subprocess.run(
        [
            "git",
            f"--git-dir={args.authorization_git_dir}",
            "cat-file",
            "blob",
            args.authorization_git_blob,
        ],
        check=True,
        capture_output=True,
    ).stdout
    if external_authorization_bytes != authorization_bytes:
        raise ValueError("external authorization blob bytes mismatch")
    external_claim = json.loads(external_claim_bytes)
    expected_external_claim = {
        "contract": EXTERNAL_CLAIM_CONTRACT,
        "authorization_git_blob": args.authorization_git_blob,
        "authorization_ref": args.authorization_ref,
        "authorization_sha256": sha256_file(args.authorization),
        "container_image_sha256": args.container_image_sha256,
        "slurm_job_id": args.slurm_job_id,
        "source_commitment_sha256": source_commitment_sha256,
    }
    if external_claim != expected_external_claim:
        raise ValueError("external authorization claim content mismatch")
    if consumed.get("authorization_sha256") != sha256_file(args.authorization):
        raise ValueError("consumed authorization hash mismatch")
    if consumed.get("source_commitment_sha256") != source_commitment_sha256:
        raise ValueError("consumed source commitment mismatch")
    if consumed.get("authorization_git_blob") != args.authorization_git_blob:
        raise ValueError("consumed authorization Git-blob mismatch")
    if consumed.get("authorization_ref") != args.authorization_ref:
        raise ValueError("consumed authorization ref mismatch")
    if consumed.get("external_claim_git_blob") != external_claim_git_blob:
        raise ValueError("consumed external-claim Git-blob mismatch")
    if consumed.get("external_claim_sha256") != sha256_file(
        args.external_authorization_claim
    ):
        raise ValueError("consumed external-claim hash mismatch")
    if consumed.get("container_image_sha256") != args.container_image_sha256:
        raise ValueError("consumed container-image mismatch")
    if consumed.get("slurm_job_id") != args.slurm_job_id:
        raise ValueError("consumed Slurm-job mismatch")
    if result.get("slurm_job_id") != args.slurm_job_id:
        raise ValueError("runner/checker Slurm-job mismatch")
    if result.get("authorization_git_blob") != args.authorization_git_blob:
        raise ValueError("runner/checker authorization Git-blob mismatch")
    if result.get("authorization_ref") != args.authorization_ref:
        raise ValueError("runner/checker authorization ref mismatch")
    if result.get("external_claim_git_blob") != external_claim_git_blob:
        raise ValueError("runner/checker external-claim Git-blob mismatch")
    if result.get("external_claim_sha256") != sha256_file(
        args.external_authorization_claim
    ):
        raise ValueError("runner/checker external-claim hash mismatch")
    if result.get("container_image_sha256") != args.container_image_sha256:
        raise ValueError("runner/checker container-image mismatch")
    authorization = json.loads(args.authorization.read_text())
    expected_authorization = {
        "contract": AUTHORIZATION_CONTRACT,
        "source_commitment_sha256": source_commitment_sha256,
        "epochs": EXPECTED_EPOCHS,
        "steps": EXPECTED_STEPS,
        "authorized": True,
        "container_image_sha256": args.container_image_sha256,
        "slurm_job_id": args.slurm_job_id,
        "authorization_ref": args.authorization_ref,
    }
    if set(authorization) != set(expected_authorization):
        raise ValueError("authorization schema mismatch")
    for field, expected_value in expected_authorization.items():
        if authorization.get(field) != expected_value:
            raise ValueError(f"authorization field mismatch: {field}")
    if result.get("authorization_sha256") != consumed["authorization_sha256"]:
        raise ValueError("runner/checker authorization mismatch")
    if result.get("consumed_authorization_sha256") != sha256_file(
        args.consumed_authorization
    ):
        raise ValueError("runner/checker consumed-marker mismatch")
    inventory = json.loads((root / ".auto/frozen/all_label_inventory.json").read_text())
    data_commitment = json.loads(
        (root / "data/all_atom_observer_v1/commitment.json").read_text()
    )
    verify_source_manifest_complete(root, args.source_commitment, data_commitment)
    bmrb_ids = {
        str(entity["entity_uid"]): str(entity["bmrb_id"])
        for entity in data_commitment["entities"]
        if entity.get("split") == "train"
        and entity.get("observer_fold") in {"A", "B"}
    }
    verify_ccc_absent_from_loss(
        root / "gpuopt/candidates/corrected_k8_dynamic_coordinate.py"
    )
    verify_model_phase_eligibility_guards(
        root / "gpuopt/candidates/corrected_k8_dynamic_coordinate.py"
    )
    verify_control_factorial(
        root / "gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py"
    )
    verify_project_import_guard(
        root / "gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py"
    )
    if result.get("ccc_used_in_loss") is not False:
        raise ValueError("source runner does not attest CCC-free loss")
    if result.get("ccc_used_only_for_source_oof_scoring") is not True:
        raise ValueError("CCC role mismatch")
    if result.get("no_coordinate_control") != (
        "independently_optimized_q_reference_with_geometry_jacobian_zeroed"
    ):
        raise ValueError("source no-coordinate control mismatch")
    if result.get("uniform_q_control") != (
        "independently_optimized_coordinate_reference_with_q_fixed_uniform"
    ):
        raise ValueError("source uniform-q control mismatch")
    if result.get("correction") != (
        "source_subset_eligibility_precedes_anchor_selection_normalization_"
        "observer_fitting_reference_calibration_and_assimilation"
    ):
        raise ValueError("corrected K8 eligibility-order declaration mismatch")
    folds: dict[str, dict[str, float | int | list[float]]] = {}
    frames = []
    for fold in ("A", "B"):
        path = args.result.parent / f"source_oof_{fold}.parquet"
        if result["output_sha256"].get(path.name) != sha256_file(path):
            raise ValueError(f"source OOF hash mismatch: {fold}")
        frame = pd.read_parquet(path)
        frames.append(frame)
        expected_targets, support_ids = committed_source_targets(
            root, data_commitment, fold
        )
        if frame.duplicated(["entity_uid", "target_id"]).any():
            raise ValueError(f"source OOF identity overlap: {fold}")
        if set(frame["held_half"].astype(int)) != {0, 1}:
            raise ValueError(f"source OOF half coverage mismatch: {fold}")
        if int(frame.groupby("sequence_cluster_id")["held_half"].nunique().max()) != 1:
            raise ValueError(f"source OOF sequence-cluster overlap: {fold}")
        fold_entities = sorted(
            [
                entity
                for entity in data_commitment["entities"]
                if entity.get("split") == "train"
                and entity.get("observer_fold") == fold
            ],
            key=lambda entity: str(entity["entity_uid"]),
        )
        half_by_entity = independent_sequence_cluster_halves(fold_entities)
        actual_half_by_entity = {
            str(entity_uid): int(halves.iloc[0])
            for entity_uid, halves in frame.groupby("entity_uid")["held_half"]
            if len(set(halves.astype(int))) == 1
        }
        if actual_half_by_entity != half_by_entity:
            raise ValueError(f"source OOF entity-half assignment mismatch: {fold}")
        expected_eligible_parts = []
        reported_eligibility = result["folds"][fold].get("eligibility", {})
        for held_half in (0, 1):
            held_uids = {
                entity_uid
                for entity_uid, half in half_by_entity.items()
                if half == held_half
            }
            train_uids = set(half_by_entity) - held_uids
            train_input = expected_targets[
                expected_targets["entity_uid"].astype(str).isin(train_uids)
            ].reset_index(drop=True)
            held_input = expected_targets[
                expected_targets["entity_uid"].astype(str).isin(held_uids)
            ].reset_index(drop=True)
            _train_selected, expected_train_receipt = independent_eligibility_receipt(
                train_input,
                frozen_atom_ids=inventory["eligible_atom_ids"],
                fold=fold,
                held_half=held_half,
                role="observer_training",
            )
            held_selected, expected_held_receipt = independent_eligibility_receipt(
                held_input,
                frozen_atom_ids=inventory["eligible_atom_ids"],
                fold=fold,
                held_half=held_half,
                role="evidence_assimilation",
            )
            expected_receipts = {
                "observer_training": expected_train_receipt,
                "evidence_assimilation": expected_held_receipt,
            }
            if reported_eligibility.get(str(held_half)) != expected_receipts:
                raise ValueError(
                    f"source eligibility receipt mismatch: {fold}/{held_half}"
                )
            actual_half = frame[frame["held_half"].astype(int).eq(held_half)].copy()
            verify_runner_targets(actual_half, held_selected, f"{fold}/{held_half}")
            expected_eligible_parts.append(held_selected)
        verify_runner_targets(
            frame,
            pd.concat(expected_eligible_parts, ignore_index=True),
            fold,
        )
        candidate, candidate_labels = macro_score(
            frame, "candidate", inventory["eligible_atom_ids"]
        )
        control, control_labels = macro_score(
            frame, "control", inventory["eligible_atom_ids"]
        )
        uniform_q, uniform_q_labels = macro_score(
            frame, "uniform_q", inventory["eligible_atom_ids"]
        )
        expected = result["folds"][fold]
        if not (
            candidate_labels
            == control_labels
            == uniform_q_labels
            == int(expected["defined_label_count"])
        ):
            raise ValueError(f"source OOF label inventory mismatch: {fold}")
        if abs(candidate - float(expected["source_oof_candidate_macro_atom_id_ccc"])) > 1e-12:
            raise ValueError(f"candidate arithmetic mismatch: {fold}")
        if abs(control - float(expected["source_oof_jacobian_zero_control_macro_atom_id_ccc"])) > 1e-12:
            raise ValueError(f"control arithmetic mismatch: {fold}")
        if abs(
            uniform_q - float(expected["source_oof_uniform_q_macro_atom_id_ccc"])
        ) > 1e-12:
            raise ValueError(f"uniform-q arithmetic mismatch: {fold}")
        family_diagnostics = {
            "candidate": atom_family_macro_score(
                frame, "candidate", inventory["eligible_atom_ids"]
            ),
            "no_coordinate": atom_family_macro_score(
                frame, "control", inventory["eligible_atom_ids"]
            ),
            "uniform_q": atom_family_macro_score(
                frame, "uniform_q", inventory["eligible_atom_ids"]
            ),
        }
        verify_family_diagnostics(
            expected.get("atom_family_macro_atom_id_ccc"),
            family_diagnostics,
            fold,
        )
        gradients = [float(value) for value in expected["data_loss_gradient_norms"]]
        if not gradients or not all(math.isfinite(value) and value > 0 for value in gradients):
            raise ValueError(f"invalid data-loss gradient receipt: {fold}")
        coordinate_audits = []
        for held_half in (0, 1):
            coordinate_path = args.result.parent / (
                f"source_coordinate_{fold}_half{held_half}.npz"
            )
            if result["output_sha256"].get(coordinate_path.name) != sha256_file(
                coordinate_path
            ):
                raise ValueError(f"source coordinate hash mismatch: {fold}/{held_half}")
            coordinate_audits.append(
                audit_coordinate_output(
                    coordinate_path,
                    root=root,
                    expected_entities=set(
                        frame.loc[
                            frame["held_half"].astype(int).eq(held_half), "entity_uid"
                        ].astype(str)
                    ),
                    support_ids=support_ids,
                    bmrb_ids=bmrb_ids,
                )
            )
        folds[fold] = {
            "candidate": candidate,
            "jacobian_zero_control": control,
            "uniform_q": uniform_q,
            "gain": candidate - control,
            "defined_label_count": candidate_labels,
            "row_count": len(frame),
            "coordinate_audits": coordinate_audits,
            "atom_family_macro_atom_id_ccc": family_diagnostics,
        }
    combined_frame = pd.concat(frames, ignore_index=True)
    combined_candidate, combined_candidate_labels = macro_score(
        combined_frame, "candidate", inventory["eligible_atom_ids"]
    )
    combined_control, combined_control_labels = macro_score(
        combined_frame, "control", inventory["eligible_atom_ids"]
    )
    combined_uniform_q, combined_uniform_q_labels = macro_score(
        combined_frame, "uniform_q", inventory["eligible_atom_ids"]
    )
    if combined_candidate_labels != len(inventory["eligible_atom_ids"]):
        raise ValueError("combined candidate does not define frozen label inventory")
    if combined_control_labels != len(inventory["eligible_atom_ids"]):
        raise ValueError("combined control does not define frozen label inventory")
    if combined_uniform_q_labels != len(inventory["eligible_atom_ids"]):
        raise ValueError("combined uniform-q does not define frozen label inventory")
    expected_combined = result["combined"]
    if abs(
        combined_candidate
        - float(expected_combined["source_oof_candidate_macro_atom_id_ccc"])
    ) > 1e-12:
        raise ValueError("combined candidate arithmetic mismatch")
    if abs(
        combined_control
        - float(expected_combined["source_oof_jacobian_zero_control_macro_atom_id_ccc"])
    ) > 1e-12:
        raise ValueError("combined control arithmetic mismatch")
    if abs(
        combined_uniform_q
        - float(expected_combined["source_oof_uniform_q_macro_atom_id_ccc"])
    ) > 1e-12:
        raise ValueError("combined uniform-q arithmetic mismatch")
    combined_family_diagnostics = {
        "candidate": atom_family_macro_score(
            combined_frame, "candidate", inventory["eligible_atom_ids"]
        ),
        "no_coordinate": atom_family_macro_score(
            combined_frame, "control", inventory["eligible_atom_ids"]
        ),
        "uniform_q": atom_family_macro_score(
            combined_frame, "uniform_q", inventory["eligible_atom_ids"]
        ),
    }
    verify_family_diagnostics(
        expected_combined.get("atom_family_macro_atom_id_ccc"),
        combined_family_diagnostics,
        "combined",
    )
    combined = {
        "candidate": combined_candidate,
        "jacobian_zero_control": combined_control,
        "uniform_q": combined_uniform_q,
        "gain": combined_candidate - combined_control,
        "defined_label_count": combined_candidate_labels,
        "row_count": len(combined_frame),
        "atom_family_macro_atom_id_ccc": combined_family_diagnostics,
    }
    selected = all(
        float(folds[fold]["gain"]) > MINIMUM_GAIN_EACH_FOLD
        for fold in ("A", "B")
    ) and float(combined["gain"]) > MINIMUM_GAIN_EACH_FOLD
    if bool(result["gate"]["pass"]) is not selected:
        raise ValueError("runner/checker source decision mismatch")
    decision = {
        "contract": DECISION_CONTRACT,
        "ccc_used_for_scoring_not_loss": True,
        "minimum_gain_each_fold": MINIMUM_GAIN_EACH_FOLD,
        "folds": folds,
        "combined": combined,
        "selected_for_k32_followup": selected,
        "result_sha256": sha256_file(args.result),
        "source_commitment_sha256": source_commitment_sha256,
        "authorization_sha256": consumed["authorization_sha256"],
        "consumed_authorization_sha256": sha256_file(args.consumed_authorization),
        "authorization_git_blob": args.authorization_git_blob,
        "authorization_ref": args.authorization_ref,
        "external_claim_git_blob": external_claim_git_blob,
        "external_claim_sha256": sha256_file(args.external_authorization_claim),
        "container_image_sha256": args.container_image_sha256,
        "slurm_job_id": args.slurm_job_id,
    }
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    descriptor = args.decision.open("x")
    with descriptor:
        json.dump(decision, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
