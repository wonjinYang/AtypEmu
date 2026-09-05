#!/usr/bin/env python3
"""Two-way source-OOF gate for the CCC-free dynamic-coordinate loss path.

CCC is used only to score frozen source-OOF predictions.  It is never part of
observer training, evidence assimilation, packing, or regularization losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project-local science modules are deliberately loaded only after the external
# authorization has been consumed in main().  These declarations let helper
# functions resolve the late-bound callables without importing project code at
# process startup.
attach_ucbshift_anchor: Any = None
build_eligibility_receipt: Any = None
calibrate_reference_bounds: Any = None
coordinate_audit: Any = None
crossfit_anchor_selection: Any = None
embedding_path: Any = None
load_fold: Any = None
normalization: Any = None
optimize_assimilation: Any = None
predict_surface: Any = None
source_eligible_row_mask: Any = None
train_observer: Any = None
verify_support_structure_receipt: Any = None


COMMITMENT_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_commitment_v1"
AUTHORIZATION_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_authorization_v1"
CONSUMED_CONTRACT = "corrected_k8_dynamic_coordinate_source_gate_consumed_v1"
EXTERNAL_CLAIM_CONTRACT = (
    "corrected_k8_dynamic_coordinate_source_gate_external_claim_v1"
)
RESULT_CONTRACT = "corrected_k8_dynamic_coordinate_source_oof_gate_v1"
EXTERNAL_AUTHORIZATION_GIT_DIR = Path(
    "/scratch/wkyu514/yang07/atypemu_external_authorizations.git"
)
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
SEQUENCE_ANCHOR_SHA256 = {
    "A": "165f34ced53aabb4b908a43716ac8f4b76e4b3990507994b4fea0aa7d4fabd58",
    "B": "97346891f5fa7e7daa41a81a73cf5c90f9bbb577f293f6d816e76ea6576553ee",
}
SEQUENCE_ANCHOR_SUMMARY_SHA256 = {
    "A": "3f64bba6c197b7318b23f1587c4f218de027e89372aec754a8d21cb876642cef",
    "B": "2419aee1fc84353749565c201275694fcc8f864d972456f276017433a27e7f38",
}
SEQUENCE_ANCHOR_RECEIPT_SHA256 = (
    "2869e7b28d08580de1046c453a6acafe04f77a45a0b1eecd6aba86ec2bd6149f"
)
EXPECTED_SUPPORT_IDS = tuple(
    f"BioEmu_{index}" for index in (1, 126, 251, 376, 501, 626, 751, 876)
)


ROW_ARRAYS = (
    "esm",
    "torsion",
    "geometry",
    "numeric",
    "categorical",
    "residue_index",
    "entity_index",
    "distance_self_jacobian",
    "distance_neighbor_jacobian",
    "distance_neighbor_residue",
    "anchor",
    "sequence_support_anchor",
    "ucb_support_anchor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("x")
    with descriptor:
        json.dump(payload, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")


def load_science_api() -> None:
    """Load project-local model code only after authorization consumption."""

    from gpuopt import source_gate_eligibility as eligibility
    from gpuopt.candidates import corrected_k8_dynamic_coordinate as candidate

    global attach_ucbshift_anchor
    global build_eligibility_receipt
    global calibrate_reference_bounds
    global coordinate_audit
    global crossfit_anchor_selection
    global embedding_path
    global load_fold
    global normalization
    global optimize_assimilation
    global predict_surface
    global source_eligible_row_mask
    global train_observer
    global verify_support_structure_receipt
    attach_ucbshift_anchor = candidate.attach_ucbshift_anchor
    build_eligibility_receipt = eligibility.build_eligibility_receipt
    calibrate_reference_bounds = candidate.calibrate_reference_bounds
    coordinate_audit = candidate.coordinate_audit
    crossfit_anchor_selection = candidate.crossfit_anchor_selection
    embedding_path = candidate.embedding_path
    load_fold = candidate.load_fold
    normalization = candidate.normalization
    optimize_assimilation = candidate.optimize_assimilation
    predict_surface = candidate.predict_surface
    source_eligible_row_mask = eligibility.source_eligible_row_mask
    train_observer = candidate.train_observer
    verify_support_structure_receipt = candidate.verify_support_structure_receipt


def verify_output_paths_fresh(
    output: Path,
    consumed_authorization: Path | None = None,
    external_authorization_claim: Path | None = None,
) -> None:
    generated = [output]
    if consumed_authorization is not None:
        generated.append(consumed_authorization)
    if external_authorization_claim is not None:
        generated.append(external_authorization_claim)
    generated.extend(output.parent / f"source_oof_{fold}.parquet" for fold in ("A", "B"))
    generated.extend(
        output.parent / f"source_coordinate_{fold}_half{held_half}.npz"
        for fold in ("A", "B")
        for held_half in (0, 1)
    )
    existing = [path for path in generated if path.exists()]
    if existing:
        raise FileExistsError(f"corrected K8 output already exists: {existing[0]}")


def source_commitment_identity(path: Path) -> str:
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
    files = commitment.get("files")
    if not isinstance(files, dict) or not STATIC_COMMITTED_FILES.issubset(files):
        raise ValueError("source commitment omits a required static file")
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("source commitment file manifest is malformed")
    return sha256_file(path)


def verify_source_commitment(root: Path, path: Path) -> str:
    commitment = json.loads(path.read_text())
    for relative, expected in commitment["files"].items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"source commitment escapes root: {relative}")
        if sha256_file(candidate) != expected:
            raise ValueError(f"source commitment hash mismatch: {relative}")
    return sha256_file(path)


def expected_source_manifest(root: Path, data_commitment: dict[str, Any]) -> set[str]:
    """Enumerate every transitive source byte consumed by the corrected gate."""

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
        paths.add(embedding_path(embedding_cache, str(entity["sequence"])))
        support_ids = tuple(str(value) for value in entity["support_ids"])
        if support_ids != EXPECTED_SUPPORT_IDS:
            raise ValueError("source commitment entity support roster is not frozen K=8")
        for support_id in support_ids:
            paths.add(
                root / "data/BioEmu" / bmrb_id / f"{bmrb_id}_{support_id}.pdb"
            )
    return {str(path.relative_to(root)) for path in paths}


def verify_source_manifest_complete(
    root: Path, commitment_path: Path, data_commitment: dict[str, Any]
) -> None:
    commitment = json.loads(commitment_path.read_text())
    declared = set(commitment["files"])
    expected = expected_source_manifest(root, data_commitment)
    source_entity_count = sum(
        entity.get("split") == "train"
        and entity.get("observer_fold") in {"A", "B"}
        for entity in data_commitment["entities"]
    )
    if commitment.get("source_entity_count") != source_entity_count:
        raise ValueError("source commitment entity count mismatch")
    if declared != expected:
        raise ValueError(
            "source commitment manifest mismatch: "
            f"missing={sorted(expected - declared)[:3]} "
            f"extra={sorted(declared - expected)[:3]}"
        )


def consume_authorization(
    path: Path,
    consumed_path: Path,
    external_claim_path: Path,
    *,
    source_commitment_sha256: str,
    epochs: int,
    steps: int,
    authorization_git_blob: str,
    authorization_ref: str,
    container_image_sha256: str,
    slurm_job_id: str,
    authorization_git_dir: Path | None = None,
) -> tuple[str, str, str, str]:
    if authorization_git_dir is None:
        authorization_git_dir = EXTERNAL_AUTHORIZATION_GIT_DIR
    if not slurm_job_id.isdigit():
        raise ValueError("source gate requires a concrete numeric Slurm job ID")
    expected_ref = (
        f"refs/atypemu-authorizations/corrected-k8/job-{slurm_job_id}"
    )
    if authorization_ref != expected_ref:
        raise ValueError("source-gate authorization ref/job mismatch")
    authorization = json.loads(path.read_text())
    if authorization.get("contract") != AUTHORIZATION_CONTRACT:
        raise ValueError("source-gate authorization contract mismatch")
    expected = {
        "source_commitment_sha256": source_commitment_sha256,
        "epochs": epochs,
        "steps": steps,
        "authorized": True,
        "container_image_sha256": container_image_sha256,
        "slurm_job_id": slurm_job_id,
        "authorization_ref": authorization_ref,
    }
    if set(authorization) != {"contract", *expected}:
        raise ValueError("source-gate authorization schema mismatch")
    for field, value in expected.items():
        if authorization.get(field) != value:
            raise ValueError(f"source-gate authorization mismatch: {field}")
    authorization_sha256 = sha256_file(path)
    authorization_bytes = path.read_bytes()
    computed_git_blob = hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(authorization_bytes)}\0".encode() + authorization_bytes
    ).hexdigest()
    if computed_git_blob != authorization_git_blob:
        raise ValueError("authorization Git-blob binding mismatch")
    resolved_blob = subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "rev-parse",
            f"{authorization_ref}^{{blob}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_blob != authorization_git_blob:
        raise ValueError("external authorization ref is not fresh")
    external_claim = {
        "contract": EXTERNAL_CLAIM_CONTRACT,
        "authorization_git_blob": authorization_git_blob,
        "authorization_ref": authorization_ref,
        "authorization_sha256": authorization_sha256,
        "container_image_sha256": container_image_sha256,
        "slurm_job_id": slurm_job_id,
        "source_commitment_sha256": source_commitment_sha256,
    }
    write_json_new(external_claim_path, external_claim)
    external_claim_sha256 = sha256_file(external_claim_path)
    external_claim_git_blob = subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "hash-object",
            "-w",
            str(external_claim_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "update-ref",
            authorization_ref,
            external_claim_git_blob,
            authorization_git_blob,
        ],
        check=True,
    )
    consumed = {
        "contract": CONSUMED_CONTRACT,
        "authorization_sha256": authorization_sha256,
        "source_commitment_sha256": source_commitment_sha256,
        "authorization_git_blob": authorization_git_blob,
        "authorization_ref": authorization_ref,
        "external_claim_git_blob": external_claim_git_blob,
        "external_claim_sha256": external_claim_sha256,
        "container_image_sha256": container_image_sha256,
        "slurm_job_id": slurm_job_id,
    }
    consumed_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = consumed_path.open("x")
    with descriptor:
        json.dump(consumed, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")
    return (
        authorization_sha256,
        sha256_file(consumed_path),
        external_claim_sha256,
        external_claim_git_blob,
    )


def subset_rows(values: dict[str, Any], row_mask: np.ndarray) -> dict[str, Any]:
    """Subset every row surface and remap entity/residue indices exactly once."""

    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.shape != (len(values["frame"]),) or not row_mask.any():
        raise ValueError("source subset has no selected rows")
    old_entities = sorted(
        set(int(value) for value in np.asarray(values["entity_index"])[row_mask])
    )
    entity_remap = {old: new for new, old in enumerate(old_entities)}
    output = dict(values)
    output["frame"] = values["frame"].loc[row_mask].reset_index(drop=True)
    for key in ROW_ARRAYS:
        if key in values:
            output[key] = np.asarray(values[key])[row_mask].copy()
    output["entities"] = [values["entities"][index] for index in old_entities]
    output["entity_index"] = np.asarray(
        [entity_remap[int(value)] for value in values["entity_index"][row_mask]],
        dtype=np.int64,
    )

    old_residue_index = np.asarray(values["residue_index"])[row_mask]
    old_residues = sorted(set(int(value) for value in old_residue_index))
    residue_remap = {old: new for new, old in enumerate(old_residues)}
    output["residue_index"] = np.asarray(
        [residue_remap[int(value)] for value in old_residue_index], dtype=np.int64
    )
    output["residue_keys"] = [values["residue_keys"][index] for index in old_residues]
    neighbor = output["distance_neighbor_residue"]
    remapped_neighbor = np.full_like(neighbor, -1)
    for old, new in residue_remap.items():
        remapped_neighbor[neighbor == old] = new
    output["distance_neighbor_residue"] = remapped_neighbor
    return output


def subset_fold(values: dict[str, Any], entity_numbers: list[int]) -> dict[str, Any]:
    selected = set(entity_numbers)
    return subset_rows(
        values,
        np.asarray(
            [int(value) in selected for value in values["entity_index"]], dtype=bool
        ),
    )


def restrict_source_subset_eligibility(
    values: dict[str, Any],
    *,
    frozen_atom_ids: list[str],
    fold: str,
    held_half: int,
    role: str,
) -> dict[str, Any]:
    """Apply target-only label eligibility before any fit or assimilation phase."""

    frame = values["frame"].reset_index(drop=True)
    row_mask = np.asarray(
        source_eligible_row_mask(
            frame["atom_id"].astype(str).tolist(),
            frame["target_value"].astype(float).tolist(),
            frozen_atom_ids,
        ),
        dtype=bool,
    )
    selected = subset_rows(values, row_mask)
    receipt = build_eligibility_receipt(
        frame,
        selected["frame"],
        frozen_atom_ids=frozen_atom_ids,
        fold=fold,
        held_half=held_half,
        role=role,
    )
    selected["source_eligibility_receipt"] = receipt
    return selected


def sequence_cluster_halves(entities: list[dict[str, Any]]) -> dict[int, int]:
    clusters: dict[str, list[int]] = {}
    for index, entity in enumerate(entities):
        clusters.setdefault(str(entity["sequence_cluster_id"]), []).append(index)
    loads = [0, 0]
    assignment: dict[int, int] = {}
    for _cluster_id, members in sorted(
        clusters.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        half = min(range(2), key=lambda index: (loads[index], index))
        for member in members:
            assignment[member] = half
        loads[half] += len(members)
    if not all(loads):
        raise ValueError("source fold cannot form two sequence-cluster-disjoint halves")
    return assignment


def attach_sequence_anchor(values: dict[str, Any], fold: str, root: Path) -> None:
    anchor_root = root / "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0" / fold
    prediction_path = anchor_root / "predictions.parquet"
    summary_path = anchor_root / "summary.json"
    if sha256_file(prediction_path) != SEQUENCE_ANCHOR_SHA256[fold]:
        raise ValueError(f"sequence anchor hash mismatch: {fold}")
    if sha256_file(summary_path) != SEQUENCE_ANCHOR_SUMMARY_SHA256[fold]:
        raise ValueError(f"sequence anchor summary hash mismatch: {fold}")
    if sha256_file(anchor_root.parent / "receipt.json") != SEQUENCE_ANCHOR_RECEIPT_SHA256:
        raise ValueError("sequence anchor receipt hash mismatch")
    anchor = pd.read_parquet(prediction_path, columns=("target_id", "prediction"))
    anchor["entity_uid"] = anchor["target_id"].astype(str).str.split(
        ":target:", n=1
    ).str[0]
    if anchor.duplicated(["entity_uid", "target_id"]).any():
        raise ValueError(f"duplicate sequence-anchor identity: {fold}")
    anchor_values = anchor.set_index(["entity_uid", "target_id"])["prediction"]
    frame_key = pd.MultiIndex.from_frame(values["frame"][["entity_uid", "target_id"]])
    aligned = pd.Series(
        anchor_values.reindex(frame_key).to_numpy(), index=values["frame"].index
    )
    if aligned.isna().any() or not np.isfinite(aligned.to_numpy(float)).all():
        raise ValueError(f"incomplete sequence anchor: {fold}")
    values["anchor"] = aligned.to_numpy(dtype=np.float32)


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


def macro_atom_id_ccc(
    frame: pd.DataFrame, prediction: np.ndarray, eligible_atom_ids: list[str]
) -> tuple[float, dict[str, float]]:
    per_label: dict[str, float] = {}
    atom = frame["atom_id"].astype(str).to_numpy()
    target = frame["target_value"].to_numpy(dtype=np.float64)
    for atom_id in eligible_atom_ids:
        rows = atom == atom_id
        atom_target = target[rows]
        if len(atom_target) < 2 or float(np.var(atom_target)) <= 1.0e-15:
            continue
        value = concordance(atom_target, prediction[rows])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite prediction-defined source CCC: {atom_id}")
        per_label[atom_id] = value
    if not per_label:
        raise ValueError("source OOF gate has no defined Atom_ID CCC")
    return float(np.mean(list(per_label.values()))), per_label


def atom_family_macro_diagnostics(per_label: dict[str, float]) -> dict[str, float]:
    diagnostics = {}
    for family in ("C", "H", "N"):
        values = [value for atom_id, value in per_label.items() if atom_id[0] == family]
        if not values:
            raise ValueError(f"source OOF has no defined {family} Atom_ID labels")
        diagnostics[family] = float(np.mean(values))
    return diagnostics


def aggregate_prediction(
    values: dict[str, Any], surface: np.ndarray, q: torch.Tensor
) -> np.ndarray:
    q_array = q.detach().cpu().numpy()
    return np.sum(surface * q_array[values["entity_index"]], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commitment", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--consumed-authorization", type=Path, required=True)
    parser.add_argument("--external-authorization-claim", type=Path, required=True)
    parser.add_argument("--authorization-git-blob", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--container-image-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs != EXPECTED_EPOCHS or args.steps != EXPECTED_STEPS:
        raise ValueError("corrected K8 gate requires frozen 1024 epochs and 100 steps")
    verify_output_paths_fresh(
        args.output,
        args.consumed_authorization,
        args.external_authorization_claim,
    )
    root = args.root.resolve()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    source_commitment_sha256 = source_commitment_identity(args.source_commitment)
    (
        authorization_sha256,
        consumed_authorization_sha256,
        external_claim_sha256,
        external_claim_git_blob,
    ) = consume_authorization(
        args.authorization,
        args.consumed_authorization,
        args.external_authorization_claim,
        source_commitment_sha256=source_commitment_sha256,
        epochs=args.epochs,
        steps=args.steps,
        authorization_git_blob=args.authorization_git_blob,
        authorization_ref=args.authorization_ref,
        container_image_sha256=args.container_image_sha256,
        slurm_job_id=slurm_job_id,
    )
    if verify_source_commitment(root, args.source_commitment) != source_commitment_sha256:
        raise ValueError("verified source commitment identity mismatch")
    load_science_api()
    invalidation_path = root / (
        "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/"
        "post_handoff_audit.json"
    )
    if sha256_file(invalidation_path) != CANONICAL_INVALIDATION_SHA256:
        raise ValueError("canonical Job 134710 invalidation audit mismatch")
    data_root = root / "data/all_atom_observer_v1"
    inventory = json.loads((root / ".auto/frozen/all_label_inventory.json").read_text())
    commitment = json.loads((data_root / "commitment.json").read_text())
    verify_source_manifest_complete(root, args.source_commitment, commitment)
    atom_index = {"<UNK>": 0}
    atom_index.update(
        {
            atom_id: index
            for index, atom_id in enumerate(inventory["eligible_atom_ids"], start=1)
        }
    )
    embedding_cache = data_root.parent / "all_atom_shared_q_e2e_v0/esm2_cache"
    structure_root = data_root.parent / "BioEmu"
    ucb_root = data_root.parent / "all_atom_shared_q_e2e_v0/ucbshift_x_anchor_v0"
    device = torch.device("cuda")
    result: dict[str, Any] = {
        "contract": RESULT_CONTRACT,
        "correction": (
            "source_subset_eligibility_precedes_anchor_selection_normalization_"
            "observer_fitting_reference_calibration_and_assimilation"
        ),
        "ccc_used_only_for_source_oof_scoring": True,
        "ccc_used_in_loss": False,
        "no_coordinate_control": (
            "independently_optimized_q_reference_with_geometry_jacobian_zeroed"
        ),
        "uniform_q_control": (
            "independently_optimized_coordinate_reference_with_q_fixed_uniform"
        ),
        "folds": {},
        "output_sha256": {},
        "source_commitment_sha256": source_commitment_sha256,
        "authorization_sha256": authorization_sha256,
        "consumed_authorization_sha256": consumed_authorization_sha256,
        "authorization_git_blob": args.authorization_git_blob,
        "authorization_ref": args.authorization_ref,
        "external_claim_git_blob": external_claim_git_blob,
        "external_claim_sha256": external_claim_sha256,
        "container_image_sha256": args.container_image_sha256,
        "slurm_job_id": slurm_job_id,
        "epochs": args.epochs,
        "steps": args.steps,
        "support_structure_receipt_audit": verify_support_structure_receipt(
            data_root, structure_root
        ),
    }

    combined_oof_frames = []
    for fold_number, fold in enumerate(("A", "B")):
        entities = sorted(
            (
                entity
                for entity in commitment["entities"]
                if entity.get("split") == "train"
                and entity.get("observer_fold") == fold
            ),
            key=lambda entity: str(entity["entity_uid"]),
        )
        for entity in entities:
            if tuple(str(value) for value in entity["support_ids"]) != EXPECTED_SUPPORT_IDS:
                raise ValueError(f"source support roster differs from frozen K=8: {fold}")
        values = load_fold(
            data_root,
            entities,
            atom_index=atom_index,
            embedding_cache=embedding_cache,
            structure_root=structure_root,
        )
        attach_sequence_anchor(values, fold, root)
        attach_ucbshift_anchor(values, ucb_root)
        oof_rows: list[pd.DataFrame] = []
        data_gradient_norms: list[float] = []
        regularizer_gradient_norms: list[float] = []
        eligibility_receipts: dict[str, dict[str, Any]] = {}
        expected_oof_row_count = 0
        half_assignment = sequence_cluster_halves(entities)
        for held_half in (0, 1):
            held_entities = [
                index
                for index in range(len(entities))
                if half_assignment[index] == held_half
            ]
            train_entities = [
                index
                for index in range(len(entities))
                if half_assignment[index] != held_half
            ]
            held_clusters = {
                str(entities[index]["sequence_cluster_id"])
                for index in held_entities
            }
            train_clusters = {
                str(entities[index]["sequence_cluster_id"])
                for index in train_entities
            }
            if held_clusters & train_clusters:
                raise ValueError(f"source OOF shares a sequence cluster: {fold}")
            train = subset_fold(values, train_entities)
            evaluation = subset_fold(values, held_entities)
            train = restrict_source_subset_eligibility(
                train,
                frozen_atom_ids=inventory["eligible_atom_ids"],
                fold=fold,
                held_half=held_half,
                role="observer_training",
            )
            evaluation = restrict_source_subset_eligibility(
                evaluation,
                frozen_atom_ids=inventory["eligible_atom_ids"],
                fold=fold,
                held_half=held_half,
                role="evidence_assimilation",
            )
            eligibility_receipts[str(held_half)] = {
                "observer_training": train["source_eligibility_receipt"],
                "evidence_assimilation": evaluation[
                    "source_eligibility_receipt"
                ],
            }
            expected_oof_row_count += len(evaluation["frame"])
            crossfit_anchor_selection(
                train,
                evaluation,
                refine_by_comp_id=fold == "B",
            )
            train, evaluation = normalization(train, evaluation)
            reference_bounds = calibrate_reference_bounds(train)
            model = train_observer(
                train,
                atom_levels=len(atom_index),
                device=device,
                seed=2026090300 + 10 * fold_number + held_half,
                epochs=args.epochs,
            )
            delta, q, reference, data_gradient, regularizer_gradient = (
                optimize_assimilation(
                    model,
                    evaluation,
                    device=device,
                    reference_bounds_ppm=reference_bounds,
                    steps=args.steps,
                )
            )
            control_delta, control_q, control_reference, _, _ = optimize_assimilation(
                model,
                evaluation,
                device=device,
                reference_bounds_ppm=reference_bounds,
                steps=args.steps,
                geometry_jacobian_scale=0.0,
            )
            uniform_delta, uniform_q, uniform_reference, _, _ = optimize_assimilation(
                model,
                evaluation,
                device=device,
                reference_bounds_ppm=reference_bounds,
                steps=args.steps,
                fixed_uniform_q=True,
            )
            candidate_surface = predict_surface(
                model,
                evaluation,
                device=device,
                delta_raw=delta,
                reference_offset=reference,
            )
            control_surface = predict_surface(
                model,
                evaluation,
                device=device,
                delta_raw=control_delta,
                reference_offset=control_reference,
                geometry_jacobian_scale=0.0,
            )
            uniform_surface = predict_surface(
                model,
                evaluation,
                device=device,
                delta_raw=uniform_delta,
                reference_offset=uniform_reference,
            )
            part = evaluation["frame"].copy()
            cluster_by_entity = {
                str(entity["entity_uid"]): str(entity["sequence_cluster_id"])
                for entity in evaluation["entities"]
            }
            part["sequence_cluster_id"] = part["entity_uid"].astype(str).map(
                cluster_by_entity
            )
            if part["sequence_cluster_id"].isna().any():
                raise ValueError(f"missing source sequence cluster: {fold}")
            part["candidate"] = aggregate_prediction(evaluation, candidate_surface, q)
            part["control"] = aggregate_prediction(
                evaluation, control_surface, control_q
            )
            part["uniform_q"] = aggregate_prediction(
                evaluation, uniform_surface, uniform_q
            )
            part["held_half"] = held_half
            oof_rows.append(part)
            coordinate_path = args.output.parent / (
                f"source_coordinate_{fold}_half{held_half}.npz"
            )
            if coordinate_path.exists():
                raise FileExistsError(coordinate_path)
            coordinate_audit(
                evaluation,
                delta,
                structure_root=structure_root,
                output=coordinate_path,
            )
            result["output_sha256"][coordinate_path.name] = sha256_file(
                coordinate_path
            )
            data_gradient_norms.append(data_gradient)
            regularizer_gradient_norms.append(regularizer_gradient)
            del (
                model,
                train,
                evaluation,
                delta,
                q,
                reference,
                control_delta,
                control_q,
                control_reference,
                uniform_delta,
                uniform_q,
                uniform_reference,
                candidate_surface,
                control_surface,
                uniform_surface,
            )
            torch.cuda.empty_cache()
        oof = pd.concat(oof_rows, ignore_index=True)
        if (
            oof.duplicated(["entity_uid", "target_id"]).any()
            or len(oof) != expected_oof_row_count
        ):
            raise ValueError(f"OOF identity coverage mismatch: {fold}")
        candidate_score, candidate_labels = macro_atom_id_ccc(
            oof, oof["candidate"].to_numpy(float), inventory["eligible_atom_ids"]
        )
        control_score, control_labels = macro_atom_id_ccc(
            oof, oof["control"].to_numpy(float), inventory["eligible_atom_ids"]
        )
        uniform_score, uniform_labels = macro_atom_id_ccc(
            oof, oof["uniform_q"].to_numpy(float), inventory["eligible_atom_ids"]
        )
        if not (
            set(candidate_labels) == set(control_labels) == set(uniform_labels)
        ):
            raise ValueError(f"source control label inventory mismatch: {fold}")
        oof_path = args.output.parent / f"source_oof_{fold}.parquet"
        with oof_path.open("xb") as handle:
            oof[
                [
                    "target_id",
                    "entity_uid",
                    "atom_id",
                    "target_value",
                    "sequence_cluster_id",
                    "held_half",
                    "candidate",
                    "control",
                    "uniform_q",
                ]
            ].to_parquet(handle, index=False)
        result["output_sha256"][oof_path.name] = sha256_file(oof_path)
        result["folds"][fold] = {
            "source_oof_candidate_macro_atom_id_ccc": candidate_score,
            "source_oof_jacobian_zero_control_macro_atom_id_ccc": control_score,
            "source_oof_uniform_q_macro_atom_id_ccc": uniform_score,
            "candidate_minus_control": candidate_score - control_score,
            "defined_label_count": len(candidate_labels),
            "row_count": len(oof),
            "data_loss_gradient_norms": data_gradient_norms,
            "regularizer_gradient_norms": regularizer_gradient_norms,
            "eligibility": eligibility_receipts,
            "atom_family_macro_atom_id_ccc": {
                "candidate": atom_family_macro_diagnostics(candidate_labels),
                "no_coordinate": atom_family_macro_diagnostics(control_labels),
                "uniform_q": atom_family_macro_diagnostics(uniform_labels),
            },
        }
        combined_oof_frames.append(oof)
        del values

    combined_oof = pd.concat(combined_oof_frames, ignore_index=True)
    combined_candidate_score, combined_candidate_labels = macro_atom_id_ccc(
        combined_oof,
        combined_oof["candidate"].to_numpy(float),
        inventory["eligible_atom_ids"],
    )
    combined_control_score, combined_control_labels = macro_atom_id_ccc(
        combined_oof,
        combined_oof["control"].to_numpy(float),
        inventory["eligible_atom_ids"],
    )
    combined_uniform_score, combined_uniform_labels = macro_atom_id_ccc(
        combined_oof,
        combined_oof["uniform_q"].to_numpy(float),
        inventory["eligible_atom_ids"],
    )
    if set(combined_candidate_labels) != set(inventory["eligible_atom_ids"]):
        raise ValueError("combined source OOF does not define the frozen label inventory")
    if set(combined_control_labels) != set(inventory["eligible_atom_ids"]):
        raise ValueError("combined control does not define the frozen label inventory")
    if set(combined_uniform_labels) != set(inventory["eligible_atom_ids"]):
        raise ValueError("combined uniform-q does not define the frozen label inventory")
    result["combined"] = {
        "source_oof_candidate_macro_atom_id_ccc": combined_candidate_score,
        "source_oof_jacobian_zero_control_macro_atom_id_ccc": combined_control_score,
        "source_oof_uniform_q_macro_atom_id_ccc": combined_uniform_score,
        "candidate_minus_control": combined_candidate_score - combined_control_score,
        "defined_label_count": len(combined_candidate_labels),
        "row_count": len(combined_oof),
        "atom_family_macro_atom_id_ccc": {
            "candidate": atom_family_macro_diagnostics(combined_candidate_labels),
            "no_coordinate": atom_family_macro_diagnostics(combined_control_labels),
            "uniform_q": atom_family_macro_diagnostics(combined_uniform_labels),
        },
    }
    result["gate"] = {
        "minimum_gain_each_fold": 0.0005,
        "both_folds_pass": all(
            result["folds"][fold]["candidate_minus_control"] > 0.0005
            for fold in ("A", "B")
        ),
        "combined_frozen_inventory_pass": (
            result["combined"]["candidate_minus_control"] > 0.0005
        ),
        "all_data_loss_gradients_finite_positive": all(
            math.isfinite(value) and value > 0.0
            for fold in ("A", "B")
            for value in result["folds"][fold]["data_loss_gradient_norms"]
        ),
    }
    result["gate"]["pass"] = all(result["gate"].values())
    if verify_source_commitment(root, args.source_commitment) != source_commitment_sha256:
        raise ValueError("source commitment changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = args.output.open("x")
    with descriptor:
        json.dump(result, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")
    print(json.dumps(result["gate"], sort_keys=True))
    for fold in ("A", "B"):
        print(
            f"source_fold_{fold}_candidate_minus_control="
            f"{result['folds'][fold]['candidate_minus_control']:.12f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
