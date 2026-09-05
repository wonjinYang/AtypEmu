#!/usr/bin/env python3
"""Freeze and materialize a source-only K=32 dynamic-distance cache.

This module deliberately reads only the source feature parquet files.  The
coordinates are the already materialized, fixed K=32 parent asset.  In
particular, no label/value table is needed to construct this cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


CONTRACT = "k32_dynamic_distance_cache_source_commitment_v1"
RECEIPT_CONTRACT = "k32_dynamic_distance_cache_receipt_v1"
PARENT_RECEIPT_CONTRACT = "k32_complete_coordinate_support_asset_v1"
SCRIPT_RELATIVE = "gpuopt/materialize_k32_dynamic_distance_cache.py"
PLAN_RELATIVE = "gpuopt/preunblind/k32_dynamic_distance_cache_plan_v1.json"
PARENT_RELATIVE = "data/all_atom_observer_v1/commitment.json"
PARENT_RECEIPT_RELATIVE = (
    ".auto/runs/k32_complete_coordinate_supports_v4_receipt.json"
)
PARENT_SOURCE_COMMITMENT_RELATIVE = ".auto/staging/k32_source_commitment_v4.json"
PARENT_ASSET_RELATIVE = "data/k32_complete_coordinate_supports_v4"
FEATURE_ROOT_RELATIVE = "data/all_atom_observer_v1/features"
OUTPUT_ROOT_RELATIVE = "data/k32_dynamic_distance_cache_v1"
SOURCE_COMMITMENT_RELATIVE = (
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
)
RECEIPT_RELATIVE = ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json"
CORRECTED_K8_DECISION_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/decision.json"
)
CORRECTED_K8_BACKPRESSURE_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "backpressure_recovery_receipt.json"
)
CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "corrected_k8_source_commitment.json"
)
CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "consumed_authorization.json"
)
CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/external_claim.json"
)
BOUND_PREREQUISITES = (
    PLAN_RELATIVE,
    PARENT_SOURCE_COMMITMENT_RELATIVE,
    CORRECTED_K8_DECISION_RELATIVE,
    CORRECTED_K8_BACKPRESSURE_RELATIVE,
    CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE,
    CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE,
    CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE,
)
EXPECTED_ENTITY_COUNT = 135
EXPECTED_FEATURE_ROW_COUNT = 127_285
EXPECTED_TARGET_SUPPORT_ROWS = EXPECTED_FEATURE_ROW_COUNT * 32
EXPECTED_TARGET_AVAILABLE_ROWS = 4_072_968
EXPECTED_TARGET_MISSING_ROWS = 152
ELEMENTS = ("H", "C", "N", "O", "S")
DISTANCE_FILL = 10.0
CHI_COUNT = 4

# This is intentionally the same ordered roster as
# gpuopt/check_k32_complete_coordinate_supports.py::EXPECTED_SUPPORTS.  Keeping
# it in the commitment builder makes the commitment independently auditable and
# avoids importing a checker (or a candidate) while materializing science data.
EXPECTED_SUPPORTS = (
    1,
    32,
    63,
    94,
    126,
    157,
    188,
    221,
    251,
    281,
    312,
    344,
    376,
    407,
    438,
    469,
    501,
    533,
    565,
    595,
    626,
    656,
    687,
    719,
    751,
    781,
    811,
    843,
    876,
    906,
    937,
    968,
)
EXPECTED_K8_SUPPORT_IDS = tuple(
    f"BioEmu_{index}" for index in (1, 126, 251, 376, 501, 626, 751, 876)
)
SUPPORT_IDS = tuple(f"BioEmu_{index}" for index in EXPECTED_SUPPORTS)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Write one durable JSON artifact, never replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _path_under(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes root: {path}")
    return resolved


def _script_path(root: Path) -> Path:
    candidate = root / SCRIPT_RELATIVE
    if not candidate.is_file():
        raise FileNotFoundError(f"root-bound materializer missing: {candidate}")
    return candidate


def _source_entities(parent: dict[str, Any]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in parent.get("entities", []):
        if raw.get("split") != "train" or raw.get("observer_fold") not in {"A", "B"}:
            continue
        entity_uid = str(raw.get("entity_uid", ""))
        bmrb_id = str(raw.get("bmrb_id", ""))
        if not entity_uid or not bmrb_id:
            raise ValueError("source entity lacks entity_uid or bmrb_id")
        if re.fullmatch(r"bmr[0-9]+", bmrb_id) is None:
            raise ValueError(f"invalid source BMRB path token: {bmrb_id!r}")
        if entity_uid in seen:
            raise ValueError(f"duplicate source entity: {entity_uid}")
        seen.add(entity_uid)
        selected.append(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "split": "train",
                "observer_fold": str(raw["observer_fold"]),
            }
        )
    if len(selected) != EXPECTED_ENTITY_COUNT:
        raise ValueError(
            f"source roster must contain {EXPECTED_ENTITY_COUNT} entities, got {len(selected)}"
        )
    return sorted(selected, key=lambda row: (row["entity_uid"], row["bmrb_id"]))


def _entity_roster(parent: dict[str, Any]) -> list[dict[str, str]]:
    return _source_entities(parent)


def _parent_output_roster(
    root: Path,
    receipt: dict[str, Any],
    entities: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if receipt.get("contract") != PARENT_RECEIPT_CONTRACT:
        raise ValueError("parent K32 receipt contract mismatch")
    for field in (
        "target_values_read",
        "source_gate_authorized",
        "formal_evaluation_authorized",
    ):
        if receipt.get(field) is not False:
            raise ValueError(f"parent K32 receipt scope mismatch: {field}")
    by_entity = {row["entity_uid"]: row for row in entities}
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in receipt.get("outputs", []):
        entity_uid = str(raw.get("entity_uid", ""))
        support_index = int(raw.get("support_index", -1))
        key = (entity_uid, support_index)
        if key in observed:
            raise ValueError(f"duplicate parent K32 output: {key}")
        if entity_uid not in by_entity:
            continue
        if support_index not in EXPECTED_SUPPORTS:
            raise ValueError(f"parent K32 source support roster mismatch: {key}")
        bmrb_id = by_entity[entity_uid]["bmrb_id"]
        support_id = f"BioEmu_{support_index}"
        expected_relative = (
            f"{PARENT_ASSET_RELATIVE}/{bmrb_id}/{bmrb_id}_{support_id}.pdb"
        )
        if str(raw.get("bmrb_id")) != bmrb_id:
            raise ValueError(f"parent K32 BMRB identity mismatch: {key}")
        if str(raw.get("support_id")) != support_id:
            raise ValueError(f"parent K32 support identity mismatch: {key}")
        if str(raw.get("output_relative_path")) != expected_relative:
            raise ValueError(f"parent K32 output path mismatch: {key}")
        output = _path_under(root, root / expected_relative)
        expected_hash = str(raw.get("output_sha256", ""))
        if len(expected_hash) != 64 or not output.is_file():
            raise ValueError(f"missing parent K32 output: {key}")
        if sha256_file(output) != expected_hash:
            raise ValueError(f"parent K32 output hash mismatch: {key}")
        observed[key] = {
            "entity_uid": entity_uid,
            "bmrb_id": bmrb_id,
            "support_id": support_id,
            "support_index": support_index,
            "relative_path": expected_relative,
            "sha256": expected_hash,
        }
    expected_keys = {
        (entity["entity_uid"], support) for entity in entities for support in EXPECTED_SUPPORTS
    }
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))[:3]
        extra = sorted(set(observed) - expected_keys)[:3]
        raise ValueError(f"parent K32 source roster mismatch: missing={missing}, extra={extra}")
    return [
        observed[(entity["entity_uid"], support)]
        for entity in entities
        for support in EXPECTED_SUPPORTS
    ]


def _feature_roster(root: Path, entities: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_bmrb = {entity["bmrb_id"]: entity["entity_uid"] for entity in entities}
    if len(by_bmrb) != len(entities):
        raise ValueError("source feature files require one BMRB ID per entity")
    rows = []
    for bmrb_id, entity_uid in sorted(by_bmrb.items()):
        relative = f"{FEATURE_ROOT_RELATIVE}/{bmrb_id}.parquet"
        path = _path_under(root, root / relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing source feature parquet: {path}")
        columns = set(pq.read_schema(path).names)
        forbidden = {"target_value"} & columns
        if forbidden:
            raise ValueError(f"target-bearing source feature parquet: {relative}: {forbidden}")
        required = {"entity_uid", "split", "observer_fold", "support_id"}
        if not required <= columns:
            raise ValueError(f"source feature schema is incomplete: {relative}")
        scope = pd.read_parquet(path, columns=sorted(required))
        if (
            set(scope["entity_uid"].astype(str).unique()) != {entity_uid}
            or set(scope["split"].astype(str).unique()) != {"train"}
            or not set(scope["observer_fold"].astype(str).unique()) <= {"A", "B"}
            or set(scope["support_id"].astype(str).unique())
            != set(EXPECTED_K8_SUPPORT_IDS)
        ):
            raise ValueError(f"source feature scope mismatch: {relative}")
        rows.append(
            {
                "bmrb_id": bmrb_id,
                "entity_uid": entity_uid,
                "relative_path": relative,
                "sha256": sha256_file(path),
                "row_count": int(len(scope)),
            }
        )
    if len(rows) != EXPECTED_ENTITY_COUNT:
        raise ValueError(
            f"source feature roster must contain {EXPECTED_ENTITY_COUNT} files, got {len(rows)}"
        )
    return rows


def _commitment_payload(
    root: Path,
    parent_path: Path,
    parent_receipt_path: Path,
) -> dict[str, Any]:
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    parent_source_path = _path_under(root, root / PARENT_SOURCE_COMMITMENT_RELATIVE)
    if not parent_source_path.is_file():
        raise FileNotFoundError(f"missing parent K32 source commitment: {parent_source_path}")
    parent_source = json.loads(parent_source_path.read_text(encoding="utf-8"))
    if (
        parent_source.get("contract") != "k32_support_asset_source_commitment_v1"
        or parent_source.get("target_values_read") is not False
        or parent_source.get("source_gate_authorized") is not False
        or parent_source.get("formal_evaluation_authorized") is not False
        or tuple(parent_source.get("support_indices", [])) != EXPECTED_SUPPORTS
        or parent_source.get("support_count") != 32
        or parent_source.get("pair_count") != 5_184
    ):
        raise ValueError("parent K32 source commitment contract mismatch")
    if parent_receipt.get("source_commitment_sha256") != sha256_file(parent_source_path):
        raise ValueError("parent K32 receipt/source-commitment binding mismatch")
    entities = _entity_roster(parent)
    pdb_files = _parent_output_roster(root, parent_receipt, entities)
    features = _feature_roster(root, entities)
    script = _script_path(root)
    bound_files = []
    for relative in BOUND_PREREQUISITES:
        path = _path_under(root, root / relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing bound prerequisite: {path}")
        bound_files.append({"relative_path": relative, "sha256": sha256_file(path)})
    corrected_k8_decision = json.loads(
        (root / CORRECTED_K8_DECISION_RELATIVE).read_text(encoding="utf-8")
    )
    corrected_k8_backpressure = json.loads(
        (root / CORRECTED_K8_BACKPRESSURE_RELATIVE).read_text(encoding="utf-8")
    )
    plan = json.loads((root / PLAN_RELATIVE).read_text(encoding="utf-8"))
    if (
        plan.get("contract") != "atypemu-k32-dynamic-distance-source-cache-plan-v1"
        or plan.get("scope", {}).get("entity_count") != EXPECTED_ENTITY_COUNT
        or plan.get("scope", {}).get("target_identity_count")
        != EXPECTED_FEATURE_ROW_COUNT
        or plan.get("scope", {}).get("target_support_count")
        != EXPECTED_TARGET_SUPPORT_ROWS
        or plan.get("scope", {}).get("support_count") != 32
        or plan.get("provenance", {}).get("parent_k32_receipt_sha256")
        != sha256_file(parent_receipt_path)
        or plan.get("provenance", {}).get("corrected_k8_decision_sha256")
        != sha256_file(root / CORRECTED_K8_DECISION_RELATIVE)
        or plan.get("provenance", {}).get(
            "corrected_k8_backpressure_recovery_receipt_sha256"
        )
        != sha256_file(root / CORRECTED_K8_BACKPRESSURE_RELATIVE)
        or plan.get("acceptance", {}).get("target_values_read") is not False
        or plan.get("acceptance", {}).get("source_gate_authorized") is not False
        or plan.get("acceptance", {}).get("formal_evaluation_authorized") is not False
    ):
        raise ValueError("dynamic-distance plan contract mismatch")
    k8_source_path = root / CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE
    k8_consumed_path = root / CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE
    k8_claim_path = root / CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE
    if (
        corrected_k8_decision.get("contract")
        != "corrected_k8_dynamic_coordinate_source_oof_check_v1"
        or corrected_k8_decision.get("selected_for_k32_followup") is not True
        or corrected_k8_decision.get("source_commitment_sha256")
        != sha256_file(k8_source_path)
        or corrected_k8_decision.get("consumed_authorization_sha256")
        != sha256_file(k8_consumed_path)
        or corrected_k8_decision.get("external_claim_sha256")
        != sha256_file(k8_claim_path)
    ):
        raise ValueError("corrected-K8 decision binding mismatch")
    if (
        corrected_k8_backpressure.get("contract")
        != "corrected_k8_job135711_backpressure_recovery_receipt_v1"
        or corrected_k8_backpressure.get("backpressure_checks_passed") is not True
        or corrected_k8_backpressure.get("source_science_rerun") is not False
        or corrected_k8_backpressure.get("formal_metrics_opened") is not False
        or corrected_k8_backpressure.get("decision_sha256")
        != sha256_file(root / CORRECTED_K8_DECISION_RELATIVE)
        or corrected_k8_backpressure.get("source_commitment_sha256")
        != corrected_k8_decision.get("source_commitment_sha256")
    ):
        raise ValueError("corrected-K8 backpressure recovery binding mismatch")
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "support_indices": list(EXPECTED_SUPPORTS),
        "support_ids": list(SUPPORT_IDS),
        "entity_count": len(entities),
        "expected_feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "expected_target_support_rows": EXPECTED_TARGET_SUPPORT_ROWS,
        "expected_target_available_rows": EXPECTED_TARGET_AVAILABLE_ROWS,
        "expected_target_missing_rows": EXPECTED_TARGET_MISSING_ROWS,
        "feature_file_count": len(features),
        "pdb_file_count": len(pdb_files),
        "script_relative_path": SCRIPT_RELATIVE,
        "script_sha256": sha256_file(script),
        "parent_commitment_relative_path": PARENT_RELATIVE,
        "parent_commitment_sha256": sha256_file(parent_path),
        "parent_receipt_relative_path": PARENT_RECEIPT_RELATIVE,
        "parent_receipt_sha256": sha256_file(parent_receipt_path),
        "bound_files": bound_files,
        "entities": entities,
        "features": features,
        "pdb_files": pdb_files,
    }
    payload["commitment_sha256"] = canonical_sha256(payload)
    return payload


def freeze(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    requested_output = Path(args.output)
    output = _path_under(
        root, requested_output if requested_output.is_absolute() else root / requested_output
    )
    if output != root / SOURCE_COMMITMENT_RELATIVE:
        raise ValueError("dynamic-distance source commitment path is not canonical")
    parent_path = _path_under(
        root, root / getattr(args, "parent_commitment", Path(PARENT_RELATIVE))
    )
    parent_receipt_path = _path_under(
        root, root / getattr(args, "parent_receipt", Path(PARENT_RECEIPT_RELATIVE))
    )
    if parent_path != root / PARENT_RELATIVE:
        raise ValueError("parent commitment path is not canonical")
    if parent_receipt_path != root / PARENT_RECEIPT_RELATIVE:
        raise ValueError("parent K32 receipt path is not canonical")
    if not parent_path.is_file() or not parent_receipt_path.is_file():
        raise FileNotFoundError("missing parent commitment or K32 receipt")
    payload = _commitment_payload(root, parent_path, parent_receipt_path)
    write_json_new(output, payload)
    print(
        json.dumps(
            {
                "commitment_sha256": sha256_file(output),
                "entity_count": payload["entity_count"],
                "pdb_file_count": payload["pdb_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_commitment_schema(root: Path, commitment: dict[str, Any]) -> None:
    if commitment.get("contract") != CONTRACT:
        raise ValueError("dynamic-distance commitment contract mismatch")
    for field in (
        "target_values_read",
        "source_gate_authorized",
        "formal_evaluation_authorized",
    ):
        if commitment.get(field) is not False:
            raise ValueError(f"dynamic-distance commitment scope mismatch: {field}")
    if tuple(int(value) for value in commitment.get("support_indices", ())) != EXPECTED_SUPPORTS:
        raise ValueError("dynamic-distance support roster mismatch")
    if tuple(str(value) for value in commitment.get("support_ids", ())) != SUPPORT_IDS:
        raise ValueError("dynamic-distance support ID roster mismatch")
    exact_counts = {
        "entity_count": EXPECTED_ENTITY_COUNT,
        "feature_file_count": EXPECTED_ENTITY_COUNT,
        "pdb_file_count": EXPECTED_ENTITY_COUNT * len(EXPECTED_SUPPORTS),
        "expected_feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "expected_target_support_rows": EXPECTED_TARGET_SUPPORT_ROWS,
        "expected_target_available_rows": EXPECTED_TARGET_AVAILABLE_ROWS,
        "expected_target_missing_rows": EXPECTED_TARGET_MISSING_ROWS,
    }
    for field, expected in exact_counts.items():
        if commitment.get(field) != expected:
            raise ValueError(f"dynamic-distance exact count mismatch: {field}")
    without_hash = dict(commitment)
    stored = without_hash.pop("commitment_sha256", None)
    if canonical_sha256(without_hash) != stored:
        raise ValueError("dynamic-distance commitment canonical hash mismatch")
    script = _script_path(root)
    if commitment.get("script_relative_path") != SCRIPT_RELATIVE:
        raise ValueError("dynamic-distance script path mismatch")
    if commitment.get("script_sha256") != sha256_file(script):
        raise ValueError("dynamic-distance script hash mismatch")
    if commitment.get("parent_commitment_relative_path") != PARENT_RELATIVE:
        raise ValueError("parent commitment path binding mismatch")
    if commitment.get("parent_receipt_relative_path") != PARENT_RECEIPT_RELATIVE:
        raise ValueError("parent K32 receipt path binding mismatch")
    expected_bound = []
    for relative in BOUND_PREREQUISITES:
        path = _path_under(root, root / relative)
        expected_bound.append({"relative_path": relative, "sha256": sha256_file(path)})
    if commitment.get("bound_files") != expected_bound:
        raise ValueError("dynamic-distance prerequisite binding mismatch")
    if commitment.get("entity_count") != len(commitment.get("entities", [])):
        raise ValueError("dynamic-distance entity count mismatch")
    if commitment.get("feature_file_count") != len(commitment.get("features", [])):
        raise ValueError("dynamic-distance feature count mismatch")
    if commitment.get("pdb_file_count") != len(commitment.get("pdb_files", [])):
        raise ValueError("dynamic-distance PDB count mismatch")


def verify_commitment(root: Path, path: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    path = Path(path).resolve()
    commitment = json.loads(path.read_text(encoding="utf-8"))
    _verify_commitment_schema(root, commitment)
    parent_path = _path_under(root, root / PARENT_RELATIVE)
    parent_receipt_path = _path_under(root, root / PARENT_RECEIPT_RELATIVE)
    expected = _commitment_payload(root, parent_path, parent_receipt_path)
    if commitment != expected:
        raise ValueError("dynamic-distance commitment replay mismatch")
    return commitment


def pdb_records(path: Path) -> dict[str, Any]:
    """Parse coordinates with the parent asset's TER/chain segment identity."""

    names: list[str] = []
    resnames: list[str] = []
    chains: list[str] = []
    seq_ids: list[int] = []
    insertions: list[str] = []
    elements: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    residue_keys: list[tuple[int, str, int, str]] = []
    residues: dict[tuple[int, str, int, str], dict[str, np.ndarray]] = {}
    residue_order: list[tuple[int, str, int, str]] = []
    raw_lines = path.read_bytes().decode("ascii", errors="replace").splitlines()
    segment = 0
    for line in raw_lines:
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        seq_id = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper() or next(
            (character for character in name if character.isalpha()), ""
        )
        coordinate = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        key = (segment, chain, seq_id, insertion)
        if key not in residues:
            residues[key] = {}
            residue_order.append(key)
        if name in residues[key]:
            raise ValueError(f"duplicate atom identity: {path}:{key}:{name}")
        residues[key][name] = coordinate
        names.append(name)
        resnames.append(resname)
        chains.append(chain)
        seq_ids.append(seq_id)
        insertions.append(insertion)
        elements.append(element)
        coordinates.append(tuple(float(value) for value in coordinate))
        residue_keys.append(key)
    array = np.asarray(coordinates, dtype=np.float64)
    if not names or not np.isfinite(array).all():
        raise ValueError(f"empty or nonfinite PDB: {path}")
    return {
        "path": path,
        "name": np.asarray(names, dtype=str),
        "resname": np.asarray(resnames, dtype=str),
        "chain": np.asarray(chains, dtype=str),
        "seq_id": np.asarray(seq_ids, dtype=np.int64),
        "insertion": np.asarray(insertions, dtype=str),
        "element": np.asarray(elements, dtype=str),
        "coordinate": array,
        "residue_key": tuple(residue_keys),
        "residues": residues,
        "residue_order": residue_order,
    }


def _record_residue_keys(records: dict[str, Any]) -> tuple[tuple[int, str, int, str], ...]:
    if "residue_key" in records:
        return tuple(records["residue_key"])
    return tuple(
        (0, str(chain).strip() or "_", int(seq_id), "")
        for chain, seq_id in zip(records["chain"], records["seq_id"], strict=True)
    )


def _chi_coordinate_velocities(records: dict[str, Any]) -> np.ndarray:
    """Return exact d(atom xyz)/d(chi), in radians, for chi1..chi4."""

    coordinates = np.asarray(records["coordinate"], dtype=np.float64)
    names = np.asarray(records["name"], dtype=str)
    resnames = np.asarray(records["resname"], dtype=str)
    keys = _record_residue_keys(records)
    velocities = np.zeros((len(coordinates), CHI_COUNT, 3), dtype=np.float64)
    by_residue: dict[tuple[int, str, int, str], list[int]] = {}
    for index, key in enumerate(keys):
        by_residue.setdefault(key, []).append(index)
    for key in sorted(by_residue):
        indices = np.asarray(by_residue[key], dtype=np.int64)
        adjacency = {int(index): set() for index in indices}
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                hydrogen = names[first].startswith("H") or names[second].startswith("H")
                # 1.50 Å includes S-H (about 1.34 Å) as well as C/N/O-H.
                cutoff = 1.50 if hydrogen else 1.95
                if np.linalg.norm(coordinates[first] - coordinates[second]) <= cutoff:
                    adjacency[int(first)].add(int(second))
                    adjacency[int(second)].add(int(first))
        resname = str(resnames[indices[0]])
        for dimension, (proximal_name, distal_name) in enumerate(
            SIDECHAIN_CHI_BONDS.get(resname, ())
        ):
            proximal = next(
                (int(index) for index in indices if names[index] == proximal_name), None
            )
            distal_axis = next(
                (int(index) for index in indices if names[index] == distal_name), None
            )
            if proximal is None or distal_axis is None:
                continue
            distal = {distal_axis}
            stack = [distal_axis]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {proximal, distal_axis}:
                        continue
                    if neighbor not in distal:
                        distal.add(neighbor)
                        stack.append(neighbor)
            if any(names[index] in {"N", "CA", "C", "O", "OXT"} for index in distal):
                continue
            axis = coordinates[distal_axis] - coordinates[proximal]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12:
                continue
            unit = axis / axis_norm
            rotated = np.asarray(sorted(distal), dtype=np.int64)
            velocities[rotated, dimension] = np.cross(
                unit, coordinates[rotated] - coordinates[proximal]
            )
    return velocities


def _target_rows(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"target_id", "seq_id", "comp_id", "atom_id"}
    if not required.issubset(targets.columns):
        raise ValueError(f"target identity columns missing: {sorted(required - set(targets))}")
    frame = targets.copy()
    frame["target_id"] = frame["target_id"].astype(str)
    frame["comp_id"] = frame["comp_id"].astype(str).str.strip().str.upper()
    frame["atom_id"] = frame["atom_id"].astype(str).str.strip().str.upper()
    numeric = pd.to_numeric(frame["seq_id"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("source feature seq_id is nonfinite")
    if not np.equal(numeric.to_numpy(float), np.floor(numeric.to_numpy(float))).all():
        raise ValueError("source feature seq_id is not integral")
    frame["seq_id"] = numeric.astype(np.int64)
    frame = frame.sort_values("target_id", kind="stable")
    first = frame.drop_duplicates("target_id", keep="first").copy()
    identity = ["seq_id", "comp_id", "atom_id"]
    inconsistent = frame.groupby("target_id", sort=False)[identity].nunique(dropna=False)
    bad = inconsistent.index[inconsistent.gt(1).any(axis=1)]
    if len(bad):
        raise ValueError(f"inconsistent source target identity: {bad[0]}")
    if first["target_id"].duplicated().any() or first.empty:
        raise ValueError("source feature target identity is empty or duplicated")
    return first.reset_index(drop=True)


def _support_dynamic_geometry(
    records: dict[str, Any], targets: pd.DataFrame
) -> dict[str, np.ndarray]:
    coordinates = np.asarray(records["coordinate"], dtype=np.float64)
    names = np.asarray(records["name"], dtype=str)
    resnames = np.asarray(records["resname"], dtype=str)
    seq_ids = np.asarray(records["seq_id"], dtype=np.int32)
    elements = np.asarray(records["element"], dtype=str)
    residue_keys = _record_residue_keys(records)
    velocities = _chi_coordinate_velocities(records)
    distances = np.full((len(targets), len(ELEMENTS)), DISTANCE_FILL, dtype=np.float32)
    self_jacobian = np.zeros((len(targets), len(ELEMENTS), CHI_COUNT), dtype=np.float32)
    neighbor_jacobian = np.zeros_like(self_jacobian)
    neighbor_seq_ids = np.full((len(targets), len(ELEMENTS)), -1, dtype=np.int32)
    target_atom_available = np.zeros(len(targets), dtype=bool)
    distance_available = np.zeros((len(targets), len(ELEMENTS)), dtype=bool)
    residue_codes = np.empty(len(residue_keys), dtype=np.int64)
    residue_code: dict[tuple[int, str, int, str], int] = {}
    identity_indices: dict[tuple[int, str, str], list[int]] = {}
    for index, residue_key in enumerate(residue_keys):
        residue_codes[index] = residue_code.setdefault(residue_key, len(residue_code))
        identity_indices.setdefault(
            (int(seq_ids[index]), str(resnames[index]), str(names[index])), []
        ).append(index)

    target_indices = np.full(len(targets), -1, dtype=np.int64)
    for row_number, row in enumerate(
        targets[["seq_id", "comp_id", "atom_id"]].itertuples(index=False)
    ):
        atom_name = "H" if str(row.atom_id).upper() == "HN" else str(row.atom_id).upper()
        candidates = identity_indices.get(
            (int(row.seq_id), str(row.comp_id).upper(), atom_name), []
        )
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous target atom identity: {row.seq_id}:{row.comp_id}:{atom_name}"
            )
        if candidates:
            target_indices[row_number] = candidates[0]
    target_rows = np.flatnonzero(target_indices >= 0)
    target_atom_available[target_rows] = True

    for descriptor, element in enumerate(ELEMENTS):
        indices = np.flatnonzero(elements == element)
        if not len(indices) or not len(target_rows):
            continue
        tree = cKDTree(coordinates[indices])
        query_count = min(32, len(indices))
        target_atoms = target_indices[target_rows]
        target_coordinates = coordinates[target_atoms]
        candidate_distances, local_indices = tree.query(
            target_coordinates, k=query_count
        )
        candidate_distances = np.asarray(candidate_distances, dtype=np.float64)
        local_indices = np.asarray(local_indices, dtype=np.int64)
        if query_count == 1:
            candidate_distances = candidate_distances[:, None]
            local_indices = local_indices[:, None]
        candidate_atoms = indices[local_indices]
        valid = residue_codes[candidate_atoms] != residue_codes[target_atoms, None]
        nearest_atoms = np.full(len(target_rows), -1, dtype=np.int64)
        nearest_distances = np.full(len(target_rows), np.inf, dtype=np.float64)
        has_neighbor = valid.any(axis=1)
        if has_neighbor.any():
            masked_distances = np.where(valid, candidate_distances, np.inf)
            best_distance = masked_distances.min(axis=1)
            tied = valid & (candidate_distances == best_distance[:, None])
            tied_atoms = np.where(tied, candidate_atoms, np.iinfo(np.int64).max)
            chosen = tied_atoms.min(axis=1)
            nearest_atoms[has_neighbor] = chosen[has_neighbor]
            nearest_distances[has_neighbor] = best_distance[has_neighbor]
            # Resolve the only bounded-query ambiguity: an exact nearest-distance
            # tie crossing the k=32 boundary.  This branch is rare but makes the
            # selected PDB atom independent of cKDTree's internal tie ordering.
            boundary_ties = has_neighbor & (
                candidate_distances[:, -1] == best_distance
            )
            for local_row in np.flatnonzero(boundary_ties):
                target_atom = int(target_atoms[local_row])
                radius = float(best_distance[local_row]) * (1.0 + 1.0e-12) + 1.0e-12
                local = tree.query_ball_point(target_coordinates[local_row], r=radius)
                eligible = indices[np.asarray(local, dtype=np.int64)]
                eligible = eligible[
                    residue_codes[eligible] != residue_codes[target_atom]
                ]
                exact = np.linalg.norm(
                    coordinates[eligible] - coordinates[target_atom], axis=1
                )
                best = float(exact.min())
                nearest_atoms[local_row] = int(eligible[exact == best].min())
                nearest_distances[local_row] = best

        # The bounded query is sufficient for ordinary residues. Fall back only
        # when all 32 candidates belong to the target residue.
        for local_row in np.flatnonzero(~has_neighbor):
            target_atom = int(target_atoms[local_row])
            eligible = indices[residue_codes[indices] != residue_codes[target_atom]]
            if not len(eligible):
                continue
            exact = np.linalg.norm(
                coordinates[eligible] - coordinates[target_atom], axis=1
            )
            best = float(exact.min())
            nearest_atoms[local_row] = int(eligible[exact == best].min())
            nearest_distances[local_row] = best

        resolved = np.flatnonzero(nearest_atoms >= 0)
        if not len(resolved):
            continue
        rows = target_rows[resolved]
        target_atoms = target_indices[rows]
        neighbor_atoms = nearest_atoms[resolved]
        distance = nearest_distances[resolved]
        distances[rows, descriptor] = distance.astype(np.float32)
        distance_available[rows, descriptor] = True
        neighbor_seq_ids[rows, descriptor] = seq_ids[neighbor_atoms]
        nonzero = distance > 1.0e-12
        rows = rows[nonzero]
        target_atoms = target_atoms[nonzero]
        neighbor_atoms = neighbor_atoms[nonzero]
        direction = (
            coordinates[target_atoms] - coordinates[neighbor_atoms]
        ) / distance[nonzero, None]
        self_jacobian[rows, descriptor] = np.einsum(
            "ncd,nd->nc", velocities[target_atoms], direction
        ).astype(np.float32)
        neighbor_jacobian[rows, descriptor] = -np.einsum(
            "ncd,nd->nc", velocities[neighbor_atoms], direction
        ).astype(np.float32)
    return {
        "distances": distances,
        "distance_self_jacobian": self_jacobian,
        "distance_neighbor_jacobian": neighbor_jacobian,
        "distance_neighbor_seq_ids": neighbor_seq_ids,
        "target_atom_available": target_atom_available,
        "distance_available": distance_available,
    }


def support_dynamic_geometry(path: Path, targets: pd.DataFrame) -> dict[str, np.ndarray]:
    return _support_dynamic_geometry(pdb_records(Path(path)), _target_rows(targets))


def support_interresidue_distance_jacobian(
    path: Path, targets: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper returning exact self/neighbor chi derivatives."""

    result = support_dynamic_geometry(path, targets)
    return (
        result["distance_self_jacobian"],
        result["distance_neighbor_jacobian"],
        result["distance_neighbor_seq_ids"],
    )


def _read_source_targets(path: Path, entity_uid: str) -> pd.DataFrame:
    # Explicit columns are important: this call cannot request or deserialize a
    # value column, even when an old feature file happens to contain one.
    columns = (
        "entity_uid",
        "target_id",
        "seq_id",
        "comp_id",
        "atom_id",
        "support_id",
    )
    frame = pd.read_parquet(path, columns=columns)
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"source feature identity columns missing: {path}")
    frame = frame.loc[:, list(columns)]
    frame = frame[frame["entity_uid"].astype(str).eq(entity_uid)].copy()
    if frame.duplicated(["target_id", "support_id"]).any():
        raise ValueError(f"duplicate source target/support identity: {path}")
    if set(frame["support_id"].astype(str).unique()) != set(EXPECTED_K8_SUPPORT_IDS):
        raise ValueError(f"source feature support roster mismatch: {path}")
    if not frame.groupby("target_id", sort=False)["support_id"].size().eq(8).all():
        raise ValueError(f"source feature target/support cardinality mismatch: {path}")
    return _target_rows(frame)


def _stable_ids_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _output_name(entity: dict[str, str], entities: list[dict[str, str]]) -> str:
    same_bmrb = sum(row["bmrb_id"] == entity["bmrb_id"] for row in entities)
    if same_bmrb == 1:
        return f"{entity['bmrb_id']}.npz"
    suffix = hashlib.sha256(entity["entity_uid"].encode()).hexdigest()[:16]
    return f"{entity['bmrb_id']}__{suffix}.npz"


def _write_entity_npz(
    output: Path,
    entity: dict[str, str],
    targets: pd.DataFrame,
    support_results: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    distances = np.stack([result["distances"] for result in support_results], axis=1)
    self_jacobian = np.stack(
        [result["distance_self_jacobian"] for result in support_results], axis=1
    )
    neighbor_jacobian = np.stack(
        [result["distance_neighbor_jacobian"] for result in support_results], axis=1
    )
    neighbor_seq_ids = np.stack(
        [result["distance_neighbor_seq_ids"] for result in support_results], axis=1
    )
    atom_available = np.stack(
        [result["target_atom_available"] for result in support_results], axis=1
    )
    distance_available = np.stack(
        [result["distance_available"] for result in support_results], axis=1
    )
    target_ids = np.asarray(targets["target_id"].astype(str).tolist(), dtype=str)
    support_ids = np.asarray(SUPPORT_IDS, dtype=str)
    if len(support_results) != len(SUPPORT_IDS):
        raise ValueError("dynamic-distance materialization requires exactly 32 supports")
    payload = {
        "entity_uid": np.asarray([entity["entity_uid"]], dtype=str),
        "bmrb_id": np.asarray([entity["bmrb_id"]], dtype=str),
        "target_ids": target_ids,
        "support_ids": support_ids,
        "support_indices": np.asarray(EXPECTED_SUPPORTS, dtype=np.int16),
        "seq_ids": targets["seq_id"].to_numpy(dtype=np.int32),
        "comp_ids": np.asarray(targets["comp_id"].astype(str).tolist(), dtype=str),
        "atom_ids": np.asarray(targets["atom_id"].astype(str).tolist(), dtype=str),
        "element_order": np.asarray(ELEMENTS, dtype=str),
        "nearest_interresidue_distances_angstrom": distances,
        "distance_self_jacobian": self_jacobian,
        "distance_neighbor_jacobian": neighbor_jacobian,
        "distance_neighbor_seq_ids": neighbor_seq_ids,
        "target_atom_available": atom_available,
        "distance_available": distance_available,
    }
    for name in (
        "nearest_interresidue_distances_angstrom",
        "distance_self_jacobian",
        "distance_neighbor_jacobian",
    ):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"nonfinite dynamic-distance array: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return {
        "entity_uid": entity["entity_uid"],
        "bmrb_id": entity["bmrb_id"],
        "output_relative_path": str(output),
        "output_sha256": sha256_file(output),
        "target_count": len(target_ids),
        "target_support_row_count": len(target_ids) * len(SUPPORT_IDS),
        "target_available_count": int(atom_available.sum()),
        "target_missing_count": int(atom_available.size - atom_available.sum()),
        "target_ids_sha256": _stable_ids_hash(target_ids),
        "support_ids": list(SUPPORT_IDS),
        "array_shapes": {
            "nearest_interresidue_distances_angstrom": list(distances.shape),
            "distance_self_jacobian": list(self_jacobian.shape),
            "distance_neighbor_jacobian": list(neighbor_jacobian.shape),
            "distance_neighbor_seq_ids": list(neighbor_seq_ids.shape),
            "target_atom_available": list(atom_available.shape),
            "distance_available": list(distance_available.shape),
        },
    }


def materialize(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    def resolve_argument(raw: Path) -> Path:
        return _path_under(root, raw if raw.is_absolute() else root / raw)

    source_commitment = resolve_argument(Path(args.source_commitment))
    output_root = resolve_argument(Path(args.output_root))
    receipt_path = resolve_argument(Path(args.receipt))
    if source_commitment != root / SOURCE_COMMITMENT_RELATIVE:
        raise ValueError("dynamic-distance source commitment path is not canonical")
    if output_root != root / OUTPUT_ROOT_RELATIVE:
        raise ValueError("dynamic-distance output root is not canonical")
    if receipt_path != root / RECEIPT_RELATIVE:
        raise ValueError("dynamic-distance receipt path is not canonical")
    partial_root = output_root.with_name(f"{output_root.name}.partial")
    partial_receipt = receipt_path.with_name(f"{receipt_path.name}.partial")
    if output_root.exists() or partial_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if receipt_path.exists() or partial_receipt.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    commitment = verify_commitment(root, source_commitment)
    entities = list(commitment["entities"])
    feature_paths = {
        row["bmrb_id"]: root / row["relative_path"] for row in commitment["features"]
    }
    pdb_paths = {
        (row["entity_uid"], int(row["support_index"])): root / row["relative_path"]
        for row in commitment["pdb_files"]
    }
    partial_root.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for entity in entities:
        targets = _read_source_targets(feature_paths[entity["bmrb_id"]], entity["entity_uid"])
        results = [
            _support_dynamic_geometry(
                pdb_records(pdb_paths[(entity["entity_uid"], support)]), targets
            )
            for support in EXPECTED_SUPPORTS
        ]
        output = _path_under(
            partial_root, partial_root / _output_name(entity, entities)
        )
        record = _write_entity_npz(output, entity, targets, results)
        record["output_relative_path"] = str(
            (output_root / output.name).relative_to(root)
        )
        outputs.append(record)
    target_count = sum(int(record["target_count"]) for record in outputs)
    target_support_row_count = sum(
        int(record["target_support_row_count"]) for record in outputs
    )
    target_available_count = sum(
        int(record["target_available_count"]) for record in outputs
    )
    target_missing_count = sum(int(record["target_missing_count"]) for record in outputs)
    exact_counts = {
        "entity_count": len(outputs),
        "target_count": target_count,
        "target_support_row_count": target_support_row_count,
        "target_available_count": target_available_count,
        "target_missing_count": target_missing_count,
    }
    expected_counts = {
        "entity_count": EXPECTED_ENTITY_COUNT,
        "target_count": EXPECTED_FEATURE_ROW_COUNT,
        "target_support_row_count": EXPECTED_TARGET_SUPPORT_ROWS,
        "target_available_count": EXPECTED_TARGET_AVAILABLE_ROWS,
        "target_missing_count": EXPECTED_TARGET_MISSING_ROWS,
    }
    if exact_counts != expected_counts:
        raise ValueError(
            f"dynamic-distance exact output count mismatch: {exact_counts} != {expected_counts}"
        )
    receipt = {
        "contract": RECEIPT_CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "source_commitment_sha256": sha256_file(source_commitment),
        "parent_receipt_sha256": commitment["parent_receipt_sha256"],
        "support_indices": list(EXPECTED_SUPPORTS),
        "support_ids": list(SUPPORT_IDS),
        "entity_count": len(outputs),
        "output_count": len(outputs),
        **exact_counts,
        "outputs": outputs,
    }
    write_json_new(partial_receipt, receipt)
    partial_root.rename(output_root)
    os.link(partial_receipt, receipt_path)
    partial_receipt.unlink()
    print(
        json.dumps(
            {
                "entity_count": len(outputs),
                "receipt_sha256": sha256_file(receipt_path),
                "source_commitment_sha256": receipt["source_commitment_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--parent-commitment", type=Path, default=Path(PARENT_RELATIVE))
    freeze_parser.add_argument("--parent-receipt", type=Path, default=Path(PARENT_RECEIPT_RELATIVE))
    materialize_parser = commands.add_parser("materialize")
    materialize_parser.add_argument("--root", type=Path, required=True)
    materialize_parser.add_argument("--source-commitment", type=Path, required=True)
    materialize_parser.add_argument(
        "--output-root", type=Path, default=Path(OUTPUT_ROOT_RELATIVE)
    )
    materialize_parser.add_argument("--receipt", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "freeze":
        return freeze(args)
    if args.command == "materialize":
        return materialize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
