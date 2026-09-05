#!/usr/bin/env python3
"""Independent, target-unread verification of the K=32 distance cache.

This checker intentionally does not import the cache materializer.  The source
feature read below requests only identity columns, and the coordinate replay is
implemented locally so that a passing receipt is evidence from an independent
implementation rather than a second call into the producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# These values are deliberately literals, not imports from the producer.
SOURCE_CONTRACT = "k32_dynamic_distance_cache_source_commitment_v1"
RECEIPT_CONTRACT = "k32_dynamic_distance_cache_receipt_v1"
SUMMARY_CONTRACT = "k32_dynamic_distance_cache_independent_check_v1"
PLAN_CONTRACT = "atypemu-k32-dynamic-distance-source-cache-plan-v1"
PARENT_RECEIPT_CONTRACT = "k32_complete_coordinate_support_asset_v1"
PARENT_SOURCE_CONTRACT = "k32_support_asset_source_commitment_v1"
SCRIPT_RELATIVE = "gpuopt/materialize_k32_dynamic_distance_cache.py"
PLAN_RELATIVE = "gpuopt/preunblind/k32_dynamic_distance_cache_plan_v1.json"
PARENT_RELATIVE = "data/all_atom_observer_v1/commitment.json"
PARENT_RECEIPT_RELATIVE = ".auto/runs/k32_complete_coordinate_supports_v4_receipt.json"
PARENT_SOURCE_RELATIVE = ".auto/staging/k32_source_commitment_v4.json"
PARENT_SOURCE_COMMITMENT_RELATIVE = PARENT_SOURCE_RELATIVE
PARENT_ASSET_RELATIVE = "data/k32_complete_coordinate_supports_v4"
FEATURE_ROOT_RELATIVE = "data/all_atom_observer_v1/features"
OUTPUT_ROOT_RELATIVE = "data/k32_dynamic_distance_cache_v1"
SOURCE_RELATIVE = ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
# Public aliases mirror the artifact names used by the producer without
# importing any producer module.
SOURCE_COMMITMENT_RELATIVE = SOURCE_RELATIVE
RECEIPT_RELATIVE = ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json"
CORRECTED_K8_DECISION_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/decision.json"
)
CORRECTED_K8_BACKPRESSURE_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "backpressure_recovery_receipt.json"
)
CORRECTED_K8_SOURCE_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "corrected_k8_source_commitment.json"
)
CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE = CORRECTED_K8_SOURCE_RELATIVE
CORRECTED_K8_CONSUMED_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
    "consumed_authorization.json"
)
CORRECTED_K8_CLAIM_RELATIVE = (
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/external_claim.json"
)
CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE = CORRECTED_K8_CONSUMED_RELATIVE
CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE = CORRECTED_K8_CLAIM_RELATIVE
BOUND_PREREQUISITES = (
    PLAN_RELATIVE,
    PARENT_SOURCE_RELATIVE,
    CORRECTED_K8_DECISION_RELATIVE,
    CORRECTED_K8_BACKPRESSURE_RELATIVE,
    CORRECTED_K8_SOURCE_RELATIVE,
    CORRECTED_K8_CONSUMED_RELATIVE,
    CORRECTED_K8_CLAIM_RELATIVE,
)
EXPECTED_ENTITY_COUNT = 135
EXPECTED_FEATURE_ROW_COUNT = 127_285
EXPECTED_TARGET_SUPPORT_ROWS = 4_073_120
EXPECTED_TARGET_AVAILABLE_ROWS = 4_072_968
EXPECTED_TARGET_MISSING_ROWS = 152
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
EXPECTED_SUPPORT_IDS = tuple(f"BioEmu_{value}" for value in EXPECTED_SUPPORTS)
SUPPORT_IDS = EXPECTED_SUPPORT_IDS
# The feature files retain only the historical K=8 source support columns.
SOURCE_SUPPORT_IDS = tuple(
    f"BioEmu_{value}" for value in (1, 126, 251, 376, 501, 626, 751, 876)
)
ELEMENTS = ("H", "C", "N", "O", "S")
CHI_NAMES = ("chi1", "chi2", "chi3", "chi4")
CHI_COUNT = 4
DISTANCE_FILL = 10.0
IDENTITY_COLUMNS = (
    "entity_uid",
    "target_id",
    "seq_id",
    "comp_id",
    "atom_id",
    "support_id",
)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _under(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes root: {candidate}")
    return resolved


def _canonical_arg(root: Path, value: Path, relative: str) -> Path:
    candidate = _under(root, value if value.is_absolute() else root / value)
    expected = root / relative
    if candidate != expected:
        raise ValueError(f"noncanonical artifact path: {relative}")
    return candidate


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _source_entities(parent: dict[str, Any]) -> list[dict[str, str]]:
    selected = []
    seen_entities: set[str] = set()
    seen_bmrb: set[str] = set()
    for raw in parent.get("entities", []):
        if raw.get("split") != "train" or raw.get("observer_fold") not in {"A", "B"}:
            continue
        entity_uid = str(raw.get("entity_uid", ""))
        bmrb_id = str(raw.get("bmrb_id", ""))
        if not entity_uid or not bmrb_id:
            raise ValueError("source entity lacks identity")
        if entity_uid in seen_entities or bmrb_id in seen_bmrb:
            raise ValueError("duplicate source entity or BMRB identity")
        seen_entities.add(entity_uid)
        seen_bmrb.add(bmrb_id)
        selected.append(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "split": "train",
                "observer_fold": str(raw["observer_fold"]),
            }
        )
    if len(selected) != EXPECTED_ENTITY_COUNT:
        raise ValueError(f"source entity count mismatch: {len(selected)}")
    return sorted(selected, key=lambda row: (row["entity_uid"], row["bmrb_id"]))


def _verify_flags(payload: dict[str, Any], label: str) -> None:
    for field in ("target_values_read", "source_gate_authorized", "formal_evaluation_authorized"):
        if payload.get(field) is not False:
            raise ValueError(f"{label} scope flag is not false: {field}")


def _verify_supports(payload: dict[str, Any], label: str) -> None:
    try:
        supports = tuple(int(value) for value in payload.get("support_indices", ()))
        support_ids = tuple(str(value) for value in payload.get("support_ids", ()))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} support roster is malformed") from error
    if supports != EXPECTED_SUPPORTS or support_ids != EXPECTED_SUPPORT_IDS:
        raise ValueError(f"{label} support roster mismatch")


def _verify_plan(root: Path, parent_receipt_path: Path) -> None:
    plan_path = _under(root, root / PLAN_RELATIVE)
    plan = _json(plan_path)
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError("dynamic-distance plan contract mismatch")
    scope = plan.get("scope", {})
    if (
        scope.get("entity_count") != EXPECTED_ENTITY_COUNT
        or scope.get("support_count") != 32
        or scope.get("target_identity_count") != EXPECTED_FEATURE_ROW_COUNT
        or scope.get("target_support_count") != EXPECTED_TARGET_SUPPORT_ROWS
    ):
        raise ValueError("dynamic-distance plan count mismatch")
    provenance = plan.get("provenance", {})
    if provenance.get("parent_k32_receipt_sha256") != sha256_file(parent_receipt_path):
        raise ValueError("dynamic-distance plan/parent receipt hash mismatch")
    for field, relative in (
        ("corrected_k8_decision_sha256", CORRECTED_K8_DECISION_RELATIVE),
        (
            "corrected_k8_backpressure_recovery_receipt_sha256",
            CORRECTED_K8_BACKPRESSURE_RELATIVE,
        ),
    ):
        if provenance.get(field) != sha256_file(_under(root, root / relative)):
            raise ValueError(f"dynamic-distance plan/{relative} hash mismatch")
    acceptance = plan.get("acceptance", {})
    for field in ("target_values_read", "source_gate_authorized", "formal_evaluation_authorized"):
        if acceptance.get(field) is not False:
            raise ValueError(f"dynamic-distance plan scope mismatch: {field}")


def _verify_source_commitment(root: Path, source_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    source = _json(source_path)
    if source.get("contract") != SOURCE_CONTRACT:
        raise ValueError("source commitment contract mismatch")
    _verify_flags(source, "source commitment")
    _verify_supports(source, "source commitment")
    exact_counts = {
        "entity_count": EXPECTED_ENTITY_COUNT,
        "feature_file_count": EXPECTED_ENTITY_COUNT,
        "pdb_file_count": EXPECTED_ENTITY_COUNT * 32,
        "expected_feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "expected_target_support_rows": EXPECTED_TARGET_SUPPORT_ROWS,
        "expected_target_available_rows": EXPECTED_TARGET_AVAILABLE_ROWS,
        "expected_target_missing_rows": EXPECTED_TARGET_MISSING_ROWS,
    }
    for field, expected in exact_counts.items():
        if source.get(field) != expected:
            raise ValueError(f"source commitment exact count mismatch: {field}")
    unhashed = dict(source)
    stored = unhashed.pop("commitment_sha256", None)
    if canonical_sha256(unhashed) != stored:
        raise ValueError("source commitment canonical hash mismatch")
    if source.get("script_relative_path") != SCRIPT_RELATIVE:
        raise ValueError("source commitment script path mismatch")
    script = _under(root, root / SCRIPT_RELATIVE)
    if not script.is_file() or source.get("script_sha256") != sha256_file(script):
        raise ValueError("source commitment script hash mismatch")
    if source.get("parent_commitment_relative_path") != PARENT_RELATIVE:
        raise ValueError("source parent commitment path mismatch")
    if source.get("parent_receipt_relative_path") != PARENT_RECEIPT_RELATIVE:
        raise ValueError("source parent receipt path mismatch")
    bound = [
        {"relative_path": relative, "sha256": sha256_file(_under(root, root / relative))}
        for relative in BOUND_PREREQUISITES
    ]
    if source.get("bound_files") != bound:
        raise ValueError("source commitment prerequisite hash/path mismatch")
    entities = source.get("entities")
    if not isinstance(entities, list) or len(entities) != EXPECTED_ENTITY_COUNT:
        raise ValueError("source commitment entity roster mismatch")
    expected_entities = sorted(
        [
            {
                "entity_uid": str(row.get("entity_uid", "")),
                "bmrb_id": str(row.get("bmrb_id", "")),
                "split": "train",
                "observer_fold": str(row.get("observer_fold", "")),
            }
            for row in entities
        ],
        key=lambda row: (row["entity_uid"], row["bmrb_id"]),
    )
    if len({row["entity_uid"] for row in expected_entities}) != EXPECTED_ENTITY_COUNT:
        raise ValueError("source commitment entity roster is not unique")
    if entities != expected_entities:
        raise ValueError("source commitment entity order or fields mismatch")
    by_bmrb = {row["bmrb_id"]: row for row in entities}
    if len(by_bmrb) != EXPECTED_ENTITY_COUNT:
        raise ValueError("source commitment BMRB roster is not unique")

    features = source.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_ENTITY_COUNT:
        raise ValueError("source commitment feature roster mismatch")
    expected_features = []
    for entity in sorted(entities, key=lambda row: row["bmrb_id"]):
        relative = f"{FEATURE_ROOT_RELATIVE}/{entity['bmrb_id']}.parquet"
        path = _under(root, root / relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing source feature parquet: {relative}")
        expected_features.append(
            {
                "bmrb_id": entity["bmrb_id"],
                "entity_uid": entity["entity_uid"],
                "relative_path": relative,
                "sha256": sha256_file(path),
                "row_count": None,
            }
        )
    for observed, expected in zip(features, expected_features, strict=True):
        if set(observed) != set(expected) or any(
            observed.get(key) != value for key, value in expected.items() if value is not None
        ):
            raise ValueError("source feature manifest path/hash mismatch")
        if not isinstance(observed.get("row_count"), int) or observed["row_count"] < 1:
            raise ValueError("source feature row-count metadata is malformed")
        expected["row_count"] = observed["row_count"]
    if features != expected_features:
        raise ValueError("source feature manifest order mismatch")

    pdb_files = source.get("pdb_files")
    expected_pdb_files = []
    for entity in entities:
        for support in EXPECTED_SUPPORTS:
            support_id = f"BioEmu_{support}"
            relative = (
                f"{PARENT_ASSET_RELATIVE}/{entity['bmrb_id']}/"
                f"{entity['bmrb_id']}_{support_id}.pdb"
            )
            path = _under(root, root / relative)
            if not path.is_file():
                raise FileNotFoundError(f"missing parent PDB: {relative}")
            expected_pdb_files.append(
                {
                    "entity_uid": entity["entity_uid"],
                    "bmrb_id": entity["bmrb_id"],
                    "support_id": support_id,
                    "support_index": support,
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                }
            )
    if not isinstance(pdb_files, list) or pdb_files != expected_pdb_files:
        raise ValueError("source parent PDB manifest mismatch")
    if len({(row["entity_uid"], row["support_index"]) for row in pdb_files}) != len(pdb_files):
        raise ValueError("source parent PDB roster is not unique")
    return source, entities


def _verify_parent_inputs(
    root: Path, source: dict[str, Any], entities: list[dict[str, str]]
) -> Path:
    parent_path = _under(root, root / PARENT_RELATIVE)
    parent_receipt_path = _under(root, root / PARENT_RECEIPT_RELATIVE)
    if source.get("parent_commitment_sha256") != sha256_file(parent_path):
        raise ValueError("source parent commitment hash mismatch")
    if source.get("parent_receipt_sha256") != sha256_file(parent_receipt_path):
        raise ValueError("source parent receipt hash mismatch")
    parent = _json(parent_path)
    if _source_entities(parent) != entities:
        raise ValueError("source entity roster differs from parent commitment")
    receipt = _json(parent_receipt_path)
    if receipt.get("contract") != PARENT_RECEIPT_CONTRACT:
        raise ValueError("parent receipt contract mismatch")
    _verify_flags(receipt, "parent receipt")
    parent_source_path = _under(root, root / PARENT_SOURCE_RELATIVE)
    parent_source = _json(parent_source_path)
    if parent_source.get("contract") != PARENT_SOURCE_CONTRACT:
        raise ValueError("parent source commitment contract mismatch")
    _verify_flags(parent_source, "parent source commitment")
    if tuple(parent_source.get("support_indices", ())) != EXPECTED_SUPPORTS:
        raise ValueError("parent source support roster mismatch")
    if parent_source.get("support_count") != 32 or parent_source.get("pair_count") != 5_184:
        raise ValueError("parent source cardinality mismatch")
    if receipt.get("source_commitment_sha256") != sha256_file(parent_source_path):
        raise ValueError("parent receipt/source commitment hash mismatch")

    expected = {
        (entity["entity_uid"], support)
        for entity in entities
        for support in EXPECTED_SUPPORTS
    }
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in receipt.get("outputs", []):
        entity_uid = str(raw.get("entity_uid", ""))
        try:
            support = int(raw.get("support_index", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("parent output support index is malformed") from error
        key = (entity_uid, support)
        if entity_uid not in {row["entity_uid"] for row in entities}:
            continue
        if key in observed:
            raise ValueError(f"duplicate parent output: {key}")
        entity = next(row for row in entities if row["entity_uid"] == entity_uid)
        support_id = f"BioEmu_{support}"
        relative = (
            f"{PARENT_ASSET_RELATIVE}/{entity['bmrb_id']}/"
            f"{entity['bmrb_id']}_{support_id}.pdb"
        )
        if support not in EXPECTED_SUPPORTS:
            raise ValueError(f"parent support roster mismatch: {key}")
        if (
            raw.get("bmrb_id") != entity["bmrb_id"]
            or raw.get("support_id") != support_id
            or raw.get("output_relative_path") != relative
        ):
            raise ValueError(f"parent output identity/path mismatch: {key}")
        path = _under(root, root / relative)
        if raw.get("output_sha256") != sha256_file(path):
            raise ValueError(f"parent output hash mismatch: {key}")
        observed[key] = raw
    if set(observed) != expected:
        raise ValueError("parent output roster is incomplete or has extras")
    _verify_plan(root, parent_receipt_path)
    return parent_receipt_path


def _target_rows(frame: pd.DataFrame, entity_uid: str) -> pd.DataFrame:
    if tuple(frame.columns) != IDENTITY_COLUMNS:
        raise ValueError("source feature read did not contain exactly identity columns")
    frame = frame[frame["entity_uid"].astype(str).eq(entity_uid)].copy()
    if frame.empty:
        raise ValueError(f"source entity has no feature rows: {entity_uid}")
    frame["target_id"] = frame["target_id"].astype(str)
    frame["comp_id"] = frame["comp_id"].astype(str).str.strip().str.upper()
    frame["atom_id"] = frame["atom_id"].astype(str).str.strip().str.upper()
    frame["entity_uid"] = frame["entity_uid"].astype(str)
    frame["support_id"] = frame["support_id"].astype(str)
    numeric = pd.to_numeric(frame["seq_id"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("source feature sequence identity is nonfinite")
    if not np.equal(numeric.to_numpy(float), np.floor(numeric.to_numpy(float))).all():
        raise ValueError("source feature sequence identity is not integral")
    frame["seq_id"] = numeric.astype(np.int32)
    if set(frame["support_id"]) != set(SOURCE_SUPPORT_IDS):
        raise ValueError(f"source feature support roster mismatch: {entity_uid}")
    if frame.duplicated(["target_id", "support_id"]).any():
        raise ValueError(f"duplicate source target/support identity: {entity_uid}")
    if not frame.groupby("target_id", sort=False)["support_id"].size().eq(8).all():
        raise ValueError(f"source target/support cardinality mismatch: {entity_uid}")
    ordered = frame.sort_values("target_id", kind="stable")
    identity = ["seq_id", "comp_id", "atom_id"]
    inconsistent = ordered.groupby("target_id", sort=False)[identity].nunique(dropna=False)
    if inconsistent.gt(1).to_numpy().any():
        raise ValueError(f"inconsistent source target identity: {entity_uid}")
    result = ordered.drop_duplicates("target_id", keep="first").reset_index(drop=True)
    if result["target_id"].duplicated().any() or result.empty:
        raise ValueError(f"source target identity is duplicated: {entity_uid}")
    return result.loc[:, list(IDENTITY_COLUMNS)]


def read_source_targets(path: Path, entity_uid: str) -> pd.DataFrame:
    """Read exactly the six non-value identity columns from a source feature file."""

    frame = pd.read_parquet(path, columns=IDENTITY_COLUMNS)
    if set(frame["entity_uid"].astype(str)) != {str(entity_uid)}:
        raise ValueError(f"source feature entity scope mismatch: {path}")
    if tuple(frame.columns) != IDENTITY_COLUMNS:
        # Some parquet engines preserve physical rather than requested order.
        if not set(IDENTITY_COLUMNS).issubset(frame.columns):
            raise ValueError(f"source identity columns missing: {path}")
        frame = frame.loc[:, list(IDENTITY_COLUMNS)]
    return _target_rows(frame, entity_uid)


def _stable_ids_hash(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(str(value).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _string_dtype(values: Any) -> np.dtype:
    return np.asarray(values, dtype=str).dtype


def _expect_array(arrays: Any, key: str, shape: tuple[int, ...], dtype: np.dtype | type, expected: Any = None) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"NPZ key missing: {key}")
    value = np.asarray(arrays[key])
    if value.shape != shape:
        raise ValueError(f"NPZ shape mismatch: {key}: {value.shape} != {shape}")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(f"NPZ dtype mismatch: {key}: {value.dtype} != {expected_dtype}")
    if expected is not None and not np.array_equal(value, np.asarray(expected, dtype=expected_dtype)):
        raise ValueError(f"NPZ identity mismatch: {key}")
    return value


def verify_npz(path: Path, entity: dict[str, str], targets: pd.DataFrame) -> dict[str, Any]:
    """Validate one cache file and return arrays/diagnostics without pickle support."""

    n = len(targets)
    target_ids = targets["target_id"].astype(str).to_numpy()
    seq_ids = targets["seq_id"].to_numpy(dtype=np.int32)
    comp_ids = targets["comp_id"].astype(str).to_numpy()
    atom_ids = targets["atom_id"].astype(str).to_numpy()
    expected_keys = {
        "entity_uid",
        "bmrb_id",
        "target_ids",
        "support_ids",
        "support_indices",
        "seq_ids",
        "comp_ids",
        "atom_ids",
        "element_order",
        "nearest_interresidue_distances_angstrom",
        "distance_self_jacobian",
        "distance_neighbor_jacobian",
        "distance_neighbor_seq_ids",
        "target_atom_available",
        "distance_available",
    }
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_keys:
                raise ValueError(f"NPZ key inventory mismatch: {path.name}")
            _expect_array(arrays, "entity_uid", (1,), _string_dtype([entity["entity_uid"]]), [entity["entity_uid"]])
            _expect_array(arrays, "bmrb_id", (1,), _string_dtype([entity["bmrb_id"]]), [entity["bmrb_id"]])
            _expect_array(arrays, "target_ids", (n,), _string_dtype(target_ids), target_ids)
            _expect_array(arrays, "support_ids", (32,), _string_dtype(EXPECTED_SUPPORT_IDS), EXPECTED_SUPPORT_IDS)
            _expect_array(arrays, "support_indices", (32,), np.int16, np.asarray(EXPECTED_SUPPORTS, dtype=np.int16))
            _expect_array(arrays, "seq_ids", (n,), np.int32, seq_ids)
            _expect_array(arrays, "comp_ids", (n,), _string_dtype(comp_ids), comp_ids)
            _expect_array(arrays, "atom_ids", (n,), _string_dtype(atom_ids), atom_ids)
            _expect_array(arrays, "element_order", (5,), _string_dtype(ELEMENTS), ELEMENTS)
            distances = _expect_array(arrays, "nearest_interresidue_distances_angstrom", (n, 32, 5), np.float32)
            self_jacobian = _expect_array(arrays, "distance_self_jacobian", (n, 32, 5, 4), np.float32)
            neighbor_jacobian = _expect_array(arrays, "distance_neighbor_jacobian", (n, 32, 5, 4), np.float32)
            neighbor_seq_ids = _expect_array(arrays, "distance_neighbor_seq_ids", (n, 32, 5), np.int32)
            atom_available = _expect_array(arrays, "target_atom_available", (n, 32), np.bool_)
            distance_available = _expect_array(arrays, "distance_available", (n, 32, 5), np.bool_)
            for key, value in {
                "nearest_interresidue_distances_angstrom": distances,
                "distance_self_jacobian": self_jacobian,
                "distance_neighbor_jacobian": neighbor_jacobian,
            }.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite NPZ array: {key}")
            if np.any(distance_available & ~atom_available[:, :, None]):
                raise ValueError("distance availability exceeds target atom availability")
            unavailable = ~distance_available
            if not np.array_equal(distances[unavailable], np.full(int(unavailable.sum()), DISTANCE_FILL, np.float32)):
                raise ValueError("unavailable distances are not filled with 10")
            if not np.all(distances[distance_available] > 0.0):
                raise ValueError("available distances are not positive")
            if not np.all(self_jacobian[unavailable] == 0.0) or not np.all(neighbor_jacobian[unavailable] == 0.0):
                raise ValueError("unavailable Jacobians are not zero")
            if not np.all(neighbor_seq_ids[unavailable] == -1):
                raise ValueError("unavailable neighbor sequence IDs are not -1")
            if np.any(~atom_available & distance_available.any(axis=2)):
                raise ValueError("missing target atom has an available distance")
            diagnostics = {
                "target_count": n,
                "target_support_row_count": n * 32,
                "target_available_count": int(atom_available.sum()),
                "target_missing_count": int(atom_available.size - atom_available.sum()),
                "target_ids_sha256": _stable_ids_hash(target_ids),
                "nonzero_jacobian_counts": {
                    chi: {
                        "self": int(np.count_nonzero(self_jacobian[:, :, :, index])),
                        "neighbor": int(np.count_nonzero(neighbor_jacobian[:, :, :, index])),
                    }
                    for index, chi in enumerate(CHI_NAMES)
                },
                "arrays": {
                    "nearest_interresidue_distances_angstrom": distances.copy(),
                    "distance_self_jacobian": self_jacobian.copy(),
                    "distance_neighbor_jacobian": neighbor_jacobian.copy(),
                    "distance_neighbor_seq_ids": neighbor_seq_ids.copy(),
                    "target_atom_available": atom_available.copy(),
                    "distance_available": distance_available.copy(),
                },
            }
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"unable to load NPZ with allow_pickle=False: {path}") from error
    return diagnostics


def _pdb_records(path: Path) -> dict[str, Any]:
    names: list[str] = []
    resnames: list[str] = []
    chains: list[str] = []
    seq_ids: list[int] = []
    elements: list[str] = []
    coordinates: list[np.ndarray] = []
    residue_keys: list[tuple[int, str, int, str]] = []
    residues: dict[tuple[int, str, int, str], dict[str, np.ndarray]] = {}
    segment = 0
    for line in path.read_bytes().decode("ascii", errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        seq_id = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper() or next((char for char in name if char.isalpha()), "")
        coordinate = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64
        )
        key = (segment, chain, seq_id, insertion)
        if name in residues.setdefault(key, {}):
            raise ValueError(f"duplicate atom identity: {path}:{key}:{name}")
        residues[key][name] = coordinate
        names.append(name)
        resnames.append(resname)
        chains.append(chain)
        seq_ids.append(seq_id)
        elements.append(element)
        coordinates.append(coordinate)
        residue_keys.append(key)
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    if not names or not np.isfinite(coordinate_array).all():
        raise ValueError(f"empty or nonfinite parent PDB: {path}")
    return {
        "name": np.asarray(names, dtype=str),
        "resname": np.asarray(resnames, dtype=str),
        "chain": np.asarray(chains, dtype=str),
        "seq_id": np.asarray(seq_ids, dtype=np.int32),
        "element": np.asarray(elements, dtype=str),
        "coordinate": coordinate_array,
        "residue_key": tuple(residue_keys),
        "residues": residues,
    }


def _velocities(records: dict[str, Any]) -> np.ndarray:
    coordinates = records["coordinate"]
    names = records["name"]
    resnames = records["resname"]
    keys = records["residue_key"]
    result = np.zeros((len(coordinates), CHI_COUNT, 3), dtype=np.float64)
    by_residue: dict[tuple[int, str, int, str], list[int]] = {}
    for index, key in enumerate(keys):
        by_residue.setdefault(key, []).append(index)
    for key in sorted(by_residue):
        indices = np.asarray(by_residue[key], dtype=np.int64)
        adjacency = {int(index): set() for index in indices}
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                cutoff = 1.50 if names[first].startswith("H") or names[second].startswith("H") else 1.95
                if np.linalg.norm(coordinates[first] - coordinates[second]) <= cutoff:
                    adjacency[int(first)].add(int(second))
                    adjacency[int(second)].add(int(first))
        for dimension, (proximal_name, distal_name) in enumerate(SIDECHAIN_CHI_BONDS.get(str(resnames[indices[0]]), ())):
            proximal = next((int(index) for index in indices if names[index] == proximal_name), None)
            distal_axis = next((int(index) for index in indices if names[index] == distal_name), None)
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
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-12:
                continue
            unit = axis / norm
            rotated = np.asarray(sorted(distal), dtype=np.int64)
            result[rotated, dimension] = np.cross(unit, coordinates[rotated] - coordinates[proximal])
    return result


def _replay_geometry(records: dict[str, Any], targets: pd.DataFrame) -> dict[str, np.ndarray]:
    coordinates = records["coordinate"]
    names = records["name"]
    resnames = records["resname"]
    seq_ids = records["seq_id"]
    elements = records["element"]
    residue_keys = records["residue_key"]
    velocity = _velocities(records)
    n = len(targets)
    distances = np.full((n, 5), DISTANCE_FILL, dtype=np.float32)
    self_jacobian = np.zeros((n, 5, 4), dtype=np.float32)
    neighbor_jacobian = np.zeros_like(self_jacobian)
    neighbor_seq_ids = np.full((n, 5), -1, dtype=np.int32)
    target_atom_available = np.zeros(n, dtype=bool)
    distance_available = np.zeros((n, 5), dtype=bool)
    residue_codes = np.empty(len(residue_keys), dtype=np.int64)
    code_by_key: dict[tuple[int, str, int, str], int] = {}
    identity_indices: dict[tuple[int, str, str], list[int]] = {}
    for index, key in enumerate(residue_keys):
        residue_codes[index] = code_by_key.setdefault(key, len(code_by_key))
        identity_indices.setdefault((int(seq_ids[index]), str(resnames[index]), str(names[index])), []).append(index)
    target_indices = np.full(n, -1, dtype=np.int64)
    for row_number, row in enumerate(targets[["seq_id", "comp_id", "atom_id"]].itertuples(index=False)):
        atom_name = "H" if str(row.atom_id).upper() == "HN" else str(row.atom_id).upper()
        candidates = identity_indices.get((int(row.seq_id), str(row.comp_id).upper(), atom_name), [])
        if len(candidates) > 1:
            raise ValueError(f"ambiguous target atom identity: {row.seq_id}:{row.comp_id}:{atom_name}")
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
        candidate_distances, local_indices = tree.query(coordinates[target_atoms], k=query_count)
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
            masked = np.where(valid, candidate_distances, np.inf)
            best = masked.min(axis=1)
            tied = valid & (candidate_distances == best[:, None])
            chosen = np.where(tied, candidate_atoms, np.iinfo(np.int64).max).min(axis=1)
            nearest_atoms[has_neighbor] = chosen[has_neighbor]
            nearest_distances[has_neighbor] = best[has_neighbor]
            boundary_ties = has_neighbor & (candidate_distances[:, -1] == best)
            for local_row in np.flatnonzero(boundary_ties):
                target_atom = int(target_atoms[local_row])
                radius = float(best[local_row]) * (1.0 + 1.0e-12) + 1.0e-12
                local = tree.query_ball_point(coordinates[target_atom], r=radius)
                eligible = indices[np.asarray(local, dtype=np.int64)]
                eligible = eligible[residue_codes[eligible] != residue_codes[target_atom]]
                exact = np.linalg.norm(coordinates[eligible] - coordinates[target_atom], axis=1)
                nearest_atoms[local_row] = int(eligible[exact == exact.min()].min())
                nearest_distances[local_row] = float(exact.min())
        for local_row in np.flatnonzero(~has_neighbor):
            target_atom = int(target_atoms[local_row])
            eligible = indices[residue_codes[indices] != residue_codes[target_atom]]
            if not len(eligible):
                continue
            exact = np.linalg.norm(coordinates[eligible] - coordinates[target_atom], axis=1)
            nearest_atoms[local_row] = int(eligible[exact == exact.min()].min())
            nearest_distances[local_row] = float(exact.min())
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
        direction = (coordinates[target_atoms] - coordinates[neighbor_atoms]) / distance[nonzero, None]
        self_jacobian[rows, descriptor] = np.einsum("ncd,nd->nc", velocity[target_atoms], direction).astype(np.float32)
        neighbor_jacobian[rows, descriptor] = -np.einsum("ncd,nd->nc", velocity[neighbor_atoms], direction).astype(np.float32)
    return {
        "nearest_interresidue_distances_angstrom": distances,
        "distance_self_jacobian": self_jacobian,
        "distance_neighbor_jacobian": neighbor_jacobian,
        "distance_neighbor_seq_ids": neighbor_seq_ids,
        "target_atom_available": target_atom_available,
        "distance_available": distance_available,
    }


def _sample_indices(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("cannot sample an empty target identity")
    return np.asarray(sorted({0, count // 2, count - 1}), dtype=np.int64)


def _replay_entity(
    root: Path,
    entity: dict[str, str],
    targets: pd.DataFrame,
    output_arrays: dict[str, np.ndarray],
    pdb_by_support: dict[int, Path],
) -> int:
    sample = _sample_indices(len(targets))
    sample_targets = targets.iloc[sample].reset_index(drop=True)
    for support_position, support in enumerate(EXPECTED_SUPPORTS):
        replayed = _replay_geometry(_pdb_records(pdb_by_support[support]), sample_targets)
        for key, observed in replayed.items():
            expected = output_arrays[key]
            actual = expected[sample, support_position]
            if not np.array_equal(actual, observed):
                raise ValueError(
                    f"independent geometry replay mismatch: {entity['entity_uid']}:{support}:{key}"
                )
    return int(len(sample) * len(EXPECTED_SUPPORTS))


def check(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    source_path = _canonical_arg(root, Path(args.source_commitment), SOURCE_RELATIVE)
    receipt_path = _canonical_arg(root, Path(args.receipt), RECEIPT_RELATIVE)
    output_root = _canonical_arg(
        root,
        Path(getattr(args, "output_root", OUTPUT_ROOT_RELATIVE)),
        OUTPUT_ROOT_RELATIVE,
    )
    if not output_root.is_dir():
        raise FileNotFoundError(f"missing canonical cache output root: {output_root}")
    source = _json(source_path)
    receipt = _json(receipt_path)
    if receipt.get("contract") != RECEIPT_CONTRACT:
        raise ValueError("dynamic-distance receipt contract mismatch")
    _verify_flags(receipt, "dynamic-distance receipt")
    _verify_supports(receipt, "dynamic-distance receipt")
    source, entities = _verify_source_commitment(root, source_path)
    parent_receipt_path = _verify_parent_inputs(root, source, entities)
    source_hash = sha256_file(source_path)
    if receipt.get("source_commitment_sha256") != source_hash:
        raise ValueError("receipt/source commitment hash mismatch")
    if receipt.get("parent_receipt_sha256") != sha256_file(parent_receipt_path):
        raise ValueError("receipt/parent receipt hash mismatch")
    exact_receipt_counts = {
        "entity_count": EXPECTED_ENTITY_COUNT,
        "output_count": EXPECTED_ENTITY_COUNT,
        "target_count": EXPECTED_FEATURE_ROW_COUNT,
        "target_support_row_count": EXPECTED_TARGET_SUPPORT_ROWS,
        "target_available_count": EXPECTED_TARGET_AVAILABLE_ROWS,
        "target_missing_count": EXPECTED_TARGET_MISSING_ROWS,
    }
    for field, expected in exact_receipt_counts.items():
        if receipt.get(field) != expected:
            raise ValueError(f"receipt exact count mismatch: {field}")

    feature_by_entity: dict[str, Path] = {}
    targets_by_entity: dict[str, pd.DataFrame] = {}
    for feature in source["features"]:
        entity_uid = str(feature["entity_uid"])
        path = _under(root, root / str(feature["relative_path"]))
        feature_by_entity[entity_uid] = path
        targets = read_source_targets(path, entity_uid)
        # The frozen feature manifest counts the eight K=8 support rows per
        # identity; the cache contains one identity row and expands it to K=32.
        if len(targets) * 8 != int(feature["row_count"]):
            raise ValueError(f"source feature row-count mismatch: {entity_uid}")
        targets_by_entity[entity_uid] = targets
    if set(feature_by_entity) != {entity["entity_uid"] for entity in entities}:
        raise ValueError("source feature/entity roster mismatch")

    pdb_by_identity = {
        (str(row["entity_uid"]), int(row["support_index"])): _under(root, root / str(row["relative_path"]))
        for row in source["pdb_files"]
    }
    output_rows = receipt.get("outputs")
    if not isinstance(output_rows, list) or len(output_rows) != EXPECTED_ENTITY_COUNT:
        raise ValueError("cache output roster length mismatch")
    by_entity: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        entity_uid = str(row.get("entity_uid", ""))
        if entity_uid in by_entity:
            raise ValueError(f"duplicate cache output entity: {entity_uid}")
        by_entity[entity_uid] = row
    if set(by_entity) != {entity["entity_uid"] for entity in entities}:
        raise ValueError("cache output entity roster mismatch")
    expected_names = set()
    for entity in entities:
        same_bmrb = sum(row["bmrb_id"] == entity["bmrb_id"] for row in entities)
        name = (
            f"{entity['bmrb_id']}.npz"
            if same_bmrb == 1
            else f"{entity['bmrb_id']}__{hashlib.sha256(entity['entity_uid'].encode()).hexdigest()[:16]}.npz"
        )
        expected_names.add(name)
    actual_entries = {
        str(path.relative_to(output_root)) for path in output_root.rglob("*")
    }
    if actual_entries != expected_names:
        raise ValueError(
            f"cache output file roster mismatch: "
            f"{sorted(actual_entries ^ expected_names)[:3]}"
        )

    total = {key: 0 for key in ("target_count", "target_support_row_count", "target_available_count", "target_missing_count")}
    replay_rows = 0
    nonzero: dict[str, dict[str, int]] = {
        chi: {"self": 0, "neighbor": 0} for chi in CHI_NAMES
    }
    for entity in entities:
        uid = entity["entity_uid"]
        row = by_entity[uid]
        targets = targets_by_entity[uid]
        n = len(targets)
        same_bmrb = sum(candidate["bmrb_id"] == entity["bmrb_id"] for candidate in entities)
        expected_relative = f"{OUTPUT_ROOT_RELATIVE}/" + (
            f"{entity['bmrb_id']}.npz"
            if same_bmrb == 1
            else f"{entity['bmrb_id']}__{hashlib.sha256(uid.encode()).hexdigest()[:16]}.npz"
        )
        if row.get("output_relative_path") != expected_relative:
            raise ValueError(f"noncanonical cache output path: {uid}")
        output_path = _under(root, root / expected_relative)
        if row.get("output_sha256") != sha256_file(output_path):
            raise ValueError(f"cache output hash mismatch: {uid}")
        expected_record = {
            "entity_uid": uid,
            "bmrb_id": entity["bmrb_id"],
            "target_count": n,
            "target_support_row_count": n * 32,
            "target_available_count": None,
            "target_missing_count": None,
            "target_ids_sha256": _stable_ids_hash(targets["target_id"].astype(str).to_numpy()),
            "support_ids": list(EXPECTED_SUPPORT_IDS),
            "array_shapes": {
                "nearest_interresidue_distances_angstrom": [n, 32, 5],
                "distance_self_jacobian": [n, 32, 5, 4],
                "distance_neighbor_jacobian": [n, 32, 5, 4],
                "distance_neighbor_seq_ids": [n, 32, 5],
                "target_atom_available": [n, 32],
                "distance_available": [n, 32, 5],
            },
        }
        observed = verify_npz(output_path, entity, targets)
        expected_record["target_available_count"] = observed["target_available_count"]
        expected_record["target_missing_count"] = observed["target_missing_count"]
        for key, value in expected_record.items():
            if row.get(key) != value:
                raise ValueError(f"cache output metadata mismatch: {uid}:{key}")
        total["target_count"] += n
        total["target_support_row_count"] += n * 32
        total["target_available_count"] += observed["target_available_count"]
        total["target_missing_count"] += observed["target_missing_count"]
        for chi in CHI_NAMES:
            for side in ("self", "neighbor"):
                nonzero[chi][side] += observed["nonzero_jacobian_counts"][chi][side]
        replay_pdbs = {
            support: pdb_by_identity[(uid, support)] for support in EXPECTED_SUPPORTS
        }
        replay_rows += _replay_entity(root, entity, targets, observed["arrays"], replay_pdbs)
    if total != {key: exact_receipt_counts[key] for key in total}:
        raise ValueError(f"cache aggregate count mismatch: {total}")
    summary = {
        "contract": SUMMARY_CONTRACT,
        "passed": True,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "entity_count": EXPECTED_ENTITY_COUNT,
        "output_count": EXPECTED_ENTITY_COUNT,
        "support_count": 32,
        "target_count": total["target_count"],
        "target_support_row_count": total["target_support_row_count"],
        "verified_target_support_rows": total["target_support_row_count"],
        "target_available_count": total["target_available_count"],
        "target_missing_count": total["target_missing_count"],
        "source_commitment_sha256": source_hash,
        "receipt_sha256": sha256_file(receipt_path),
        "parent_receipt_sha256": sha256_file(parent_receipt_path),
        "replay_sample_target_support_rows": replay_rows,
        "replay_sample_spans_every_entity": True,
        "replay_sample_spans_every_support": True,
        "per_chi_nonzero_jacobian_counts": nonzero,
        "jacobian_nonzero_counts": nonzero,
        "errors": [],
    }
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--source-commitment", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument(
        "--output-root", type=Path, default=Path(OUTPUT_ROOT_RELATIVE)
    )
    result.add_argument("--summary", "--decision", dest="summary", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        summary = check(args)
        code = 0
    except Exception as error:
        summary = {
            "contract": SUMMARY_CONTRACT,
            "passed": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "target_values_read": False,
            "source_gate_authorized": False,
            "formal_evaluation_authorized": False,
        }
        code = 1
    write_json_new(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    if code == 0:
        print("METRIC verified_target_support_rows=4073120")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
