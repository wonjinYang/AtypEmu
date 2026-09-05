#!/usr/bin/env python3
"""Fail-closed, target-unread checker for the nested support-count catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

STUDY_ID = "atypemu_nested_support_count_v1"
SHARD_CONTRACT = "atypemu_nested_support_count_v1_catalog_shard_v2"
RECEIPT_CONTRACT = "atypemu_nested_support_count_v1_catalog_receipt_v2"
SHARD_ARTIFACT_KIND = "target_unread_coordinate_catalog_shard_not_authorization"
RECEIPT_ARTIFACT_KIND = "target_unread_coordinate_catalog_receipt_not_authorization"
SHARD_COUNT = 27
ENTITY_COUNT = 135
SUPPORT_INDEXES = frozenset(range(1, 1001))
SOURCE_SUPPORT_INDEXES = (
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
SOURCE_SUPPORT_IDS = tuple(f"BioEmu_{index}" for index in SOURCE_SUPPORT_INDEXES)
SOURCE_COMMITMENT_RELATIVE_PATH = (
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
)
SOURCE_SCRIPT_RELATIVE_PATH = "gpuopt/materialize_k32_dynamic_distance_cache.py"
SOURCE_PARENT_COMMITMENT_RELATIVE_PATH = "data/all_atom_observer_v1/commitment.json"
SOURCE_PARENT_RECEIPT_RELATIVE_PATH = (
    ".auto/runs/k32_complete_coordinate_supports_v4_receipt.json"
)
SOURCE_BOUND_RELATIVE_PATHS = (
    "gpuopt/preunblind/k32_dynamic_distance_cache_plan_v1.json",
    ".auto/staging/k32_source_commitment_v4.json",
    ".auto/consolidation/corrected_k8_job135711_evidence_20260905/decision.json",
    (
        ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
        "backpressure_recovery_receipt.json"
    ),
    (
        ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
        "corrected_k8_source_commitment.json"
    ),
    (
        ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
        "consumed_authorization.json"
    ),
    (
        ".auto/consolidation/corrected_k8_job135711_evidence_20260905/"
        "external_claim.json"
    ),
)
EXPECTED_SUMMARY_RAW_SHA256 = (
    "737aaff560041ab7b64a750d011469a43563ab8a307af518eea3b98903f99b41"
)
EXPECTED_SHARD_ARCHIVE_RAW_SHA256 = (
    "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2"
)
EXPECTED_ROSTER_RAW_SHA256 = (
    "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
)
EXPECTED_SOURCE_COMMITMENT_RAW_SHA256 = (
    "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8"
)
EXPECTED_PRODUCER_RAW_SHA256 = (
    "4bd7c9463b34a9d6c268784fd13b45a01bf18c5e97905c0a5a4efd1cd7da0c28"
)
LEVELS = (32, 128, 768, 1536)
EXPECTED_ALL_ATOM_COUNT_DIAGNOSTICS = {
    "reference_topology_entity_count_ge_k": {
        "32": 130,
        "128": 121,
        "768": 108,
        "1536": 0,
    },
    "largest_single_topology_variant_entity_count_ge_k": {
        "32": 135,
        "128": 135,
        "768": 117,
        "1536": 0,
    },
}
BENIGN_ARCHIVE_MEMBERS = frozenset({"catalog_summary.json", "SHA256SUMS"})
MAX_JSON_MEMBER_BYTES = 64 * 1024 * 1024

SHARD_FIELDS = frozenset(
    {
        "artifact_kind",
        "authorization_consumed",
        "contract",
        "entities",
        "entity_count",
        "outer_or_formal_metrics_opened",
        "roster_sha256",
        "science_executed",
        "shard_count",
        "shard_index",
        "source_commitment_relative_path",
        "source_commitment_sha256",
        "source_scores_read",
        "study_id",
        "target_values_read",
    }
)
ROSTER_FIELDS = frozenset(
    {
        "artifact_kind",
        "authorization_consumed",
        "contract",
        "entities",
        "entity_count",
        "outer_or_formal_metrics_opened",
        "source_commitment_relative_path",
        "source_commitment_sha256",
        "source_scores_read",
        "study_id",
        "target_values_read",
    }
)
ROSTER_ENTITY_FIELDS = frozenset(
    {
        "bmrb_id",
        "canonical_all_atom_topology_sha256",
        "canonical_atom_count",
        "canonical_heavy_atom_count",
        "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256",
        "canonical_reference_relative_path",
        "canonical_reference_support_index",
        "entity_uid",
        "observer_fold",
        "split",
    }
)
SOURCE_COMMITMENT_FIELDS = frozenset(
    {
        "bound_files",
        "commitment_sha256",
        "contract",
        "entities",
        "entity_count",
        "expected_feature_row_count",
        "expected_target_available_rows",
        "expected_target_missing_rows",
        "expected_target_support_rows",
        "feature_file_count",
        "features",
        "formal_evaluation_authorized",
        "parent_commitment_relative_path",
        "parent_commitment_sha256",
        "parent_receipt_relative_path",
        "parent_receipt_sha256",
        "pdb_file_count",
        "pdb_files",
        "script_relative_path",
        "script_sha256",
        "source_gate_authorized",
        "support_ids",
        "support_indices",
        "target_values_read",
    }
)
SOURCE_ENTITY_FIELDS = frozenset({"bmrb_id", "entity_uid", "observer_fold", "split"})
SOURCE_FEATURE_FIELDS = frozenset(
    {"bmrb_id", "entity_uid", "relative_path", "row_count", "sha256"}
)
SOURCE_PDB_FIELDS = frozenset(
    {
        "bmrb_id",
        "entity_uid",
        "relative_path",
        "sha256",
        "support_id",
        "support_index",
    }
)
SOURCE_BOUND_FILE_FIELDS = frozenset({"relative_path", "sha256"})
ENTITY_FIELDS = frozenset(
    {
        "all_atom_topology_variant_count",
        "bmrb_id",
        "canonical_all_atom_topology_sha256",
        "canonical_atom_count",
        "canonical_filename_count",
        "canonical_heavy_atom_count",
        "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256",
        "canonical_reference_relative_path",
        "canonical_reference_support_index",
        "catalog_digest",
        "entity_uid",
        "files",
        "heavy_topology_compatible_unique_coordinate_count",
        "hydrogen_topology_variation_present",
        "invalid",
        "invalid_count",
        "missing_indices_1_to_1000",
        "observer_fold",
        "split",
        "unexpected_entries",
        "valid_index_digest",
    }
)
FILE_FIELDS = frozenset(
    {
        "all_atom_coordinate_sha256",
        "all_atom_topology_matches_reference",
        "all_atom_topology_sha256",
        "atom_count",
        "heavy_atom_coordinate_sha256",
        "heavy_atom_count",
        "heavy_atom_topology_sha256",
        "pdb_sha256",
        "support_index",
    }
)
SUMMARY_FIELDS = frozenset(
    {
        "artifact_kind",
        "authorization_consumed",
        "catalog_shard_count",
        "contract",
        "entities",
        "entity_count",
        "full_catalog_digest",
        "level_count_feasibility",
        "outer_or_formal_metrics_opened",
        "roster_sha256",
        "science_executed",
        "source_commitment_relative_path",
        "source_commitment_sha256",
        "source_method_composition",
        "source_scores_read",
        "study_id",
        "target_values_read",
    }
)
SUMMARY_ENTITY_FIELDS = (
    "bmrb_id",
    "all_atom_topology_variant_count",
    "canonical_filename_count",
    "catalog_digest",
    "entity_uid",
    "heavy_topology_compatible_unique_coordinate_count",
    "hydrogen_topology_variation_present",
    "invalid_count",
    "missing_indices_1_to_1000",
    "observer_fold",
    "split",
    "unexpected_entries",
    "valid_index_digest",
)
LEVEL_FIELDS = frozenset(
    {
        "all_entities_count_feasible",
        "entity_count_feasible",
        "maximum_entity_shortfall",
        "total_shortfall",
    }
)
SENTINEL_FIELDS = (
    "authorization_consumed",
    "outer_or_formal_metrics_opened",
    "science_executed",
    "source_scores_read",
    "target_values_read",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
ENTITY_UID_RE = re.compile(r"bmrb:([1-9][0-9]*):entity:([1-9][0-9]*)\Z")


class Checks:
    """Count only invariants that have successfully been checked."""

    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
        self.count += 1


def _is_int(value: object) -> bool:
    return type(value) is int


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _decode_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {label}")  # noqa: TRY004
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return _decode_json(path.read_bytes(), str(path))


def _require_fields(
    payload: dict[str, Any], expected: frozenset[str], label: str, checks: Checks
) -> None:
    checks.require(set(payload) == expected, f"{label} schema mismatch")


def _require_sha256(value: object, label: str, checks: Checks) -> str:
    checks.require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"invalid SHA-256: {label}",
    )
    return str(value)


def _require_positive_int(value: object, label: str, checks: Checks) -> int:
    checks.require(_is_int(value) and value > 0, f"positive integer required: {label}")
    return int(value)


def _require_nonnegative_int(value: object, label: str, checks: Checks) -> int:
    checks.require(
        _is_int(value) and value >= 0, f"nonnegative integer required: {label}"
    )
    return int(value)


def _require_safe_relative_path(value: object, label: str, checks: Checks) -> str:
    checks.require(isinstance(value, str) and value != "", f"invalid path: {label}")
    path = str(value)
    parts = path.split("/")
    checks.require(
        not path.startswith("/")
        and "\\" not in path
        and all(part not in {"", ".", ".."} for part in parts),
        f"unsafe relative path: {label}",
    )
    return path


def _require_false_fields(
    payload: dict[str, Any], fields: tuple[str, ...], label: str, checks: Checks
) -> None:
    for field in fields:
        checks.require(
            type(payload.get(field)) is bool and payload[field] is False,
            f"{label} sentinel is not false: {field}",
        )


def _require_false_sentinels(
    payload: dict[str, Any], label: str, checks: Checks
) -> None:
    _require_false_fields(payload, SENTINEL_FIELDS, label, checks)


def _sha256_file(path: Path, label: str) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"unable to read {label}: {error}") from error
    return digest.hexdigest()


def _require_raw_sha256(path: Path, expected: str, label: str, checks: Checks) -> str:
    actual = _sha256_file(path, label)
    checks.require(actual == expected, f"{label} raw SHA-256 mismatch")
    return actual


def _require_canonical_identity(
    row: dict[str, Any], label: str, checks: Checks
) -> tuple[str, str]:
    bmrb_id = row["bmrb_id"]
    entity_uid = row["entity_uid"]
    checks.require(
        isinstance(bmrb_id, str) and BMRB_RE.fullmatch(bmrb_id) is not None,
        f"noncanonical BMRB ID: {label}",
    )
    checks.require(
        isinstance(entity_uid, str) and ENTITY_UID_RE.fullmatch(entity_uid) is not None,
        f"noncanonical entity UID: {label}",
    )
    bmrb_match = BMRB_RE.fullmatch(str(bmrb_id))
    uid_match = ENTITY_UID_RE.fullmatch(str(entity_uid))
    assert bmrb_match is not None and uid_match is not None
    checks.require(
        uid_match.group(1) == bmrb_match.group(1) and uid_match.group(2) == "1",
        f"entity UID and BMRB ID differ: {label}",
    )
    return str(bmrb_id), str(entity_uid)


def _validate_roster(
    roster: dict[str, Any], checks: Checks
) -> tuple[dict[str, dict[str, Any]], tuple[str, str]]:
    _require_fields(roster, ROSTER_FIELDS, "entity roster", checks)
    checks.require(
        roster["artifact_kind"]
        == "target_unread_structural_entity_roster_not_authorization",
        "wrong entity roster artifact kind",
    )
    checks.require(
        roster["contract"] == "atypemu_nested_support_count_v1_entity_roster_v3",
        "wrong entity roster contract",
    )
    checks.require(roster["study_id"] == STUDY_ID, "wrong entity roster study ID")
    _require_false_fields(
        roster,
        (
            "authorization_consumed",
            "outer_or_formal_metrics_opened",
            "source_scores_read",
            "target_values_read",
        ),
        "entity roster",
        checks,
    )
    checks.require(
        _is_int(roster["entity_count"]) and roster["entity_count"] == ENTITY_COUNT,
        "entity roster count mismatch",
    )
    source_relative_path = _require_safe_relative_path(
        roster["source_commitment_relative_path"],
        "entity roster.source_commitment_relative_path",
        checks,
    )
    checks.require(
        source_relative_path == SOURCE_COMMITMENT_RELATIVE_PATH,
        "entity roster source commitment path mismatch",
    )
    source_sha256 = _require_sha256(
        roster["source_commitment_sha256"],
        "entity roster.source_commitment_sha256",
        checks,
    )
    checks.require(
        source_sha256 == EXPECTED_SOURCE_COMMITMENT_RAW_SHA256,
        "entity roster source commitment hash mismatch",
    )
    checks.require(
        isinstance(roster["entities"], list), "entity roster entities is not a list"
    )
    checks.require(
        len(roster["entities"]) == ENTITY_COUNT,
        "entity roster entity-list count mismatch",
    )

    by_uid: dict[str, dict[str, Any]] = {}
    bmrb_ids: set[str] = set()
    for position, raw_row in enumerate(roster["entities"]):
        label = f"entity roster.entities[{position}]"
        checks.require(
            isinstance(raw_row, dict), f"entity roster row is not an object: {label}"
        )
        row = raw_row
        _require_fields(row, ROSTER_ENTITY_FIELDS, label, checks)
        bmrb_id, entity_uid = _require_canonical_identity(row, label, checks)
        checks.require(
            row["split"] == "train", f"entity roster split is not train: {label}"
        )
        checks.require(
            row["observer_fold"] in {"A", "B"},
            f"invalid entity roster observer fold: {label}",
        )
        for field in (
            "canonical_all_atom_topology_sha256",
            "canonical_heavy_topology_sha256",
            "canonical_reference_pdb_sha256",
        ):
            _require_sha256(row[field], f"{label}.{field}", checks)
        atom_count = _require_positive_int(
            row["canonical_atom_count"], f"{label}.canonical_atom_count", checks
        )
        heavy_atom_count = _require_positive_int(
            row["canonical_heavy_atom_count"],
            f"{label}.canonical_heavy_atom_count",
            checks,
        )
        checks.require(
            heavy_atom_count <= atom_count,
            f"entity roster heavy atom count exceeds all-atom count: {label}",
        )
        checks.require(
            _is_int(row["canonical_reference_support_index"])
            and row["canonical_reference_support_index"] == 1,
            f"entity roster reference support index is not exactly 1: {label}",
        )
        expected_reference_path = (
            f"data/k32_complete_coordinate_supports_v4/{bmrb_id}/{bmrb_id}_BioEmu_1.pdb"
        )
        checks.require(
            _require_safe_relative_path(
                row["canonical_reference_relative_path"],
                f"{label}.canonical_reference_relative_path",
                checks,
            )
            == expected_reference_path,
            f"noncanonical entity roster reference path: {label}",
        )
        checks.require(
            entity_uid not in by_uid, f"duplicate entity roster UID: {label}"
        )
        checks.require(
            bmrb_id not in bmrb_ids, f"duplicate entity roster BMRB ID: {label}"
        )
        by_uid[entity_uid] = row
        bmrb_ids.add(bmrb_id)
    checks.require(
        len(by_uid) == ENTITY_COUNT and len(bmrb_ids) == ENTITY_COUNT,
        "duplicate or incomplete entity roster",
    )
    return by_uid, (source_relative_path, source_sha256)


def _validate_source_commitment(
    source: dict[str, Any], roster_by_uid: dict[str, dict[str, Any]], checks: Checks
) -> None:
    _require_fields(source, SOURCE_COMMITMENT_FIELDS, "source commitment", checks)
    checks.require(
        source["contract"] == "k32_dynamic_distance_cache_source_commitment_v1",
        "wrong source commitment contract",
    )
    for field in (
        "target_values_read",
        "source_gate_authorized",
        "formal_evaluation_authorized",
    ):
        checks.require(
            type(source[field]) is bool and source[field] is False,
            f"source commitment scope flag is not false: {field}",
        )
    checks.require(
        isinstance(source["support_indices"], list)
        and tuple(source["support_indices"]) == SOURCE_SUPPORT_INDEXES,
        "source commitment support-index roster mismatch",
    )
    checks.require(
        isinstance(source["support_ids"], list)
        and tuple(source["support_ids"]) == SOURCE_SUPPORT_IDS,
        "source commitment support-ID roster mismatch",
    )
    exact_counts = {
        "entity_count": ENTITY_COUNT,
        "feature_file_count": ENTITY_COUNT,
        "pdb_file_count": ENTITY_COUNT * len(SOURCE_SUPPORT_INDEXES),
        "expected_feature_row_count": 127_285,
        "expected_target_support_rows": 4_073_120,
        "expected_target_available_rows": 4_072_968,
        "expected_target_missing_rows": 152,
    }
    for field, expected in exact_counts.items():
        checks.require(
            _is_int(source[field]) and source[field] == expected,
            f"source commitment exact count mismatch: {field}",
        )
    _require_sha256(
        source["commitment_sha256"], "source commitment.commitment_sha256", checks
    )
    without_commitment_hash = dict(source)
    stored_commitment_hash = without_commitment_hash.pop("commitment_sha256")
    checks.require(
        stored_commitment_hash == _canonical_json_sha256(without_commitment_hash),
        "source commitment canonical hash mismatch",
    )
    checks.require(
        source["script_relative_path"] == SOURCE_SCRIPT_RELATIVE_PATH,
        "source commitment script path mismatch",
    )
    _require_sha256(source["script_sha256"], "source commitment.script_sha256", checks)
    checks.require(
        source["parent_commitment_relative_path"]
        == SOURCE_PARENT_COMMITMENT_RELATIVE_PATH,
        "source commitment parent commitment path mismatch",
    )
    _require_sha256(
        source["parent_commitment_sha256"],
        "source commitment.parent_commitment_sha256",
        checks,
    )
    checks.require(
        source["parent_receipt_relative_path"] == SOURCE_PARENT_RECEIPT_RELATIVE_PATH,
        "source commitment parent receipt path mismatch",
    )
    _require_sha256(
        source["parent_receipt_sha256"],
        "source commitment.parent_receipt_sha256",
        checks,
    )

    bound_files = source["bound_files"]
    checks.require(
        isinstance(bound_files, list), "source commitment bound files is not a list"
    )
    checks.require(
        len(bound_files) == len(SOURCE_BOUND_RELATIVE_PATHS),
        "source commitment bound-file count mismatch",
    )
    for position, (raw_bound, expected_path) in enumerate(
        zip(bound_files, SOURCE_BOUND_RELATIVE_PATHS)
    ):
        label = f"source commitment.bound_files[{position}]"
        checks.require(
            isinstance(raw_bound, dict), f"bound file is not an object: {label}"
        )
        _require_fields(raw_bound, SOURCE_BOUND_FILE_FIELDS, label, checks)
        checks.require(
            raw_bound["relative_path"] == expected_path,
            f"source commitment bound-file path mismatch: {label}",
        )
        _require_sha256(raw_bound["sha256"], f"{label}.sha256", checks)

    source_entities = source["entities"]
    checks.require(
        isinstance(source_entities, list), "source commitment entities is not a list"
    )
    checks.require(
        len(source_entities) == ENTITY_COUNT,
        "source commitment entity-list count mismatch",
    )
    source_by_uid: dict[str, dict[str, Any]] = {}
    for position, raw_entity in enumerate(source_entities):
        label = f"source commitment.entities[{position}]"
        checks.require(
            isinstance(raw_entity, dict),
            f"source commitment entity is not an object: {label}",
        )
        entity = raw_entity
        _require_fields(entity, SOURCE_ENTITY_FIELDS, label, checks)
        bmrb_id, entity_uid = _require_canonical_identity(entity, label, checks)
        checks.require(
            entity["split"] == "train", f"source entity split is not train: {label}"
        )
        checks.require(
            entity["observer_fold"] in {"A", "B"},
            f"invalid source entity observer fold: {label}",
        )
        checks.require(
            entity_uid not in source_by_uid, f"duplicate source entity UID: {label}"
        )
        source_by_uid[entity_uid] = entity
        roster_entity = roster_by_uid.get(entity_uid)
        checks.require(
            roster_entity is not None, f"source entity is absent from roster: {label}"
        )
        assert roster_entity is not None
        checks.require(
            (bmrb_id, entity_uid, entity["observer_fold"], entity["split"])
            == (
                roster_entity["bmrb_id"],
                roster_entity["entity_uid"],
                roster_entity["observer_fold"],
                roster_entity["split"],
            ),
            f"source entity identity/fold differs from roster: {label}",
        )
    checks.require(
        set(source_by_uid) == set(roster_by_uid),
        "source entity roster differs from entity roster",
    )

    features = source["features"]
    checks.require(
        isinstance(features, list), "source commitment features is not a list"
    )
    checks.require(
        len(features) == ENTITY_COUNT, "source commitment feature-list count mismatch"
    )
    feature_uids: set[str] = set()
    total_feature_rows = 0
    for position, raw_feature in enumerate(features):
        label = f"source commitment.features[{position}]"
        checks.require(
            isinstance(raw_feature, dict), f"source feature is not an object: {label}"
        )
        feature = raw_feature
        _require_fields(feature, SOURCE_FEATURE_FIELDS, label, checks)
        entity_uid = feature["entity_uid"]
        roster_entity = roster_by_uid.get(entity_uid)
        checks.require(
            roster_entity is not None, f"source feature is absent from roster: {label}"
        )
        assert roster_entity is not None
        checks.require(
            feature["bmrb_id"] == roster_entity["bmrb_id"],
            f"source feature BMRB differs from roster: {label}",
        )
        checks.require(
            feature["relative_path"]
            == f"data/all_atom_observer_v1/features/{roster_entity['bmrb_id']}.parquet",
            f"source feature path mismatch: {label}",
        )
        _require_sha256(feature["sha256"], f"{label}.sha256", checks)
        row_count = _require_positive_int(
            feature["row_count"], f"{label}.row_count", checks
        )
        checks.require(
            entity_uid not in feature_uids, f"duplicate source feature UID: {label}"
        )
        feature_uids.add(entity_uid)
        total_feature_rows += row_count
    checks.require(
        feature_uids == set(roster_by_uid),
        "source feature roster differs from entity roster",
    )
    checks.require(
        total_feature_rows == source["expected_feature_row_count"] * 8,
        "source feature row total mismatch",
    )

    pdb_files = source["pdb_files"]
    checks.require(
        isinstance(pdb_files, list), "source commitment PDB files is not a list"
    )
    checks.require(
        len(pdb_files) == ENTITY_COUNT * len(SOURCE_SUPPORT_INDEXES),
        "source commitment PDB-list count mismatch",
    )
    pdb_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for position, raw_pdb in enumerate(pdb_files):
        label = f"source commitment.pdb_files[{position}]"
        checks.require(
            isinstance(raw_pdb, dict), f"source PDB record is not an object: {label}"
        )
        pdb = raw_pdb
        _require_fields(pdb, SOURCE_PDB_FIELDS, label, checks)
        entity_uid = pdb["entity_uid"]
        roster_entity = roster_by_uid.get(entity_uid)
        checks.require(
            roster_entity is not None,
            f"source PDB entity is absent from roster: {label}",
        )
        assert roster_entity is not None
        support_index = _require_positive_int(
            pdb["support_index"], f"{label}.support_index", checks
        )
        checks.require(
            support_index in SOURCE_SUPPORT_INDEXES,
            f"source PDB support index mismatch: {label}",
        )
        checks.require(
            pdb["bmrb_id"] == roster_entity["bmrb_id"],
            f"source PDB BMRB differs from roster: {label}",
        )
        checks.require(
            pdb["support_id"] == f"BioEmu_{support_index}",
            f"source PDB support ID mismatch: {label}",
        )
        expected_path = (
            f"data/k32_complete_coordinate_supports_v4/{roster_entity['bmrb_id']}/"
            f"{roster_entity['bmrb_id']}_BioEmu_{support_index}.pdb"
        )
        checks.require(
            pdb["relative_path"] == expected_path,
            f"source PDB path mismatch: {label}",
        )
        _require_sha256(pdb["sha256"], f"{label}.sha256", checks)
        key = (entity_uid, support_index)
        checks.require(
            key not in pdb_by_identity, f"duplicate source PDB record: {label}"
        )
        pdb_by_identity[key] = pdb
    expected_pdb_keys = {
        (entity_uid, support_index)
        for entity_uid in roster_by_uid
        for support_index in SOURCE_SUPPORT_INDEXES
    }
    checks.require(
        set(pdb_by_identity) == expected_pdb_keys,
        "source PDB roster does not contain exactly 4,320 records",
    )
    for entity_uid, roster_entity in roster_by_uid.items():
        reference_key = (entity_uid, roster_entity["canonical_reference_support_index"])
        reference = pdb_by_identity[reference_key]
        checks.require(
            reference["relative_path"]
            == roster_entity["canonical_reference_relative_path"],
            f"source PDB reference path differs from roster: {entity_uid}",
        )
        checks.require(
            reference["sha256"] == roster_entity["canonical_reference_pdb_sha256"],
            f"source PDB reference hash differs from roster: {entity_uid}",
        )


def _require_evidence_bindings(
    observed: tuple[str, str, str],
    expected: tuple[str, str, str],
    label: str,
    checks: Checks,
) -> None:
    checks.require(observed == expected, f"{label} source/roster binding mismatch")


def _summary_projection(entity: dict[str, Any]) -> dict[str, Any]:
    return {field: entity[field] for field in SUMMARY_ENTITY_FIELDS}


def _validate_file(
    record: object,
    entity: dict[str, Any],
    label: str,
    checks: Checks,
) -> int:
    checks.require(isinstance(record, dict), f"file record is not an object: {label}")
    file_record = record
    _require_fields(file_record, FILE_FIELDS, label, checks)
    support_index = _require_positive_int(
        file_record["support_index"], f"{label}.support_index", checks
    )
    checks.require(
        support_index in SUPPORT_INDEXES,
        f"support index outside 1..1000: {label}",
    )
    atom_count = _require_positive_int(
        file_record["atom_count"], f"{label}.atom_count", checks
    )
    heavy_atom_count = _require_positive_int(
        file_record["heavy_atom_count"], f"{label}.heavy_atom_count", checks
    )
    checks.require(
        heavy_atom_count <= atom_count, f"heavy atom count exceeds atom count: {label}"
    )
    checks.require(
        type(file_record["all_atom_topology_matches_reference"]) is bool,
        f"all-atom topology flag is not boolean: {label}",
    )
    for field in (
        "all_atom_coordinate_sha256",
        "all_atom_topology_sha256",
        "heavy_atom_coordinate_sha256",
        "heavy_atom_topology_sha256",
        "pdb_sha256",
    ):
        _require_sha256(file_record[field], f"{label}.{field}", checks)
    checks.require(
        file_record["heavy_atom_topology_sha256"]
        == entity["canonical_heavy_topology_sha256"],
        f"heavy topology differs from canonical topology: {label}",
    )
    expected_matches_reference = (
        file_record["all_atom_topology_sha256"]
        == entity["canonical_all_atom_topology_sha256"]
    )
    checks.require(
        file_record["all_atom_topology_matches_reference"]
        is expected_matches_reference,
        f"all-atom topology reference flag disagrees with digest: {label}",
    )
    return support_index


def _validate_entity(entity: object, label: str, checks: Checks) -> dict[str, Any]:
    checks.require(isinstance(entity, dict), f"entity is not an object: {label}")
    row = entity
    _require_fields(row, ENTITY_FIELDS, label, checks)

    bmrb_id = row["bmrb_id"]
    entity_uid = row["entity_uid"]
    checks.require(
        isinstance(bmrb_id, str) and BMRB_RE.fullmatch(bmrb_id) is not None,
        f"noncanonical BMRB ID: {label}",
    )
    checks.require(
        isinstance(entity_uid, str) and ENTITY_UID_RE.fullmatch(entity_uid) is not None,
        f"noncanonical entity UID: {label}",
    )
    uid_match = ENTITY_UID_RE.fullmatch(str(entity_uid))
    bmrb_match = BMRB_RE.fullmatch(str(bmrb_id))
    assert uid_match is not None and bmrb_match is not None
    checks.require(
        uid_match.group(1) == bmrb_match.group(1),
        f"entity UID and BMRB ID differ: {label}",
    )
    checks.require(row["split"] == "train", f"entity split is not train: {label}")
    checks.require(
        row["observer_fold"] in {"A", "B"}, f"invalid observer fold: {label}"
    )

    for field in (
        "canonical_all_atom_topology_sha256",
        "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256",
        "catalog_digest",
        "valid_index_digest",
    ):
        _require_sha256(row[field], f"{label}.{field}", checks)
    canonical_atom_count = _require_positive_int(
        row["canonical_atom_count"], f"{label}.canonical_atom_count", checks
    )
    canonical_heavy_atom_count = _require_positive_int(
        row["canonical_heavy_atom_count"],
        f"{label}.canonical_heavy_atom_count",
        checks,
    )
    checks.require(
        canonical_heavy_atom_count <= canonical_atom_count,
        f"canonical heavy atom count exceeds all-atom count: {label}",
    )
    canonical_filename_count = _require_nonnegative_int(
        row["canonical_filename_count"],
        f"{label}.canonical_filename_count",
        checks,
    )
    invalid_count = _require_nonnegative_int(
        row["invalid_count"], f"{label}.invalid_count", checks
    )
    variant_count = _require_positive_int(
        row["all_atom_topology_variant_count"],
        f"{label}.all_atom_topology_variant_count",
        checks,
    )
    compatible_count = _require_nonnegative_int(
        row["heavy_topology_compatible_unique_coordinate_count"],
        f"{label}.heavy_topology_compatible_unique_coordinate_count",
        checks,
    )
    checks.require(
        type(row["hydrogen_topology_variation_present"]) is bool,
        f"hydrogen topology variation flag is not boolean: {label}",
    )
    checks.require(isinstance(row["files"], list), f"files is not a list: {label}")
    checks.require(isinstance(row["invalid"], list), f"invalid is not a list: {label}")
    checks.require(
        invalid_count == len(row["invalid"]), f"invalid count mismatch: {label}"
    )
    checks.require(
        isinstance(row["unexpected_entries"], list) and row["unexpected_entries"] == [],
        f"unexpected entries are present: {label}",
    )
    checks.require(
        canonical_filename_count == len(row["files"]) + invalid_count,
        f"canonical filename count mismatch: {label}",
    )

    reference_index = _require_positive_int(
        row["canonical_reference_support_index"],
        f"{label}.canonical_reference_support_index",
        checks,
    )
    checks.require(
        reference_index in SUPPORT_INDEXES,
        f"reference support index outside 1..1000: {label}",
    )
    expected_reference_path = (
        f"data/k32_complete_coordinate_supports_v4/{bmrb_id}/"
        f"{bmrb_id}_BioEmu_{reference_index}.pdb"
    )
    checks.require(
        _require_safe_relative_path(
            row["canonical_reference_relative_path"],
            f"{label}.canonical_reference_relative_path",
            checks,
        )
        == expected_reference_path,
        f"noncanonical reference path: {label}",
    )

    support_indexes: list[int] = []
    by_index: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(row["files"]):
        support_index = _validate_file(
            record, row, f"{label}.files[{position}]", checks
        )
        support_indexes.append(support_index)
        by_index[support_index] = record
    checks.require(
        support_indexes == sorted(support_indexes),
        f"files are not support-index sorted: {label}",
    )
    checks.require(
        len(by_index) == len(support_indexes), f"duplicate support index: {label}"
    )
    checks.require(reference_index in by_index, f"reference support is absent: {label}")
    reference = by_index[reference_index]
    checks.require(
        reference["all_atom_topology_sha256"]
        == row["canonical_all_atom_topology_sha256"],
        f"canonical all-atom topology does not match reference: {label}",
    )
    checks.require(
        reference["heavy_atom_topology_sha256"]
        == row["canonical_heavy_topology_sha256"],
        f"canonical heavy topology does not match reference: {label}",
    )
    checks.require(
        reference["pdb_sha256"] == row["canonical_reference_pdb_sha256"],
        f"canonical reference PDB digest mismatch: {label}",
    )
    checks.require(
        reference["atom_count"] == canonical_atom_count,
        f"canonical all-atom count mismatch: {label}",
    )
    checks.require(
        reference["heavy_atom_count"] == canonical_heavy_atom_count,
        f"canonical heavy-atom count mismatch: {label}",
    )

    invalid_indexes: list[int] = []
    for position, invalid in enumerate(row["invalid"]):
        invalid_label = f"{label}.invalid[{position}]"
        checks.require(
            isinstance(invalid, dict)
            and set(invalid) == {"error", "pdb_sha256", "support_index"},
            f"invalid-record schema mismatch: {invalid_label}",
        )
        checks.require(
            isinstance(invalid["error"], str) and bool(invalid["error"]),
            f"empty invalid-record error: {invalid_label}",
        )
        _require_sha256(invalid["pdb_sha256"], f"{invalid_label}.pdb_sha256", checks)
        invalid_index = _require_positive_int(
            invalid["support_index"], f"{invalid_label}.support_index", checks
        )
        checks.require(
            invalid_index in SUPPORT_INDEXES,
            f"invalid-record support index outside 1..1000: {invalid_label}",
        )
        invalid_indexes.append(invalid_index)
    observed_indexes = support_indexes + invalid_indexes
    checks.require(
        len(set(observed_indexes)) == len(observed_indexes),
        f"support index repeated across valid/invalid records: {label}",
    )
    expected_missing = sorted(SUPPORT_INDEXES - set(observed_indexes))
    missing = row["missing_indices_1_to_1000"]
    checks.require(
        isinstance(missing, list) and all(_is_int(value) for value in missing),
        f"missing indices are not integer list: {label}",
    )
    checks.require(
        missing == expected_missing, f"missing-index complement mismatch: {label}"
    )
    checks.require(
        row["catalog_digest"] == _canonical_json_sha256(row["files"]),
        f"catalog digest mismatch: {label}",
    )
    checks.require(
        row["valid_index_digest"] == _canonical_json_sha256(support_indexes),
        f"valid-index digest mismatch: {label}",
    )
    topology_variants = {record["all_atom_topology_sha256"] for record in row["files"]}
    checks.require(
        variant_count == len(topology_variants),
        f"all-atom topology variant count mismatch: {label}",
    )
    checks.require(
        row["hydrogen_topology_variation_present"] == (variant_count > 1),
        f"hydrogen topology variation flag mismatch: {label}",
    )
    compatible_coordinates = {
        record["heavy_atom_coordinate_sha256"]
        for record in row["files"]
        if record["heavy_atom_topology_sha256"]
        == row["canonical_heavy_topology_sha256"]
    }
    checks.require(
        compatible_count == len(compatible_coordinates),
        f"heavy-topology compatible coordinate count mismatch: {label}",
    )
    return row


def _require_catalog_entity_matches_roster(
    entity: dict[str, Any],
    roster_by_uid: dict[str, dict[str, Any]],
    label: str,
    checks: Checks,
) -> None:
    roster_entity = roster_by_uid.get(entity["entity_uid"])
    checks.require(
        roster_entity is not None,
        f"catalog entity is absent from entity roster: {label}",
    )
    assert roster_entity is not None
    for field in ROSTER_ENTITY_FIELDS:
        checks.require(
            entity[field] == roster_entity[field],
            f"catalog entity differs from entity roster: {label}.{field}",
        )


def _read_shards(archive_path: Path, checks: Checks) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    benign_members: set[str] = set()
    shard_members: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                normalized_name = member.name.removeprefix("./")
                if member.isdir():
                    checks.require(
                        normalized_name in {"", "."},
                        f"forbidden archive directory: {member.name}",
                    )
                    continue
                checks.require(
                    member.isfile(), f"forbidden archive member: {member.name}"
                )
                if normalized_name in BENIGN_ARCHIVE_MEMBERS:
                    checks.require(
                        normalized_name not in benign_members,
                        f"duplicate benign archive member: {member.name}",
                    )
                    benign_members.add(normalized_name)
                    continue
                checks.require(
                    re.fullmatch(
                        r"shard_(?:[0-9]|1[0-9]|2[0-6])\.json", normalized_name
                    )
                    is not None,
                    f"forbidden archive member name: {member.name}",
                )
                checks.require(
                    normalized_name not in shard_members,
                    f"duplicate shard archive member: {member.name}",
                )
                shard_members.add(normalized_name)
                checks.require(
                    member.size <= MAX_JSON_MEMBER_BYTES,
                    f"oversized JSON shard: {member.name}",
                )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unable to read archive member: {member.name}")
                with handle:
                    shards.append(_decode_json(handle.read(), f"archive:{member.name}"))
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"invalid shard archive: {error}") from error
    checks.require(
        len(shards) == SHARD_COUNT, "archive does not contain exactly 27 shards"
    )
    return shards


def _validate_shards(
    shards: list[dict[str, Any]],
    roster_by_uid: dict[str, dict[str, Any]],
    expected_bindings: tuple[str, str, str],
    checks: Checks,
) -> list[dict[str, Any]]:
    all_entities: list[dict[str, Any]] = []
    indexes: set[int] = set()
    for position, shard in enumerate(shards):
        label = f"shard[{position}]"
        _require_fields(shard, SHARD_FIELDS, label, checks)
        checks.require(
            shard["contract"] == SHARD_CONTRACT, f"wrong shard contract: {label}"
        )
        checks.require(
            shard["artifact_kind"] == SHARD_ARTIFACT_KIND,
            f"wrong shard artifact kind: {label}",
        )
        checks.require(shard["study_id"] == STUDY_ID, f"wrong study ID: {label}")
        _require_false_sentinels(shard, label, checks)
        checks.require(
            _is_int(shard["shard_count"]) and shard["shard_count"] == SHARD_COUNT,
            f"wrong shard count: {label}",
        )
        checks.require(
            _is_int(shard["shard_index"]) and 0 <= shard["shard_index"] < SHARD_COUNT,
            f"invalid shard index: {label}",
        )
        indexes.add(shard["shard_index"])
        shard_bindings = (
            _require_sha256(shard["roster_sha256"], f"{label}.roster_sha256", checks),
            _require_safe_relative_path(
                shard["source_commitment_relative_path"],
                f"{label}.source_commitment_relative_path",
                checks,
            ),
            _require_sha256(
                shard["source_commitment_sha256"],
                f"{label}.source_commitment_sha256",
                checks,
            ),
        )
        _require_evidence_bindings(shard_bindings, expected_bindings, label, checks)
        checks.require(
            isinstance(shard["entities"], list), f"entities is not a list: {label}"
        )
        checks.require(
            _is_int(shard["entity_count"])
            and shard["entity_count"] == len(shard["entities"]),
            f"entity count mismatch: {label}",
        )
        for entity_position, entity in enumerate(shard["entities"]):
            entity_label = f"{label}.entities[{entity_position}]"
            catalog_entity = _validate_entity(entity, entity_label, checks)
            _require_catalog_entity_matches_roster(
                catalog_entity, roster_by_uid, entity_label, checks
            )
            all_entities.append(catalog_entity)
    checks.require(
        indexes == set(range(SHARD_COUNT)), "shard indexes are not exactly 0..26"
    )
    return all_entities


def _expected_level_count_feasibility(
    entities: list[dict[str, Any]],
) -> dict[str, dict[str, int | bool]]:
    result: dict[str, dict[str, int | bool]] = {}
    for level in LEVELS:
        shortfalls = [
            max(0, level - entity["heavy_topology_compatible_unique_coordinate_count"])
            for entity in entities
        ]
        result[str(level)] = {
            "all_entities_count_feasible": all(
                shortfall == 0 for shortfall in shortfalls
            ),
            "entity_count_feasible": sum(shortfall == 0 for shortfall in shortfalls),
            "maximum_entity_shortfall": max(shortfalls),
            "total_shortfall": sum(shortfalls),
        }
    return result


def _validate_level_count_feasibility(
    observed: object, entities: list[dict[str, Any]], checks: Checks
) -> None:
    checks.require(isinstance(observed, dict), "level feasibility is not an object")
    expected = _expected_level_count_feasibility(entities)
    checks.require(
        set(observed) == set(expected),
        "level feasibility levels are not 32/128/768/1536",
    )
    for level, expected_row in expected.items():
        row = observed[level]
        checks.require(
            isinstance(row, dict), f"level feasibility is not an object: {level}"
        )
        _require_fields(row, LEVEL_FIELDS, f"level feasibility {level}", checks)
        checks.require(
            type(row["all_entities_count_feasible"]) is bool,
            f"level feasibility all flag is not boolean: {level}",
        )
        for field in (
            "entity_count_feasible",
            "maximum_entity_shortfall",
            "total_shortfall",
        ):
            _require_nonnegative_int(
                row[field], f"level feasibility {level}.{field}", checks
            )
        checks.require(
            row == expected_row, f"level feasibility arithmetic mismatch: {level}"
        )


def _validate_all_atom_count_diagnostics(
    entities: list[dict[str, Any]], checks: Checks
) -> None:
    reference_counts: dict[str, int] = {}
    largest_variant_counts: dict[str, int] = {}
    for level in LEVELS:
        reference_counts[str(level)] = 0
        largest_variant_counts[str(level)] = 0
        for entity in entities:
            topology_counts: dict[str, int] = {}
            for record in entity["files"]:
                topology = str(record["all_atom_topology_sha256"])
                topology_counts[topology] = topology_counts.get(topology, 0) + 1
            reference_count = topology_counts.get(
                str(entity["canonical_all_atom_topology_sha256"]), 0
            )
            largest_variant_count = max(topology_counts.values(), default=0)
            reference_counts[str(level)] += reference_count >= level
            largest_variant_counts[str(level)] += largest_variant_count >= level
    checks.require(
        {
            "reference_topology_entity_count_ge_k": reference_counts,
            "largest_single_topology_variant_entity_count_ge_k": (
                largest_variant_counts
            ),
        }
        == EXPECTED_ALL_ATOM_COUNT_DIAGNOSTICS,
        "all-atom topology-count diagnostics mismatch",
    )


def _validate_summary(
    summary: dict[str, Any],
    entities: list[dict[str, Any]],
    bindings: tuple[str, str, str],
    checks: Checks,
) -> None:
    _require_fields(summary, SUMMARY_FIELDS, "summary", checks)
    checks.require(
        summary["contract"] == RECEIPT_CONTRACT, "wrong summary receipt contract"
    )
    checks.require(
        summary["artifact_kind"] == RECEIPT_ARTIFACT_KIND,
        "wrong summary receipt artifact kind",
    )
    checks.require(summary["study_id"] == STUDY_ID, "wrong summary study ID")
    _require_false_sentinels(summary, "summary", checks)
    checks.require(
        _is_int(summary["catalog_shard_count"])
        and summary["catalog_shard_count"] == SHARD_COUNT,
        "summary shard count mismatch",
    )
    checks.require(
        _is_int(summary["entity_count"]) and summary["entity_count"] == ENTITY_COUNT,
        "summary entity count mismatch",
    )
    checks.require(
        isinstance(summary["source_method_composition"], dict)
        and set(summary["source_method_composition"]) == {"BioEmu"}
        and type(summary["source_method_composition"]["BioEmu"]) is float
        and summary["source_method_composition"]["BioEmu"] == 1.0,
        "source method composition is not exactly BioEmu:1.0",
    )
    summary_bindings = (
        _require_sha256(summary["roster_sha256"], "summary.roster_sha256", checks),
        _require_safe_relative_path(
            summary["source_commitment_relative_path"],
            "summary.source_commitment_relative_path",
            checks,
        ),
        _require_sha256(
            summary["source_commitment_sha256"],
            "summary.source_commitment_sha256",
            checks,
        ),
    )
    _require_evidence_bindings(summary_bindings, bindings, "summary", checks)
    _require_sha256(
        summary["full_catalog_digest"], "summary.full_catalog_digest", checks
    )

    checks.require(
        len(entities) == ENTITY_COUNT, "catalog does not contain 135 entities"
    )
    entity_uids = [entity["entity_uid"] for entity in entities]
    bmrb_ids = [entity["bmrb_id"] for entity in entities]
    checks.require(
        len(set(entity_uids)) == ENTITY_COUNT, "entity UID roster is not unique"
    )
    checks.require(len(set(bmrb_ids)) == ENTITY_COUNT, "BMRB roster is not unique")
    expected_rows = sorted(
        (_summary_projection(entity) for entity in entities),
        key=lambda row: row["entity_uid"],
    )
    checks.require(
        isinstance(summary["entities"], list), "summary entities is not a list"
    )
    checks.require(
        summary["entities"] == expected_rows,
        "summary entity projection or entity order mismatch",
    )
    checks.require(
        summary["full_catalog_digest"]
        == _canonical_json_sha256(sorted(entities, key=lambda row: row["entity_uid"])),
        "full catalog digest mismatch",
    )
    _validate_level_count_feasibility(
        summary["level_count_feasibility"], entities, checks
    )


def check_catalog(
    summary_path: Path,
    shard_archive_path: Path,
    roster_path: Path,
    source_commitment_path: Path,
    producer_path: Path,
) -> int:
    """Validate only bound, target-unread catalog evidence artifacts."""
    checks = Checks()
    summary_raw = summary_path.read_bytes()
    summary_sha256 = hashlib.sha256(summary_raw).hexdigest()
    checks.require(
        summary_sha256 == EXPECTED_SUMMARY_RAW_SHA256,
        "summary raw SHA-256 mismatch",
    )
    roster_raw = roster_path.read_bytes()
    roster_sha256 = hashlib.sha256(roster_raw).hexdigest()
    checks.require(
        roster_sha256 == EXPECTED_ROSTER_RAW_SHA256,
        "entity roster raw SHA-256 mismatch",
    )
    source_raw = source_commitment_path.read_bytes()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    checks.require(
        source_sha256 == EXPECTED_SOURCE_COMMITMENT_RAW_SHA256,
        "source commitment raw SHA-256 mismatch",
    )
    _require_raw_sha256(producer_path, EXPECTED_PRODUCER_RAW_SHA256, "producer", checks)
    _require_raw_sha256(
        shard_archive_path,
        EXPECTED_SHARD_ARCHIVE_RAW_SHA256,
        "shard archive",
        checks,
    )

    summary = _decode_json(summary_raw, str(summary_path))
    roster = _decode_json(roster_raw, str(roster_path))
    source_commitment = _decode_json(source_raw, str(source_commitment_path))
    roster_by_uid, roster_source_binding = _validate_roster(roster, checks)
    _validate_source_commitment(source_commitment, roster_by_uid, checks)
    bindings = (roster_sha256, *roster_source_binding)
    checks.require(
        bindings == (roster_sha256, SOURCE_COMMITMENT_RELATIVE_PATH, source_sha256),
        "entity roster/source commitment binding mismatch",
    )
    shards = _read_shards(shard_archive_path, checks)
    _require_raw_sha256(
        shard_archive_path,
        EXPECTED_SHARD_ARCHIVE_RAW_SHA256,
        "shard archive changed during validation",
        checks,
    )
    entities = _validate_shards(shards, roster_by_uid, bindings, checks)
    _validate_summary(summary, entities, bindings, checks)
    _validate_all_atom_count_diagnostics(entities, checks)
    return checks.count


def _self_test_entity() -> dict[str, Any]:
    digest = "0" * 64
    file_row = {
        "all_atom_coordinate_sha256": digest,
        "all_atom_topology_matches_reference": True,
        "all_atom_topology_sha256": digest,
        "atom_count": 2,
        "heavy_atom_coordinate_sha256": digest,
        "heavy_atom_count": 1,
        "heavy_atom_topology_sha256": digest,
        "pdb_sha256": digest,
        "support_index": 1,
    }
    return {
        "all_atom_topology_variant_count": 1,
        "bmrb_id": "bmr1",
        "canonical_all_atom_topology_sha256": digest,
        "canonical_atom_count": 2,
        "canonical_filename_count": 1,
        "canonical_heavy_atom_count": 1,
        "canonical_heavy_topology_sha256": digest,
        "canonical_reference_pdb_sha256": digest,
        "canonical_reference_relative_path": (
            "data/k32_complete_coordinate_supports_v4/bmr1/bmr1_BioEmu_1.pdb"
        ),
        "canonical_reference_support_index": 1,
        "catalog_digest": _canonical_json_sha256([file_row]),
        "entity_uid": "bmrb:1:entity:1",
        "files": [file_row],
        "heavy_topology_compatible_unique_coordinate_count": 1,
        "hydrogen_topology_variation_present": False,
        "invalid": [],
        "invalid_count": 0,
        "missing_indices_1_to_1000": list(range(2, 1001)),
        "observer_fold": "A",
        "split": "train",
        "unexpected_entries": [],
        "valid_index_digest": _canonical_json_sha256([1]),
    }


def _self_test_roster() -> dict[str, Any]:
    digest = "0" * 64
    entities = []
    for index in range(1, ENTITY_COUNT + 1):
        bmrb_id = f"bmr{index}"
        entities.append(
            {
                "bmrb_id": bmrb_id,
                "canonical_all_atom_topology_sha256": digest,
                "canonical_atom_count": 2,
                "canonical_heavy_atom_count": 1,
                "canonical_heavy_topology_sha256": digest,
                "canonical_reference_pdb_sha256": digest,
                "canonical_reference_relative_path": (
                    f"data/k32_complete_coordinate_supports_v4/{bmrb_id}/"
                    f"{bmrb_id}_BioEmu_1.pdb"
                ),
                "canonical_reference_support_index": 1,
                "entity_uid": f"bmrb:{index}:entity:1",
                "observer_fold": "A" if index % 2 else "B",
                "split": "train",
            }
        )
    return {
        "artifact_kind": "target_unread_structural_entity_roster_not_authorization",
        "authorization_consumed": False,
        "contract": "atypemu_nested_support_count_v1_entity_roster_v3",
        "entities": entities,
        "entity_count": ENTITY_COUNT,
        "outer_or_formal_metrics_opened": False,
        "source_commitment_relative_path": SOURCE_COMMITMENT_RELATIVE_PATH,
        "source_commitment_sha256": EXPECTED_SOURCE_COMMITMENT_RAW_SHA256,
        "source_scores_read": False,
        "study_id": STUDY_ID,
        "target_values_read": False,
    }


def _self_test_source_commitment(roster: dict[str, Any]) -> dict[str, Any]:
    digest = "0" * 64
    entities = [
        {
            "bmrb_id": row["bmrb_id"],
            "entity_uid": row["entity_uid"],
            "observer_fold": row["observer_fold"],
            "split": row["split"],
        }
        for row in roster["entities"]
    ]
    features = []
    pdb_files = []
    for position, entity in enumerate(entities):
        row_count = 1 if position < ENTITY_COUNT - 1 else 127_285 * 8 - position
        features.append(
            {
                "bmrb_id": entity["bmrb_id"],
                "entity_uid": entity["entity_uid"],
                "relative_path": (
                    f"data/all_atom_observer_v1/features/{entity['bmrb_id']}.parquet"
                ),
                "row_count": row_count,
                "sha256": digest,
            }
        )
        for support_index in SOURCE_SUPPORT_INDEXES:
            support_id = f"BioEmu_{support_index}"
            pdb_files.append(
                {
                    "bmrb_id": entity["bmrb_id"],
                    "entity_uid": entity["entity_uid"],
                    "relative_path": (
                        f"data/k32_complete_coordinate_supports_v4/{entity['bmrb_id']}/"
                        f"{entity['bmrb_id']}_{support_id}.pdb"
                    ),
                    "sha256": digest,
                    "support_id": support_id,
                    "support_index": support_index,
                }
            )
    source: dict[str, Any] = {
        "bound_files": [
            {"relative_path": path, "sha256": digest}
            for path in SOURCE_BOUND_RELATIVE_PATHS
        ],
        "contract": "k32_dynamic_distance_cache_source_commitment_v1",
        "entities": entities,
        "entity_count": ENTITY_COUNT,
        "expected_feature_row_count": 127_285,
        "expected_target_available_rows": 4_072_968,
        "expected_target_missing_rows": 152,
        "expected_target_support_rows": 4_073_120,
        "feature_file_count": ENTITY_COUNT,
        "features": features,
        "formal_evaluation_authorized": False,
        "parent_commitment_relative_path": SOURCE_PARENT_COMMITMENT_RELATIVE_PATH,
        "parent_commitment_sha256": digest,
        "parent_receipt_relative_path": SOURCE_PARENT_RECEIPT_RELATIVE_PATH,
        "parent_receipt_sha256": digest,
        "pdb_file_count": ENTITY_COUNT * len(SOURCE_SUPPORT_INDEXES),
        "pdb_files": pdb_files,
        "script_relative_path": SOURCE_SCRIPT_RELATIVE_PATH,
        "script_sha256": digest,
        "source_gate_authorized": False,
        "support_ids": list(SOURCE_SUPPORT_IDS),
        "support_indices": list(SOURCE_SUPPORT_INDEXES),
        "target_values_read": False,
    }
    source["commitment_sha256"] = _canonical_json_sha256(source)
    return source


def _expect_self_test_rejection(name: str, operation: Any) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError(f"self-test accepted tampered {name}")


def self_test() -> int:
    """Exercise catalog and independent evidence-binding rejection paths."""
    _validate_entity(_self_test_entity(), "self-test valid", Checks())
    catalog_cases: list[tuple[str, dict[str, Any]]] = []
    bad_digest = _self_test_entity()
    bad_digest["catalog_digest"] = "f" * 64
    catalog_cases.append(("catalog digest", bad_digest))
    bad_schema = _self_test_entity()
    bad_schema["extra"] = False
    catalog_cases.append(("unknown entity field", bad_schema))
    bad_flag = _self_test_entity()
    bad_flag["files"][0]["all_atom_topology_matches_reference"] = False
    catalog_cases.append(("topology reference flag", bad_flag))
    for name, entity in catalog_cases:
        _expect_self_test_rejection(
            name,
            lambda entity=entity, name=name: _validate_entity(
                copy.deepcopy(entity), f"self-test {name}", Checks()
            ),
        )

    roster = _self_test_roster()
    roster_by_uid, _ = _validate_roster(copy.deepcopy(roster), Checks())
    source = _self_test_source_commitment(roster)
    _validate_source_commitment(copy.deepcopy(source), roster_by_uid, Checks())

    bad_source_fold = copy.deepcopy(source)
    bad_source_fold["entities"][0]["observer_fold"] = "B"
    bad_source_fold["commitment_sha256"] = _canonical_json_sha256(
        {
            key: value
            for key, value in bad_source_fold.items()
            if key != "commitment_sha256"
        }
    )
    bad_source_entity = copy.deepcopy(source)
    bad_source_entity["entities"][0]["bmrb_id"] = "bmr999"
    bad_source_entity["entities"][0]["entity_uid"] = "bmrb:999:entity:1"
    bad_source_entity["commitment_sha256"] = _canonical_json_sha256(
        {
            key: value
            for key, value in bad_source_entity.items()
            if key != "commitment_sha256"
        }
    )
    bad_reference_roster = copy.deepcopy(roster)
    bad_reference_roster["entities"][0]["canonical_reference_pdb_sha256"] = "f" * 64
    bad_reference_by_uid, _ = _validate_roster(bad_reference_roster, Checks())

    evidence_cases = (
        ("empty roster", lambda: _validate_roster({}, Checks())),
        (
            "empty source commitment",
            lambda: _validate_source_commitment({}, roster_by_uid, Checks()),
        ),
        (
            "changed source fold",
            lambda: _validate_source_commitment(
                bad_source_fold, roster_by_uid, Checks()
            ),
        ),
        (
            "changed source entity",
            lambda: _validate_source_commitment(
                bad_source_entity, roster_by_uid, Checks()
            ),
        ),
        (
            "changed roster reference",
            lambda: _validate_source_commitment(source, bad_reference_by_uid, Checks()),
        ),
        (
            "mismatched binding",
            lambda: _require_evidence_bindings(
                ("0" * 64, SOURCE_COMMITMENT_RELATIVE_PATH, "1" * 64),
                ("1" * 64, SOURCE_COMMITMENT_RELATIVE_PATH, "1" * 64),
                "self-test",
                Checks(),
            ),
        ),
    )
    for name, operation in evidence_cases:
        _expect_self_test_rejection(name, operation)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "bound-evidence"
        path.write_bytes(b"bound")
        expected_hash = _sha256_file(path, "self-test bound evidence")
        _require_raw_sha256(path, expected_hash, "self-test bound evidence", Checks())
        path.write_bytes(path.read_bytes() + b" appended")
        _expect_self_test_rejection(
            "appended bound evidence bytes",
            lambda: _require_raw_sha256(
                path, expected_hash, "self-test bound evidence", Checks()
            ),
        )
    return len(catalog_cases) + len(evidence_cases) + 3


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path)
    result.add_argument("--shard-archive", type=Path)
    result.add_argument("--entity-roster", type=Path)
    result.add_argument("--source-commitment", type=Path)
    result.add_argument("--producer", type=Path)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    evidence_paths = (
        args.summary,
        args.shard_archive,
        args.entity_roster,
        args.source_commitment,
        args.producer,
    )
    if any(path is None for path in evidence_paths) and not all(
        path is None for path in evidence_paths
    ):
        parser().error(
            "summary, shard archive, roster, source commitment, and producer are "
            "required together"
        )
    if not args.self_test and any(path is None for path in evidence_paths):
        parser().error("evidence paths are required unless --self-test is used alone")
    try:
        check_count = 0
        if all(path is not None for path in evidence_paths):
            check_count = check_catalog(
                args.summary,
                args.shard_archive,
                args.entity_roster,
                args.source_commitment,
                args.producer,
            )
        if args.self_test:
            check_count += self_test()
    except Exception as error:  # noqa: BLE001 - checker failures must not emit success.
        print(f"FAIL {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"METRIC support_catalog_checker_checks={check_count}")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
