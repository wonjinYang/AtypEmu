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
LEVELS = (32, 128, 768, 1536)
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
        raise ValueError(f"JSON object required: {label}")
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


def _require_false_sentinels(
    payload: dict[str, Any], label: str, checks: Checks
) -> None:
    for field in SENTINEL_FIELDS:
        checks.require(
            type(payload.get(field)) is bool and payload[field] is False,
            f"{label} sentinel is not false: {field}",
        )


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
        isinstance(entity_uid, str)
        and ENTITY_UID_RE.fullmatch(entity_uid) is not None,
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
    checks.require(
        reference_index in by_index, f"reference support is absent: {label}"
    )
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
    topology_variants = {
        record["all_atom_topology_sha256"] for record in row["files"]
    }
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
                    re.fullmatch(r"shard_(?:[0-9]|1[0-9]|2[0-6])\.json", normalized_name)
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
    shards: list[dict[str, Any]], checks: Checks
) -> tuple[list[dict[str, Any]], tuple[str, str, str]]:
    all_entities: list[dict[str, Any]] = []
    indexes: set[int] = set()
    bindings: tuple[str, str, str] | None = None
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
        if bindings is None:
            bindings = shard_bindings
        checks.require(
            shard_bindings == bindings, f"source/roster bindings differ: {label}"
        )
        checks.require(
            isinstance(shard["entities"], list), f"entities is not a list: {label}"
        )
        checks.require(
            _is_int(shard["entity_count"])
            and shard["entity_count"] == len(shard["entities"]),
            f"entity count mismatch: {label}",
        )
        for entity_position, entity in enumerate(shard["entities"]):
            all_entities.append(
                _validate_entity(entity, f"{label}.entities[{entity_position}]", checks)
            )
    checks.require(
        indexes == set(range(SHARD_COUNT)), "shard indexes are not exactly 0..26"
    )
    checks.require(bindings is not None, "no shard bindings")
    return all_entities, bindings


def _expected_level_count_feasibility(
    entities: list[dict[str, Any]]
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
    checks.require(
        summary_bindings == bindings, "summary source/roster binding mismatch"
    )
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


def check_catalog(summary_path: Path, shard_archive_path: Path) -> int:
    """Validate only the supplied JSON receipt and unextracted JSON shard archive."""
    checks = Checks()
    summary = _read_json(summary_path)
    shards = _read_shards(shard_archive_path, checks)
    entities, bindings = _validate_shards(shards, checks)
    _validate_summary(summary, entities, bindings, checks)
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


def self_test() -> int:
    """Exercise valid, digest, exact-schema, and topology-flag paths."""
    _validate_entity(_self_test_entity(), "self-test valid", Checks())
    cases: list[tuple[str, dict[str, Any]]] = []
    bad_digest = _self_test_entity()
    bad_digest["catalog_digest"] = "f" * 64
    cases.append(("catalog digest", bad_digest))
    bad_schema = _self_test_entity()
    bad_schema["extra"] = False
    cases.append(("unknown entity field", bad_schema))
    bad_flag = _self_test_entity()
    bad_flag["files"][0]["all_atom_topology_matches_reference"] = False
    cases.append(("topology reference flag", bad_flag))
    for name, entity in cases:
        try:
            _validate_entity(copy.deepcopy(entity), f"self-test {name}", Checks())
        except ValueError:
            continue
        raise AssertionError(f"self-test accepted tampered {name}")
    return len(cases) + 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--shard-archive", type=Path, required=True)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        check_count = check_catalog(args.summary, args.shard_archive)
        if args.self_test:
            check_count += self_test()
    except Exception as error:  # A checker must never issue success metrics on failure.
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
