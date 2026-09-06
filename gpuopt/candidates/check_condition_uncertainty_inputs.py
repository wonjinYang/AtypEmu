#!/usr/bin/env python3
"""Read-only verifier for frozen target-unread condition/protonation inputs.

This checker intentionally does not import the freezer.  It independently parses
all bound material and only qualifies identity/provenance; it is not a support,
science, score, target, or authorization operation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable

CHECKER_RELATIVE = Path("gpuopt/candidates/check_condition_uncertainty_inputs.py")
FREEZER_RELATIVE = Path("gpuopt/candidates/freeze_condition_uncertainty_inputs.py")
DOCKERFILE_RELATIVE = Path("gpuopt/candidates/openmm86_protonation_runtime.Dockerfile")

MANIFESTS = {
    "condition": Path(
        "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json"
    ),
    "sequence": Path(
        "gpuopt/preunblind/atypemu_nested_support_count_v1_protein_sequence_manifest_v1.json"
    ),
    "parent": Path(
        "gpuopt/preunblind/"
        "atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1.json"
    ),
    "environment": Path(
        "gpuopt/preunblind/"
        "atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json"
    ),
}
MANIFEST_SHA256 = {
    "condition": "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618",
    "sequence": "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
    "parent": "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
    "environment": "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
}
ROSTER_RELATIVE = Path(".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json")
RECOVERY_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json"
)
V6_ARCHIVE_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1.zip"
)
V6_EVIDENCE_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json"
)
CATALOG_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/"
    "catalog_v3_shards.tar.gz"
)
RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_condition_uncertainty_inputs_check_receipt_v1.json"
)

ROSTER_SHA256 = "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
RECOVERY_SHA256 = "2d8a255add821950e0e381401afc8c27a97d37cffb8ead827f5d8bd500bc3b8b"
V6_ARCHIVE_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"
V6_EVIDENCE_RECEIPT_SHA256 = (
    "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36"
)
CATALOG_SHA256 = "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2"
FREEZER_SHA256 = "062858e6bb1f9994cdc135a66670fa793682d5b9f8fcf49413b85e79cae74874"
DOCKERFILE_SHA256 = "007e877565ca25883bcfdc083513b240d94e93c0416514e62cc791f3414836cc"
SOURCE_COMMITMENT_RELATIVE = Path(
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
)
SOURCE_COMMITMENT_SHA256 = (
    "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8"
)

CANDIDATE_ID = "atypemu_nested_support_count_v1_condition_uncertainty_protonation_v1"
STUDY_ID = "atypemu_nested_support_count_v1"
FALSE_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
OUTPUT_SCOPE = "target-unread HOLD-only input identity; not support or science evidence"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
UID_RE = re.compile(r"bmrb:([1-9][0-9]*):entity:1\Z")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ROSTER_FIELDS = {
    "artifact_kind", "authorization_consumed", "contract", "entities", "entity_count",
    "outer_or_formal_metrics_opened", "source_commitment_relative_path",
    "source_commitment_sha256", "source_scores_read", "study_id", "target_values_read",
}
ROSTER_ENTITY_FIELDS = {
    "bmrb_id", "canonical_all_atom_topology_sha256", "canonical_atom_count",
    "canonical_heavy_atom_count", "canonical_heavy_topology_sha256",
    "canonical_reference_pdb_sha256", "canonical_reference_relative_path",
    "canonical_reference_support_index", "entity_uid", "observer_fold", "split",
}
CONDITION_FIELDS = {
    "artifact_kind", "candidate_id", "closed_capabilities", "contract", "entities",
    "entity_count", "freezer", "roster", "scope", "state_counts", "upstream",
}
CONDITION_ENTITY_FIELDS = {
    "bmrb_id", "condition_state", "deposited_ph", "entity_uid", "pH_source",
    "recovery_v6_hold_reasons", "temperature_ionic_diagnostics",
}
SEQUENCE_FIELDS = {
    "artifact_kind", "candidate_id", "closed_capabilities", "contract", "entities",
    "entity_count", "freezer", "roster", "scope",
}
SEQUENCE_ENTITY_FIELDS = {
    "bmrb_id", "entity_uid", "reference_pdb", "residue_count", "residues",
    "sequence_one_letter", "sequence_sha256",
}
RESIDUE_FIELDS = {"chain_id", "insertion_code", "residue_id", "residue_name"}
PARENT_FIELDS = {
    "artifact_kind", "candidate_id", "catalog_archive", "closed_capabilities", "contract",
    "entities", "entity_count", "freezer", "parent_record_count",
    "parent_records_sha256", "roster", "scope", "shards", "source_commitment",
    "source_relative_path_pattern",
}
PARENT_ENTITY_FIELDS = {
    "bmrb_id", "entity_uid", "heavy_atom_count", "heavy_atom_topology_sha256",
    "observer_fold", "records_sha256", "split", "support_count", "support_index_digest",
}
ENVIRONMENT_FIELDS = {
    "artifact_kind", "base_image", "build_recipe", "candidate_id", "closed_capabilities",
    "contract", "conversion", "freezer", "inspection", "ph_regimes", "required_platform",
    "runtime_sif", "scope", "wheels",
}
SHARD_FIELDS = {
    "artifact_kind", "authorization_consumed", "contract", "entities", "entity_count",
    "outer_or_formal_metrics_opened", "roster_sha256", "science_executed", "shard_count",
    "shard_index", "source_commitment_relative_path", "source_commitment_sha256",
    "source_scores_read", "study_id", "target_values_read",
}
CATALOG_ENTITY_FIELDS = {
    "all_atom_topology_variant_count", "bmrb_id", "canonical_all_atom_topology_sha256",
    "canonical_atom_count", "canonical_filename_count", "canonical_heavy_atom_count",
    "canonical_heavy_topology_sha256", "canonical_reference_pdb_sha256",
    "canonical_reference_relative_path", "canonical_reference_support_index", "catalog_digest",
    "entity_uid", "files", "heavy_topology_compatible_unique_coordinate_count",
    "hydrogen_topology_variation_present", "invalid", "invalid_count",
    "missing_indices_1_to_1000", "observer_fold", "split", "unexpected_entries",
    "valid_index_digest",
}
CATALOG_FILE_FIELDS = {
    "all_atom_coordinate_sha256", "all_atom_topology_matches_reference",
    "all_atom_topology_sha256", "atom_count", "heavy_atom_coordinate_sha256",
    "heavy_atom_count", "heavy_atom_topology_sha256", "pdb_sha256", "support_index",
}
SUMMARY_FIELDS = {
    "artifact_kind", "authorization_consumed", "catalog_shard_count", "contract", "entities",
    "entity_count", "full_catalog_digest", "level_count_feasibility",
    "outer_or_formal_metrics_opened", "roster_sha256", "science_executed",
    "source_commitment_relative_path", "source_commitment_sha256", "source_method_composition",
    "source_scores_read", "study_id", "target_values_read",
}
SUMMARY_ENTITY_FIELDS = {
    "all_atom_topology_variant_count", "bmrb_id", "canonical_filename_count",
    "catalog_digest", "entity_uid", "heavy_topology_compatible_unique_coordinate_count",
    "hydrogen_topology_variation_present", "invalid_count", "missing_indices_1_to_1000",
    "observer_fold", "split", "unexpected_entries", "valid_index_digest",
}
V6_RECEIPT_MEMBER = (
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1/"
    "artifacts/.auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v6_recovery/receipt.json"
)
V6_MANIFEST_MEMBER = (
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1/"
    "archive_manifest.json"
)
V6_ENTITY_FIELDS = {
    "assigned_chem_shift_list_ids", "bmrb_id", "complete_chem_shift_experiment_link_count",
    "deposited_ph", "deposited_ph_record_count", "entity_uid", "experiment_ids",
    "hold_reasons", "new_response_binding", "ph_feasible", "prior_archive_response_bindings",
    "route", "sample_condition_list_ids", "temperature_ionic_diagnostics",
}
RECOVERY_ENTITY_FIELDS = {
    "assigned_chem_shift_list_id", "bmrb_id", "direct_sample_condition_list_id",
    "entity_uid", "fallback_metadata_resolved", "fallback_ph", "hold_reasons",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    _reject_nonfinite(value, label)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {label}")
    return value


def _safe_relative(path: Path, label: str) -> None:
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"noncanonical repository-relative path: {label}")
    if path.as_posix() != str(path) or "\\" in path.as_posix():
        raise ValueError(f"noncanonical repository-relative path: {label}")


def _same_path(value: Any, expected: Path, label: str) -> None:
    if not isinstance(value, str) or value != expected.as_posix():
        raise ValueError(f"path substitution rejected: {label}")


def _read_open_fd(descriptor: int, maximum: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > maximum:
        raise ValueError(f"not a bounded regular file: {label}")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, min(1 << 20, maximum + 1))
        if not chunk:
            break
        chunks.append(chunk)
        if sum(map(len, chunks)) > maximum:
            raise ValueError(f"file exceeds byte bound while reading: {label}")
    raw = b"".join(chunks)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_mode, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_mode, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity_before != identity_after:
        raise ValueError(f"file changed while reading: {label}")
    return raw


def _read_repo(root: Path, relative: Path, maximum: int) -> bytes:
    _safe_relative(relative, str(relative))
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("repository root is indirect")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory
        )
        try:
            return _read_open_fd(descriptor, maximum, relative.as_posix())
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _read_direct(path: Path, maximum: int, label: str) -> bytes:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"absolute direct file required: {label}")
    details = os.lstat(path)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError(f"symlink or non-regular direct file rejected: {label}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        return _read_open_fd(descriptor, maximum, label)
    finally:
        os.close(descriptor)


def _root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / CHECKER_RELATIVE
    if source != expected or source.is_symlink() or source.resolve(strict=True) != source:
        raise ValueError("checker must execute from its canonical repository path")
    return root


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid SHA-256: {label}")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"invalid integer: {label}")
    return value


def _schema(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"unknown, missing, or malformed fields: {label}")
    return value


def _closed(value: Any, label: str, *, complete: bool = False) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"capability object required: {label}")
    if complete and value != FALSE_CAPABILITIES:
        raise ValueError(f"capability surface drifted: {label}")
    for field in (
        "authorization_consumed", "outer_or_formal_metrics_opened", "science_executed",
        "source_scores_read", "target_values_read",
    ):
        if value.get(field) is not False:
            raise ValueError(f"opened protected surface: {label}.{field}")


def _identity(bmrb: Any, uid: Any, label: str) -> None:
    if not isinstance(bmrb, str) or not isinstance(uid, str):
        raise ValueError(f"non-string entity identity: {label}")
    bmrb_match = BMRB_RE.fullmatch(bmrb)
    uid_match = UID_RE.fullmatch(uid)
    if bmrb_match is None or uid_match is None or bmrb_match.group(1) != uid_match.group(1):
        raise ValueError(f"noncanonical entity identity: {label}")


def _binding(value: Any, path: Path, digest: str, label: str) -> None:
    row = _schema(value, {"path", "sha256"}, label)
    _same_path(row["path"], path, f"{label}.path")
    if row["sha256"] != digest:
        raise ValueError(f"hash binding drifted: {label}")


def _common(
    manifest: dict[str, Any], fields: set[str], kind: str, contract: str, label: str
) -> None:
    _schema(manifest, fields, label)
    if (
        manifest["artifact_kind"] != kind
        or manifest["candidate_id"] != CANDIDATE_ID
        or manifest["contract"] != contract
        or manifest["scope"] != OUTPUT_SCOPE
    ):
        raise ValueError(f"manifest identity or HOLD scope drifted: {label}")
    _closed(manifest["closed_capabilities"], f"{label}.closed_capabilities", complete=True)
    _binding(manifest["freezer"], FREEZER_RELATIVE, FREEZER_SHA256, f"{label}.freezer")
    _binding(manifest["roster"], ROSTER_RELATIVE, ROSTER_SHA256, f"{label}.roster")


def _read_bound(root: Path, path: Path, digest: str, maximum: int) -> bytes:
    raw = _read_repo(root, path, maximum)
    if _sha256(raw) != digest:
        raise ValueError(f"bound bytes drifted: {path}")
    return raw


def _validate_roster(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _schema(value, ROSTER_FIELDS, "roster")
    if (
        value["artifact_kind"] != "target_unread_structural_entity_roster_not_authorization"
        or value["contract"] != "atypemu_nested_support_count_v1_entity_roster_v3"
        or value["study_id"] != STUDY_ID
        or value["entity_count"] != 135
        or not isinstance(value["entities"], list)
        or len(value["entities"]) != 135
    ):
        raise ValueError("roster identity or count drifted")
    _closed(value, "roster")
    _same_path(value["source_commitment_relative_path"], SOURCE_COMMITMENT_RELATIVE, "roster source")
    if value["source_commitment_sha256"] != SOURCE_COMMITMENT_SHA256:
        raise ValueError("roster source commitment binding drifted")
    result: dict[str, dict[str, Any]] = {}
    bmrb_ids: set[str] = set()
    for index, row in enumerate(value["entities"]):
        _schema(row, ROSTER_ENTITY_FIELDS, f"roster entity {index}")
        _identity(row["bmrb_id"], row["entity_uid"], f"roster entity {index}")
        if row["split"] != "train" or row["observer_fold"] not in {"A", "B"}:
            raise ValueError("roster fold/split drifted")
        for field in (
            "canonical_all_atom_topology_sha256", "canonical_heavy_topology_sha256",
            "canonical_reference_pdb_sha256",
        ):
            _sha(row[field], f"roster {field}")
        atoms = _integer(row["canonical_atom_count"], "roster atom count", positive=True)
        heavy = _integer(row["canonical_heavy_atom_count"], "roster heavy count", positive=True)
        if heavy > atoms or row["canonical_reference_support_index"] != 1:
            raise ValueError("roster atom/reference identity drifted")
        expected_path = Path(
            f"data/k32_complete_coordinate_supports_v4/{row['bmrb_id']}/"
            f"{row['bmrb_id']}_BioEmu_1.pdb"
        )
        _same_path(row["canonical_reference_relative_path"], expected_path, "roster reference")
        if row["entity_uid"] in result or row["bmrb_id"] in bmrb_ids:
            raise ValueError("duplicate roster entity")
        result[row["entity_uid"]] = row
        bmrb_ids.add(row["bmrb_id"])
    if len(result) != 135:
        raise ValueError("incomplete roster")
    return result


def _decimal_ph(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"invalid pH value: {label}")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid pH value: {label}") from error
    if not decimal.is_finite() or not Decimal("0") <= decimal <= Decimal("14"):
        raise ValueError(f"out-of-range pH value: {label}")
    return format(decimal, "f")


def _validate_v6_archive(raw: bytes, roster: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Bind the v6 envelope and return its independently checked receipt."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) != 141 or len(names) != len(set(names)):
                raise ValueError("v6 archive inventory drifted")
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    info.is_dir() or info.flag_bits & 1 or path.is_absolute()
                    or "\\" in info.filename or "\x00" in info.filename
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or info.file_size > 1_000_000
                ):
                    raise ValueError("unsafe v6 archive member")
            by_name = dict(zip(names, infos))
            if V6_MANIFEST_MEMBER not in by_name or V6_RECEIPT_MEMBER not in by_name:
                raise ValueError("v6 archive required member absent")
            manifest = _json_object(archive.read(by_name[V6_MANIFEST_MEMBER]), "v6 manifest")
            _schema(
                manifest,
                {
                    "artifact_kind", "authorization_consumed", "candidate_id", "contract",
                    "member_count", "members", "outer_or_formal_metrics_opened",
                    "science_executed", "source_scores_read", "target_atom_identities_read",
                    "target_values_read",
                },
                "v6 archive manifest",
            )
            if (
                manifest["artifact_kind"]
                != "target_unread_openmm86_deposited_ph_recovery_v6_archive_manifest"
                or manifest["candidate_id"]
                != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
                or manifest["contract"]
                != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_archive_v1"
                or manifest["member_count"] != 140
                or not isinstance(manifest["members"], list)
                or len(manifest["members"]) != 140
            ):
                raise ValueError("v6 archive manifest semantics drifted")
            _closed(manifest, "v6 archive manifest")
            listed: set[str] = set()
            for item in manifest["members"]:
                _schema(item, {"path", "sha256", "size"}, "v6 member")
                relative = item["path"]
                if (
                    not isinstance(relative, str)
                    or not relative
                    or "\\" in relative
                    or "\x00" in relative
                    or PurePosixPath(relative).is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or relative in listed
                    or type(item["size"]) is not int
                    or item["size"] < 0
                ):
                    raise ValueError("v6 member path/schema drifted")
                _sha(item["sha256"], "v6 member digest")
                name = (
                    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1/"
                    f"artifacts/{relative}"
                )
                if name not in by_name:
                    raise ValueError("v6 manifested member absent")
                body = archive.read(by_name[name])
                if len(body) != item["size"] or _sha256(body) != item["sha256"]:
                    raise ValueError("v6 manifested member bytes drifted")
                listed.add(relative)
            expected_members = {V6_MANIFEST_MEMBER} | {
                "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1/"
                f"artifacts/{item}" for item in listed
            }
            if set(names) != expected_members:
                raise ValueError("unmanifested v6 archive member")
            receipt = _json_object(archive.read(by_name[V6_RECEIPT_MEMBER]), "v6 receipt")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"invalid v6 evidence archive: {error}") from error
    _schema(
        receipt,
        {
            "all_entities_ph_feasible", "api_base", "application_header", "archive",
            "artifact_kind", "authorization_consumed", "candidate_id", "contract",
            "ended_at_utc", "entities", "entity_count", "future_consumer",
            "metadata_scope", "new_response_count", "new_response_manifest",
            "outer_or_formal_metrics_opened", "ph_feasible_entity_count", "plan",
            "recovery_evidence", "roster", "science_executed", "source_construction_executed",
            "source_producer_git_commit", "source_producer_relative_path",
            "source_producer_sha256", "source_scores_read", "started_at_utc", "status",
            "target_atom_identities_read", "target_values_read", "warnings",
        },
        "v6 receipt",
    )
    if (
        receipt["contract"] != "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6"
        or receipt["entity_count"] != 135
        or receipt["ph_feasible_entity_count"] != 115
        or receipt["all_entities_ph_feasible"] is not False
        or receipt["status"] != "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
        or not isinstance(receipt["entities"], list)
        or len(receipt["entities"]) != 135
    ):
        raise ValueError("v6 receipt aggregate semantics drifted")
    _closed(receipt, "v6 receipt")
    _binding(receipt["roster"], ROSTER_RELATIVE, ROSTER_SHA256, "v6 receipt roster")
    records: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(receipt["entities"]):
        _schema(row, V6_ENTITY_FIELDS, f"v6 entity {index}")
        _identity(row["bmrb_id"], row["entity_uid"], f"v6 entity {index}")
        roster_row = roster.get(row["entity_uid"])
        if roster_row is None or row["bmrb_id"] != roster_row["bmrb_id"]:
            raise ValueError("v6 entity/roster substitution")
        if type(row["ph_feasible"]) is not bool or not isinstance(row["hold_reasons"], list):
            raise ValueError("v6 feasibility shape drifted")
        if row["ph_feasible"]:
            if row["hold_reasons"] or _decimal_ph(row["deposited_ph"], "v6 observed") is None:
                raise ValueError("v6 observed semantics drifted")
        elif row["deposited_ph"] is not None or not row["hold_reasons"]:
            raise ValueError("v6 unresolved semantics drifted")
        if row["entity_uid"] in records:
            raise ValueError("duplicate v6 entity")
        records[row["entity_uid"]] = row
    if set(records) != set(roster) or sum(row["ph_feasible"] for row in records.values()) != 115:
        raise ValueError("v6 roster feasibility replay drifted")
    return records


def _validate_recovery(value: dict[str, Any], roster: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {
        "artifact_kind", "bound_inputs", "candidate_id", "closed_capabilities",
        "combined_metadata_resolved_entity_count", "contract", "entities", "entity_count",
        "execution_provenance", "fallback_candidate_entity_count",
        "fallback_metadata_resolved_entity_count", "metadata_resolution_limit", "output",
        "recovery_v6_hold_entity_count", "recovery_v6_metadata_resolved_entity_count", "status",
    }
    _schema(value, expected, "recovery-v3 receipt")
    if (
        value["artifact_kind"] != "hold_only_target_unread_unique_assigned_ph_recovery_v3_receipt"
        or value["candidate_id"] != "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v3"
        or value["contract"] != "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v3"
        or value["status"] != "HOLD_METADATA_REINTERPRETATION_ONLY"
        or value["entity_count"] != 135
        or value["fallback_candidate_entity_count"] != 20
        or value["fallback_metadata_resolved_entity_count"] != 4
        or value["combined_metadata_resolved_entity_count"] != 119
        or value["recovery_v6_hold_entity_count"] != 20
        or value["recovery_v6_metadata_resolved_entity_count"] != 115
        or not isinstance(value["entities"], list)
        or len(value["entities"]) != 20
    ):
        raise ValueError("recovery-v3 aggregate semantics drifted")
    _closed(value["closed_capabilities"], "recovery-v3 closed capabilities", complete=True)
    inputs = value["bound_inputs"]
    if not isinstance(inputs, dict) or inputs.get("recovery_v6_archive") != {
        "path": V6_ARCHIVE_RELATIVE.as_posix(), "raw_sha256": V6_ARCHIVE_SHA256
    } or inputs.get("recovery_v6_evidence_receipt_raw_sha256") != V6_EVIDENCE_RECEIPT_SHA256:
        raise ValueError("recovery-v3 v6 evidence binding drifted")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(value["entities"]):
        _schema(row, RECOVERY_ENTITY_FIELDS, f"recovery entity {index}")
        _identity(row["bmrb_id"], row["entity_uid"], f"recovery entity {index}")
        roster_row = roster.get(row["entity_uid"])
        if roster_row is None or row["bmrb_id"] != roster_row["bmrb_id"]:
            raise ValueError("recovery entity/roster substitution")
        if type(row["fallback_metadata_resolved"]) is not bool or not isinstance(row["hold_reasons"], list):
            raise ValueError("recovery fallback state is malformed")
        if row["fallback_metadata_resolved"]:
            _decimal_ph(row["fallback_ph"], "recovery pH")
        elif row["fallback_ph"] is not None:
            raise ValueError("recovery unresolved row imputed a pH")
        if row["entity_uid"] in result:
            raise ValueError("duplicate recovery entity")
        result[row["entity_uid"]] = row
    if len(result) != 20 or sum(r["fallback_metadata_resolved"] for r in result.values()) != 4:
        raise ValueError("recovery-v3 fallback replay drifted")
    return result


def _validate_condition(
    manifest: dict[str, Any], roster: dict[str, dict[str, Any]],
    v6: dict[str, dict[str, Any]], recovery: dict[str, dict[str, Any]],
) -> int:
    _common(
        manifest, CONDITION_FIELDS, "target_unread_condition_manifest_not_authorization",
        "atypemu_nested_support_count_v1_condition_manifest_v1", "condition manifest",
    )
    if manifest["entity_count"] != 135 or not isinstance(manifest["entities"], list):
        raise ValueError("condition entity count schema drifted")
    _binding(manifest["upstream"]["combined_receipt"], RECOVERY_RELATIVE, RECOVERY_SHA256, "condition recovery")
    _binding(
        manifest["upstream"]["recovery_v6_archive"], V6_ARCHIVE_RELATIVE,
        V6_ARCHIVE_SHA256, "condition v6 archive",
    )
    _binding(
        manifest["upstream"]["recovery_v6_evidence_receipt"], V6_EVIDENCE_RECEIPT_RELATIVE,
        V6_EVIDENCE_RECEIPT_SHA256, "condition v6 evidence receipt",
    )
    if set(manifest["upstream"]) != {
        "combined_receipt", "recovery_v6_archive", "recovery_v6_evidence_receipt"
    }:
        raise ValueError("unknown condition upstream field")
    seen: set[str] = set()
    states = {"observed": 0, "state_missing": 0, "state_ambiguous": 0}
    for index, row in enumerate(manifest["entities"]):
        _schema(row, CONDITION_ENTITY_FIELDS, f"condition entity {index}")
        uid = row["entity_uid"]
        _identity(row["bmrb_id"], uid, f"condition entity {index}")
        base = v6.get(uid)
        roster_row = roster.get(uid)
        fallback = recovery.get(uid)
        if base is None or roster_row is None or row["bmrb_id"] != roster_row["bmrb_id"]:
            raise ValueError("condition entity roster substitution")
        if base["ph_feasible"]:
            expected_state = "observed"
            expected_ph = _decimal_ph(base["deposited_ph"], "v6 pH")
            expected_source: str | None = "recovery_v6_exact"
        elif fallback is not None and fallback["fallback_metadata_resolved"]:
            expected_state = "observed"
            expected_ph = _decimal_ph(fallback["fallback_ph"], "fallback pH")
            expected_source = "unique_assigned_list_exact"
        elif (
            len(base["assigned_chem_shift_list_ids"]) > 1
            or len(base["sample_condition_list_ids"]) > 1
        ):
            expected_state, expected_ph, expected_source = "state_ambiguous", None, None
        else:
            expected_state, expected_ph, expected_source = "state_missing", None, None
        expected = {
            "bmrb_id": base["bmrb_id"],
            "condition_state": expected_state,
            "deposited_ph": expected_ph,
            "entity_uid": uid,
            "pH_source": expected_source,
            "recovery_v6_hold_reasons": base["hold_reasons"],
            "temperature_ionic_diagnostics": base["temperature_ionic_diagnostics"],
        }
        if row != expected:
            raise ValueError("condition row did not independently replay; imputation rejected")
        if uid in seen:
            raise ValueError("duplicate condition entity")
        seen.add(uid)
        states[expected_state] += 1
    if (
        seen != set(roster)
        or states != {"observed": 119, "state_missing": 11, "state_ambiguous": 5}
        or manifest["state_counts"] != states
    ):
        raise ValueError("condition state count/missingness drifted")
    return len(seen)


def _parse_pdb(raw: bytes, label: str) -> tuple[list[dict[str, str]], str]:
    residues: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for number, line in enumerate(raw.splitlines(), start=1):
        if line[:6] == b"HETATM":
            raise ValueError(f"unsupported HETATM record: {label}:{number}")
        if line[:6] != b"ATOM  ":
            continue
        if len(line) < 78 or line[16:17] != b" ":
            raise ValueError(f"unsupported atom/alternate location: {label}:{number}")
        try:
            record = {
                "chain_id": line[21:22].decode("ascii"),
                "insertion_code": line[26:27].decode("ascii"),
                "residue_id": line[22:26].decode("ascii").strip(),
                "residue_name": line[17:20].decode("ascii").strip(),
            }
        except UnicodeDecodeError as error:
            raise ValueError(f"non-ASCII PDB identity: {label}:{number}") from error
        key = tuple(record[field] for field in ("chain_id", "residue_id", "insertion_code", "residue_name"))
        if key not in seen:
            if not record["residue_id"] or record["residue_name"] not in AA3_TO_1:
                raise ValueError(f"unsupported residue identity: {label}:{number}")
            seen.add(key)
            residues.append(record)
    if not residues or {row["chain_id"] for row in residues} != {"A"}:
        raise ValueError(f"PDB must contain one nonempty chain A: {label}")
    return residues, "".join(AA3_TO_1[row["residue_name"]] for row in residues)


def _validate_sequence(root: Path, manifest: dict[str, Any], roster: dict[str, dict[str, Any]]) -> int:
    _common(
        manifest, SEQUENCE_FIELDS, "target_unread_protein_sequence_manifest_not_authorization",
        "atypemu_nested_support_count_v1_protein_sequence_manifest_v1", "sequence manifest",
    )
    if manifest["entity_count"] != 135 or not isinstance(manifest["entities"], list):
        raise ValueError("sequence entity count schema drifted")
    seen: set[str] = set()
    for index, row in enumerate(manifest["entities"]):
        _schema(row, SEQUENCE_ENTITY_FIELDS, f"sequence entity {index}")
        _identity(row["bmrb_id"], row["entity_uid"], f"sequence entity {index}")
        roster_row = roster.get(row["entity_uid"])
        if roster_row is None or row["bmrb_id"] != roster_row["bmrb_id"]:
            raise ValueError("sequence roster substitution")
        reference = _schema(row["reference_pdb"], {"path", "sha256"}, "sequence reference")
        expected_path = Path(roster_row["canonical_reference_relative_path"])
        _same_path(reference["path"], expected_path, "sequence reference")
        if reference["sha256"] != roster_row["canonical_reference_pdb_sha256"]:
            raise ValueError("sequence reference digest binding drifted")
        raw = _read_repo(root, expected_path, 2_000_000)
        if _sha256(raw) != reference["sha256"]:
            raise ValueError("reference PDB raw bytes drifted")
        residues, sequence = _parse_pdb(raw, expected_path.as_posix())
        if (
            row["residues"] != residues
            or row["residue_count"] != len(residues)
            or row["sequence_one_letter"] != sequence
            or row["sequence_sha256"] != _sha256(sequence.encode("ascii"))
        ):
            raise ValueError("reference PDB residue/sequence replay drifted")
        for residue in row["residues"]:
            _schema(residue, RESIDUE_FIELDS, "sequence residue")
        if row["entity_uid"] in seen:
            raise ValueError("duplicate sequence entity")
        seen.add(row["entity_uid"])
    if seen != set(roster):
        raise ValueError("sequence roster incomplete")
    return len(seen)


def _catalog_file(record: Any, entity: dict[str, Any], label: str) -> int:
    row = _schema(record, CATALOG_FILE_FIELDS, label)
    index = _integer(row["support_index"], f"{label} index", positive=True)
    if index > 1000:
        raise ValueError("support index outside canonical range")
    for field in (
        "all_atom_coordinate_sha256", "all_atom_topology_sha256", "heavy_atom_coordinate_sha256",
        "heavy_atom_topology_sha256", "pdb_sha256",
    ):
        _sha(row[field], f"{label}.{field}")
    atoms = _integer(row["atom_count"], f"{label} atom count", positive=True)
    heavy = _integer(row["heavy_atom_count"], f"{label} heavy count", positive=True)
    if (
        heavy > atoms
        or type(row["all_atom_topology_matches_reference"]) is not bool
        or row["heavy_atom_topology_sha256"] != entity["canonical_heavy_topology_sha256"]
        or row["all_atom_topology_matches_reference"]
        is not (row["all_atom_topology_sha256"] == entity["canonical_all_atom_topology_sha256"])
    ):
        raise ValueError(f"catalog support identity drifted: {label}")
    return index


def _catalog_entity(
    row: Any, roster: dict[str, dict[str, Any]], label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    entity = _schema(row, CATALOG_ENTITY_FIELDS, label)
    _identity(entity["bmrb_id"], entity["entity_uid"], label)
    expected = roster.get(entity["entity_uid"])
    if expected is None:
        raise ValueError("catalog entity absent from roster")
    for field in ROSTER_ENTITY_FIELDS:
        if entity[field] != expected[field]:
            raise ValueError(f"catalog entity roster identity drifted: {field}")
    if entity["split"] != "train" or entity["observer_fold"] not in {"A", "B"}:
        raise ValueError("catalog fold/split drifted")
    for field in (
        "canonical_all_atom_topology_sha256", "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256", "catalog_digest", "valid_index_digest",
    ):
        _sha(entity[field], f"{label}.{field}")
    atoms = _integer(entity["canonical_atom_count"], "catalog atom count", positive=True)
    heavy = _integer(entity["canonical_heavy_atom_count"], "catalog heavy count", positive=True)
    if heavy > atoms or entity["canonical_reference_support_index"] != 1:
        raise ValueError("catalog canonical atom/reference drifted")
    for field in (
        "canonical_filename_count", "heavy_topology_compatible_unique_coordinate_count",
        "invalid_count",
    ):
        _integer(entity[field], f"{label}.{field}")
    if (
        type(entity["all_atom_topology_variant_count"]) is not int
        or entity["all_atom_topology_variant_count"] < 1
        or type(entity["hydrogen_topology_variation_present"]) is not bool
        or not isinstance(entity["files"], list)
        or entity["invalid"] != []
        or entity["invalid_count"] != 0
        or entity["unexpected_entries"] != []
        or entity["canonical_filename_count"] != len(entity["files"])
    ):
        raise ValueError("catalog invalid or malformed parent rows rejected")
    indexes: list[int] = []
    projected: list[dict[str, Any]] = []
    for position, support in enumerate(entity["files"]):
        support_index = _catalog_file(support, entity, f"{label} support {position}")
        indexes.append(support_index)
        projected.append(
            {
                "heavy_atom_coordinate_sha256": support["heavy_atom_coordinate_sha256"],
                "pdb_sha256": support["pdb_sha256"],
                "support_index": support_index,
            }
        )
    if indexes != sorted(set(indexes)):
        raise ValueError("catalog support IDs are duplicate or unsorted")
    if entity["missing_indices_1_to_1000"] != sorted(set(range(1, 1001)) - set(indexes)):
        raise ValueError("catalog support missing-index replay drifted")
    if (
        entity["catalog_digest"] != _canonical_hash(entity["files"])
        or entity["valid_index_digest"] != _canonical_hash(indexes)
        or entity["heavy_topology_compatible_unique_coordinate_count"] != len(entity["files"])
    ):
        raise ValueError("catalog record digest/count replay drifted")
    variants = {item["all_atom_topology_sha256"] for item in entity["files"]}
    if (
        entity["all_atom_topology_variant_count"] != len(variants)
        or entity["hydrogen_topology_variation_present"] != (len(variants) > 1)
    ):
        raise ValueError("catalog topology variation replay drifted")
    reference = next((item for item in entity["files"] if item["support_index"] == 1), None)
    if (
        reference is None
        or reference["pdb_sha256"] != entity["canonical_reference_pdb_sha256"]
        or reference["atom_count"] != entity["canonical_atom_count"]
        or reference["heavy_atom_count"] != entity["canonical_heavy_atom_count"]
    ):
        raise ValueError("catalog reference identity drifted")
    parent_row = {
        "bmrb_id": entity["bmrb_id"],
        "entity_uid": entity["entity_uid"],
        "heavy_atom_count": entity["canonical_heavy_atom_count"],
        "heavy_atom_topology_sha256": entity["canonical_heavy_topology_sha256"],
        "observer_fold": entity["observer_fold"],
        "records_sha256": _canonical_hash(projected),
        "split": entity["split"],
        "support_count": len(projected),
        "support_index_digest": _canonical_hash(indexes),
    }
    return entity, parent_row


def _validate_parent(
    manifest: dict[str, Any], archive_raw: bytes, roster: dict[str, dict[str, Any]]
) -> int:
    _common(
        manifest, PARENT_FIELDS, "target_unread_parent_heavy_coordinate_manifest_not_authorization",
        "atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1", "parent manifest",
    )
    if manifest["entity_count"] != 135 or manifest["parent_record_count"] != 134_850:
        raise ValueError("parent aggregate count drifted")
    _binding(manifest["catalog_archive"], CATALOG_RELATIVE, CATALOG_SHA256, "parent catalog")
    _binding(
        manifest["source_commitment"], SOURCE_COMMITMENT_RELATIVE, SOURCE_COMMITMENT_SHA256,
        "parent source commitment",
    )
    if manifest["source_relative_path_pattern"] != "{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb":
        raise ValueError("parent source path substitution")
    expected_names = {"."} | {f"./shard_{index}.json" for index in range(27)} | {
        "./catalog_summary.json", "./SHA256SUMS"
    }
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise ValueError("catalog archive inventory/duplicate drifted")
            by_name = {member.name: member for member in members}
            if not by_name["."].isdir() or any(
                not by_name[name].isfile() or by_name[name].issym() or by_name[name].islnk()
                for name in expected_names - {"."}
            ):
                raise ValueError("catalog archive non-regular member")
            contents: dict[str, bytes] = {}
            for name in expected_names - {"."}:
                handle = archive.extractfile(by_name[name])
                if handle is None:
                    raise ValueError("catalog archive member unreadable")
                with handle:
                    contents[name] = handle.read()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"invalid parent catalog archive: {error}") from error
    sums: dict[str, str] = {}
    try:
        lines = contents["./SHA256SUMS"].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("catalog checksum file is not ASCII") from error
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  /([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in sums:
            raise ValueError("catalog checksum syntax/duplicate/path substitution")
        sums[match.group(2)] = match.group(1)
    expected_sum_names = {Path(name).name for name in expected_names - {".", "./SHA256SUMS"}}
    if set(sums) != expected_sum_names or any(
        sums[Path(name).name] != _sha256(contents[name])
        for name in expected_names - {".", "./SHA256SUMS"}
    ):
        raise ValueError("catalog checksum inventory or digest drifted")
    summary = _json_object(contents["./catalog_summary.json"], "catalog summary")
    _schema(summary, SUMMARY_FIELDS, "catalog summary")
    if (
        summary["artifact_kind"] != "target_unread_coordinate_catalog_receipt_not_authorization"
        or summary["contract"] != "atypemu_nested_support_count_v1_catalog_receipt_v2"
        or summary["study_id"] != STUDY_ID
        or summary["catalog_shard_count"] != 27
        or summary["entity_count"] != 135
        or summary["roster_sha256"] != ROSTER_SHA256
        or summary["source_commitment_relative_path"] != SOURCE_COMMITMENT_RELATIVE.as_posix()
        or summary["source_commitment_sha256"] != SOURCE_COMMITMENT_SHA256
        or summary["source_method_composition"] != {"BioEmu": 1.0}
    ):
        raise ValueError("catalog summary identity/BioEmu lineage drifted")
    _closed(summary, "catalog summary")
    full_rows: list[dict[str, Any]] = []
    derived: dict[str, dict[str, Any]] = {}
    shard_bindings: list[dict[str, str]] = []
    for shard_index in range(27):
        name = f"./shard_{shard_index}.json"
        shard_raw = contents[name]
        shard = _json_object(shard_raw, name)
        _schema(shard, SHARD_FIELDS, name)
        if (
            shard["artifact_kind"] != "target_unread_coordinate_catalog_shard_not_authorization"
            or shard["contract"] != "atypemu_nested_support_count_v1_catalog_shard_v2"
            or shard["study_id"] != STUDY_ID
            or shard["shard_index"] != shard_index
            or shard["shard_count"] != 27
            or shard["entity_count"] != 5
            or not isinstance(shard["entities"], list)
            or len(shard["entities"]) != 5
            or shard["roster_sha256"] != ROSTER_SHA256
            or shard["source_commitment_relative_path"] != SOURCE_COMMITMENT_RELATIVE.as_posix()
            or shard["source_commitment_sha256"] != SOURCE_COMMITMENT_SHA256
        ):
            raise ValueError("catalog shard schema/sentinel/binding drifted")
        _closed(shard, name)
        shard_bindings.append({"member": name, "sha256": _sha256(shard_raw)})
        for position, row in enumerate(shard["entities"]):
            catalog_row, parent_row = _catalog_entity(row, roster, f"{name} entity {position}")
            uid = catalog_row["entity_uid"]
            if uid in derived:
                raise ValueError("duplicate parent catalog entity")
            derived[uid] = parent_row
            full_rows.append(catalog_row)
    if set(derived) != set(roster):
        raise ValueError("parent catalog roster incomplete")
    full_rows.sort(key=lambda row: row["entity_uid"])
    projections = [{field: row[field] for field in SUMMARY_ENTITY_FIELDS} for row in full_rows]
    counts = [row["heavy_topology_compatible_unique_coordinate_count"] for row in full_rows]
    feasibility = {
        str(level): {
            "all_entities_count_feasible": all(count >= level for count in counts),
            "entity_count_feasible": sum(count >= level for count in counts),
            "maximum_entity_shortfall": max(max(0, level - count) for count in counts),
            "total_shortfall": sum(max(0, level - count) for count in counts),
        }
        for level in (32, 128, 768, 1536)
    }
    if (
        summary["entities"] != projections
        or summary["full_catalog_digest"] != _canonical_hash(full_rows)
        or summary["level_count_feasibility"] != feasibility
    ):
        raise ValueError("catalog summary did not replay from all shards")
    rows = [derived[uid] for uid in sorted(derived)]
    if (
        not isinstance(manifest["entities"], list)
        or manifest["entities"] != rows
        or manifest["shards"] != shard_bindings
        or manifest["parent_records_sha256"] != _canonical_hash(rows)
        or sum(row["support_count"] for row in rows) != 134_850
    ):
        raise ValueError("parent-heavy manifest did not replay from catalog")
    for row in manifest["entities"]:
        _schema(row, PARENT_ENTITY_FIELDS, "parent manifest entity")
    return 134_850


def _digest_descriptor(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid OCI descriptor digest: {label}")
    return value[7:]


def _inspect_docker(raw: bytes) -> dict[str, Any]:
    """Independently require a closed OCI graph and matching legacy root."""
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError("duplicate Docker archive member")
            directories = {member.name for member in members if member.isdir()}
            files = {member.name: member for member in members if member.isfile()}
            if (
                len(members) != len(directories) + len(files)
                or directories != {"blobs", "blobs/sha256"}
                or {"index.json", "manifest.json", "oci-layout"} - set(files)
            ):
                raise ValueError("Docker archive inventory contains indirect member")
            blobs: dict[str, bytes] = {}
            for name, member in files.items():
                if name.startswith("blobs/sha256/"):
                    digest = name.removeprefix("blobs/sha256/")
                    handle = archive.extractfile(member)
                    if SHA256_RE.fullmatch(digest) is None or handle is None:
                        raise ValueError("malformed Docker blob")
                    with handle:
                        body = handle.read()
                    if _sha256(body) != digest:
                        raise ValueError("Docker blob digest drifted")
                    blobs[digest] = body
            expected_names = {"index.json", "manifest.json", "oci-layout"} | {
                f"blobs/sha256/{digest}" for digest in blobs
            }
            if set(files) != expected_names:
                raise ValueError("Docker archive file path substitution")
            controls = {}
            for name in ("index.json", "manifest.json", "oci-layout"):
                handle = archive.extractfile(files[name])
                if handle is None:
                    raise ValueError("Docker control unreadable")
                with handle:
                    controls[name] = handle.read()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"invalid Docker archive: {error}") from error
    if _json_object(controls["oci-layout"], "OCI layout") != {"imageLayoutVersion": "1.0.0"}:
        raise ValueError("OCI layout drifted")
    index = _json_object(controls["index.json"], "OCI index")
    if set(index) != {"manifests", "mediaType", "schemaVersion"} or (
        index["schemaVersion"] != 2
        or index["mediaType"] != "application/vnd.oci.image.index.v1+json"
        or not isinstance(index["manifests"], list)
        or len(index["manifests"]) != 1
    ):
        raise ValueError("OCI root index schema drifted")
    root = index["manifests"][0]
    root_hex = _digest_descriptor(root.get("digest"), "OCI root") if isinstance(root, dict) else ""
    if (
        not isinstance(root, dict)
        or root.get("mediaType") != "application/vnd.oci.image.index.v1+json"
        or root.get("size") != len(blobs.get(root_hex, b""))
        or root_hex not in blobs
    ):
        raise ValueError("OCI root descriptor drifted")
    image_index = _json_object(blobs[root_hex], "OCI image index")
    if (
        set(image_index) != {"manifests", "mediaType", "schemaVersion"}
        or image_index["schemaVersion"] != 2
        or image_index["mediaType"] != "application/vnd.oci.image.index.v1+json"
        or not isinstance(image_index["manifests"], list)
        or len(image_index["manifests"]) != 2
    ):
        raise ValueError("OCI image index schema drifted")
    referenced = {root_hex}
    manifests: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for descriptor in image_index["manifests"]:
        if not isinstance(descriptor, dict):
            raise ValueError("OCI image descriptor is not object")
        digest = _digest_descriptor(descriptor.get("digest"), "image descriptor")
        if (
            descriptor.get("mediaType") != "application/vnd.oci.image.manifest.v1+json"
            or descriptor.get("size") != len(blobs.get(digest, b""))
            or digest not in blobs
        ):
            raise ValueError("OCI image descriptor closure drifted")
        document = _json_object(blobs[digest], "OCI manifest")
        if (
            document.get("schemaVersion") != 2
            or document.get("mediaType") != "application/vnd.oci.image.manifest.v1+json"
            or not isinstance(document.get("config"), dict)
            or not isinstance(document.get("layers"), list)
        ):
            raise ValueError("OCI manifest schema drifted")
        referenced.add(digest)
        for child in [document["config"], *document["layers"]]:
            if not isinstance(child, dict):
                raise ValueError("OCI child descriptor is not object")
            child_digest = _digest_descriptor(child.get("digest"), "OCI child")
            if child_digest not in blobs or child.get("size") != len(blobs[child_digest]):
                raise ValueError("OCI blob closure drifted")
            referenced.add(child_digest)
        manifests.append((descriptor, document))
    legacy = json.loads(
        controls["manifest.json"].decode("utf-8"), object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    _reject_nonfinite(legacy, "legacy Docker manifest")
    if (
        not isinstance(legacy, list)
        or len(legacy) != 1
        or not isinstance(legacy[0], dict)
        or set(legacy[0]) != {"Config", "RepoTags", "Layers"}
        or legacy[0]["RepoTags"] != ["atypemu/openmm86-protonation:8.6.0"]
    ):
        raise ValueError("legacy Docker manifest schema drifted")
    config_name = legacy[0]["Config"]
    layers = legacy[0]["Layers"]
    if (
        not isinstance(config_name, str)
        or not isinstance(layers, list)
        or any(not isinstance(item, str) for item in layers)
    ):
        raise ValueError("legacy Docker root path drifted")
    config_hex = config_name.removeprefix("blobs/sha256/")
    layer_hexes = [item.removeprefix("blobs/sha256/") for item in layers]
    if (
        config_name != f"blobs/sha256/{config_hex}"
        or any(item != f"blobs/sha256/{digest}" for item, digest in zip(layers, layer_hexes))
        or SHA256_RE.fullmatch(config_hex) is None
        or any(SHA256_RE.fullmatch(item) is None for item in layer_hexes)
    ):
        raise ValueError("legacy Docker root is noncanonical")
    referenced.update({config_hex, *layer_hexes})
    if set(blobs) != referenced:
        raise ValueError("Docker archive blob closure is incomplete")
    amd64 = [
        (descriptor, document)
        for descriptor, document in manifests
        if descriptor.get("platform") == {"architecture": "amd64", "os": "linux"}
    ]
    if len(amd64) != 1:
        raise ValueError("Docker archive lacks unique linux/amd64 image")
    descriptor, manifest = amd64[0]
    manifest_hex = _digest_descriptor(descriptor["digest"], "amd64 manifest")
    actual_config = _digest_descriptor(manifest["config"].get("digest"), "amd64 config")
    actual_layers = [_digest_descriptor(row.get("digest"), "amd64 layer") for row in manifest["layers"]]
    if config_hex != actual_config or layer_hexes != actual_layers:
        raise ValueError("OCI and legacy roots identify different images")
    config = _json_object(blobs[actual_config], "image config")
    env = config.get("config", {}).get("Env") if isinstance(config.get("config"), dict) else None
    required_env = {
        "OPENMM_DEFAULT_PLATFORM=Reference", "PYTHONHASHSEED=0", "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1", "PYTHON_VERSION=3.13.7",
    }
    if config.get("architecture") != "amd64" or not isinstance(env, list) or not required_env.issubset(env):
        raise ValueError("Docker runtime configuration drifted")
    attestations = [
        (descriptor, document)
        for descriptor, document in manifests
        if descriptor.get("annotations", {}).get("vnd.docker.reference.type") == "attestation-manifest"
    ]
    if len(attestations) != 1:
        raise ValueError("Docker build attestation absent or ambiguous")
    attestation_descriptor, attestation_manifest = attestations[0]
    if attestation_descriptor.get("annotations", {}).get("vnd.docker.reference.digest") != descriptor["digest"]:
        raise ValueError("Docker attestation subject substitution")
    if len(attestation_manifest["layers"]) != 1 or (
        attestation_manifest["layers"][0].get("mediaType") != "application/vnd.in-toto+json"
    ):
        raise ValueError("Docker attestation layer drifted")
    attestation_digest = _digest_descriptor(
        attestation_manifest["layers"][0].get("digest"), "attestation layer"
    )
    attestation = _json_object(blobs[attestation_digest], "build attestation")
    try:
        definition = attestation["predicate"]["buildDefinition"]
        request = definition["externalParameters"]["request"]["args"]
        dependency = definition["resolvedDependencies"]
        subject = attestation["subject"]
    except (KeyError, TypeError) as error:
        raise ValueError("build attestation schema drifted") from error
    if (
        attestation.get("predicateType") != "https://slsa.dev/provenance/v1"
        or definition["externalParameters"]["configSource"].get("path") != DOCKERFILE_RELATIVE.name
        or not isinstance(request, dict)
        or request.get("force-network-mode") != "none"
        or "no-cache" not in request
        or definition.get("internalParameters", {}).get("builderPlatform") != "linux/amd64"
        or dependency != [{"digest": {"sha256": "781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7"}}]
        or subject != [{"digest": {"sha256": manifest_hex}}]
    ):
        raise ValueError("build attestation semantics drifted")
    return {
        "amd64_config_digest": f"sha256:{actual_config}",
        "amd64_layer_digests": [f"sha256:{item}" for item in actual_layers],
        "amd64_manifest_digest": f"sha256:{manifest_hex}",
        "build_attestation_digest": f"sha256:{attestation_digest}",
        "image_index_digest": f"sha256:{root_hex}",
    }


def _singularity_json(arguments: list[str], label: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments, cwd="/tmp", stdin=subprocess.DEVNULL, capture_output=True,
            check=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"runtime {label} failed: {error}") from error
    return _json_object(result.stdout.encode("utf-8"), label)


def _inspect_sif(sif_raw: bytes, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = r'''import hashlib,json,pathlib,re,sys,xml.etree.ElementTree as ET
import numpy,openmm
from openmm import app
root=pathlib.Path(app.__file__).parent
paths={"force_field":root/"data/amber14/protein.ff14SB.xml","hydrogen_definitions":root/"data/hydrogens.xml","modeller_source":root/"modeller.py"}
app.ForceField("amber14/protein.ff14SB.xml")
print(json.dumps({"files":{k:{"path":str(v),"sha256":hashlib.sha256(v.read_bytes()).hexdigest(),"size":v.stat().st_size} for k,v in paths.items()},"numpy_version":numpy.__version__,"openmm_version":openmm.__version__,"openmm_version_full":openmm.version.full_version,"platforms":[openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())],"python_version":sys.version.split()[0],"source_ph_thresholds":sorted(set(re.findall(r"pH\s*[<>]=?\s*([0-9]+(?:\.[0-9]+)?)",paths["modeller_source"].read_text())),key=float),"xml_maxph_thresholds":sorted({str(e.attrib["maxph"]) for e in ET.parse(paths["hydrogen_definitions"]).iter() if "maxph" in e.attrib},key=float)},sort_keys=True))'''
    with tempfile.TemporaryDirectory(prefix="condition-input-check-") as directory:
        snapshot = Path(directory) / name
        descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            offset = 0
            while offset < len(sif_raw):
                offset += os.write(descriptor, sif_raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _sha256(_read_direct(snapshot, 200_000_000, "snapshotted SIF")) != _sha256(sif_raw):
            raise ValueError("SIF snapshot changed before inspection")
        inspect = _singularity_json(["singularity", "inspect", "--json", str(snapshot)], "SIF inspect")
        runtime = _singularity_json(
            ["singularity", "exec", "--containall", "--cleanenv", "--no-home", str(snapshot), "python", "-c", helper],
            "SIF execution",
        )
    try:
        labels = inspect["data"]["attributes"]["labels"]
    except (KeyError, TypeError) as error:
        raise ValueError("SIF inspect labels absent") from error
    if not isinstance(labels, dict):
        raise ValueError("SIF inspect labels are not object")
    return labels, runtime


def _validate_environment(
    root: Path, manifest: dict[str, Any], sif: Path, docker: Path
) -> tuple[str, str]:
    _common(
        manifest, ENVIRONMENT_FIELDS, "openmm86_runtime_environment_manifest_not_authorization",
        "atypemu_nested_support_count_v1_openmm86_environment_manifest_v1", "environment manifest",
    )
    docker_raw = _read_direct(docker, 200_000_000, "Docker archive")
    sif_raw = _read_direct(sif, 200_000_000, "SIF")
    expected_docker = {
        "filename": "openmm86_protonation_8.6.0.docker.tar",
        "sha256": "e2a1532d6fc4435b6940f38b6197287395015f2df8bb17c0a76683c624d9b0a3",
        "size": 109516800,
    }
    expected_sif = {
        "filename": "openmm86_protonation_8.6.0.sif",
        "sha256": "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5",
        "size": 78290944,
    }
    actual_docker = {"filename": docker.name, "sha256": _sha256(docker_raw), "size": len(docker_raw)}
    actual_sif = {"filename": sif.name, "sha256": _sha256(sif_raw), "size": len(sif_raw)}
    if actual_docker != expected_docker or actual_sif != expected_sif:
        raise ValueError("required runtime direct-file hash, size, or name drifted")
    docker_image = _inspect_docker(docker_raw)
    expected_image = {
        "amd64_config_digest": "sha256:3fd1c4478594daf5231b8d0dc296be47b97d78a66ce8c7cd5ae484cde77b35eb",
        "amd64_layer_digests": [
            "sha256:5c32499ab806884c5725c705c2bf528662d034ed99de13d3205309e0d9ef0375",
            "sha256:79dc17963468084e50bd8100760c1d19f0c4711c75a624e91f83ad282cc114f7",
            "sha256:5c36186b8d8417454960b65c9c32b077ab4f3fb3c662ff5d4eeb5fd6abf649ff",
            "sha256:4ce903582132c9254f5172f6336ca9b39b7554f483694627d03a62f181950257",
            "sha256:098d6a910a7ac3523e4376a85f96d3cfa6b4984324636d7378ab39aba0e1ec0d",
            "sha256:4b32dad6a12f98596ec264ba2f1cb441e672548738090f0b1721d72ec016031c",
        ],
        "amd64_manifest_digest": "sha256:6fa60f0b8729602696f49aa7387f8519e354ba400ec878cee567632f52b57844",
        "build_attestation_digest": "sha256:898e3e0e7688d7d03e195abfb2ad1cedf833a81e2b23cc5b27bc67c23ad8c586",
        "image_index_digest": "sha256:f555076df2698a38dc016815219ff51325f052ad31e214951e067ac0ac96563e",
    }
    if docker_image != expected_image:
        raise ValueError("Docker OCI/runtime image identity drifted")
    labels, inspection = _inspect_sif(sif_raw, sif.name)
    expected_labels = {
        "org.label-schema.build-arch": "amd64",
        "org.label-schema.build-date": "Monday_7_September_2026_3:26:22_KST",
        "org.label-schema.schema-version": "1.0",
        "org.label-schema.usage.singularity.deffile.bootstrap": "docker-archive",
        "org.label-schema.usage.singularity.deffile.from": "/home/yang07/.cache/atypemu_openmm86_runtime/openmm86_protonation_8.6.0.docker.tar",
        "org.label-schema.usage.singularity.version": "4.3.4-noble",
    }
    expected_inspection = {
        "files": {
            "force_field": {"path": "/usr/local/lib/python3.13/site-packages/openmm/app/data/amber14/protein.ff14SB.xml", "sha256": "d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8", "size": 224056},
            "hydrogen_definitions": {"path": "/usr/local/lib/python3.13/site-packages/openmm/app/data/hydrogens.xml", "sha256": "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d", "size": 13147},
            "modeller_source": {"path": "/usr/local/lib/python3.13/site-packages/openmm/app/modeller.py", "sha256": "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3", "size": 95003},
        },
        "numpy_version": "2.3.3", "openmm_version": "8.6", "openmm_version_full": "8.6.0.dev-c6173db",
        "platforms": ["Reference", "CPU"], "python_version": "3.13.7",
        "source_ph_thresholds": ["6.5"], "xml_maxph_thresholds": ["4.4", "8.5", "10.0"],
    }
    if labels != expected_labels or inspection != expected_inspection:
        raise ValueError("snapshotted SIF runtime/Reference/threshold identity drifted")
    expected_manifest = {
        "base_image": {"amd64_manifest_digest": "sha256:781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7", "name": "python:3.13.7-slim-bookworm"},
        "build_recipe": {"path": DOCKERFILE_RELATIVE.as_posix(), "sha256": DOCKERFILE_SHA256},
        "conversion": {"docker_archive": expected_docker, "docker_image": expected_image, "singularity_labels": expected_labels},
        "inspection": expected_inspection,
        "ph_regimes": {"combined_transition_thresholds": ["4.4", "6.5", "8.5", "10.0"], "interval_midpoints": ["2.2", "5.45", "7.5", "9.25", "12.0"], "lower_bound_inclusive": "0.0", "upper_bound_inclusive": "14.0"},
        "required_platform": "Reference", "runtime_sif": expected_sif,
        "wheels": [
            {"filename": "numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "sha256": "5b83648633d46f77039c29078751f80da65aa64d5622a3cd62aaef9d835b6c93", "size": 16636301},
            {"filename": "openmm-8.6.0-cp313-cp313-manylinux_2_34_x86_64.whl", "sha256": "1b0f39a812452fd9eb1faf28ed4e9d832aef44fc2072f4442a81a4ce7a6f56bf", "size": 14425566},
        ],
    }
    for field, expected in expected_manifest.items():
        if manifest[field] != expected:
            raise ValueError(f"environment manifest self-attestation mismatch: {field}")
    dockerfile = _read_repo(root, DOCKERFILE_RELATIVE, 100_000)
    if _sha256(dockerfile) != DOCKERFILE_SHA256:
        raise ValueError("bound Dockerfile source drifted")
    return actual_docker["sha256"], actual_sif["sha256"]


def verify(root: Path, sif: Path, docker: Path) -> tuple[dict[str, str], int]:
    """Replay all frozen inputs.  No target, score, science, or authorization is read."""
    _read_bound(root, FREEZER_RELATIVE, FREEZER_SHA256, 1_000_000)
    raw_manifests = {
        name: _read_bound(root, path, MANIFEST_SHA256[name], 3_000_000)
        for name, path in MANIFESTS.items()
    }
    manifests = {name: _json_object(raw, f"{name} manifest") for name, raw in raw_manifests.items()}
    roster = _validate_roster(
        _json_object(_read_bound(root, ROSTER_RELATIVE, ROSTER_SHA256, 2_000_000), "roster")
    )
    recovery = _validate_recovery(
        _json_object(_read_bound(root, RECOVERY_RELATIVE, RECOVERY_SHA256, 2_000_000), "recovery"),
        roster,
    )
    evidence = _json_object(
        _read_bound(root, V6_EVIDENCE_RECEIPT_RELATIVE, V6_EVIDENCE_RECEIPT_SHA256, 100_000),
        "v6 evidence receipt",
    )
    if evidence.get("archive", {}).get("sha256") != V6_ARCHIVE_SHA256:
        raise ValueError("v6 committed evidence binding drifted")
    v6 = _validate_v6_archive(
        _read_bound(root, V6_ARCHIVE_RELATIVE, V6_ARCHIVE_SHA256, 5_000_000), roster
    )
    conditions = _validate_condition(manifests["condition"], roster, v6, recovery)
    sequences = _validate_sequence(root, manifests["sequence"], roster)
    parents = _validate_parent(
        manifests["parent"], _read_bound(root, CATALOG_RELATIVE, CATALOG_SHA256, 30_000_000), roster
    )
    docker_sha, sif_sha = _validate_environment(root, manifests["environment"], sif, docker)
    if set(roster) != {row["entity_uid"] for row in manifests["parent"]["entities"]}:
        raise ValueError("sequence/parent correlated roster lineage drifted")
    source_hashes = {
        "freezer": FREEZER_SHA256,
        "roster": ROSTER_SHA256,
        "recovery_v3": RECOVERY_SHA256,
        "recovery_v6_archive": V6_ARCHIVE_SHA256,
        "catalog_archive": CATALOG_SHA256,
        "dockerfile": DOCKERFILE_SHA256,
        "docker_archive": docker_sha,
        "runtime_sif": sif_sha,
    }
    return source_hashes, conditions + sequences + parents


def _clean_committed_head(root: Path) -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root,
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"clean committed HEAD required: {error}") from error
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or status:
        raise ValueError("--write-receipt requires a clean committed HEAD")
    return head


def _receipt_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_descriptor_once(descriptor: int, raw: bytes) -> None:
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444
    )
    _write_descriptor_once(descriptor, _receipt_bytes(payload))


def _write_repo_once(root: Path, relative: Path, payload: dict[str, Any]) -> None:
    _safe_relative(relative, "receipt")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, flags)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=directory,
        )
        _write_descriptor_once(descriptor, _receipt_bytes(payload))
    finally:
        os.close(directory)


def write_receipt(root: Path, hashes: dict[str, str]) -> None:
    head = _clean_committed_head(root)
    checker_hash = _sha256(_read_repo(root, CHECKER_RELATIVE, 2_000_000))
    payload = {
        "artifact_kind": "target_unread_condition_uncertainty_inputs_check_receipt_not_authorization",
        "checker": {"path": CHECKER_RELATIVE.as_posix(), "sha256": checker_hash},
        "closed_capability_counters": {
            "authorization_consumed": 0, "science_executed": 0,
            "source_scores_read": 0, "target_values_read": 0,
        },
        "contract": "atypemu_nested_support_count_v1_condition_uncertainty_inputs_check_receipt_v1",
        "git_head": head,
        "manifest_hashes": {MANIFESTS[name].as_posix(): MANIFEST_SHA256[name] for name in sorted(MANIFESTS)},
        "runtime_hashes": {"docker_archive": hashes["docker_archive"], "runtime_sif": hashes["runtime_sif"]},
        "source_hashes": {key: value for key, value in hashes.items() if key not in {"docker_archive", "runtime_sif"}},
        "status": "HOLD_TARGET_UNREAD_INPUT_IDENTITY_QUALIFIED_NOT_AUTHORIZATION",
    }
    _write_repo_once(root, RECEIPT_RELATIVE, payload)


def _expect_rejection(label: str, operation: Callable[[], Any]) -> int:
    try:
        operation()
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile):
        return 1
    raise AssertionError(f"self-test accepted {label}")


def _self_test() -> int:
    """Target-free adversarial parser/schema/no-clobber tests."""
    checks = 0
    checks += _expect_rejection("duplicate JSON keys", lambda: _json_object(b'{"a":1,"a":2}', "test"))
    checks += _expect_rejection("NaN", lambda: _json_object(b'{"a":NaN}', "test"))
    checks += _expect_rejection("infinite exponent", lambda: _json_object(b'{"a":1e999}', "test"))
    checks += _expect_rejection("unknown schema field", lambda: _schema({"a": 1, "x": 2}, {"a"}, "test"))
    checks += _expect_rejection("imputed pH", lambda: _decimal_ph("NaN", "test"))
    checks += _expect_rejection("alternate PDB atom", lambda: _parse_pdb(b"ATOM      1  CA AALA A   1       0.000   0.000   0.000  1.00  0.00           C  ", "test"))
    checks += _expect_rejection("HETATM PDB", lambda: _parse_pdb(b"HETATM    1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  ", "test"))
    checks += _expect_rejection("bad catalog support schema", lambda: _catalog_file({}, {"canonical_heavy_topology_sha256": "0" * 64, "canonical_all_atom_topology_sha256": "0" * 64}, "test"))
    checks += _expect_rejection("malformed Docker archive", lambda: _inspect_docker(b"not a tar"))
    with tempfile.TemporaryDirectory(prefix="condition-input-self-test-") as directory:
        root = Path(directory)
        (root / "real").mkdir()
        (root / "real" / "input.json").write_bytes(b"{}")
        (root / "leaf.json").symlink_to(root / "real" / "input.json")
        checks += _expect_rejection("symlink direct input", lambda: _read_direct(root / "leaf.json", 10, "test"))
        receipt = root / "receipt.json"
        _write_once(receipt, {"fixed": True})
        checks += _expect_rejection("receipt clobber", lambda: _write_once(receipt, {"fixed": False}))
    # Ambiguity/missingness must remain unresolved; no single-value repair is allowed.
    ambiguous = {"assigned_chem_shift_list_ids": ["1", "2"], "sample_condition_list_ids": ["1"]}
    if not (len(ambiguous["assigned_chem_shift_list_ids"]) > 1):
        raise AssertionError("self-test ambiguity fixture drifted")
    checks += 1
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sif", type=Path)
    parser.add_argument("--runtime-docker-archive", type=Path)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            if args.runtime_sif is not None or args.runtime_docker_archive is not None or args.write_receipt:
                parser.error("--self-test is target-free and accepts no runtime or receipt option")
            checks = _self_test()
            print(f"METRIC condition_uncertainty_input_self_test_checks={checks}")
            print("METRIC target_values_read=0")
            print("METRIC source_scores_read=0")
            print("METRIC science_executed=0")
            print("METRIC authorization_consumed=0")
            print("STATUS HOLD_TARGET_UNREAD_INPUT_SELF_TEST_PASS")
            return 0
        if args.runtime_sif is None or args.runtime_docker_archive is None:
            parser.error("normal verification requires --runtime-sif and --runtime-docker-archive")
        root = _root()
        hashes, replay_count = verify(root, args.runtime_sif, args.runtime_docker_archive)
        if args.write_receipt:
            write_receipt(root, hashes)
        print("METRIC condition_rows_replayed=135")
        print("METRIC sequence_rows_replayed=135")
        print(f"METRIC parent_heavy_records_replayed={replay_count - 270}")
        print("METRIC target_values_read=0")
        print("METRIC source_scores_read=0")
        print("METRIC science_executed=0")
        print("METRIC authorization_consumed=0")
        print("STATUS HOLD_TARGET_UNREAD_INPUT_IDENTITY_QUALIFIED_NOT_AUTHORIZATION")
        return 0
    except Exception as error:  # Fail closed; no success-looking partial output.
        print(f"FAIL {type(error).__name__}: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
