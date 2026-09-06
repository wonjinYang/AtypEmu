"""Freeze target-unread inputs for the HOLD-only protonation proposal."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_condition_uncertainty_protonation_v1"
SCRIPT_RELATIVE = Path("gpuopt/candidates/freeze_condition_uncertainty_inputs.py")
DOCKERFILE_RELATIVE = Path("gpuopt/candidates/openmm86_protonation_runtime.Dockerfile")
ROSTER_RELATIVE = Path(".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json")
SOURCE_COMMITMENT_RELATIVE = Path(
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
)
RECOVERY_RELATIVE = Path(".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json")
V6_EVIDENCE_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json"
)
V6_ARCHIVE_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1.zip"
)
CATALOG_ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/"
    "catalog_v3_shards.tar.gz"
)
OUTPUTS = {
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
        "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json"
    ),
}
EXPECTED_HASHES = {
    ROSTER_RELATIVE: "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9",
    SOURCE_COMMITMENT_RELATIVE: "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    RECOVERY_RELATIVE: "2d8a255add821950e0e381401afc8c27a97d37cffb8ead827f5d8bd500bc3b8b",
    V6_EVIDENCE_RECEIPT_RELATIVE: "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36",
    V6_ARCHIVE_RELATIVE: "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36",
    CATALOG_ARCHIVE_RELATIVE: "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2",
}
FALSE_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHARD_KEYS = {
    "artifact_kind", "authorization_consumed", "contract", "entities",
    "entity_count", "outer_or_formal_metrics_opened", "roster_sha256",
    "science_executed", "shard_count", "shard_index",
    "source_commitment_relative_path", "source_commitment_sha256",
    "source_scores_read", "study_id", "target_values_read",
}
CATALOG_ENTITY_KEYS = {
    "all_atom_topology_variant_count", "bmrb_id", "canonical_all_atom_topology_sha256",
    "canonical_atom_count", "canonical_filename_count", "canonical_heavy_atom_count",
    "canonical_heavy_topology_sha256", "canonical_reference_pdb_sha256",
    "canonical_reference_relative_path", "canonical_reference_support_index",
    "catalog_digest", "entity_uid", "files",
    "heavy_topology_compatible_unique_coordinate_count",
    "hydrogen_topology_variation_present", "invalid", "invalid_count",
    "missing_indices_1_to_1000", "observer_fold", "split", "unexpected_entries",
    "valid_index_digest",
}
CATALOG_FILE_KEYS = {
    "all_atom_coordinate_sha256", "all_atom_topology_matches_reference",
    "all_atom_topology_sha256", "atom_count", "heavy_atom_coordinate_sha256",
    "heavy_atom_count", "heavy_atom_topology_sha256", "pdb_sha256", "support_index",
}
SUMMARY_KEYS = {
    "artifact_kind", "authorization_consumed", "catalog_shard_count", "contract",
    "entities", "entity_count", "full_catalog_digest", "level_count_feasibility",
    "outer_or_formal_metrics_opened", "roster_sha256", "science_executed",
    "source_commitment_relative_path", "source_commitment_sha256",
    "source_method_composition", "source_scores_read", "study_id", "target_values_read",
}
SUMMARY_ENTITY_KEYS = {
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def read_regular(path: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink input rejected: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"invalid input file: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        if len(raw) != before.st_size or identity(before) != identity(after):
            raise ValueError(f"input changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def bound_root() -> Path:
    script = Path(__file__).absolute()
    root = script.parents[2]
    if script != root / SCRIPT_RELATIVE or script.is_symlink() or script.resolve() != script:
        raise ValueError("freezer must run from its canonical repository path")
    return root


def checked_input(root: Path, relative: Path, maximum_bytes: int) -> bytes:
    raw = read_regular(root / relative, maximum_bytes)
    if sha256(raw) != EXPECTED_HASHES[relative]:
        raise ValueError(f"bound input hash drifted: {relative}")
    return raw


def common(kind: str, script_hash: str) -> dict[str, Any]:
    return {
        "artifact_kind": kind,
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": FALSE_CAPABILITIES,
        "freezer": {"path": SCRIPT_RELATIVE.as_posix(), "sha256": script_hash},
        "scope": "target-unread HOLD-only input identity; not support or science evidence",
    }


def decimal_string(value: Any) -> str:
    number = Decimal(str(value))
    if not number.is_finite() or not Decimal("0") <= number <= Decimal("14"):
        raise ValueError(f"invalid pH: {value!r}")
    return format(number, "f")


def v6_receipt(archive_raw: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)) or V6_RECEIPT_MEMBER not in names:
            raise ValueError("v6 evidence archive member inventory drifted")
        info = archive.getinfo(V6_RECEIPT_MEMBER)
        if info.is_dir() or (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
            raise ValueError("v6 receipt member is not a regular file")
        return parse_json(archive.read(info), "v6 receipt")


def make_condition_manifest(
    roster: dict[str, Any], recovery: dict[str, Any], v6: dict[str, Any], script_hash: str
) -> dict[str, Any]:
    if not (
        recovery.get("entity_count") == 135
        and recovery.get("combined_metadata_resolved_entity_count") == 119
        and recovery.get("fallback_metadata_resolved_entity_count") == 4
        and len(recovery.get("entities", [])) == 20
        and recovery.get("closed_capabilities") == FALSE_CAPABILITIES
        and recovery["bound_inputs"]["recovery_v6_archive"]["raw_sha256"]
        == EXPECTED_HASHES[V6_ARCHIVE_RELATIVE]
    ):
        raise ValueError("combined condition receipt semantics drifted")
    if not (
        v6.get("entity_count") == 135
        and len(v6.get("entities", [])) == 135
        and v6.get("ph_feasible_entity_count") == 115
        and v6.get("status") == "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
        and all(v6.get(field) is False for field in (
            "authorization_consumed", "outer_or_formal_metrics_opened", "science_executed",
            "source_construction_executed", "source_scores_read", "target_atom_identities_read",
            "target_values_read",
        ))
    ):
        raise ValueError("v6 condition receipt semantics drifted")
    roster_by_uid = {row["entity_uid"]: row for row in roster["entities"]}
    v6_by_uid = {row["entity_uid"]: row for row in v6["entities"]}
    fallback_by_uid = {row["entity_uid"]: row for row in recovery["entities"]}
    v6_hold_uids = {uid for uid, row in v6_by_uid.items() if not row["ph_feasible"]}
    fallback_resolved = {
        uid for uid, row in fallback_by_uid.items() if row["fallback_metadata_resolved"]
    }
    if (
        len(roster_by_uid) != 135
        or set(v6_by_uid) != set(roster_by_uid)
        or len(fallback_by_uid) != 20
        or set(fallback_by_uid) != v6_hold_uids
        or len(fallback_resolved) != 4
    ):
        raise ValueError("condition and roster identities differ")
    rows = []
    for uid in sorted(roster_by_uid):
        base = v6_by_uid[uid]
        fallback = fallback_by_uid.get(uid)
        if base["bmrb_id"] != roster_by_uid[uid]["bmrb_id"]:
            raise ValueError(f"BMRB identity differs: {uid}")
        if fallback is not None and fallback["bmrb_id"] != base["bmrb_id"]:
            raise ValueError(f"fallback BMRB identity differs: {uid}")
        if base["ph_feasible"]:
            state, ph, source = "observed", decimal_string(base["deposited_ph"]), "recovery_v6_exact"
        elif fallback is not None and fallback["fallback_metadata_resolved"]:
            state, ph, source = "observed", decimal_string(fallback["fallback_ph"]), "unique_assigned_list_exact"
        elif len(base["assigned_chem_shift_list_ids"]) > 1 or len(base["sample_condition_list_ids"]) > 1:
            state, ph, source = "state_ambiguous", None, None
        else:
            state, ph, source = "state_missing", None, None
        rows.append({
            "bmrb_id": base["bmrb_id"],
            "condition_state": state,
            "deposited_ph": ph,
            "entity_uid": uid,
            "pH_source": source,
            "recovery_v6_hold_reasons": base["hold_reasons"],
            "temperature_ionic_diagnostics": base["temperature_ionic_diagnostics"],
        })
    counts = {state: sum(row["condition_state"] == state for row in rows) for state in (
        "observed", "state_missing", "state_ambiguous"
    )}
    if counts != {"observed": 119, "state_missing": 11, "state_ambiguous": 5}:
        raise ValueError(f"condition-state counts drifted: {counts}")
    return {
        **common("target_unread_condition_manifest_not_authorization", script_hash),
        "contract": "atypemu_nested_support_count_v1_condition_manifest_v1",
        "entities": rows,
        "entity_count": 135,
        "roster": {
            "path": ROSTER_RELATIVE.as_posix(),
            "sha256": EXPECTED_HASHES[ROSTER_RELATIVE],
        },
        "state_counts": counts,
        "upstream": {
            "combined_receipt": {"path": RECOVERY_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[RECOVERY_RELATIVE]},
            "recovery_v6_archive": {"path": V6_ARCHIVE_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[V6_ARCHIVE_RELATIVE]},
            "recovery_v6_evidence_receipt": {"path": V6_EVIDENCE_RECEIPT_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[V6_EVIDENCE_RECEIPT_RELATIVE]},
        },
    }


def parse_sequence(raw: bytes) -> tuple[list[dict[str, str]], str]:
    residues: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for line in raw.splitlines():
        if line[:6] not in {b"ATOM  ", b"HETATM"}:
            continue
        if len(line) < 78 or line[:6] != b"ATOM  " or line[16:17] != b" ":
            raise ValueError("reference PDB contains unsupported atom record or alternate location")
        record = {
            "chain_id": line[21:22].decode("ascii"),
            "insertion_code": line[26:27].decode("ascii"),
            "residue_id": line[22:26].decode("ascii").strip(),
            "residue_name": line[17:20].decode("ascii").strip(),
        }
        key = tuple(record[field] for field in ("chain_id", "residue_id", "insertion_code", "residue_name"))
        if key not in seen:
            if record["residue_name"] not in AA3_TO_1:
                raise ValueError(f"nonstandard residue: {record['residue_name']}")
            seen.add(key)
            residues.append(record)
    if not residues or {row["chain_id"] for row in residues} != {"A"}:
        raise ValueError("reference PDB must contain one nonempty chain A")
    sequence = "".join(AA3_TO_1[row["residue_name"]] for row in residues)
    return residues, sequence


def make_sequence_manifest(root: Path, roster: dict[str, Any], script_hash: str) -> dict[str, Any]:
    rows = []
    for entity in sorted(roster["entities"], key=lambda row: row["entity_uid"]):
        relative = Path(entity["canonical_reference_relative_path"])
        raw = read_regular(root / relative, 2_000_000)
        if sha256(raw) != entity["canonical_reference_pdb_sha256"]:
            raise ValueError(f"reference PDB hash drifted: {relative}")
        residues, sequence = parse_sequence(raw)
        rows.append({
            "bmrb_id": entity["bmrb_id"],
            "entity_uid": entity["entity_uid"],
            "reference_pdb": {"path": relative.as_posix(), "sha256": sha256(raw)},
            "residue_count": len(residues),
            "residues": residues,
            "sequence_one_letter": sequence,
            "sequence_sha256": sha256(sequence.encode("ascii")),
        })
    if len(rows) != 135:
        raise ValueError("sequence entity count drifted")
    return {
        **common("target_unread_protein_sequence_manifest_not_authorization", script_hash),
        "contract": "atypemu_nested_support_count_v1_protein_sequence_manifest_v1",
        "entities": rows,
        "entity_count": 135,
        "roster": {"path": ROSTER_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[ROSTER_RELATIVE]},
    }


def make_parent_manifest(archive_raw: bytes, roster: dict[str, Any], script_hash: str) -> dict[str, Any]:
    roster_by_uid = {row["entity_uid"]: row for row in roster["entities"]}
    entities: dict[str, dict[str, Any]] = {}
    full_rows = []
    shard_bindings = []
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        all_members = archive.getmembers()
        expected_shards = {f"./shard_{index}.json" for index in range(27)}
        expected_files = expected_shards | {"./catalog_summary.json", "./SHA256SUMS"}
        expected_members = {"."} | expected_files
        if (
            len(all_members) != len(expected_members)
            or {member.name for member in all_members} != expected_members
            or any(member.name == "." and not member.isdir() for member in all_members)
            or any(member.name != "." and not member.isfile() for member in all_members)
        ):
            raise ValueError("catalog archive member inventory drifted")
        members = [member for member in all_members if member.isfile()]
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("catalog archive has duplicate file members")
        by_name = {member.name: member for member in members}
        member_raw = {
            name: archive.extractfile(by_name[name]).read() for name in expected_files
        }
        checksum_rows = {}
        for line in member_raw["./SHA256SUMS"].decode("ascii").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
            if match is None:
                raise ValueError("invalid catalog SHA256SUMS line")
            basename = Path(match.group(2)).name
            if basename in checksum_rows:
                raise ValueError("duplicate catalog SHA256SUMS basename")
            checksum_rows[basename] = match.group(1)
        expected_checksum_names = {Path(name).name for name in expected_files - {"./SHA256SUMS"}}
        if set(checksum_rows) != expected_checksum_names or any(
            checksum_rows[Path(name).name] != sha256(member_raw[name])
            for name in expected_files - {"./SHA256SUMS"}
        ):
            raise ValueError("catalog SHA256SUMS binding drifted")
        summary = parse_json(member_raw["./catalog_summary.json"], "catalog summary")
        if not (
            set(summary) == SUMMARY_KEYS
            and summary["artifact_kind"] == "target_unread_coordinate_catalog_receipt_not_authorization"
            and summary["contract"] == "atypemu_nested_support_count_v1_catalog_receipt_v2"
            and summary["study_id"] == "atypemu_nested_support_count_v1"
            and summary["catalog_shard_count"] == 27
            and summary["entity_count"] == len(summary["entities"]) == 135
            and summary["roster_sha256"] == EXPECTED_HASHES[ROSTER_RELATIVE]
            and summary["source_commitment_relative_path"] == SOURCE_COMMITMENT_RELATIVE.as_posix()
            and summary["source_commitment_sha256"] == EXPECTED_HASHES[SOURCE_COMMITMENT_RELATIVE]
            and summary["source_method_composition"] == {"BioEmu": 1.0}
            and all(summary[field] is False for field in (
                "authorization_consumed", "outer_or_formal_metrics_opened", "science_executed",
                "source_scores_read", "target_values_read",
            ))
        ):
            raise ValueError("catalog summary contract drifted")
        for name in sorted(expected_shards, key=lambda value: int(value.split("_")[1].split(".")[0])):
            shard_raw = member_raw[name]
            shard = parse_json(shard_raw, name)
            index = int(name.split("_")[1].split(".")[0])
            if not (
                set(shard) == SHARD_KEYS
                and shard["artifact_kind"] == "target_unread_coordinate_catalog_shard_not_authorization"
                and shard["contract"] == "atypemu_nested_support_count_v1_catalog_shard_v2"
                and shard["study_id"] == "atypemu_nested_support_count_v1"
                and shard["shard_index"] == index
                and shard["shard_count"] == 27
                and shard["entity_count"] == len(shard["entities"]) == 5
                and shard["roster_sha256"] == EXPECTED_HASHES[ROSTER_RELATIVE]
                and shard["source_commitment_relative_path"] == SOURCE_COMMITMENT_RELATIVE.as_posix()
                and shard["source_commitment_sha256"] == EXPECTED_HASHES[SOURCE_COMMITMENT_RELATIVE]
                and all(shard[field] is False for field in (
                    "authorization_consumed", "outer_or_formal_metrics_opened", "science_executed",
                    "source_scores_read", "target_values_read",
                ))
            ):
                raise ValueError(f"catalog shard contract drifted: {name}")
            shard_bindings.append({"member": name, "sha256": sha256(shard_raw)})
            for row in shard["entities"]:
                uid = row["entity_uid"]
                expected = roster_by_uid.get(uid)
                if (
                    set(row) != CATALOG_ENTITY_KEYS
                    or expected is None
                    or uid in entities
                    or row["bmrb_id"] != expected["bmrb_id"]
                    or row["observer_fold"] != expected["observer_fold"]
                    or row["split"] != expected["split"]
                    or row["canonical_heavy_atom_count"] != expected["canonical_heavy_atom_count"]
                    or row["canonical_heavy_topology_sha256"] != expected["canonical_heavy_topology_sha256"]
                    or row["canonical_reference_pdb_sha256"] != expected["canonical_reference_pdb_sha256"]
                    or row["canonical_reference_relative_path"] != expected["canonical_reference_relative_path"]
                    or row["canonical_reference_support_index"] != expected["canonical_reference_support_index"]
                    or row["invalid"] != []
                    or row["invalid_count"] != 0
                    or row["unexpected_entries"] != []
                ):
                    raise ValueError(f"catalog entity identity drifted: {uid}")
                files = row["files"]
                if (
                    row["catalog_digest"] != canonical_sha256(files)
                    or row["canonical_filename_count"] != len(files)
                    or row["heavy_topology_compatible_unique_coordinate_count"] != len(files)
                ):
                    raise ValueError(f"catalog digest drifted: {uid}")
                projected = []
                indexes = []
                for support in files:
                    if set(support) != CATALOG_FILE_KEYS:
                        raise ValueError(f"catalog support schema drifted: {uid}")
                    index_value = support["support_index"]
                    indexes.append(index_value)
                    hashes = (
                        support["pdb_sha256"], support["heavy_atom_topology_sha256"],
                        support["heavy_atom_coordinate_sha256"],
                    )
                    if not (isinstance(index_value, int) and 1 <= index_value <= 1000 and all(SHA256_RE.fullmatch(value) for value in hashes)):
                        raise ValueError(f"invalid parent support record: {uid}")
                    if support["heavy_atom_count"] != expected["canonical_heavy_atom_count"] or support["heavy_atom_topology_sha256"] != expected["canonical_heavy_topology_sha256"]:
                        raise ValueError(f"parent heavy topology drifted: {uid}")
                    projected.append({
                        "heavy_atom_coordinate_sha256": support["heavy_atom_coordinate_sha256"],
                        "pdb_sha256": support["pdb_sha256"],
                        "support_index": index_value,
                    })
                if indexes != sorted(set(indexes)) or row["valid_index_digest"] != canonical_sha256(indexes):
                    raise ValueError(f"parent support indexes drifted: {uid}")
                if row["missing_indices_1_to_1000"] != sorted(set(range(1, 1001)) - set(indexes)):
                    raise ValueError(f"parent missing-index list drifted: {uid}")
                reference = next((support for support in files if support["support_index"] == expected["canonical_reference_support_index"]), None)
                if reference is None or reference["pdb_sha256"] != expected["canonical_reference_pdb_sha256"]:
                    raise ValueError(f"parent reference binding drifted: {uid}")
                entities[uid] = {
                    "bmrb_id": row["bmrb_id"],
                    "entity_uid": uid,
                    "heavy_atom_count": expected["canonical_heavy_atom_count"],
                    "heavy_atom_topology_sha256": expected["canonical_heavy_topology_sha256"],
                    "observer_fold": expected["observer_fold"],
                    "records_sha256": canonical_sha256(projected),
                    "split": expected["split"],
                    "support_count": len(projected),
                    "support_index_digest": canonical_sha256(indexes),
                }
                full_rows.append(row)
    if set(entities) != set(roster_by_uid):
        raise ValueError("parent manifest entity roster incomplete")
    full_rows.sort(key=lambda row: row["entity_uid"])
    summary_projection = [
        {key: row[key] for key in SUMMARY_ENTITY_KEYS} for row in full_rows
    ]
    counts = [row["heavy_topology_compatible_unique_coordinate_count"] for row in full_rows]
    level_status = {
        str(level): {
            "all_entities_count_feasible": all(count >= level for count in counts),
            "entity_count_feasible": sum(count >= level for count in counts),
            "maximum_entity_shortfall": max(max(0, level - count) for count in counts),
            "total_shortfall": sum(max(0, level - count) for count in counts),
        }
        for level in (32, 128, 768, 1536)
    }
    if (
        summary["entities"] != summary_projection
        or summary["full_catalog_digest"] != canonical_sha256(full_rows)
        or summary["level_count_feasibility"] != level_status
    ):
        raise ValueError("catalog summary does not replay from shards")
    rows = [entities[uid] for uid in sorted(entities)]
    support_count = sum(row["support_count"] for row in rows)
    if support_count != 134_850:
        raise ValueError("parent support count drifted")
    return {
        **common("target_unread_parent_heavy_coordinate_manifest_not_authorization", script_hash),
        "catalog_archive": {"path": CATALOG_ARCHIVE_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[CATALOG_ARCHIVE_RELATIVE]},
        "contract": "atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1",
        "entities": rows,
        "entity_count": 135,
        "parent_record_count": support_count,
        "parent_records_sha256": canonical_sha256(rows),
        "roster": {"path": ROSTER_RELATIVE.as_posix(), "sha256": EXPECTED_HASHES[ROSTER_RELATIVE]},
        "shards": shard_bindings,
        "source_commitment": {
            "path": SOURCE_COMMITMENT_RELATIVE.as_posix(),
            "sha256": EXPECTED_HASHES[SOURCE_COMMITMENT_RELATIVE],
        },
        "source_relative_path_pattern": "{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb",
    }


def inspect_runtime(sif: Path) -> dict[str, Any]:
    helper = r'''import hashlib,json,pathlib,re,xml.etree.ElementTree as ET
import numpy,openmm
from openmm import app
root=pathlib.Path(app.__file__).parent
paths={"force_field":root/"data/amber14/protein.ff14SB.xml","hydrogen_definitions":root/"data/hydrogens.xml","modeller_source":root/"modeller.py"}
app.ForceField("amber14/protein.ff14SB.xml")
xml_thresholds=sorted({str(e.attrib["maxph"]) for e in ET.parse(paths["hydrogen_definitions"]).iter() if "maxph" in e.attrib},key=float)
source_thresholds=sorted(set(re.findall(r"pH\s*[<>]=?\s*([0-9]+(?:\.[0-9]+)?)",paths["modeller_source"].read_text())),key=float)
print(json.dumps({"files":{k:{"path":str(v),"sha256":hashlib.sha256(v.read_bytes()).hexdigest(),"size":v.stat().st_size} for k,v in paths.items()},"numpy_version":numpy.__version__,"openmm_version":openmm.__version__,"openmm_version_full":openmm.version.full_version,"platforms":[openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())],"python_version":__import__("sys").version.split()[0],"source_ph_thresholds":source_thresholds,"xml_maxph_thresholds":xml_thresholds},sort_keys=True))'''
    result = subprocess.run(
        ["singularity", "exec", "--containall", "--cleanenv", "--no-home", str(sif), "python", "-c", helper],
        cwd="/tmp", check=True, capture_output=True, text=True, timeout=60,
    )
    return parse_json(result.stdout.encode(), "runtime inspection")


def inspect_docker_archive(raw: bytes) -> dict[str, Any]:
    def digest_hex(value: Any) -> str:
        if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("Docker archive descriptor has invalid digest")
        return value.removeprefix("sha256:")

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        if any(not (member.isfile() or member.isdir()) for member in members):
            raise ValueError("Docker archive has a non-file member")
        directory_members = [member.name for member in members if member.isdir()]
        directory_names = set(directory_members)
        file_members = {member.name: member for member in members if member.isfile()}
        if len(file_members) != sum(member.isfile() for member in members):
            raise ValueError("Docker archive has duplicate file members")
        control_names = {"index.json", "manifest.json", "oci-layout"}
        if (
            len(directory_members) != 2
            or len(directory_names) != 2
            or directory_names != {"blobs", "blobs/sha256"}
        ):
            raise ValueError("Docker archive directory inventory drifted")
        blobs = {}
        for name, member in file_members.items():
            if not name.startswith("blobs/sha256/"):
                continue
            digest = name.removeprefix("blobs/sha256/")
            blob = archive.extractfile(member).read()
            if not SHA256_RE.fullmatch(digest) or sha256(blob) != digest:
                raise ValueError("Docker archive blob digest drifted")
            blobs[digest] = blob
        expected_file_names = control_names | {f"blobs/sha256/{digest}" for digest in blobs}
        if set(file_members) != expected_file_names:
            raise ValueError("Docker archive file inventory drifted")
        control_raw = {
            name: archive.extractfile(file_members[name]).read() for name in control_names
        }
        if parse_json(control_raw["oci-layout"], "OCI layout") != {"imageLayoutVersion": "1.0.0"}:
            raise ValueError("Docker archive OCI layout drifted")
        index = parse_json(control_raw["index.json"], "OCI index")
        if not (
            index.get("schemaVersion") == 2
            and index.get("mediaType") == "application/vnd.oci.image.index.v1+json"
            and isinstance(index.get("manifests"), list)
            and len(index["manifests"]) == 1
        ):
            raise ValueError("Docker archive OCI index drifted")
        root_descriptor = index["manifests"][0]
        image_digest = root_descriptor["digest"]
        image_hex = digest_hex(image_digest)
        if (
            root_descriptor.get("mediaType") != "application/vnd.oci.image.index.v1+json"
            or image_hex not in blobs
            or root_descriptor["size"] != len(blobs[image_hex])
        ):
            raise ValueError("Docker archive image index blob missing")
        image_index = parse_json(blobs[image_hex], "image index")
        if not (
            image_index.get("schemaVersion") == 2
            and image_index.get("mediaType") == "application/vnd.oci.image.index.v1+json"
            and isinstance(image_index.get("manifests"), list)
            and len(image_index["manifests"]) == 2
        ):
            raise ValueError("Docker archive image index drifted")
        referenced = {image_hex}
        for descriptor in image_index["manifests"]:
            digest = digest_hex(descriptor["digest"])
            if (
                descriptor.get("mediaType")
                != "application/vnd.oci.image.manifest.v1+json"
                or digest not in blobs
                or descriptor["size"] != len(blobs[digest])
            ):
                raise ValueError("Docker archive image descriptor drifted")
            referenced.add(digest)
            descriptor_manifest = parse_json(blobs[digest], "image descriptor manifest")
            if not (
                descriptor_manifest.get("schemaVersion") == 2
                and descriptor_manifest.get("mediaType")
                == "application/vnd.oci.image.manifest.v1+json"
                and isinstance(descriptor_manifest.get("layers"), list)
            ):
                raise ValueError("Docker archive descriptor manifest drifted")
            for child in [descriptor_manifest["config"], *descriptor_manifest["layers"]]:
                child_digest = digest_hex(child["digest"])
                if child_digest not in blobs or child["size"] != len(blobs[child_digest]):
                    raise ValueError("Docker archive image child descriptor drifted")
                referenced.add(child_digest)
        legacy = json.loads(
            control_raw["manifest.json"], object_pairs_hook=reject_duplicate_keys
        )
        if not (
            isinstance(legacy, list)
            and len(legacy) == 1
            and set(legacy[0]) == {"Config", "RepoTags", "Layers"}
            and legacy[0]["RepoTags"] == ["atypemu/openmm86-protonation:8.6.0"]
        ):
            raise ValueError("Docker archive legacy manifest drifted")
        legacy_config_hex = legacy[0]["Config"].removeprefix("blobs/sha256/")
        legacy_layer_hexes = [
            value.removeprefix("blobs/sha256/") for value in legacy[0]["Layers"]
        ]
        referenced.update({legacy_config_hex, *legacy_layer_hexes})
        if set(blobs) != referenced:
            raise ValueError("Docker archive contains an unreferenced blob")
        amd64 = [
            row for row in image_index["manifests"]
            if row.get("platform") == {"architecture": "amd64", "os": "linux"}
        ]
        if len(amd64) != 1:
            raise ValueError("Docker archive has no unique linux/amd64 image")
        manifest_digest = amd64[0]["digest"]
        manifest_hex = digest_hex(manifest_digest)
        manifest = parse_json(blobs[manifest_hex], "amd64 image manifest")
        config_digest = manifest["config"]["digest"]
        config_hex = digest_hex(config_digest)
        config = parse_json(blobs[config_hex], "amd64 image config")
        layers = [row["digest"] for row in manifest["layers"]]
        if (
            legacy_config_hex != config_hex
            or legacy_layer_hexes != [digest.removeprefix("sha256:") for digest in layers]
            or parse_json(blobs[legacy_config_hex], "legacy image config") != config
        ):
            raise ValueError("Docker archive roots do not identify the same image")
        required_env = {
            "OPENMM_DEFAULT_PLATFORM=Reference", "PYTHONHASHSEED=0",
            "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTHON_VERSION=3.13.7",
        }
        if config.get("architecture") != "amd64" or not required_env.issubset(config["config"]["Env"]):
            raise ValueError("Docker archive image configuration drifted")
        attestations = [
            row for row in image_index["manifests"]
            if row.get("annotations", {}).get("vnd.docker.reference.type")
            == "attestation-manifest"
        ]
        if (
            len(attestations) != 1
            or attestations[0]["annotations"].get("vnd.docker.reference.digest")
            != manifest_digest
        ):
            raise ValueError("Docker archive build attestation descriptor drifted")
        attestation_manifest = parse_json(
            blobs[digest_hex(attestations[0]["digest"])], "build attestation manifest"
        )
        if (
            len(attestation_manifest["layers"]) != 1
            or attestation_manifest["layers"][0]["mediaType"]
            != "application/vnd.in-toto+json"
        ):
            raise ValueError("Docker archive build attestation layer drifted")
        attestation_digest = attestation_manifest["layers"][0]["digest"]
        attestation = parse_json(blobs[digest_hex(attestation_digest)], "build attestation")
        definition = attestation["predicate"]["buildDefinition"]
        request_args = definition["externalParameters"]["request"]["args"]
        dependencies = definition["resolvedDependencies"]
        subjects = attestation["subject"]
        if not (
            attestation.get("predicateType") == "https://slsa.dev/provenance/v1"
            and definition["externalParameters"]["configSource"]["path"]
            == DOCKERFILE_RELATIVE.name
            and request_args.get("force-network-mode") == "none"
            and "no-cache" in request_args
            and definition["internalParameters"]["builderPlatform"] == "linux/amd64"
            and len(dependencies) == 1
            and dependencies[0]["digest"]
            == {"sha256": "781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7"}
            and len(subjects) == 1
            and subjects[0]["digest"] == {"sha256": manifest_hex}
        ):
            raise ValueError("Docker archive build attestation semantics drifted")
        return {
            "amd64_config_digest": config_digest,
            "amd64_layer_digests": layers,
            "amd64_manifest_digest": manifest_digest,
            "build_attestation_digest": attestation_digest,
            "image_index_digest": image_digest,
        }


def inspect_singularity(sif: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["singularity", "inspect", "--json", str(sif)],
        cwd="/tmp", check=True, capture_output=True, text=True, timeout=60,
    )
    value = parse_json(result.stdout.encode(), "Singularity inspection")
    labels = value["data"]["attributes"]["labels"]
    if not isinstance(labels, dict):
        raise ValueError("Singularity labels missing")
    return labels


def make_environment_manifest(
    root: Path, sif: Path, docker_archive: Path, script_hash: str
) -> dict[str, Any]:
    if not sif.is_absolute() or sif.is_symlink():
        raise ValueError("runtime SIF must be a direct absolute regular file")
    if not docker_archive.is_absolute() or docker_archive.is_symlink():
        raise ValueError("runtime Docker archive must be a direct absolute regular file")
    sif_raw = read_regular(sif, 200_000_000)
    docker_raw = read_regular(docker_archive, 200_000_000)
    docker_inspection = inspect_docker_archive(docker_raw)
    with tempfile.TemporaryDirectory(prefix="openmm86-sif-snapshot-") as directory:
        snapshot = Path(directory) / sif.name
        snapshot.write_bytes(sif_raw)
        snapshot.chmod(0o444)
        singularity_labels = inspect_singularity(snapshot)
        inspection = inspect_runtime(snapshot)
    expected = {
        "numpy_version": "2.3.3", "openmm_version": "8.6",
        "openmm_version_full": "8.6.0.dev-c6173db", "platforms": ["Reference", "CPU"],
        "python_version": "3.13.7", "source_ph_thresholds": ["6.5"],
        "xml_maxph_thresholds": ["4.4", "8.5", "10.0"],
    }
    if any(inspection.get(key) != value for key, value in expected.items()):
        raise ValueError("OpenMM runtime identity or pH thresholds drifted")
    if (
        singularity_labels.get("org.label-schema.build-arch") != "amd64"
        or singularity_labels.get("org.label-schema.usage.singularity.deffile.bootstrap")
        != "docker-archive"
        or singularity_labels.get("org.label-schema.usage.singularity.deffile.from")
        != str(docker_archive)
        or not singularity_labels.get("org.label-schema.usage.singularity.version")
        or sha256(read_regular(sif, 200_000_000)) != sha256(sif_raw)
    ):
        raise ValueError("Singularity conversion provenance or stable hash drifted")
    dockerfile_raw = read_regular(root / DOCKERFILE_RELATIVE, 100_000)
    combined = sorted({Decimal(value) for value in inspection["source_ph_thresholds"] + inspection["xml_maxph_thresholds"]})
    boundaries = [Decimal("0"), *combined, Decimal("14")]
    midpoints = [format((left + right) / 2, "f") for left, right in zip(boundaries, boundaries[1:])]
    return {
        **common("openmm86_runtime_environment_manifest_not_authorization", script_hash),
        "base_image": {
            "amd64_manifest_digest": "sha256:781449467ffb6f04218f09b1ecdcdc7d22b289ee5da9ec498b024e24ad7a6db7",
            "name": "python:3.13.7-slim-bookworm",
        },
        "build_recipe": {"path": DOCKERFILE_RELATIVE.as_posix(), "sha256": sha256(dockerfile_raw)},
        "contract": "atypemu_nested_support_count_v1_openmm86_environment_manifest_v1",
        "conversion": {
            "docker_archive": {
                "filename": docker_archive.name,
                "sha256": sha256(docker_raw),
                "size": len(docker_raw),
            },
            "docker_image": docker_inspection,
            "singularity_labels": singularity_labels,
        },
        "inspection": inspection,
        "ph_regimes": {
            "combined_transition_thresholds": [format(value, "f") for value in combined],
            "interval_midpoints": midpoints,
            "lower_bound_inclusive": "0.0",
            "upper_bound_inclusive": "14.0",
        },
        "required_platform": "Reference",
        "runtime_sif": {"filename": sif.name, "sha256": sha256(sif_raw), "size": len(sif_raw)},
        "wheels": [
            {"filename": "numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "sha256": "5b83648633d46f77039c29078751f80da65aa64d5622a3cd62aaef9d835b6c93", "size": 16636301},
            {"filename": "openmm-8.6.0-cp313-cp313-manylinux_2_34_x86_64.whl", "sha256": "1b0f39a812452fd9eb1faf28ed4e9d832aef44fc2072f4442a81a4ce7a6f56bf", "size": 14425566},
        ],
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def build(root: Path, sif: Path, docker_archive: Path) -> dict[str, dict[str, Any]]:
    script_hash = sha256(read_regular(root / SCRIPT_RELATIVE, 1_000_000))
    roster = parse_json(checked_input(root, ROSTER_RELATIVE, 2_000_000), "roster")
    if roster.get("entity_count") != 135 or len(roster.get("entities", [])) != 135:
        raise ValueError("roster count drifted")
    source_commitment_raw = checked_input(root, SOURCE_COMMITMENT_RELATIVE, 2_000_000)
    source_commitment = parse_json(source_commitment_raw, "source commitment")
    if not (
        sha256(source_commitment_raw) == roster["source_commitment_sha256"]
        and source_commitment.get("contract")
        == "k32_dynamic_distance_cache_source_commitment_v1"
        and source_commitment.get("entity_count") == 135
        and len(source_commitment.get("entities", [])) == 135
        and len(source_commitment.get("support_ids", [])) == 32
        and source_commitment.get("source_gate_authorized") is False
        and source_commitment.get("formal_evaluation_authorized") is False
        and source_commitment.get("target_values_read") is False
    ):
        raise ValueError("roster source commitment binding drifted")
    recovery = parse_json(checked_input(root, RECOVERY_RELATIVE, 2_000_000), "condition recovery")
    v6_evidence = parse_json(checked_input(root, V6_EVIDENCE_RECEIPT_RELATIVE, 100_000), "v6 evidence receipt")
    v6_archive_raw = checked_input(root, V6_ARCHIVE_RELATIVE, 5_000_000)
    if v6_evidence["archive"]["sha256"] != sha256(v6_archive_raw) or v6_evidence["closed_capabilities"] != FALSE_CAPABILITIES:
        raise ValueError("v6 evidence binding drifted")
    catalog_raw = checked_input(root, CATALOG_ARCHIVE_RELATIVE, 30_000_000)
    return {
        "condition": make_condition_manifest(roster, recovery, v6_receipt(v6_archive_raw), script_hash),
        "sequence": make_sequence_manifest(root, roster, script_hash),
        "parent": make_parent_manifest(catalog_raw, roster, script_hash),
        "environment": make_environment_manifest(root, sif, docker_archive, script_hash),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sif", type=Path, required=True)
    parser.add_argument("--runtime-docker-archive", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = bound_root()
    manifests = build(root, args.runtime_sif, args.runtime_docker_archive)
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="condition-input-freeze-") as directory:
            for name, value in manifests.items():
                path = Path(directory) / OUTPUTS[name].name
                write_exclusive(path, value)
                if parse_json(path.read_bytes(), name) != value:
                    raise AssertionError(f"round-trip failed: {name}")
        print("STATUS PASS_TARGET_UNREAD_INPUT_FREEZER_SELF_TEST")
        return 0
    for name, value in manifests.items():
        write_exclusive(root / OUTPUTS[name], value)
    print("METRIC condition_manifest_entities=135")
    print("METRIC observed_condition_entities=119")
    print("METRIC parent_heavy_records=134850")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_INPUT_MANIFESTS_FROZEN_SMOKE_UNRUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
