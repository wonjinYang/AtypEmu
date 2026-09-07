#!/usr/bin/env python3
"""Validate the target-unread HOLD-only ff15ipq all-support plan."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import tarfile
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
SCRIPT_RELATIVE = Path(
    "gpuopt/candidates/check_ff15ipq_all_support_qualification_plan.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_qualification_plan_v1.json"
)
ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/"
    "catalog_v3_shards.tar.gz"
)
CONDITION_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json"
)
ARCHIVE_SHA256 = "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2"
CONDITION_SHA256 = "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618"
SHA = re.compile(r"[0-9a-f]{64}\Z")
UID = re.compile(r"bmrb:([0-9]+):entity:1\Z")
MIDPOINTS = ["2.2", "5.45", "7.5", "9.25", "12.0"]
INPUTS = {
    "base_openmm_environment_manifest": (
        Path(
            "gpuopt/preunblind/"
            "atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json"
        ),
        "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
    ),
    "catalog_shard_archive": (ARCHIVE_RELATIVE, ARCHIVE_SHA256),
    "condition_manifest": (CONDITION_RELATIVE, CONDITION_SHA256),
    "condition_policy": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "condition_uncertainty_protonation_plan_v1.json"
        ),
        "37af7bd203fbe105178657663879f1737c905aebb12d4d9115d61773dc1e8843",
    ),
    "ff15ipq_template_scan_receipt": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "cohort_support1_ff15ipq_template_scan_receipt_v1.json"
        ),
        "a106871b1fc0504f90d92e412ff5242f0f2db995e384650b0befe8b6c0fc9bb5",
    ),
    "openmm_environment_manifest": (
        Path(
            "gpuopt/preunblind/"
            "atypemu_nested_support_count_v1_openmm86_ff15ipq_environment_manifest_v1.json"
        ),
        "f40b75666b440b464f094f63038a67755c577704998337b3d7ec0ca1451e7a10",
    ),
    "parent_heavy_coordinate_manifest": (
        Path(
            "gpuopt/preunblind/"
            "atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1.json"
        ),
        "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
    ),
    "protein_sequence_manifest": (
        Path(
            "gpuopt/preunblind/"
            "atypemu_nested_support_count_v1_protein_sequence_manifest_v1.json"
        ),
        "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
    ),
    "support1_postrun_audit": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "cohort_support1_ff15ipq_preflight_postrun_audit_receipt_v1.json"
        ),
        "c6deaea251ff9730a6b455724e767f507004aba3045f6e1854d7cf207dee2d16",
    ),
    "support1_result": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "cohort_support1_ff15ipq_preflight_result_receipt_v1.json"
        ),
        "f40bdebf94077fdea6070fd432f2d932ae750a390c583fde77111e0102aa0ec9",
    ),
    "runtime_input_projection": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "ff15ipq_all_support_input_projection_v1.json.gz"
        ),
        "e9b4861216d99e8136df872958a50f568337fc8a6b4991b25df2b906dcee5046",
    ),
    "runtime_input_projection_producer": (
        Path("gpuopt/candidates/freeze_ff15ipq_all_support_input_projection.py"),
        "dbff43191c3cd150ef6ebc160f6aea1c7c035634ce9ed9adb7a3849a2107f89b",
    ),
}
CONDITION_KEYS = {
    "bmrb_id",
    "condition_state",
    "deposited_ph",
    "entity_uid",
    "pH_source",
    "recovery_v6_hold_reasons",
    "temperature_ionic_diagnostics",
}
SHARD_KEYS = {
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
ENTITY_KEYS = {
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
FILE_KEYS = {
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


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reject_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=reject_dupes, parse_constant=reject_constant
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys differ")


def exact_false_map(value: Any, keys: set[str], label: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(item is not False for item in value.values())
    ):
        raise ValueError(f"{label} is not an exact false-valued map")


def checked_support_indexes(files: list[dict[str, Any]]) -> list[int]:
    indexes = [item["support_index"] for item in files]
    if indexes != sorted(set(indexes)) or not all(
        type(index) is int and 1 <= index <= 1000 for index in indexes
    ):
        raise ValueError("support indexes are not unique ordered BioEmu indexes")
    return indexes


def checked_missing_indices(value: Any, present: list[int]) -> list[int]:
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        raise TypeError("missing-support indexes are not exact integers")
    expected = [index for index in range(1, 1001) if index not in set(present)]
    if value != expected:
        raise ValueError("catalog missing-index complement mismatch")
    return expected


def checked_bmrb_identity(uid: Any, bmrb_id: Any) -> str:
    if not isinstance(uid, str):
        raise TypeError("entity UID is not a string")
    match = UID.fullmatch(uid)
    if match is None or bmrb_id != f"bmr{match.group(1)}":
        raise ValueError("BMRB identity mismatch")
    return uid


def root_file(root: Path, relative: Path) -> bytes:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("unsafe repository path")
    directory_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    file_fd: int | None = None
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {relative}")
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(file_fd)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise ValueError(f"file changed during read: {relative}")
        return b"".join(chunks)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def archive_json(raw: bytes) -> list[dict[str, Any]]:
    shards: dict[int, dict[str, Any]] = {}
    expected_files = {
        "./SHA256SUMS",
        "./catalog_summary.json",
        *(f"./shard_{index}.json" for index in range(27)),
    }
    seen: set[str] = set()
    root_seen = False
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            if name == "." and member.isdir():
                if root_seen:
                    raise ValueError("duplicate catalog archive root")
                root_seen = True
                continue
            if name not in expected_files or name in seen or not member.isfile():
                raise ValueError("unexpected, duplicate, or non-regular archive member")
            seen.add(name)
            match = re.fullmatch(r"\./shard_([0-9]+)\.json", name)
            if match is None:
                continue
            index = int(match.group(1))
            if index in shards:
                raise ValueError("duplicate catalog shard")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("catalog shard cannot be read")
            shards[index] = json_object(handle.read(), name)
    if not root_seen or seen != expected_files or set(shards) != set(range(27)):
        raise ValueError("catalog archive member set is incomplete")
    return [shards[index] for index in range(27)]


def validate(root: Path) -> int:
    checks = 0
    plan_raw = root_file(root, PLAN_RELATIVE)
    plan = json_object(plan_raw, "plan")
    exact_keys(
        plan,
        {
            "artifact_kind",
            "authorization",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "design_provenance",
            "execution_contract",
            "frozen_inputs",
            "input_access_policy",
            "openmm_policy",
            "output_contract",
            "physicality_gate",
            "policy_lineage",
            "qualification_condition_branches",
            "roster",
            "scope_limits",
            "stage_order",
            "state",
            "study_id",
            "validator",
        },
        "plan",
    )
    if plan["candidate_id"] != CANDIDATE_ID:
        raise ValueError("candidate mismatch")
    if plan["state"] != "HOLD_INPUT_PROJECTION_FROZEN_IMPLEMENTATION_ABSENT_UNRUN":
        raise ValueError("plan escaped HOLD")
    if plan["execution_contract"]["source_commitment"] != "ABSENT_UNFROZEN":
        raise ValueError("source commitment unexpectedly present")
    if plan["openmm_policy"] != {
        "add_hydrogens_call": (
            "Modeller.addHydrogens(forcefield, pH=branch_ph, variants=None, "
            "platform=Reference)"
        ),
        "forcefield": "amber14/protein.ff15ipq.xml",
        "forcefield_sha256": (
            "0085c9dc2818a28501f6074a0bb46b994a9787bf4010ff86417fe014b11ab622"
        ),
        "input_hydrogens": "remove all H/D/T before protonation",
        "openmm_exact_version": "8.6.0.dev-c6173db",
        "parent_heavy_atoms": (
            "exact ordered atom identity and 0.001-A parsed coordinate preservation"
        ),
        "platform": "Reference",
        "randomness": (
            "candidate/entity/support/branch/role SHA256-derived seed with isolated "
            "process and every exposed RNG seeded"
        ),
        "unsupported_residue_or_template": (
            "fail the entire candidate without entity or residue exception"
        ),
        "variants": None,
    }:
        raise ValueError("plan ff15ipq policy drifted")
    lineage = plan["policy_lineage"]
    if (
        not isinstance(lineage, dict)
        or "protein.ff14SB.xml is replaced uniformly by amber14/protein.ff15ipq.xml"
        not in lineage.get("superseded_field", "")
        or "No predecessor authorization is reusable"
        not in lineage.get("unchanged_authorization_rule", "")
    ):
        raise ValueError("ff15ipq policy supersession is not explicit")
    expected_authorization = {
        "consumption_allowed": False,
        "existing_authorizations_reusable": False,
        "request_allowed": False,
    }
    exact_false_map(plan["authorization"], set(expected_authorization), "authorization")
    capability_keys = {
        "authorization_consumed",
        "outer_or_formal_metrics_opened",
        "science_executed",
        "source_construction_executed",
        "source_scores_read",
        "target_atom_identities_read",
        "target_values_read",
    }
    exact_false_map(plan["closed_capabilities"], capability_keys, "closed capabilities")
    validator = plan["validator"]
    script_raw = root_file(root, SCRIPT_RELATIVE)
    if set(validator) != {"path", "sha256"} or validator != {
        "path": SCRIPT_RELATIVE.as_posix(),
        "sha256": digest(script_raw),
    }:
        raise ValueError("validator self-binding mismatch")
    expected_branches = [
        {"condition_branch_id": f"regime-{index}", "proposal_pH": ph}
        for index, ph in enumerate(MIDPOINTS)
    ]
    if plan["qualification_condition_branches"] != {
        "observed": "one observed-0 branch using the exact deposited_ph string",
        "unresolved": expected_branches,
        "use_in_final_support_mapping": (
            "Qualification materializes all five possible states per unresolved "
            "parent support. A later target-unread diversity/order commitment selects "
            "exactly one frozen branch per ranked support using the already frozen "
            "balanced mapping rule; this plan does not define the final support ensemble."
        ),
    }:
        raise ValueError("qualification condition branches drifted")
    access = plan["input_access_policy"]
    forbidden = access.get("forbidden_generator_fields")
    if (
        access.get("worker_raw_catalog_access_allowed") is not False
        or access.get("runtime_pdb_path_template")
        != "data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb"
        or not isinstance(forbidden, list)
        or set(forbidden)
        != {
            "assigned_atom_inventory",
            "atom_id",
            "atom_name",
            "chemical_shift",
            "eligibility_rows",
            "observer_fold",
            "source_or_outer_scores",
            "split",
            "target_id",
            "target_uid",
            "target_value",
        }
    ):
        raise ValueError("runtime input-access closure drifted")
    checks += 5

    if set(plan["frozen_inputs"]) != set(INPUTS):
        raise ValueError("frozen-input set drifted")
    input_raw: dict[str, bytes] = {}
    for name, (relative, expected) in INPUTS.items():
        if plan["frozen_inputs"][name] != {
            "path": relative.as_posix(),
            "sha256": expected,
        }:
            raise ValueError(f"plan binding mismatch: {name}")
        raw = root_file(root, relative)
        if digest(raw) != expected:
            raise ValueError(f"input hash mismatch: {name}")
        input_raw[name] = raw
        checks += 2

    condition = json_object(input_raw["condition_manifest"], "condition manifest")
    exact_keys(
        condition,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "entities",
            "entity_count",
            "freezer",
            "roster",
            "scope",
            "state_counts",
            "upstream",
        },
        "condition manifest",
    )
    if condition["entity_count"] != 135 or condition["state_counts"] != {
        "observed": 119,
        "state_ambiguous": 5,
        "state_missing": 11,
    }:
        raise ValueError("condition manifest summary drifted")
    exact_false_map(
        condition["closed_capabilities"], capability_keys, "condition capabilities"
    )
    policy = json_object(input_raw["condition_policy"], "condition policy")
    unresolved_rule = policy["condition_policy"]["unresolved_rule"]
    if (
        unresolved_rule["lower_ph_bound_inclusive"] != "0.0"
        or unresolved_rule["upper_ph_bound_inclusive"] != "14.0"
        or policy["support_mapping"]["parent_support_order"] != "ABSENT_UNQUALIFIED"
    ):
        raise ValueError("condition policy boundary drifted")
    environment = json_object(
        input_raw["openmm_environment_manifest"], "OpenMM environment"
    )
    if (
        environment["base_environment_manifest"]
        != {
            "path": INPUTS["base_openmm_environment_manifest"][0].as_posix(),
            "sha256": INPUTS["base_openmm_environment_manifest"][1],
        }
        or environment["force_field"]
        != {
            "relative_path": "amber14/protein.ff15ipq.xml",
            "sha256": "0085c9dc2818a28501f6074a0bb46b994a9787bf4010ff86417fe014b11ab622",
        }
        or environment["container"]
        != {
            "filename": "openmm86_protonation_8.6.0.sif",
            "sha256": "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5",
            "size": 78290944,
        }
        or environment["openmm_exact_version"] != "8.6.0.dev-c6173db"
        or environment["platform"] != "Reference"
        or environment["ph_regimes"]["interval_midpoints"] != MIDPOINTS
    ):
        raise ValueError("ff15ipq OpenMM environment binding drifted")
    entities = condition.get("entities")
    if not isinstance(entities, list) or len(entities) != 135:
        raise ValueError("condition entity count mismatch")
    states: dict[str, tuple[int, str, str, str | None]] = {}
    for row in entities:
        if not isinstance(row, dict):
            raise TypeError("condition row is not an object")
        exact_keys(row, CONDITION_KEYS, "condition entity")
        uid, state, bmrb_id = (
            row["entity_uid"],
            row["condition_state"],
            row["bmrb_id"],
        )
        if not isinstance(uid, str) or uid in states:
            raise ValueError("condition identity mismatch")
        checked_bmrb_identity(uid, bmrb_id)
        if state == "observed":
            ph = row.get("deposited_ph")
            if (
                not isinstance(ph, str)
                or not math.isfinite(float(ph))
                or not 0.0 <= float(ph) <= 14.0
            ):
                raise ValueError("observed pH is invalid")
            states[uid] = (1, bmrb_id, state, ph)
        elif state in {"state_missing", "state_ambiguous"}:
            if row.get("deposited_ph") is not None:
                raise ValueError("unresolved state was imputed")
            states[uid] = (5, bmrb_id, state, None)
        else:
            raise ValueError("unknown condition state")
    multiplicities = [value[0] for value in states.values()]
    if multiplicities.count(1) != 119 or multiplicities.count(5) != 16:
        raise ValueError("condition-state quota mismatch")
    checks += 135

    seen: set[str] = set()
    supports = 0
    support_states = 0
    counts: list[int] = []
    projection_entries: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(
        archive_json(input_raw["catalog_shard_archive"])
    ):
        exact_keys(shard, SHARD_KEYS, "catalog shard")
        if shard.get("shard_index") != shard_index or shard.get("shard_count") != 27:
            raise ValueError("catalog shard identity mismatch")
        rows = shard.get("entities")
        if not isinstance(rows, list) or shard.get("entity_count") != len(rows):
            raise ValueError("catalog shard entity count mismatch")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("catalog entity is not an object")
            exact_keys(row, ENTITY_KEYS, "catalog entity")
            uid, files = row.get("entity_uid"), row.get("files")
            if not isinstance(uid, str) or uid in seen or uid not in states:
                raise ValueError("catalog entity identity mismatch")
            if row.get("bmrb_id") != states[uid][1]:
                raise ValueError("catalog/condition BMRB identity mismatch")
            if not isinstance(files, list):
                raise TypeError("catalog files are absent")
            for item in files:
                if not isinstance(item, dict):
                    raise TypeError("catalog support is not an object")
                exact_keys(item, FILE_KEYS, "catalog support")
            checked_support_indexes(files)
            if any(
                not isinstance(item.get("pdb_sha256"), str)
                or SHA.fullmatch(item["pdb_sha256"]) is None
                for item in files
            ):
                raise ValueError("catalog PDB hash is invalid")
            seen.add(uid)
            count = len(files)
            indexes = [item["support_index"] for item in files]
            missing = checked_missing_indices(row["missing_indices_1_to_1000"], indexes)
            state, deposited_ph = states[uid][2], states[uid][3]
            branches = (
                [{"condition_branch_id": "observed-0", "proposal_pH": deposited_ph}]
                if state == "observed"
                else [
                    {
                        "condition_branch_id": f"regime-{index}",
                        "proposal_pH": ph,
                    }
                    for index, ph in enumerate(MIDPOINTS)
                ]
            )
            projection_entries.append(
                {
                    "bmrb_id": states[uid][1],
                    "condition_branches": branches,
                    "condition_state": state,
                    "entity_uid": uid,
                    "missing_support_indices": missing,
                    "supports": [
                        {
                            "parent_raw_pdb_sha256": item["pdb_sha256"],
                            "support_index": item["support_index"],
                        }
                        for item in files
                    ],
                }
            )
            counts.append(count)
            supports += count
            support_states += count * states[uid][0]
            checks += count
    if seen != set(states):
        raise ValueError("catalog/condition roster mismatch")
    if (
        supports != 134850
        or support_states != 198758
        or min(counts) != 991
        or max(counts) != 1000
    ):
        raise ValueError("all-support cardinality mismatch")

    projection = json_object(
        gzip.decompress(input_raw["runtime_input_projection"]),
        "runtime input projection",
    )
    exact_keys(
        projection,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "counts",
            "entries",
            "producer",
            "runtime_allowed_fields",
            "source_path_template",
            "state",
            "upstream",
        },
        "runtime input projection",
    )
    exact_false_map(
        projection["closed_capabilities"], capability_keys, "projection capabilities"
    )
    projection_entries.sort(key=lambda item: item["entity_uid"])
    if (
        projection["candidate_id"] != CANDIDATE_ID
        or projection["artifact_kind"]
        != "target_unread_minimal_all_support_runtime_input_projection"
        or projection["contract"]
        != "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1"
        or projection["counts"]
        != {
            "entity_count": 135,
            "qualification_candidate_state_count": support_states,
            "support_count": supports,
        }
        or projection["entries"] != projection_entries
        or projection["producer"]
        != {
            "path": INPUTS["runtime_input_projection_producer"][0].as_posix(),
            "sha256": INPUTS["runtime_input_projection_producer"][1],
        }
        or projection["source_path_template"]
        != "data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb"
        or projection["runtime_allowed_fields"]
        != [
            "bmrb_id",
            "condition_branch_id",
            "condition_state",
            "entity_uid",
            "missing_support_indices",
            "parent_raw_pdb_sha256",
            "proposal_pH",
            "support_index",
        ]
        or projection["state"] != "HOLD_RUNTIME_INPUT_ONLY_IMPLEMENTATION_UNRUN"
        or projection["upstream"]
        != {
            "catalog_shard_archive_sha256": ARCHIVE_SHA256,
            "condition_manifest_sha256": CONDITION_SHA256,
        }
    ):
        raise ValueError("runtime input projection replay mismatch")
    roster = plan["roster"]
    output = plan["output_contract"]
    if (
        roster["entity_count"] != 135
        or roster["observed_entity_count"] != 119
        or roster["unresolved_entity_count"] != 16
        or roster["support_count"] != supports
        or roster["support_condition_state_count"] != support_states
        or roster["support_count_range_per_entity"] != [991, 1000]
        or output["entity_archives"] != 135
        or output["support_count"] != supports
        or output["qualification_candidate_state_count"] != support_states
        or output["support_condition_state_count"] != support_states
    ):
        raise ValueError("plan cardinality claims drifted")
    return checks + 8


def self_test() -> int:
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"):
        try:
            json_object(raw, "self-test")
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid JSON accepted")
    for value in ({}, {"x": 0}, {"x": None}):
        try:
            exact_false_map(value, {"x"}, "self-test")
        except ValueError:
            pass
        else:
            raise AssertionError("non-exact false map accepted")
    for files in (
        [{"support_index": True}],
        [{"support_index": 1}, {"support_index": 1}],
        [{"support_index": 0}],
    ):
        try:
            checked_support_indexes(files)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid support indexes accepted")
    for missing in ([True, *range(2, 1001)], [1.0, *range(2, 1001)]):
        try:
            checked_missing_indices(missing, [])
        except TypeError:
            pass
        else:
            raise AssertionError("non-integer missing-support index accepted")
    for uid, bmrb_id in (("bmrb:1:entity:1", "bmr2"), ("other:1", "bmr1")):
        try:
            checked_bmrb_identity(uid, bmrb_id)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid BMRB identity accepted")
    for name, kind in (
        ("nested/shard_0.json", "file"),
        ("./extra.json", "file"),
        ("./shard_0.json", "symlink"),
    ):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "./shard_1.json"
                archive.addfile(member)
            else:
                member.size = 2
                archive.addfile(member, io.BytesIO(b"{}"))
        try:
            archive_json(buffer.getvalue())
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe archive member accepted")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for _ in range(2):
            member = tarfile.TarInfo(".")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    try:
        archive_json(buffer.getvalue())
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate archive root accepted")
    return 19


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(f"STATUS PASS_HOLD_ALL_SUPPORT_PLAN_SELF_TEST checks={self_test()}")
        return 0
    actual = Path(__file__).absolute()
    root = actual.parents[2]
    if actual != root / Path("gpuopt/candidates") / actual.name or actual.is_symlink():
        raise ValueError("validator must run from its canonical repository path")
    checks = validate(root)
    print(f"STATUS PASS_HOLD_ALL_SUPPORT_PLAN_ONLY checks={checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
