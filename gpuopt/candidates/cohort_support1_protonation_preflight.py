# ruff: noqa: TRY004
"""Generate the unexecuted HOLD support-1 protonation preflight when separately released.

This file is deliberately inert until a later launcher supplies a committed source
binding and a sealed, read-only stage.  ``--self-test`` is static and needs neither
OpenMM nor the frozen manifests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any

CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_v1"
)
SUPPORT_INDEX = 1
MIDPOINTS = ("2.2", "5.45", "7.5", "9.25", "12.0")
ENTITY_COUNT = 135
STATE_COUNT = 199
INPUT_HASHES = {
    "environment.json": "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
    "condition.json": "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618",
    "sequence.json": "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
    "parent.json": "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
}
PLAN_NAME = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_plan_v1.json"
)
COMMITMENT_NAME = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_"
    "source_commitment_v7.json"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_"
    "source_commitment_v7"
)
MAX_PARALLEL_WORKERS = 8
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
RUNTIME_FILE_HASHES = {
    "data/amber14/protein.ff14SB.xml": "d9f9779c09d67cd5f8bc657692f174ffab14c469dfd06d560ac1899fa7e976b8",
    "data/hydrogens.xml": "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d",
    "modeller.py": "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3",
}


@dataclass(frozen=True)
class State:
    entity_uid: str
    branch_id: str
    ph: str
    pdb_relative: str
    pdb_sha256: str
    heavy_atom_count: int
    heavy_topology_sha256: str
    heavy_coordinate_sha256: str
    sequence: str
    residues: tuple[tuple[str, str, str, str], ...]


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def safe_token(entity_uid: str) -> str:
    if not isinstance(entity_uid, str) or not entity_uid or "\x00" in entity_uid:
        raise ValueError("invalid entity_uid")
    slug = SLUG_RE.sub("-", entity_uid).strip(".-")[:48] or "entity"
    return f"{slug}-{sha256(entity_uid.encode())[:20]}"


def seed_for(entity_uid: str, branch_id: str, role: str) -> int:
    if role not in {"generation-repeat-0", "generation-repeat-1", "checker-replay"}:
        raise ValueError("unknown seed role")
    # Process role must not alter the scientific RNG stream: exact repeat and
    # independent-replay equality are the determinism test.
    material = "\0".join(
        (CANDIDATE_ID, entity_uid, str(SUPPORT_INDEX), branch_id)
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _require_frozen_source(inputs: Path) -> None:
    """Refuse every non-self-test entry point before opening cohort inputs."""
    scripts = inputs.parent / "scripts"
    commitment_raw = read_regular(scripts, COMMITMENT_NAME)
    commitment = parse_json(commitment_raw, "source commitment")
    if set(commitment) != {"candidate_id", "contract", "files", "git_commit", "status"}:
        raise ValueError("source commitment schema drifted")
    if not (
        commitment["candidate_id"] == CANDIDATE_ID
        and commitment["contract"] == COMMITMENT_CONTRACT
        and commitment["status"] == "FROZEN_COMMITTED"
        and isinstance(commitment["git_commit"], str)
        and GIT_COMMIT_RE.fullmatch(commitment["git_commit"])
    ):
        raise PermissionError("source commitment is absent or not frozen")
    expected = {
        "gpuopt/candidates/check_cohort_support1_protonation_preflight.py": (
            "check_cohort_support1_protonation_preflight.py"
        ),
        "gpuopt/candidates/cohort_support1_protonation_preflight.py": (
            "cohort_support1_protonation_preflight.py"
        ),
        "gpuopt/candidates/launch_cohort_support1_protonation_preflight.py": (
            "launch_cohort_support1_protonation_preflight.py"
        ),
        "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_"
        "protonation_preflight_plan_v1.json": PLAN_NAME,
    }
    files = commitment["files"]
    if not isinstance(files, dict) or set(files) != set(expected):
        raise ValueError("source commitment inventory drifted")
    for repository_path, staged_name in expected.items():
        wanted = files[repository_path]
        if (
            not isinstance(wanted, str)
            or SHA256_RE.fullmatch(wanted) is None
            or sha256(read_regular(scripts, staged_name)) != wanted
        ):
            raise ValueError(f"source commitment hash mismatch: {repository_path}")
    plan = parse_json(read_regular(scripts, PLAN_NAME), "frozen plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("state") == "HOLD_PREFLIGHT_SOURCE_FROZEN_UNRUN"
        and plan.get("source_commitment", {}).get("status") == "FROZEN_COMMITTED"
    ):
        raise PermissionError("plan has not released frozen HOLD preflight execution")


def pythonhashseed(seed: int) -> str:
    """CPython accepts only a 32-bit PYTHONHASHSEED; retain the full RNG seed."""
    return str(seed % (2**32))


def _decimal_ph(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("pH must be a string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid pH") from error
    if not number.is_finite() or not Decimal(0) <= number <= Decimal(14):
        raise ValueError("pH out of range")
    return format(number, "f")


def states_from_condition(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Derive exactly one observed branch or all five unresolved branches."""
    output: list[tuple[str, str, str]] = []
    observed = unresolved = 0
    seen: set[str] = set()
    for row in rows:
        if set(row) != {
            "bmrb_id",
            "condition_state",
            "deposited_ph",
            "entity_uid",
            "pH_source",
            "recovery_v6_hold_reasons",
            "temperature_ionic_diagnostics",
        }:
            raise ValueError("condition entity schema drifted")
        uid, kind = row["entity_uid"], row["condition_state"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("condition entity identity is invalid or duplicate")
        seen.add(uid)
        if kind == "observed":
            output.append((uid, "observed-0", _decimal_ph(row["deposited_ph"])))
            observed += 1
        elif kind in {"state_missing", "state_ambiguous"}:
            if row["deposited_ph"] is not None:
                raise ValueError("unresolved condition has an imputed pH")
            output.extend(
                (uid, f"regime-{index}", ph) for index, ph in enumerate(MIDPOINTS)
            )
            unresolved += 1
        else:
            raise ValueError("unknown condition state")
    if len(rows) != ENTITY_COUNT or observed != 119 or unresolved != 16:
        raise ValueError("condition cohort count drifted")
    if len(output) != STATE_COUNT:
        raise ValueError("condition state count drifted")
    return sorted(output)


def _under(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative!r}")
    current = root
    for part in candidate.parts:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink rejected: {relative}")
    return current


def read_regular(root: Path, relative: str, maximum_bytes: int = 4_000_000) -> bytes:
    path = _under(root, relative)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"invalid regular input: {relative}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, min(1_048_576, maximum_bytes + 1)):
            chunks.append(chunk)
            if sum(map(len, chunks)) > maximum_bytes:
                raise ValueError(f"input too large: {relative}")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"input changed while reading: {relative}")
        return raw
    finally:
        os.close(descriptor)


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid sha256: {label}")
    return value


def _residue_tuple(row: Any) -> tuple[str, str, str, str]:
    if not isinstance(row, dict) or set(row) != {
        "chain_id",
        "insertion_code",
        "residue_id",
        "residue_name",
    }:
        raise ValueError("residue schema drifted")
    values = tuple(
        row[field]
        for field in ("chain_id", "residue_id", "insertion_code", "residue_name")
    )
    if not all(isinstance(value, str) for value in values) or values[3] not in AA3_TO_1:
        raise ValueError("invalid standard residue")
    return values


def _load_roster(inputs: Path) -> list[State]:
    raw_inputs = {name: read_regular(inputs, name) for name in INPUT_HASHES}
    for name, expected in INPUT_HASHES.items():
        if sha256(raw_inputs[name]) != expected:
            raise ValueError(f"frozen input hash drifted: {name}")
    condition = parse_json(raw_inputs["condition.json"], "condition manifest")
    sequence = parse_json(raw_inputs["sequence.json"], "sequence manifest")
    parent = parse_json(raw_inputs["parent.json"], "parent manifest")
    environment = parse_json(raw_inputs["environment.json"], "environment manifest")
    _require_keys(
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
    _require_keys(
        sequence,
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
        },
        "sequence manifest",
    )
    _require_keys(
        parent,
        {
            "artifact_kind",
            "candidate_id",
            "catalog_archive",
            "closed_capabilities",
            "contract",
            "entities",
            "entity_count",
            "freezer",
            "parent_record_count",
            "parent_records_sha256",
            "roster",
            "scope",
            "shards",
            "source_commitment",
            "source_relative_path_pattern",
        },
        "parent manifest",
    )
    _require_keys(
        environment,
        {
            "artifact_kind",
            "base_image",
            "build_recipe",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "conversion",
            "freezer",
            "inspection",
            "ph_regimes",
            "required_platform",
            "runtime_sif",
            "scope",
            "wheels",
        },
        "environment manifest",
    )
    if not (
        condition["entity_count"]
        == sequence["entity_count"]
        == parent["entity_count"]
        == ENTITY_COUNT
        and len(condition["entities"])
        == len(sequence["entities"])
        == len(parent["entities"])
        == ENTITY_COUNT
        and condition["state_counts"]
        == {"observed": 119, "state_ambiguous": 5, "state_missing": 11}
        and environment["required_platform"] == "Reference"
        and environment["inspection"]["openmm_version_full"] == "8.6.0.dev-c6173db"
        and environment["runtime_sif"]["sha256"]
        == "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
    ):
        raise ValueError("frozen manifest invariant drifted")
    condition_states = states_from_condition(condition["entities"])
    sequence_by_uid: dict[str, dict[str, Any]] = {}
    for row in sequence["entities"]:
        _require_keys(
            row,
            {
                "bmrb_id",
                "entity_uid",
                "reference_pdb",
                "residue_count",
                "residues",
                "sequence_one_letter",
                "sequence_sha256",
            },
            "sequence entity",
        )
        uid = row["entity_uid"]
        reference = row["reference_pdb"]
        if (
            not isinstance(uid, str)
            or uid in sequence_by_uid
            or not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or _require_hash(reference["sha256"], "reference PDB")
            != reference["sha256"]
        ):
            raise ValueError("sequence identity or reference drifted")
        residues = tuple(_residue_tuple(item) for item in row["residues"])
        expected_sequence = "".join(AA3_TO_1[item[3]] for item in residues)
        if (
            row["residue_count"] != len(residues)
            or row["sequence_one_letter"] != expected_sequence
        ):
            raise ValueError("sequence residue identity drifted")
        if sha256(expected_sequence.encode("ascii")) != _require_hash(
            row["sequence_sha256"], "sequence"
        ):
            raise ValueError("sequence hash drifted")
        sequence_by_uid[uid] = row
    parent_by_uid: dict[str, dict[str, Any]] = {}
    for row in parent["entities"]:
        _require_keys(
            row,
            {
                "bmrb_id",
                "entity_uid",
                "heavy_atom_count",
                "heavy_atom_topology_sha256",
                "observer_fold",
                "records_sha256",
                "split",
                "support_count",
                "support_index_digest",
            },
            "parent entity",
        )
        uid = row["entity_uid"]
        if (
            not isinstance(uid, str)
            or uid in parent_by_uid
            or not isinstance(row["support_count"], int)
            or row["support_count"] < SUPPORT_INDEX
            or not isinstance(row["heavy_atom_count"], int)
        ):
            raise ValueError("parent support count drifted")
        for field in (
            "heavy_atom_topology_sha256",
            "records_sha256",
            "support_index_digest",
        ):
            _require_hash(row[field], field)
        parent_by_uid[uid] = row
    inventory = parse_json(
        read_regular(inputs, "support1_inventory.json"), "support-1 inventory"
    )
    _require_keys(
        inventory,
        {"contract", "entities", "entity_count", "support_index"},
        "support-1 inventory",
    )
    if (
        inventory["entity_count"] != ENTITY_COUNT
        or inventory["support_index"] != SUPPORT_INDEX
        or len(inventory["entities"]) != ENTITY_COUNT
    ):
        raise ValueError("support-1 inventory count drifted")
    inventory_by_uid: dict[str, dict[str, Any]] = {}
    for row in inventory["entities"]:
        _require_keys(
            row,
            {
                "entity_uid",
                "heavy_coordinate_sha256",
                "heavy_topology_sha256",
                "path",
                "pdb_sha256",
            },
            "support-1 inventory entity",
        )
        uid = row["entity_uid"]
        if (
            not isinstance(uid, str)
            or uid in inventory_by_uid
            or not isinstance(row["path"], str)
        ):
            raise ValueError("support-1 inventory identity drifted")
        for field in ("heavy_coordinate_sha256", "heavy_topology_sha256", "pdb_sha256"):
            _require_hash(row[field], field)
        inventory_by_uid[uid] = row
    condition_uids = {uid for uid, _, _ in condition_states}
    condition_bmrb = {
        row["entity_uid"]: row["bmrb_id"] for row in condition["entities"]
    }
    if len(condition_bmrb) != ENTITY_COUNT or not all(
        isinstance(value, str) and value for value in condition_bmrb.values()
    ):
        raise ValueError("condition BMRB identity roster drifted")
    if not (
        condition_uids
        == set(sequence_by_uid)
        == set(parent_by_uid)
        == set(inventory_by_uid)
    ):
        raise ValueError("manifest cross-join identity drifted")
    states: list[State] = []
    for uid, branch_id, ph in condition_states:
        sequence_row, parent_row, support = (
            sequence_by_uid[uid],
            parent_by_uid[uid],
            inventory_by_uid[uid],
        )
        if not (
            condition_bmrb[uid] == sequence_row["bmrb_id"] == parent_row["bmrb_id"]
        ):
            raise ValueError("BMRB identity cross-join drifted")
        if support["heavy_topology_sha256"] != parent_row["heavy_atom_topology_sha256"]:
            raise ValueError("support heavy topology does not bind parent")
        # The raw PDB hash is frozen by the sequence manifest; the staged inventory
        # additionally commits its parsed heavy coordinate record before OpenMM runs.
        if support["pdb_sha256"] != sequence_row["reference_pdb"]["sha256"]:
            raise ValueError("support-1 PDB does not bind sequence manifest")
        states.append(
            State(
                uid,
                branch_id,
                ph,
                support["path"],
                support["pdb_sha256"],
                parent_row["heavy_atom_count"],
                support["heavy_topology_sha256"],
                support["heavy_coordinate_sha256"],
                sequence_row["sequence_one_letter"],
                tuple(_residue_tuple(item) for item in sequence_row["residues"]),
            )
        )
    if len(states) != STATE_COUNT:
        raise ValueError("cross-joined state count drifted")
    return states


def _pdb_records(
    raw: bytes,
) -> tuple[list[dict[str, Any]], tuple[tuple[str, str, str, str], ...], str]:
    records: list[dict[str, Any]] = []
    residues: list[tuple[str, str, str, str]] = []
    residue_seen: set[tuple[str, str, str, str]] = set()
    segment = 0
    for number, line in enumerate(raw.splitlines(), 1):
        if line.startswith(b"TER"):
            segment += 1
            continue
        if line[:6] not in {b"ATOM  ", b"HETATM"}:
            continue
        if line[:6] != b"ATOM  " or len(line) < 78 or line[16:17] != b" ":
            raise ValueError(f"unsupported PDB atom record at line {number}")
        try:
            identity = {
                "_record_type": line[:6].decode("ascii").strip(),
                "_segment": segment,
                "atom_name": line[12:16].decode("ascii").strip(),
                "chain_id": line[21:22].decode("ascii"),
                "element": line[76:78].decode("ascii").strip().upper(),
                "insertion_code": line[26:27].decode("ascii"),
                "residue_id": line[22:26].decode("ascii").strip(),
                "residue_name": line[17:20].decode("ascii").strip(),
            }
            xyz = [
                Decimal(line[start:stop].decode("ascii").strip())
                for start, stop in ((30, 38), (38, 46), (46, 54))
            ]
        except (UnicodeDecodeError, InvalidOperation) as error:
            raise ValueError(f"invalid PDB atom record at line {number}") from error
        if (
            not identity["atom_name"]
            or not identity["element"]
            or not all(value.is_finite() for value in xyz)
        ):
            raise ValueError(f"invalid PDB atom fields at line {number}")
        if any(-999.999 > value or value > 9999.999 for value in xyz):
            raise ValueError(f"PDB coordinate out of range at line {number}")
        identity["x"], identity["y"], identity["z"] = (
            format(value, ".3f") for value in xyz
        )
        records.append(identity)
        residue = tuple(
            identity[field]
            for field in ("chain_id", "residue_id", "insertion_code", "residue_name")
        )
        if residue not in residue_seen:
            if residue[3] not in AA3_TO_1:
                raise ValueError("nonstandard residue in support-1 PDB")
            residue_seen.add(residue)
            residues.append(residue)
    if not records:
        raise ValueError("support-1 PDB has no ATOM records")
    identities = [
        tuple(
            record[key]
            for key in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate PDB atom identity")
    return records, tuple(residues), "".join(AA3_TO_1[row[3]] for row in residues)


def _heavy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in records
        if row["element"] not in {"H", "D", "T"}
    ]


def _heavy_records_numerically_exact(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> bool:
    identity = (
        "atom_name",
        "chain_id",
        "element",
        "insertion_code",
        "residue_id",
        "residue_name",
    )
    return len(left) == len(right) and all(
        all(a[key] == b[key] for key in identity)
        and all(Decimal(a[key]) == Decimal(b[key]) for key in ("x", "y", "z"))
        for a, b in zip(left, right, strict=True)
    )


def _heavy_hashes(records: list[dict[str, Any]]) -> tuple[str, str]:
    parsed_heavy = [row for row in records if row["element"] not in {"H", "D", "T"}]
    topology = hashlib.sha256()
    for row in parsed_heavy:
        identity = (
            row["_record_type"],
            row["atom_name"].upper(),
            row["residue_name"].upper(),
            row["_segment"],
            row["chain_id"].strip() or "_",
            int(row["residue_id"]),
            row["insertion_code"].strip(),
            row["element"],
        )
        topology.update(json.dumps(identity, separators=(",", ":")).encode())
        topology.update(b"\n")
    coordinates = [
        {
            key: row[key]
            for key in (
                "atom_name",
                "chain_id",
                "element",
                "insertion_code",
                "residue_id",
                "residue_name",
                "x",
                "y",
                "z",
            )
        }
        for row in parsed_heavy
    ]
    return topology.hexdigest(), sha256(canonical_bytes(coordinates))


def _require_runtime(openmm: Any, app: Any) -> None:
    if openmm.version.full_version != "8.6.0.dev-c6173db":
        raise ValueError("OpenMM exact version drifted")
    root = Path(app.__file__).parent
    for relative, expected in RUNTIME_FILE_HASHES.items():
        if sha256((root / relative).read_bytes()) != expected:
            raise ValueError(f"OpenMM runtime file hash drifted: {relative}")


def _atom_key(atom: Any) -> tuple[str, str, str, str, str, str]:
    residue = atom.residue
    element = atom.element.symbol.upper() if atom.element is not None else ""
    return (
        residue.chain.id,
        str(residue.id),
        residue.insertionCode,
        residue.name,
        atom.name,
        element,
    )


def _openmm_heavy_records(
    topology: Any, positions: Any, unit: Any
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for atom, position in zip(topology.atoms(), positions):
        if atom.element is None or atom.element.atomic_number == 1:
            continue
        chain, residue_id, insertion, residue, name, element = _atom_key(atom)
        xyz = position.value_in_unit(unit.angstrom)
        result.append(
            {
                "atom_name": name,
                "chain_id": chain,
                "element": element,
                "insertion_code": insertion,
                "residue_id": residue_id,
                "residue_name": residue,
                "x": format(Decimal(str(float(xyz[0]))), ".3f"),
                "y": format(Decimal(str(float(xyz[1]))), ".3f"),
                "z": format(Decimal(str(float(xyz[2]))), ".3f"),
            }
        )
    return result


def _distance(left: Any, right: Any, unit: Any) -> float:
    vector = left - right
    return float(
        math.sqrt(
            sum(float(value * value) for value in vector.value_in_unit(unit.angstrom))
        )
    )


def hydrogen_distance_ok(parent_element: str, distance: float) -> bool:
    lower, upper = (1.1, 1.5) if parent_element == "S" else (0.7, 1.3)
    return lower <= distance <= upper


def _physicality(
    modeller: Any, forcefield: Any, platform: Any, unit: Any, openmm: Any
) -> dict[str, Any]:
    atoms = list(modeller.topology.atoms())
    positions = list(modeller.positions)
    if len(atoms) != len(positions) or not atoms:
        raise ValueError("atom-position inventory invalid")
    coordinates = [position.value_in_unit(unit.angstrom) for position in positions]
    if not all(math.isfinite(float(value)) for xyz in coordinates for value in xyz):
        raise ValueError("non-finite coordinate")
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in atoms}
    atom_index = {atom: index for index, atom in enumerate(atoms)}
    for left, right in modeller.topology.bonds():
        adjacency[left].append(right)
        adjacency[right].append(left)
        left_element = left.element.symbol.upper() if left.element else ""
        right_element = right.element.symbol.upper() if right.element else ""
        if left_element != "H" and right_element != "H":
            distance = _distance(
                positions[atom_index[left]], positions[atom_index[right]], unit
            )
            if not 1.0 <= distance <= 2.3:
                raise ValueError("heavy-heavy bond distance outside 1.0..2.3 A")
    for index, left in enumerate(positions):
        for right in positions[index + 1 :]:
            if _distance(left, right, unit) < 0.5:
                raise ValueError("distinct atoms closer than 0.5 A")
    limits = {"H": 1, "C": 4, "N": 4, "O": 2, "S": 6}
    hydrogen_count = 0
    for atom in atoms:
        element = atom.element.symbol.upper() if atom.element else ""
        if element not in limits or len(adjacency[atom]) > limits[element]:
            raise ValueError("invalid elemental valence")
        if element != "H":
            continue
        hydrogen_count += 1
        parents = [
            other
            for other in adjacency[atom]
            if (other.element and other.element.atomic_number != 1)
        ]
        if len(parents) != 1 or len(adjacency[atom]) != 1:
            raise ValueError("hydrogen lacks exactly one heavy parent")
        parent = parents[0]
        parent_element = parent.element.symbol.upper()
        distance = _distance(
            positions[atom_index[atom]], positions[atom_index[parent]], unit
        )
        if not hydrogen_distance_ok(parent_element, distance):
            raise ValueError("hydrogen-heavy distance outside physicality bound")
        others = [other for other in adjacency[parent] if other is not atom]
        if others:
            first = others[0]
            a = positions[atom_index[atom]] - positions[atom_index[parent]]
            b = positions[atom_index[first]] - positions[atom_index[parent]]
            av, bv = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom)
            cosine = sum(float(x * y) for x, y in zip(av, bv)) / math.sqrt(
                sum(float(x * x) for x in av) * sum(float(y * y) for y in bv)
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if not 55.0 <= angle <= 180.0:
                raise ValueError("hydrogen bond angle outside physicality bound")
    if hydrogen_count == 0:
        raise ValueError("OpenMM produced no hydrogens")
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        for left, right in pairwise(residues):
            left_c = next((atom for atom in left.atoms() if atom.name == "C"), None)
            right_n = next((atom for atom in right.atoms() if atom.name == "N"), None)
            if left_c is not None and right_n is not None:
                distance = _distance(
                    positions[atom_index[left_c]], positions[atom_index[right_n]], unit
                )
                if not 1.0 <= distance <= 1.8:
                    raise ValueError("backbone continuity failed")
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPositions(modeller.positions)
        energy = float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context
        del integrator
    if not math.isfinite(energy):
        raise ValueError("potential energy is non-finite")
    return {
        "atom_count": len(atoms),
        "hydrogen_count": hydrogen_count,
        "potential_energy_kj_per_mol": energy,
    }


def _signature(topology: Any) -> list[dict[str, str]]:
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in topology.atoms()}
    for left, right in topology.bonds():
        adjacency[left].append(right)
        adjacency[right].append(left)
    output = []
    for atom in topology.atoms():
        if atom.element is None or atom.element.atomic_number != 1:
            continue
        parent = next(
            (
                other
                for other in adjacency[atom]
                if other.element and other.element.atomic_number != 1
            ),
            None,
        )
        if parent is None:
            raise ValueError("signature hydrogen lacks heavy parent")
        chain, residue_id, insertion, residue, name, _ = _atom_key(atom)
        output.append(
            {
                "chain_id": chain,
                "hydrogen_name": name,
                "heavy_parent": parent.name,
                "insertion_code": insertion,
                "residue_id": residue_id,
                "residue_name": residue,
            }
        )
    return sorted(output, key=canonical_bytes)


def _exclusive(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _state_dir(output: Path, state: State, repeat: int) -> Path:
    if repeat not in (0, 1):
        raise ValueError("repeat must be zero or one")
    return output / f"{safe_token(state.entity_uid)}--{state.branch_id}--repeat{repeat}"


def _worker(
    inputs: Path, output: Path, entity_uid: str, branch_id: str, repeat: int
) -> None:
    _require_frozen_source(inputs)
    states = _load_roster(inputs)
    state = next(
        (
            item
            for item in states
            if item.entity_uid == entity_uid and item.branch_id == branch_id
        ),
        None,
    )
    if state is None:
        raise ValueError("requested state is absent from cross-joined roster")
    role = f"generation-repeat-{repeat}"
    seed = seed_for(entity_uid, branch_id, role)
    if os.environ.get("PYTHONHASHSEED") != pythonhashseed(seed):
        raise ValueError("worker does not have its derived PYTHONHASHSEED")
    random.seed(seed)
    raw_pdb = read_regular(inputs, state.pdb_relative)
    if sha256(raw_pdb) != state.pdb_sha256:
        raise ValueError("staged support-1 raw PDB hash drifted")
    source_records, residues, sequence = _pdb_records(raw_pdb)
    topology_hash, coordinate_hash = _heavy_hashes(source_records)
    if residues != state.residues or sequence != state.sequence:
        raise ValueError("staged support-1 parsed sequence identity drifted")
    if len(_heavy_records(source_records)) != state.heavy_atom_count:
        raise ValueError("staged support-1 heavy atom count drifted")
    if (
        topology_hash != state.heavy_topology_sha256
        or coordinate_hash != state.heavy_coordinate_sha256
    ):
        raise ValueError("staged support-1 parsed heavy record hash drifted")
    try:
        import numpy  # type: ignore[import-not-found]
        import openmm  # type: ignore[import-not-found]
        from openmm import Platform, app, unit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("frozen OpenMM runtime is required for a worker") from error
    numpy.random.seed(seed % (2**32))
    _require_runtime(openmm, app)
    pdb = app.PDBFile(str(_under(inputs, state.pdb_relative)))
    source_heavy = _heavy_records(source_records)
    if len(list(pdb.topology.atoms())) != len(source_records):
        raise ValueError("PDBFile atom inventory differs from parsed source")
    pre_heavy = _openmm_heavy_records(pdb.topology, pdb.positions, unit)
    if not _heavy_records_numerically_exact(pre_heavy, source_heavy):
        raise ValueError("PDBFile did not preserve parsed input heavy records exactly")
    source_keys = {
        tuple(
            row[key]
            for key in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for row in source_heavy
    }
    modeller = app.Modeller(pdb.topology, pdb.positions)
    delete = [
        atom for atom in modeller.topology.atoms() if _atom_key(atom) not in source_keys
    ]
    modeller.delete(delete)  # Deletes every input hydrogen isotope before pH selection.
    if not _heavy_records_numerically_exact(
        _openmm_heavy_records(modeller.topology, modeller.positions, unit),
        source_heavy,
    ):
        raise ValueError("heavy records changed while removing input hydrogens")
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")
    platform = Platform.getPlatformByName("Reference")
    variants = modeller.addHydrogens(
        forcefield, pH=float(state.ph), variants=None, platform=platform
    )
    post_heavy = _openmm_heavy_records(modeller.topology, modeller.positions, unit)
    if not _heavy_records_numerically_exact(post_heavy, source_heavy):
        raise ValueError("addHydrogens changed parsed parent heavy records")
    physicality = _physicality(modeller, forcefield, platform, unit, openmm)
    directory = _state_dir(output, state, repeat)
    os.mkdir(directory, 0o700)
    pdb_path = directory / "protonated.pdb"
    descriptor = os.open(pdb_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", closefd=False) as handle:
            # writeFile() adds a wall-clock-dependent header.  Model+footer is
            # the complete deterministic coordinate/bond payload we compare.
            app.PDBFile.writeModel(
                modeller.topology, modeller.positions, handle, keepIds=True
            )
            app.PDBFile.writeFooter(modeller.topology, handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(pdb_path, 0o444)
    emitted_records, _, _ = _pdb_records(read_regular(directory, "protonated.pdb"))
    if not _heavy_records_numerically_exact(
        _heavy_records(emitted_records), source_heavy
    ):
        raise ValueError("emitted PDB parsed heavy records are not exact")
    metadata = {
        "branch_id": state.branch_id,
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "entity_token": safe_token(state.entity_uid),
        "entity_uid": state.entity_uid,
        "emitted_parsed_heavy_records_exact": True,
        "input_support_raw_sha256": state.pdb_sha256,
        "parsed_heavy_coordinate_sha256": coordinate_hash,
        "parsed_heavy_records_exact": True,
        "parsed_heavy_topology_sha256": topology_hash,
        "ph": state.ph,
        "physicality": physicality,
        "protonated_pdb_sha256": sha256(read_regular(directory, "protonated.pdb")),
        "protonation_signature": _signature(modeller.topology),
        "repeat": repeat,
        "returned_variants": list(variants),
        "support_index": SUPPORT_INDEX,
    }
    _exclusive(directory / "metadata.json", canonical_bytes(metadata) + b"\n")
    os.chmod(directory, 0o555)


def _run(inputs: Path, output: Path) -> None:
    _require_frozen_source(inputs)
    states = _load_roster(inputs)
    os.mkdir(output, 0o700)  # O_EXCL equivalent for the single run root.
    tasks: list[tuple[list[str], dict[str, str]]] = []
    for state in states:
        for repeat in (0, 1):
            role = f"generation-repeat-{repeat}"
            seed = seed_for(state.entity_uid, state.branch_id, role)
            environment = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": pythonhashseed(seed),
                "PREFLIGHT_RNG_SEED": str(seed),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--inputs",
                str(inputs),
                "--output",
                str(output),
                "--entity-uid",
                state.entity_uid,
                "--branch-id",
                state.branch_id,
                "--repeat",
                str(repeat),
            ]
            tasks.append((command, environment))
    for offset in range(0, len(tasks), MAX_PARALLEL_WORKERS):
        batch = tasks[offset : offset + MAX_PARALLEL_WORKERS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [
                pool.submit(subprocess.run, command, check=True, env=environment)
                for command, environment in batch
            ]
            for future in futures:
                future.result()
    expected = {
        _state_dir(output, state, repeat).name for state in states for repeat in (0, 1)
    }
    if len(expected) != 398 or {path.name for path in output.iterdir()} != expected:
        raise ValueError("generator output inventory or quota drifted")
    _exclusive(
        output / "generator_inventory.json",
        canonical_bytes(
            {
                "candidate_id": CANDIDATE_ID,
                "closed_capabilities": CLOSED_CAPABILITIES,
                "repeat_count": 2,
                "state_count": STATE_COUNT,
                "support_index": SUPPORT_INDEX,
            }
        )
        + b"\n",
    )
    os.chmod(output, 0o555)


def self_test() -> int:
    base = {
        "atom_name": "CA",
        "chain_id": "A",
        "element": "C",
        "insertion_code": "",
        "residue_id": "1",
        "residue_name": "ALA",
        "x": "-0.000",
        "y": "1.000",
        "z": "2.000",
    }
    equivalent = dict(base, x="0.000")
    changed = dict(equivalent, atom_name="CB")
    if not _heavy_records_numerically_exact([base], [equivalent]):
        raise AssertionError("signed zero must be numerically exact")
    if _heavy_records_numerically_exact([base], [changed]):
        raise AssertionError("heavy identity drift was accepted")
    rows = []
    for index in range(119):
        rows.append(
            {
                "bmrb_id": str(index),
                "condition_state": "observed",
                "deposited_ph": "7.0",
                "entity_uid": f"observed/{index}",
                "pH_source": "x",
                "recovery_v6_hold_reasons": [],
                "temperature_ionic_diagnostics": {},
            }
        )
    for index in range(16):
        rows.append(
            {
                "bmrb_id": f"u{index}",
                "condition_state": "state_missing" if index < 11 else "state_ambiguous",
                "deposited_ph": None,
                "entity_uid": f"unresolved/{index}",
                "pH_source": None,
                "recovery_v6_hold_reasons": [],
                "temperature_ionic_diagnostics": {},
            }
        )
    assert len(states_from_condition(rows)) == STATE_COUNT
    assert safe_token("same/slash") != safe_token("same:slash")
    assert hydrogen_distance_ok("S", 1.4) and not hydrogen_distance_ok("C", 1.4)
    for bad in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
        try:
            parse_json(bad, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed JSON accepted")
    return 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--entity-uid")
    parser.add_argument("--branch-id")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(f"STATUS PASS_STATIC_SELF_TEST checks={self_test()}")
        return 0
    if args.inputs is None or args.output is None:
        parser.error("--inputs and --output are required")
    if args.worker:
        if args.entity_uid is None or args.branch_id is None or args.repeat is None:
            parser.error("worker requires entity, branch, and repeat")
        _worker(args.inputs, args.output, args.entity_uid, args.branch_id, args.repeat)
    else:
        _run(args.inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
