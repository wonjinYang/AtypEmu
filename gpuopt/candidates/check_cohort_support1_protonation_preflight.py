# ruff: noqa: TRY004
"""Independently replay the HOLD support-1 protonation preflight.

The checker intentionally owns its manifest validation, PDB parsing, chemistry audit,
and child-process replay.  It never imports the generator.
"""

from __future__ import annotations

import argparse
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
ENTITY_COUNT, STATE_COUNT = 135, 199
INPUT_HASHES = {
    "environment.json": "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
    "condition.json": "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618",
    "sequence.json": "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
    "parent.json": "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
}
FROZEN_PLAN_FILE = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_plan_v1.json"
)
FROZEN_COMMITMENT_FILE = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_"
    "source_commitment_v2.json"
)
FROZEN_COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_"
    "source_commitment_v2"
)
AA = {
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
HEX = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SLUG = re.compile(r"[^a-zA-Z0-9_.-]+")
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
class ReplayState:
    uid: str
    branch: str
    ph: str
    pdb_path: str
    pdb_hash: str
    heavy_count: int
    heavy_topology: str
    heavy_coordinates: str
    sequence: str
    residues: tuple[tuple[str, str, str, str], ...]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def no_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for key, value in pairs:
        if key in answer:
            raise ValueError(f"duplicate JSON key: {key}")
        answer[key] = value
    return answer


def object_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        answer = json.loads(raw, object_pairs_hook=no_dupes, parse_constant=no_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(answer, dict):
        raise ValueError(f"{label} is not an object")
    return answer


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def token(uid: str) -> str:
    if not isinstance(uid, str) or not uid or "\x00" in uid:
        raise ValueError("invalid entity uid")
    return (
        f"{SLUG.sub('-', uid).strip('.-')[:48] or 'entity'}-{digest(uid.encode())[:20]}"
    )


def checker_seed(uid: str, branch: str) -> int:
    material = "\0".join((CANDIDATE_ID, uid, str(SUPPORT_INDEX), branch))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def require_committed_sources(inputs: Path) -> None:
    """Independently require the staged, exact four-file source commitment."""
    scripts = inputs.parent / "scripts"
    receipt = object_json(regular(scripts, FROZEN_COMMITMENT_FILE), "source commitment")
    exact_keys(
        receipt,
        {"candidate_id", "contract", "files", "git_commit", "status"},
        "source commitment",
    )
    if not (
        receipt["candidate_id"] == CANDIDATE_ID
        and receipt["contract"] == FROZEN_COMMITMENT_CONTRACT
        and receipt["status"] == "FROZEN_COMMITTED"
        and isinstance(receipt["git_commit"], str)
        and GIT_COMMIT.fullmatch(receipt["git_commit"])
    ):
        raise PermissionError("checker source commitment is not frozen")
    staged = {
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
        "protonation_preflight_plan_v1.json": FROZEN_PLAN_FILE,
    }
    hashes = receipt["files"]
    if not isinstance(hashes, dict) or set(hashes) != set(staged):
        raise ValueError("checker source commitment inventory drifted")
    for repository_name, stage_name in staged.items():
        wanted = hashes[repository_name]
        if (
            not isinstance(wanted, str)
            or HEX.fullmatch(wanted) is None
            or digest(regular(scripts, stage_name)) != wanted
        ):
            raise ValueError(f"checker source hash mismatch: {repository_name}")
    plan = object_json(regular(scripts, FROZEN_PLAN_FILE), "frozen plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("state") == "HOLD_PREFLIGHT_SOURCE_FROZEN_UNRUN"
        and plan.get("source_commitment", {}).get("status") == "FROZEN_COMMITTED"
    ):
        raise PermissionError("checker plan is not source-frozen for HOLD execution")


def pythonhashseed(seed: int) -> str:
    return str(seed % (2**32))


def _path(root: Path, relative: str) -> Path:
    piece = Path(relative)
    if (
        piece.is_absolute()
        or not piece.parts
        or any(item in {"", ".", ".."} for item in piece.parts)
    ):
        raise ValueError("unsafe relative path")
    current = root
    for item in piece.parts:
        current /= item
        data = os.lstat(current)
        if stat.S_ISLNK(data.st_mode):
            raise ValueError(f"symlink input rejected: {relative}")
    return current


def regular(root: Path, relative: str, maximum: int = 4_000_000) -> bytes:
    descriptor = os.open(
        _path(root, relative), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"invalid regular file: {relative}")
        pieces: list[bytes] = []
        total = 0
        while block := os.read(descriptor, 1_048_576):
            total += len(block)
            if total > maximum:
                raise ValueError(f"input too large: {relative}")
            pieces.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"input changed while reading: {relative}")
        return b"".join(pieces)
    finally:
        os.close(descriptor)


def exact_keys(row: dict[str, Any], names: set[str], label: str) -> None:
    if set(row) != names:
        raise ValueError(f"{label} exact schema mismatch")


def hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX.fullmatch(value) is None:
        raise ValueError(f"bad {label} SHA256")
    return value


def ph_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("pH must be string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid pH") from error
    if not number.is_finite() or number < 0 or number > 14:
        raise ValueError("pH range")
    return format(number, "f")


def derive_branches(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    ids: set[str] = set()
    seen_observed = seen_open = 0
    fields = {
        "bmrb_id",
        "condition_state",
        "deposited_ph",
        "entity_uid",
        "pH_source",
        "recovery_v6_hold_reasons",
        "temperature_ionic_diagnostics",
    }
    for row in rows:
        exact_keys(row, fields, "condition row")
        uid = row["entity_uid"]
        if not isinstance(uid, str) or not uid or uid in ids:
            raise ValueError("condition identities not unique")
        ids.add(uid)
        if row["condition_state"] == "observed":
            records.append((uid, "observed-0", ph_string(row["deposited_ph"])))
            seen_observed += 1
        elif row["condition_state"] in {"state_missing", "state_ambiguous"}:
            if row["deposited_ph"] is not None:
                raise ValueError("unresolved state was assigned pH")
            records += [
                (uid, f"regime-{number}", ph) for number, ph in enumerate(MIDPOINTS)
            ]
            seen_open += 1
        else:
            raise ValueError("unknown condition state")
    if len(rows) != ENTITY_COUNT or (seen_observed, seen_open, len(records)) != (
        119,
        16,
        STATE_COUNT,
    ):
        raise ValueError("condition branch quota mismatch")
    return sorted(records)


def residue(row: Any) -> tuple[str, str, str, str]:
    if not isinstance(row, dict):
        raise ValueError("residue is not object")
    exact_keys(
        row, {"chain_id", "insertion_code", "residue_id", "residue_name"}, "residue"
    )
    answer = tuple(
        row[name]
        for name in ("chain_id", "residue_id", "insertion_code", "residue_name")
    )
    if not all(isinstance(item, str) for item in answer) or answer[3] not in AA:
        raise ValueError("unsupported residue")
    return answer


def independent_roster(inputs: Path) -> list[ReplayState]:
    raw = {name: regular(inputs, name) for name in INPUT_HASHES}
    if any(digest(raw[name]) != expected for name, expected in INPUT_HASHES.items()):
        raise ValueError("bound manifest raw hash mismatch")
    condition = object_json(raw["condition.json"], "condition manifest")
    sequence = object_json(raw["sequence.json"], "sequence manifest")
    parent = object_json(raw["parent.json"], "parent manifest")
    runtime = object_json(raw["environment.json"], "environment manifest")
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
    exact_keys(
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
    exact_keys(
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
    exact_keys(
        runtime,
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
        and runtime["required_platform"] == "Reference"
        and runtime["inspection"]["openmm_version_full"] == "8.6.0.dev-c6173db"
        and runtime["runtime_sif"]["sha256"]
        == "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
    ):
        raise ValueError("manifest invariant mismatch")
    sequence_rows: dict[str, dict[str, Any]] = {}
    for row in sequence["entities"]:
        exact_keys(
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
            "sequence row",
        )
        uid, ref = row["entity_uid"], row["reference_pdb"]
        if (
            not isinstance(uid, str)
            or uid in sequence_rows
            or not isinstance(ref, dict)
        ):
            raise ValueError("bad sequence uid or reference")
        exact_keys(ref, {"path", "sha256"}, "reference PDB")
        if not isinstance(ref["path"], str):
            raise ValueError("PDB path invalid")
        hash_value(ref["sha256"], "reference PDB")
        residues = tuple(residue(value) for value in row["residues"])
        letters = "".join(AA[value[3]] for value in residues)
        if (
            row["residue_count"] != len(residues)
            or row["sequence_one_letter"] != letters
            or digest(letters.encode())
            != hash_value(row["sequence_sha256"], "sequence")
        ):
            raise ValueError("sequence validation failed")
        sequence_rows[uid] = row
    parent_rows: dict[str, dict[str, Any]] = {}
    for row in parent["entities"]:
        exact_keys(
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
            "parent row",
        )
        uid = row["entity_uid"]
        if (
            not isinstance(uid, str)
            or uid in parent_rows
            or not isinstance(row["support_count"], int)
            or row["support_count"] < 1
        ):
            raise ValueError("parent support-1 absent")
        if not isinstance(row["heavy_atom_count"], int):
            raise ValueError("parent heavy count invalid")
        for name in (
            "heavy_atom_topology_sha256",
            "records_sha256",
            "support_index_digest",
        ):
            hash_value(row[name], name)
        parent_rows[uid] = row
    inventory = object_json(
        regular(inputs, "support1_inventory.json"), "support inventory"
    )
    exact_keys(
        inventory,
        {"contract", "entities", "entity_count", "support_index"},
        "support inventory",
    )
    if (
        inventory["entity_count"] != ENTITY_COUNT
        or inventory["support_index"] != 1
        or len(inventory["entities"]) != ENTITY_COUNT
    ):
        raise ValueError("support inventory quota mismatch")
    support_rows: dict[str, dict[str, Any]] = {}
    for row in inventory["entities"]:
        exact_keys(
            row,
            {
                "entity_uid",
                "heavy_coordinate_sha256",
                "heavy_topology_sha256",
                "path",
                "pdb_sha256",
            },
            "support inventory row",
        )
        uid = row["entity_uid"]
        if (
            not isinstance(uid, str)
            or uid in support_rows
            or not isinstance(row["path"], str)
        ):
            raise ValueError("support inventory identity invalid")
        for name in ("heavy_coordinate_sha256", "heavy_topology_sha256", "pdb_sha256"):
            hash_value(row[name], name)
        support_rows[uid] = row
    branches = derive_branches(condition["entities"])
    condition_bmrb = {
        row["entity_uid"]: row["bmrb_id"] for row in condition["entities"]
    }
    if len(condition_bmrb) != ENTITY_COUNT or not all(
        isinstance(value, str) and value for value in condition_bmrb.values()
    ):
        raise ValueError("independent condition BMRB roster failed")
    if (
        {item[0] for item in branches} != set(sequence_rows)
        or set(sequence_rows) != set(parent_rows)
        or set(parent_rows) != set(support_rows)
    ):
        raise ValueError("independent manifest cross-join failed")
    output: list[ReplayState] = []
    for uid, branch, ph in branches:
        seq, parent_row, support = (
            sequence_rows[uid],
            parent_rows[uid],
            support_rows[uid],
        )
        if (
            condition_bmrb[uid] != seq["bmrb_id"]
            or seq["bmrb_id"] != parent_row["bmrb_id"]
            or support["pdb_sha256"] != seq["reference_pdb"]["sha256"]
            or support["heavy_topology_sha256"]
            != parent_row["heavy_atom_topology_sha256"]
        ):
            raise ValueError("identity, raw PDB, or parent topology binding failed")
        output.append(
            ReplayState(
                uid,
                branch,
                ph,
                support["path"],
                support["pdb_sha256"],
                parent_row["heavy_atom_count"],
                support["heavy_topology_sha256"],
                support["heavy_coordinate_sha256"],
                seq["sequence_one_letter"],
                tuple(residue(item) for item in seq["residues"]),
            )
        )
    if len(output) != STATE_COUNT:
        raise ValueError("independent roster state quota failed")
    return output


def pdb_records(
    raw: bytes,
) -> tuple[list[dict[str, str]], tuple[tuple[str, str, str, str], ...], str]:
    atoms: list[dict[str, str]] = []
    chain_residues: list[tuple[str, str, str, str]] = []
    seen_residues: set[tuple[str, str, str, str]] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if line[:6] not in {b"ATOM  ", b"HETATM"}:
            continue
        if line[:6] != b"ATOM  " or len(line) < 78 or line[16:17] != b" ":
            raise ValueError(f"invalid PDB atom line {line_number}")
        try:
            xyz = [
                Decimal(line[start:stop].decode().strip())
                for start, stop in ((30, 38), (38, 46), (46, 54))
            ]
            atom = {
                "atom_name": line[12:16].decode().strip(),
                "chain_id": line[21:22].decode(),
                "element": line[76:78].decode().strip().upper(),
                "insertion_code": line[26:27].decode(),
                "residue_id": line[22:26].decode().strip(),
                "residue_name": line[17:20].decode().strip(),
                "x": format(xyz[0], ".3f"),
                "y": format(xyz[1], ".3f"),
                "z": format(xyz[2], ".3f"),
            }
        except (InvalidOperation, UnicodeDecodeError) as error:
            raise ValueError(f"malformed PDB atom line {line_number}") from error
        if (
            not atom["atom_name"]
            or not atom["element"]
            or not all(value.is_finite() for value in xyz)
        ):
            raise ValueError("missing PDB atom identity")
        atoms.append(atom)
        residue_id = tuple(
            atom[name]
            for name in ("chain_id", "residue_id", "insertion_code", "residue_name")
        )
        if residue_id not in seen_residues:
            if residue_id[3] not in AA:
                raise ValueError("nonstandard PDB residue")
            seen_residues.add(residue_id)
            chain_residues.append(residue_id)
    identities = [
        tuple(
            atom[name]
            for name in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for atom in atoms
    ]
    if not atoms or len(identities) != len(set(identities)):
        raise ValueError("empty or duplicate PDB topology")
    return atoms, tuple(chain_residues), "".join(AA[item[3]] for item in chain_residues)


def heavy_hashes(atoms: list[dict[str, str]]) -> tuple[str, str, list[dict[str, str]]]:
    heavy = [atom for atom in atoms if atom["element"] not in {"H", "D", "T"}]
    identity_names = (
        "atom_name",
        "chain_id",
        "element",
        "insertion_code",
        "residue_id",
        "residue_name",
    )
    topology = [{name: atom[name] for name in identity_names} for atom in heavy]
    coordinates = [
        {name: atom[name] for name in (*identity_names, "x", "y", "z")}
        for atom in heavy
    ]
    return digest(canonical(topology)), digest(canonical(coordinates)), heavy


def require_runtime(openmm: Any, app: Any) -> None:
    if openmm.version.full_version != "8.6.0.dev-c6173db":
        raise ValueError("OpenMM exact version drifted")
    root = Path(app.__file__).parent
    for relative, expected in RUNTIME_FILE_HASHES.items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError(f"OpenMM runtime file hash drifted: {relative}")


def atom_id(atom: Any) -> tuple[str, str, str, str, str, str]:
    residue = atom.residue
    element = atom.element.symbol.upper() if atom.element else ""
    return (
        residue.chain.id,
        str(residue.id),
        residue.insertionCode,
        residue.name,
        atom.name,
        element,
    )


def mm_heavy(topology: Any, positions: Any, unit: Any) -> list[dict[str, str]]:
    answer: list[dict[str, str]] = []
    for atom, position in zip(topology.atoms(), positions):
        if atom.element is None or atom.element.atomic_number == 1:
            continue
        chain, number, insertion, residue_name, atom_name, element = atom_id(atom)
        xyz = position.value_in_unit(unit.angstrom)
        answer.append(
            {
                "atom_name": atom_name,
                "chain_id": chain,
                "element": element,
                "insertion_code": insertion,
                "residue_id": number,
                "residue_name": residue_name,
                "x": format(Decimal(str(float(xyz[0]))), ".3f"),
                "y": format(Decimal(str(float(xyz[1]))), ".3f"),
                "z": format(Decimal(str(float(xyz[2]))), ".3f"),
            }
        )
    return answer


def dist(a: Any, b: Any, unit: Any) -> float:
    value = (a - b).value_in_unit(unit.angstrom)
    return math.sqrt(sum(float(component) ** 2 for component in value))


def h_distance(parent: str, value: float) -> bool:
    return (1.1 <= value <= 1.5) if parent == "S" else (0.7 <= value <= 1.3)


def independent_physicality(
    modeller: Any, ff: Any, platform: Any, unit: Any, openmm: Any
) -> dict[str, Any]:
    atoms, positions = list(modeller.topology.atoms()), list(modeller.positions)
    if not atoms or len(atoms) != len(positions):
        raise ValueError("invalid topology-position count")
    if not all(
        math.isfinite(float(x))
        for position in positions
        for x in position.value_in_unit(unit.angstrom)
    ):
        raise ValueError("nonfinite position")
    index = {atom: number for number, atom in enumerate(atoms)}
    bonded = {atom: [] for atom in atoms}
    for a, b in modeller.topology.bonds():
        bonded[a].append(b)
        bonded[b].append(a)
        ea = a.element.symbol.upper() if a.element else ""
        eb = b.element.symbol.upper() if b.element else ""
        if (
            ea != "H"
            and eb != "H"
            and not 1.0 <= dist(positions[index[a]], positions[index[b]], unit) <= 2.3
        ):
            raise ValueError("heavy bond geometry")
    for number, left in enumerate(positions):
        if any(dist(left, right, unit) < 0.5 for right in positions[number + 1 :]):
            raise ValueError("overlapping distinct atoms")
    maximum = {"H": 1, "C": 4, "N": 4, "O": 2, "S": 6}
    hydrogens = 0
    for atom, neighbors in bonded.items():
        element = atom.element.symbol.upper() if atom.element else ""
        if element not in maximum or len(neighbors) > maximum[element]:
            raise ValueError("valence audit failed")
        if element != "H":
            continue
        hydrogens += 1
        parent = [
            other
            for other in neighbors
            if other.element and other.element.atomic_number != 1
        ]
        if len(parent) != 1 or len(neighbors) != 1:
            raise ValueError("invalid hydrogen parent")
        parent_atom = parent[0]
        parent_element = parent_atom.element.symbol.upper()
        if not h_distance(
            parent_element,
            dist(positions[index[atom]], positions[index[parent_atom]], unit),
        ):
            raise ValueError("hydrogen distance audit failed")
        other = next((item for item in bonded[parent_atom] if item is not atom), None)
        if other is not None:
            u = positions[index[atom]] - positions[index[parent_atom]]
            v = positions[index[other]] - positions[index[parent_atom]]
            uv, vv = u.value_in_unit(unit.angstrom), v.value_in_unit(unit.angstrom)
            denominator = math.sqrt(
                sum(float(x) ** 2 for x in uv) * sum(float(x) ** 2 for x in vv)
            )
            angle = math.degrees(
                math.acos(
                    max(
                        -1.0,
                        min(
                            1.0, sum(float(x * y) for x, y in zip(uv, vv)) / denominator
                        ),
                    )
                )
            )
            if angle < 55.0 or angle > 180.0:
                raise ValueError("hydrogen-angle audit failed")
    if hydrogens == 0:
        raise ValueError("no added hydrogens")
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        for former, latter in pairwise(residues):
            c_atom = next((atom for atom in former.atoms() if atom.name == "C"), None)
            n_atom = next((atom for atom in latter.atoms() if atom.name == "N"), None)
            if (
                c_atom
                and n_atom
                and not 1.0
                <= dist(positions[index[c_atom]], positions[index[n_atom]], unit)
                <= 1.8
            ):
                raise ValueError("backbone continuity audit failed")
    system = ff.createSystem(modeller.topology)
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
        raise ValueError("energy audit failed")
    return {
        "atom_count": len(atoms),
        "hydrogen_count": hydrogens,
        "potential_energy_kj_per_mol": energy,
    }


def signature(topology: Any) -> list[dict[str, str]]:
    joins = {atom: [] for atom in topology.atoms()}
    for a, b in topology.bonds():
        joins[a].append(b)
        joins[b].append(a)
    answer = []
    for atom, neighbors in joins.items():
        if atom.element is None or atom.element.atomic_number != 1:
            continue
        parent = next(
            (
                item
                for item in neighbors
                if item.element and item.element.atomic_number != 1
            ),
            None,
        )
        if parent is None:
            raise ValueError("signature parent absent")
        chain, number, insertion, residue_name, hydrogen_name, _ = atom_id(atom)
        answer.append(
            {
                "chain_id": chain,
                "hydrogen_name": hydrogen_name,
                "heavy_parent": parent.name,
                "insertion_code": insertion,
                "residue_id": number,
                "residue_name": residue_name,
            }
        )
    return sorted(answer, key=canonical)


def exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.chmod(path, 0o444)


def replay_worker(
    inputs: Path, generated: Path, output: Path, uid: str, branch: str
) -> None:
    require_committed_sources(inputs)
    state = next(
        (
            item
            for item in independent_roster(inputs)
            if item.uid == uid and item.branch == branch
        ),
        None,
    )
    if state is None:
        raise ValueError("requested replay state absent")
    seed = checker_seed(uid, branch)
    if os.environ.get("PYTHONHASHSEED") != pythonhashseed(seed):
        raise ValueError("replay child lacks its derived PYTHONHASHSEED")
    random.seed(seed)
    raw = regular(inputs, state.pdb_path)
    if digest(raw) != state.pdb_hash:
        raise ValueError("support raw PDB drift")
    source, residues, sequence = pdb_records(raw)
    topology_hash, coordinate_hash, heavy = heavy_hashes(source)
    if (
        len(heavy) != state.heavy_count
        or residues != state.residues
        or sequence != state.sequence
        or topology_hash != state.heavy_topology
        or coordinate_hash != state.heavy_coordinates
    ):
        raise ValueError("support parsed identity/coordinate audit failed")
    try:
        import numpy  # type: ignore[import-not-found]
        import openmm  # type: ignore[import-not-found]
        from openmm import Platform, app, unit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("frozen OpenMM runtime unavailable") from error
    numpy.random.seed(seed % (2**32))
    require_runtime(openmm, app)
    pdb = app.PDBFile(str(_path(inputs, state.pdb_path)))
    if len(list(pdb.topology.atoms())) != len(source):
        raise ValueError("PDBFile atom inventory differs from parsed source")
    if mm_heavy(pdb.topology, pdb.positions, unit) != heavy:
        raise ValueError("independent pre-addHydrogens heavy identity audit failed")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    required = {
        tuple(
            atom[name]
            for name in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for atom in heavy
    }
    modeller.delete(
        [atom for atom in modeller.topology.atoms() if atom_id(atom) not in required]
    )
    if mm_heavy(modeller.topology, modeller.positions, unit) != heavy:
        raise ValueError("independent hydrogen removal changed heavy coordinates")
    ff = app.ForceField("amber14/protein.ff14SB.xml")
    reference = Platform.getPlatformByName("Reference")
    variants = modeller.addHydrogens(
        ff, pH=float(state.ph), variants=None, platform=reference
    )
    if mm_heavy(modeller.topology, modeller.positions, unit) != heavy:
        raise ValueError("independent post-addHydrogens heavy audit failed")
    audit = independent_physicality(modeller, ff, reference, unit, openmm)
    destination = output / f"{token(uid)}--{branch}"
    os.mkdir(destination, 0o700)
    pdb_file = destination / "replay.pdb"
    fd = os.open(pdb_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "w", encoding="ascii", closefd=False) as handle:
            app.PDBFile.writeModel(
                modeller.topology, modeller.positions, handle, keepIds=True
            )
            app.PDBFile.writeFooter(modeller.topology, handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.chmod(pdb_file, 0o444)
    emitted, _, _ = pdb_records(regular(destination, "replay.pdb"))
    _, _, emitted_heavy = heavy_hashes(emitted)
    if emitted_heavy != heavy:
        raise ValueError("independent emitted PDB heavy audit failed")
    replay = {
        "branch_id": branch,
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "entity_token": token(uid),
        "entity_uid": uid,
        "emitted_parsed_heavy_records_exact": True,
        "input_support_raw_sha256": state.pdb_hash,
        "parsed_heavy_coordinate_sha256": coordinate_hash,
        "parsed_heavy_records_exact": True,
        "parsed_heavy_topology_sha256": topology_hash,
        "ph": state.ph,
        "physicality": audit,
        "protonated_pdb_sha256": digest(regular(destination, "replay.pdb")),
        "protonation_signature": signature(modeller.topology),
        "returned_variants": list(variants),
        "support_index": 1,
    }
    exclusive(destination / "replay.json", canonical(replay) + b"\n")
    os.chmod(destination, 0o555)


def generated_metadata(
    generated: Path, state: ReplayState, repeat: int
) -> tuple[bytes, dict[str, Any]]:
    directory = f"{token(state.uid)}--{state.branch}--repeat{repeat}"
    raw_pdb = regular(generated, f"{directory}/protonated.pdb")
    metadata = object_json(
        regular(generated, f"{directory}/metadata.json"), "generator metadata"
    )
    exact_keys(
        metadata,
        {
            "branch_id",
            "candidate_id",
            "closed_capabilities",
            "entity_token",
            "entity_uid",
            "emitted_parsed_heavy_records_exact",
            "input_support_raw_sha256",
            "parsed_heavy_coordinate_sha256",
            "parsed_heavy_records_exact",
            "parsed_heavy_topology_sha256",
            "ph",
            "physicality",
            "protonated_pdb_sha256",
            "protonation_signature",
            "repeat",
            "returned_variants",
            "support_index",
        },
        "generator metadata",
    )
    if not (
        metadata["candidate_id"] == CANDIDATE_ID
        and metadata["closed_capabilities"] == CLOSED_CAPABILITIES
        and metadata["entity_uid"] == state.uid
        and metadata["emitted_parsed_heavy_records_exact"] is True
        and metadata["entity_token"] == token(state.uid)
        and metadata["branch_id"] == state.branch
        and metadata["repeat"] == repeat
        and metadata["support_index"] == 1
        and metadata["ph"] == state.ph
        and metadata["input_support_raw_sha256"] == state.pdb_hash
        and metadata["parsed_heavy_topology_sha256"] == state.heavy_topology
        and metadata["parsed_heavy_coordinate_sha256"] == state.heavy_coordinates
        and metadata["protonated_pdb_sha256"] == digest(raw_pdb)
        and metadata["parsed_heavy_records_exact"] is True
    ):
        raise ValueError("generator metadata binding mismatch")
    return raw_pdb, metadata


def check(inputs: Path, generated: Path, output: Path) -> None:
    require_committed_sources(inputs)
    states = independent_roster(inputs)
    inventory = object_json(
        regular(generated, "generator_inventory.json"), "generator inventory"
    )
    exact_keys(
        inventory,
        {
            "candidate_id",
            "closed_capabilities",
            "repeat_count",
            "state_count",
            "support_index",
        },
        "generator inventory",
    )
    if inventory != {
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "repeat_count": 2,
        "state_count": STATE_COUNT,
        "support_index": SUPPORT_INDEX,
    }:
        raise ValueError("generator inventory binding mismatch")
    expected_generated = {
        f"{token(state.uid)}--{state.branch}--repeat{repeat}"
        for state in states
        for repeat in (0, 1)
    } | {"generator_inventory.json"}
    if (
        len(expected_generated) != 399
        or {item.name for item in generated.iterdir()} != expected_generated
    ):
        raise ValueError("generator output inventory or quota failed")
    os.mkdir(output, 0o700)
    for state in states:
        p0, m0 = generated_metadata(generated, state, 0)
        p1, m1 = generated_metadata(generated, state, 1)
        compare0, compare1 = dict(m0), dict(m1)
        compare0.pop("repeat")
        compare1.pop("repeat")
        if p0 != p1 or canonical(compare0) != canonical(compare1):
            raise ValueError("generation repeat equality failed")
        seed = checker_seed(state.uid, state.branch)
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
            "--generated",
            str(generated),
            "--output",
            str(output),
            "--entity-uid",
            state.uid,
            "--branch-id",
            state.branch,
        ]
        subprocess.run(command, check=True, env=environment)
        replay_dir = f"{token(state.uid)}--{state.branch}"
        replay_pdb = regular(output, f"{replay_dir}/replay.pdb")
        replay = object_json(
            regular(output, f"{replay_dir}/replay.json"), "replay metadata"
        )
        if replay_pdb != p0 or canonical(replay) != canonical(compare0):
            raise ValueError("independent checker replay equality failed")
    expected = {f"{token(state.uid)}--{state.branch}" for state in states}
    if {item.name for item in output.iterdir()} != expected:
        raise ValueError("checker output inventory or quota failed")
    exclusive(
        output / "result.json",
        canonical(
            {
                "candidate_id": CANDIDATE_ID,
                "checker_replay_count": STATE_COUNT,
                "closed_capabilities": CLOSED_CAPABILITIES,
                "generation_repeat_count": 2,
                "state_count": STATE_COUNT,
                "support_index": 1,
            }
        )
        + b"\n",
    )
    os.chmod(output, 0o555)


def self_test() -> int:
    synthetic = []
    for number in range(119):
        synthetic.append(
            {
                "bmrb_id": str(number),
                "condition_state": "observed",
                "deposited_ph": "6.8",
                "entity_uid": f"a/{number}",
                "pH_source": "test",
                "recovery_v6_hold_reasons": [],
                "temperature_ionic_diagnostics": {},
            }
        )
    for number in range(16):
        synthetic.append(
            {
                "bmrb_id": f"u{number}",
                "condition_state": "state_missing",
                "deposited_ph": None,
                "entity_uid": f"u/{number}",
                "pH_source": None,
                "recovery_v6_hold_reasons": [],
                "temperature_ionic_diagnostics": {},
            }
        )
    assert len(derive_branches(synthetic)) == 199
    assert token("a/b") != token("a:b")
    assert h_distance("S", 1.4) and not h_distance("N", 1.4)
    for raw in (b'{"x":1,"x":1}', b'{"x":Infinity}', b"null"):
        try:
            object_json(raw, "test")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON accepted")
    own_source = Path(__file__).read_text(encoding="utf-8")
    forbidden = "cohort_support1_protonation_preflight" + " import"
    if forbidden in own_source:
        raise AssertionError("checker imports generator")
    return 6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--entity-uid")
    parser.add_argument("--branch-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(f"STATUS PASS_STATIC_INDEPENDENT_CHECKER_SELF_TEST checks={self_test()}")
        return 0
    if args.inputs is None or args.generated is None or args.output is None:
        parser.error("--inputs, --generated, and --output are required")
    if args.worker:
        if args.entity_uid is None or args.branch_id is None:
            parser.error("worker requires entity and branch")
        replay_worker(
            args.inputs, args.generated, args.output, args.entity_uid, args.branch_id
        )
    else:
        check(args.inputs, args.generated, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
