"""Inert shared parsing and OpenMM helpers for the ff15ipq bounded smoke."""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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
RUNTIME_FILE_HASHES = {
    "data/amber14/protein.ff15ipq.xml": (
        "0085c9dc2818a28501f6074a0bb46b994a9787bf4010ff86417fe014b11ab622"
    ),
    "data/hydrogens.xml": (
        "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d"
    ),
    "modeller.py": "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
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
                raise ValueError("nonstandard residue in source PDB")
            residue_seen.add(residue)
            residues.append(residue)
    if not records:
        raise ValueError("source PDB has no ATOM records")
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
