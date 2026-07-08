"""PDB normalization/completion helpers for UCBShift-facing BioEmu exports.

BioEmu detached samples can be backbone-heavy-atom structures.  UCBShift expects
conventional residue numbering, backbone protons, and enough standard residue
atoms for its SPARTA+/SHIFTX feature builders to keep residue rows.  This module
does not claim to reconstruct physically exact sidechain conformations; it
creates a deterministic, provenance-marked completion scaffold so generated
support can be evaluated by the offline UCBShift judge instead of failing before
posterior inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPLETION_MODE = "ucbshift_idealized_sidechain_completion_v1"
HYDROGEN_ONLY_COMPLETION_MODE = "ucbshift_backbone_hydrogen_completion_v1"

_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

_SIDECHAIN_OFFSETS: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "ARG": [
        ("CG", (1.52, 0.00, 0.00)),
        ("CD", (2.95, 0.35, 0.10)),
        ("NE", (4.30, 0.05, 0.25)),
        ("CZ", (5.58, 0.20, 0.10)),
        ("NH1", (6.35, 1.15, 0.10)),
        ("NH2", (6.45, -0.70, -0.10)),
    ],
    "ASN": [("CG", (1.50, 0.00, 0.00)), ("OD1", (2.15, 0.95, 0.00)), ("ND2", (2.35, -0.85, 0.00))],
    "ASP": [("CG", (1.50, 0.00, 0.00)), ("OD1", (2.15, 0.95, 0.00)), ("OD2", (2.35, -0.85, 0.00))],
    "CYS": [("SG", (1.82, 0.10, 0.05))],
    "GLN": [
        ("CG", (1.52, 0.00, 0.00)),
        ("CD", (2.95, 0.25, 0.10)),
        ("OE1", (3.55, 1.20, 0.10)),
        ("NE2", (3.75, -0.75, -0.10)),
    ],
    "GLU": [
        ("CG", (1.52, 0.00, 0.00)),
        ("CD", (2.95, 0.25, 0.10)),
        ("OE1", (3.55, 1.20, 0.10)),
        ("OE2", (3.75, -0.75, -0.10)),
    ],
    "HIS": [
        ("CG", (1.48, 0.00, 0.00)),
        ("ND1", (2.15, 1.05, 0.00)),
        ("CD2", (2.25, -1.05, 0.00)),
        ("CE1", (3.35, 0.65, 0.00)),
        ("NE2", (3.40, -0.65, 0.00)),
    ],
    "ILE": [("CG1", (1.30, 0.92, 0.35)), ("CG2", (1.30, -0.92, -0.35)), ("CD1", (2.72, 1.22, 0.52))],
    "LEU": [("CG", (1.52, 0.00, 0.00)), ("CD1", (2.35, 0.95, 0.30)), ("CD2", (2.35, -0.95, -0.30))],
    "LYS": [
        ("CG", (1.52, 0.00, 0.00)),
        ("CD", (2.95, 0.35, 0.10)),
        ("CE", (4.35, 0.05, 0.25)),
        ("NZ", (5.72, 0.22, 0.05)),
    ],
    "MET": [("CG", (1.52, 0.00, 0.00)), ("SD", (2.95, 0.35, 0.10)), ("CE", (4.45, 0.05, 0.25))],
    "PHE": [
        ("CG", (1.50, 0.00, 0.00)),
        ("CD1", (2.20, 1.20, 0.00)),
        ("CD2", (2.20, -1.20, 0.00)),
        ("CE1", (3.55, 1.20, 0.00)),
        ("CE2", (3.55, -1.20, 0.00)),
        ("CZ", (4.25, 0.00, 0.00)),
    ],
    "PRO": [("CG", (1.35, 0.75, 0.20)), ("CD", (0.95, 1.95, 0.45))],
    "SER": [("OG", (1.42, 0.20, 0.20))],
    "THR": [("OG1", (1.32, 0.90, 0.20)), ("CG2", (1.32, -0.90, -0.20))],
    "TRP": [
        ("CG", (1.50, 0.00, 0.00)),
        ("CD1", (2.20, 1.05, 0.00)),
        ("CD2", (2.35, -0.82, 0.00)),
        ("NE1", (3.48, 0.78, 0.00)),
        ("CE2", (3.58, -0.45, 0.00)),
        ("CE3", (3.02, -2.05, 0.00)),
        ("CZ2", (4.78, -1.10, 0.00)),
        ("CZ3", (4.18, -2.65, 0.00)),
        ("CH2", (5.22, -2.08, 0.00)),
    ],
    "TYR": [
        ("CG", (1.50, 0.00, 0.00)),
        ("CD1", (2.20, 1.20, 0.00)),
        ("CD2", (2.20, -1.20, 0.00)),
        ("CE1", (3.55, 1.20, 0.00)),
        ("CE2", (3.55, -1.20, 0.00)),
        ("CZ", (4.25, 0.00, 0.00)),
        ("OH", (5.58, 0.00, 0.00)),
    ],
    "VAL": [("CG1", (1.30, 0.92, 0.35)), ("CG2", (1.30, -0.92, -0.35))],
}


@dataclass(frozen=True)
class PDBCompletionResult:
    """Summary for one UCBShift-ready PDB completion."""

    input_path: Path
    output_path: Path
    mode: str
    raw_atom_count: int
    completed_atom_count: int
    added_atom_count: int
    added_heavy_atom_count: int
    added_hydrogen_count: int
    residue_count: int
    residue_number_offset: int
    sequence_resname_overrides: int
    backbone_rebuilt: bool
    backbone_bond_ready_fraction_before: float
    backbone_bond_ready_fraction_after: float


def complete_pdb_for_ucbshift(
    input_path: Path,
    output_path: Path,
    *,
    sequence: str = "",
    mode: str = COMPLETION_MODE,
    complete_heavy_atoms: bool = True,
) -> PDBCompletionResult:
    """Write a UCBShift-ready, provenance-marked completion of ``input_path``."""

    atom_rows = [
        parsed
        for line in input_path.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        for parsed in [_parse_pdb_atom_line(line)]
        if parsed is not None
    ]
    if not atom_rows:
        raise ValueError(f"No parseable ATOM/HETATM records in PDB: {input_path}")

    raw_atom_count = len(atom_rows)
    residue_numbers = [int(row["resseq"]) for row in atom_rows]
    offset = 1 - min(residue_numbers) if min(residue_numbers) <= 0 else 0
    sequence_resnames = [_ONE_TO_THREE.get(letter.upper(), "") for letter in sequence.strip()]
    sequence_override_count = 0
    residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str]] = []
    for row in atom_rows:
        row = dict(row)
        row["chain_id"] = str(row.get("chain_id") or "A")[:1] or "A"
        row["resseq"] = int(row["resseq"]) + offset
        key = (str(row["chain_id"]), int(row["resseq"]), str(row.get("icode") or " "))
        if key not in residues:
            residue_index = len(order)
            resname = str(row["resname"]).upper()
            if residue_index < len(sequence_resnames) and sequence_resnames[residue_index]:
                if resname != sequence_resnames[residue_index]:
                    sequence_override_count += 1
                resname = sequence_resnames[residue_index]
            residues[key] = {"resname": resname, "atoms": {}, "atom_order": []}
            order.append(key)
        atom_name = str(row["atom_name"]).strip()
        if atom_name not in residues[key]["atoms"]:
            residues[key]["atom_order"].append(atom_name)
        residues[key]["atoms"][atom_name] = row

    bond_fraction_before = _backbone_bond_ready_fraction(residues, order)
    backbone_rebuilt = bool(len(order) >= 3 and bond_fraction_before < 0.80)
    if backbone_rebuilt:
        _regularize_backbone_from_ca_trace(residues, order)
    bond_fraction_after = _backbone_bond_ready_fraction(residues, order)

    added_hydrogen = 0
    for index, key in enumerate(order):
        residue = residues[key]
        previous_atoms = residues[order[index - 1]]["atoms"] if index > 0 else {}
        _, hydrogen_count = _complete_residue(
            residue,
            previous_atoms,
            complete_heavy_atoms=complete_heavy_atoms,
        )
        added_hydrogen += hydrogen_count

    output_lines: list[str] = []
    serial = 1
    for chain_id, resseq, icode in order:
        residue = residues[(chain_id, resseq, icode)]
        resname = str(residue["resname"])
        atom_names = _ucbshift_atom_order(
            resname,
            list(residue["atom_order"]),
            residue["atoms"],
        )
        for atom_name in atom_names:
            atom = residue["atoms"][atom_name]
            output_lines.append(
                _format_pdb_atom_line(
                    serial=serial,
                    atom_name=atom_name,
                    resname=resname,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    xyz=np.asarray(atom["xyz"], dtype=float),
                    occupancy=float(atom.get("occupancy", 1.0)),
                    bfactor=float(atom.get("bfactor", 0.0)),
                    element=str(atom.get("element") or _infer_element(atom_name)),
                )
            )
            serial += 1
    output_lines.append("TER\n")
    output_lines.append("END\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(output_lines))
    completed_atom_count = serial - 1
    added_atom_count = max(0, completed_atom_count - raw_atom_count)
    added_heavy_count = max(0, added_atom_count - added_hydrogen)
    return PDBCompletionResult(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        raw_atom_count=raw_atom_count,
        completed_atom_count=completed_atom_count,
        added_atom_count=added_atom_count,
        added_heavy_atom_count=added_heavy_count,
        added_hydrogen_count=added_hydrogen,
        residue_count=len(order),
        residue_number_offset=offset,
        sequence_resname_overrides=sequence_override_count,
        backbone_rebuilt=backbone_rebuilt,
        backbone_bond_ready_fraction_before=bond_fraction_before,
        backbone_bond_ready_fraction_after=bond_fraction_after,
    )


def _complete_residue(
    residue: dict[str, Any],
    previous_atoms: dict[str, dict[str, Any]],
    *,
    complete_heavy_atoms: bool = True,
) -> tuple[int, int]:
    atoms: dict[str, dict[str, Any]] = residue["atoms"]
    resname = str(residue["resname"]).upper()
    heavy_added = 0
    hydrogen_added = 0

    if (
        complete_heavy_atoms
        and resname != "GLY"
        and "CB" not in atoms
        and "CA" in atoms
    ):
        basis = _local_basis(atoms)
        atoms["CB"] = _new_atom("CB", _atom_xyz(atoms["CA"]) + basis[0] * 1.53)
        residue["atom_order"].append("CB")
        heavy_added += 1

    basis = _local_basis(atoms)
    if complete_heavy_atoms and resname in _SIDECHAIN_OFFSETS and "CB" in atoms:
        cb_xyz = _atom_xyz(atoms["CB"])
        for atom_name, offset in _SIDECHAIN_OFFSETS[resname]:
            if atom_name in atoms:
                continue
            atoms[atom_name] = _new_atom(
                atom_name,
                cb_xyz
                + basis[0] * offset[0]
                + basis[1] * offset[1]
                + basis[2] * offset[2],
            )
            residue["atom_order"].append(atom_name)
            heavy_added += 1

    h_count = _add_backbone_hydrogens(residue, previous_atoms, basis)
    hydrogen_added += h_count
    return heavy_added, hydrogen_added


def _add_backbone_hydrogens(
    residue: dict[str, Any],
    previous_atoms: dict[str, dict[str, Any]],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int:
    atoms: dict[str, dict[str, Any]] = residue["atoms"]
    resname = str(residue["resname"]).upper()
    added = 0
    if resname != "PRO" and "H" not in atoms and "N" in atoms:
        n_xyz = _atom_xyz(atoms["N"])
        if "C" in previous_atoms:
            direction = n_xyz - _atom_xyz(previous_atoms["C"])
        elif "CA" in atoms:
            direction = n_xyz - _atom_xyz(atoms["CA"])
        else:
            direction = -basis[0]
        atoms["H"] = _new_atom("H", n_xyz + _unit_vector(direction) * 1.0, element="H")
        residue["atom_order"].append("H")
        added += 1

    if "CA" not in atoms:
        return added
    ca_xyz = _atom_xyz(atoms["CA"])
    cb_term = ca_xyz - _atom_xyz(atoms["CB"]) if "CB" in atoms else basis[1]
    direction = np.zeros(3, dtype=float)
    if "N" in atoms:
        direction += ca_xyz - _atom_xyz(atoms["N"])
    if "C" in atoms:
        direction += ca_xyz - _atom_xyz(atoms["C"])
    direction += cb_term
    ha_direction = _unit_vector(direction if np.linalg.norm(direction) > 1.0e-6 else basis[1])
    if resname == "GLY":
        if "HA2" not in atoms:
            atoms["HA2"] = _new_atom("HA2", ca_xyz + _unit_vector(ha_direction + basis[2]) * 1.09, element="H")
            residue["atom_order"].append("HA2")
            added += 1
        if "HA3" not in atoms:
            atoms["HA3"] = _new_atom("HA3", ca_xyz + _unit_vector(ha_direction - basis[2]) * 1.09, element="H")
            residue["atom_order"].append("HA3")
            added += 1
        if "HA" not in atoms:
            atoms["HA"] = _new_atom("HA", ca_xyz + ha_direction * 1.09, element="H")
            residue["atom_order"].append("HA")
            added += 1
        return added

    if "HA" not in atoms:
        atoms["HA"] = _new_atom("HA", ca_xyz + ha_direction * 1.09, element="H")
        residue["atom_order"].append("HA")
        added += 1
    return added


def _local_basis(
    atoms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ca = _atom_xyz(atoms["CA"]) if "CA" in atoms else np.zeros(3, dtype=float)
    if "CB" in atoms and np.linalg.norm(_atom_xyz(atoms["CB"]) - ca) > 1.0e-6:
        e1 = _unit_vector(_atom_xyz(atoms["CB"]) - ca)
    else:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        if "N" in atoms:
            direction += ca - _atom_xyz(atoms["N"])
        if "C" in atoms:
            direction += ca - _atom_xyz(atoms["C"])
        e1 = _unit_vector(direction)

    if "N" in atoms and "C" in atoms:
        axis = _unit_vector(_atom_xyz(atoms["C"]) - _atom_xyz(atoms["N"]))
    elif "C" in atoms:
        axis = _unit_vector(_atom_xyz(atoms["C"]) - ca)
    elif "N" in atoms:
        axis = _unit_vector(ca - _atom_xyz(atoms["N"]))
    else:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    e2 = np.cross(axis, e1)
    if np.linalg.norm(e2) < 1.0e-6:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, e1))) > 0.95:
            trial = np.array([0.0, 0.0, 1.0], dtype=float)
        e2 = np.cross(trial, e1)
    e2 = _unit_vector(e2)
    e3 = _unit_vector(np.cross(e1, e2))
    return e1, e2, e3


def _backbone_bond_ready_fraction(
    residues: dict[tuple[str, int, str], dict[str, Any]],
    order: list[tuple[str, int, str]],
) -> float:
    distances: list[float] = []
    for left_key, right_key in zip(order, order[1:]):
        left = residues[left_key]["atoms"]
        right = residues[right_key]["atoms"]
        if "C" not in left or "N" not in right:
            continue
        distances.append(float(np.linalg.norm(_atom_xyz(left["C"]) - _atom_xyz(right["N"]))))
    if not distances:
        return 0.0 if len(order) >= 2 else 1.0
    ready = sum(1 for distance in distances if 0.75 <= distance <= 2.10)
    return float(ready / len(distances))


def _regularize_backbone_from_ca_trace(
    residues: dict[tuple[str, int, str], dict[str, Any]],
    order: list[tuple[str, int, str]],
) -> None:
    ca_points: list[np.ndarray] = []
    for index, key in enumerate(order):
        atoms = residues[key]["atoms"]
        if "CA" in atoms:
            ca_points.append(_atom_xyz(atoms["CA"]))
            continue
        ca_points.append(np.array([float(index) * 3.8, 0.0, 0.0], dtype=float))
    if not ca_points:
        return

    regularized_ca = [np.asarray(ca_points[0], dtype=float)]
    previous_direction = np.array([1.0, 0.0, 0.0], dtype=float)
    for index in range(1, len(ca_points)):
        raw_direction = np.asarray(ca_points[index], dtype=float) - np.asarray(
            ca_points[index - 1],
            dtype=float,
        )
        direction = _unit_vector(raw_direction)
        if np.linalg.norm(raw_direction) < 1.0e-6:
            direction = previous_direction
        regularized_ca.append(regularized_ca[-1] + direction * 3.80)
        previous_direction = direction

    for index, key in enumerate(order):
        residue = residues[key]
        atoms: dict[str, dict[str, Any]] = residue["atoms"]
        ca = regularized_ca[index]
        prev_direction = (
            _unit_vector(regularized_ca[index] - regularized_ca[index - 1])
            if index > 0
            else None
        )
        next_direction = (
            _unit_vector(regularized_ca[index + 1] - regularized_ca[index])
            if index + 1 < len(regularized_ca)
            else None
        )
        if index == 0 and len(regularized_ca) > 1:
            tangent = next_direction if next_direction is not None else np.array([1.0, 0.0, 0.0], dtype=float)
        elif index + 1 == len(regularized_ca) and index > 0:
            tangent = prev_direction if prev_direction is not None else np.array([1.0, 0.0, 0.0], dtype=float)
        elif prev_direction is not None and next_direction is not None:
            tangent = _unit_vector(prev_direction + next_direction)
        else:
            tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = _perpendicular_unit(tangent)
        old_ca = _atom_xyz(atoms["CA"]) if "CA" in atoms else ca
        old_cb_vector = (
            _atom_xyz(atoms["CB"]) - old_ca
            if "CB" in atoms
            else normal * 1.53
        )
        cb_direction = _unit_vector(old_cb_vector)
        if abs(float(np.dot(cb_direction, tangent))) > 0.92:
            cb_direction = normal

        n_direction = prev_direction if prev_direction is not None else tangent
        c_direction = next_direction if next_direction is not None else tangent
        _set_or_add_atom(residue, "N", ca - n_direction * 1.20)
        _set_or_add_atom(residue, "CA", ca)
        _set_or_add_atom(residue, "C", ca + c_direction * 1.20)
        _set_or_add_atom(residue, "O", ca + c_direction * 1.68 + normal * 1.05, element="O")
        if str(residue["resname"]).upper() != "GLY":
            _set_or_add_atom(residue, "CB", ca + cb_direction * 1.53)


def _perpendicular_unit(vector: np.ndarray) -> np.ndarray:
    unit = _unit_vector(vector)
    trial = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(unit, trial))) > 0.90:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    return _unit_vector(np.cross(unit, trial))


def _set_or_add_atom(
    residue: dict[str, Any],
    atom_name: str,
    xyz: np.ndarray,
    *,
    element: str | None = None,
) -> None:
    atoms: dict[str, dict[str, Any]] = residue["atoms"]
    if atom_name in atoms:
        atoms[atom_name]["xyz"] = np.asarray(xyz, dtype=float)
        atoms[atom_name]["element"] = element or atoms[atom_name].get("element") or _infer_element(atom_name)
        return
    atoms[atom_name] = _new_atom(atom_name, xyz, element=element)
    residue["atom_order"].append(atom_name)


def _parse_pdb_atom_line(line: str) -> dict[str, Any] | None:
    try:
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain_id = line[21:22].strip() or "A"
        resseq = int(line[22:26])
        icode = line[26:27] or " "
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
        occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
        bfactor = float(line[60:66]) if line[60:66].strip() else 0.0
        element = line[76:78].strip() or _infer_element(atom_name)
    except Exception:
        return None
    if not atom_name or not resname:
        return None
    return {
        "atom_name": atom_name,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "xyz": xyz,
        "occupancy": occupancy,
        "bfactor": bfactor,
        "element": element,
    }


def _format_pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    xyz: np.ndarray,
    occupancy: float,
    bfactor: float,
    element: str,
) -> str:
    x, y, z = [float(value) for value in xyz]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id[:1] or 'A'}"
        f"{int(resseq):4d}{(icode or ' ')[:1]}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{float(occupancy):6.2f}{float(bfactor):6.2f}"
        f"          {element.strip()[:2].upper():>2s}\n"
    )


def _new_atom(atom_name: str, xyz: np.ndarray, *, element: str | None = None) -> dict[str, Any]:
    return {
        "atom_name": atom_name,
        "xyz": np.asarray(xyz, dtype=float),
        "occupancy": 1.0,
        "bfactor": 0.0,
        "element": element or _infer_element(atom_name),
    }


def _atom_xyz(atom: dict[str, Any]) -> np.ndarray:
    return np.asarray(atom["xyz"], dtype=float)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1.0e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _infer_element(atom_name: str) -> str:
    stripped = "".join(ch for ch in str(atom_name).strip() if ch.isalpha())
    if not stripped:
        return "C"
    if len(stripped) >= 2 and stripped[:2].upper() in {"FE", "ZN", "MG", "MN", "CA"}:
        return stripped[:2].upper()
    return stripped[0].upper()


def _ucbshift_atom_order(
    resname: str,
    atom_order: list[str],
    atoms: dict[str, dict[str, Any]],
) -> list[str]:
    preferred = ["N", "H", "CA", "HA", "HA2", "HA3", "C", "O", "OXT", "CB"]
    sidechain = [atom for atom, _ in _SIDECHAIN_OFFSETS.get(str(resname).upper(), [])]
    ordered: list[str] = []
    for atom_name in preferred + sidechain + atom_order + sorted(atoms):
        if atom_name in atoms and atom_name not in ordered:
            ordered.append(atom_name)
    return ordered
