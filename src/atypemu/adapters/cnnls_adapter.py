"""Helpers for reading precomputed chemical-shift files from existing pipelines."""

from __future__ import annotations

import csv
from pathlib import Path

from atypemu.types import canonical_atom_name


_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "?": "UNK",
}


def normalize_candidate_key(path: str | Path) -> str:
    """Return a stable candidate key derived from a path.

    Args:
        path: Input file path.

    Returns:
        Normalized candidate identifier.
    """
    name = Path(path).name
    for suffix in [".pdb.cs", ".tab", ".csv", ".json", ".pdb"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def build_prediction_map(directory: str | Path) -> dict[str, str]:
    """Map normalized candidate IDs to prediction sidecar paths."""
    root = Path(directory)
    mapping: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        mapping[normalize_candidate_key(path)] = str(path)
    return mapping


def load_candidate_shift_file(
    path: str | Path, file_format: str | None
) -> dict[tuple[int, str], float]:
    """Load one candidate's predicted shifts.

    Args:
        path: Input prediction file.
        file_format: Optional explicit format name.

    Returns:
        Mapping from ``(seq_id, atom_id)`` to predicted shift value.
    """
    resolved_format = (file_format or Path(path).suffix.lstrip(".")).lower()
    if resolved_format in {"json"}:
        return _load_json(path)
    if resolved_format in {"shiftx2", "pdb.cs", "cs"}:
        return _load_shiftx2(path)
    if resolved_format in {"sparta+", "sparta"}:
        return _load_sparta(path)
    if resolved_format in {"ucbshift", "csv"}:
        try:
            return _load_ucbshift(path)
        except Exception:
            return _load_shiftx2(path)
    raise ValueError(f"Unsupported chemical-shift format: {resolved_format}")


def _load_json(path: str | Path) -> dict[tuple[int, str], float]:
    """Load chemical shifts from a simple JSON sidecar."""
    import json

    payload = json.loads(Path(path).read_text())
    mapping: dict[tuple[int, str], float] = {}
    if isinstance(payload, list):
        for item in payload:
            mapping[(int(item["seq_id"]), canonical_atom_name(item["atom_id"]))] = (
                float(item["value"])
            )
        return mapping
    for seq_key, atom_map in payload.items():
        for atom_id, value in atom_map.items():
            mapping[(int(seq_key), canonical_atom_name(atom_id))] = float(value)
    return mapping


def _load_shiftx2(path: str | Path) -> dict[tuple[int, str], float]:
    """Load per-candidate SHIFTX2-style predictions."""
    mapping: dict[tuple[int, str], float] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = int(row["NUM"])
            atom_id = canonical_atom_name(row["ATOMNAME"])
            value = float(row["SHIFT"])
            mapping[(seq_id, atom_id)] = value
    return mapping


def _load_sparta(path: str | Path) -> dict[tuple[int, str], float]:
    """Load per-candidate SPARTA+ predictions."""
    mapping: dict[tuple[int, str], float] = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("REMARK", "DATA", "VARS", "FORMAT")):
            continue
        tokens = stripped.split()
        if len(tokens) < 5:
            continue
        seq_id = int(tokens[0])
        atom_id = canonical_atom_name(tokens[2])
        value = float(tokens[4])
        mapping[(seq_id, atom_id)] = value
    return mapping


def _load_ucbshift(path: str | Path) -> dict[tuple[int, str], float]:
    """Load per-candidate UCBSHIFT predictions."""
    mapping: dict[tuple[int, str], float] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = int(row["RESNUM"])
            for atom_id in ["H", "CA", "CB", "C", "N"]:
                for key in [
                    f"{atom_id}_UCBShift",
                    f"{atom_id}_X",
                    f"{atom_id}_Y",
                ]:
                    if key not in row or row[key] in {"", ".", "?"}:
                        continue
                    mapping[(seq_id, canonical_atom_name(atom_id))] = float(row[key])
                    break
    return mapping
