"""Secondary chemical-shift reference helpers.

The v1 table is intentionally small and deterministic.  It is used as a
physics-aware centering layer, not as a claim of high-precision random-coil
prediction.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


ATOM_FAMILY_BY_ATOM = {
    "H": "HN",
    "HN": "HN",
    "N": "N",
    "CA": "CA",
    "CB": "CB",
    "C": "C'",
}

FAMILY_DEFAULTS = {
    "HN": 8.24,
    "N": 119.8,
    "CA": 55.9,
    "CB": 38.5,
    "C'": 175.7,
}

# Approximate residue-specific random-coil centering values in ppm.  The
# strongest use is removing residue-type and atom-family offsets before CCC
# optimization; missing cells fall back to FAMILY_DEFAULTS.
RANDOM_COIL_REFERENCE = {
    "ALA": {"HN": 8.24, "N": 123.0, "CA": 52.5, "CB": 19.0, "C'": 177.8},
    "ARG": {"HN": 8.25, "N": 120.5, "CA": 56.1, "CB": 30.6, "C'": 176.3},
    "ASN": {"HN": 8.35, "N": 118.5, "CA": 53.0, "CB": 38.7, "C'": 175.4},
    "ASP": {"HN": 8.31, "N": 120.5, "CA": 54.1, "CB": 40.8, "C'": 176.2},
    "CYS": {"HN": 8.28, "N": 119.5, "CA": 58.0, "CB": 28.0, "C'": 175.0},
    "GLN": {"HN": 8.27, "N": 120.0, "CA": 55.8, "CB": 29.2, "C'": 176.2},
    "GLU": {"HN": 8.29, "N": 120.2, "CA": 56.4, "CB": 29.8, "C'": 176.4},
    "GLY": {"HN": 8.32, "N": 109.5, "CA": 45.2, "CB": 38.5, "C'": 174.5},
    "HIS": {"HN": 8.30, "N": 119.0, "CA": 56.0, "CB": 29.8, "C'": 175.1},
    "ILE": {"HN": 8.19, "N": 121.5, "CA": 61.3, "CB": 38.7, "C'": 176.1},
    "LEU": {"HN": 8.24, "N": 121.0, "CA": 54.9, "CB": 42.2, "C'": 176.5},
    "LYS": {"HN": 8.25, "N": 120.5, "CA": 56.2, "CB": 32.9, "C'": 176.4},
    "MET": {"HN": 8.24, "N": 120.0, "CA": 55.4, "CB": 32.8, "C'": 176.0},
    "PHE": {"HN": 8.31, "N": 120.0, "CA": 58.0, "CB": 39.5, "C'": 175.5},
    "PRO": {"HN": 8.24, "N": 135.0, "CA": 63.1, "CB": 31.8, "C'": 176.8},
    "SER": {"HN": 8.28, "N": 116.0, "CA": 58.2, "CB": 63.7, "C'": 174.6},
    "THR": {"HN": 8.24, "N": 114.0, "CA": 61.7, "CB": 69.8, "C'": 174.5},
    "TRP": {"HN": 8.25, "N": 121.0, "CA": 57.5, "CB": 29.6, "C'": 176.0},
    "TYR": {"HN": 8.28, "N": 120.0, "CA": 57.8, "CB": 38.8, "C'": 175.5},
    "VAL": {"HN": 8.20, "N": 121.5, "CA": 62.3, "CB": 32.7, "C'": 176.0},
}


def parse_chemical_shift_target_id(target_id: str) -> dict[str, str | int | None]:
    """Parse ``cs:{chain}:{seq_id}:{comp_id}:{atom_id}`` target IDs."""
    parts = str(target_id).split(":")
    if len(parts) != 5 or parts[0] != "cs":
        return {
            "chain_id": "_",
            "residue_index": 0,
            "residue_name": "",
            "atom_name": "",
            "atom_family": None,
        }
    try:
        residue_index = int(parts[2])
    except ValueError:
        residue_index = 0
    atom_name = parts[4].upper()
    return {
        "chain_id": parts[1],
        "residue_index": max(residue_index, 0),
        "residue_name": parts[3].upper(),
        "atom_name": atom_name,
        "atom_family": ATOM_FAMILY_BY_ATOM.get(atom_name),
    }


def random_coil_reference(residue_name: str, atom_family: str | None) -> float:
    """Return the deterministic v1 random-coil reference in ppm."""
    if atom_family is None:
        return math.nan
    family = str(atom_family)
    residue = str(residue_name or "").upper()
    return float(
        RANDOM_COIL_REFERENCE.get(residue, {}).get(
            family,
            FAMILY_DEFAULTS.get(family, math.nan),
        )
    )


def random_coil_reference_for_target_id(target_id: str) -> float:
    """Return the random-coil reference for one chemical-shift target ID."""
    parsed = parse_chemical_shift_target_id(target_id)
    return random_coil_reference(
        residue_name=str(parsed["residue_name"]),
        atom_family=parsed["atom_family"] if parsed["atom_family"] else None,
    )


def random_coil_baselines_for_target_ids(target_ids: Iterable[str]) -> np.ndarray:
    """Return a float array of random-coil references for target IDs."""
    return np.asarray(
        [random_coil_reference_for_target_id(target_id) for target_id in target_ids],
        dtype=np.float64,
    )


def secondary_shift_vector(values: np.ndarray, target_ids: list[str]) -> np.ndarray:
    """Return values centered by residue/atom random-coil references."""
    baselines = random_coil_baselines_for_target_ids(target_ids)
    return np.asarray(values, dtype=np.float64) - baselines
