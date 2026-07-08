"""pRCI helpers for chemical-shift-derived flexibility profiles.

The original Random Coil Index (RCI) uses optimized coefficients and a full
random-coil correction workflow. This module implements a conservative
posterior RCI-like adapter (pRCI) for AtypEmu monitor diagnostics when an exact
external RCI profile is not supplied.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


ATOM_RANDOM_COIL_SCALES = {
    "HN": 0.55,
    "N": 3.0,
    "CA": 2.2,
    "CB": 2.4,
    "C'": 1.7,
}

GENERIC_RANDOM_COIL_SHIFTS = {
    "HN": 8.25,
    "N": 120.0,
    "CA": 55.0,
    "CB": 33.0,
    "C'": 176.0,
}

# Approximate backbone random-coil reference shifts. These are intentionally
# labeled pRCI, not exact RCI, because they do not include the full neighbor,
# pH, temperature, and optimized-combination corrections from the RCI server.
RANDOM_COIL_SHIFTS = {
    "ALA": {"HN": 8.24, "N": 123.8, "CA": 52.5, "CB": 19.0, "C'": 177.8},
    "ARG": {"HN": 8.24, "N": 120.5, "CA": 56.1, "CB": 30.6, "C'": 176.3},
    "ASN": {"HN": 8.35, "N": 119.4, "CA": 53.0, "CB": 38.6, "C'": 175.0},
    "ASP": {"HN": 8.33, "N": 120.6, "CA": 54.0, "CB": 40.8, "C'": 176.2},
    "CYS": {"HN": 8.30, "N": 119.5, "CA": 58.3, "CB": 28.2, "C'": 174.6},
    "GLN": {"HN": 8.26, "N": 119.8, "CA": 55.7, "CB": 29.2, "C'": 176.1},
    "GLU": {"HN": 8.28, "N": 120.2, "CA": 56.4, "CB": 29.9, "C'": 176.8},
    "GLY": {"HN": 8.33, "N": 108.8, "CA": 45.1, "C'": 174.9},
    "HIS": {"HN": 8.34, "N": 119.7, "CA": 55.2, "CB": 29.8, "C'": 175.1},
    "ILE": {"HN": 8.17, "N": 121.3, "CA": 61.3, "CB": 38.8, "C'": 176.1},
    "LEU": {"HN": 8.25, "N": 121.8, "CA": 54.9, "CB": 42.2, "C'": 176.8},
    "LYS": {"HN": 8.24, "N": 121.2, "CA": 56.2, "CB": 32.9, "C'": 176.4},
    "MET": {"HN": 8.27, "N": 120.5, "CA": 55.3, "CB": 32.6, "C'": 176.2},
    "PHE": {"HN": 8.32, "N": 120.3, "CA": 58.0, "CB": 39.5, "C'": 175.4},
    "PRO": {"CA": 63.1, "CB": 31.7, "C'": 176.6},
    "SER": {"HN": 8.31, "N": 115.7, "CA": 58.2, "CB": 63.8, "C'": 174.6},
    "THR": {"HN": 8.25, "N": 113.6, "CA": 61.7, "CB": 69.7, "C'": 174.7},
    "TRP": {"HN": 8.21, "N": 121.5, "CA": 57.4, "CB": 29.5, "C'": 175.8},
    "TYR": {"HN": 8.21, "N": 120.2, "CA": 57.9, "CB": 38.8, "C'": 175.6},
    "VAL": {"HN": 8.19, "N": 120.0, "CA": 62.3, "CB": 32.7, "C'": 175.8},
}


def compute_prci_profile_from_posterior(
    posterior_frame: pd.DataFrame,
    *,
    smooth: bool = True,
) -> pd.DataFrame:
    """Compute a pRCI flexibility profile from chemical-shift posterior rows.

    Args:
        posterior_frame: Rows with at least ``entity_uid``, ``target_id``,
            ``atom_family``, and ``target_value``. ``posterior_std`` is optional.
        smooth: Whether to apply a three-residue centered smoothing pass.

    Returns:
        Long-form RCI adapter profile with ``rci_flexibility`` in ``[0, 1]``.
    """

    required = {"entity_uid", "target_id", "atom_family", "target_value"}
    if posterior_frame.empty or not required.issubset(posterior_frame.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in posterior_frame.iterrows():
        metadata = _target_metadata(str(row["target_id"]))
        if metadata is None:
            continue
        chain_id, residue_index, residue_name, atom_family = metadata
        if str(row.get("atom_family")) != atom_family:
            continue
        target_value = _finite_float(row.get("target_value"))
        if target_value is None:
            continue
        reference = _random_coil_shift(residue_name, atom_family)
        scale = ATOM_RANDOM_COIL_SCALES.get(atom_family)
        if reference is None or scale is None:
            continue
        secondary_shift = target_value - reference
        normalized_shift = abs(secondary_shift) / max(scale, 1e-6)
        posterior_std = _finite_float(row.get("posterior_std"))
        uncertainty_score = (
            None if posterior_std is None else posterior_std / max(scale, 1e-6)
        )
        rows.append(
            {
                "entity_uid": str(row["entity_uid"]),
                "chain_id": chain_id,
                "residue_index": int(residue_index),
                "residue_name": residue_name,
                "residue_id": f"{chain_id}:{residue_index}:{residue_name}",
                "atom_family": atom_family,
                "target_id": str(row["target_id"]),
                "target_value": float(target_value),
                "random_coil_shift": float(reference),
                "secondary_shift": float(secondary_shift),
                "normalized_secondary_shift": float(normalized_shift),
                "posterior_uncertainty_score": uncertainty_score,
            }
        )
    if not rows:
        return pd.DataFrame()

    atom_frame = pd.DataFrame(rows)
    residue_rows: list[dict[str, Any]] = []
    for residue_key, residue_frame in atom_frame.groupby(
        ["entity_uid", "chain_id", "residue_index", "residue_name", "residue_id"],
        sort=True,
    ):
        shift_scores = pd.to_numeric(
            residue_frame["normalized_secondary_shift"],
            errors="coerce",
        ).dropna()
        if shift_scores.empty:
            continue
        secondary_score = float(np.mean(np.clip(shift_scores, 0.0, 12.0)))
        uncertainty_scores = pd.to_numeric(
            residue_frame["posterior_uncertainty_score"],
            errors="coerce",
        ).dropna()
        uncertainty_score = (
            math.nan
            if uncertainty_scores.empty
            else float(np.mean(np.clip(uncertainty_scores, 0.0, 12.0)))
        )
        coil_proximity = math.exp(-secondary_score)
        uncertainty_flexibility = (
            0.0
            if not math.isfinite(uncertainty_score)
            else 1.0 - math.exp(-uncertainty_score)
        )
        flexibility = float(
            np.clip(0.82 * coil_proximity + 0.18 * uncertainty_flexibility, 0.0, 1.0)
        )
        residue_rows.append(
            {
                "entity_uid": residue_key[0],
                "chain_id": residue_key[1],
                "residue_index": int(residue_key[2]),
                "residue_name": residue_key[3],
                "residue_id": residue_key[4],
                "rci_value": float(secondary_score),
                "rci_flexibility": flexibility,
                "rci_source": "pRCI_random_coil_proximity",
                "prci_secondary_shift_score": float(secondary_score),
                "prci_uncertainty_score": (
                    None if not math.isfinite(uncertainty_score) else uncertainty_score
                ),
                "residue_evidence_count": int(len(residue_frame)),
                "atom_families": "|".join(
                    sorted(residue_frame["atom_family"].astype(str).unique())
                ),
            }
        )
    profile = pd.DataFrame(residue_rows)
    if profile.empty:
        return profile
    if smooth:
        profile = _smooth_profile(profile)
    return profile.sort_values(
        ["entity_uid", "chain_id", "residue_index"],
        kind="stable",
    ).reset_index(drop=True)


def _smooth_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Apply entity/chain-local three-residue smoothing to pRCI flexibility."""

    smoothed = profile.copy()
    smoothed["rci_flexibility_raw"] = smoothed["rci_flexibility"]
    smoothed["rci_value_raw"] = smoothed["rci_value"]
    for _, indices in smoothed.groupby(
        ["entity_uid", "chain_id"], sort=False
    ).groups.items():
        ordered = smoothed.loc[list(indices)].sort_values("residue_index")
        for column in ["rci_flexibility", "rci_value"]:
            values = ordered[column].to_numpy(dtype=np.float64)
            if values.size < 3:
                smooth_values = values
            else:
                padded = np.pad(values, (1, 1), mode="edge")
                smooth_values = np.asarray(
                    [
                        float(np.mean(padded[index : index + 3]))
                        for index in range(values.size)
                    ],
                    dtype=np.float64,
                )
            smoothed.loc[ordered.index, column] = smooth_values
    smoothed["rci_flexibility"] = smoothed["rci_flexibility"].clip(0.0, 1.0)
    return smoothed


def _target_metadata(target_id: str) -> tuple[str, int, str, str] | None:
    """Parse AtypEmu chemical-shift target metadata."""

    if not target_id.startswith("cs:"):
        return None
    parts = target_id.split(":")
    if len(parts) < 5:
        return None
    try:
        residue_index = int(parts[2])
    except ValueError:
        return None
    atom_family = {
        "H": "HN",
        "N": "N",
        "CA": "CA",
        "CB": "CB",
        "C": "C'",
    }.get(parts[-1])
    if atom_family is None:
        return None
    return parts[1], residue_index, parts[3].upper(), atom_family


def _random_coil_shift(residue_name: str, atom_family: str) -> float | None:
    """Return an approximate random-coil shift for one residue and atom."""

    residue_table = RANDOM_COIL_SHIFTS.get(str(residue_name).upper(), {})
    return residue_table.get(atom_family, GENERIC_RANDOM_COIL_SHIFTS.get(atom_family))


def _finite_float(value: Any) -> float | None:
    """Convert a value to a finite float when possible."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
