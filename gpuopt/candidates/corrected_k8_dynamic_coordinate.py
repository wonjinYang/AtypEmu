"""Corrected K=8 source-gate observer with dynamic complete-coordinate response."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import freesasa
from scipy.spatial import cKDTree
from torch import nn

from gpuopt.source_gate_eligibility import validate_eligibility_receipt


# Required by deterministic CUDA GEMM when torch deterministic algorithms are
# enabled.  This is set before the candidate creates a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


AA3 = (
    "<UNK>",
    "<NTERM>",
    "<CTERM>",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
AA_INDEX = {name: index for index, name in enumerate(AA3)}
AA1_TO_3 = dict(zip("ARNDCQEGHILKMFPSTWYV", AA3[3:], strict=True))
TORSION_NAMES = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
TORSION_COLUMNS = tuple(
    f"{name}_{suffix}"
    for name in TORSION_NAMES
    for suffix in ("sin", "cos", "available")
)
ESM_MODEL = "esm2_t30_150M_UR50D"
ESM_DIM = 640
ACTUATOR_SPECS = (
    ("phi", 0.0),
    ("psi", 0.0),
    ("chi1", 0.05),
    ("chi2", 0.05),
    ("chi3", 0.05),
    ("chi4", 0.05),
)
SIDECHAIN_CHI_BONDS = {
    "ARG": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")),
    "ASN": (("CA", "CB"), ("CB", "CG")),
    "ASP": (("CA", "CB"), ("CB", "CG")),
    "CYS": (("CA", "CB"),),
    "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
    "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD")),
    "HIS": (("CA", "CB"), ("CB", "CG")),
    "ILE": (("CA", "CB"), ("CB", "CG1")),
    "LEU": (("CA", "CB"), ("CB", "CG")),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD")),
    "PHE": (("CA", "CB"), ("CB", "CG")),
    "SER": (("CA", "CB"),),
    "THR": (("CA", "CB"),),
    "TRP": (("CA", "CB"), ("CB", "CG")),
    "TYR": (("CA", "CB"), ("CB", "CG")),
    "VAL": (("CA", "CB"),),
}
SUPPORT_COUNT = 8
UCB_ANCHOR_FILE_COUNT = 128
UCB_ANCHOR_AGGREGATE_SHA256 = (
    "2ceb7f3b551aefb027708ff9f8acdce263052a249bd7f82e7338afd9f48cc871"
)
OBSERVER_RESIDUAL_GAIN = 0.1
STRUCTURAL_RESPONSE_GAIN = 1.0
REFERENCE_ELEMENTS = {"H": 0, "C": 1, "N": 2}
REFERENCE_BOUND_LIMITS = ((0.1, 1.0), (0.5, 4.0), (1.0, 8.0))
NUMERIC_COLUMNS = (
    "relative_position",
    "log_length",
    "solution_ph",
    "solution_temperature_k",
    "solution_ionic_strength_mm",
    "solution_pressure_atm",
    "solution_ph_available",
    "solution_temperature_k_available",
    "solution_ionic_strength_mm_available",
    "solution_pressure_atm_available",
)
GEOMETRY_DISTANCE_COLUMNS = tuple(
    column
    for element in ("H", "C", "N", "O", "S")
    for column in (
        f"nearest_{element}_distance_angstrom",
        f"nearest_interresidue_{element}_distance_angstrom",
    )
) + (
    "hbond_acceptor_distance_angstrom",
    "nearest_aromatic_ring_distance_angstrom",
)
GEOMETRY_COLUMNS = (
    "target_residue_atom_count",
    *tuple(
        column
        for element in ("H", "C", "N", "O", "S")
        for column in (
            f"nearest_{element}_distance_angstrom",
            f"nearest_interresidue_{element}_distance_angstrom",
            *(f"interresidue_{element}_contact_count_{radius}a" for radius in (2, 3, 4, 5)),
        )
    ),
    "hbond_acceptor_distance_angstrom",
    "hbond_donor_h_acceptor_cosine",
    "nearest_aromatic_ring_distance_angstrom",
    "ring_current_geometry_factor_inverse_a3",
    "aromatic_ring_count",
    "geometry_available",
    "geometry_feature_complete",
)
SASA_COLUMNS = (
    "target_atom_sasa_angstrom2",
    "target_residue_sasa_angstrom2",
    "target_atom_sasa_available",
    "target_residue_sasa_available",
)
OBSERVER_GEOMETRY_COLUMNS = (*GEOMETRY_COLUMNS, *SASA_COLUMNS)
DYNAMIC_DISTANCE_COLUMNS = tuple(
    f"nearest_interresidue_{element}_distance_angstrom"
    for element in ("H", "C", "N", "O", "S")
)
DYNAMIC_DISTANCE_INDICES = tuple(
    OBSERVER_GEOMETRY_COLUMNS.index(column) for column in DYNAMIC_DISTANCE_COLUMNS
)
CHI_ACTUATOR_OFFSET = 2
FEATURE_COLUMNS = (
    "entity_uid",
    "target_id",
    "support_id",
    "seq_id",
    "comp_id",
    "atom_id",
    *NUMERIC_COLUMNS[2:],
    *GEOMETRY_COLUMNS,
    *TORSION_COLUMNS,
)
SOURCE_ROW_ARRAYS = (
    "esm",
    "torsion",
    "geometry",
    "numeric",
    "categorical",
    "residue_index",
    "entity_index",
    "distance_self_jacobian",
    "distance_neighbor_jacobian",
    "distance_neighbor_residue",
    "anchor",
    "sequence_support_anchor",
    "ucb_support_anchor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_source_eligibility(values: dict[str, Any]) -> None:
    """Require the corrected source-subset row contract at every model phase."""

    receipt = values.get("source_eligibility_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("model phase lacks source eligibility receipt")
    validate_eligibility_receipt(values["frame"], receipt)
    row_count = len(values["frame"])
    for key in SOURCE_ROW_ARRAYS:
        if key not in values:
            raise ValueError(f"model phase lacks row-aligned array: {key}")
        if len(values[key]) != row_count:
            raise ValueError(
                f"model phase row-alignment mismatch: {key} "
                f"has {len(values[key])} rows, expected {row_count}"
            )


def verify_support_structure_receipt(
    data_root: Path, structure_root: Path
) -> dict[str, Any]:
    """Verify every consumed support PDB against the frozen feature receipt."""

    receipt = json.loads((data_root / "feature_receipt.json").read_text())
    inputs: list[tuple[str, Path, str]] = []
    for row in receipt["files"]:
        bmrb_id = str(row["bmrb_id"])
        for support_index, expected in sorted(row["structure_sha256"].items()):
            identity = f"{bmrb_id}:BioEmu_{support_index}"
            path = (
                structure_root
                / bmrb_id
                / f"{bmrb_id}_BioEmu_{support_index}.pdb"
            )
            inputs.append((identity, path, str(expected)))
    aggregate = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=8) as executor:
        actual_hashes = executor.map(
            sha256_file, (path for _identity, path, _expected in inputs)
        )
        for (identity, path, expected), actual in zip(
            inputs, actual_hashes, strict=True
        ):
            if actual != expected:
                raise ValueError(f"support structure hash mismatch: {path}")
            aggregate.update(identity.encode() + b"\0" + actual.encode() + b"\n")
    return {
        "support_structure_count": len(inputs),
        "aggregate_sha256": aggregate.hexdigest(),
    }


def attach_ucbshift_anchor(
    values: dict[str, Any], root: Path
) -> dict[str, float | int | str]:
    paths = sorted(root.glob("*.npz"))
    if len(paths) != UCB_ANCHOR_FILE_COUNT:
        raise ValueError(f"unexpected UCBShift-X anchor file count: {len(paths)}")
    aggregate = hashlib.sha256()
    for path in paths:
        aggregate.update(path.name.encode() + b"\0")
        aggregate.update(bytes.fromhex(sha256_file(path)))
    actual_hash = aggregate.hexdigest()
    if actual_hash != UCB_ANCHOR_AGGREGATE_SHA256:
        raise ValueError("UCBShift-X anchor aggregate hash mismatch")

    sequence_anchor = np.asarray(values["anchor"], dtype=np.float32)
    sequence_support = np.repeat(sequence_anchor[:, None], SUPPORT_COUNT, axis=1)
    ucb_support = np.full_like(sequence_support, np.nan)
    target_rows = {
        (str(entity_uid), str(target_id)): row
        for row, (entity_uid, target_id) in enumerate(
            values["frame"][["entity_uid", "target_id"]].itertuples(
                index=False, name=None
            )
        )
    }
    replaced = np.zeros_like(sequence_support, dtype=bool)
    for entity in values["entities"]:
        path = root / f"{entity['bmrb_id']}.npz"
        if not path.exists():
            continue
        with np.load(path) as payload:
            support_ids = [str(value) for value in payload["support_ids"]]
            if support_ids != values["support_ids"]:
                raise ValueError(f"UCBShift-X support order mismatch: {path}")
            target_ids = payload["target_ids"].astype(str)
            prediction = payload["prediction_ppm"].astype(np.float32)
        if prediction.shape != (len(target_ids), SUPPORT_COUNT):
            raise ValueError(f"invalid UCBShift-X anchor shape: {path}")
        for source_row, target_id in enumerate(target_ids):
            destination = target_rows.get((str(entity["entity_uid"]), target_id))
            if destination is None:
                continue
            finite = np.isfinite(prediction[source_row])
            ucb_support[destination, finite] = prediction[source_row, finite]
            replaced[destination, finite] = True
    values["sequence_support_anchor"] = sequence_support
    values["ucb_support_anchor"] = ucb_support
    return {
        "aggregate_sha256": actual_hash,
        "file_count": len(paths),
        "ucb_finite_values": int(replaced.sum()),
        "total_values": int(replaced.size),
        "ucb_finite_fraction": float(replaced.mean()),
    }


def crossfit_anchor_selection(
    train: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    refine_by_comp_id: bool = False,
) -> dict[str, dict[str, float | int | str]]:
    require_source_eligibility(train)
    require_source_eligibility(evaluation)
    selection: dict[str, dict[str, float | int | str]] = {}
    train_target = train["frame"]["target_value"].to_numpy(dtype=np.float64)
    train_atom = train["frame"]["atom_id"].astype(str).to_numpy()

    def ccc(target: np.ndarray, prediction: np.ndarray) -> float:
        if len(target) < 2:
            return 0.0
        target_mean = float(target.mean())
        prediction_mean = float(prediction.mean())
        target_centered = target - target_mean
        prediction_centered = prediction - prediction_mean
        denominator = (
            float(np.mean(target_centered**2))
            + float(np.mean(prediction_centered**2))
            + (target_mean - prediction_mean) ** 2
        )
        if denominator <= 1.0e-15:
            return 0.0
        return 2.0 * float(np.mean(target_centered * prediction_centered)) / denominator

    for atom_id in sorted(set(train_atom)):
        rows = np.flatnonzero(train_atom == atom_id)
        ucb = train["ucb_support_anchor"][rows].mean(axis=1)
        finite = np.isfinite(ucb)
        sequence = train["sequence_support_anchor"][rows, 0]
        sequence_ccc = ccc(train_target[rows][finite], sequence[finite])
        ucb_ccc = ccc(train_target[rows][finite], ucb[finite])
        source = (
            "ucbshift_x"
            if finite.sum() >= 50 and ucb_ccc > sequence_ccc + 0.02
            else "sequence"
        )
        selection[atom_id] = {
            "source": source,
            "source_train_rows": int(finite.sum()),
            "source_train_sequence_ccc": sequence_ccc,
            "source_train_ucbshift_x_ccc": ucb_ccc,
        }

    cell_selection: dict[str, dict[str, float | int | str]] = {}
    if refine_by_comp_id:
        train_comp = train["frame"]["comp_id"].astype(str).to_numpy()
        for comp_id, atom_id in sorted(set(zip(train_comp, train_atom, strict=True))):
            rows = np.flatnonzero((train_comp == comp_id) & (train_atom == atom_id))
            ucb = train["ucb_support_anchor"][rows].mean(axis=1)
            finite = np.isfinite(ucb)
            sequence = train["sequence_support_anchor"][rows, 0]
            sequence_ccc = ccc(train_target[rows][finite], sequence[finite])
            ucb_ccc = ccc(train_target[rows][finite], ucb[finite])
            inherited = str(selection[atom_id]["source"])
            source = inherited
            if finite.sum() >= 50:
                if ucb_ccc > sequence_ccc + 0.02:
                    source = "ucbshift_x"
                elif sequence_ccc > ucb_ccc + 0.02:
                    source = "sequence"
            cell_selection[f"{comp_id}|{atom_id}"] = {
                "source": source,
                "inherited_atom_id_source": inherited,
                "source_train_rows": int(finite.sum()),
                "source_train_sequence_ccc": sequence_ccc,
                "source_train_ucbshift_x_ccc": ucb_ccc,
            }

    for values in (train, evaluation):
        sequence = values["sequence_support_anchor"]
        ucb = values["ucb_support_anchor"]
        selected = sequence.copy()
        for atom_id, receipt in selection.items():
            if receipt["source"] != "ucbshift_x":
                continue
            rows = values["frame"]["atom_id"].astype(str).eq(atom_id).to_numpy()
            finite = np.isfinite(ucb[rows])
            selected_rows = selected[rows]
            selected_rows[finite] = ucb[rows][finite]
            selected[rows] = selected_rows
        for cell, receipt in cell_selection.items():
            comp_id, atom_id = cell.split("|", 1)
            rows = (
                values["frame"]["comp_id"].astype(str).eq(comp_id)
                & values["frame"]["atom_id"].astype(str).eq(atom_id)
            ).to_numpy()
            if not rows.any():
                continue
            selected[rows] = sequence[rows]
            if receipt["source"] == "ucbshift_x":
                finite = np.isfinite(ucb[rows])
                selected_rows = selected[rows]
                selected_rows[finite] = ucb[rows][finite]
                selected[rows] = selected_rows
        values["support_anchor"] = selected
    train["cell_anchor_selection_receipt"] = {
        "enabled": refine_by_comp_id,
        "minimum_finite_rows": 50,
        "required_ccc_advantage": 0.02,
        "cells": cell_selection,
    }
    return selection


def embedding_path(cache: Path, sequence: str) -> Path:
    digest = hashlib.sha1(f"{ESM_MODEL}|{ESM_DIM}|{sequence}".encode()).hexdigest()[:16]
    return cache / f"esm2_{digest}.npz"


def residue_index(value: object) -> int:
    token = str(value or "").strip().upper()
    return AA_INDEX.get(AA1_TO_3.get(token, token), 0)


def support_sasa_features(path: Path, targets: pd.DataFrame) -> np.ndarray:
    """All-hydrogen FreeSASA features using target identity but never values."""

    freesasa.setVerbosity(freesasa.nowarnings)
    structure = freesasa.Structure(
        str(path),
        options={
            "hetatm": False,
            "hydrogen": True,
            "join-models": False,
            "skip-unknown": False,
            "halt-at-unknown": False,
        },
    )
    parameters = freesasa.Parameters(
        {
            "algorithm": freesasa.LeeRichards,
            "probe-radius": 1.4,
            "n-slices": 20,
            "n-threads": 1,
        }
    )
    result = freesasa.calc(structure, parameters)
    atom_area: dict[tuple[int, str, str], float] = {}
    residue_area: dict[tuple[int, str], float] = {}
    for index in range(structure.nAtoms()):
        seq_id = int(str(structure.residueNumber(index)).strip())
        comp_id = str(structure.residueName(index)).strip().upper()
        atom_id = str(structure.atomName(index)).strip().upper()
        atom_key = (seq_id, comp_id, atom_id)
        if atom_key in atom_area:
            raise ValueError(f"ambiguous SASA atom identity in {path}: {atom_key}")
        area = float(result.atomArea(index))
        if not math.isfinite(area) or area < 0.0:
            raise ValueError(f"invalid SASA atom area in {path}: {atom_key}")
        atom_area[atom_key] = area
        residue_key = (seq_id, comp_id)
        residue_area[residue_key] = residue_area.get(residue_key, 0.0) + area

    output = np.zeros((len(targets), len(SASA_COLUMNS)), dtype=np.float32)
    for row_number, row in enumerate(
        targets[["seq_id", "comp_id", "atom_id"]].itertuples(index=False)
    ):
        seq_id = int(row.seq_id)
        comp_id = str(row.comp_id).strip().upper()
        atom_id = str(row.atom_id).strip().upper()
        atom_alias = "H" if atom_id == "HN" else atom_id
        atom_key = (seq_id, comp_id, atom_alias)
        residue_key = (seq_id, comp_id)
        if atom_key in atom_area:
            output[row_number, 0] = atom_area[atom_key]
            output[row_number, 2] = 1.0
        if residue_key in residue_area:
            output[row_number, 1] = residue_area[residue_key]
            output[row_number, 3] = 1.0
    return output


def _pdb_coordinate_records(path: Path) -> dict[str, np.ndarray]:
    """Read the exact coordinate fields used by the emitted-coordinate path."""

    names: list[str] = []
    resnames: list[str] = []
    chains: list[str] = []
    seq_ids: list[int] = []
    elements: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip().upper()
        element = line[76:78].strip().upper()
        if not element:
            element = next((character for character in name if character.isalpha()), "")
        names.append(name)
        resnames.append(line[17:20].strip().upper())
        chains.append(line[21:22])
        seq_ids.append(int(line[22:26]))
        elements.append(element)
        coordinates.append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    if not coordinates:
        raise ValueError(f"no coordinates in support structure: {path}")
    return {
        "name": np.asarray(names, dtype=str),
        "resname": np.asarray(resnames, dtype=str),
        "chain": np.asarray(chains, dtype=str),
        "seq_id": np.asarray(seq_ids, dtype=np.int64),
        "element": np.asarray(elements, dtype=str),
        "coordinate": np.asarray(coordinates, dtype=np.float64),
    }


def _chi_coordinate_velocities(records: dict[str, np.ndarray]) -> np.ndarray:
    """Analytic d(position)/d(chi) for the exact side-chain rotation emitter."""

    coordinates = records["coordinate"]
    velocities = np.zeros((len(coordinates), 4, 3), dtype=np.float64)
    for chain, seq_id in sorted(set(zip(records["chain"], records["seq_id"], strict=True))):
        indices = np.flatnonzero(
            (records["chain"] == chain) & (records["seq_id"] == seq_id)
        )
        if len(indices) == 0:
            continue
        adjacency = {int(index): set() for index in indices}
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                hydrogen = (
                    str(records["name"][first]).startswith("H")
                    or str(records["name"][second]).startswith("H")
                )
                cutoff = 1.25 if hydrogen else 1.95
                if np.linalg.norm(coordinates[first] - coordinates[second]) <= cutoff:
                    adjacency[int(first)].add(int(second))
                    adjacency[int(second)].add(int(first))
        resname = str(records["resname"][indices[0]])
        for dimension, (proximal_name, distal_name) in enumerate(
            SIDECHAIN_CHI_BONDS.get(resname, ())
        ):
            proximal = next(
                (
                    int(index)
                    for index in indices
                    if records["name"][index] == proximal_name
                ),
                None,
            )
            distal_axis = next(
                (
                    int(index)
                    for index in indices
                    if records["name"][index] == distal_name
                ),
                None,
            )
            if proximal is None or distal_axis is None:
                continue
            distal = {distal_axis}
            stack = [distal_axis]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {proximal, distal_axis}:
                        continue
                    if neighbor not in distal:
                        distal.add(neighbor)
                        stack.append(neighbor)
            if any(
                records["name"][index] in {"N", "CA", "C", "O", "OXT"}
                for index in distal
            ):
                continue
            axis = coordinates[distal_axis] - coordinates[proximal]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12:
                continue
            unit = axis / axis_norm
            rotated = np.asarray(sorted(distal), dtype=np.int64)
            velocities[rotated, dimension] = np.cross(
                unit, coordinates[rotated] - coordinates[proximal]
            )
    return velocities


def support_interresidue_distance_jacobian(
    path: Path,
    targets: pd.DataFrame,
    *,
    entity_uid: str,
    residue_lookup: dict[tuple[str, int], int],
    cached_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearize retained inter-residue distances through emitted chi rotations.

    The returned derivatives are with respect to physical chi angles in radians.
    Each distance receives both the target-residue and nearest-neighbor-residue
    contribution.  No chemical-shift target values are read here.
    """

    records = _pdb_coordinate_records(path)
    coordinates = records["coordinate"]
    velocities = _chi_coordinate_velocities(records)
    row_count = len(targets)
    descriptor_count = len(DYNAMIC_DISTANCE_COLUMNS)
    self_jacobian = np.zeros((row_count, descriptor_count, 4), dtype=np.float32)
    neighbor_jacobian = np.zeros_like(self_jacobian)
    neighbor_residue = np.full((row_count, descriptor_count), -1, dtype=np.int64)

    element_indices: dict[str, np.ndarray] = {}
    element_trees: dict[str, cKDTree] = {}
    for element in ("H", "C", "N", "O", "S"):
        indices = np.flatnonzero(records["element"] == element)
        if len(indices):
            element_indices[element] = indices
            element_trees[element] = cKDTree(coordinates[indices])

    identities: dict[tuple[int, str, str], int] = {}
    for index, (chain, seq_id, name) in enumerate(
        zip(records["chain"], records["seq_id"], records["name"], strict=True)
    ):
        key = (int(seq_id), str(name), str(chain))
        if key in identities:
            raise ValueError(f"ambiguous target atom in {path}: {key}")
        identities[key] = index
    chains = sorted(set(records["chain"]))
    if len(chains) != 1:
        raise ValueError(f"expected one coordinate chain in {path}: {chains}")
    chain = str(chains[0])

    for row_number, row in enumerate(
        targets[["seq_id", "atom_id"]].itertuples(index=False)
    ):
        seq_id = int(row.seq_id)
        atom_name = "H" if str(row.atom_id).upper() == "HN" else str(row.atom_id).upper()
        target_index = identities.get((seq_id, atom_name, chain))
        if target_index is None:
            if np.any(np.asarray(cached_distances[row_number], dtype=float) < 9.999):
                raise ValueError(
                    f"cached geometry has distances for missing atom in {path}: "
                    f"{seq_id}:{atom_name}"
                )
            continue
        target_coordinate = coordinates[target_index]
        for descriptor, element in enumerate(("H", "C", "N", "O", "S")):
            if element not in element_trees:
                continue
            indices = element_indices[element]
            query_count = min(32, len(indices))
            _distance, local = element_trees[element].query(
                target_coordinate, k=query_count
            )
            local = np.atleast_1d(local)
            nearest_index = next(
                (
                    int(indices[int(candidate)])
                    for candidate in local
                    if int(records["seq_id"][indices[int(candidate)]]) != seq_id
                ),
                None,
            )
            if nearest_index is None:
                nearest_index = next(
                    (
                        int(index)
                        for index in indices
                        if int(records["seq_id"][index]) != seq_id
                    ),
                    None,
                )
            if nearest_index is None:
                continue
            displacement = target_coordinate - coordinates[nearest_index]
            distance = float(np.linalg.norm(displacement))
            cached = float(cached_distances[row_number, descriptor])
            if math.isfinite(cached) and cached < 9.999 and abs(distance - cached) > 2.0e-3:
                raise ValueError(
                    f"coordinate/cached distance mismatch in {path}: "
                    f"{seq_id}:{atom_name}:{element} {distance} != {cached}"
                )
            if distance <= 1.0e-12:
                continue
            direction = displacement / distance
            self_jacobian[row_number, descriptor] = (
                velocities[target_index] @ direction
            ).astype(np.float32)
            neighbor_seq_id = int(records["seq_id"][nearest_index])
            neighbor_number = residue_lookup.get((entity_uid, neighbor_seq_id), -1)
            if neighbor_number >= 0:
                neighbor_residue[row_number, descriptor] = neighbor_number
                neighbor_jacobian[row_number, descriptor] = (
                    -(velocities[nearest_index] @ direction)
                ).astype(np.float32)
    return self_jacobian, neighbor_jacobian, neighbor_residue


def load_fold(
    data_root: Path,
    entities: list[dict[str, Any]],
    *,
    atom_index: dict[str, int],
    embedding_cache: Path,
    structure_root: Path,
) -> dict[str, Any]:
    feature_receipt = json.loads((data_root / "feature_receipt.json").read_text())
    receipted_supports = {
        str(row["bmrb_id"]): {
            f"BioEmu_{support_index}" for support_index in row["structure_sha256"]
        }
        for row in feature_receipt["files"]
    }
    frames: list[pd.DataFrame] = []
    esm_arrays: list[np.ndarray] = []
    torsion_arrays: list[np.ndarray] = []
    geometry_arrays: list[np.ndarray] = []
    sasa_arrays: list[np.ndarray] = []
    distance_self_jacobian_arrays: list[np.ndarray] = []
    distance_neighbor_jacobian_arrays: list[np.ndarray] = []
    distance_neighbor_residue_arrays: list[np.ndarray] = []
    support_ids_reference: list[str] | None = None
    residue_lookup: dict[tuple[str, int], int] = {}
    residue_keys: list[tuple[str, int]] = []
    sasa_atom_available_count = 0
    sasa_residue_available_count = 0
    sasa_target_support_count = 0

    for entity_number, entity in enumerate(entities):
        entity_uid = str(entity["entity_uid"])
        bmrb_id = str(entity["bmrb_id"])
        sequence = str(entity["sequence"])
        support_ids = [str(value) for value in entity["support_ids"]]
        if len(support_ids) != SUPPORT_COUNT:
            raise ValueError(f"unexpected support count: {entity_uid}")
        if set(support_ids) != receipted_supports.get(bmrb_id):
            raise ValueError(f"consumed support set differs from receipt: {entity_uid}")
        if support_ids_reference is None:
            support_ids_reference = support_ids
        elif support_ids != support_ids_reference:
            raise ValueError("support ordering differs across entities")

        feature = pd.read_parquet(
            data_root / "features" / f"{bmrb_id}.parquet", columns=FEATURE_COLUMNS
        )
        target = pd.read_parquet(
            data_root / "targets" / f"{bmrb_id}.parquet",
            columns=("entity_uid", "target_id", "target_value"),
        )
        feature = feature[feature["entity_uid"].astype(str).eq(entity_uid)].copy()
        target = target[target["entity_uid"].astype(str).eq(entity_uid)].copy()
        target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce")
        target = target[target["target_value"].map(math.isfinite)].copy()
        feature = feature[feature["target_id"].isin(target["target_id"])].copy()
        feature["support_id"] = pd.Categorical(
            feature["support_id"].astype(str), categories=support_ids, ordered=True
        )
        feature = feature.sort_values(["target_id", "support_id"], kind="stable")
        if not (
            feature.groupby("target_id", observed=True)["support_id"].nunique()
            == SUPPORT_COUNT
        ).all():
            raise ValueError(f"incomplete support surface: {entity_uid}")
        first = feature.drop_duplicates("target_id", keep="first").copy()
        target = target[["target_id", "target_value"]]
        first = first.merge(target, on="target_id", validate="one_to_one")
        if len(first) != len(target):
            raise ValueError(f"target/feature mismatch: {entity_uid}")
        for seq_id in first["seq_id"].astype(int).unique():
            key = (entity_uid, int(seq_id))
            if key not in residue_lookup:
                residue_lookup[key] = len(residue_keys)
                residue_keys.append(key)
        first["residue_number"] = [
            residue_lookup[(entity_uid, int(seq_id))]
            for seq_id in first["seq_id"].astype(int)
        ]

        sequence_esm = np.load(embedding_path(embedding_cache, sequence))["features"]
        seq_index = first["seq_id"].to_numpy(dtype=np.int64) - 1
        if np.any(seq_index < 0) or np.any(seq_index >= len(sequence_esm)):
            raise ValueError(f"sequence embedding mismatch: {entity_uid}")
        esm_arrays.append(sequence_esm[seq_index].astype(np.float16))

        torsion = (
            feature[list(TORSION_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(len(first), SUPPORT_COUNT, len(TORSION_COLUMNS))
        )
        torsion_arrays.append(torsion)
        geometry_frame = feature[list(GEOMETRY_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        geometry_frame.loc[:, list(GEOMETRY_DISTANCE_COLUMNS)] = geometry_frame[
            list(GEOMETRY_DISTANCE_COLUMNS)
        ].fillna(10.0)
        geometry = (
            geometry_frame.fillna(0.0)
            .to_numpy(dtype=np.float32)
            .reshape(len(first), SUPPORT_COUNT, len(GEOMETRY_COLUMNS))
        )
        if not np.isfinite(geometry).all():
            raise ValueError(f"nonfinite geometry surface: {entity_uid}")
        geometry_arrays.append(geometry)
        support_sasa = []
        distance_self_jacobian = []
        distance_neighbor_jacobian = []
        distance_neighbor_residue = []
        dynamic_geometry_indices = [
            GEOMETRY_COLUMNS.index(column) for column in DYNAMIC_DISTANCE_COLUMNS
        ]
        for support_number, support_id in enumerate(support_ids):
            structure_path = (
                structure_root
                / bmrb_id
                / f"{bmrb_id}_{support_id}.pdb"
            )
            support_sasa.append(support_sasa_features(structure_path, first))
            self_jacobian, neighbor_jacobian, neighbor_residue = (
                support_interresidue_distance_jacobian(
                    structure_path,
                    first,
                    entity_uid=entity_uid,
                    residue_lookup=residue_lookup,
                    cached_distances=geometry[
                        :, support_number, dynamic_geometry_indices
                    ],
                )
            )
            distance_self_jacobian.append(self_jacobian)
            distance_neighbor_jacobian.append(neighbor_jacobian)
            distance_neighbor_residue.append(neighbor_residue)
        sasa = np.stack(support_sasa, axis=1)
        if not np.isfinite(sasa).all():
            raise ValueError(f"nonfinite SASA surface: {entity_uid}")
        expected_atom_available = (
            feature["geometry_feature_complete"]
            .astype(bool)
            .to_numpy()
            .reshape(len(first), SUPPORT_COUNT)
        )
        actual_atom_available = sasa[:, :, 2] > 0.5
        if not np.array_equal(actual_atom_available, expected_atom_available):
            raise ValueError(f"SASA/geometry atom coverage mismatch: {entity_uid}")
        sasa_atom_available_count += int(actual_atom_available.sum())
        sasa_residue_available_count += int((sasa[:, :, 3] > 0.5).sum())
        sasa_target_support_count += int(actual_atom_available.size)
        sasa_arrays.append(sasa)
        distance_self_jacobian_arrays.append(
            np.stack(distance_self_jacobian, axis=1)
        )
        distance_neighbor_jacobian_arrays.append(
            np.stack(distance_neighbor_jacobian, axis=1)
        )
        distance_neighbor_residue_arrays.append(
            np.stack(distance_neighbor_residue, axis=1)
        )

        first["entity_number"] = entity_number
        first["comp_number"] = first["comp_id"].map(residue_index)
        first["atom_number"] = (
            first["atom_id"].astype(str).map(atom_index).fillna(0).astype(int)
        )
        first["previous_number"] = [
            AA_INDEX["<NTERM>"] if seq_id <= 1 else residue_index(sequence[seq_id - 2])
            for seq_id in first["seq_id"].astype(int)
        ]
        first["next_number"] = [
            AA_INDEX["<CTERM>"]
            if seq_id >= len(sequence)
            else residue_index(sequence[seq_id])
            for seq_id in first["seq_id"].astype(int)
        ]
        first["relative_position"] = (first["seq_id"].astype(float) - 1.0) / max(
            len(sequence) - 1, 1
        )
        first["log_length"] = math.log(max(len(sequence), 1)) / 6.0
        frames.append(first)

    sasa_atom_coverage = sasa_atom_available_count / sasa_target_support_count
    sasa_residue_coverage = sasa_residue_available_count / sasa_target_support_count
    if min(sasa_atom_coverage, sasa_residue_coverage) < 0.999:
        raise ValueError(
            "SASA coverage below frozen gate: "
            f"atom={sasa_atom_coverage}, residue={sasa_residue_coverage}"
        )
    frame = pd.concat(frames, ignore_index=True)
    numeric = (
        frame[list(NUMERIC_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return {
        "frame": frame[
            ["entity_uid", "target_id", "comp_id", "atom_id", "target_value"]
        ].copy(),
        "esm": np.concatenate(esm_arrays),
        "torsion": np.concatenate(torsion_arrays),
        "geometry": np.concatenate(
            (
                np.concatenate(geometry_arrays),
                np.concatenate(sasa_arrays),
            ),
            axis=2,
        ),
        "numeric": numeric,
        "categorical": frame[
            ["comp_number", "atom_number", "previous_number", "next_number"]
        ].to_numpy(dtype=np.int64),
        "entity_index": frame["entity_number"].to_numpy(dtype=np.int64),
        "residue_index": frame["residue_number"].to_numpy(dtype=np.int64),
        "distance_self_jacobian": np.concatenate(distance_self_jacobian_arrays),
        "distance_neighbor_jacobian": np.concatenate(
            distance_neighbor_jacobian_arrays
        ),
        "distance_neighbor_residue": np.concatenate(
            distance_neighbor_residue_arrays
        ),
        "residue_keys": residue_keys,
        "support_ids": support_ids_reference or [],
        "entities": entities,
        "sasa_audit": {
            "target_support_count": sasa_target_support_count,
            "atom_available_count": sasa_atom_available_count,
            "residue_available_count": sasa_residue_available_count,
            "atom_coverage": sasa_atom_coverage,
            "residue_coverage": sasa_residue_coverage,
            "atom_availability_matches_geometry_complete": True,
        },
    }


def normalization(
    train: dict[str, Any], evaluation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_source_eligibility(train)
    require_source_eligibility(evaluation)
    mean = train["numeric"].mean(axis=0)
    scale = train["numeric"].std(axis=0).clip(min=1.0e-5)
    train["numeric"] = ((train["numeric"] - mean) / scale).astype(np.float32)
    evaluation["numeric"] = ((evaluation["numeric"] - mean) / scale).astype(np.float32)
    geometry_mean = train["geometry"].mean(axis=(0, 1), keepdims=True)
    geometry_scale = train["geometry"].std(axis=(0, 1), keepdims=True).clip(min=1.0e-5)
    train["geometry"] = ((train["geometry"] - geometry_mean) / geometry_scale).astype(
        np.float32
    )
    evaluation["geometry"] = (
        (evaluation["geometry"] - geometry_mean) / geometry_scale
    ).astype(np.float32)
    dynamic_scale = geometry_scale[0, 0, list(DYNAMIC_DISTANCE_INDICES)]
    for values in (train, evaluation):
        values["distance_self_jacobian"] = (
            values["distance_self_jacobian"] / dynamic_scale[None, None, :, None]
        ).astype(np.float32)
        values["distance_neighbor_jacobian"] = (
            values["distance_neighbor_jacobian"]
            / dynamic_scale[None, None, :, None]
        ).astype(np.float32)

    train_frame = train["frame"]
    cell = train_frame.groupby(["comp_id", "atom_id"])["target_value"].agg(
        ["mean", "std"]
    )
    atom = train_frame.groupby("atom_id")["target_value"].agg(["mean", "std"])
    element_frame = train_frame.assign(
        element=train_frame["atom_id"].astype(str).str[0]
    )
    element = element_frame.groupby("element")["target_value"].agg(["mean", "std"])
    global_mean = float(train_frame["target_value"].mean())
    global_scale = max(float(train_frame["target_value"].std()), 0.1)

    def attach(values: dict[str, Any]) -> None:
        centers = []
        scales = []
        for row in values["frame"][["comp_id", "atom_id"]].itertuples(index=False):
            key = (row.comp_id, row.atom_id)
            if key in cell.index:
                center, width = cell.loc[key, ["mean", "std"]]
            elif row.atom_id in atom.index:
                center, width = atom.loc[row.atom_id, ["mean", "std"]]
            elif str(row.atom_id)[0] in element.index:
                center, width = element.loc[str(row.atom_id)[0], ["mean", "std"]]
            else:
                center, width = global_mean, global_scale
            centers.append(float(center))
            scales.append(
                max(float(width) if math.isfinite(float(width)) else 0.1, 0.1)
            )
        if "support_anchor" in values:
            values["center"] = np.asarray(
                values["support_anchor"], dtype=np.float32
            ).mean(axis=1)
        elif "anchor" in values:
            values["center"] = np.asarray(values["anchor"], dtype=np.float32)
        else:
            values["center"] = np.asarray(centers, dtype=np.float32)
        values["scale"] = np.asarray(scales, dtype=np.float32)
        target = values["frame"]["target_value"].to_numpy(dtype=np.float32)
        values["normalized_target"] = (target - values["center"]) / values["scale"]

    attach(train)
    attach(evaluation)
    return train, evaluation


def fold_copy(values: dict[str, Any]) -> dict[str, Any]:
    """Copy only arrays mutated by direction-specific train normalization."""

    output = dict(values)
    output["numeric"] = values["numeric"].copy()
    output["geometry"] = values["geometry"].copy()
    output["distance_self_jacobian"] = values["distance_self_jacobian"].copy()
    output["distance_neighbor_jacobian"] = values[
        "distance_neighbor_jacobian"
    ].copy()
    return output


def calibrate_reference_bounds(values: dict[str, Any]) -> np.ndarray:
    """Estimate robust nuisance bounds from the opposite source fold only."""

    require_source_eligibility(values)
    target = values["frame"]["target_value"].to_numpy(dtype=np.float64)
    residual = target - np.asarray(values["center"], dtype=np.float64)
    atom = values["frame"]["atom_id"].astype(str).to_numpy(dtype=str)
    entity = np.asarray(values["entity_index"], dtype=np.int64)
    bounds = []
    for element, index in REFERENCE_ELEMENTS.items():
        medians = []
        for entity_index in range(len(values["entities"])):
            rows = (entity == entity_index) & np.char.startswith(atom, element)
            if rows.sum() >= 3:
                medians.append(float(np.median(residual[rows])))
        lower, upper = REFERENCE_BOUND_LIMITS[index]
        estimate = np.quantile(np.abs(medians), 1.0) if medians else lower
        bounds.append(float(np.clip(estimate, lower, upper)))
    return np.asarray(bounds, dtype=np.float32)


class CoordinateObserver(nn.Module):
    def __init__(
        self, atom_levels: int, width: int = 96, *, use_sasa: bool = True
    ) -> None:
        super().__init__()
        self.comp = nn.Embedding(len(AA3), 12)
        self.atom = nn.Embedding(atom_levels, 24)
        self.neighbor = nn.Embedding(len(AA3), 8)
        self.esm = nn.Sequential(
            nn.LayerNorm(ESM_DIM), nn.Linear(ESM_DIM, 48), nn.SiLU()
        )
        self.numeric = nn.Sequential(nn.Linear(10, 16), nn.SiLU())
        self.geometry_dimensions = len(
            OBSERVER_GEOMETRY_COLUMNS if use_sasa else GEOMETRY_COLUMNS
        )
        self.geometry = nn.Sequential(
            nn.Linear(self.geometry_dimensions, 32), nn.SiLU()
        )
        self.torsion = nn.Sequential(nn.Linear(len(TORSION_COLUMNS), 24), nn.SiLU())
        total = 12 + 24 + 2 * 8 + 48 + 16 + 32 + 24
        self.readout = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        esm: torch.Tensor,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        geometry: torch.Tensor,
        torsion: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.cat(
            (
                self.comp(categorical[:, 0]),
                self.atom(categorical[:, 1]),
                self.neighbor(categorical[:, 2]),
                self.neighbor(categorical[:, 3]),
                self.esm(esm),
                self.numeric(numeric),
                self.geometry(geometry[:, : self.geometry_dimensions]),
                self.torsion(torsion),
            ),
            dim=1,
        )
        return self.readout(state).squeeze(1)


def tensor(
    values: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def atom_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("atom_id")["target_id"].transform("count").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    return (weight / weight.mean()).astype(np.float32)


def configure_deterministic_torch(seed: int) -> None:
    """Make the fixed source-training seed reproducible across fresh processes."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def train_observer(
    data: dict[str, Any],
    *,
    atom_levels: int,
    device: torch.device,
    seed: int,
    epochs: int = 1024,
    batch_size: int = 4096,
    use_sasa: bool = True,
) -> CoordinateObserver:
    require_source_eligibility(data)
    configure_deterministic_torch(seed)
    model = CoordinateObserver(atom_levels, use_sasa=use_sasa).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, weight_decay=1.0e-4, foreach=False
    )
    esm = tensor(data["esm"], device, torch.float32)
    categorical = tensor(data["categorical"], device, torch.long)
    numeric = tensor(data["numeric"], device, torch.float32)
    geometry = tensor(data["geometry"], device, torch.float32)
    torsion = tensor(data["torsion"], device, torch.float32)
    target = tensor(data["normalized_target"], device, torch.float32)
    weight = tensor(atom_weights(data["frame"]), device, torch.float32)
    size = len(target)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(size, generator=generator, device=device)
        model.train()
        for start in range(0, size, batch_size):
            row = order[start : start + batch_size]
            count = len(row)
            prediction = model(
                esm[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                categorical[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                numeric[row].repeat_interleave(SUPPORT_COUNT, dim=0),
                geometry[row].reshape(-1, geometry.shape[-1]),
                torsion[row].reshape(-1, torsion.shape[-1]),
            ).reshape(count, SUPPORT_COUNT)
            loss = torch.mean(
                weight[row]
                * torch.nn.functional.smooth_l1_loss(
                    prediction.mean(dim=1), target[row], reduction="none"
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def actuator_delta(delta_raw: torch.Tensor) -> torch.Tensor:
    bounds = delta_raw.new_tensor([bound for _name, bound in ACTUATOR_SPECS])
    return bounds * torch.tanh(delta_raw)


def actuate_torsions(
    torsion: torch.Tensor,
    residue_index_tensor: torch.Tensor,
    delta_raw: torch.Tensor,
) -> torch.Tensor:
    delta = actuator_delta(delta_raw[residue_index_tensor])
    output = torsion.clone()
    for dimension, (name, _bound) in enumerate(ACTUATOR_SPECS):
        sin_index = TORSION_COLUMNS.index(f"{name}_sin")
        cos_index = TORSION_COLUMNS.index(f"{name}_cos")
        available_index = TORSION_COLUMNS.index(f"{name}_available")
        angle = delta[..., dimension] * torsion[..., available_index]
        sin_value = torsion[..., sin_index]
        cos_value = torsion[..., cos_index]
        output[..., sin_index] = sin_value * torch.cos(angle) + cos_value * torch.sin(
            angle
        )
        output[..., cos_index] = cos_value * torch.cos(angle) - sin_value * torch.sin(
            angle
        )
    return output


def actuate_coordinate_geometry(
    geometry: torch.Tensor,
    residue_index_tensor: torch.Tensor,
    delta_raw: torch.Tensor,
    self_jacobian: torch.Tensor,
    neighbor_jacobian: torch.Tensor,
    neighbor_residue: torch.Tensor,
    *,
    jacobian_scale: float = 1.0,
) -> torch.Tensor:
    """Apply the local complete-coordinate chi Jacobian to retained distances."""

    physical_delta = actuator_delta(delta_raw)[..., CHI_ACTUATOR_OFFSET:]
    self_delta = physical_delta[residue_index_tensor]
    change = torch.einsum("bksc,bkc->bks", self_jacobian, self_delta)
    safe_neighbor = neighbor_residue.clamp_min(0)
    support_index = torch.arange(
        SUPPORT_COUNT, device=geometry.device, dtype=torch.long
    ).view(1, SUPPORT_COUNT, 1).expand_as(safe_neighbor)
    neighbor_delta = physical_delta[safe_neighbor, support_index]
    neighbor_delta = neighbor_delta * (neighbor_residue >= 0).unsqueeze(-1)
    change = change + torch.einsum(
        "bksc,bksc->bks", neighbor_jacobian, neighbor_delta
    )
    additive = torch.zeros_like(geometry)
    additive[..., list(DYNAMIC_DISTANCE_INDICES)] = change
    return geometry + float(jacobian_scale) * additive


def coordinate_response(
    model: CoordinateObserver,
    esm: torch.Tensor,
    categorical: torch.Tensor,
    numeric: torch.Tensor,
    base_geometry: torch.Tensor,
    active_geometry: torch.Tensor,
    torsion: torch.Tensor,
) -> torch.Tensor:
    """Coordinate response from recomputed geometry, never edited torsion features."""

    count = len(esm)
    repeated_esm = esm.repeat_interleave(SUPPORT_COUNT, dim=0)
    repeated_categorical = categorical.repeat_interleave(SUPPORT_COUNT, dim=0)
    repeated_numeric = numeric.repeat_interleave(SUPPORT_COUNT, dim=0)
    base = model(
        repeated_esm,
        repeated_categorical,
        repeated_numeric,
        base_geometry.reshape(-1, base_geometry.shape[-1]),
        torsion.reshape(-1, torsion.shape[-1]),
    ).reshape(count, SUPPORT_COUNT)
    active = model(
        repeated_esm,
        repeated_categorical,
        repeated_numeric,
        active_geometry.reshape(-1, active_geometry.shape[-1]),
        torsion.reshape(-1, torsion.shape[-1]),
    ).reshape(count, SUPPORT_COUNT)
    base_mean = base.mean(dim=1, keepdim=True)
    return OBSERVER_RESIDUAL_GAIN * base_mean + STRUCTURAL_RESPONSE_GAIN * (
        active - base_mean
    )


def predict_surface(
    model: CoordinateObserver,
    values: dict[str, Any],
    *,
    device: torch.device,
    delta_raw: torch.Tensor | None,
    reference_offset: np.ndarray | None = None,
    geometry_jacobian_scale: float = 1.0,
    chunk_size: int = 2048,
) -> np.ndarray:
    require_source_eligibility(values)
    output = []
    for start in range(0, len(values["frame"]), chunk_size):
        stop = min(start + chunk_size, len(values["frame"]))
        esm = tensor(values["esm"][start:stop], device, torch.float32)
        categorical = tensor(values["categorical"][start:stop], device, torch.long)
        numeric = tensor(values["numeric"][start:stop], device, torch.float32)
        geometry = tensor(values["geometry"][start:stop], device, torch.float32)
        torsion = tensor(values["torsion"][start:stop], device, torch.float32)
        active_geometry = geometry
        if delta_raw is not None:
            residues = tensor(values["residue_index"][start:stop], device, torch.long)
            active_geometry = actuate_coordinate_geometry(
                geometry,
                residues,
                delta_raw,
                tensor(
                    values["distance_self_jacobian"][start:stop],
                    device,
                    torch.float32,
                ),
                tensor(
                    values["distance_neighbor_jacobian"][start:stop],
                    device,
                    torch.float32,
                ),
                tensor(
                    values["distance_neighbor_residue"][start:stop],
                    device,
                    torch.long,
                ),
                jacobian_scale=geometry_jacobian_scale,
            )
        flat_prediction = coordinate_response(
            model, esm, categorical, numeric, geometry, active_geometry, torsion
        )
        output.append(flat_prediction.detach().cpu().numpy())
    normalized = np.concatenate(output)
    anchor = values.get("support_anchor", values["center"][:, None])
    prediction = np.asarray(anchor) + values["scale"][:, None] * normalized
    if reference_offset is not None:
        element_index = np.asarray(
            [REFERENCE_ELEMENTS[str(atom_id)[0]] for atom_id in values["frame"]["atom_id"]],
            dtype=np.int64,
        )
        prediction = prediction + reference_offset[
            values["entity_index"], element_index
        ][:, None]
    return prediction


def optimize_assimilation(
    model: CoordinateObserver,
    values: dict[str, Any],
    *,
    device: torch.device,
    reference_bounds_ppm: np.ndarray,
    steps: int = 100,
    chunk_size: int = 2048,
    geometry_jacobian_scale: float = 1.0,
    fixed_uniform_q: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, float, float]:
    require_source_eligibility(values)
    residue_count = len(values["residue_keys"])
    entity_count = len(values["entities"])
    delta_raw = nn.Parameter(
        torch.zeros(residue_count, SUPPORT_COUNT, len(ACTUATOR_SPECS), device=device)
    )
    q_logits = nn.Parameter(torch.zeros(entity_count, SUPPORT_COUNT, device=device))
    reference_raw = nn.Parameter(torch.zeros(entity_count, 3, device=device))
    parameters: list[nn.Parameter] = [delta_raw, reference_raw]
    if not fixed_uniform_q:
        parameters.append(q_logits)
    optimizer = torch.optim.Adam(parameters, lr=0.08, foreach=False)
    weights = atom_weights(values["frame"])
    element_index = tensor(
        np.asarray(
            [REFERENCE_ELEMENTS[str(atom_id)[0]] for atom_id in values["frame"]["atom_id"]],
            dtype=np.int64,
        ),
        device,
        torch.long,
    )
    reference_bounds = tensor(
        np.asarray(reference_bounds_ppm, dtype=np.float32), device, torch.float32
    )
    row_count = len(values["frame"])

    def backward_data_objective() -> None:
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            esm = tensor(values["esm"][start:stop], device, torch.float32)
            categorical = tensor(values["categorical"][start:stop], device, torch.long)
            numeric = tensor(values["numeric"][start:stop], device, torch.float32)
            geometry = tensor(values["geometry"][start:stop], device, torch.float32)
            base_torsion = tensor(values["torsion"][start:stop], device, torch.float32)
            residues = tensor(values["residue_index"][start:stop], device, torch.long)
            entities = tensor(values["entity_index"][start:stop], device, torch.long)
            target = tensor(
                values["normalized_target"][start:stop], device, torch.float32
            )
            weight = tensor(weights[start:stop], device, torch.float32)
            active_geometry = actuate_coordinate_geometry(
                geometry,
                residues,
                delta_raw,
                tensor(
                    values["distance_self_jacobian"][start:stop],
                    device,
                    torch.float32,
                ),
                tensor(
                    values["distance_neighbor_jacobian"][start:stop],
                    device,
                    torch.float32,
                ),
                tensor(
                    values["distance_neighbor_residue"][start:stop],
                    device,
                    torch.long,
                ),
                jacobian_scale=geometry_jacobian_scale,
            )
            support_prediction = coordinate_response(
                model,
                esm,
                categorical,
                numeric,
                geometry,
                active_geometry,
                base_torsion,
            )
            anchor_deviation = tensor(
                (
                    values["support_anchor"][start:stop]
                    - values["center"][start:stop, None]
                )
                / values["scale"][start:stop, None],
                device,
                torch.float32,
            )
            support_prediction = support_prediction + anchor_deviation
            q = (
                torch.full_like(q_logits[entities], 1.0 / SUPPORT_COUNT)
                if fixed_uniform_q
                else torch.softmax(q_logits[entities], dim=1)
            )
            aggregate = torch.sum(q * support_prediction, dim=1)
            reference_offset = reference_bounds * torch.tanh(reference_raw)
            aggregate = aggregate + reference_offset[
                entities, element_index[start:stop]
            ] / tensor(values["scale"][start:stop], device, torch.float32)
            loss = torch.sum(weight * torch.square(aggregate - target)) / row_count
            loss.backward()

    def regularizer_objective() -> torch.Tensor:
        q = (
            torch.full_like(q_logits, 1.0 / SUPPORT_COUNT)
            if fixed_uniform_q
            else torch.softmax(q_logits, dim=1)
        )
        regularizer = 3.0e-2 * torch.mean(torch.square(torch.tanh(delta_raw)))
        if not fixed_uniform_q:
            regularizer = regularizer + 3.0e-3 * torch.mean(
                torch.sum(
                    q * torch.log((q * SUPPORT_COUNT).clamp_min(1.0e-12)), dim=1
                )
            )
        regularizer = regularizer + 3.0e-3 * torch.mean(
            torch.square(torch.tanh(reference_raw))
        )
        return regularizer

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        backward_data_objective()
        regularizer_objective().backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    backward_data_objective()
    data_gradient = (
        delta_raw.grad.detach().clone()
        if delta_raw.grad is not None
        else torch.zeros_like(delta_raw)
    )
    gradient_norm = float(data_gradient.norm().cpu())
    optimizer.zero_grad(set_to_none=True)
    regularizer_objective().backward()
    regularizer_gradient_norm = float(delta_raw.grad.norm().detach().cpu())
    if not torch.isfinite(data_gradient).all() or (
        geometry_jacobian_scale != 0.0 and gradient_norm <= 0.0
    ):
        raise ValueError("chemical-shift data loss has no finite coordinate gradient")
    reference_offset = (
        reference_bounds * torch.tanh(reference_raw.detach())
    ).cpu().numpy()
    return (
        delta_raw.detach(),
        (
            torch.full_like(q_logits.detach(), 1.0 / SUPPORT_COUNT)
            if fixed_uniform_q
            else torch.softmax(q_logits.detach(), dim=1)
        ),
        reference_offset,
        gradient_norm,
        regularizer_gradient_norm,
    )


def surface_frame(values: dict[str, Any], prediction: np.ndarray) -> pd.DataFrame:
    frames = []
    identity = values["frame"][["entity_uid", "target_id"]].reset_index(drop=True)
    for support_number, support_id in enumerate(values["support_ids"]):
        part = identity.copy()
        part["support_id"] = support_id
        part["support_prediction"] = prediction[:, support_number]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def q_frame(values: dict[str, Any], q: np.ndarray) -> pd.DataFrame:
    rows = []
    for entity_number, entity in enumerate(values["entities"]):
        weights = np.asarray(q[entity_number], dtype=np.float64)
        weights = weights / weights.sum()
        weights[-1] = 1.0 - float(weights[:-1].sum())
        for support_number, support_id in enumerate(values["support_ids"]):
            rows.append(
                {
                    "entity_uid": str(entity["entity_uid"]),
                    "support_id": support_id,
                    "posterior_weight": float(weights[support_number]),
                }
            )
    return pd.DataFrame(rows)


def rotate_about_axis(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    shifted = points - origin
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        shifted * cosine
        + np.cross(unit, shifted) * sine
        + np.outer(shifted @ unit, unit) * (1.0 - cosine)
        + origin
    )


def _coordinate_state(
    values: dict[str, Any],
    delta: np.ndarray,
    *,
    structure_root: Path,
    entity_number: int,
    support_number: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Emit one exact complete-coordinate state for the physicality audit."""

    entity = values["entities"][entity_number]
    entity_uid = str(entity["entity_uid"])
    support_id = values["support_ids"][support_number]
    path = (
        structure_root
        / str(entity["bmrb_id"])
        / f"{entity['bmrb_id']}_{support_id}.pdb"
    )

    records = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        records.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22],
                "seq_id": int(line[22:26]),
                "coord": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ),
            }
        )
    base = np.stack([record["coord"] for record in records])
    conditioned = base.copy()
    relevant = {
        seq_id: delta[index, support_number]
        for index, (uid, seq_id) in enumerate(values["residue_keys"])
        if uid == entity_uid
    }
    for seq_id, angles in sorted(relevant.items()):
        indices = [
            index for index, record in enumerate(records) if record["seq_id"] == seq_id
        ]
        if not indices:
            continue
        chain = records[indices[0]]["chain"]
        indices = [index for index in indices if records[index]["chain"] == chain]
        n_atom = next(
            (index for index in indices if records[index]["name"] == "N"), None
        )
        ca = next((index for index in indices if records[index]["name"] == "CA"), None)
        c_atom = next(
            (index for index in indices if records[index]["name"] == "C"), None
        )
        phi, psi = (float(value) for value in angles[:2])
        if n_atom is not None and ca is not None and abs(phi) > 1.0e-12:
            n_side = {"N", "H", "H1", "H2", "H3"}
            rotated = [
                index
                for index, record in enumerate(records)
                if record["chain"] == chain
                and (
                    record["seq_id"] > seq_id
                    or (record["seq_id"] == seq_id and record["name"] not in n_side)
                )
            ]
            conditioned[rotated] = rotate_about_axis(
                conditioned[rotated],
                conditioned[n_atom],
                conditioned[ca] - conditioned[n_atom],
                phi,
            )
        if ca is not None and c_atom is not None and abs(psi) > 1.0e-12:
            rotated = [
                index
                for index, record in enumerate(records)
                if record["chain"] == chain
                and (
                    record["seq_id"] > seq_id
                    or (record["seq_id"] == seq_id and record["name"] in {"O", "OXT"})
                )
            ]
            conditioned[rotated] = rotate_about_axis(
                conditioned[rotated],
                conditioned[ca],
                conditioned[c_atom] - conditioned[ca],
                psi,
            )
        adjacency = {index: set() for index in indices}
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                hydrogen = records[first]["name"].startswith("H") or records[second][
                    "name"
                ].startswith("H")
                cutoff = 1.25 if hydrogen else 1.95
                if np.linalg.norm(conditioned[first] - conditioned[second]) <= cutoff:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        resname = records[indices[0]]["resname"]
        bonds = SIDECHAIN_CHI_BONDS.get(resname, ())
        for angle, (proximal_name, distal_name) in zip(angles[2:], bonds, strict=False):
            angle = float(angle)
            if abs(angle) <= 1.0e-12:
                continue
            proximal = next(
                (index for index in indices if records[index]["name"] == proximal_name),
                None,
            )
            distal_axis = next(
                (index for index in indices if records[index]["name"] == distal_name),
                None,
            )
            if proximal is None or distal_axis is None:
                continue
            distal = {distal_axis}
            stack = [distal_axis]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {proximal, distal_axis}:
                        continue
                    if neighbor not in distal:
                        distal.add(neighbor)
                        stack.append(neighbor)
            if any(
                records[index]["name"] in {"N", "CA", "C", "O", "OXT"}
                for index in distal
            ):
                continue
            rotated = sorted(distal)
            conditioned[rotated] = rotate_about_axis(
                conditioned[rotated],
                conditioned[proximal],
                conditioned[distal_axis] - conditioned[proximal],
                angle,
            )
    return base, conditioned, records


def coordinate_audit(
    values: dict[str, Any],
    delta_raw: torch.Tensor,
    *,
    structure_root: Path,
    output: Path,
) -> None:
    """Emit every posterior support after exact bounded coordinate actuation."""

    require_source_eligibility(values)
    delta = actuator_delta(delta_raw).cpu().numpy()
    availability = np.zeros_like(delta)
    for row_number, residue_number in enumerate(values["residue_index"]):
        for dimension, (name, _bound) in enumerate(ACTUATOR_SPECS):
            available_index = TORSION_COLUMNS.index(f"{name}_available")
            availability[residue_number, :, dimension] = np.maximum(
                availability[residue_number, :, dimension],
                values["torsion"][row_number, :, available_index],
            )
    delta = delta * availability

    conditioned_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    atom_states: list[np.ndarray] = []
    atom_names: list[np.ndarray] = []
    atom_elements: list[np.ndarray] = []
    atom_seq_ids: list[np.ndarray] = []
    state_entities: list[str] = []
    state_supports: list[str] = []
    for entity_number, entity in enumerate(values["entities"]):
        for support_number, support_id in enumerate(values["support_ids"]):
            base, conditioned, records = _coordinate_state(
                values,
                delta,
                structure_root=structure_root,
                entity_number=entity_number,
                support_number=support_number,
            )
            state_number = len(conditioned_states)
            base_states.append(base)
            conditioned_states.append(conditioned)
            atom_states.append(np.full(len(base), state_number, dtype=np.int32))
            atom_names.append(
                np.asarray([record["name"] for record in records], dtype="U4")
            )
            atom_elements.append(
                np.asarray(
                    [
                        next(
                            (
                                character
                                for character in str(record["name"])
                                if character.isalpha()
                            ),
                            "",
                        ).upper()
                        for record in records
                    ],
                    dtype="U2",
                )
            )
            atom_seq_ids.append(
                np.asarray([record["seq_id"] for record in records], dtype=np.int32)
            )
            state_entities.append(str(entity["entity_uid"]))
            state_supports.append(str(support_id))
    conditioned = np.concatenate(conditioned_states, axis=0).astype(np.float32)
    base = np.concatenate(base_states, axis=0).astype(np.float32)
    atom_state = np.concatenate(atom_states)
    with output.open("xb") as handle:
        np.savez(
            handle,
            conditioned_coordinates=conditioned,
            no_evidence_coordinates=base,
            atom_mask=np.ones(len(base), dtype=bool),
            atom_state_index=atom_state,
            atom_name=np.concatenate(atom_names),
            atom_element=np.concatenate(atom_elements),
            atom_seq_id=np.concatenate(atom_seq_ids),
            state_entity_uid=np.asarray(state_entities, dtype="U128"),
            state_support_id=np.asarray(state_supports, dtype="U32"),
        )
