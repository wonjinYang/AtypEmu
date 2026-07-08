"""Analytic NOE back-calculation for simple proton-proton restraints."""

from __future__ import annotations

import numpy as np
from Bio.PDB.Polypeptide import is_aa

from atypemu.structures.pool import StructureLoader
from atypemu.types import (
    CandidatePool,
    NMRTargetBundle,
    ObservableMatrix,
    canonical_atom_name,
)


def _coordinate(residue, atom_names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first matching atom coordinate for a residue."""
    for atom_name in atom_names:
        if atom_name in residue:
            return residue[atom_name].coord
    return None


class NOEHead:
    """Compute simple proton-proton NOE observables as ``r^-6`` values."""

    def __init__(self) -> None:
        """Initialize the NOE head."""
        self._loader = StructureLoader()

    def build(self, pool: CandidatePool, bundle: NMRTargetBundle) -> ObservableMatrix:
        """Construct the NOE observable matrix."""
        targets = bundle.noe_restraints
        values = np.zeros((len(targets), len(pool.records)), dtype=float)
        mask = np.zeros_like(values, dtype=bool)
        target_ids = [target.target_id() for target in targets]

        for column, record in enumerate(pool.records):
            structure = self._loader.load_structure(record.structure_path)
            residue_map = self._build_residue_map(structure)
            for row, target in enumerate(targets):
                inverse_sixth = self._compute_inverse_sixth(residue_map, target)
                if inverse_sixth is None:
                    continue
                values[row, column] = inverse_sixth
                mask[row, column] = True

        return ObservableMatrix(
            label="noe_restraints",
            values=values,
            mask=mask,
            target_ids=target_ids,
            candidate_ids=pool.candidate_ids,
            transform="inverse_sixth",
        )

    def _build_residue_map(self, structure) -> dict[int, object]:
        """Build a single-chain sequence map for one structure."""
        mapping: dict[int, object] = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    mapping[residue.id[1]] = residue
            break
        return mapping

    def _compute_inverse_sixth(
        self, residue_map: dict[int, object], target
    ) -> float | None:
        """Compute ``distance^-6`` for one simple NOE restraint."""
        residue_1 = residue_map.get(target.seq_id_1)
        residue_2 = residue_map.get(target.seq_id_2)
        if residue_1 is None or residue_2 is None:
            return None

        atom_names_1 = (canonical_atom_name(target.atom_id_1), "H")
        atom_names_2 = (canonical_atom_name(target.atom_id_2), "H")
        coord_1 = _coordinate(residue_1, atom_names_1)
        coord_2 = _coordinate(residue_2, atom_names_2)
        if coord_1 is None or coord_2 is None:
            return None

        distance = float(np.linalg.norm(coord_1 - coord_2))
        if distance <= 0.0:
            return None
        return distance**-6
