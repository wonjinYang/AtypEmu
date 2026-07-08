"""Analytic J-coupling back-calculation for AtypEmu."""

from __future__ import annotations

import math

import numpy as np
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import Vector, calc_dihedral

from atypemu.structures.pool import StructureLoader
from atypemu.types import CandidatePool, NMRTargetBundle, ObservableMatrix


def _coordinate(residue, atom_names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first matching atom coordinate for a residue."""
    for atom_name in atom_names:
        if atom_name in residue:
            return residue[atom_name].coord
    return None


class JCouplingHead:
    """Back-calculate ``3J_HNHA`` from backbone phi torsions."""

    def __init__(
        self,
        a: float = 6.51,
        b: float = -1.76,
        c: float = 1.60,
        phase_offset_deg: float = 60.0,
    ) -> None:
        """Initialize Karplus parameters.

        Args:
            a: Quadratic Karplus coefficient.
            b: Linear Karplus coefficient.
            c: Constant Karplus coefficient.
            phase_offset_deg: Phase offset in degrees.
        """
        self.a = a
        self.b = b
        self.c = c
        self.phase_offset = math.radians(phase_offset_deg)
        self._loader = StructureLoader()

    def build(self, pool: CandidatePool, bundle: NMRTargetBundle) -> ObservableMatrix:
        """Construct the J-coupling observable matrix."""
        targets = bundle.j_couplings
        values = np.zeros((len(targets), len(pool.records)), dtype=float)
        mask = np.zeros_like(values, dtype=bool)
        target_ids = [target.target_id() for target in targets]

        for column, record in enumerate(pool.records):
            structure = self._loader.load_structure(record.structure_path)
            residue_map = self._build_residue_map(structure)
            for row, target in enumerate(targets):
                coupling = self._compute_three_j_hnha(residue_map, target.seq_id)
                if coupling is None:
                    continue
                values[row, column] = coupling
                mask[row, column] = True

        return ObservableMatrix(
            label="j_couplings",
            values=values,
            mask=mask,
            target_ids=target_ids,
            candidate_ids=pool.candidate_ids,
            transform="identity",
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

    def _compute_three_j_hnha(
        self, residue_map: dict[int, object], seq_id: int
    ) -> float | None:
        """Compute ``3J_HNHA`` from the local backbone phi torsion."""
        residue = residue_map.get(seq_id)
        previous = residue_map.get(seq_id - 1)
        if residue is None or previous is None:
            return None

        c_prev = _coordinate(previous, ("C",))
        n_atom = _coordinate(residue, ("N",))
        ca_atom = _coordinate(residue, ("CA",))
        c_atom = _coordinate(residue, ("C",))
        if any(value is None for value in (c_prev, n_atom, ca_atom, c_atom)):
            return None

        phi = calc_dihedral(
            Vector(c_prev),
            Vector(n_atom),
            Vector(ca_atom),
            Vector(c_atom),
        )
        shifted = phi - self.phase_offset
        return self.a * math.cos(shifted) ** 2 + self.b * math.cos(shifted) + self.c
