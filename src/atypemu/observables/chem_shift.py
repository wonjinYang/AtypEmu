"""Chemical-shift observable matrix construction."""

from __future__ import annotations

import numpy as np

from atypemu.adapters.cnnls_adapter import load_candidate_shift_file
from atypemu.types import (
    CandidatePool,
    NMRTargetBundle,
    ObservableMatrix,
    canonical_atom_name,
)


class ChemicalShiftMatrixBuilder:
    """Build candidate-by-target chemical-shift matrices."""

    def build(self, pool: CandidatePool, bundle: NMRTargetBundle) -> ObservableMatrix:
        """Construct the chemical-shift matrix.

        Args:
            pool: Candidate pool manifest.
            bundle: Experimental target bundle.

        Returns:
            Dense observable matrix with a boolean validity mask.
        """
        targets = bundle.chemical_shifts
        values = np.zeros((len(targets), len(pool.records)), dtype=float)
        mask = np.zeros_like(values, dtype=bool)
        target_ids = [target.target_id() for target in targets]
        cache: dict[str, dict[tuple[int, str], float]] = {}

        for column, record in enumerate(pool.records):
            if not record.chemical_shift_path:
                continue
            cache_key = f"{record.chemical_shift_path}:{record.chemical_shift_format}"
            if cache_key not in cache:
                cache[cache_key] = load_candidate_shift_file(
                    record.chemical_shift_path,
                    record.chemical_shift_format,
                )
            mapping = cache[cache_key]
            for row, target in enumerate(targets):
                key = (target.seq_id, canonical_atom_name(target.atom_id))
                if key not in mapping:
                    continue
                values[row, column] = mapping[key]
                mask[row, column] = True

        return ObservableMatrix(
            label="chemical_shifts",
            values=values,
            mask=mask,
            target_ids=target_ids,
            candidate_ids=pool.candidate_ids,
            transform="identity",
        )
