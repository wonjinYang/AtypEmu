"""Thin adapters used by the AtypEmu offline core."""

from atypemu.adapters.bioemu_protocols import PriorBatchSampler, PosteriorPotential
from atypemu.adapters.cnnls_adapter import (
    build_prediction_map,
    load_candidate_shift_file,
    normalize_candidate_key,
)

__all__ = [
    "PosteriorPotential",
    "PriorBatchSampler",
    "build_prediction_map",
    "load_candidate_shift_file",
    "normalize_candidate_key",
]
