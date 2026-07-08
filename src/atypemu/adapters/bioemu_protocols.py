"""Protocol definitions for future BioEmu integration."""

from __future__ import annotations

from typing import Protocol


class PriorBatchSampler(Protocol):
    """Future interface for BioEmu-like conformer generators."""

    def sample_batch(self, sequence: str, batch_size: int) -> list[str]:
        """Return candidate structure paths sampled for one sequence."""


class PosteriorPotential(Protocol):
    """Future interface for posterior potentials used during steering."""

    def score_batch(self, sequence: str, structure_paths: list[str]) -> float:
        """Return a scalar posterior score for one candidate batch."""
