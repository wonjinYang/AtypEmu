"""Shared CLI helpers for AtypEmu commands."""

from __future__ import annotations

import csv
from pathlib import Path

from atypemu.types import (
    CandidatePool,
    EnergyBreakdown,
    NMRTargetBundle,
    ObservableBundle,
    WeightSolution,
)


def load_bundle(path: str | Path) -> NMRTargetBundle:
    """Load an NMR target bundle from JSON."""
    return NMRTargetBundle.from_json(path)


def load_pool(path: str | Path) -> CandidatePool:
    """Load a candidate pool from JSONL."""
    return CandidatePool.from_jsonl(path)


def load_observables(path: str | Path) -> ObservableBundle:
    """Load observable matrices from NPZ."""
    return ObservableBundle.from_npz(path)


def save_weights_csv(path: str | Path, candidate_ids: list[str], weights) -> None:
    """Write weights to a small CSV file."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "weight"])
        for candidate_id, weight in zip(candidate_ids, weights, strict=True):
            writer.writerow([candidate_id, float(weight)])


def load_breakdown_as_solution(path: str | Path) -> WeightSolution:
    """Load a fit result and expose it as a weight solution."""
    breakdown = EnergyBreakdown.from_json(path)
    return WeightSolution(
        method="loaded",
        weights=breakdown.weights,
        energy=breakdown.energy,
        converged=breakdown.converged,
        iterations=breakdown.iterations,
        diagnostics=breakdown.diagnostics,
    )
