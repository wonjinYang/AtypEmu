"""Evaluation utilities for fitted AtypEmu weight solutions."""

from __future__ import annotations

import math

import numpy as np

from atypemu.energy.posterior import MultiObservablePosteriorEnergy
from atypemu.types import (
    EvaluationReport,
    NMRTargetBundle,
    ObservableBundle,
    WeightSolution,
)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation for finite values."""
    if x.size < 2 or y.size < 2:
        return math.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


class Evaluator:
    """Compute channel metrics and summary diagnostics for AtypEmu."""

    def __init__(self, energy_model: MultiObservablePosteriorEnergy) -> None:
        """Initialize the evaluator with the same energy model used for fitting."""
        self.energy_model = energy_model

    def evaluate(
        self,
        bundle: NMRTargetBundle,
        observables: ObservableBundle,
        solution: WeightSolution,
    ) -> EvaluationReport:
        """Evaluate one fitted weight solution.

        Args:
            bundle: Experimental target bundle.
            observables: Candidate observable matrices.
            solution: Optimized simplex weights.

        Returns:
            Evaluation report.
        """
        predictions = self.energy_model.predictions(solution.weights, observables)
        channel_metrics: dict[str, dict[str, float]] = {}
        global_metrics: dict[str, float] = {
            "energy": float(solution.energy),
            "ess": float(1.0 / np.sum(np.square(solution.weights))),
            "entropy": float(
                -np.sum(
                    np.clip(solution.weights, 1e-12, None)
                    * np.log(np.clip(solution.weights, 1e-12, None))
                )
            ),
            "top3_weight": float(np.sort(solution.weights)[-3:].sum()),
        }

        if "chemical_shifts" in predictions:
            target = np.asarray(
                [item.value for item in bundle.chemical_shifts], dtype=float
            )
            pred = predictions["chemical_shifts"]
            mask = np.isfinite(pred) & np.isfinite(target)
            residual = pred[mask] - target[mask]
            ss_res = float(np.sum(np.square(residual)))
            ss_tot = float(np.sum(np.square(target[mask] - np.mean(target[mask]))))
            channel_metrics["chemical_shifts"] = {
                "mae": float(np.mean(np.abs(residual))) if residual.size else math.nan,
                "correlation": _correlation(pred[mask], target[mask]),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
            }

        if "j_couplings" in predictions:
            target = np.asarray(
                [item.value for item in bundle.j_couplings], dtype=float
            )
            pred = predictions["j_couplings"]
            mask = np.isfinite(pred) & np.isfinite(target)
            residual = pred[mask] - target[mask]
            channel_metrics["j_couplings"] = {
                "mae": float(np.mean(np.abs(residual))) if residual.size else math.nan,
                "correlation": _correlation(pred[mask], target[mask]),
            }

        if "noe_restraints" in predictions:
            target = np.asarray(
                [item.target_value for item in bundle.noe_restraints], dtype=float
            )
            sigma = np.asarray(
                [item.uncertainty for item in bundle.noe_restraints], dtype=float
            )
            pred = predictions["noe_restraints"]
            mask = np.isfinite(pred) & np.isfinite(target)
            residual = pred[mask] - target[mask]
            satisfied = []
            for prediction, restraint in zip(pred, bundle.noe_restraints, strict=True):
                if not np.isfinite(prediction):
                    continue
                lower_ok = (
                    restraint.lower_bound is None or prediction >= restraint.lower_bound
                )
                upper_ok = (
                    restraint.upper_bound is None or prediction <= restraint.upper_bound
                )
                satisfied.append(lower_ok and upper_ok)
            channel_metrics["noe_restraints"] = {
                "mae": float(np.mean(np.abs(residual))) if residual.size else math.nan,
                "distance_z_score": (
                    float(np.mean(residual / sigma[mask]))
                    if residual.size
                    else math.nan
                ),
                "satisfaction_rate": (
                    float(np.mean(satisfied)) if satisfied else math.nan
                ),
            }

        return EvaluationReport(
            metrics=global_metrics,
            channel_metrics=channel_metrics,
            diagnostics=solution.diagnostics,
        )
