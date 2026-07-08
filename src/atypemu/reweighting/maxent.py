"""Mirror-descent style MaxEnt reweighting on the simplex."""

from __future__ import annotations

import numpy as np

from atypemu.errors import ReweightingError
from atypemu.reweighting.base import BaseReweighter, GradientFn, ObjectiveFn
from atypemu.types import WeightSolution


class MaxEntReweighter(BaseReweighter):
    """Optimize a simplex objective with exponentiated-gradient updates."""

    method_name = "maxent"

    def __init__(
        self,
        max_iter: int = 1000,
        learning_rate: float = 0.1,
        tolerance: float = 1e-8,
        weight_floor: float = 1e-12,
    ) -> None:
        """Initialize the optimizer."""
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.weight_floor = weight_floor

    def fit(
        self,
        num_candidates: int,
        objective_fn: ObjectiveFn,
        gradient_fn: GradientFn | None = None,
        prior_weights: np.ndarray | None = None,
    ) -> WeightSolution:
        """Fit one simplex weight vector with exponentiated gradients."""
        if gradient_fn is None:
            raise ReweightingError("MaxEnt reweighting requires an objective gradient.")

        weights = (
            prior_weights.copy()
            if prior_weights is not None
            else np.full(num_candidates, 1.0 / num_candidates)
        )
        previous_objective = objective_fn(weights)
        converged = False

        for iteration in range(1, self.max_iter + 1):
            gradient = gradient_fn(weights)
            logits = np.log(np.clip(weights, self.weight_floor, None))
            logits -= self.learning_rate * gradient
            logits -= logits.max()
            updated = np.exp(logits)
            updated /= max(updated.sum(), self.weight_floor)
            objective = objective_fn(updated)

            if (
                np.linalg.norm(updated - weights, ord=1) < self.tolerance
                or abs(objective - previous_objective) < self.tolerance
            ):
                weights = updated
                previous_objective = objective
                converged = True
                break

            weights = updated
            previous_objective = objective

        return WeightSolution(
            method=self.method_name,
            weights=weights,
            energy=float(previous_objective),
            converged=converged,
            iterations=iteration,
            diagnostics={"learning_rate": self.learning_rate},
        )
