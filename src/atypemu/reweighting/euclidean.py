"""Euclidean simplex reweighting using SLSQP."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from atypemu.reweighting.base import BaseReweighter, GradientFn, ObjectiveFn
from atypemu.types import WeightSolution


class EuclideanSimplexReweighter(BaseReweighter):
    """Solve simplex-constrained objectives with Euclidean-style optimization."""

    method_name = "euclidean"

    def __init__(self, max_iter: int = 500, tolerance: float = 1e-10) -> None:
        """Initialize the optimizer."""
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(
        self,
        num_candidates: int,
        objective_fn: ObjectiveFn,
        gradient_fn: GradientFn | None = None,
        prior_weights: np.ndarray | None = None,
    ) -> WeightSolution:
        """Fit one weight vector with SLSQP."""
        weights_0 = (
            prior_weights.copy()
            if prior_weights is not None
            else np.full(num_candidates, 1.0 / num_candidates)
        )
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(num_candidates)]
        result = minimize(
            objective_fn,
            weights_0,
            jac=gradient_fn,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tolerance},
        )
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= max(weights.sum(), 1e-12)
        return WeightSolution(
            method=self.method_name,
            weights=weights,
            energy=float(objective_fn(weights)),
            converged=bool(result.success),
            iterations=int(result.nit),
            diagnostics={"status": float(result.status)},
        )
