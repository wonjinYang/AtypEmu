#!/usr/bin/env python3
"""Independent synthetic checks for the certified shared-q reference solvers."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from shared_q_support_diagnostic import kl_support_projection


def objective(
    q: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    prior: np.ndarray,
    kl_coefficient: float,
) -> float:
    residual = a @ q - y
    return float(
        0.5 * np.dot(weight * residual, residual)
        + kl_coefficient * np.dot(q, np.log(q / prior))
    )


def check_against_independent_primal_solver() -> None:
    rng = np.random.default_rng(20260907)
    for _ in range(64):
        rows = int(rng.integers(2, 8))
        supports = int(rng.integers(2, 8))
        a = rng.normal(size=(rows, supports))
        y = rng.normal(size=rows)
        weight = rng.uniform(0.2, 3.0, size=rows)
        prior = rng.uniform(0.1, 2.0, size=supports)
        prior /= prior.sum()
        coefficient = float(rng.uniform(0.03, 2.0))

        certified = kl_support_projection(
            a,
            y,
            weight=weight,
            prior=prior,
            regularization=coefficient,
            tolerance=1e-11,
        )
        reference = minimize(
            objective,
            prior,
            args=(a, y, weight, prior, coefficient),
            method="SLSQP",
            bounds=[(1e-14, 1.0)] * supports,
            constraints={"type": "eq", "fun": lambda q: float(q.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 5000},
        )
        assert reference.success, reference.message
        reference_value = objective(reference.x, a, y, weight, prior, coefficient)
        assert certified["converged"]
        assert certified["lower_bound"] <= reference_value + 2e-8
        assert abs(certified["objective"] - reference_value) <= 2e-8
        assert certified["primal_dual_gap"] <= certified["gap_tolerance"]
        assert abs(float(np.sum(certified["q"])) - 1.0) <= 1e-12
        assert np.all(np.asarray(certified["q"]) > 0.0)


if __name__ == "__main__":
    check_against_independent_primal_solver()
    print(json.dumps({"independent_primal_cases": 64, "status": "PASS"}))
