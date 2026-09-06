"""Fixed-observer support diagnostic; no data loading or experiment entrypoint.

Run this file for synthetic checks only. Production use requires a separately
frozen source-only protocol. WLS bounds are not bounds on macro CCC.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize


def support_projection(surface, target, weight=None, *, tolerance=1e-8, maxiter=2000):
    """Bound min_q 0.5 * sum_i w_i (Aq-y)_i**2 on one entry simplex.

    Rows must include the full frozen observation cohort, columns complete
    conformers, and weights frozen positive precision/group weights. A failure
    to converge returns an interval, never a claimed support ceiling. No
    reference-offset fitting, KL, geometry repair, or row masking is performed.
    """
    a = np.asarray(surface, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) == 0 or y.shape != (a.shape[0],) or w.shape != y.shape:
        raise ValueError("expected nonempty [rows, supports] and matching target/weight")
    if not all(np.isfinite(v).all() for v in (a, y, w)) or np.any(w <= 0):
        raise ValueError("finite complete surface and positive weights required")
    if not np.isfinite(tolerance) or tolerance <= 0 or maxiter < 1:
        raise ValueError("positive tolerance and iteration budget required")

    def objective(q):
        residual = a @ q - y
        return 0.5 * np.dot(w * residual, residual), a.T @ (w * residual)

    initial = np.full(a.shape[1], 1.0 / a.shape[1])
    result = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * a.shape[1],
        constraints={"type": "eq", "fun": lambda q: q.sum() - 1.0,
                     "jac": lambda q: np.ones_like(q)},
        options={"ftol": tolerance * 0.01, "maxiter": maxiter},
    )
    if not np.isfinite(result.x).all():
        raise ValueError("optimizer returned nonfinite population")
    q = np.clip(result.x, 0.0, 1.0)
    if q.sum() <= 0:
        raise ValueError("optimizer returned an empty population")
    q /= q.sum()
    upper, gradient = objective(q)
    # Convex first-order minorant minimized at a simplex vertex gives a bound.
    gap = max(0.0, float(q @ gradient - gradient.min()))
    lower = max(0.0, float(upper) - gap)
    if not np.isfinite([upper, gap, lower]).all():
        raise ValueError("nonfinite objective or bound")
    return {
        "q": q, "prediction": a @ q,
        "wls_lower_bound": lower, "wls_upper_bound": float(upper),
        "optimality_gap": gap,
        "converged": bool(result.success and gap <= tolerance * (1.0 + abs(upper))),
        "solver_message": str(result.message),
    }


def self_test():
    # Rowwise reachability does not imply one shared population can fit both.
    conflict = np.array([[0., 1.], [1., 0.]])
    fit = support_projection(conflict, [1., 1.])
    assert fit["converged"]
    assert np.allclose(fit["q"], [0.5, 0.5])
    assert abs(fit["wls_lower_bound"] - 0.25) < 1e-8
    duplicated = support_projection(np.tile(conflict, (1, 8)), [1., 1.])
    assert abs(duplicated["wls_lower_bound"] - 0.25) < 1e-8
    expanded = support_projection(np.column_stack((conflict, [1., 1.])), [1., 1.])
    assert expanded["converged"] and expanded["wls_upper_bound"] < 1e-8
    singleton = support_projection([[0.], [0.]], [1., 2.], [2., 1.])
    assert abs(singleton["wls_lower_bound"] - 3.0) < 1e-8

    rng = np.random.default_rng(20260906)
    for _ in range(12):
        a, y = rng.normal(size=(7, 4)), rng.normal(size=7)
        w = rng.uniform(0.2, 2.0, size=7)
        fit = support_projection(a, y, w)
        samples = rng.dirichlet(np.ones(4), size=300)
        costs = 0.5 * ((samples @ a.T - y) ** 2 @ w)
        assert fit["wls_lower_bound"] <= costs.min() + 1e-9
        assert fit["wls_lower_bound"] <= fit["wls_upper_bound"] + 1e-9

    for a, y, w in [([[np.nan]], [1.], None), ([[1.]], [1.], [0.]),
                    ([[1.]], [np.inf], None), ([[1.]], [1., 2.], None)]:
        try:
            support_projection(a, y, w)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")

    # Per-label CCC >= c is a convex quadratic sublevel for c >= 0.
    y, prediction = rng.normal(size=(2, 20))
    c = 0.95
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    covariance = np.mean((y - y.mean()) * (prediction - prediction.mean()))
    quadratic = c * np.mean(prediction ** 2) - 2 * np.mean(
        ((y - y.mean()) + c * y.mean()) * prediction
    ) + c * np.mean(y ** 2)
    assert np.isclose(quadratic, c * denominator - 2 * covariance)
    print(json.dumps({"synthetic_checks": "PASS", "production_data_read": False,
                      "conflict_wls": 0.25, "expanded_wls": expanded["wls_upper_bound"]}))


if __name__ == "__main__":
    self_test()
