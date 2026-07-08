"""Common reweighting interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from atypemu.types import WeightSolution

ObjectiveFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]


class BaseReweighter(ABC):
    """Abstract base class for simplex reweighting methods."""

    method_name: str = "base"

    @abstractmethod
    def fit(
        self,
        num_candidates: int,
        objective_fn: ObjectiveFn,
        gradient_fn: GradientFn | None = None,
        prior_weights: np.ndarray | None = None,
    ) -> WeightSolution:
        """Fit one simplex-constrained weight vector."""
