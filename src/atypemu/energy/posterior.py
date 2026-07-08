"""Multi-observable posterior energy for the AtypEmu offline core."""

from __future__ import annotations

import math

import numpy as np

from atypemu.reweighting.base import BaseReweighter
from atypemu.types import EnergyBreakdown, NMRTargetBundle, ObservableBundle


class ChannelNormalizer:
    """Channel-level Gaussian negative log-likelihood helpers."""

    @staticmethod
    def gaussian_nll(residual: float, sigma: float) -> float:
        """Return one Gaussian negative log-likelihood."""
        variance = max(sigma**2, 1e-12)
        return 0.5 * ((residual**2) / variance + math.log(2.0 * math.pi * variance))


class LikelihoodAggregator:
    """Bundle-level observable predictions and likelihood aggregation."""

    @staticmethod
    def normalized_prediction(
        row_values: np.ndarray,
        row_mask: np.ndarray,
        weights: np.ndarray,
        transform: str,
    ) -> tuple[float | None, np.ndarray]:
        """Compute one masked ensemble prediction and its weight gradient."""
        validity = row_mask.astype(float)
        denominator = float(np.dot(validity, weights))
        if denominator <= 1e-12:
            return None, np.zeros_like(weights)

        numerator = float(np.dot(validity * row_values, weights))
        averaged = numerator / denominator
        gradient_base = validity * (row_values - averaged) / denominator

        if transform == "identity":
            return averaged, gradient_base
        if transform == "inverse_sixth":
            if averaged <= 1e-12:
                return None, np.zeros_like(weights)
            prediction = averaged ** (-1.0 / 6.0)
            derivative = (-1.0 / 6.0) * averaged ** (-7.0 / 6.0)
            return prediction, derivative * gradient_base
        raise ValueError(f"Unsupported observable transform: {transform}")


class MultiObservablePosteriorEnergy:
    """Compute and optimize the AtypEmu multi-observable posterior energy."""

    def __init__(
        self,
        reweighter: BaseReweighter,
        beta_cs: float = 1.0,
        beta_j: float = 1.0,
        beta_noe: float = 1.0,
        lambda_reg: float = 0.01,
        regularizer: str = "maxent",
        default_cs_sigma: float = 1.0,
    ) -> None:
        """Initialize the posterior energy model."""
        self.reweighter = reweighter
        self.beta_cs = beta_cs
        self.beta_j = beta_j
        self.beta_noe = beta_noe
        self.lambda_reg = lambda_reg
        self.regularizer = regularizer
        self.default_cs_sigma = default_cs_sigma

    def score(
        self,
        bundle: NMRTargetBundle,
        observables: ObservableBundle,
        prior_weights: np.ndarray | None = None,
    ) -> EnergyBreakdown:
        """Fit weights and return a full energy breakdown."""
        num_candidates = len(observables.candidate_ids())
        prior = (
            prior_weights.copy()
            if prior_weights is not None
            else np.full(num_candidates, 1.0 / num_candidates)
        )
        solution = self.reweighter.fit(
            num_candidates=num_candidates,
            objective_fn=lambda weights: self.objective(
                weights, bundle, observables, prior
            ),
            gradient_fn=lambda weights: self.gradient(
                weights, bundle, observables, prior
            ),
            prior_weights=prior,
        )
        channel_scores, diagnostics = self.channel_scores(
            solution.weights, bundle, observables, prior
        )
        return EnergyBreakdown(
            energy=solution.energy,
            weights=solution.weights,
            channel_scores=channel_scores,
            ess=self._effective_sample_size(solution.weights),
            entropy=self._entropy(solution.weights),
            iterations=solution.iterations,
            converged=solution.converged,
            diagnostics=diagnostics | solution.diagnostics,
        )

    def objective(
        self,
        weights: np.ndarray,
        bundle: NMRTargetBundle,
        observables: ObservableBundle,
        prior_weights: np.ndarray,
    ) -> float:
        """Return the total posterior energy for one weight vector."""
        channel_scores, _ = self.channel_scores(
            weights, bundle, observables, prior_weights
        )
        total = (
            self.beta_cs * channel_scores.get("chemical_shifts", 0.0)
            + self.beta_j * channel_scores.get("j_couplings", 0.0)
            + self.beta_noe * channel_scores.get("noe_restraints", 0.0)
        )
        if self.regularizer == "maxent":
            clipped = np.clip(weights, 1e-12, None)
            prior = np.clip(prior_weights, 1e-12, None)
            total += self.lambda_reg * float(np.sum(clipped * np.log(clipped / prior)))
        return float(total)

    def gradient(
        self,
        weights: np.ndarray,
        bundle: NMRTargetBundle,
        observables: ObservableBundle,
        prior_weights: np.ndarray,
    ) -> np.ndarray:
        """Return the posterior-energy gradient for one weight vector."""
        gradient = np.zeros_like(weights, dtype=float)

        if observables.chemical_shifts is not None:
            gradient += self.beta_cs * self._channel_gradient(
                weights,
                observables.chemical_shifts,
                [target.value for target in bundle.chemical_shifts],
                [
                    (
                        target.uncertainty
                        if target.uncertainty is not None
                        else self.default_cs_sigma
                    )
                    for target in bundle.chemical_shifts
                ],
            )
        if observables.j_couplings is not None:
            gradient += self.beta_j * self._channel_gradient(
                weights,
                observables.j_couplings,
                [target.value for target in bundle.j_couplings],
                [
                    target.uncertainty if target.uncertainty is not None else 0.5
                    for target in bundle.j_couplings
                ],
            )
        if observables.noe_restraints is not None:
            gradient += self.beta_noe * self._channel_gradient(
                weights,
                observables.noe_restraints,
                [target.target_value for target in bundle.noe_restraints],
                [target.uncertainty for target in bundle.noe_restraints],
            )

        if self.regularizer == "maxent":
            clipped = np.clip(weights, 1e-12, None)
            prior = np.clip(prior_weights, 1e-12, None)
            gradient += self.lambda_reg * (np.log(clipped / prior) + 1.0)

        return gradient

    def channel_scores(
        self,
        weights: np.ndarray,
        bundle: NMRTargetBundle,
        observables: ObservableBundle,
        prior_weights: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return per-channel scores and diagnostics for one weight vector."""
        scores: dict[str, float] = {}
        diagnostics: dict[str, float] = {}

        if observables.chemical_shifts is not None:
            score, count = self._channel_score(
                weights,
                observables.chemical_shifts,
                [target.value for target in bundle.chemical_shifts],
                [
                    (
                        target.uncertainty
                        if target.uncertainty is not None
                        else self.default_cs_sigma
                    )
                    for target in bundle.chemical_shifts
                ],
            )
            scores["chemical_shifts"] = score
            diagnostics["chemical_shifts_count"] = float(count)
        if observables.j_couplings is not None:
            score, count = self._channel_score(
                weights,
                observables.j_couplings,
                [target.value for target in bundle.j_couplings],
                [
                    target.uncertainty if target.uncertainty is not None else 0.5
                    for target in bundle.j_couplings
                ],
            )
            scores["j_couplings"] = score
            diagnostics["j_couplings_count"] = float(count)
        if observables.noe_restraints is not None:
            score, count = self._channel_score(
                weights,
                observables.noe_restraints,
                [target.target_value for target in bundle.noe_restraints],
                [target.uncertainty for target in bundle.noe_restraints],
            )
            scores["noe_restraints"] = score
            diagnostics["noe_restraints_count"] = float(count)
        diagnostics["prior_mass"] = float(prior_weights.sum())
        return scores, diagnostics

    def predictions(
        self,
        weights: np.ndarray,
        observables: ObservableBundle,
    ) -> dict[str, np.ndarray]:
        """Return per-channel ensemble predictions."""
        predictions: dict[str, np.ndarray] = {}
        for name, matrix in observables.iter_channels():
            channel_predictions: list[float] = []
            for row in range(matrix.values.shape[0]):
                prediction, _ = LikelihoodAggregator.normalized_prediction(
                    matrix.values[row],
                    matrix.mask[row],
                    weights,
                    matrix.transform,
                )
                channel_predictions.append(np.nan if prediction is None else prediction)
            predictions[name] = np.asarray(channel_predictions, dtype=float)
        return predictions

    def _channel_score(
        self,
        weights: np.ndarray,
        matrix,
        target_values: list[float],
        sigmas: list[float],
    ) -> tuple[float, int]:
        """Return the average NLL for one observable channel."""
        total = 0.0
        count = 0
        for row in range(matrix.values.shape[0]):
            prediction, _ = LikelihoodAggregator.normalized_prediction(
                matrix.values[row],
                matrix.mask[row],
                weights,
                matrix.transform,
            )
            if prediction is None:
                continue
            sigma = max(float(sigmas[row]), 1e-12)
            total += ChannelNormalizer.gaussian_nll(
                prediction - float(target_values[row]),
                sigma,
            )
            count += 1
        return (total / count if count else 0.0), count

    def _channel_gradient(
        self,
        weights: np.ndarray,
        matrix,
        target_values: list[float],
        sigmas: list[float],
    ) -> np.ndarray:
        """Return the average NLL gradient for one observable channel."""
        gradient = np.zeros_like(weights, dtype=float)
        count = 0
        for row in range(matrix.values.shape[0]):
            prediction, prediction_grad = LikelihoodAggregator.normalized_prediction(
                matrix.values[row],
                matrix.mask[row],
                weights,
                matrix.transform,
            )
            if prediction is None:
                continue
            sigma = max(float(sigmas[row]), 1e-12)
            gradient += (
                (prediction - float(target_values[row])) / (sigma**2)
            ) * prediction_grad
            count += 1
        if count:
            gradient /= count
        return gradient

    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Return the effective sample size."""
        return float(1.0 / np.sum(np.square(weights)))

    def _entropy(self, weights: np.ndarray) -> float:
        """Return the Shannon entropy of the latent weights."""
        clipped = np.clip(weights, 1e-12, None)
        return float(-np.sum(clipped * np.log(clipped)))
