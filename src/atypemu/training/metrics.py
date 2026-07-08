"""Metric helpers for staged AtypEmu student training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atypemu.training.data import PreparedObservableChannel, PreparedTeacherExample
from atypemu.training.config import StudentTrainingConfig
from atypemu.training.secondary_shift import (
    random_coil_baselines_for_target_ids,
    random_coil_reference_for_target_id,
)


ATOM_FAMILY_MAP = {
    "H": "HN",
    "HN": "HN",
    "N": "N",
    "CA": "CA",
    "CB": "CB",
    "C": "C'",
}
NOE_METRIC_NAMES = (
    "noe_satisfaction_rate",
    "noe_violation_rate",
    "noe_mean_violation",
    "noe_rmse_z",
)


@dataclass(slots=True)
class EvaluatedChannel:
    """One evaluated observable channel for a prepared training example."""

    name: str
    predictions: np.ndarray
    targets: np.ndarray
    sigmas: np.ndarray
    target_ids: list[str]
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    posterior_stats: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Return the number of usable measurements in the channel."""
        return int(self.predictions.size)


@dataclass(slots=True)
class EvaluatedExample:
    """All metrics derived from one prepared example and one model prediction."""

    entity_uid: str
    bmrb_id: str
    split: str
    support_size: int
    teacher_kl: float
    teacher_js: float
    teacher_top10_mass_overlap: float
    teacher_ess_abs_error: float
    teacher_entropy_abs_error: float
    channels: dict[str, EvaluatedChannel]
    predicted_weights: np.ndarray | None = None
    teacher_weights: np.ndarray | None = None
    chemical_shift_residual_mu: np.ndarray | None = None
    chemical_shift_moment_mu: np.ndarray | None = None


def evaluate_prepared_example(
    example: PreparedTeacherExample,
    predicted_weights: np.ndarray,
    chemical_shift_residual_mu: np.ndarray | None = None,
    chemical_shift_moment_mu: np.ndarray | None = None,
) -> EvaluatedExample:
    """Evaluate one prepared example against predicted support weights.

    Args:
        example: Prepared teacher example.
        predicted_weights: Predicted candidate weights in pool order.

    Returns:
        Per-example metric payload.
    """
    teacher = np.asarray(example.teacher_weights, dtype=np.float64)
    predicted = np.asarray(predicted_weights, dtype=np.float64)
    teacher = teacher / np.clip(teacher.sum(), 1e-12, None)
    predicted = predicted / np.clip(predicted.sum(), 1e-12, None)
    clipped_teacher = np.clip(teacher, 1e-12, None)
    clipped_predicted = np.clip(predicted, 1e-12, None)
    mixture = 0.5 * (clipped_teacher + clipped_predicted)

    teacher_kl = float(
        np.sum(clipped_teacher * (np.log(clipped_teacher) - np.log(clipped_predicted)))
    )
    teacher_js = float(
        0.5 * np.sum(clipped_teacher * (np.log(clipped_teacher) - np.log(mixture)))
        + 0.5
        * np.sum(clipped_predicted * (np.log(clipped_predicted) - np.log(mixture)))
    )
    teacher_top10_mass_overlap = float(
        np.minimum(
            clipped_teacher[
                _topk_union_indices(clipped_teacher, clipped_predicted, 10)
            ],
            clipped_predicted[
                _topk_union_indices(clipped_teacher, clipped_predicted, 10)
            ],
        ).sum()
    )
    teacher_ess_abs_error = abs(
        _effective_support_size(predicted) - _effective_support_size(teacher)
    )
    teacher_entropy_abs_error = abs(_entropy(predicted) - _entropy(teacher))

    evaluated_channels: dict[str, EvaluatedChannel] = {}
    for channel_name, channel in example.channels.items():
        evaluated = _evaluate_channel(
            channel_name,
            predicted,
            channel,
            chemical_shift_residual_mu=chemical_shift_residual_mu,
            chemical_shift_moment_mu=chemical_shift_moment_mu,
        )
        if evaluated is not None:
            evaluated_channels[channel_name] = evaluated

    return EvaluatedExample(
        entity_uid=example.entity_uid,
        bmrb_id=example.metadata.get("bmrb_id", ""),
        split=example.split,
        support_size=int(predicted.size),
        teacher_kl=teacher_kl,
        teacher_js=teacher_js,
        teacher_top10_mass_overlap=teacher_top10_mass_overlap,
        teacher_ess_abs_error=float(teacher_ess_abs_error),
        teacher_entropy_abs_error=float(teacher_entropy_abs_error),
        channels=evaluated_channels,
        predicted_weights=predicted.astype(np.float64, copy=False),
        teacher_weights=teacher.astype(np.float64, copy=False),
        chemical_shift_residual_mu=(
            None
            if chemical_shift_residual_mu is None
            else np.asarray(chemical_shift_residual_mu, dtype=np.float64)
        ),
        chemical_shift_moment_mu=(
            None
            if chemical_shift_moment_mu is None
            else np.asarray(chemical_shift_moment_mu, dtype=np.float64)
        ),
    )


def aggregate_epoch_metrics(
    payloads: list[EvaluatedExample],
    split: str,
    report_micro_metrics: bool,
) -> dict[str, Any]:
    """Aggregate per-example payloads into metric rows and flat summaries.

    Args:
        payloads: Evaluated example payloads for one split.
        split: Split label such as ``train`` or ``val``.
        report_micro_metrics: Whether to emit micro-averaged metrics.

    Returns:
        Dictionary with ``metric_rows``, ``metric_summary``, and ``eligible_counts``.
    """
    metric_rows: list[dict[str, Any]] = []
    metric_summary: dict[str, float] = {}
    eligible_counts: dict[str, dict[str, int]] = {}

    teacher_metrics = {
        "teacher_kl": [payload.teacher_kl for payload in payloads],
        "teacher_js": [payload.teacher_js for payload in payloads],
        "teacher_top10_mass_overlap": [
            payload.teacher_top10_mass_overlap for payload in payloads
        ],
        "teacher_ess_abs_error": [
            payload.teacher_ess_abs_error for payload in payloads
        ],
        "teacher_entropy_abs_error": [
            payload.teacher_entropy_abs_error for payload in payloads
        ],
    }
    support_sizes = [payload.support_size for payload in payloads]
    for metric_name, values in teacher_metrics.items():
        _append_scalar_metric(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name=metric_name,
            values=values,
            measurement_counts=support_sizes,
            report_micro_metrics=report_micro_metrics,
        )

    channel_map = {
        "chemical_shifts": "cs",
        "j_couplings": "j",
        "noe_restraints": "noe",
    }
    for channel_name, prefix in channel_map.items():
        channel_payloads = [
            payload.channels[channel_name]
            for payload in payloads
            if channel_name in payload.channels
            and payload.channels[channel_name].count > 0
        ]
        if not channel_payloads:
            continue

        predictions = [channel.predictions for channel in channel_payloads]
        targets = [channel.targets for channel in channel_payloads]
        sigmas = [channel.sigmas for channel in channel_payloads]
        counts = [channel.count for channel in channel_payloads]

        if prefix in {"cs", "j"}:
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name=f"{prefix}_mae_ppm" if prefix == "cs" else "j_mae_hz",
                per_example_values=[
                    float(np.mean(np.abs(prediction - target)))
                    for prediction, target in zip(predictions, targets, strict=True)
                ],
                micro_value=_mae(np.concatenate(predictions), np.concatenate(targets)),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name=f"{prefix}_rmse_ppm" if prefix == "cs" else "j_rmse_hz",
                per_example_values=[
                    _rmse(prediction, target)
                    for prediction, target in zip(predictions, targets, strict=True)
                ],
                micro_value=_rmse(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name=f"{prefix}_rmse_z",
                per_example_values=[
                    _rmse_z(prediction, target, sigma)
                    for prediction, target, sigma in zip(
                        predictions, targets, sigmas, strict=True
                    )
                ],
                micro_value=_rmse_z(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                    np.concatenate(sigmas),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )

        if prefix == "cs":
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="cs_pearson_r",
                per_example_values=[
                    _safe_pearson(prediction, target)
                    for prediction, target in zip(predictions, targets, strict=True)
                ],
                micro_value=_safe_pearson(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="cs_ccc",
                per_example_values=[
                    _safe_ccc(prediction, target)
                    for prediction, target in zip(predictions, targets, strict=True)
                ],
                micro_value=_safe_ccc(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="cs_ccc_loss",
                per_example_values=[
                    _ccc_loss_value(prediction, target)
                    for prediction, target in zip(predictions, targets, strict=True)
                ],
                micro_value=_ccc_loss_value(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            posterior_channels = [
                channel
                for channel in channel_payloads
                if {"crps", "nll", "q05", "q10", "q25", "q75", "q90", "q95"}
                <= set(channel.posterior_stats)
            ]
            if posterior_channels:
                posterior_counts = [channel.count for channel in posterior_channels]
                crps_arrays = [
                    channel.posterior_stats["crps"] for channel in posterior_channels
                ]
                nll_arrays = [
                    channel.posterior_stats["nll"] for channel in posterior_channels
                ]
                coverage_50_arrays = [
                    (
                        (channel.targets >= channel.posterior_stats["q25"])
                        & (channel.targets <= channel.posterior_stats["q75"])
                    ).astype(np.float64)
                    for channel in posterior_channels
                ]
                coverage_80_arrays = [
                    (
                        (channel.targets >= channel.posterior_stats["q10"])
                        & (channel.targets <= channel.posterior_stats["q90"])
                    ).astype(np.float64)
                    for channel in posterior_channels
                ]
                coverage_95_arrays = [
                    (
                        (channel.targets >= channel.posterior_stats["q05"])
                        & (channel.targets <= channel.posterior_stats["q95"])
                    ).astype(np.float64)
                    for channel in posterior_channels
                ]
                interval_width_arrays = [
                    channel.posterior_stats["q95"] - channel.posterior_stats["q05"]
                    for channel in posterior_channels
                ]
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name="cs_crps",
                    per_example_values=[
                        float(np.mean(values)) for values in crps_arrays
                    ],
                    micro_value=float(np.mean(np.concatenate(crps_arrays))),
                    counts=posterior_counts,
                    report_micro_metrics=report_micro_metrics,
                )
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name="cs_nll",
                    per_example_values=[
                        float(np.mean(values)) for values in nll_arrays
                    ],
                    micro_value=float(np.mean(np.concatenate(nll_arrays))),
                    counts=posterior_counts,
                    report_micro_metrics=report_micro_metrics,
                )
                for metric_name, coverage_arrays in [
                    ("cs_coverage_50", coverage_50_arrays),
                    ("cs_coverage_80", coverage_80_arrays),
                    ("cs_coverage_95", coverage_95_arrays),
                ]:
                    _append_vector_metric(
                        metric_rows=metric_rows,
                        metric_summary=metric_summary,
                        eligible_counts=eligible_counts,
                        split=split,
                        metric_name=metric_name,
                        per_example_values=[
                            float(np.mean(values)) for values in coverage_arrays
                        ],
                        micro_value=float(np.mean(np.concatenate(coverage_arrays))),
                        counts=posterior_counts,
                        report_micro_metrics=report_micro_metrics,
                    )
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name="cs_interval_width",
                    per_example_values=[
                        float(np.mean(values)) for values in interval_width_arrays
                    ],
                    micro_value=float(np.mean(np.concatenate(interval_width_arrays))),
                    counts=posterior_counts,
                    report_micro_metrics=report_micro_metrics,
                )
            for family_name in sorted(set(ATOM_FAMILY_MAP.values())):
                family_predictions: list[np.ndarray] = []
                family_targets: list[np.ndarray] = []
                family_counts: list[int] = []
                family_example_mae: list[float] = []
                family_example_rmse: list[float] = []
                family_example_ccc: list[float] = []
                family_example_ccc_loss: list[float] = []
                for channel in channel_payloads:
                    family_mask = np.asarray(
                        [
                            _atom_family_from_target_id(target_id) == family_name
                            for target_id in channel.target_ids
                        ],
                        dtype=bool,
                    )
                    if not np.any(family_mask):
                        continue
                    family_prediction = channel.predictions[family_mask]
                    family_target = channel.targets[family_mask]
                    family_predictions.append(family_prediction)
                    family_targets.append(family_target)
                    family_counts.append(int(family_prediction.size))
                    family_example_mae.append(_mae(family_prediction, family_target))
                    family_example_rmse.append(_rmse(family_prediction, family_target))
                    family_example_ccc.append(
                        _safe_ccc(family_prediction, family_target)
                    )
                    family_example_ccc_loss.append(
                        _ccc_loss_value(family_prediction, family_target)
                    )
                if not family_predictions:
                    continue
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name=f"cs_{family_name}_mae_ppm",
                    per_example_values=family_example_mae,
                    micro_value=_mae(
                        np.concatenate(family_predictions),
                        np.concatenate(family_targets),
                    ),
                    counts=family_counts,
                    report_micro_metrics=report_micro_metrics,
                )
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name=f"cs_{family_name}_ccc",
                    per_example_values=family_example_ccc,
                    micro_value=_safe_ccc(
                        np.concatenate(family_predictions),
                        np.concatenate(family_targets),
                    ),
                    counts=family_counts,
                    report_micro_metrics=report_micro_metrics,
                )
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name=f"cs_{family_name}_ccc_loss",
                    per_example_values=family_example_ccc_loss,
                    micro_value=_ccc_loss_value(
                        np.concatenate(family_predictions),
                        np.concatenate(family_targets),
                    ),
                    counts=family_counts,
                    report_micro_metrics=report_micro_metrics,
                )
                _append_vector_metric(
                    metric_rows=metric_rows,
                    metric_summary=metric_summary,
                    eligible_counts=eligible_counts,
                    split=split,
                    metric_name=f"cs_{family_name}_rmse_ppm",
                    per_example_values=family_example_rmse,
                    micro_value=_rmse(
                        np.concatenate(family_predictions),
                        np.concatenate(family_targets),
                    ),
                    counts=family_counts,
                    report_micro_metrics=report_micro_metrics,
                )
            _append_family_balanced_ccc_metrics(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                channel_payloads=channel_payloads,
                report_micro_metrics=report_micro_metrics,
            )
            _append_secondary_chemical_shift_metrics(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                channel_payloads=channel_payloads,
                report_micro_metrics=report_micro_metrics,
            )

        if prefix == "noe":
            lower_bounds = [
                (
                    channel.lower_bounds
                    if channel.lower_bounds is not None
                    else np.full(channel.count, np.nan, dtype=np.float32)
                )
                for channel in channel_payloads
            ]
            upper_bounds = [
                (
                    channel.upper_bounds
                    if channel.upper_bounds is not None
                    else np.full(channel.count, np.nan, dtype=np.float32)
                )
                for channel in channel_payloads
            ]
            satisfaction_lists = [
                _noe_satisfaction_mask(prediction, lower, upper)
                for prediction, lower, upper in zip(
                    predictions, lower_bounds, upper_bounds, strict=True
                )
            ]
            violation_amounts = [
                _noe_violation_amount(prediction, lower, upper)
                for prediction, lower, upper in zip(
                    predictions, lower_bounds, upper_bounds, strict=True
                )
            ]
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="noe_satisfaction_rate",
                per_example_values=[
                    float(np.mean(satisfaction.astype(np.float64)))
                    for satisfaction in satisfaction_lists
                ],
                micro_value=float(
                    np.mean(np.concatenate(satisfaction_lists).astype(np.float64))
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="noe_violation_rate",
                per_example_values=[
                    float(np.mean((~satisfaction).astype(np.float64)))
                    for satisfaction in satisfaction_lists
                ],
                micro_value=float(
                    np.mean((~np.concatenate(satisfaction_lists)).astype(np.float64))
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="noe_mean_violation",
                per_example_values=[
                    float(np.mean(violation)) for violation in violation_amounts
                ],
                micro_value=float(np.mean(np.concatenate(violation_amounts))),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )
            _append_vector_metric(
                metric_rows=metric_rows,
                metric_summary=metric_summary,
                eligible_counts=eligible_counts,
                split=split,
                metric_name="noe_rmse_z",
                per_example_values=[
                    _rmse_z(prediction, target, sigma)
                    for prediction, target, sigma in zip(
                        predictions, targets, sigmas, strict=True
                    )
                ],
                micro_value=_rmse_z(
                    np.concatenate(predictions),
                    np.concatenate(targets),
                    np.concatenate(sigmas),
                ),
                counts=counts,
                report_micro_metrics=report_micro_metrics,
            )

    return {
        "metric_rows": metric_rows,
        "metric_summary": metric_summary,
        "eligible_counts": eligible_counts,
    }


def _append_family_balanced_ccc_metrics(
    metric_rows: list[dict[str, Any]],
    metric_summary: dict[str, float],
    eligible_counts: dict[str, dict[str, int]],
    split: str,
    channel_payloads: list[EvaluatedChannel],
    report_micro_metrics: bool,
) -> None:
    """Append family-balanced CCC metrics for multi-atom shift agreement."""
    per_example_ccc: list[float] = []
    per_example_loss: list[float] = []
    per_example_counts: list[int] = []
    all_family_ccc: list[float] = []
    all_family_loss: list[float] = []
    all_family_counts: list[int] = []
    for channel in channel_payloads:
        example_ccc: list[float] = []
        example_loss: list[float] = []
        example_count = 0
        for family_name in sorted(set(ATOM_FAMILY_MAP.values())):
            family_mask = np.asarray(
                [
                    _atom_family_from_target_id(target_id) == family_name
                    for target_id in channel.target_ids
                ],
                dtype=bool,
            )
            if np.count_nonzero(family_mask) < 2:
                continue
            ccc = _safe_ccc(
                channel.predictions[family_mask],
                channel.targets[family_mask],
            )
            loss = _ccc_loss_value(
                channel.predictions[family_mask],
                channel.targets[family_mask],
            )
            if not math.isfinite(ccc) or not math.isfinite(loss):
                continue
            count = int(np.count_nonzero(family_mask))
            example_ccc.append(ccc)
            example_loss.append(loss)
            example_count += count
            all_family_ccc.append(ccc)
            all_family_loss.append(loss)
            all_family_counts.append(count)
        if example_ccc:
            per_example_ccc.append(float(np.mean(example_ccc)))
            per_example_loss.append(float(np.mean(example_loss)))
            per_example_counts.append(example_count)
    if not per_example_ccc:
        return
    _append_vector_metric(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name="cs_family_ccc",
        per_example_values=per_example_ccc,
        micro_value=float(np.mean(all_family_ccc)),
        counts=per_example_counts,
        report_micro_metrics=report_micro_metrics,
    )
    _append_vector_metric(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name="cs_family_ccc_loss",
        per_example_values=per_example_loss,
        micro_value=float(np.mean(all_family_loss)),
        counts=per_example_counts,
        report_micro_metrics=report_micro_metrics,
    )


def _append_secondary_chemical_shift_metrics(
    metric_rows: list[dict[str, Any]],
    metric_summary: dict[str, float],
    eligible_counts: dict[str, dict[str, int]],
    split: str,
    channel_payloads: list[EvaluatedChannel],
    report_micro_metrics: bool,
) -> None:
    """Append secondary-shift CCC metrics after random-coil centering."""
    secondary_payloads = [
        _secondary_channel_arrays(channel) for channel in channel_payloads
    ]
    secondary_payloads = [
        payload for payload in secondary_payloads if payload["predictions"].size >= 2
    ]
    if not secondary_payloads:
        return
    predictions = [payload["predictions"] for payload in secondary_payloads]
    targets = [payload["targets"] for payload in secondary_payloads]
    counts = [int(payload["predictions"].size) for payload in secondary_payloads]
    _append_vector_metric(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name="cs_secondary_ccc",
        per_example_values=[
            _safe_ccc(prediction, target)
            for prediction, target in zip(predictions, targets, strict=True)
        ],
        micro_value=_safe_ccc(np.concatenate(predictions), np.concatenate(targets)),
        counts=counts,
        report_micro_metrics=report_micro_metrics,
    )
    _append_vector_metric(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name="cs_secondary_ccc_loss",
        per_example_values=[
            _ccc_loss_value(prediction, target)
            for prediction, target in zip(predictions, targets, strict=True)
        ],
        micro_value=_ccc_loss_value(
            np.concatenate(predictions),
            np.concatenate(targets),
        ),
        counts=counts,
        report_micro_metrics=report_micro_metrics,
    )

    family_example_values: list[float] = []
    family_example_losses: list[float] = []
    family_example_counts: list[int] = []
    all_family_values: list[float] = []
    all_family_losses: list[float] = []
    for payload in secondary_payloads:
        example_values: list[float] = []
        example_losses: list[float] = []
        example_count = 0
        for family_name in sorted(set(ATOM_FAMILY_MAP.values())):
            family_mask = payload["families"] == family_name
            if np.count_nonzero(family_mask) < 2:
                continue
            ccc = _safe_ccc(
                payload["predictions"][family_mask],
                payload["targets"][family_mask],
            )
            loss = _ccc_loss_value(
                payload["predictions"][family_mask],
                payload["targets"][family_mask],
            )
            if not math.isfinite(ccc) or not math.isfinite(loss):
                continue
            count = int(np.count_nonzero(family_mask))
            example_values.append(ccc)
            example_losses.append(loss)
            example_count += count
            all_family_values.append(ccc)
            all_family_losses.append(loss)
        if example_values:
            family_example_values.append(float(np.mean(example_values)))
            family_example_losses.append(float(np.mean(example_losses)))
            family_example_counts.append(example_count)
    if family_example_values:
        _append_vector_metric(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name="cs_secondary_family_ccc",
            per_example_values=family_example_values,
            micro_value=float(np.mean(all_family_values)),
            counts=family_example_counts,
            report_micro_metrics=report_micro_metrics,
        )
        _append_vector_metric(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name="cs_secondary_family_ccc_loss",
            per_example_values=family_example_losses,
            micro_value=float(np.mean(all_family_losses)),
            counts=family_example_counts,
            report_micro_metrics=report_micro_metrics,
        )

    for family_name in sorted(set(ATOM_FAMILY_MAP.values())):
        per_example_ccc: list[float] = []
        per_example_loss: list[float] = []
        per_example_counts: list[int] = []
        family_predictions: list[np.ndarray] = []
        family_targets: list[np.ndarray] = []
        for payload in secondary_payloads:
            family_mask = payload["families"] == family_name
            if np.count_nonzero(family_mask) < 2:
                continue
            prediction = payload["predictions"][family_mask]
            target = payload["targets"][family_mask]
            per_example_ccc.append(_safe_ccc(prediction, target))
            per_example_loss.append(_ccc_loss_value(prediction, target))
            per_example_counts.append(int(prediction.size))
            family_predictions.append(prediction)
            family_targets.append(target)
        if not family_predictions:
            continue
        _append_vector_metric(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name=f"cs_secondary_{family_name}_ccc",
            per_example_values=per_example_ccc,
            micro_value=_safe_ccc(
                np.concatenate(family_predictions),
                np.concatenate(family_targets),
            ),
            counts=per_example_counts,
            report_micro_metrics=report_micro_metrics,
        )
        _append_vector_metric(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name=f"cs_secondary_{family_name}_ccc_loss",
            per_example_values=per_example_loss,
            micro_value=_ccc_loss_value(
                np.concatenate(family_predictions),
                np.concatenate(family_targets),
            ),
            counts=per_example_counts,
            report_micro_metrics=report_micro_metrics,
        )


def _secondary_channel_arrays(channel: EvaluatedChannel) -> dict[str, np.ndarray]:
    """Return usable secondary-shift prediction/target arrays for one channel."""
    baselines = random_coil_baselines_for_target_ids(channel.target_ids)
    families = np.asarray(
        [_atom_family_from_target_id(target_id) for target_id in channel.target_ids],
        dtype=object,
    )
    usable = np.isfinite(baselines)
    return {
        "predictions": channel.predictions[usable] - baselines[usable],
        "targets": channel.targets[usable] - baselines[usable],
        "families": families[usable],
    }


def metric_rows_from_loss_summary(
    loss_summary: dict[str, float],
    split: str,
    example_count: int,
) -> list[dict[str, Any]]:
    """Convert averaged loss values into registry-style metric rows.

    Args:
        loss_summary: Averaged loss dictionary.
        split: Split label.
        example_count: Number of examples seen in the epoch.

    Returns:
        Loss rows compatible with the metric registry schema.
    """
    rows: list[dict[str, Any]] = []
    for metric_name, value in sorted(loss_summary.items()):
        rows.append(
            {
                "metric_name": metric_name,
                "split": split,
                "aggregation": "macro",
                "value": float(value),
                "eligible_examples": int(example_count),
                "eligible_measurements": 0,
                "tier": "loss",
            }
        )
    return rows


def select_checkpoint_metric(
    metric_summary: dict[str, float],
    split: str,
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Build the checkpoint-selection tuple for one epoch.

    Args:
        metric_summary: Flat metric summary for one epoch.
        split: Split used for checkpoint selection.
        config: Student-training configuration.

    Returns:
        Serializable checkpoint-selection record.
    """
    metric_names = [config.checkpoint_metric, *config.checkpoint_tie_breakers]
    values: list[float] = []
    resolved: dict[str, float] = {}
    for metric_name in metric_names:
        key = (
            str(metric_name)
            if str(metric_name).startswith(f"{split}_")
            else f"{split}_{metric_name}"
        )
        value = float(metric_summary.get(key, math.inf))
        values.append(_checkpoint_sort_value(metric_name, value))
        resolved[key] = value
    return {
        "split": split,
        "primary": config.checkpoint_metric,
        "tie_breakers": list(config.checkpoint_tie_breakers),
        "values": resolved,
        "comparison_key": values,
    }


def _checkpoint_sort_value(metric_name: str, value: float) -> float:
    """Return a lower-is-better value for checkpoint tuple sorting."""
    if not math.isfinite(value):
        return math.inf
    lowered = metric_name.lower()
    higher_is_better_tokens = (
        "ccc",
        "entropy",
        "ess",
        "pearson",
        "coverage",
        "satisfaction",
        "overlap",
        "win_rate",
    )
    lower_is_better_tokens = (
        "loss",
        "kl",
        "nll",
        "mae",
        "rmse",
        "violation",
        "error",
        "chi2",
    )
    if any(token in lowered for token in lower_is_better_tokens):
        return float(value)
    if any(token in lowered for token in higher_is_better_tokens):
        return -float(value)
    return float(value)


def collect_chemical_shift_prediction_rows(
    payloads: list[EvaluatedExample],
    split: str,
) -> list[dict[str, Any]]:
    """Collect per-target chemical-shift predictions for reporting.

    Args:
        payloads: Evaluated examples from one split.
        split: Split label.

    Returns:
        Flat prediction rows.
    """
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        channel = payload.channels.get("chemical_shifts")
        if channel is None:
            continue
        for index, (target_id, prediction, target, sigma) in enumerate(
            zip(
                channel.target_ids,
                channel.predictions,
                channel.targets,
                channel.sigmas,
                strict=True,
            )
        ):
            rows.append(
                {
                    "entity_uid": payload.entity_uid,
                    "bmrb_id": payload.bmrb_id,
                    "split": split,
                    "target_id": target_id,
                    "atom_family": _atom_family_from_target_id(target_id),
                    "predicted_value": float(prediction),
                    "target_value": float(target),
                    "target_sigma": float(sigma),
                    "random_coil_reference": random_coil_reference_for_target_id(
                        target_id
                    ),
                    "predicted_secondary_shift": float(
                        prediction - random_coil_reference_for_target_id(target_id)
                    ),
                    "target_secondary_shift": float(
                        target - random_coil_reference_for_target_id(target_id)
                    ),
                    "posterior_std": _posterior_stat_value(
                        channel.posterior_stats,
                        "std",
                        index,
                    ),
                    "posterior_q05": _posterior_stat_value(
                        channel.posterior_stats,
                        "q05",
                        index,
                    ),
                    "posterior_q10": _posterior_stat_value(
                        channel.posterior_stats,
                        "q10",
                        index,
                    ),
                    "posterior_q25": _posterior_stat_value(
                        channel.posterior_stats,
                        "q25",
                        index,
                    ),
                    "posterior_q50": _posterior_stat_value(
                        channel.posterior_stats,
                        "q50",
                        index,
                    ),
                    "posterior_q75": _posterior_stat_value(
                        channel.posterior_stats,
                        "q75",
                        index,
                    ),
                    "posterior_q90": _posterior_stat_value(
                        channel.posterior_stats,
                        "q90",
                        index,
                    ),
                    "posterior_q95": _posterior_stat_value(
                        channel.posterior_stats,
                        "q95",
                        index,
                    ),
                    "posterior_crps": _posterior_stat_value(
                        channel.posterior_stats,
                        "crps",
                        index,
                    ),
                    "posterior_nll": _posterior_stat_value(
                        channel.posterior_stats,
                        "nll",
                        index,
                    ),
                }
            )
    return rows


def _append_scalar_metric(
    metric_rows: list[dict[str, Any]],
    metric_summary: dict[str, float],
    eligible_counts: dict[str, dict[str, int]],
    split: str,
    metric_name: str,
    values: list[float],
    measurement_counts: list[int],
    report_micro_metrics: bool,
) -> None:
    """Append scalar macro and micro metrics to the registry payloads."""
    valid_pairs = [
        (float(value), int(count))
        for value, count in zip(values, measurement_counts, strict=True)
        if math.isfinite(float(value))
    ]
    if not valid_pairs:
        return
    macro_value = float(np.mean([value for value, _ in valid_pairs]))
    total_measurements = int(sum(count for _, count in valid_pairs))
    total_examples = len(valid_pairs)
    _append_metric_row(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name=metric_name,
        aggregation="macro",
        value=macro_value,
        eligible_examples=total_examples,
        eligible_measurements=total_measurements,
    )
    if report_micro_metrics:
        micro_value = float(
            np.average(
                np.asarray([value for value, _ in valid_pairs], dtype=float),
                weights=np.asarray([count for _, count in valid_pairs], dtype=float),
            )
        )
        _append_metric_row(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name=metric_name,
            aggregation="micro",
            value=micro_value,
            eligible_examples=total_examples,
            eligible_measurements=total_measurements,
        )


def _append_vector_metric(
    metric_rows: list[dict[str, Any]],
    metric_summary: dict[str, float],
    eligible_counts: dict[str, dict[str, int]],
    split: str,
    metric_name: str,
    per_example_values: list[float],
    micro_value: float,
    counts: list[int],
    report_micro_metrics: bool,
) -> None:
    """Append one measurement-derived metric to the registry payloads."""
    valid_pairs = [
        (float(value), int(count))
        for value, count in zip(per_example_values, counts, strict=True)
        if math.isfinite(float(value))
    ]
    if not valid_pairs:
        return
    total_measurements = int(sum(count for _, count in valid_pairs))
    total_examples = len(valid_pairs)
    macro_value = float(np.mean([value for value, _ in valid_pairs]))
    _append_metric_row(
        metric_rows=metric_rows,
        metric_summary=metric_summary,
        eligible_counts=eligible_counts,
        split=split,
        metric_name=metric_name,
        aggregation="macro",
        value=macro_value,
        eligible_examples=total_examples,
        eligible_measurements=total_measurements,
    )
    if report_micro_metrics and math.isfinite(float(micro_value)):
        _append_metric_row(
            metric_rows=metric_rows,
            metric_summary=metric_summary,
            eligible_counts=eligible_counts,
            split=split,
            metric_name=metric_name,
            aggregation="micro",
            value=float(micro_value),
            eligible_examples=total_examples,
            eligible_measurements=total_measurements,
        )


def _append_metric_row(
    metric_rows: list[dict[str, Any]],
    metric_summary: dict[str, float],
    eligible_counts: dict[str, dict[str, int]],
    split: str,
    metric_name: str,
    aggregation: str,
    value: float,
    eligible_examples: int,
    eligible_measurements: int,
) -> None:
    """Append one metric row and its flat summary entry."""
    row = {
        "metric_name": metric_name,
        "split": split,
        "aggregation": aggregation,
        "value": float(value),
        "eligible_examples": int(eligible_examples),
        "eligible_measurements": int(eligible_measurements),
        "tier": _metric_tier(metric_name),
    }
    metric_rows.append(row)
    metric_summary[f"{split}_{metric_name}_{aggregation}"] = float(value)
    eligible_counts[f"{split}_{metric_name}_{aggregation}"] = {
        "eligible_examples": int(eligible_examples),
        "eligible_measurements": int(eligible_measurements),
    }


def _evaluate_channel(
    channel_name: str,
    predicted_weights: np.ndarray,
    channel: PreparedObservableChannel,
    chemical_shift_residual_mu: np.ndarray | None = None,
    chemical_shift_moment_mu: np.ndarray | None = None,
) -> EvaluatedChannel | None:
    """Evaluate one prepared channel under predicted candidate weights."""
    weights = predicted_weights.astype(np.float64, copy=False)
    validity = channel.mask.astype(np.float64, copy=False)
    values = _channel_values_with_residual(
        channel_name=channel_name,
        channel=channel,
        chemical_shift_residual_mu=chemical_shift_residual_mu,
    )
    denominator = (validity * weights[None, :]).sum(axis=1)
    usable = denominator > 1e-8
    if not np.any(usable):
        return None
    if (
        channel_name == "chemical_shifts"
        and chemical_shift_moment_mu is not None
        and chemical_shift_moment_mu.shape[0] == channel.target_values.shape[0]
    ):
        predictions = chemical_shift_moment_mu.astype(np.float64, copy=False)
    else:
        weighted = (validity * values * weights[None, :]).sum(axis=1) / np.clip(
            denominator, 1e-8, None
        )
        if channel.transform == "inverse_sixth":
            predictions = np.clip(weighted, 1e-8, None) ** (-1.0 / 6.0)
        else:
            predictions = weighted
    lower_bounds = None
    if channel.lower_bounds is not None:
        lower_bounds = channel.lower_bounds.astype(np.float64, copy=False)[usable]
    upper_bounds = None
    if channel.upper_bounds is not None:
        upper_bounds = channel.upper_bounds.astype(np.float64, copy=False)[usable]
    posterior_stats: dict[str, np.ndarray] = {}
    if channel_name == "chemical_shifts":
        (
            posterior_std,
            posterior_q05,
            posterior_q10,
            posterior_q25,
            posterior_q50,
            posterior_q75,
            posterior_q90,
            posterior_q95,
            posterior_crps,
            posterior_nll,
        ) = _chemical_shift_posterior_stats(
            weights=weights,
            channel=channel,
            usable=usable,
            chemical_shift_residual_mu=chemical_shift_residual_mu,
        )
        posterior_stats = {
            "std": posterior_std,
            "q05": posterior_q05,
            "q10": posterior_q10,
            "q25": posterior_q25,
            "q50": posterior_q50,
            "q75": posterior_q75,
            "q90": posterior_q90,
            "q95": posterior_q95,
            "crps": posterior_crps,
            "nll": posterior_nll,
        }
    return EvaluatedChannel(
        name=channel_name,
        predictions=predictions[usable].astype(np.float64, copy=False),
        targets=channel.target_values.astype(np.float64, copy=False)[usable],
        sigmas=np.clip(
            channel.target_sigmas.astype(np.float64, copy=False)[usable],
            1e-6,
            None,
        ),
        target_ids=[
            target_id
            for target_id, is_usable in zip(channel.target_ids, usable, strict=True)
            if is_usable
        ],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        posterior_stats=posterior_stats,
    )


def _chemical_shift_posterior_stats(
    weights: np.ndarray,
    channel: PreparedObservableChannel,
    usable: np.ndarray,
    chemical_shift_residual_mu: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Return posterior summaries for one chemical-shift channel."""
    values = _channel_values_with_residual(
        channel_name="chemical_shifts",
        channel=channel,
        chemical_shift_residual_mu=chemical_shift_residual_mu,
    )
    masks = channel.mask.astype(bool, copy=False)
    targets = channel.target_values.astype(np.float64, copy=False)
    sigmas = np.clip(channel.target_sigmas.astype(np.float64, copy=False), 1e-6, None)

    posterior_std: list[float] = []
    posterior_q05: list[float] = []
    posterior_q10: list[float] = []
    posterior_q25: list[float] = []
    posterior_q50: list[float] = []
    posterior_q75: list[float] = []
    posterior_q90: list[float] = []
    posterior_q95: list[float] = []
    posterior_crps: list[float] = []
    posterior_nll: list[float] = []

    usable_indices = np.flatnonzero(usable)
    for index in usable_indices:
        value_row = values[index]
        mask_row = masks[index]
        sample_values = value_row[mask_row]
        sample_weights = weights[mask_row]
        sample_weights = sample_weights / np.clip(sample_weights.sum(), 1e-12, None)
        mean_value = float(np.sum(sample_weights * sample_values))
        variance = float(np.sum(sample_weights * np.square(sample_values - mean_value)))
        posterior_std.append(float(np.sqrt(max(variance, 0.0))))
        posterior_q05.append(_weighted_quantile(sample_values, sample_weights, 0.05))
        posterior_q10.append(_weighted_quantile(sample_values, sample_weights, 0.10))
        posterior_q25.append(_weighted_quantile(sample_values, sample_weights, 0.25))
        posterior_q50.append(_weighted_quantile(sample_values, sample_weights, 0.50))
        posterior_q75.append(_weighted_quantile(sample_values, sample_weights, 0.75))
        posterior_q90.append(_weighted_quantile(sample_values, sample_weights, 0.90))
        posterior_q95.append(_weighted_quantile(sample_values, sample_weights, 0.95))
        posterior_crps.append(
            _empirical_crps(
                sample_values=sample_values,
                sample_weights=sample_weights,
                target_value=float(targets[index]),
            )
        )
        posterior_nll.append(
            _mixture_nll(
                sample_values=sample_values,
                sample_weights=sample_weights,
                target_value=float(targets[index]),
                kernel_sigma=float(sigmas[index]),
            )
        )

    return tuple(
        np.asarray(values_list, dtype=np.float64)
        for values_list in [
            posterior_std,
            posterior_q05,
            posterior_q10,
            posterior_q25,
            posterior_q50,
            posterior_q75,
            posterior_q90,
            posterior_q95,
            posterior_crps,
            posterior_nll,
        ]
    )


def _channel_values_with_residual(
    channel_name: str,
    channel: PreparedObservableChannel,
    chemical_shift_residual_mu: np.ndarray | None,
) -> np.ndarray:
    """Return channel values with optional atom-family residual corrections."""
    values = channel.values.astype(np.float64, copy=False)
    if channel_name != "chemical_shifts" or chemical_shift_residual_mu is None:
        return values
    residual = np.asarray(chemical_shift_residual_mu, dtype=np.float64)
    if residual.shape == values.shape:
        return values + residual
    if residual.ndim != 2 or residual.shape[0] != values.shape[1]:
        return values
    adjusted = values.copy()
    for family_index, family_name in enumerate(["HN", "N", "CA", "CB", "C'"]):
        row_mask = np.asarray(
            [
                _atom_family_from_target_id(target_id) == family_name
                for target_id in channel.target_ids
            ],
            dtype=bool,
        )
        if np.any(row_mask) and family_index < residual.shape[1]:
            adjusted[row_mask] += residual[:, family_index][None, :]
    return adjusted


def _effective_support_size(weights: np.ndarray) -> float:
    """Return the effective support size of one probability vector."""
    return float(1.0 / np.sum(np.square(np.clip(weights, 1e-12, None))))


def _entropy(weights: np.ndarray) -> float:
    """Return entropy of one probability vector."""
    clipped = np.clip(weights, 1e-12, None)
    return float(-np.sum(clipped * np.log(clipped)))


def _topk_union_indices(
    teacher_weights: np.ndarray,
    predicted_weights: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return the union of teacher and predicted top-k support indices."""
    teacher_indices = np.argsort(teacher_weights)[-min(k, teacher_weights.size) :]
    predicted_indices = np.argsort(predicted_weights)[-min(k, predicted_weights.size) :]
    return np.unique(np.concatenate([teacher_indices, predicted_indices]))


def _safe_pearson(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Pearson correlation for two one-dimensional arrays."""
    if predictions.size < 2 or targets.size < 2:
        return math.nan
    if np.allclose(predictions, predictions[0]) or np.allclose(targets, targets[0]):
        return math.nan
    return float(np.corrcoef(predictions, targets)[0, 1])


def _safe_ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient for paired values."""
    if predictions.size < 2 or targets.size < 2:
        return math.nan
    predictions = predictions.astype(np.float64, copy=False)
    targets = targets.astype(np.float64, copy=False)
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    covariance = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denominator = pred_var + target_var + float(np.square(pred_mean - target_mean))
    if denominator <= 1e-12:
        return math.nan
    return float((2.0 * covariance) / denominator)


def _ccc_loss_value(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the minimization-friendly CCC loss, ``1 - CCC``."""
    ccc = _safe_ccc(predictions, targets)
    if not math.isfinite(ccc):
        return math.nan
    return float(1.0 - ccc)


def _mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the mean absolute error."""
    if predictions.size == 0:
        return math.nan
    return float(np.mean(np.abs(predictions - targets)))


def _rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the root mean squared error."""
    if predictions.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(predictions - targets))))


def _rmse_z(predictions: np.ndarray, targets: np.ndarray, sigmas: np.ndarray) -> float:
    """Return RMSE in units of experimental uncertainty."""
    if predictions.size == 0:
        return math.nan
    z_residual = (predictions - targets) / np.clip(sigmas, 1e-6, None)
    return float(np.sqrt(np.mean(np.square(z_residual))))


def _noe_satisfaction_mask(
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Return one boolean mask of NOE restraint satisfaction."""
    lower_ok = np.isnan(lower_bounds) | (predictions >= lower_bounds)
    upper_ok = np.isnan(upper_bounds) | (predictions <= upper_bounds)
    return lower_ok & upper_ok


def _noe_violation_amount(
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Return the per-restraint absolute NOE violation amount."""
    lower_violation = np.where(
        np.isnan(lower_bounds),
        0.0,
        np.clip(lower_bounds - predictions, 0.0, None),
    )
    upper_violation = np.where(
        np.isnan(upper_bounds),
        0.0,
        np.clip(predictions - upper_bounds, 0.0, None),
    )
    return lower_violation + upper_violation


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return one canonical atom-family label from a chemical-shift target ID."""
    if not target_id.startswith("cs:"):
        return None
    atom_name = target_id.rsplit(":", 1)[-1]
    return ATOM_FAMILY_MAP.get(atom_name)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return one weighted empirical quantile."""
    if values.size == 0:
        return math.nan
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order] / np.clip(weights[order].sum(), 1e-12, None)
    cdf = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cdf, quantile, side="left"))
    index = min(max(index, 0), ordered_values.size - 1)
    return float(ordered_values[index])


def _empirical_crps(
    sample_values: np.ndarray,
    sample_weights: np.ndarray,
    target_value: float,
) -> float:
    """Return the CRPS for one weighted empirical distribution."""
    absolute_target = np.sum(sample_weights * np.abs(sample_values - target_value))
    pairwise = np.abs(sample_values[:, None] - sample_values[None, :])
    pair_weights = sample_weights[:, None] * sample_weights[None, :]
    return float(absolute_target - 0.5 * np.sum(pair_weights * pairwise))


def _mixture_nll(
    sample_values: np.ndarray,
    sample_weights: np.ndarray,
    target_value: float,
    kernel_sigma: float,
) -> float:
    """Return the negative log-likelihood under a Gaussian-kernel mixture."""
    sigma = max(float(kernel_sigma), 1e-3)
    normalization = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * np.square((target_value - sample_values) / sigma)
    density = float(np.sum(sample_weights * normalization * np.exp(exponent)))
    return float(-np.log(max(density, 1e-12)))


def _posterior_stat_value(
    posterior_stats: dict[str, np.ndarray],
    name: str,
    index: int,
) -> float | None:
    """Return one scalar posterior stat value when present."""
    if name not in posterior_stats:
        return None
    values = posterior_stats[name]
    if index >= values.size:
        return None
    return float(values[index])


def _metric_tier(metric_name: str) -> str:
    """Return the metric tier label for one metric name."""
    if metric_name.startswith("teacher_"):
        return "density_fidelity"
    if (
        metric_name.startswith("cs_")
        or metric_name.startswith("j_")
        or metric_name in NOE_METRIC_NAMES
    ):
        return "observable_quality"
    return "loss"
