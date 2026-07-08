"""Multi-observable approximation adapters for NMR posterior training."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from atypemu.training.data import PreparedObservableChannel, PreparedTeacherExample

if TYPE_CHECKING:
    import torch


ADAPTER_CHANNELS = ("chemical_shifts", "j_couplings", "noe_restraints")
DEFAULT_CHANNEL_WEIGHTS = {
    "chemical_shifts": 1.0,
    "j_couplings": 0.10,
    "noe_restraints": 0.10,
    "saxs": 0.0,
}
CHANNEL_ADAPTER_TYPES = {
    "chemical_shifts": "affine_scalar_ccc",
    "j_couplings": "robust_scalar_hz",
    "noe_restraints": "bounded_inverse_sixth",
    "saxs": "profile_scale_background",
}


def observable_adapter_oracle_kl_loss(
    *,
    predicted_weights: "torch.Tensor",
    channels: dict[str, PreparedObservableChannel],
    config: Any,
    allows_target_conditioning: bool,
) -> "torch.Tensor":
    """Distill a product-of-experts observable oracle into posterior weights.

    Args:
        predicted_weights: Model posterior weights on the candidate simplex.
        channels: Prepared observable channels for one example.
        config: Student-training configuration.
        allows_target_conditioning: Whether the active task may use observed
            target-derived evidence.

    Returns:
        KL(target oracle || predicted) for the joint observable oracle.

    Raises:
        ValueError: If this target-derived loss is enabled for a prior task.
    """
    import torch

    weight = float(getattr(config, "observable_oracle_distillation_weight", 0.0))
    if weight <= 0.0:
        return predicted_weights.new_tensor(0.0)
    if not allows_target_conditioning:
        raise ValueError(
            "observable_oracle_distillation_weight uses target-derived evidence "
            "and is only valid for posterior NMR tasks."
        )
    scores = joint_observable_scores_torch(
        channels=channels,
        predicted_weights=predicted_weights,
        channel_weights=dict(getattr(config, "observable_channel_weights", {})),
    )
    if scores is None:
        return predicted_weights.new_tensor(0.0)
    temperature = max(
        float(getattr(config, "observable_oracle_temperature", 0.5)), 1e-6
    )
    target_weights = torch.softmax(scores / temperature, dim=0).detach()
    clipped_target = target_weights.clamp_min(1e-8)
    clipped_predicted = predicted_weights.clamp_min(1e-8)
    return torch.sum(
        clipped_target * (torch.log(clipped_target) - torch.log(clipped_predicted))
    )


def joint_observable_scores_torch(
    *,
    channels: dict[str, PreparedObservableChannel],
    predicted_weights: "torch.Tensor",
    channel_weights: dict[str, float],
) -> "torch.Tensor | None":
    """Return standardized product-of-experts scores for all available channels."""
    import torch

    terms: list[torch.Tensor] = []
    reliabilities: list[float] = []
    for channel_name in ADAPTER_CHANNELS:
        channel = channels.get(channel_name)
        if channel is None:
            continue
        base_weight = float(
            channel_weights.get(
                channel_name,
                DEFAULT_CHANNEL_WEIGHTS.get(channel_name, 0.0),
            )
        )
        if base_weight <= 0.0:
            continue
        score = channel_candidate_scores_torch(channel_name, channel)
        if score is None:
            continue
        finite = torch.isfinite(score)
        if torch.count_nonzero(finite) < 2:
            continue
        standardized = score.clone()
        finite_values = standardized[finite]
        standardized[finite] = (finite_values - torch.mean(finite_values)) / torch.std(
            finite_values, unbiased=False
        ).clamp_min(1e-6)
        standardized = torch.nan_to_num(
            standardized,
            nan=-5.0,
            neginf=-5.0,
            posinf=5.0,
        )
        reliability = base_weight * channel_reliability_torch(channel)
        if reliability <= 0.0:
            continue
        terms.append(standardized)
        reliabilities.append(reliability)
    if not terms:
        return None
    weights = torch.tensor(
        reliabilities,
        dtype=predicted_weights.dtype,
        device=predicted_weights.device,
    ).clamp_min(0.0)
    stacked = torch.stack([term.to(predicted_weights.dtype) for term in terms])
    return torch.sum(stacked * weights.unsqueeze(1), dim=0) / weights.sum().clamp_min(
        1e-8
    )


def channel_candidate_scores_torch(
    channel_name: str,
    channel: PreparedObservableChannel,
) -> "torch.Tensor | None":
    """Return one higher-is-better candidate score vector for a channel."""
    import torch

    if channel.values.ndim != 2 or channel.values.shape[1] == 0:
        return None
    values = channel.values
    mask = channel.mask.bool()
    if torch.count_nonzero(mask) == 0:
        return None
    if channel_name == "noe_restraints":
        return _noe_candidate_scores_torch(channel)

    target = channel.target_values.to(values.dtype).unsqueeze(1)
    sigma = channel.target_sigmas.to(values.dtype).clamp_min(1e-6).unsqueeze(1)
    residual = (values - target) / sigma
    if channel_name == "j_couplings":
        delta = residual.new_tensor(4.0)
        absolute = torch.abs(residual)
        quadratic = torch.minimum(absolute, delta)
        loss = 0.5 * torch.square(quadratic) + delta * (absolute - quadratic)
    else:
        loss = torch.square(residual)
    counts = mask.sum(dim=0).to(values.dtype)
    scores = -(loss * mask.to(values.dtype)).sum(dim=0) / counts.clamp_min(1.0)
    return scores.masked_fill(counts <= 0, -100.0)


def channel_reliability_torch(channel: PreparedObservableChannel) -> float:
    """Return a simple coverage-based reliability for one sparse channel."""
    try:
        mask = channel.mask
        if mask.numel() == 0:
            return 0.0
        target_coverage = float(mask.any(dim=1).float().mean().detach().cpu().item())
        candidate_coverage = float(mask.any(dim=0).float().mean().detach().cpu().item())
        measurement_factor = min(math.log1p(int(mask.shape[0])) / math.log(32.0), 1.0)
    except Exception:
        return 0.0
    return max(target_coverage * candidate_coverage * measurement_factor, 0.0)


def _noe_candidate_scores_torch(channel: PreparedObservableChannel) -> "torch.Tensor":
    """Return NOE scores from distance-bound violation in physical distance space."""
    import torch

    values = channel.values.clamp_min(1e-12)
    distances = values.pow(-1.0 / 6.0)
    mask = channel.mask.bool()
    target = channel.target_values.to(values.dtype).unsqueeze(1)
    sigma = channel.target_sigmas.to(values.dtype).clamp_min(1e-6).unsqueeze(1)
    violation = torch.zeros_like(distances)
    if channel.lower_bounds is not None:
        lower = channel.lower_bounds.to(values.dtype).unsqueeze(1)
        lower_mask = torch.isfinite(lower)
        violation = violation + torch.where(
            lower_mask,
            torch.relu(lower - distances),
            torch.zeros_like(violation),
        )
    if channel.upper_bounds is not None:
        upper = channel.upper_bounds.to(values.dtype).unsqueeze(1)
        upper_mask = torch.isfinite(upper)
        violation = violation + torch.where(
            upper_mask,
            torch.relu(distances - upper),
            torch.zeros_like(violation),
        )
    no_bounds = violation == 0.0
    residual = torch.where(
        no_bounds,
        torch.abs(distances - target) / sigma,
        violation / sigma,
    )
    counts = mask.sum(dim=0).to(values.dtype)
    scores = -(residual * mask.to(values.dtype)).sum(dim=0) / counts.clamp_min(1.0)
    return scores.masked_fill(counts <= 0, -100.0)


def build_multi_observable_artifacts(
    *,
    output_dir: str | Path,
    train_examples: list[PreparedTeacherExample],
    val_examples: list[PreparedTeacherExample],
    train_epoch: dict[str, Any],
    val_epoch: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Write optional multi-observable oracle diagnostics for one best checkpoint."""
    output_dir_path = Path(output_dir)
    arrays_dir = output_dir_path / "reports" / "arrays"
    metrics_dir = output_dir_path / "reports" / "metrics"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    evaluated = {
        "train": _evaluated_by_uid(train_epoch.get("evaluated_examples", [])),
        "val": _evaluated_by_uid(val_epoch.get("evaluated_examples", [])),
    }
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    all_examples = {"train": train_examples, "val": val_examples}
    for split, examples in all_examples.items():
        for example in examples:
            predicted = None
            evaluated_example = evaluated.get(split, {}).get(example.entity_uid)
            if evaluated_example is not None:
                predicted = evaluated_example.predicted_weights
            example_rows, example_weight_rows = _example_adapter_rows(
                example=example,
                split=split,
                predicted_weights=predicted,
                config=config,
            )
            rows.extend(example_rows)
            weight_rows.extend(example_weight_rows)

    frame = pd.DataFrame(rows)
    weight_frame = pd.DataFrame(weight_rows)
    if not frame.empty:
        frame.to_parquet(arrays_dir / "observable_adapter_summary.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(
            arrays_dir / "observable_adapter_summary.parquet",
            index=False,
        )
    if not weight_frame.empty:
        weight_frame.to_parquet(
            arrays_dir / "observable_oracle_weights.parquet",
            index=False,
        )
    else:
        pd.DataFrame().to_parquet(
            arrays_dir / "observable_oracle_weights.parquet",
            index=False,
        )

    report = _adapter_report_from_rows(frame)
    support_report = _support_report_from_rows(frame)
    report["artifacts"] = {
        "summary_table": "reports/arrays/observable_adapter_summary.parquet",
        "oracle_weights": "reports/arrays/observable_oracle_weights.parquet",
        "support_report": "reports/metrics/multi_observable_support_ceiling_report.json",
    }
    _write_json(output_dir_path / "observable_adapter_report.json", report)
    _write_json(metrics_dir / "observable_adapter_report.json", report)
    _write_json(
        output_dir_path / "multi_observable_support_ceiling_report.json",
        support_report,
    )
    _write_json(
        metrics_dir / "multi_observable_support_ceiling_report.json",
        support_report,
    )
    return report


def _example_adapter_rows(
    *,
    example: PreparedTeacherExample,
    split: str,
    predicted_weights: np.ndarray | None,
    config: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return summary and oracle-weight rows for one example."""
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    candidate_count = int(len(example.teacher_weights))
    if candidate_count == 0:
        return rows, weight_rows
    uniform = np.full(candidate_count, 1.0 / candidate_count, dtype=np.float64)
    teacher = _normalize_weights(example.teacher_weights)
    model = None if predicted_weights is None else _normalize_weights(predicted_weights)
    oracle_weights_by_channel: dict[str, np.ndarray] = {}

    for channel_name in ADAPTER_CHANNELS:
        channel = example.channels.get(channel_name)
        if channel is None:
            continue
        scores = channel_candidate_scores_numpy(channel_name, channel)
        if scores is None:
            continue
        oracle = _softmax_np(
            scores,
            temperature=float(getattr(config, "observable_oracle_temperature", 0.5)),
        )
        oracle_weights_by_channel[channel_name] = oracle
        metric_name = primary_metric_name(channel_name)
        uniform_score = primary_metric(channel_name, channel, uniform)
        teacher_score = primary_metric(channel_name, channel, teacher)
        oracle_score = primary_metric(channel_name, channel, oracle)
        model_score = (
            None if model is None else primary_metric(channel_name, channel, model)
        )
        rows.append(
            {
                "entity_uid": example.entity_uid,
                "split": split,
                "observable": channel_name,
                "adapter_type": CHANNEL_ADAPTER_TYPES[channel_name],
                "metric_name": metric_name,
                "eligible_measurements": int(
                    np.count_nonzero(channel.mask.any(axis=1))
                ),
                "candidate_count": candidate_count,
                "uniform_score": _json_float(uniform_score),
                "teacher_score": _json_float(teacher_score),
                "model_score": _json_float(model_score),
                "oracle_score": _json_float(oracle_score),
                "oracle_minus_model": _json_float(
                    None if model_score is None else oracle_score - model_score
                ),
                "oracle_minus_uniform": _json_float(oracle_score - uniform_score),
                "oracle_method": _oracle_method(channel_name),
            }
        )
        for candidate_index, candidate_id in enumerate(example.candidate_ids):
            weight_rows.append(
                {
                    "entity_uid": example.entity_uid,
                    "split": split,
                    "observable": channel_name,
                    "candidate_index": candidate_index,
                    "candidate_id": candidate_id,
                    "teacher_weight": float(teacher[candidate_index]),
                    "model_weight": (
                        None if model is None else float(model[candidate_index])
                    ),
                    "oracle_weight": float(oracle[candidate_index]),
                    "oracle_score": _json_float(scores[candidate_index]),
                }
            )

    if len(oracle_weights_by_channel) >= 2:
        conflict = _mean_pairwise_js(list(oracle_weights_by_channel.values()))
        for row in rows[-len(oracle_weights_by_channel) :]:
            row["observable_conflict_score"] = _json_float(conflict)
    return rows, weight_rows


def channel_candidate_scores_numpy(
    channel_name: str,
    channel: PreparedObservableChannel,
) -> np.ndarray | None:
    """Return a candidate-level oracle score vector for one numpy channel."""
    values = np.asarray(channel.values, dtype=np.float64)
    mask = np.asarray(channel.mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] == 0 or not np.any(mask):
        return None
    if channel_name == "noe_restraints":
        return _noe_candidate_scores_numpy(channel)
    targets = np.asarray(channel.target_values, dtype=np.float64)[:, None]
    sigmas = np.clip(np.asarray(channel.target_sigmas, dtype=np.float64), 1e-6, None)[
        :, None
    ]
    residual = (values - targets) / sigmas
    if channel_name == "j_couplings":
        absolute = np.abs(residual)
        delta = 4.0
        quadratic = np.minimum(absolute, delta)
        loss = 0.5 * np.square(quadratic) + delta * (absolute - quadratic)
    else:
        loss = np.square(residual)
    counts = mask.sum(axis=0).astype(np.float64)
    scores = -np.sum(loss * mask, axis=0) / np.clip(counts, 1.0, None)
    scores[counts <= 0] = -100.0
    return scores


def _noe_candidate_scores_numpy(channel: PreparedObservableChannel) -> np.ndarray:
    """Return NOE candidate scores in distance-bound space."""
    values = np.clip(np.asarray(channel.values, dtype=np.float64), 1e-12, None)
    distances = np.power(values, -1.0 / 6.0)
    mask = np.asarray(channel.mask, dtype=bool)
    targets = np.asarray(channel.target_values, dtype=np.float64)[:, None]
    sigmas = np.clip(np.asarray(channel.target_sigmas, dtype=np.float64), 1e-6, None)[
        :, None
    ]
    violation = np.zeros_like(distances)
    if channel.lower_bounds is not None:
        lower = np.asarray(channel.lower_bounds, dtype=np.float64)[:, None]
        violation += np.where(
            np.isfinite(lower), np.maximum(lower - distances, 0.0), 0.0
        )
    if channel.upper_bounds is not None:
        upper = np.asarray(channel.upper_bounds, dtype=np.float64)[:, None]
        violation += np.where(
            np.isfinite(upper), np.maximum(distances - upper, 0.0), 0.0
        )
    residual = np.where(
        violation > 0.0,
        violation / sigmas,
        np.abs(distances - targets) / sigmas,
    )
    counts = mask.sum(axis=0).astype(np.float64)
    scores = -np.sum(residual * mask, axis=0) / np.clip(counts, 1.0, None)
    scores[counts <= 0] = -100.0
    return scores


def primary_metric_name(channel_name: str) -> str:
    """Return the canonical higher-is-better support metric for one channel."""
    return {
        "chemical_shifts": "cs_family_ccc",
        "j_couplings": "negative_j_rmse_hz",
        "noe_restraints": "noe_satisfaction_rate",
    }.get(channel_name, "support_score")


def primary_metric(
    channel_name: str,
    channel: PreparedObservableChannel,
    weights: np.ndarray,
) -> float:
    """Return the primary higher-is-better metric for one weighted channel."""
    predictions, usable = _channel_expectations_numpy(channel, weights)
    if not np.any(usable):
        return float("nan")
    targets = np.asarray(channel.target_values, dtype=np.float64)
    if channel_name == "chemical_shifts":
        return _chemical_shift_family_ccc(
            predictions[usable],
            targets[usable],
            [target_id for target_id, ok in zip(channel.target_ids, usable) if ok],
        )
    if channel_name == "j_couplings":
        return -float(
            np.sqrt(np.mean(np.square(predictions[usable] - targets[usable])))
        )
    if channel_name == "noe_restraints":
        lower = (
            np.full(predictions.shape, np.nan, dtype=np.float64)
            if channel.lower_bounds is None
            else np.asarray(channel.lower_bounds, dtype=np.float64)
        )
        upper = (
            np.full(predictions.shape, np.nan, dtype=np.float64)
            if channel.upper_bounds is None
            else np.asarray(channel.upper_bounds, dtype=np.float64)
        )
        satisfied = _noe_satisfaction(predictions[usable], lower[usable], upper[usable])
        return float(np.mean(satisfied.astype(np.float64)))
    return float("nan")


def _channel_expectations_numpy(
    channel: PreparedObservableChannel,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numpy weighted expectations matching the trainer semantics."""
    values = np.asarray(channel.values, dtype=np.float64)
    mask = np.asarray(channel.mask, dtype=bool)
    weights = _normalize_weights(weights)
    validity = mask.astype(np.float64)
    denominator = (validity * weights[None, :]).sum(axis=1)
    usable = denominator > 1e-8
    averaged = (validity * values * weights[None, :]).sum(axis=1) / np.clip(
        denominator,
        1e-8,
        None,
    )
    if channel.transform == "inverse_sixth":
        return np.power(np.clip(averaged, 1e-12, None), -1.0 / 6.0), usable
    return averaged, usable


def _chemical_shift_family_ccc(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_ids: list[str],
) -> float:
    """Return atom-family macro CCC for chemical shifts."""
    values = []
    for family in ["HN", "N", "CA", "CB", "C'"]:
        mask = np.asarray(
            [
                _atom_family_from_target_id(target_id) == family
                for target_id in target_ids
            ],
            dtype=bool,
        )
        if np.count_nonzero(mask) < 2:
            continue
        score = _safe_ccc(predictions[mask], targets[mask])
        if math.isfinite(score):
            values.append(score)
    if not values:
        return _safe_ccc(predictions, targets)
    return float(np.mean(values))


def _noe_satisfaction(
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Return NOE bound satisfaction booleans."""
    ok = np.ones(predictions.shape, dtype=bool)
    finite_lower = np.isfinite(lower_bounds)
    finite_upper = np.isfinite(upper_bounds)
    ok[finite_lower] &= predictions[finite_lower] >= lower_bounds[finite_lower]
    ok[finite_upper] &= predictions[finite_upper] <= upper_bounds[finite_upper]
    return ok


def _adapter_report_from_rows(frame: pd.DataFrame) -> dict[str, Any]:
    """Build the public JSON adapter report from summary rows."""
    report: dict[str, Any] = {
        "status": "ok" if not frame.empty else "no_observable_rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "channels": {},
        "saxs": {
            "status": "waiting_for_candidate_profile_matrix",
            "adapter_type": CHANNEL_ADAPTER_TYPES["saxs"],
        },
    }
    if frame.empty:
        return report
    for observable, subset in frame.groupby("observable", dropna=False):
        clean = subset.copy()
        report["channels"][str(observable)] = {
            "status": "ok",
            "adapter_type": str(clean["adapter_type"].iloc[0]),
            "metric_name": str(clean["metric_name"].iloc[0]),
            "examples": int(clean["entity_uid"].nunique()),
            "eligible_measurements": int(clean["eligible_measurements"].sum()),
            "uniform_score_macro": _mean_or_none(clean.get("uniform_score")),
            "model_score_macro": _mean_or_none(clean.get("model_score")),
            "oracle_score_macro": _mean_or_none(clean.get("oracle_score")),
            "oracle_minus_model_macro": _mean_or_none(clean.get("oracle_minus_model")),
            "oracle_minus_uniform_macro": _mean_or_none(
                clean.get("oracle_minus_uniform")
            ),
            "observable_conflict_score": _mean_or_none(
                clean.get("observable_conflict_score")
            ),
        }
    return report


def _support_report_from_rows(frame: pd.DataFrame) -> dict[str, Any]:
    """Build the multi-observable support-ceiling report."""
    report = {
        "status": "ok" if not frame.empty else "no_observable_rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_model_claim": "NMR-conditioned posterior reconstruction",
        "entries": {},
    }
    if frame.empty:
        return report
    for observable, subset in frame.groupby("observable", dropna=False):
        report["entries"][str(observable)] = {
            "adapter_type": str(subset["adapter_type"].iloc[0]),
            "metric_name": str(subset["metric_name"].iloc[0]),
            "support_ceiling": _mean_or_none(subset.get("oracle_score")),
            "model_value": _mean_or_none(subset.get("model_score")),
            "support_gap": _mean_or_none(subset.get("oracle_minus_model")),
            "diagnosis": _diagnose_gap(_mean_or_none(subset.get("oracle_minus_model"))),
        }
    report["entries"]["saxs"] = {
        "adapter_type": CHANNEL_ADAPTER_TYPES["saxs"],
        "status": "waiting_for_candidate_profile_matrix",
        "diagnosis": "candidate_profile_matrix_missing",
    }
    return report


def _evaluated_by_uid(evaluated_examples: list[Any]) -> dict[str, Any]:
    """Index evaluated examples by entity uid."""
    return {
        str(getattr(example, "entity_uid", "")): example
        for example in evaluated_examples
        if getattr(example, "entity_uid", None)
    }


def _softmax_np(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Return stable softmax weights from candidate scores."""
    finite_scores = np.nan_to_num(scores.astype(np.float64), nan=-100.0)
    finite_scores = finite_scores / max(float(temperature), 1e-6)
    finite_scores -= np.max(finite_scores)
    weights = np.exp(np.clip(finite_scores, -80.0, 80.0))
    return _normalize_weights(weights)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Return one finite simplex vector."""
    parsed = np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0)
    parsed = np.clip(parsed, 0.0, None)
    total = float(parsed.sum())
    if total <= 0.0:
        return np.full(parsed.shape, 1.0 / max(parsed.size, 1), dtype=np.float64)
    return parsed / total


def _mean_pairwise_js(weights: list[np.ndarray]) -> float:
    """Return mean pairwise Jensen-Shannon divergence between oracle weights."""
    values = []
    for left_index, left in enumerate(weights):
        for right in weights[left_index + 1 :]:
            values.append(
                _js_divergence(_normalize_weights(left), _normalize_weights(right))
            )
    return float(np.mean(values)) if values else 0.0


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jensen-Shannon divergence for two simplex vectors."""
    left = np.clip(left, 1e-12, None)
    right = np.clip(right, 1e-12, None)
    mixture = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * (np.log(left) - np.log(mixture)))
        + 0.5 * np.sum(right * (np.log(right) - np.log(mixture)))
    )


def _safe_ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's CCC with a finite fallback."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    finite = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[finite]
    targets = targets[finite]
    if predictions.size < 2:
        return float("nan")
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    covariance = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    if denominator <= 1e-12:
        return float("nan")
    return float(2.0 * covariance / denominator)


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return the canonical chemical-shift family for one target id."""
    if not str(target_id).startswith("cs:"):
        return None
    return {"H": "HN", "HN": "HN", "N": "N", "CA": "CA", "CB": "CB", "C": "C'"}.get(
        str(target_id).rsplit(":", 1)[-1]
    )


def _oracle_method(channel_name: str) -> str:
    """Return the approximation method name for one adapter."""
    return {
        "chemical_shifts": "affine_calibrated_softmax_ccc_proxy",
        "j_couplings": "student_t_huber_softmax_ls_proxy",
        "noe_restraints": "inverse_sixth_bound_hinge_softmax_proxy",
    }.get(channel_name, "unsupported")


def _mean_or_none(series: pd.Series | None) -> float | None:
    """Return a finite mean from one optional pandas series."""
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _diagnose_gap(gap: float | None) -> str:
    """Classify an oracle-model gap."""
    if gap is None:
        return "missing_model_or_oracle"
    if gap > 0.10:
        return "model_or_objective_gap"
    if gap > 0.03:
        return "moderate_gap"
    return "near_adapter_support_ceiling"


def _json_float(value: float | None) -> float | None:
    """Return a JSON-safe finite float."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact with stable formatting."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
