"""Objective helpers and support-ceiling utilities for AtypEmu training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def teacher_kl_loss(
    predicted_weights: "torch.Tensor",
    teacher_weights: "torch.Tensor",
) -> "torch.Tensor":
    """Return KL(teacher || predicted) over one candidate support."""
    clipped_teacher = teacher_weights.clamp_min(1e-8)
    clipped_predicted = predicted_weights.clamp_min(1e-8)
    return (clipped_teacher * (clipped_teacher.log() - clipped_predicted.log())).sum()


def support_entropy(weights: "torch.Tensor") -> "torch.Tensor":
    """Return entropy of one candidate-weight distribution."""
    clipped = weights.clamp_min(1e-8)
    return -(clipped * clipped.log()).sum()


def posterior_evidence_kl_loss(
    *,
    predicted_weights: "torch.Tensor",
    evidence_log_likelihood: "torch.Tensor | None",
    evidence_temperature: float,
) -> "torch.Tensor":
    """Distill the explicit evidence likelihood into posterior support weights."""
    import torch

    if evidence_log_likelihood is None:
        return predicted_weights.new_tensor(0.0)
    evidence = torch.nan_to_num(
        evidence_log_likelihood.to(predicted_weights.dtype),
        nan=-100.0,
        neginf=-100.0,
        posinf=10.0,
    )
    target_weights = torch.softmax(
        evidence / max(float(evidence_temperature), 1e-6),
        dim=0,
    ).detach()
    clipped_target = target_weights.clamp_min(1e-8)
    clipped_predicted = predicted_weights.clamp_min(1e-8)
    return torch.sum(
        clipped_target * (torch.log(clipped_target) - torch.log(clipped_predicted))
    )


def build_support_ceiling_report(
    learnability_report: dict[str, Any],
    *,
    task_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build a first-class candidate-support ceiling report from learnability rows."""
    val_metrics = learnability_report.get("variant_metrics", {}).get("val", {})
    entries = {
        "uniform": _metric_value(val_metrics, "uniform"),
        "teacher": _metric_value(val_metrics, "teacher"),
        "best_single_candidate_by_family_ccc": _metric_value(
            val_metrics,
            "best_family_ccc_onehot",
        ),
        "best_mixture_oracle": _metric_value(val_metrics, "family_weighted_oracle"),
        "per_family_oracle": _metric_value(val_metrics, "per_family_best_candidate"),
    }
    task_name = str(task_contract.get("task_name") or "")
    current = _metric_value(val_metrics, "current_model")
    entries["prior_model"] = current if task_name == "prior_sequence_only" else None
    entries["posterior_model"] = current if task_name.startswith("posterior_") else None
    model_value = entries["posterior_model"]
    if model_value is None:
        model_value = entries["prior_model"]
    support_ceiling = _max_finite(
        [
            entries["best_single_candidate_by_family_ccc"],
            entries["best_mixture_oracle"],
            entries["per_family_oracle"],
        ]
    )
    support_gap = (
        support_ceiling - model_value
        if support_ceiling is not None and model_value is not None
        else None
    )
    return {
        "status": "ok" if support_ceiling is not None else "missing_oracle_metrics",
        "metric": "cs_family_ccc_macro",
        "task_contract": dict(task_contract),
        "entries": entries,
        "support_ceiling": support_ceiling,
        "support_gap": support_gap,
        "diagnosis": _support_diagnosis(support_ceiling, support_gap),
    }


def _metric_value(
    variant_metrics: dict[str, Any],
    variant_name: str,
) -> float | None:
    """Return one JSON-safe support-ceiling metric value."""
    value = variant_metrics.get(variant_name, {}).get("cs_family_ccc_macro")
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _max_finite(values: list[float | None]) -> float | None:
    """Return the max finite value in one optional list."""
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _support_diagnosis(
    support_ceiling: float | None,
    support_gap: float | None,
) -> str:
    """Classify whether the current bottleneck is support or model gap."""
    if support_ceiling is None:
        return "insufficient_oracle_metrics"
    if support_ceiling < 0.35:
        return "candidate_or_observable_support_limited"
    if support_gap is not None and support_gap > 0.05:
        return "model_or_objective_gap"
    return "near_current_support_ceiling"
