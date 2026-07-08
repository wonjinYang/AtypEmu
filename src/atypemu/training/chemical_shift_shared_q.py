"""One-shared-q posterior inference for chemical-shift predictor sidecars.

The fast CS predictor is only useful for the end-to-end goal if its conformer
sidecars can be reduced through the same scientific contract as UCBShift:
one entity/support-level posterior population shared by all atom families.
This module keeps that bridge small and auditable.  It consumes materialized
per-support CS predictions, infers one shared q per entity, writes posterior
mean rows, and evaluates those rows with the existing UCBShift-parity metrics.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from atypemu.training.cs_prediction_parity import (
    PREDICTION_OOD_FLAG_ALIASES,
    PREDICTION_OOD_SCORE_ALIASES,
    PREDICTION_UNCERTAINTY_ALIASES,
    PREDICTION_VALUE_ALIASES,
    ChemicalShiftParityConfig,
    canonical_atom_family,
    evaluate_chemical_shift_prediction_parity,
    load_table,
    normalize_target_frame,
    resolve_join_keys,
    write_parity_outputs,
    _boolean_series,
    _column_from_aliases,
    _normalize_common_columns,
)

if TYPE_CHECKING:
    import torch


SHARED_Q_POSTERIOR_KIND = "atypemu_cs_predictor_one_shared_q_posterior_v1"
SHARED_Q_INTERFACE_KIND = "one_shared_q_posterior_solver"
SUPPORT_ID_ALIASES: tuple[str, ...] = (
    "support_id",
    "conformer_id",
    "candidate_id",
    "sample_id",
    "structure_id",
    "pdb_id",
)
PRIOR_LOG_PROB_ALIASES: tuple[str, ...] = (
    "prior_log_prob",
    "log_prior",
    "bioemu_score_log_prob",
    "bioemu_log_prob",
    "log_prob",
)
VALID_SUPPORT_ALIASES: tuple[str, ...] = (
    "support_valid",
    "conformer_valid",
    "decoder_valid",
    "generated_support_valid",
    "valid",
    "validity_passed",
)
DEFAULT_FAMILY_PPM_SCALES: dict[str, float] = {
    "HN": 2.0,
    "N": 18.75,
    "CA": 16.25,
    "CB": 22.5,
    "C'": 10.0,
}


@dataclass(frozen=True, slots=True)
class ChemicalShiftSharedQConfig:
    """Configuration for tabular one-shared-q posterior inference."""

    energy_temperature: float = 1.0
    energy_scale: float = 1.0
    prior_log_prob_weight: float = 1.0
    target_sigma_floor: float = 1.0e-3
    family_ppm_scales: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_PPM_SCALES)
    )
    uncertainty_penalty_weight: float = 0.0
    uncertainty_threshold: float | None = None
    high_uncertainty_penalty: float = 64.0
    ood_score_threshold: float | None = None
    ood_penalty: float = 1.0e6
    incomplete_support_penalty: float = 64.0
    min_posterior_ess: float = 1.0
    min_posterior_entropy: float = 0.0
    max_posterior_top_mass: float = 1.0
    min_valid_mass: float = 1.0
    require_support_id: bool = True
    posterior_solver_kind: str = "energy_softmax_ccc_refined"
    ccc_refinement_steps: int = 64
    ccc_refinement_learning_rate: float = 5.0e-2
    ccc_refinement_residual_weight: float = 1.0
    ccc_refinement_energy_weight: float = 5.0e-2
    ccc_refinement_kl_weight: float = 1.0e-2
    ccc_refinement_floor: float = 0.95
    ccc_refinement_floor_weight: float = 2.0
    ccc_refinement_macro_floor: float = 0.95
    ccc_refinement_macro_floor_weight: float = 2.0
    ccc_refinement_macro_reward_weight: float = 0.25
    ccc_refinement_posterior_health_weight: float = 1.0
    ccc_refinement_floor_aggregation: str = "active_set"
    ccc_refinement_min_family_points: int = 2
    ccc_coordinate_refinement_passes: int = 2
    ccc_coordinate_candidate_limit: int = 16
    ccc_coordinate_mix_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    ccc_coordinate_target_fit_candidates: bool = True
    posterior_label: str = "cs_predictor_one_shared_q_posterior_mean"


@dataclass(frozen=True, slots=True)
class ChemicalShiftSharedQSweepConfig:
    """Configuration grid for selecting a one-shared-q posterior solver."""

    energy_temperatures: tuple[float, ...] = (0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    ccc_refinement_floor_weights: tuple[float, ...] = (0.0, 2.0, 5.0, 20.0)
    ccc_refinement_macro_floor_weights: tuple[float, ...] = (0.0, 2.0, 5.0, 20.0)
    ccc_refinement_macro_reward_weights: tuple[float, ...] = (0.0, 0.25, 1.0)
    ccc_refinement_posterior_health_weights: tuple[float, ...] = (0.0, 1.0, 5.0)
    ccc_refinement_energy_weights: tuple[float, ...] = (0.0, 0.05)
    ccc_refinement_kl_weights: tuple[float, ...] = (0.0, 0.01)
    ccc_refinement_residual_weights: tuple[float, ...] = (0.0, 1.0)
    max_configs: int = 256
    selection_split: str | None = None
    selection_metric: str = "family_gate_then_min_macro_ccc"


def infer_chemical_shift_shared_q_posterior(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    posterior_config: ChemicalShiftSharedQConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
) -> dict[str, Any]:
    """Infer one shared q and evaluate posterior means against target shifts."""

    cfg = posterior_config or ChemicalShiftSharedQConfig()
    prediction_frame = _normalize_support_prediction_frame(predictions, cfg)
    target_frame = normalize_target_frame(targets)
    join_keys = resolve_join_keys(
        prediction_frame=prediction_frame,
        target_frame=target_frame,
        requested_join_keys=None,
    )
    pairs = _build_support_prediction_target_pairs(
        prediction_frame=prediction_frame,
        target_frame=target_frame,
        join_keys=join_keys,
        splits=splits,
    )
    if pairs.empty:
        raise ValueError("No support prediction-target rows joined")

    support_weights, entity_health = _infer_support_weights(
        pairs=pairs,
        target_frame=target_frame,
        join_keys=join_keys,
        config=cfg,
    )
    posterior_predictions = _posterior_mean_predictions(
        pairs=pairs,
        support_weights=support_weights,
    )
    parity = evaluate_chemical_shift_prediction_parity(
        predictions=posterior_predictions,
        targets=targets,
        splits=splits,
        config=parity_config
        or ChemicalShiftParityConfig(
            gate_split="val",
            prediction_label=cfg.posterior_label,
            require_uncertainty=True,
            require_ood=True,
            uncertainty_threshold=cfg.uncertainty_threshold,
            ood_score_threshold=cfg.ood_score_threshold,
            max_high_uncertainty_positive_reward_fraction=0.0,
        ),
    )
    summary = _shared_q_summary(
        parity=parity,
        support_weights=support_weights,
        entity_health=entity_health,
        config=cfg,
        join_keys=join_keys,
    )
    return {
        "summary": summary,
        "posterior_predictions": posterior_predictions,
        "support_weights": support_weights,
        "support_pairs": pairs,
        "parity": parity,
        "posterior_config": cfg,
    }


def infer_shared_q_posterior_from_paths(
    *,
    predictions_path: str | Path,
    targets_path: str | Path,
    splits_path: str | Path | None,
    posterior_predictions_output: str | Path,
    support_weights_output: str | Path,
    summary_output: str | Path,
    metrics_output: str | Path | None = None,
    pairs_output: str | Path | None = None,
    posterior_config: ChemicalShiftSharedQConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
) -> dict[str, Any]:
    """CLI-oriented helper for one-shared-q posterior artifacts."""

    predictions = load_table(predictions_path)
    targets = load_table(targets_path)
    splits = (
        load_table(splits_path)
        if splits_path and Path(splits_path).exists()
        else None
    )
    result = infer_chemical_shift_shared_q_posterior(
        predictions=predictions,
        targets=targets,
        splits=splits,
        posterior_config=posterior_config,
        parity_config=parity_config,
    )
    _write_table(result["posterior_predictions"], Path(posterior_predictions_output))
    _write_table(result["support_weights"], Path(support_weights_output))
    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n"
    )
    write_parity_outputs(
        result=result["parity"],
        summary_path=summary_path.with_name(f"{summary_path.stem}.parity.json"),
        metrics_path=metrics_output,
        pairs_path=pairs_output,
    )
    return result["summary"]


def sweep_chemical_shift_shared_q_posterior(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    posterior_config: ChemicalShiftSharedQConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
    sweep_config: ChemicalShiftSharedQSweepConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded solver sweep and return the best one-shared-q posterior."""

    base_config = posterior_config or ChemicalShiftSharedQConfig()
    grid_config = sweep_config or ChemicalShiftSharedQSweepConfig()
    rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    best_index = -1
    for index, candidate_config in enumerate(
        _iter_shared_q_sweep_configs(base_config, grid_config)
    ):
        try:
            result = infer_chemical_shift_shared_q_posterior(
                predictions=predictions,
                targets=targets,
                splits=splits,
                posterior_config=candidate_config,
                parity_config=parity_config,
            )
        except Exception as exc:
            rows.append(
                {
                    "sweep_index": index,
                    "decision": "error",
                    "error": str(exc),
                    **_shared_q_config_row(candidate_config),
                }
            )
            continue
        row = _shared_q_sweep_result_row(
            index=index,
            result=result,
            config=candidate_config,
            sweep_config=grid_config,
        )
        rows.append(row)
        score = _shared_q_sweep_score(
            row,
            selection_metric=grid_config.selection_metric,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_result = result
            best_index = index
    if best_result is None:
        raise ValueError("No shared-q sweep configuration produced a valid result")

    sweep_results = pd.DataFrame(rows)
    best_summary = dict(best_result["summary"])
    best_summary["sweep"] = {
        "sweep_kind": "atypemu_cs_predictor_one_shared_q_solver_sweep_v1",
        "selection_metric": _normalized_sweep_selection_metric(
            grid_config.selection_metric
        ),
        "selection_split": _sweep_selection_split(best_result, grid_config),
        "evaluated_config_count": int(
            (sweep_results.get("decision", pd.Series(dtype=object)) != "error").sum()
        ),
        "error_config_count": int(
            (sweep_results.get("decision", pd.Series(dtype=object)) == "error").sum()
        ),
        "best_sweep_index": int(best_index),
        "best_score": list(best_score or ()),
        "best_config": _shared_q_config_row(best_result["posterior_config"]),
    }
    best_result = dict(best_result)
    best_result["summary"] = best_summary
    best_result["sweep_results"] = sweep_results
    best_result["sweep_summary"] = best_summary["sweep"]
    return best_result


def sweep_shared_q_posterior_from_paths(
    *,
    predictions_path: str | Path,
    targets_path: str | Path,
    splits_path: str | Path | None,
    posterior_predictions_output: str | Path,
    support_weights_output: str | Path,
    summary_output: str | Path,
    sweep_report_output: str | Path | None = None,
    metrics_output: str | Path | None = None,
    pairs_output: str | Path | None = None,
    posterior_config: ChemicalShiftSharedQConfig | None = None,
    parity_config: ChemicalShiftParityConfig | None = None,
    sweep_config: ChemicalShiftSharedQSweepConfig | None = None,
) -> dict[str, Any]:
    """CLI-oriented helper for selecting and writing the best shared-q posterior."""

    predictions = load_table(predictions_path)
    targets = load_table(targets_path)
    splits = (
        load_table(splits_path)
        if splits_path and Path(splits_path).exists()
        else None
    )
    result = sweep_chemical_shift_shared_q_posterior(
        predictions=predictions,
        targets=targets,
        splits=splits,
        posterior_config=posterior_config,
        parity_config=parity_config,
        sweep_config=sweep_config,
    )
    _write_table(result["posterior_predictions"], Path(posterior_predictions_output))
    _write_table(result["support_weights"], Path(support_weights_output))
    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n"
    )
    if sweep_report_output is not None:
        _write_table(result["sweep_results"], Path(sweep_report_output))
    write_parity_outputs(
        result=result["parity"],
        summary_path=summary_path.with_name(f"{summary_path.stem}.parity.json"),
        metrics_path=metrics_output,
        pairs_path=pairs_output,
    )
    return result["summary"]


def _iter_shared_q_sweep_configs(
    base_config: ChemicalShiftSharedQConfig,
    sweep_config: ChemicalShiftSharedQSweepConfig,
) -> list[ChemicalShiftSharedQConfig]:
    configs: list[ChemicalShiftSharedQConfig] = []
    seen: set[tuple[float, ...]] = set()
    max_configs = max(int(sweep_config.max_configs), 1)

    for temperature in _finite_grid(sweep_config.energy_temperatures):
        baseline = replace(
            base_config,
            energy_temperature=temperature,
            posterior_solver_kind="energy_softmax",
            ccc_refinement_steps=0,
        )
        _append_unique_shared_q_config(configs, seen, baseline, max_configs)

    product = itertools.product(
        _finite_grid(sweep_config.energy_temperatures),
        _finite_grid(sweep_config.ccc_refinement_floor_weights),
        _finite_grid(sweep_config.ccc_refinement_macro_floor_weights),
        _finite_grid(sweep_config.ccc_refinement_macro_reward_weights),
        _finite_grid(sweep_config.ccc_refinement_posterior_health_weights),
        _finite_grid(sweep_config.ccc_refinement_energy_weights),
        _finite_grid(sweep_config.ccc_refinement_kl_weights),
        _finite_grid(sweep_config.ccc_refinement_residual_weights),
    )
    for (
        temperature,
        floor_weight,
        macro_floor_weight,
        macro_reward_weight,
        posterior_health_weight,
        energy_weight,
        kl_weight,
        residual_weight,
    ) in product:
        candidate = replace(
            base_config,
            energy_temperature=temperature,
            posterior_solver_kind="energy_softmax_ccc_refined",
            ccc_refinement_floor_weight=floor_weight,
            ccc_refinement_macro_floor_weight=macro_floor_weight,
            ccc_refinement_macro_reward_weight=macro_reward_weight,
            ccc_refinement_posterior_health_weight=posterior_health_weight,
            ccc_refinement_energy_weight=energy_weight,
            ccc_refinement_kl_weight=kl_weight,
            ccc_refinement_residual_weight=residual_weight,
        )
        _append_unique_shared_q_config(configs, seen, candidate, max_configs)
        if len(configs) >= max_configs:
            break
    return configs


def _finite_grid(values: Sequence[float]) -> tuple[float, ...]:
    out = []
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric):
            out.append(numeric)
    return tuple(out)


def _append_unique_shared_q_config(
    configs: list[ChemicalShiftSharedQConfig],
    seen: set[tuple[float, ...]],
    config: ChemicalShiftSharedQConfig,
    max_configs: int,
) -> None:
    if len(configs) >= max_configs:
        return
    key = (
        round(float(config.energy_temperature), 12),
        float(config.posterior_solver_kind == "energy_softmax_ccc_refined"),
        round(float(config.ccc_refinement_floor), 12),
        round(float(config.ccc_refinement_floor_weight), 12),
        round(float(config.ccc_refinement_macro_floor), 12),
        round(float(config.ccc_refinement_macro_floor_weight), 12),
        round(float(config.ccc_refinement_macro_reward_weight), 12),
        round(float(config.ccc_refinement_posterior_health_weight), 12),
        round(float(config.ccc_refinement_energy_weight), 12),
        round(float(config.ccc_refinement_kl_weight), 12),
        round(float(config.ccc_refinement_residual_weight), 12),
        float(_floor_aggregation_kind(config) == "active_set"),
        float(bool(config.ccc_coordinate_target_fit_candidates)),
    )
    if key in seen:
        return
    seen.add(key)
    configs.append(config)


def _shared_q_sweep_result_row(
    *,
    index: int,
    result: dict[str, Any],
    config: ChemicalShiftSharedQConfig,
    sweep_config: ChemicalShiftSharedQSweepConfig,
) -> dict[str, Any]:
    summary = result["summary"]
    split_name = _sweep_selection_split(result, sweep_config)
    split_summary = (
        summary.get("parity_summary", {})
        .get("split_summaries", {})
        .get(split_name, {})
    )
    ccc_refinement = summary.get("ccc_refinement", {})
    failing = split_summary.get("failing_required_families", [])
    missing = split_summary.get("missing_required_families", [])
    macro_ccc = _safe_float(split_summary.get("family_macro_ccc"))
    macro_floor = float(config.ccc_refinement_macro_floor)
    macro_gap = (
        max(0.0, macro_floor - macro_ccc) if math.isfinite(macro_ccc) else math.nan
    )
    return {
        "sweep_index": int(index),
        "decision": str(summary.get("decision")),
        "selection_split": split_name,
        "family_min_ccc": _safe_float(split_summary.get("family_min_ccc")),
        "family_macro_ccc": macro_ccc,
        "passes_family_ccc_gate": bool(split_summary.get("passes_family_ccc_gate")),
        "passes_macro_family_ccc_gate": bool(
            math.isfinite(macro_ccc) and macro_ccc + 1.0e-12 >= macro_floor
        ),
        "family_macro_ccc_gap_to_floor": macro_gap,
        "posterior_health_gate_passed": bool(
            summary.get("posterior_health_gate_passed")
        ),
        "predictor_reliability_gate_passed": bool(
            summary.get("predictor_reliability_gate_passed")
        ),
        "missing_required_families": ",".join(map(str, missing)),
        "failing_required_families": ",".join(map(str, failing)),
        "support_count": int(summary.get("support_count") or 0),
        "entity_count": int(summary.get("entity_count") or 0),
        "ccc_refinement_max_abs_weight_delta": _safe_float(
            ccc_refinement.get("max_abs_weight_delta")
        ),
        "ccc_refinement_mean_abs_weight_delta": _safe_float(
            ccc_refinement.get("mean_abs_weight_delta")
        ),
        "ccc_refinement_best_objective": _safe_float(
            ccc_refinement.get("best_objective")
        ),
        "ccc_coordinate_refinement_updates": int(
            ccc_refinement.get("coordinate_update_count") or 0
        ),
        **_shared_q_config_row(config),
    }


def _sweep_selection_split(
    result: dict[str, Any],
    sweep_config: ChemicalShiftSharedQSweepConfig,
) -> str:
    if sweep_config.selection_split:
        return str(sweep_config.selection_split)
    summary = result.get("summary", {})
    gate_split = summary.get("gate_split")
    return str(gate_split or "val")


def _shared_q_sweep_score(
    row: dict[str, Any],
    *,
    selection_metric: str | None = "family_gate_then_min_macro_ccc",
) -> tuple[float, ...]:
    decision_bonus = 1.0 if str(row.get("decision")) == "pass" else 0.0
    family_gate = 1.0 if bool(row.get("passes_family_ccc_gate")) else 0.0
    macro_gate = 1.0 if bool(row.get("passes_macro_family_ccc_gate")) else 0.0
    health = 1.0 if bool(row.get("posterior_health_gate_passed")) else 0.0
    reliability = 1.0 if bool(row.get("predictor_reliability_gate_passed")) else 0.0
    min_ccc = _score_float(row.get("family_min_ccc"))
    macro_ccc = _score_float(row.get("family_macro_ccc"))
    objective = _score_float(row.get("ccc_refinement_best_objective"))
    support_count = _score_float(row.get("support_count"))
    metric = _normalized_sweep_selection_metric(selection_metric)
    common_prefix = (decision_bonus, family_gate, macro_gate, health, reliability)
    if metric in {"family_gate_then_min_macro_ccc", "family_min_ccc_then_macro"}:
        return (*common_prefix, min_ccc, macro_ccc, -objective, support_count)
    if metric == "family_gate_then_macro_ccc":
        return (*common_prefix, macro_ccc, min_ccc, -objective, support_count)
    if metric == "objective_then_macro_ccc":
        return (*common_prefix, -objective, macro_ccc, min_ccc, support_count)
    raise ValueError(f"Unsupported shared-q sweep selection_metric={selection_metric!r}")


def _normalized_sweep_selection_metric(value: str | None) -> str:
    metric = str(value or "family_gate_then_min_macro_ccc").strip().lower()
    metric = metric.replace("-", "_")
    aliases = {
        "goal": "family_gate_then_min_macro_ccc",
        "goal_safe": "family_gate_then_min_macro_ccc",
        "family_gate_min": "family_gate_then_min_macro_ccc",
        "family_gate_then_min": "family_gate_then_min_macro_ccc",
        "family_gate_then_min_macro": "family_gate_then_min_macro_ccc",
        "family_gate_then_min_ccc": "family_gate_then_min_macro_ccc",
        "family_gate_then_min_ccc_then_macro_ccc": (
            "family_gate_then_min_macro_ccc"
        ),
        "family_min": "family_min_ccc_then_macro",
        "family_min_then_macro": "family_min_ccc_then_macro",
        "family_min_ccc_then_macro_ccc": "family_min_ccc_then_macro",
        "macro": "family_gate_then_macro_ccc",
        "macro_then_min": "family_gate_then_macro_ccc",
        "macro_family_ccc_then_min": "family_gate_then_macro_ccc",
        "family_macro_ccc_then_min": "family_gate_then_macro_ccc",
        "family_gate_macro": "family_gate_then_macro_ccc",
        "objective": "objective_then_macro_ccc",
        "objective_then_macro": "objective_then_macro_ccc",
    }
    return aliases.get(metric, metric)


def _shared_q_config_row(config: ChemicalShiftSharedQConfig) -> dict[str, Any]:
    return {
        "posterior_solver_kind": str(config.posterior_solver_kind),
        "energy_temperature": float(config.energy_temperature),
        "energy_scale": float(config.energy_scale),
        "prior_log_prob_weight": float(config.prior_log_prob_weight),
        "ccc_refinement_steps": int(config.ccc_refinement_steps),
        "ccc_refinement_floor": float(config.ccc_refinement_floor),
        "ccc_refinement_floor_weight": float(config.ccc_refinement_floor_weight),
        "ccc_refinement_macro_floor": float(config.ccc_refinement_macro_floor),
        "ccc_refinement_macro_floor_weight": float(
            config.ccc_refinement_macro_floor_weight
        ),
        "ccc_refinement_macro_reward_weight": float(
            config.ccc_refinement_macro_reward_weight
        ),
        "ccc_refinement_posterior_health_weight": float(
            config.ccc_refinement_posterior_health_weight
        ),
        "ccc_refinement_floor_aggregation": _floor_aggregation_kind(config),
        "ccc_refinement_energy_weight": float(config.ccc_refinement_energy_weight),
        "ccc_refinement_kl_weight": float(config.ccc_refinement_kl_weight),
        "ccc_refinement_residual_weight": float(
            config.ccc_refinement_residual_weight
        ),
        "ccc_coordinate_refinement_passes": int(
            config.ccc_coordinate_refinement_passes
        ),
        "ccc_coordinate_candidate_limit": int(
            config.ccc_coordinate_candidate_limit
        ),
        "ccc_coordinate_target_fit_candidates": bool(
            config.ccc_coordinate_target_fit_candidates
        ),
    }


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _score_float(value: Any) -> float:
    numeric = _safe_float(value)
    return numeric if math.isfinite(numeric) else -1.0e12


def _normalize_support_prediction_frame(
    frame: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> pd.DataFrame:
    normalized = _normalize_common_columns(frame)
    value_series, _ = _column_from_aliases(
        frame,
        PREDICTION_VALUE_ALIASES,
        required=True,
        label="prediction value",
    )
    normalized["predicted_value"] = pd.to_numeric(value_series, errors="coerce")
    support_id, support_column = _column_from_aliases(
        frame,
        SUPPORT_ID_ALIASES,
        required=False,
        label="support id",
    )
    if support_column is None:
        if config.require_support_id:
            raise ValueError(
                "Prediction table must include support_id/conformer_id for "
                "one-shared-q posterior inference"
            )
        support_id = pd.Series("support:0", index=frame.index)
    normalized["support_id"] = support_id.astype("string").str.strip()
    prior_log_prob, _ = _column_from_aliases(
        frame,
        PRIOR_LOG_PROB_ALIASES,
        required=False,
        label="prior log probability",
    )
    normalized["prior_log_prob"] = pd.to_numeric(prior_log_prob, errors="coerce")
    valid, valid_column = _column_from_aliases(
        frame,
        VALID_SUPPORT_ALIASES,
        required=False,
        label="support validity",
    )
    normalized["support_valid"] = (
        _boolean_series(valid).fillna(True)
        if valid_column is not None
        else pd.Series(True, index=frame.index)
    )
    uncertainty, _ = _column_from_aliases(
        frame,
        PREDICTION_UNCERTAINTY_ALIASES,
        required=False,
        label="prediction uncertainty",
    )
    normalized["prediction_uncertainty"] = pd.to_numeric(
        uncertainty,
        errors="coerce",
    )
    ood_score, _ = _column_from_aliases(
        frame,
        PREDICTION_OOD_SCORE_ALIASES,
        required=False,
        label="prediction OOD score",
    )
    normalized["prediction_ood_score"] = pd.to_numeric(ood_score, errors="coerce")
    ood_flag, _ = _column_from_aliases(
        frame,
        PREDICTION_OOD_FLAG_ALIASES,
        required=False,
        label="prediction OOD flag",
    )
    normalized["prediction_ood_flag"] = _boolean_series(ood_flag).fillna(False)
    normalized = normalized.dropna(
        subset=["entity_uid", "atom_family", "support_id", "predicted_value"]
    )
    return normalized.reset_index(drop=True)


def _build_support_prediction_target_pairs(
    *,
    prediction_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    join_keys: Sequence[str],
    splits: pd.DataFrame | None,
) -> pd.DataFrame:
    joined = prediction_frame.merge(
        target_frame,
        on=list(join_keys),
        how="inner",
        suffixes=("_prediction", "_target"),
    )
    result = pd.DataFrame(index=joined.index)
    result["entity_uid"] = joined["entity_uid"].astype(str)
    result["support_id"] = joined["support_id"].astype(str)
    for column in ["target_id", "seq_id", "comp_id", "atom_id", "atom_family"]:
        result[column] = _coalesce_joined_column(joined, column, join_keys)
    result["atom_family"] = result["atom_family"].map(canonical_atom_family)
    result["predicted_value"] = pd.to_numeric(joined["predicted_value"], errors="coerce")
    result["target_value"] = pd.to_numeric(joined["target_value"], errors="coerce")
    result["target_sigma"] = pd.to_numeric(
        joined.get("target_sigma", math.nan),
        errors="coerce",
    )
    result["prior_log_prob"] = pd.to_numeric(
        joined.get("prior_log_prob", 0.0),
        errors="coerce",
    )
    result["support_valid"] = _boolean_series(
        joined.get("support_valid", pd.Series(True, index=joined.index))
    ).fillna(True)
    result["prediction_uncertainty"] = pd.to_numeric(
        joined.get("prediction_uncertainty", math.nan),
        errors="coerce",
    )
    result["prediction_ood_score"] = pd.to_numeric(
        joined.get("prediction_ood_score", math.nan),
        errors="coerce",
    )
    result["prediction_ood_flag"] = _boolean_series(
        joined.get("prediction_ood_flag", pd.Series(False, index=joined.index))
    ).fillna(False)
    split = _coalesce_joined_column(joined, "split", join_keys)
    result["split"] = split.astype("string").str.strip()
    if splits is not None and "split" in splits.columns:
        split_frame = pd.DataFrame(
            {
                "entity_uid": _entity_uid_series(splits),
                "split_from_file": splits["split"].astype("string").str.strip(),
            }
        ).drop_duplicates("entity_uid")
        result = result.merge(split_frame, on="entity_uid", how="left")
        result["split"] = result["split"].where(
            result["split"].notna(),
            result["split_from_file"],
        )
        result = result.drop(columns=["split_from_file"])
    result["split"] = result["split"].fillna("all")
    finite = (
        np.isfinite(result["predicted_value"].to_numpy(dtype=float))
        & np.isfinite(result["target_value"].to_numpy(dtype=float))
    )
    return result.loc[finite].reset_index(drop=True)


def _infer_support_weights(
    *,
    pairs: pd.DataFrame,
    target_frame: pd.DataFrame,
    join_keys: Sequence[str],
    config: ChemicalShiftSharedQConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_counts = _expected_target_counts(target_frame, join_keys)
    rows: list[dict[str, Any]] = []
    for entity_uid, entity_pairs in pairs.groupby("entity_uid", sort=True):
        support_rows: list[dict[str, Any]] = []
        expected_count = int(expected_counts.get(str(entity_uid), 0))
        for support_id, support_pairs in entity_pairs.groupby("support_id", sort=True):
            energy_parts = _family_balanced_energy(support_pairs, config)
            row_count = int(len(support_pairs))
            coverage = (
                float(row_count / expected_count)
                if expected_count > 0
                else 1.0
            )
            incomplete_penalty = (
                max(0.0, 1.0 - coverage) * float(config.incomplete_support_penalty)
            )
            uncertainty_penalty = _uncertainty_penalty(support_pairs, config)
            ood_penalty = _ood_penalty(support_pairs, config)
            support_valid = bool(support_pairs["support_valid"].fillna(True).all())
            energy = (
                float(config.energy_scale) * energy_parts["energy"]
                + incomplete_penalty
                + uncertainty_penalty
                + ood_penalty
            )
            prior = _finite_mean(support_pairs["prior_log_prob"].to_numpy(float))
            support_rows.append(
                {
                    "entity_uid": str(entity_uid),
                    "support_id": str(support_id),
                    "shared_q_energy": energy,
                    "family_balanced_residual_energy": energy_parts["energy"],
                    "family_residual_energy_min": energy_parts[
                        "family_residual_energy_min"
                    ],
                    "family_residual_energy_by_family": json.dumps(
                        energy_parts["family_residual_energy_by_family"],
                        sort_keys=True,
                    ),
                    "prior_log_prob": 0.0 if not math.isfinite(prior) else prior,
                    "support_valid": support_valid,
                    "support_row_count": row_count,
                    "expected_target_row_count": expected_count,
                    "coverage_fraction": coverage,
                    "uncertainty_penalty": uncertainty_penalty,
                    "ood_penalty": ood_penalty,
                    "incomplete_support_penalty": incomplete_penalty,
                    "atom_family_count": energy_parts["family_count"],
                }
            )
        support_table = pd.DataFrame(support_rows)
        if support_table.empty:
            continue
        q = _posterior_softmax(
            support_table=support_table,
            config=config,
        )
        support_table["posterior_weight"] = q
        support_table["posterior_weight_initial"] = q
        rows.extend(support_table.to_dict("records"))
    support_weights = pd.DataFrame(rows)
    if support_weights.empty:
        return support_weights, pd.DataFrame()
    support_weights = _refine_support_weights_for_family_ccc(
        pairs=pairs,
        support_weights=support_weights,
        config=config,
    )
    support_weights = _rank_support_weights(support_weights)
    health_rows = [
        _entity_health_row(str(entity_uid), entity_supports, config)
        for entity_uid, entity_supports in support_weights.groupby(
            "entity_uid",
            sort=True,
        )
    ]
    return support_weights, pd.DataFrame(health_rows)


def _posterior_mean_predictions(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
) -> pd.DataFrame:
    weighted = pairs.merge(
        support_weights[["entity_uid", "support_id", "posterior_weight"]],
        on=["entity_uid", "support_id"],
        how="inner",
    )
    weighted["weighted_prediction"] = (
        weighted["predicted_value"] * weighted["posterior_weight"]
    )
    weighted["weighted_second_moment"] = (
        weighted["predicted_value"] ** 2 * weighted["posterior_weight"]
    )
    uncertainty = weighted["prediction_uncertainty"].fillna(0.0)
    weighted["weighted_uncertainty2"] = (
        uncertainty ** 2 * weighted["posterior_weight"]
    )
    key_columns = ["entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "atom_family"]
    rows: list[dict[str, Any]] = []
    for key, group in weighted.groupby(key_columns, dropna=False, sort=True):
        weight_sum = float(group["posterior_weight"].sum())
        if weight_sum <= 0.0:
            continue
        mean = float(group["weighted_prediction"].sum() / weight_sum)
        second = float(group["weighted_second_moment"].sum() / weight_sum)
        uncertainty2 = float(group["weighted_uncertainty2"].sum() / weight_sum)
        variance = max(second - mean * mean, 0.0) + max(uncertainty2, 0.0)
        posterior_ood_score = float(
            (
                group["prediction_ood_score"].fillna(0.0)
                * group["posterior_weight"]
            ).sum()
            / weight_sum
        )
        ood_mass = float(
            group.loc[
                group["prediction_ood_flag"].fillna(False),
                "posterior_weight",
            ].sum()
        )
        row = dict(zip(key_columns, key))
        row.update(
            {
                "predicted_value": mean,
                "posterior_mean": mean,
                "prediction": mean,
                "prediction_uncertainty": math.sqrt(variance),
                "prediction_ood_score": posterior_ood_score,
                "prediction_ood_flag": bool(ood_mass > 1.0e-6),
                "prediction_reward": weight_sum,
                "prediction_reliability_gate_passed": bool(
                    ood_mass <= 1.0e-6
                ),
                "posterior_ood_mass": ood_mass,
                "posterior_weight_observed_mass": weight_sum,
                "shared_q_interface_kind": SHARED_Q_INTERFACE_KIND,
                "posterior_mean_source": SHARED_Q_POSTERIOR_KIND,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_support_weights(support_weights: pd.DataFrame) -> pd.DataFrame:
    ranked = support_weights.copy()
    ranked["posterior_weight_rank"] = (
        ranked.groupby("entity_uid")["posterior_weight"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return ranked


def _refine_support_weights_for_family_ccc(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> pd.DataFrame:
    support = support_weights.reset_index(drop=True).copy()
    if not _ccc_refinement_enabled(config):
        support["posterior_solver_kind"] = "energy_softmax"
        support["posterior_weight_refinement_delta"] = 0.0
        support["ccc_refinement_objective"] = math.nan
        return support
    if support.empty or int(support["entity_uid"].nunique()) <= 0:
        return support

    best = support.copy()
    best_objective = _support_weight_refinement_objective(
        pairs=pairs,
        support_weights=best,
        config=config,
    )
    for candidate in _ccc_refinement_candidates(support, config):
        objective = _support_weight_refinement_objective(
            pairs=pairs,
            support_weights=candidate,
            config=config,
        )
        if _objective_is_better(objective, best_objective):
            best = candidate
            best_objective = objective

    best, best_objective, coordinate_updates = _coordinate_refine_support_weights(
        pairs=pairs,
        support_weights=best,
        initial_objective=best_objective,
        config=config,
    )
    best, best_objective, gradient_updates = _gradient_refine_support_weights(
        pairs=pairs,
        support_weights=best,
        initial_objective=best_objective,
        config=config,
    )
    best["posterior_weight_refinement_delta"] = (
        best["posterior_weight"] - best["posterior_weight_initial"]
    )
    best["posterior_solver_kind"] = str(config.posterior_solver_kind)
    best["ccc_refinement_objective"] = best_objective
    best["ccc_coordinate_refinement_updates"] = int(coordinate_updates)
    best["ccc_gradient_refinement_updates"] = int(gradient_updates)
    return best


def _ccc_refinement_enabled(config: ChemicalShiftSharedQConfig) -> bool:
    solver = str(config.posterior_solver_kind or "energy_softmax").strip().lower()
    return (
        solver
        in {
            "energy_softmax_ccc_refined",
            "ccc_refined",
            "global_ccc_refined",
        }
        and int(config.ccc_refinement_steps) > 0
    )


def _ccc_refinement_candidates(
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> list[pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    seen: set[tuple[float, ...]] = set()
    max_candidates = max(int(config.ccc_refinement_steps), 1)
    for temperature in _ccc_refinement_temperature_grid(config):
        candidate = _support_weights_at_temperature(
            support_weights,
            config=config,
            temperature=temperature,
        )
        _append_unique_candidate(candidates, seen, candidate, max_candidates)
    uniform = _uniform_valid_support_weights(support_weights)
    _append_unique_candidate(candidates, seen, uniform, max_candidates)
    health_projected = _posterior_health_projected_support_weights(
        support_weights,
        config,
    )
    _append_unique_candidate(candidates, seen, health_projected, max_candidates)
    base = support_weights.copy()
    for alpha in _ccc_refinement_blend_grid(max_candidates):
        blended = base.copy()
        blended["posterior_weight"] = (
            (1.0 - alpha)
            * pd.to_numeric(base["posterior_weight"], errors="coerce").fillna(0.0)
            + alpha
            * pd.to_numeric(uniform["posterior_weight"], errors="coerce").fillna(0.0)
        )
        blended = _renormalize_support_weights_by_entity(blended)
        _append_unique_candidate(candidates, seen, blended, max_candidates)
    return candidates


def _coordinate_refine_support_weights(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
    initial_objective: float,
    config: ChemicalShiftSharedQConfig,
) -> tuple[pd.DataFrame, float, int]:
    passes = max(int(config.ccc_coordinate_refinement_passes), 0)
    if passes <= 0 or support_weights.empty:
        return support_weights, initial_objective, 0
    best = support_weights.copy()
    best_objective = initial_objective
    updates = 0
    for _ in range(passes):
        improved_this_pass = False
        for entity_uid, group in best.groupby("entity_uid", sort=True):
            indices = group.index.to_numpy(dtype=np.int64)
            entity_pairs = pairs[
                pairs["entity_uid"].astype(str).to_numpy(dtype=object)
                == str(entity_uid)
            ]
            for local_weights in _entity_coordinate_weight_candidates(
                group,
                config,
                pairs=entity_pairs,
            ):
                current = best.loc[indices, "posterior_weight"].to_numpy(
                    dtype=np.float64
                )
                if np.allclose(current, local_weights, atol=1.0e-10, rtol=1.0e-8):
                    continue
                candidate = best.copy()
                candidate.loc[indices, "posterior_weight"] = local_weights
                candidate = _renormalize_support_weights_by_entity(candidate)
                objective = _support_weight_refinement_objective(
                    pairs=pairs,
                    support_weights=candidate,
                    config=config,
                )
                if _objective_is_better(objective, best_objective):
                    best = candidate
                    best_objective = objective
                    updates += 1
                    improved_this_pass = True
                    break
        if not improved_this_pass:
            break
    return best, best_objective, updates


def _gradient_refine_support_weights(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
    initial_objective: float,
    config: ChemicalShiftSharedQConfig,
) -> tuple[pd.DataFrame, float, int]:
    steps = max(int(config.ccc_refinement_steps), 0)
    learning_rate = float(config.ccc_refinement_learning_rate)
    if steps <= 0 or learning_rate <= 0.0 or support_weights.empty:
        return support_weights, initial_objective, 0
    try:
        import torch
    except ImportError:
        return support_weights, initial_objective, 0

    prepared = _prepare_torch_refinement_inputs(
        pairs=pairs,
        support_weights=support_weights,
        config=config,
    )
    if prepared is None:
        return support_weights, initial_objective, 0

    support = prepared["support"]
    logits = torch.tensor(
        prepared["initial_logits"],
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    eligible = prepared["eligible"]
    eligible_tensor = torch.tensor(eligible, dtype=torch.bool)
    entity_groups = prepared["entity_groups"]
    base_q = torch.tensor(prepared["base_q"], dtype=torch.float64)
    energy = torch.tensor(prepared["energy"], dtype=torch.float64)
    pair_support_index = torch.tensor(
        prepared["pair_support_index"],
        dtype=torch.long,
    )
    pair_row_index = torch.tensor(prepared["pair_row_index"], dtype=torch.long)
    pair_prediction = torch.tensor(prepared["pair_prediction"], dtype=torch.float64)
    target = torch.tensor(prepared["target"], dtype=torch.float64)
    sigma = torch.tensor(prepared["sigma"], dtype=torch.float64)
    family_index = torch.tensor(prepared["family_index"], dtype=torch.long)
    row_count = int(prepared["row_count"])
    family_count = int(prepared["family_count"])

    best = support_weights.copy()
    best_objective = initial_objective
    best_q = prepared["current_q"]
    updates = 0
    for _ in range(steps):
        optimizer.zero_grad()
        q = _torch_grouped_softmax(logits, entity_groups, eligible)
        prediction, usable = _torch_posterior_predictions(
            q=q,
            pair_support_index=pair_support_index,
            pair_row_index=pair_row_index,
            pair_prediction=pair_prediction,
            row_count=row_count,
        )
        loss = _torch_refinement_loss(
            q=q,
            base_q=base_q,
            energy=energy,
            eligible=eligible_tensor,
            entity_groups=entity_groups,
            prediction=prediction,
            target=target,
            sigma=sigma,
            family_index=family_index,
            family_count=family_count,
            usable=usable,
            config=config,
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_q = (
                _torch_grouped_softmax(logits, entity_groups, eligible)
                .detach()
                .cpu()
                .numpy()
            )
        candidate = support.copy()
        candidate["posterior_weight"] = candidate_q
        candidate = _renormalize_support_weights_by_entity(candidate)
        objective = _support_weight_refinement_objective(
            pairs=pairs,
            support_weights=candidate,
            config=config,
        )
        if _objective_is_better(objective, best_objective):
            best = candidate
            best_objective = objective
            best_q = candidate["posterior_weight"].to_numpy(dtype=np.float64)
            updates += 1

    if updates <= 0:
        return support_weights, initial_objective, 0
    best["posterior_weight"] = best_q
    best = _renormalize_support_weights_by_entity(best)
    return best, best_objective, updates


def _prepare_torch_refinement_inputs(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> dict[str, Any] | None:
    support = support_weights.reset_index(drop=True).copy()
    support["support_index"] = np.arange(len(support), dtype=np.int64)
    pair_frame = pairs.merge(
        support[["entity_uid", "support_id", "support_index"]],
        on=["entity_uid", "support_id"],
        how="inner",
    )
    if pair_frame.empty:
        return None
    key_columns = [
        "entity_uid",
        "target_id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_family",
    ]
    row_keys = list(map(tuple, pair_frame[key_columns].astype(str).to_numpy()))
    row_lookup: dict[tuple[str, ...], int] = {}
    pair_row_index: list[int] = []
    target_values: list[float] = []
    sigma_values: list[float] = []
    family_values: list[str] = []
    for row_number, key in enumerate(row_keys):
        if key not in row_lookup:
            row_lookup[key] = len(row_lookup)
            target_values.append(float(pair_frame.iloc[row_number]["target_value"]))
            sigma_values.append(
                _pair_row_sigma(pair_frame.iloc[row_number], config=config)
            )
            family_values.append(str(pair_frame.iloc[row_number]["atom_family"]))
        pair_row_index.append(row_lookup[key])
    family_names = sorted(set(family_values))
    family_lookup = {family: index for index, family in enumerate(family_names)}
    current_q = (
        pd.to_numeric(support["posterior_weight"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    base_q = (
        pd.to_numeric(support["posterior_weight_initial"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    valid = support["support_valid"].fillna(True).to_numpy(dtype=bool)
    energy = pd.to_numeric(support["shared_q_energy"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    eligible = valid & np.isfinite(energy)
    if not bool(np.any(eligible)):
        return None
    energy = np.where(np.isfinite(energy), energy, 0.0)
    current_q = _renormalize_vector_by_entity(
        current_q,
        support["entity_uid"].astype(str).to_numpy(dtype=object),
        eligible=eligible,
    )
    base_q = _renormalize_vector_by_entity(
        base_q,
        support["entity_uid"].astype(str).to_numpy(dtype=object),
        eligible=eligible,
    )
    initial_logits = np.log(np.clip(current_q, 1.0e-12, None))
    initial_logits = np.where(eligible, initial_logits, 0.0)
    entity_groups = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in support.groupby("entity_uid", sort=False)
    ]
    return {
        "support": support.drop(columns=["support_index"]),
        "initial_logits": initial_logits,
        "current_q": current_q,
        "base_q": base_q,
        "energy": energy,
        "eligible": eligible,
        "entity_groups": entity_groups,
        "pair_support_index": pair_frame["support_index"].to_numpy(dtype=np.int64),
        "pair_row_index": np.asarray(pair_row_index, dtype=np.int64),
        "pair_prediction": pair_frame["predicted_value"].to_numpy(dtype=np.float64),
        "target": np.asarray(target_values, dtype=np.float64),
        "sigma": np.asarray(sigma_values, dtype=np.float64),
        "family_index": np.asarray(
            [family_lookup[family] for family in family_values],
            dtype=np.int64,
        ),
        "row_count": int(len(row_lookup)),
        "family_count": int(len(family_names)),
    }


def _pair_row_sigma(row: pd.Series, *, config: ChemicalShiftSharedQConfig) -> float:
    target_sigma = row.get("target_sigma", math.nan)
    try:
        sigma = float(target_sigma)
    except (TypeError, ValueError):
        sigma = math.nan
    if math.isfinite(sigma) and sigma > 0.0:
        return max(sigma, float(config.target_sigma_floor))
    family = str(row.get("atom_family", ""))
    fallback = float(config.family_ppm_scales.get(family, 1.0))
    return max(fallback, float(config.target_sigma_floor))


def _renormalize_vector_by_entity(
    values: np.ndarray,
    entities: np.ndarray,
    *,
    eligible: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for entity in pd.unique(pd.Series(entities)):
        mask = entities == entity
        local_eligible = mask & eligible
        if not bool(np.any(local_eligible)):
            continue
        local = np.where(local_eligible, values, 0.0)
        local = np.where(np.isfinite(local) & (local > 0.0), local, 0.0)
        total = float(np.sum(local))
        if total <= 0.0:
            out[local_eligible] = 1.0 / float(np.count_nonzero(local_eligible))
        else:
            out[local_eligible] = local[local_eligible] / total
    return out


def _entity_coordinate_weight_candidates(
    group: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
    *,
    pairs: pd.DataFrame | None = None,
) -> list[np.ndarray]:
    current = _normalize_vector(
        pd.to_numeric(group["posterior_weight"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
    valid = valid & np.isfinite(group["shared_q_energy"].to_numpy(dtype=float))
    if not bool(np.any(valid)):
        return []
    limit = max(int(config.ccc_coordinate_candidate_limit), 1)
    bases: list[np.ndarray] = [current]
    target_fit = _entity_target_fit_simplex_candidate(
        group=group,
        pairs=pairs,
        config=config,
    )
    if target_fit is not None:
        bases.append(target_fit)

    uniform = np.zeros(len(group), dtype=np.float64)
    uniform[valid] = 1.0 / float(np.count_nonzero(valid))
    bases.append(uniform)
    health_projected = _project_entity_weights_for_health(current, valid, config)
    bases.append(health_projected)

    for index in _family_residual_energy_support_indices(group, limit=limit):
        one_hot = np.zeros(len(group), dtype=np.float64)
        one_hot[index] = 1.0
        bases.append(one_hot)

    energy = pd.to_numeric(group["shared_q_energy"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    sorted_valid = [
        int(index)
        for index in np.argsort(np.where(valid & np.isfinite(energy), energy, np.inf))
        if valid[index] and math.isfinite(float(energy[index]))
    ]
    for index in sorted_valid[:limit]:
        one_hot = np.zeros(len(group), dtype=np.float64)
        one_hot[index] = 1.0
        bases.append(one_hot)
    for temperature in _ccc_refinement_temperature_grid(config)[:limit]:
        local_config = replace(
            config,
            energy_temperature=float(temperature),
            posterior_solver_kind="energy_softmax",
            ccc_refinement_steps=0,
        )
        bases.append(_posterior_softmax(support_table=group, config=local_config))

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    grid = tuple(float(value) for value in config.ccc_coordinate_mix_grid)
    for base in bases:
        base = _normalize_vector(base)
        for alpha in grid:
            alpha = min(max(float(alpha), 0.0), 1.0)
            mixed = _normalize_vector((1.0 - alpha) * current + alpha * base)
            key = tuple(np.round(mixed, 10).tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(mixed)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _entity_target_fit_simplex_candidate(
    *,
    group: pd.DataFrame,
    pairs: pd.DataFrame | None,
    config: ChemicalShiftSharedQConfig,
) -> np.ndarray | None:
    if not bool(config.ccc_coordinate_target_fit_candidates):
        return None
    if pairs is None or pairs.empty or group.empty:
        return None
    support_ids = group["support_id"].astype(str).tolist()
    if not support_ids:
        return None
    valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
    valid = valid & np.isfinite(group["shared_q_energy"].to_numpy(dtype=float))
    eligible_indices = np.flatnonzero(valid)
    if eligible_indices.size == 0:
        return None
    eligible_supports = [support_ids[int(index)] for index in eligible_indices]
    key_columns = [
        "entity_uid",
        "target_id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_family",
    ]
    pair_frame = pairs.copy()
    pair_frame["support_id"] = pair_frame["support_id"].astype(str)
    pair_frame = pair_frame[pair_frame["support_id"].isin(eligible_supports)]
    if pair_frame.empty:
        return None
    row_meta = pair_frame[key_columns + ["target_value", "target_sigma"]].drop_duplicates(
        key_columns
    )
    family_counts = row_meta["atom_family"].astype(str).value_counts().to_dict()
    rows: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []
    for _, row in row_meta.iterrows():
        row_mask = np.ones(len(pair_frame), dtype=bool)
        for column in key_columns:
            row_mask &= pair_frame[column].astype(str).to_numpy(dtype=object) == str(
                row[column]
            )
        row_pairs = pair_frame.loc[row_mask]
        values: list[float] = []
        complete = True
        for support_id in eligible_supports:
            support_values = pd.to_numeric(
                row_pairs.loc[
                    row_pairs["support_id"].astype(str) == support_id,
                    "predicted_value",
                ],
                errors="coerce",
            )
            finite = support_values[np.isfinite(support_values.to_numpy(dtype=float))]
            if finite.empty:
                complete = False
                break
            values.append(float(finite.iloc[0]))
        target_value = _safe_float(row.get("target_value"))
        if not complete or not math.isfinite(target_value):
            continue
        sigma = _pair_row_sigma(row, config=config)
        family = str(row.get("atom_family"))
        family_count = max(int(family_counts.get(family, 1)), 1)
        rows.append(values)
        targets.append(target_value)
        weights.append(1.0 / (max(float(sigma), 1.0e-12) * math.sqrt(family_count)))
    if not rows:
        return None
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    finite_rows = np.isfinite(matrix).all(axis=1) & np.isfinite(target) & np.isfinite(weight)
    if not bool(np.any(finite_rows)):
        return None
    matrix = matrix[finite_rows] * weight[finite_rows, None]
    target = target[finite_rows] * weight[finite_rows]
    try:
        local_q = _solve_sum_constrained_least_squares(matrix, target)
    except np.linalg.LinAlgError:
        return None
    local_q = _project_to_simplex(local_q)
    if local_q.size != eligible_indices.size:
        return None
    full = np.zeros(len(group), dtype=np.float64)
    full[eligible_indices] = local_q
    return _normalize_vector(full)


def _solve_sum_constrained_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    gram = matrix.T @ matrix
    rhs = matrix.T @ target
    ones = np.ones((gram.shape[0], 1), dtype=np.float64)
    kkt = np.block(
        [
            [gram, ones],
            [ones.T, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    target_rhs = np.concatenate([rhs, np.asarray([1.0], dtype=np.float64)])
    try:
        solution = np.linalg.solve(kkt, target_rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, target_rhs, rcond=None)[0]
    return np.asarray(solution[:-1], dtype=np.float64)


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    if vector.size == 0:
        return vector
    ordered = np.sort(vector)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    valid = ordered - cssv / indices > 0.0
    if not bool(np.any(valid)):
        return np.ones_like(vector) / float(vector.size)
    rho = int(np.flatnonzero(valid)[-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    return _normalize_vector(projected)


def _family_residual_energy_support_indices(
    group: pd.DataFrame,
    *,
    limit: int,
) -> list[int]:
    if "family_residual_energy_by_family" not in group.columns:
        return []
    valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
    best_by_family: dict[str, tuple[float, int]] = {}
    for local_index, (_, row) in enumerate(group.iterrows()):
        if not bool(valid[local_index]):
            continue
        energy_map = _parse_family_residual_energy_map(
            row.get("family_residual_energy_by_family")
        )
        for family, energy in energy_map.items():
            if not math.isfinite(energy):
                continue
            current = best_by_family.get(family)
            if current is None or energy < current[0]:
                best_by_family[family] = (energy, local_index)
    ordered: list[int] = []
    seen: set[int] = set()
    for _, index in sorted(best_by_family.values(), key=lambda item: (item[0], item[1])):
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
        if len(ordered) >= max(int(limit), 1):
            break
    return ordered


def _parse_family_residual_energy_map(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    out: dict[str, float] = {}
    for family, energy in dict(raw).items():
        try:
            numeric = float(energy)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            out[str(family)] = numeric
    return out


def _append_unique_candidate(
    candidates: list[pd.DataFrame],
    seen: set[tuple[float, ...]],
    candidate: pd.DataFrame,
    max_candidates: int,
) -> None:
    if len(candidates) >= max_candidates:
        return
    key = tuple(
        np.round(
            pd.to_numeric(candidate["posterior_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64),
            decimals=10,
        ).tolist()
    )
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _ccc_refinement_temperature_grid(
    config: ChemicalShiftSharedQConfig,
) -> tuple[float, ...]:
    base = max(float(config.energy_temperature), 1.0e-6)
    values = [
        base * factor
        for factor in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
    ]
    values.extend([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    finite = sorted({round(float(value), 10) for value in values if value > 0.0})
    return tuple(finite)


def _ccc_refinement_blend_grid(max_candidates: int) -> tuple[float, ...]:
    count = min(max(int(max_candidates), 2), 11)
    return tuple(float(value) for value in np.linspace(0.0, 1.0, count))


def _support_weights_at_temperature(
    support_weights: pd.DataFrame,
    *,
    config: ChemicalShiftSharedQConfig,
    temperature: float,
) -> pd.DataFrame:
    candidate = support_weights.copy()
    q = np.zeros(len(candidate), dtype=np.float64)
    for _, group in candidate.groupby("entity_uid", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        local_config = ChemicalShiftSharedQConfig(
            energy_temperature=float(temperature),
            energy_scale=float(config.energy_scale),
            prior_log_prob_weight=float(config.prior_log_prob_weight),
            target_sigma_floor=float(config.target_sigma_floor),
            family_ppm_scales=dict(config.family_ppm_scales),
            uncertainty_penalty_weight=float(config.uncertainty_penalty_weight),
            uncertainty_threshold=config.uncertainty_threshold,
            high_uncertainty_penalty=float(config.high_uncertainty_penalty),
            ood_score_threshold=config.ood_score_threshold,
            ood_penalty=float(config.ood_penalty),
            incomplete_support_penalty=float(config.incomplete_support_penalty),
            min_posterior_ess=float(config.min_posterior_ess),
            min_posterior_entropy=float(config.min_posterior_entropy),
            max_posterior_top_mass=float(config.max_posterior_top_mass),
            min_valid_mass=float(config.min_valid_mass),
            require_support_id=bool(config.require_support_id),
            posterior_solver_kind="energy_softmax",
            ccc_refinement_steps=0,
            ccc_refinement_learning_rate=float(config.ccc_refinement_learning_rate),
            ccc_refinement_residual_weight=float(
                config.ccc_refinement_residual_weight
            ),
            ccc_refinement_energy_weight=float(config.ccc_refinement_energy_weight),
            ccc_refinement_kl_weight=float(config.ccc_refinement_kl_weight),
            ccc_refinement_floor=float(config.ccc_refinement_floor),
            ccc_refinement_floor_weight=float(config.ccc_refinement_floor_weight),
            ccc_refinement_macro_floor=float(config.ccc_refinement_macro_floor),
            ccc_refinement_macro_floor_weight=float(
                config.ccc_refinement_macro_floor_weight
            ),
            ccc_refinement_macro_reward_weight=float(
                config.ccc_refinement_macro_reward_weight
            ),
            ccc_refinement_posterior_health_weight=float(
                config.ccc_refinement_posterior_health_weight
            ),
            ccc_refinement_floor_aggregation=_floor_aggregation_kind(config),
            ccc_refinement_min_family_points=int(
                config.ccc_refinement_min_family_points
            ),
            ccc_coordinate_refinement_passes=int(
                config.ccc_coordinate_refinement_passes
            ),
            ccc_coordinate_candidate_limit=int(
                config.ccc_coordinate_candidate_limit
            ),
            ccc_coordinate_mix_grid=tuple(config.ccc_coordinate_mix_grid),
            ccc_coordinate_target_fit_candidates=bool(
                config.ccc_coordinate_target_fit_candidates
            ),
            posterior_label=str(config.posterior_label),
        )
        q[indices] = _posterior_softmax(support_table=group, config=local_config)
    candidate["posterior_weight"] = q
    return candidate


def _uniform_valid_support_weights(support_weights: pd.DataFrame) -> pd.DataFrame:
    candidate = support_weights.copy()
    q = np.zeros(len(candidate), dtype=np.float64)
    for _, group in candidate.groupby("entity_uid", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
        valid = valid & np.isfinite(group["shared_q_energy"].to_numpy(dtype=float))
        if not bool(np.any(valid)):
            continue
        q[indices[valid]] = 1.0 / float(np.count_nonzero(valid))
    candidate["posterior_weight"] = q
    return candidate


def _posterior_health_projected_support_weights(
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> pd.DataFrame:
    candidate = support_weights.copy()
    q = np.zeros(len(candidate), dtype=np.float64)
    for _, group in candidate.groupby("entity_uid", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
        valid = valid & np.isfinite(group["shared_q_energy"].to_numpy(dtype=float))
        current = (
            pd.to_numeric(group["posterior_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        q[indices] = _project_entity_weights_for_health(current, valid, config)
    candidate["posterior_weight"] = q
    return candidate


def _project_entity_weights_for_health(
    weights: np.ndarray,
    valid: np.ndarray,
    config: ChemicalShiftSharedQConfig,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros_like(values, dtype=np.float64)
    eligible = valid & np.isfinite(values)
    if not bool(np.any(eligible)):
        return out
    n = int(np.count_nonzero(eligible))
    if n <= 1:
        out[eligible] = 1.0
        return out
    local = np.where(eligible, values, 0.0)
    local = np.where(np.isfinite(local) & (local > 0.0), local, 0.0)
    if float(np.sum(local)) <= 0.0:
        out[eligible] = 1.0 / float(n)
        return out
    local = local / float(np.sum(local))
    cap = _health_projection_top_mass_cap(config, n)
    if cap >= 1.0 - 1.0e-12:
        out[eligible] = local[eligible]
        return out
    projected = _project_to_capped_simplex(local[eligible], cap)
    out[eligible] = projected
    return out


def _health_projection_top_mass_cap(
    config: ChemicalShiftSharedQConfig,
    support_count: int,
) -> float:
    n = max(int(support_count), 1)
    lower = 1.0 / float(n)
    cap = 1.0
    max_top = float(config.max_posterior_top_mass)
    if math.isfinite(max_top) and max_top > 0.0:
        cap = min(cap, max_top)
    min_ess = float(config.min_posterior_ess)
    if math.isfinite(min_ess) and min_ess > 1.0:
        cap = min(cap, 1.0 / math.sqrt(min_ess))
    min_entropy = float(config.min_posterior_entropy)
    if math.isfinite(min_entropy) and min_entropy > 0.0:
        # A sufficient, conservative cap: entropy is at least -log(max q).
        cap = min(cap, math.exp(-min_entropy))
    return min(max(cap, lower), 1.0)


def _project_to_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    vector = np.where(np.isfinite(vector), vector, 0.0)
    n = int(vector.size)
    if n <= 0:
        return vector
    upper = min(max(float(cap), 1.0 / float(n)), 1.0)
    if upper >= 1.0 - 1.0e-12:
        return _normalize_vector(vector)
    low = float(np.min(vector) - upper)
    high = float(np.max(vector))
    for _ in range(80):
        theta = 0.5 * (low + high)
        projected = np.clip(vector - theta, 0.0, upper)
        total = float(np.sum(projected))
        if total > 1.0:
            low = theta
        else:
            high = theta
    projected = np.clip(vector - high, 0.0, upper)
    total = float(np.sum(projected))
    if total <= 0.0 or not math.isfinite(total):
        projected = np.ones(n, dtype=np.float64) / float(n)
    else:
        projected = projected / total
    return projected


def _renormalize_support_weights_by_entity(support_weights: pd.DataFrame) -> pd.DataFrame:
    out = support_weights.copy()
    normalized = np.zeros(len(out), dtype=np.float64)
    for _, group in out.groupby("entity_uid", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        weights = pd.to_numeric(group["posterior_weight"], errors="coerce").fillna(0.0)
        values = weights.to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
        total = float(np.sum(values))
        if total <= 0.0:
            valid = group["support_valid"].fillna(True).to_numpy(dtype=bool)
            valid = valid & np.isfinite(group["shared_q_energy"].to_numpy(dtype=float))
            if bool(np.any(valid)):
                values = np.zeros_like(values)
                values[valid] = 1.0 / float(np.count_nonzero(valid))
        else:
            values = values / total
        normalized[indices] = values
    out["posterior_weight"] = normalized
    return out


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / max(int(weights.size), 1)
    return weights / total


def _support_weight_refinement_objective(
    *,
    pairs: pd.DataFrame,
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    posterior = _posterior_mean_predictions(
        pairs=pairs,
        support_weights=support_weights,
    )
    key_columns = [
        "entity_uid",
        "target_id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_family",
    ]
    target_rows = pairs[key_columns + ["target_value"]].drop_duplicates(key_columns)
    scored = posterior.merge(target_rows, on=key_columns, how="inner")
    if scored.empty:
        return math.inf
    residual_loss = _family_balanced_residual_objective(scored, config)
    ccc_loss = _family_ccc_floor_objective(scored, config)
    macro_ccc_loss = _family_macro_ccc_floor_objective(scored, config)
    macro_ccc_reward = _family_macro_ccc_value(scored, config)
    energy_loss = _support_expected_energy(support_weights)
    kl_loss = _support_kl_from_initial(support_weights)
    posterior_health_loss = _posterior_health_refinement_penalty(
        support_weights,
        config,
    )
    objective = (
        float(config.ccc_refinement_residual_weight) * residual_loss
        + float(config.ccc_refinement_floor_weight) * ccc_loss
        + float(config.ccc_refinement_macro_floor_weight) * macro_ccc_loss
        - float(config.ccc_refinement_macro_reward_weight)
        * (macro_ccc_reward if math.isfinite(macro_ccc_reward) else 0.0)
        + float(config.ccc_refinement_posterior_health_weight)
        * posterior_health_loss
        + float(config.ccc_refinement_energy_weight) * energy_loss
        + float(config.ccc_refinement_kl_weight) * kl_loss
    )
    return objective if math.isfinite(objective) else math.inf


def _posterior_health_refinement_penalty(
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    if float(config.ccc_refinement_posterior_health_weight) <= 0.0:
        return 0.0
    if support_weights.empty:
        return 0.0
    penalties: list[float] = []
    for _, group in support_weights.groupby("entity_uid", sort=False):
        q = (
            pd.to_numeric(group["posterior_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        q = np.where(np.isfinite(q) & (q > 0.0), q, 0.0)
        if q.size <= 0:
            penalties.append(1.0)
            continue
        valid = group["support_valid"].fillna(False).to_numpy(dtype=bool)
        ess = 1.0 / max(float(np.sum(np.square(q))), 1.0e-12)
        entropy = float(-np.sum(q * np.log(np.maximum(q, 1.0e-300))))
        top_mass = float(np.max(q)) if q.size else 0.0
        valid_mass = float(np.sum(q[valid])) if valid.size == q.size else 0.0
        penalties.append(
            _floor_deficit_penalty(ess, float(config.min_posterior_ess))
            + _floor_deficit_penalty(
                entropy,
                float(config.min_posterior_entropy),
            )
            + _ceiling_excess_penalty(
                top_mass,
                float(config.max_posterior_top_mass),
            )
            + _floor_deficit_penalty(valid_mass, float(config.min_valid_mass))
        )
    finite = [value for value in penalties if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.inf


def _floor_deficit_penalty(value: float, floor: float) -> float:
    if not math.isfinite(floor) or floor <= 0.0:
        return 0.0
    if not math.isfinite(value):
        return 1.0
    gap = max(0.0, float(floor) - float(value))
    scale = max(abs(float(floor)), 1.0)
    return (gap / scale) ** 2


def _ceiling_excess_penalty(value: float, ceiling: float) -> float:
    if not math.isfinite(ceiling):
        return 0.0
    if not math.isfinite(value):
        return 1.0
    gap = max(0.0, float(value) - float(ceiling))
    if gap <= 0.0:
        return 0.0
    scale = max(1.0 - min(max(float(ceiling), 0.0), 1.0), 1.0e-6)
    return (gap / scale) ** 2


def _family_balanced_residual_objective(
    scored: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    family_values: list[float] = []
    for family, family_frame in scored.groupby("atom_family", sort=True):
        prediction = family_frame["predicted_value"].to_numpy(dtype=np.float64)
        target = family_frame["target_value"].to_numpy(dtype=np.float64)
        scale = max(float(config.family_ppm_scales.get(str(family), 1.0)), 1.0e-12)
        scaled = (prediction - target) / scale
        finite = scaled[np.isfinite(scaled)]
        if finite.size:
            family_values.append(float(np.mean(np.square(finite))))
    return float(np.mean(family_values)) if family_values else math.inf


def _family_ccc_floor_objective(
    scored: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    floor = float(config.ccc_refinement_floor)
    losses = [
        max(0.0, floor - ccc) ** 2
        for ccc in _family_ccc_values(scored, config)
        if math.isfinite(ccc)
    ]
    if not losses:
        return 0.0
    if _floor_aggregation_kind(config) == "active_set":
        return float(np.max(losses))
    return float(np.mean(losses))


def _family_macro_ccc_floor_objective(
    scored: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    macro_ccc = _family_macro_ccc_value(scored, config)
    if not math.isfinite(macro_ccc):
        return 0.0
    floor = float(config.ccc_refinement_macro_floor)
    return max(0.0, floor - macro_ccc) ** 2


def _family_macro_ccc_value(
    scored: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    values = [
        ccc for ccc in _family_ccc_values(scored, config) if math.isfinite(ccc)
    ]
    return float(np.mean(values)) if values else math.nan


def _family_ccc_values(
    scored: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> list[float]:
    values: list[float] = []
    min_points = max(int(config.ccc_refinement_min_family_points), 2)
    for _, family_frame in scored.groupby("atom_family", sort=True):
        if int(len(family_frame)) < min_points:
            continue
        prediction = family_frame["predicted_value"].to_numpy(dtype=np.float64)
        target = family_frame["target_value"].to_numpy(dtype=np.float64)
        ccc = _lin_ccc(prediction, target)
        if math.isfinite(ccc):
            values.append(ccc)
    return values


def _floor_aggregation_kind(config: ChemicalShiftSharedQConfig) -> str:
    kind = str(config.ccc_refinement_floor_aggregation or "active_set").strip().lower()
    if kind in {"active", "active_set", "worst", "worst_family", "max"}:
        return "active_set"
    if kind in {"mean", "average", "macro"}:
        return "mean"
    raise ValueError(
        "ccc_refinement_floor_aggregation must be 'active_set' or 'mean'"
    )


def _support_expected_energy(support_weights: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in support_weights.groupby("entity_uid", sort=False):
        q = pd.to_numeric(group["posterior_weight"], errors="coerce").fillna(0.0)
        energy = pd.to_numeric(group["shared_q_energy"], errors="coerce").fillna(
            math.inf
        )
        finite = np.isfinite(energy.to_numpy(dtype=float))
        if bool(np.any(finite)):
            values.append(
                float(np.sum(q.to_numpy(dtype=float)[finite] * energy.to_numpy(dtype=float)[finite]))
            )
    return float(np.mean(values)) if values else math.inf


def _support_kl_from_initial(support_weights: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in support_weights.groupby("entity_uid", sort=False):
        q = pd.to_numeric(group["posterior_weight"], errors="coerce").fillna(0.0)
        base = pd.to_numeric(
            group["posterior_weight_initial"],
            errors="coerce",
        ).fillna(0.0)
        q_values = np.clip(q.to_numpy(dtype=np.float64), 1.0e-12, None)
        base_values = np.clip(base.to_numpy(dtype=np.float64), 1.0e-12, None)
        values.append(float(np.sum(q_values * (np.log(q_values) - np.log(base_values)))))
    return float(np.mean(values)) if values else 0.0


def _objective_is_better(candidate: float, current: float) -> bool:
    if not math.isfinite(candidate):
        return False
    return (not math.isfinite(current)) or candidate + 1.0e-12 < current


def _lin_ccc(prediction: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[finite]
    target = target[finite]
    if prediction.size < 2:
        return math.nan
    pred_mean = float(np.mean(prediction))
    target_mean = float(np.mean(target))
    pred_var = float(np.mean(np.square(prediction - pred_mean)))
    target_var = float(np.mean(np.square(target - target_mean)))
    covariance = float(np.mean((prediction - pred_mean) * (target - target_mean)))
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    return math.nan if denominator <= 1.0e-12 else float(2.0 * covariance / denominator)


def _family_scale_array(
    families: pd.Series,
    config: ChemicalShiftSharedQConfig,
) -> np.ndarray:
    values = [
        float(config.family_ppm_scales.get(str(family), 1.0))
        for family in families.astype(str).tolist()
    ]
    return np.maximum(
        np.asarray(values, dtype=np.float64),
        float(config.target_sigma_floor),
    )


def _torch_grouped_softmax(
    logits: "torch.Tensor",
    entity_groups: Sequence[np.ndarray],
    eligible: np.ndarray,
) -> "torch.Tensor":
    import torch

    q = torch.zeros_like(logits)
    for group in entity_groups:
        eligible_indices = group[eligible[group]]
        if eligible_indices.size == 0:
            continue
        index = torch.as_tensor(
            eligible_indices,
            dtype=torch.long,
            device=logits.device,
        )
        q = q.index_put((index,), torch.softmax(logits.index_select(0, index), dim=0))
    return q


def _torch_posterior_predictions(
    *,
    q: "torch.Tensor",
    pair_support_index: "torch.Tensor",
    pair_row_index: "torch.Tensor",
    pair_prediction: "torch.Tensor",
    row_count: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    row_shape = (int(row_count),)
    numerator = q.new_zeros(row_shape)
    denominator = q.new_zeros(row_shape)
    pair_q = q.index_select(0, pair_support_index)
    numerator = numerator.index_add(0, pair_row_index, pair_q * pair_prediction)
    denominator = denominator.index_add(0, pair_row_index, pair_q)
    usable = denominator > 1.0e-10
    prediction = numerator / denominator.clamp_min(1.0e-10)
    return prediction, usable


def _torch_refinement_loss(
    *,
    q: "torch.Tensor",
    base_q: "torch.Tensor",
    energy: "torch.Tensor",
    eligible: "torch.Tensor",
    entity_groups: list[np.ndarray],
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    sigma: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
    config: ChemicalShiftSharedQConfig,
) -> "torch.Tensor":
    import torch

    residual_loss = _torch_family_balanced_residual_loss(
        prediction=prediction,
        target=target,
        sigma=sigma,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
    )
    ccc_loss = _torch_family_ccc_floor_loss(
        prediction=prediction,
        target=target,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
        floor=float(config.ccc_refinement_floor),
        min_points=int(config.ccc_refinement_min_family_points),
        aggregation=_floor_aggregation_kind(config),
    )
    macro_ccc_loss = _torch_family_macro_ccc_floor_loss(
        prediction=prediction,
        target=target,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
        floor=float(config.ccc_refinement_macro_floor),
        min_points=int(config.ccc_refinement_min_family_points),
    )
    macro_ccc_reward = _torch_family_macro_ccc_value(
        prediction=prediction,
        target=target,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
        min_points=int(config.ccc_refinement_min_family_points),
    )
    energy_loss = torch.sum(q * energy) / max(int(family_count), 1)
    q_safe = q.clamp_min(1.0e-12)
    base_safe = base_q.clamp_min(1.0e-12)
    kl_loss = torch.sum(q_safe * (torch.log(q_safe) - torch.log(base_safe)))
    posterior_health_loss = _torch_posterior_health_penalty(
        q=q,
        eligible=eligible,
        entity_groups=entity_groups,
        config=config,
    )
    return (
        prediction.new_tensor(float(config.ccc_refinement_residual_weight))
        * residual_loss
        + prediction.new_tensor(float(config.ccc_refinement_floor_weight))
        * ccc_loss
        + prediction.new_tensor(float(config.ccc_refinement_macro_floor_weight))
        * macro_ccc_loss
        - prediction.new_tensor(float(config.ccc_refinement_macro_reward_weight))
        * macro_ccc_reward
        + prediction.new_tensor(float(config.ccc_refinement_posterior_health_weight))
        * posterior_health_loss
        + prediction.new_tensor(float(config.ccc_refinement_energy_weight))
        * energy_loss
        + prediction.new_tensor(float(config.ccc_refinement_kl_weight))
        * kl_loss
    )


def _torch_posterior_health_penalty(
    *,
    q: "torch.Tensor",
    eligible: "torch.Tensor",
    entity_groups: list[np.ndarray],
    config: ChemicalShiftSharedQConfig,
) -> "torch.Tensor":
    import torch

    if float(config.ccc_refinement_posterior_health_weight) <= 0.0:
        return q.sum() * 0.0
    penalties = []
    for indices in entity_groups:
        if len(indices) <= 0:
            continue
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=q.device)
        local_q = q.index_select(0, index_tensor)
        local_eligible = eligible.index_select(0, index_tensor)
        ess = 1.0 / torch.sum(torch.square(local_q)).clamp_min(1.0e-12)
        entropy = -torch.sum(local_q * torch.log(local_q.clamp_min(1.0e-300)))
        top_mass = torch.max(local_q) if local_q.numel() else q.new_tensor(0.0)
        valid_mass = torch.sum(local_q[local_eligible])
        penalties.append(
            _torch_floor_deficit_penalty(
                ess,
                float(config.min_posterior_ess),
            )
            + _torch_floor_deficit_penalty(
                entropy,
                float(config.min_posterior_entropy),
            )
            + _torch_ceiling_excess_penalty(
                top_mass,
                float(config.max_posterior_top_mass),
            )
            + _torch_floor_deficit_penalty(valid_mass, float(config.min_valid_mass))
        )
    if not penalties:
        return q.sum() * 0.0
    return torch.mean(torch.stack(penalties))


def _torch_floor_deficit_penalty(
    value: "torch.Tensor",
    floor: float,
) -> "torch.Tensor":
    import torch

    if not math.isfinite(float(floor)) or float(floor) <= 0.0:
        return value * 0.0
    scale = max(abs(float(floor)), 1.0)
    return torch.square(torch.relu(value.new_tensor(float(floor)) - value) / scale)


def _torch_ceiling_excess_penalty(
    value: "torch.Tensor",
    ceiling: float,
) -> "torch.Tensor":
    import torch

    if not math.isfinite(float(ceiling)):
        return value * 0.0
    bounded_ceiling = min(max(float(ceiling), 0.0), 1.0)
    scale = max(1.0 - bounded_ceiling, 1.0e-6)
    return torch.square(torch.relu(value - value.new_tensor(float(ceiling))) / scale)


def _torch_family_balanced_residual_loss(
    *,
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    sigma: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    losses = []
    scaled = (prediction - target) / sigma.clamp_min(1.0e-10)
    for index in range(max(int(family_count), 0)):
        mask = usable & (family_index == int(index))
        if int(torch.count_nonzero(mask).detach().cpu()) <= 0:
            continue
        losses.append(torch.mean(torch.square(scaled[mask])))
    if not losses:
        return prediction.sum() * 0.0
    return torch.mean(torch.stack(losses))


def _torch_family_ccc_floor_loss(
    *,
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
    floor: float,
    min_points: int,
    aggregation: str,
) -> "torch.Tensor":
    import torch

    losses = []
    point_floor = max(int(min_points), 2)
    floor_t = prediction.new_tensor(float(floor))
    for index in range(max(int(family_count), 0)):
        mask = usable & (family_index == int(index))
        if int(torch.count_nonzero(mask).detach().cpu()) < point_floor:
            continue
        pred = prediction[mask]
        truth = target[mask]
        finite = torch.isfinite(pred) & torch.isfinite(truth)
        if int(torch.count_nonzero(finite).detach().cpu()) < point_floor:
            continue
        pred = pred[finite]
        truth = truth[finite]
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        covariance = torch.mean(pred_centered * truth_centered)
        pred_var = torch.mean(torch.square(pred_centered))
        target_var = torch.mean(torch.square(truth_centered))
        mean_delta = pred.mean() - truth.mean()
        ccc = (
            2.0
            * covariance
            / (pred_var + target_var + torch.square(mean_delta)).clamp_min(1.0e-10)
        )
        losses.append(torch.square(torch.relu(floor_t - ccc)))
    if not losses:
        return prediction.sum() * 0.0
    stacked = torch.stack(losses)
    if str(aggregation) == "active_set":
        return torch.max(stacked)
    return torch.mean(stacked)


def _torch_family_macro_ccc_floor_loss(
    *,
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
    floor: float,
    min_points: int,
) -> "torch.Tensor":
    import torch

    values = _torch_family_ccc_values(
        prediction=prediction,
        target=target,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
        min_points=min_points,
    )
    if not values:
        return prediction.sum() * 0.0
    macro_ccc = torch.mean(torch.stack(values))
    return torch.square(torch.relu(prediction.new_tensor(float(floor)) - macro_ccc))


def _torch_family_macro_ccc_value(
    *,
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
    min_points: int,
) -> "torch.Tensor":
    import torch

    values = _torch_family_ccc_values(
        prediction=prediction,
        target=target,
        family_index=family_index,
        family_count=family_count,
        usable=usable,
        min_points=min_points,
    )
    if not values:
        return prediction.sum() * 0.0
    return torch.mean(torch.stack(values))


def _torch_family_ccc_values(
    *,
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    family_index: "torch.Tensor",
    family_count: int,
    usable: "torch.Tensor",
    min_points: int,
) -> list["torch.Tensor"]:
    import torch

    values = []
    point_floor = max(int(min_points), 2)
    for index in range(max(int(family_count), 0)):
        mask = usable & (family_index == int(index))
        if int(torch.count_nonzero(mask).detach().cpu()) < point_floor:
            continue
        pred = prediction[mask]
        truth = target[mask]
        finite = torch.isfinite(pred) & torch.isfinite(truth)
        if int(torch.count_nonzero(finite).detach().cpu()) < point_floor:
            continue
        pred = pred[finite]
        truth = truth[finite]
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        covariance = torch.mean(pred_centered * truth_centered)
        pred_var = torch.mean(torch.square(pred_centered))
        target_var = torch.mean(torch.square(truth_centered))
        mean_delta = pred.mean() - truth.mean()
        values.append(
            2.0
            * covariance
            / (pred_var + target_var + torch.square(mean_delta)).clamp_min(1.0e-10)
        )
    return values


def _shared_q_summary(
    *,
    parity: dict[str, Any],
    support_weights: pd.DataFrame,
    entity_health: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
    join_keys: Sequence[str],
) -> dict[str, Any]:
    health_passed = _posterior_health_passed(entity_health, config)
    parity_summary = parity["summary"]
    gate_split = str(parity_summary.get("gate_split") or "val")
    gate_split_summary = (
        parity_summary.get("split_summaries", {}).get(gate_split, {})
    )
    macro_ccc = _safe_float(gate_split_summary.get("family_macro_ccc"))
    macro_floor = float(config.ccc_refinement_macro_floor)
    macro_gate_passed = bool(
        math.isfinite(macro_ccc) and macro_ccc + 1.0e-12 >= macro_floor
    )
    return {
        "bundle_kind": SHARED_Q_POSTERIOR_KIND,
        "shared_q_interface_kind": SHARED_Q_INTERFACE_KIND,
        "posterior_solver_kind": str(config.posterior_solver_kind),
        "posterior_population_is_one_shared_q": True,
        "posterior_mean_source": SHARED_Q_POSTERIOR_KIND,
        "decision": (
            "pass"
            if (
                parity_summary.get("decision") == "pass"
                and health_passed
                and macro_gate_passed
            )
            else "fail"
        ),
        "passes_family_ccc_gate": bool(parity_summary.get("passes_family_ccc_gate")),
        "passes_macro_family_ccc_gate": macro_gate_passed,
        "family_macro_ccc": macro_ccc,
        "family_macro_ccc_floor": macro_floor,
        "family_macro_ccc_gap_to_floor": (
            max(0.0, macro_floor - macro_ccc)
            if math.isfinite(macro_ccc)
            else math.nan
        ),
        "predictor_reliability_gate_passed": bool(
            parity_summary.get("predictor_reliability_gate_passed")
        ),
        "posterior_health_gate_passed": health_passed,
        "target_ccc": float(parity_summary.get("target_ccc") or 0.95),
        "gate_split": gate_split,
        "join_keys": list(join_keys),
        "support_count": int(len(support_weights)),
        "entity_count": int(support_weights["entity_uid"].nunique())
        if not support_weights.empty
        else 0,
        "posterior_health": entity_health.to_dict("records"),
        "ccc_refinement": _ccc_refinement_summary(support_weights, config),
        "family_summary": parity_summary.get("split_summaries", {}),
        "parity_summary": parity_summary,
    }


def _ccc_refinement_summary(
    support_weights: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> dict[str, Any]:
    if support_weights.empty:
        return {
            "enabled": _ccc_refinement_enabled(config),
            "support_count": 0,
            "max_abs_weight_delta": 0.0,
        }
    delta = pd.to_numeric(
        support_weights.get(
            "posterior_weight_refinement_delta",
            pd.Series(0.0, index=support_weights.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    objective = pd.to_numeric(
        support_weights.get(
            "ccc_refinement_objective",
            pd.Series(math.nan, index=support_weights.index),
        ),
        errors="coerce",
    )
    coordinate_updates = pd.to_numeric(
        support_weights.get(
            "ccc_coordinate_refinement_updates",
            pd.Series(0, index=support_weights.index),
        ),
        errors="coerce",
    ).fillna(0)
    gradient_updates = pd.to_numeric(
        support_weights.get(
            "ccc_gradient_refinement_updates",
            pd.Series(0, index=support_weights.index),
        ),
        errors="coerce",
    ).fillna(0)
    finite_objective = objective[np.isfinite(objective.to_numpy(dtype=float))]
    return {
        "enabled": _ccc_refinement_enabled(config),
        "solver_kind": str(config.posterior_solver_kind),
        "steps": int(config.ccc_refinement_steps),
        "learning_rate": float(config.ccc_refinement_learning_rate),
        "floor": float(config.ccc_refinement_floor),
        "floor_weight": float(config.ccc_refinement_floor_weight),
        "macro_floor": float(config.ccc_refinement_macro_floor),
        "macro_floor_weight": float(config.ccc_refinement_macro_floor_weight),
        "macro_reward_weight": float(config.ccc_refinement_macro_reward_weight),
        "posterior_health_weight": float(
            config.ccc_refinement_posterior_health_weight
        ),
        "floor_aggregation": _floor_aggregation_kind(config),
        "support_count": int(len(support_weights)),
        "refined_support_count": int((delta.abs() > 1.0e-9).sum()),
        "max_abs_weight_delta": float(delta.abs().max()) if len(delta) else 0.0,
        "mean_abs_weight_delta": float(delta.abs().mean()) if len(delta) else 0.0,
        "best_objective": (
            float(finite_objective.min()) if not finite_objective.empty else math.nan
        ),
        "coordinate_refinement_passes": int(config.ccc_coordinate_refinement_passes),
        "coordinate_candidate_limit": int(config.ccc_coordinate_candidate_limit),
        "coordinate_target_fit_candidates": bool(
            config.ccc_coordinate_target_fit_candidates
        ),
        "coordinate_update_count": int(coordinate_updates.max())
        if len(coordinate_updates)
        else 0,
        "gradient_update_count": int(gradient_updates.max())
        if len(gradient_updates)
        else 0,
    }


def _family_balanced_energy(
    frame: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> dict[str, Any]:
    family_values: list[float] = []
    family_energy: dict[str, float] = {}
    for family, family_frame in frame.groupby("atom_family", sort=True):
        residual = (
            family_frame["predicted_value"].to_numpy(dtype=float)
            - family_frame["target_value"].to_numpy(dtype=float)
        )
        sigma = _row_scales(family_frame, config)
        scaled = residual / sigma
        finite = scaled[np.isfinite(scaled)]
        if finite.size:
            value = float(np.mean(np.square(finite)))
            family_values.append(value)
            family_energy[str(family)] = value
    energy = float(np.mean(family_values)) if family_values else 0.0
    return {
        "energy": energy,
        "family_count": int(len(family_values)),
        "family_residual_energy_by_family": family_energy,
        "family_residual_energy_min": (
            float(min(family_values)) if family_values else math.nan
        ),
    }


def _row_scales(
    frame: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> np.ndarray:
    target_sigma = pd.to_numeric(
        frame.get("target_sigma", pd.Series(math.nan, index=frame.index)),
        errors="coerce",
    )
    sigma = target_sigma.to_numpy(dtype=float)
    families = frame["atom_family"].astype(str).to_numpy(dtype=object)
    fallback = np.asarray(
        [
            float(config.family_ppm_scales.get(str(family), 1.0))
            for family in families.tolist()
        ],
        dtype=float,
    )
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, fallback)
    return np.maximum(sigma, float(config.target_sigma_floor))


def _uncertainty_penalty(
    frame: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    uncertainty = pd.to_numeric(
        frame.get("prediction_uncertainty", pd.Series(math.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = uncertainty[np.isfinite(uncertainty)]
    penalty = (
        float(config.uncertainty_penalty_weight) * float(np.mean(finite))
        if finite.size
        else 0.0
    )
    threshold = config.uncertainty_threshold
    if threshold is not None and math.isfinite(float(threshold)):
        high = finite > float(threshold)
        if high.size:
            penalty += float(config.high_uncertainty_penalty) * float(np.mean(high))
    return penalty


def _ood_penalty(
    frame: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> float:
    flag = _boolean_series(
        frame.get("prediction_ood_flag", pd.Series(False, index=frame.index))
    ).fillna(False)
    ood = flag.to_numpy(dtype=bool)
    threshold = config.ood_score_threshold
    if threshold is not None and math.isfinite(float(threshold)):
        score = pd.to_numeric(
            frame.get("prediction_ood_score", pd.Series(math.nan, index=frame.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        ood = ood | (np.isfinite(score) & (score > float(threshold)))
    return float(config.ood_penalty) if bool(np.any(ood)) else 0.0


def _posterior_softmax(
    *,
    support_table: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> np.ndarray:
    valid = support_table["support_valid"].fillna(True).to_numpy(dtype=bool)
    energy = support_table["shared_q_energy"].to_numpy(dtype=float)
    prior = support_table["prior_log_prob"].fillna(0.0).to_numpy(dtype=float)
    prior = np.where(np.isfinite(prior), prior, 0.0)
    eligible = valid & np.isfinite(energy)
    if not bool(np.any(eligible)):
        raise ValueError("No valid finite-energy supports are available for shared q")
    logits = np.full(len(support_table), -np.inf, dtype=float)
    temperature = max(float(config.energy_temperature), 1.0e-12)
    logits[eligible] = (
        float(config.prior_log_prob_weight) * prior[eligible]
        - energy[eligible] / temperature
    )
    finite_logits = logits[np.isfinite(logits)]
    max_logit = float(np.max(finite_logits))
    weights = np.zeros(len(support_table), dtype=float)
    weights[eligible] = np.exp(logits[eligible] - max_logit)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Shared-q posterior normalization failed")
    return weights / total


def _entity_health_row(
    entity_uid: str,
    support_table: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> dict[str, Any]:
    eps = 1.0e-9
    q = support_table["posterior_weight"].to_numpy(dtype=float)
    q = q[np.isfinite(q)]
    ess = 1.0 / float(np.sum(np.square(q))) if q.size else 0.0
    entropy = float(-np.sum(q * np.log(np.maximum(q, 1.0e-300)))) if q.size else 0.0
    top_mass = float(np.max(q)) if q.size else 0.0
    valid_mass = float(
        support_table.loc[
            support_table["support_valid"].fillna(False), "posterior_weight"
        ].sum()
    )
    return {
        "entity_uid": entity_uid,
        "posterior_ess": ess,
        "posterior_entropy": entropy,
        "posterior_top_mass": top_mass,
        "valid_mass": valid_mass,
        "posterior_ess_safe": ess + eps >= float(config.min_posterior_ess),
        "posterior_entropy_safe": entropy + eps >= float(config.min_posterior_entropy),
        "posterior_top_mass_safe": top_mass <= float(config.max_posterior_top_mass) + eps,
        "valid_mass_safe": valid_mass + eps >= float(config.min_valid_mass),
    }


def _posterior_health_passed(
    entity_health: pd.DataFrame,
    config: ChemicalShiftSharedQConfig,
) -> bool:
    if entity_health.empty:
        return False
    checks = (
        "posterior_ess_safe",
        "posterior_entropy_safe",
        "posterior_top_mass_safe",
        "valid_mass_safe",
    )
    return all(bool(entity_health[column].all()) for column in checks)


def _expected_target_counts(
    target_frame: pd.DataFrame,
    join_keys: Sequence[str],
) -> dict[str, int]:
    unique_targets = target_frame.dropna(subset=list(join_keys)).drop_duplicates(
        list(join_keys)
    )
    return {
        str(entity_uid): int(len(group))
        for entity_uid, group in unique_targets.groupby("entity_uid", sort=False)
    }


def _coalesce_joined_column(
    joined: pd.DataFrame,
    column: str,
    join_keys: Sequence[str],
) -> pd.Series:
    if column in join_keys and column in joined.columns:
        return joined[column]
    left = f"{column}_prediction"
    right = f"{column}_target"
    if right in joined.columns and left in joined.columns:
        return joined[right].where(joined[right].notna(), joined[left])
    if right in joined.columns:
        return joined[right]
    if left in joined.columns:
        return joined[left]
    if column in joined.columns:
        return joined[column]
    return pd.Series(pd.NA, index=joined.index)


def _entity_uid_series(frame: pd.DataFrame) -> pd.Series:
    values, _ = _column_from_aliases(
        frame,
        ("entity_uid", "entry_uid", "bmrb_uid"),
        required=False,
        label="entity uid",
    )
    values = values.astype("string").str.strip()
    if values.notna().any():
        return values
    raise ValueError("split table is missing entity_uid/entry_uid/bmrb_uid")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else math.nan


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(path, sep="\t", index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True)
    elif suffix == ".json":
        frame.to_json(path, orient="records", indent=2)
    else:
        frame.to_csv(path, index=False)
