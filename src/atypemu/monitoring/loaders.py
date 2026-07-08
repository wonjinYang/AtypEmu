"""Artifact loaders for the AtypEmu training monitor.

The monitor is intentionally read-only. These helpers tolerate missing or
partially written files because training jobs may update JSON artifacts while
the Streamlit page is refreshing.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


JSON_CACHE_KEY_PREFIX = "json:"


@dataclass(frozen=True, slots=True)
class MonitorArtifactPaths:
    """Resolved artifact paths for one training run."""

    run_dir: Path
    history: Path
    summary: Path
    benchmark_report: Path
    chemical_shift_baseline_report: Path
    chemical_shift_calibration_report: Path
    learnability_report: Path
    observable_adapter_report: Path
    multi_observable_support_report: Path
    best_preview_report: Path
    moment_oracle_report: Path
    moment_consistency_report: Path
    ccc_decomposition_report: Path
    entry_family_calibration_report: Path
    residue_numbering_alignment_report: Path
    candidate_free_report: Path
    candidate_free_physics_risk_report: Path
    bioemu_latent_nmr_report: Path
    hn95_report: Path
    uncertainty_report: Path
    figures_dir: Path
    structures_dir: Path
    arrays_dir: Path
    projection_coordinates: Path
    ensemble_states: Path
    conformer_state_assignments: Path
    chemical_shift_posteriors: Path
    secondary_shift_posteriors: Path
    secondary_shift_targets: Path
    chemical_shift_joint_posteriors: Path
    chemical_shift_predictions_calibrated: Path
    ccc_support_oracle: Path
    ccc_oracle_weights: Path
    observable_adapter_summary: Path
    observable_oracle_weights: Path
    moment_predictions: Path
    moment_oracle_predictions: Path
    moment_oracle_weights: Path
    moment_oracle_state_occupancy: Path
    nmr_structural_features: Path
    measure_aware_targets: Path
    joint_nmr_posteriors: Path
    posterior_state_tokens: Path
    posterior_latent_samples: Path
    candidate_free_predictions: Path
    candidate_free_masked_predictions: Path
    candidate_free_repaired_targets: Path
    candidate_free_hn_physics_risk: Path
    candidate_free_hn_outlier_rows: Path
    candidate_free_hn_entry_bottlenecks: Path
    candidate_free_hn_strategy_audit: Path
    candidate_free_hn_ceiling_policy: Path
    candidate_free_hn_structure_proxy_audit: Path
    bioemu_latent_nmr_predictions: Path
    bioemu_latent_nmr_masked_predictions: Path
    bioemu_live_epoch_progress: Path
    bioemu_latent_sample_weights: Path
    bioemu_latent_atlas_coordinates: Path
    bioemu_posterior_moments: Path
    bioemu_posterior_bridge_audit: Path
    bioemu_ucbshift_cnnls_teacher_weights: Path
    bioemu_ucbshift_cnnls_teacher_predictions: Path
    bioemu_ucbshift_cnnls_teacher_audit: Path
    bioemu_ucbshift_cnnls_teacher_skipped: Path
    bioemu_family_conflict_audit: Path
    bioemu_physics_atlas_coordinates: Path
    bioemu_family_tangent_adapter_audit: Path
    bioemu_hn_cprime_direction_conflict_audit: Path
    bioemu_ring_physics_audit: Path
    bioemu_carbonyl_backbone_audit: Path
    bioemu_ucbshift_factor_atlas_audit: Path
    bioemu_mechanism_signed_direction_audit: Path
    bioemu_rare_regime_atlas: Path
    bioemu_rare_regime_summary: Path
    bioemu_rare_regime_acceptance: Path
    bioemu_family_chart_calibration_probe: Path
    bioemu_family_chart_calibrated_predictions: Path
    bioemu_family_affine_calibration_probe: Path
    bioemu_family_affine_calibrated_predictions: Path
    bioemu_family_calibration_strategy_probe: Path
    bioemu_family_calibration_strategy_metrics: Path
    bioemu_recommended_family_metrics: Path
    bioemu_family_calibration_strategy_predictions: Path
    bioemu_family_calibration_strategy_residue_audit: Path
    bioemu_family_calibration_strategy_residue_type_audit: Path
    bioemu_family_calibration_strategy_residue_type_guard_probe: Path
    bioemu_family_calibration_strategy_residue_type_guard_predictions: Path
    bioemu_proline_n_rare_regime_audit: Path
    bioemu_proline_n_expert_probe: Path
    bioemu_proline_n_expert_predictions: Path
    bioemu_proline_n_mobility_expert_probe: Path
    bioemu_proline_n_mobility_expert_split_guard: Path
    bioemu_proline_n_mobility_expert_predictions: Path
    bioemu_proline_n_mobility_entity_holdout: Path
    bioemu_proline_n_mobility_entity_holdout_predictions: Path
    bioemu_residue_family_expert_probe: Path
    bioemu_residue_family_expert_predictions: Path
    bioemu_residue_family_expert_policy_probe: Path
    bioemu_residue_family_expert_policy_predictions: Path
    bioemu_residue_family_expert_reliability_gate_probe: Path
    bioemu_residue_family_expert_reliability_gate_predictions: Path
    bioemu_residue_family_atlas_local_expert_probe: Path
    bioemu_residue_family_atlas_local_expert_predictions: Path
    bioemu_round_conformer_landscape_audit: Path
    bioemu_detached_conformer_probe_manifest: Path
    bioemu_detached_conformer_probe_inputs: Path
    bioemu_detached_conformer_probe_output_audit: Path
    bioemu_detached_conformer_ensemble_quality: Path
    bioemu_detached_conformer_family_quality_guard: Path
    bioemu_detached_support_expansion_action_plan: Path
    bioemu_training_artifact_consistency_audit: Path
    bioemu_detached_conformer_ensemble_pca: Path
    bioemu_detached_conformer_nmr_coupling: Path
    bioemu_detached_conformer_residue_shift_coupling: Path
    bioemu_cprime_chart_calibration_probe: Path
    bioemu_cprime_chart_calibrated_predictions: Path
    hn_regime_predictions: Path
    hn_evidence_graph_residuals: Path
    candidate_free_state_tokens: Path
    forward_residual_predictions: Path
    residue_atom_forward_residuals: Path
    rci_vs_ensemble_profile: Path
    annotated_representative_pdb: Path
    sidecar_summary: Path | None
    materialization_summary: Path | None


def artifact_paths(
    run_dir: str | Path,
    integrated_root: str | Path | None = None,
) -> MonitorArtifactPaths:
    """Return all read-only paths used by the monitor.

    Args:
        run_dir: Training output directory.
        integrated_root: Optional integrated workspace root containing teacher
            and UCBShift summaries.

    Returns:
        Resolved path container for the monitor.
    """

    run_dir_path = Path(run_dir).expanduser().resolve()
    reports_dir = run_dir_path / "reports"
    arrays_dir = reports_dir / "arrays"
    figures_dir = reports_dir / "figures"
    structures_dir = reports_dir / "structures"

    integrated_path = (
        None if integrated_root is None else Path(integrated_root).expanduser()
    )
    return MonitorArtifactPaths(
        run_dir=run_dir_path,
        history=run_dir_path / "history.json",
        summary=run_dir_path / "summary.json",
        benchmark_report=run_dir_path / "benchmark_report.json",
        chemical_shift_baseline_report=(
            run_dir_path / "chemical_shift_baseline_report.json"
        ),
        chemical_shift_calibration_report=(
            run_dir_path / "chemical_shift_calibration.json"
        ),
        learnability_report=reports_dir / "metrics" / "learnability_report.json",
        observable_adapter_report=(
            reports_dir / "metrics" / "observable_adapter_report.json"
        ),
        multi_observable_support_report=(
            reports_dir / "metrics" / "multi_observable_support_ceiling_report.json"
        ),
        best_preview_report=reports_dir / "metrics" / "best_preview_report.json",
        moment_oracle_report=reports_dir / "metrics" / "moment_oracle_report.json",
        moment_consistency_report=(
            reports_dir / "metrics" / "moment_consistency_report.json"
        ),
        ccc_decomposition_report=(
            reports_dir / "metrics" / "ccc_decomposition_report.json"
        ),
        entry_family_calibration_report=(
            reports_dir / "metrics" / "entry_family_calibration_report.json"
        ),
        residue_numbering_alignment_report=(
            reports_dir / "metrics" / "residue_numbering_alignment_report.json"
        ),
        candidate_free_report=reports_dir / "metrics" / "candidate_free_report.json",
        candidate_free_physics_risk_report=(
            reports_dir / "metrics" / "candidate_free_physics_risk_report.json"
        ),
        bioemu_latent_nmr_report=(
            reports_dir / "metrics" / "bioemu_latent_nmr_report.json"
        ),
        hn95_report=reports_dir / "metrics" / "hn95_report.json",
        uncertainty_report=run_dir_path / "uncertainty_report.json",
        figures_dir=figures_dir,
        structures_dir=structures_dir,
        arrays_dir=arrays_dir,
        projection_coordinates=arrays_dir / "projection_coordinates.parquet",
        ensemble_states=arrays_dir / "ensemble_states.parquet",
        conformer_state_assignments=(
            arrays_dir / "conformer_state_assignments.parquet"
        ),
        chemical_shift_posteriors=arrays_dir / "chemical_shift_posteriors.parquet",
        secondary_shift_posteriors=arrays_dir / "secondary_shift_posteriors.parquet",
        secondary_shift_targets=arrays_dir / "secondary_shift_targets.parquet",
        chemical_shift_joint_posteriors=(
            arrays_dir / "chemical_shift_joint_posteriors.parquet"
        ),
        chemical_shift_predictions_calibrated=(
            arrays_dir / "chemical_shift_predictions_calibrated.parquet"
        ),
        ccc_support_oracle=arrays_dir / "ccc_support_oracle.parquet",
        ccc_oracle_weights=arrays_dir / "ccc_oracle_weights.parquet",
        observable_adapter_summary=(arrays_dir / "observable_adapter_summary.parquet"),
        observable_oracle_weights=arrays_dir / "observable_oracle_weights.parquet",
        moment_predictions=arrays_dir / "moment_predictions.parquet",
        moment_oracle_predictions=arrays_dir / "moment_oracle_predictions.parquet",
        moment_oracle_weights=arrays_dir / "moment_oracle_weights.parquet",
        moment_oracle_state_occupancy=(
            arrays_dir / "moment_oracle_state_occupancy.parquet"
        ),
        nmr_structural_features=arrays_dir / "nmr_structural_features.parquet",
        measure_aware_targets=arrays_dir / "measure_aware_targets.parquet",
        joint_nmr_posteriors=arrays_dir / "joint_nmr_posteriors.parquet",
        posterior_state_tokens=arrays_dir / "posterior_state_tokens.parquet",
        posterior_latent_samples=arrays_dir / "posterior_latent_samples.parquet",
        candidate_free_predictions=arrays_dir / "candidate_free_predictions.parquet",
        candidate_free_masked_predictions=(
            arrays_dir / "candidate_free_masked_predictions.parquet"
        ),
        candidate_free_repaired_targets=(
            arrays_dir / "candidate_free_repaired_targets.parquet"
        ),
        candidate_free_hn_physics_risk=(
            arrays_dir / "candidate_free_hn_physics_risk.parquet"
        ),
        candidate_free_hn_outlier_rows=(
            arrays_dir / "candidate_free_hn_outlier_rows.parquet"
        ),
        candidate_free_hn_entry_bottlenecks=(
            arrays_dir / "candidate_free_hn_entry_bottlenecks.parquet"
        ),
        candidate_free_hn_strategy_audit=(
            arrays_dir / "candidate_free_hn_strategy_audit.parquet"
        ),
        candidate_free_hn_ceiling_policy=(
            arrays_dir / "candidate_free_hn_ceiling_policy.parquet"
        ),
        candidate_free_hn_structure_proxy_audit=(
            arrays_dir / "candidate_free_hn_structure_proxy_audit.parquet"
        ),
        bioemu_latent_nmr_predictions=(
            arrays_dir / "bioemu_latent_nmr_predictions.parquet"
        ),
        bioemu_latent_nmr_masked_predictions=(
            arrays_dir / "bioemu_latent_nmr_masked_predictions.parquet"
        ),
        bioemu_live_epoch_progress=arrays_dir / "bioemu_live_epoch_progress.json",
        bioemu_latent_sample_weights=(
            arrays_dir / "bioemu_latent_sample_weights.parquet"
        ),
        bioemu_latent_atlas_coordinates=(
            arrays_dir / "bioemu_latent_atlas_coordinates.parquet"
        ),
        bioemu_posterior_moments=arrays_dir / "bioemu_posterior_moments.parquet",
        bioemu_posterior_bridge_audit=(
            arrays_dir / "bioemu_posterior_bridge_audit.parquet"
        ),
        bioemu_ucbshift_cnnls_teacher_weights=(
            arrays_dir / "bioemu_ucbshift_cnnls_teacher_weights.parquet"
        ),
        bioemu_ucbshift_cnnls_teacher_predictions=(
            arrays_dir / "bioemu_ucbshift_cnnls_teacher_predictions.parquet"
        ),
        bioemu_ucbshift_cnnls_teacher_audit=(
            arrays_dir / "bioemu_ucbshift_cnnls_teacher_audit.parquet"
        ),
        bioemu_ucbshift_cnnls_teacher_skipped=(
            arrays_dir / "bioemu_ucbshift_cnnls_teacher_skipped.parquet"
        ),
        bioemu_family_conflict_audit=(
            arrays_dir / "bioemu_family_conflict_audit.parquet"
        ),
        bioemu_physics_atlas_coordinates=(
            arrays_dir / "bioemu_physics_atlas_coordinates.parquet"
        ),
        bioemu_family_tangent_adapter_audit=(
            arrays_dir / "bioemu_family_tangent_adapter_audit.parquet"
        ),
        bioemu_hn_cprime_direction_conflict_audit=(
            arrays_dir / "bioemu_hn_cprime_direction_conflict_audit.parquet"
        ),
        bioemu_ring_physics_audit=(
            arrays_dir / "bioemu_ring_physics_audit.parquet"
        ),
        bioemu_carbonyl_backbone_audit=(
            arrays_dir / "bioemu_carbonyl_backbone_audit.parquet"
        ),
        bioemu_ucbshift_factor_atlas_audit=(
            arrays_dir / "bioemu_ucbshift_factor_atlas_audit.parquet"
        ),
        bioemu_mechanism_signed_direction_audit=(
            arrays_dir / "bioemu_mechanism_signed_direction_audit.parquet"
        ),
        bioemu_rare_regime_atlas=(
            arrays_dir / "bioemu_rare_regime_atlas.parquet"
        ),
        bioemu_rare_regime_summary=(
            arrays_dir / "bioemu_rare_regime_summary.parquet"
        ),
        bioemu_rare_regime_acceptance=(
            arrays_dir / "bioemu_rare_regime_acceptance.parquet"
        ),
        bioemu_family_chart_calibration_probe=(
            arrays_dir / "bioemu_family_chart_calibration_probe.parquet"
        ),
        bioemu_family_chart_calibrated_predictions=(
            arrays_dir / "bioemu_family_chart_calibrated_predictions.parquet"
        ),
        bioemu_family_affine_calibration_probe=(
            arrays_dir / "bioemu_family_affine_calibration_probe.parquet"
        ),
        bioemu_family_affine_calibrated_predictions=(
            arrays_dir / "bioemu_family_affine_calibrated_predictions.parquet"
        ),
        bioemu_family_calibration_strategy_probe=(
            arrays_dir / "bioemu_family_calibration_strategy_probe.parquet"
        ),
        bioemu_family_calibration_strategy_metrics=(
            arrays_dir / "bioemu_family_calibration_strategy_metrics.parquet"
        ),
        bioemu_recommended_family_metrics=(
            arrays_dir / "bioemu_recommended_family_metrics.parquet"
        ),
        bioemu_family_calibration_strategy_predictions=(
            arrays_dir / "bioemu_family_calibration_strategy_predictions.parquet"
        ),
        bioemu_family_calibration_strategy_residue_audit=(
            arrays_dir / "bioemu_family_calibration_strategy_residue_audit.parquet"
        ),
        bioemu_family_calibration_strategy_residue_type_audit=(
            arrays_dir / "bioemu_family_calibration_strategy_residue_type_audit.parquet"
        ),
        bioemu_family_calibration_strategy_residue_type_guard_probe=(
            arrays_dir
            / "bioemu_family_calibration_strategy_residue_type_guard_probe.parquet"
        ),
        bioemu_family_calibration_strategy_residue_type_guard_predictions=(
            arrays_dir
            / "bioemu_family_calibration_strategy_residue_type_guard_predictions.parquet"
        ),
        bioemu_proline_n_rare_regime_audit=(
            arrays_dir / "bioemu_proline_n_rare_regime_audit.parquet"
        ),
        bioemu_proline_n_expert_probe=(
            arrays_dir / "bioemu_proline_n_expert_probe.parquet"
        ),
        bioemu_proline_n_expert_predictions=(
            arrays_dir / "bioemu_proline_n_expert_predictions.parquet"
        ),
        bioemu_proline_n_mobility_expert_probe=(
            arrays_dir / "bioemu_proline_n_mobility_expert_probe.parquet"
        ),
        bioemu_proline_n_mobility_expert_split_guard=(
            arrays_dir / "bioemu_proline_n_mobility_expert_split_guard.parquet"
        ),
        bioemu_proline_n_mobility_expert_predictions=(
            arrays_dir / "bioemu_proline_n_mobility_expert_predictions.parquet"
        ),
        bioemu_proline_n_mobility_entity_holdout=(
            arrays_dir / "bioemu_proline_n_mobility_entity_holdout.parquet"
        ),
        bioemu_proline_n_mobility_entity_holdout_predictions=(
            arrays_dir
            / "bioemu_proline_n_mobility_entity_holdout_predictions.parquet"
        ),
        bioemu_residue_family_expert_probe=(
            arrays_dir / "bioemu_residue_family_expert_probe.parquet"
        ),
        bioemu_residue_family_expert_predictions=(
            arrays_dir / "bioemu_residue_family_expert_predictions.parquet"
        ),
        bioemu_residue_family_expert_policy_probe=(
            arrays_dir / "bioemu_residue_family_expert_policy_probe.parquet"
        ),
        bioemu_residue_family_expert_policy_predictions=(
            arrays_dir / "bioemu_residue_family_expert_policy_predictions.parquet"
        ),
        bioemu_residue_family_expert_reliability_gate_probe=(
            arrays_dir / "bioemu_residue_family_expert_reliability_gate_probe.parquet"
        ),
        bioemu_residue_family_expert_reliability_gate_predictions=(
            arrays_dir
            / "bioemu_residue_family_expert_reliability_gate_predictions.parquet"
        ),
        bioemu_residue_family_atlas_local_expert_probe=(
            arrays_dir / "bioemu_residue_family_atlas_local_expert_probe.parquet"
        ),
        bioemu_residue_family_atlas_local_expert_predictions=(
            arrays_dir
            / "bioemu_residue_family_atlas_local_expert_predictions.parquet"
        ),
        bioemu_round_conformer_landscape_audit=(
            arrays_dir / "bioemu_round_conformer_landscape_audit.parquet"
        ),
        bioemu_detached_conformer_probe_manifest=(
            arrays_dir / "bioemu_detached_conformer_probe_manifest.parquet"
        ),
        bioemu_detached_conformer_probe_inputs=(
            arrays_dir / "bioemu_detached_conformer_probe_inputs.parquet"
        ),
        bioemu_detached_conformer_probe_output_audit=(
            arrays_dir / "bioemu_detached_conformer_probe_output_audit.parquet"
        ),
        bioemu_detached_conformer_ensemble_quality=(
            arrays_dir / "bioemu_detached_conformer_ensemble_quality.parquet"
        ),
        bioemu_detached_conformer_family_quality_guard=(
            arrays_dir / "bioemu_detached_conformer_family_quality_guard.parquet"
        ),
        bioemu_detached_support_expansion_action_plan=(
            arrays_dir / "bioemu_detached_support_expansion_action_plan.parquet"
        ),
        bioemu_training_artifact_consistency_audit=(
            arrays_dir / "bioemu_training_artifact_consistency_audit.parquet"
        ),
        bioemu_detached_conformer_ensemble_pca=(
            arrays_dir / "bioemu_detached_conformer_ensemble_pca.parquet"
        ),
        bioemu_detached_conformer_nmr_coupling=(
            arrays_dir / "bioemu_detached_conformer_nmr_coupling.parquet"
        ),
        bioemu_detached_conformer_residue_shift_coupling=(
            arrays_dir / "bioemu_detached_conformer_residue_shift_coupling.parquet"
        ),
        bioemu_cprime_chart_calibration_probe=(
            arrays_dir / "bioemu_cprime_chart_calibration_probe.parquet"
        ),
        bioemu_cprime_chart_calibrated_predictions=(
            arrays_dir / "bioemu_cprime_chart_calibrated_predictions.parquet"
        ),
        hn_regime_predictions=arrays_dir / "hn_regime_predictions.parquet",
        hn_evidence_graph_residuals=(
            arrays_dir / "hn_evidence_graph_residuals.parquet"
        ),
        candidate_free_state_tokens=arrays_dir / "candidate_free_state_tokens.parquet",
        forward_residual_predictions=(
            arrays_dir / "forward_residual_predictions.parquet"
        ),
        residue_atom_forward_residuals=(
            arrays_dir / "residue_atom_forward_residuals.parquet"
        ),
        rci_vs_ensemble_profile=arrays_dir / "rci_vs_ensemble_profile.parquet",
        annotated_representative_pdb=(structures_dir / "annotated_representative.pdb"),
        sidecar_summary=(
            None
            if integrated_path is None
            else integrated_path / "ucbshift" / "sidecar_generation_summary.json"
        ),
        materialization_summary=(
            None
            if integrated_path is None
            else integrated_path / "teachers" / "materialization_summary.json"
        ),
    )


def read_json_cached(
    path: str | Path,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any] | None:
    """Read a JSON artifact with partial-write fallback.

    Args:
        path: JSON file path.
        cache: Mutable cache keyed by path. When a file is missing or currently
            contains invalid JSON, the last successfully parsed payload is
            returned.

    Returns:
        Parsed JSON payload, the last cached payload, or ``None``.
    """

    path_obj = Path(path)
    cache_key = f"{JSON_CACHE_KEY_PREFIX}{path_obj}"
    if not path_obj.exists():
        return None if cache is None else cache.get(cache_key)
    try:
        payload = json.loads(path_obj.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None if cache is None else cache.get(cache_key)
    if cache is not None:
        cache[cache_key] = payload
    return payload


def load_optional_parquet(path: str | Path) -> pd.DataFrame:
    """Load an optional Parquet artifact.

    Args:
        path: Parquet path.

    Returns:
        DataFrame when the file exists and can be read, otherwise an empty
        DataFrame.
    """

    path_obj = Path(path)
    tsv_path = path_obj.with_suffix(".tsv")
    if not path_obj.exists() and not tsv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path_obj)
    except Exception:
        if not tsv_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(tsv_path, sep="\t")
        except Exception:
            return pd.DataFrame()


def flatten_history(history: Any) -> pd.DataFrame:
    """Flatten training history into a plot-ready metric table.

    Args:
        history: Parsed ``history.json`` payload.

    Returns:
        DataFrame with one row per metric/loss value and epoch.
    """

    if not isinstance(history, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for record in history:
        if not isinstance(record, dict):
            continue
        epoch = _safe_int(record.get("epoch"))
        if epoch is None:
            continue
        for split in ["train", "val"]:
            split_payload = record.get(split, {})
            if not isinstance(split_payload, dict):
                continue
            rows.extend(_flatten_metric_rows(epoch, split, split_payload))
        rows.extend(_flatten_checkpoint_record(epoch, record.get("checkpoint_metric")))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values(
        ["epoch", "split", "metric_name", "aggregation"],
        kind="stable",
    )
    return frame.reset_index(drop=True)


def load_monitor_snapshot(
    run_dir: str | Path,
    integrated_root: str | Path | None = None,
    json_cache: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load one read-only monitor snapshot for a training run.

    Args:
        run_dir: Training output directory.
        integrated_root: Optional integrated workspace root.
        json_cache: Mutable cache for partially written JSON artifacts.
        job_id: Optional Slurm job id for status lookup.

    Returns:
        Dictionary of parsed reports, frames, artifact paths, and job status.
    """

    paths = artifact_paths(run_dir=run_dir, integrated_root=integrated_root)
    history = read_json_cached(paths.history, json_cache) or []
    summary = read_json_cached(paths.summary, json_cache) or {}
    benchmark_report = read_json_cached(paths.benchmark_report, json_cache) or {}
    baseline_report = (
        read_json_cached(paths.chemical_shift_baseline_report, json_cache) or {}
    )
    calibration_report = (
        read_json_cached(paths.chemical_shift_calibration_report, json_cache) or {}
    )
    learnability_report = read_json_cached(paths.learnability_report, json_cache) or {}
    observable_adapter_report = (
        read_json_cached(paths.observable_adapter_report, json_cache)
        or read_json_cached(
            paths.run_dir / "observable_adapter_report.json", json_cache
        )
        or {}
    )
    multi_observable_support_report = (
        read_json_cached(paths.multi_observable_support_report, json_cache)
        or read_json_cached(
            paths.run_dir / "multi_observable_support_ceiling_report.json",
            json_cache,
        )
        or {}
    )
    best_preview_report = (
        read_json_cached(paths.best_preview_report, json_cache)
        or read_json_cached(paths.run_dir / "best_preview_report.json", json_cache)
        or {}
    )
    moment_oracle_report = (
        read_json_cached(paths.moment_oracle_report, json_cache)
        or read_json_cached(paths.run_dir / "moment_oracle_report.json", json_cache)
        or {}
    )
    moment_consistency_report = (
        read_json_cached(paths.moment_consistency_report, json_cache) or {}
    )
    ccc_decomposition_report = (
        read_json_cached(paths.ccc_decomposition_report, json_cache) or {}
    )
    entry_family_calibration_report = (
        read_json_cached(paths.entry_family_calibration_report, json_cache) or {}
    )
    residue_numbering_alignment_report = (
        read_json_cached(paths.residue_numbering_alignment_report, json_cache) or {}
    )
    candidate_free_report = (
        read_json_cached(paths.candidate_free_report, json_cache) or {}
    )
    candidate_free_physics_risk_report = (
        read_json_cached(paths.candidate_free_physics_risk_report, json_cache) or {}
    )
    bioemu_latent_nmr_report = (
        read_json_cached(paths.bioemu_latent_nmr_report, json_cache)
        or read_json_cached(paths.run_dir / "bioemu_latent_nmr_report.json", json_cache)
        or {}
    )
    bioemu_live_epoch_progress = (
        read_json_cached(paths.bioemu_live_epoch_progress, json_cache) or {}
    )
    hn95_report = read_json_cached(paths.hn95_report, json_cache) or {}
    uncertainty_report = read_json_cached(paths.uncertainty_report, json_cache) or {}
    sidecar_summary = (
        None
        if paths.sidecar_summary is None
        else read_json_cached(paths.sidecar_summary, json_cache)
    )
    materialization_summary = (
        None
        if paths.materialization_summary is None
        else read_json_cached(paths.materialization_summary, json_cache)
    )

    metric_frame = flatten_history(history)
    return {
        "paths": paths,
        "history": history,
        "summary": summary,
        "benchmark_report": benchmark_report,
        "chemical_shift_baseline_report": baseline_report,
        "chemical_shift_calibration_report": calibration_report,
        "learnability_report": learnability_report,
        "observable_adapter_report": observable_adapter_report,
        "multi_observable_support_report": multi_observable_support_report,
        "best_preview_report": best_preview_report,
        "moment_oracle_report": moment_oracle_report,
        "moment_consistency_report": moment_consistency_report,
        "ccc_decomposition_report": ccc_decomposition_report,
        "entry_family_calibration_report": entry_family_calibration_report,
        "residue_numbering_alignment_report": residue_numbering_alignment_report,
        "candidate_free_report": candidate_free_report,
        "candidate_free_physics_risk_report": candidate_free_physics_risk_report,
        "bioemu_latent_nmr_report": bioemu_latent_nmr_report,
        "bioemu_live_epoch_progress": bioemu_live_epoch_progress,
        "hn95_report": hn95_report,
        "uncertainty_report": uncertainty_report,
        "sidecar_summary": sidecar_summary,
        "materialization_summary": materialization_summary,
        "metric_frame": metric_frame,
        "projection_coordinates": load_optional_parquet(paths.projection_coordinates),
        "ensemble_states": load_optional_parquet(paths.ensemble_states),
        "conformer_state_assignments": load_optional_parquet(
            paths.conformer_state_assignments
        ),
        "chemical_shift_posteriors": load_optional_parquet(
            paths.chemical_shift_posteriors
        ),
        "secondary_shift_posteriors": load_optional_parquet(
            paths.secondary_shift_posteriors
        ),
        "secondary_shift_targets": load_optional_parquet(paths.secondary_shift_targets),
        "chemical_shift_joint_posteriors": load_optional_parquet(
            paths.chemical_shift_joint_posteriors
        ),
        "chemical_shift_predictions_calibrated": load_optional_parquet(
            paths.chemical_shift_predictions_calibrated
        ),
        "ccc_support_oracle": load_optional_parquet(paths.ccc_support_oracle),
        "ccc_oracle_weights": load_optional_parquet(paths.ccc_oracle_weights),
        "observable_adapter_summary": load_optional_parquet(
            paths.observable_adapter_summary
        ),
        "observable_oracle_weights": load_optional_parquet(
            paths.observable_oracle_weights
        ),
        "moment_predictions": load_optional_parquet(paths.moment_predictions),
        "moment_oracle_predictions": load_optional_parquet(
            paths.moment_oracle_predictions
        ),
        "moment_oracle_weights": load_optional_parquet(paths.moment_oracle_weights),
        "moment_oracle_state_occupancy": load_optional_parquet(
            paths.moment_oracle_state_occupancy
        ),
        "nmr_structural_features": load_optional_parquet(paths.nmr_structural_features),
        "measure_aware_targets": load_optional_parquet(paths.measure_aware_targets),
        "joint_nmr_posteriors": load_optional_parquet(paths.joint_nmr_posteriors),
        "posterior_state_tokens": load_optional_parquet(paths.posterior_state_tokens),
        "posterior_latent_samples": load_optional_parquet(
            paths.posterior_latent_samples
        ),
        "candidate_free_predictions": load_optional_parquet(
            paths.candidate_free_predictions
        ),
        "candidate_free_masked_predictions": load_optional_parquet(
            paths.candidate_free_masked_predictions
        ),
        "candidate_free_repaired_targets": load_optional_parquet(
            paths.candidate_free_repaired_targets
        ),
        "candidate_free_hn_physics_risk": load_optional_parquet(
            paths.candidate_free_hn_physics_risk
        ),
        "candidate_free_hn_outlier_rows": load_optional_parquet(
            paths.candidate_free_hn_outlier_rows
        ),
        "candidate_free_hn_entry_bottlenecks": load_optional_parquet(
            paths.candidate_free_hn_entry_bottlenecks
        ),
        "candidate_free_hn_strategy_audit": load_optional_parquet(
            paths.candidate_free_hn_strategy_audit
        ),
        "candidate_free_hn_ceiling_policy": load_optional_parquet(
            paths.candidate_free_hn_ceiling_policy
        ),
        "candidate_free_hn_structure_proxy_audit": load_optional_parquet(
            paths.candidate_free_hn_structure_proxy_audit
        ),
        "bioemu_latent_nmr_predictions": load_optional_parquet(
            paths.bioemu_latent_nmr_predictions
        ),
        "bioemu_latent_nmr_masked_predictions": load_optional_parquet(
            paths.bioemu_latent_nmr_masked_predictions
        ),
        "bioemu_latent_sample_weights": load_optional_parquet(
            paths.bioemu_latent_sample_weights
        ),
        "bioemu_latent_atlas_coordinates": load_optional_parquet(
            paths.bioemu_latent_atlas_coordinates
        ),
        "bioemu_posterior_moments": load_optional_parquet(
            paths.bioemu_posterior_moments
        ),
        "bioemu_posterior_bridge_audit": load_optional_parquet(
            paths.bioemu_posterior_bridge_audit
        ),
        "bioemu_ucbshift_cnnls_teacher_weights": load_optional_parquet(
            paths.bioemu_ucbshift_cnnls_teacher_weights
        ),
        "bioemu_ucbshift_cnnls_teacher_predictions": load_optional_parquet(
            paths.bioemu_ucbshift_cnnls_teacher_predictions
        ),
        "bioemu_ucbshift_cnnls_teacher_audit": load_optional_parquet(
            paths.bioemu_ucbshift_cnnls_teacher_audit
        ),
        "bioemu_ucbshift_cnnls_teacher_skipped": load_optional_parquet(
            paths.bioemu_ucbshift_cnnls_teacher_skipped
        ),
        "bioemu_family_conflict_audit": load_optional_parquet(
            paths.bioemu_family_conflict_audit
        ),
        "bioemu_physics_atlas_coordinates": load_optional_parquet(
            paths.bioemu_physics_atlas_coordinates
        ),
        "bioemu_family_tangent_adapter_audit": load_optional_parquet(
            paths.bioemu_family_tangent_adapter_audit
        ),
        "bioemu_hn_cprime_direction_conflict_audit": load_optional_parquet(
            paths.bioemu_hn_cprime_direction_conflict_audit
        ),
        "bioemu_ring_physics_audit": load_optional_parquet(
            paths.bioemu_ring_physics_audit
        ),
        "bioemu_carbonyl_backbone_audit": load_optional_parquet(
            paths.bioemu_carbonyl_backbone_audit
        ),
        "bioemu_ucbshift_factor_atlas_audit": load_optional_parquet(
            paths.bioemu_ucbshift_factor_atlas_audit
        ),
        "bioemu_mechanism_signed_direction_audit": load_optional_parquet(
            paths.bioemu_mechanism_signed_direction_audit
        ),
        "bioemu_rare_regime_atlas": load_optional_parquet(
            paths.bioemu_rare_regime_atlas
        ),
        "bioemu_rare_regime_summary": load_optional_parquet(
            paths.bioemu_rare_regime_summary
        ),
        "bioemu_rare_regime_acceptance": load_optional_parquet(
            paths.bioemu_rare_regime_acceptance
        ),
        "bioemu_family_chart_calibration_probe": load_optional_parquet(
            paths.bioemu_family_chart_calibration_probe
        ),
        "bioemu_family_chart_calibrated_predictions": load_optional_parquet(
            paths.bioemu_family_chart_calibrated_predictions
        ),
        "bioemu_family_affine_calibration_probe": load_optional_parquet(
            paths.bioemu_family_affine_calibration_probe
        ),
        "bioemu_family_affine_calibrated_predictions": load_optional_parquet(
            paths.bioemu_family_affine_calibrated_predictions
        ),
        "bioemu_family_calibration_strategy_probe": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_probe
        ),
        "bioemu_family_calibration_strategy_metrics": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_metrics
        ),
        "bioemu_recommended_family_metrics": load_optional_parquet(
            paths.bioemu_recommended_family_metrics
        ),
        "bioemu_family_calibration_strategy_predictions": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_predictions
        ),
        "bioemu_family_calibration_strategy_residue_audit": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_residue_audit
        ),
        "bioemu_family_calibration_strategy_residue_type_audit": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_residue_type_audit
        ),
        "bioemu_family_calibration_strategy_residue_type_guard_probe": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_residue_type_guard_probe
        ),
        "bioemu_family_calibration_strategy_residue_type_guard_predictions": load_optional_parquet(
            paths.bioemu_family_calibration_strategy_residue_type_guard_predictions
        ),
        "bioemu_proline_n_rare_regime_audit": load_optional_parquet(
            paths.bioemu_proline_n_rare_regime_audit
        ),
        "bioemu_proline_n_expert_probe": load_optional_parquet(
            paths.bioemu_proline_n_expert_probe
        ),
        "bioemu_proline_n_expert_predictions": load_optional_parquet(
            paths.bioemu_proline_n_expert_predictions
        ),
        "bioemu_proline_n_mobility_expert_probe": load_optional_parquet(
            paths.bioemu_proline_n_mobility_expert_probe
        ),
        "bioemu_proline_n_mobility_expert_split_guard": load_optional_parquet(
            paths.bioemu_proline_n_mobility_expert_split_guard
        ),
        "bioemu_proline_n_mobility_expert_predictions": load_optional_parquet(
            paths.bioemu_proline_n_mobility_expert_predictions
        ),
        "bioemu_proline_n_mobility_entity_holdout": load_optional_parquet(
            paths.bioemu_proline_n_mobility_entity_holdout
        ),
        "bioemu_proline_n_mobility_entity_holdout_predictions": load_optional_parquet(
            paths.bioemu_proline_n_mobility_entity_holdout_predictions
        ),
        "bioemu_residue_family_expert_probe": load_optional_parquet(
            paths.bioemu_residue_family_expert_probe
        ),
        "bioemu_residue_family_expert_predictions": load_optional_parquet(
            paths.bioemu_residue_family_expert_predictions
        ),
        "bioemu_residue_family_expert_policy_probe": load_optional_parquet(
            paths.bioemu_residue_family_expert_policy_probe
        ),
        "bioemu_residue_family_expert_policy_predictions": load_optional_parquet(
            paths.bioemu_residue_family_expert_policy_predictions
        ),
        "bioemu_residue_family_expert_reliability_gate_probe": load_optional_parquet(
            paths.bioemu_residue_family_expert_reliability_gate_probe
        ),
        "bioemu_residue_family_expert_reliability_gate_predictions": (
            load_optional_parquet(
                paths.bioemu_residue_family_expert_reliability_gate_predictions
            )
        ),
        "bioemu_residue_family_atlas_local_expert_probe": load_optional_parquet(
            paths.bioemu_residue_family_atlas_local_expert_probe
        ),
        "bioemu_residue_family_atlas_local_expert_predictions": (
            load_optional_parquet(
                paths.bioemu_residue_family_atlas_local_expert_predictions
            )
        ),
        "bioemu_round_conformer_landscape_audit": load_optional_parquet(
            paths.bioemu_round_conformer_landscape_audit
        ),
        "bioemu_detached_conformer_probe_manifest": load_optional_parquet(
            paths.bioemu_detached_conformer_probe_manifest
        ),
        "bioemu_detached_conformer_probe_inputs": load_optional_parquet(
            paths.bioemu_detached_conformer_probe_inputs
        ),
        "bioemu_detached_conformer_probe_output_audit": load_optional_parquet(
            paths.bioemu_detached_conformer_probe_output_audit
        ),
        "bioemu_detached_conformer_ensemble_quality": load_optional_parquet(
            paths.bioemu_detached_conformer_ensemble_quality
        ),
        "bioemu_detached_conformer_family_quality_guard": load_optional_parquet(
            paths.bioemu_detached_conformer_family_quality_guard
        ),
        "bioemu_detached_support_expansion_action_plan": load_optional_parquet(
            paths.bioemu_detached_support_expansion_action_plan
        ),
        "bioemu_training_artifact_consistency_audit": load_optional_parquet(
            paths.bioemu_training_artifact_consistency_audit
        ),
        "bioemu_detached_conformer_ensemble_pca": load_optional_parquet(
            paths.bioemu_detached_conformer_ensemble_pca
        ),
        "bioemu_detached_conformer_nmr_coupling": load_optional_parquet(
            paths.bioemu_detached_conformer_nmr_coupling
        ),
        "bioemu_detached_conformer_residue_shift_coupling": load_optional_parquet(
            paths.bioemu_detached_conformer_residue_shift_coupling
        ),
        "bioemu_cprime_chart_calibration_probe": load_optional_parquet(
            paths.bioemu_cprime_chart_calibration_probe
        ),
        "bioemu_cprime_chart_calibrated_predictions": load_optional_parquet(
            paths.bioemu_cprime_chart_calibrated_predictions
        ),
        "hn_regime_predictions": load_optional_parquet(paths.hn_regime_predictions),
        "hn_evidence_graph_residuals": load_optional_parquet(
            paths.hn_evidence_graph_residuals
        ),
        "candidate_free_state_tokens": load_optional_parquet(
            paths.candidate_free_state_tokens
        ),
        "forward_residual_predictions": load_optional_parquet(
            paths.forward_residual_predictions
        ),
        "residue_atom_forward_residuals": load_optional_parquet(
            paths.residue_atom_forward_residuals
        ),
        "rci_vs_ensemble_profile": load_optional_parquet(paths.rci_vs_ensemble_profile),
        "job_status": query_slurm_job(job_id),
        "current_epoch": current_epoch(history, bioemu_live_epoch_progress),
    }


def current_epoch(history: Any, live_progress: Any | None = None) -> int | None:
    """Return the most recent epoch from parsed history."""

    if not isinstance(history, list) or not history:
        if isinstance(live_progress, dict):
            return _safe_int(live_progress.get("epoch"))
        return None
    epochs = [
        _safe_int(record.get("epoch")) for record in history if isinstance(record, dict)
    ]
    epochs = [epoch for epoch in epochs if epoch is not None]
    if epochs:
        return max(epochs)
    if isinstance(live_progress, dict):
        return _safe_int(live_progress.get("epoch"))
    return None


def query_slurm_job(job_id: str | None) -> dict[str, Any] | None:
    """Return minimal Slurm status for one user-owned job id.

    The monitor only queries a specific job id. It never cancels or modifies
    jobs.
    """

    if not job_id:
        return None
    safe_job_id = "".join(ch for ch in str(job_id) if ch.isdigit())
    if not safe_job_id:
        return None
    commands = [
        [
            "squeue",
            "-j",
            safe_job_id,
            "-h",
            "-o",
            "%i|%T|%M|%D|%R|%u|%j",
        ],
        [
            "sacct",
            "-j",
            safe_job_id,
            "--format=JobIDRaw,State,Elapsed,NodeList,User,JobName",
            "--parsable2",
            "--noheader",
        ],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        line = result.stdout.strip().splitlines()[0:1]
        if not line:
            continue
        fields = line[0].split("|")
        if command[0] == "squeue" and len(fields) >= 7:
            return {
                "job_id": fields[0],
                "state": fields[1],
                "elapsed": fields[2],
                "nodes": fields[3],
                "reason_or_node": fields[4],
                "user": fields[5],
                "job_name": fields[6],
                "source": "squeue",
            }
        if command[0] == "sacct" and len(fields) >= 6:
            return {
                "job_id": fields[0],
                "state": fields[1],
                "elapsed": fields[2],
                "reason_or_node": fields[3],
                "user": fields[4],
                "job_name": fields[5],
                "source": "sacct",
            }
    return None


def _flatten_metric_rows(
    epoch: int,
    split: str,
    split_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten metric and loss rows for one split payload."""

    rows: list[dict[str, Any]] = []
    for key in ["loss_rows", "metric_rows"]:
        for row in split_payload.get(key, []) or []:
            if not isinstance(row, dict):
                continue
            value = row.get("value")
            numeric = _safe_float(value)
            if numeric is None:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "split": str(row.get("split", split)),
                    "metric_name": str(row.get("metric_name", "")),
                    "aggregation": str(row.get("aggregation", "macro")),
                    "value": numeric,
                    "eligible_examples": row.get("eligible_examples"),
                    "eligible_measurements": row.get("eligible_measurements"),
                    "tier": row.get("tier"),
                    "source": key,
                }
            )
    return rows


def _flatten_checkpoint_record(
    epoch: int,
    checkpoint_record: Any,
) -> list[dict[str, Any]]:
    """Flatten checkpoint selection payload into metric rows."""

    if not isinstance(checkpoint_record, dict):
        return []
    rows: list[dict[str, Any]] = []
    metric_name = checkpoint_record.get("metric_name")
    value = _safe_float(checkpoint_record.get("value"))
    split = str(checkpoint_record.get("split", "val"))
    if metric_name is not None and value is not None:
        rows.append(
            {
                "epoch": epoch,
                "split": split,
                "metric_name": f"{metric_name}_checkpoint",
                "aggregation": "selection",
                "value": value,
                "eligible_examples": None,
                "eligible_measurements": None,
                "tier": "checkpoint",
                "source": "checkpoint_metric",
            }
        )
    for index, item in enumerate(checkpoint_record.get("comparison_key", []) or []):
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        rows.append(
            {
                "epoch": epoch,
                "split": split,
                "metric_name": f"checkpoint_key_{index}",
                "aggregation": "selection",
                "value": numeric,
                "eligible_examples": None,
                "eligible_measurements": None,
                "tier": "checkpoint",
                "source": "checkpoint_metric",
            }
        )
    return rows


def _safe_int(value: Any) -> int | None:
    """Convert an arbitrary value to int when possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Convert an arbitrary value to finite float when possible."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric
