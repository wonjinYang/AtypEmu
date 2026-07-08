"""Configuration dataclasses for AtypEmu teacher and student training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _int_list_payload(value: Any) -> list[int]:
    """Parse a JSON list or comma-separated string into integers."""

    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    parsed: list[int] = []
    for item in items:
        parsed.append(int(item))
    return parsed


def multiseed_active_set_seed_family_weights_with_defaults(
    config: Any,
) -> dict[str, dict[str, float]]:
    """Expose global active-set family weights as default seed weights."""

    merged: dict[str, dict[str, float]] = {}
    global_weights = getattr(
        config,
        "bioemu_x0_multiseed_active_set_family_weights",
        {},
    )
    if isinstance(global_weights, dict) and global_weights:
        merged["*"] = dict(global_weights)

    seed_weights = getattr(
        config,
        "bioemu_x0_multiseed_active_set_seed_family_weights",
        {},
    )
    if not isinstance(seed_weights, dict):
        return merged

    for key, payload in seed_weights.items():
        if not isinstance(payload, dict):
            continue
        text_key = str(key)
        if text_key in {"*", "default"}:
            current = dict(merged.get(text_key, {}))
            current.update(payload)
            merged[text_key] = current
        else:
            merged[text_key] = dict(payload)
    return merged


@dataclass(slots=True)
class TeacherMaterializationConfig:
    """Configuration for offline teacher materialization."""

    selected_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    selected_sources: list[str] = field(
        default_factory=lambda: ["AF3", "BioEmu", "CALVADOS2"]
    )
    excluded_bmrb_ids: list[str] = field(default_factory=list)
    selected_entity_uids: list[str] = field(default_factory=list)
    selected_bmrb_ids: list[str] = field(default_factory=list)
    artifact_namespace: str | None = None
    chemical_shift_dir_templates: dict[str, str] = field(default_factory=dict)
    chemical_shift_formats: dict[str, str] = field(default_factory=dict)
    reweighting_method: str = "maxent"
    prior_policy: str = "source_balanced"
    lambda_reg: float = 0.01
    adaptive_lambda_ess_floor: float | None = None
    adaptive_lambda_max: float | None = None
    adaptive_lambda_growth: float = 2.0
    adaptive_lambda_max_steps: int = 1
    beta_cs: float = 1.0
    beta_j: float = 1.0
    beta_noe: float = 1.0
    require_chemical_shifts: bool = True
    refresh_outputs: bool = False
    refresh_training_bundle: bool = True
    max_examples: int | None = None
    max_candidates_per_example: int | None = None
    parse_candidate_structures: bool = True
    materialization_worker_count: int = 1
    fail_fast: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to one JSON file."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherMaterializationConfig":
        """Create one configuration object from a dictionary."""
        return cls(
            selected_splits=list(payload.get("selected_splits", ["train", "val"])),
            selected_sources=list(
                payload.get("selected_sources", ["AF3", "BioEmu", "CALVADOS2"])
            ),
            excluded_bmrb_ids=list(payload.get("excluded_bmrb_ids", [])),
            selected_entity_uids=list(payload.get("selected_entity_uids", [])),
            selected_bmrb_ids=list(payload.get("selected_bmrb_ids", [])),
            artifact_namespace=payload.get("artifact_namespace"),
            chemical_shift_dir_templates=dict(
                payload.get("chemical_shift_dir_templates", {})
            ),
            chemical_shift_formats=dict(payload.get("chemical_shift_formats", {})),
            reweighting_method=str(payload.get("reweighting_method", "maxent")),
            prior_policy=str(payload.get("prior_policy", "source_balanced")),
            lambda_reg=float(payload.get("lambda_reg", 0.01)),
            adaptive_lambda_ess_floor=(
                None
                if payload.get("adaptive_lambda_ess_floor") is None
                else float(payload["adaptive_lambda_ess_floor"])
            ),
            adaptive_lambda_max=(
                None
                if payload.get("adaptive_lambda_max") is None
                else float(payload["adaptive_lambda_max"])
            ),
            adaptive_lambda_growth=float(payload.get("adaptive_lambda_growth", 2.0)),
            adaptive_lambda_max_steps=max(
                1, int(payload.get("adaptive_lambda_max_steps", 1) or 1)
            ),
            beta_cs=float(payload.get("beta_cs", 1.0)),
            beta_j=float(payload.get("beta_j", 1.0)),
            beta_noe=float(payload.get("beta_noe", 1.0)),
            require_chemical_shifts=bool(payload.get("require_chemical_shifts", True)),
            refresh_outputs=bool(payload.get("refresh_outputs", False)),
            refresh_training_bundle=bool(payload.get("refresh_training_bundle", True)),
            max_examples=(
                None
                if payload.get("max_examples") is None
                else int(payload["max_examples"])
            ),
            max_candidates_per_example=(
                None
                if payload.get("max_candidates_per_example") is None
                else int(payload["max_candidates_per_example"])
            ),
            parse_candidate_structures=bool(
                payload.get("parse_candidate_structures", True)
            ),
            materialization_worker_count=max(
                1, int(payload.get("materialization_worker_count", 1) or 1)
            ),
            fail_fast=bool(payload.get("fail_fast", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TeacherMaterializationConfig":
        """Load the configuration from one JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(slots=True)
class StudentTrainingConfig:
    """Configuration for the first density-student training loop."""

    train_splits: list[str] = field(default_factory=lambda: ["train"])
    val_splits: list[str] = field(default_factory=lambda: ["val"])
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    checkpoint_metric: str = "teacher_kl_macro"
    checkpoint_tie_breakers: list[str] = field(
        default_factory=lambda: [
            "teacher_js_macro",
            "cs_rmse_z_macro",
            "noe_violation_rate_macro",
        ]
    )
    report_micro_metrics: bool = True
    epochs: int = 25
    learning_rate: float = 1e-3
    min_learning_rate: float = 0.0
    learning_rate_schedule: str = "constant"
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    weight_kl_weight: float = 1.0
    chemical_shift_loss_weight: float = 0.5
    chemical_shift_loss_mode: str = "mse"
    chemical_shift_huber_delta: float = 5.0
    chemical_shift_ccc_loss_weight: float = 0.0
    chemical_shift_family_balance: bool = False
    chemical_shift_family_weights: dict[str, float] = field(default_factory=dict)
    training_task: str | None = None
    enable_ccc_support_oracle: bool = False
    ccc_oracle_solver: str = "projected_simplex"
    ccc_oracle_steps: int = 80
    ccc_oracle_distillation_weight: float = 0.0
    ccc_geometry_loss_weight: float = 0.0
    posterior_evidence_kl_weight: float = 0.0
    evidence_temperature: float = 1.0
    ccc_oracle_kl_weight: float = 0.0
    ccc_oracle_temperature: float = 0.5
    enable_observable_adapters: bool = False
    observable_oracle_distillation_weight: float = 0.0
    observable_oracle_temperature: float = 0.5
    observable_channel_weights: dict[str, float] = field(
        default_factory=lambda: {
            "chemical_shifts": 1.0,
            "j_couplings": 0.10,
            "noe_restraints": 0.10,
            "saxs": 0.0,
        }
    )
    enable_moment_head: bool = False
    enable_posterior_state_tokens: bool = False
    posterior_state_count: int = 16
    enable_latent_posterior_flow: bool = False
    latent_flow_steps: int = 4
    moment_oracle_distillation_weight: float = 0.0
    moment_sample_consistency_weight: float = 0.0
    state_occupancy_distillation_weight: float = 0.0
    state_diversity_weight: float = 0.0
    enable_family_specific_moment_head: bool = False
    enable_nmr_structural_features: bool = False
    enable_local_evidence_context: bool = False
    enable_same_residue_evidence_context: bool = False
    same_residue_evidence_target_families: list[str] = field(default_factory=list)
    enable_target_set_evidence_encoder: bool = False
    enable_target_evidence_token: bool = False
    target_set_encoder_layers: int = 1
    target_set_encoder_heads: int = 4
    target_set_encoder_max_targets: int = 1024
    target_set_context_gate_init: float = 1.0
    target_set_encoder_masked_only: bool = False
    target_set_encoder_target_families: list[str] = field(default_factory=list)
    target_set_encoder_evidence_families: list[str] = field(default_factory=list)
    enable_residue_grid_evidence_encoder: bool = False
    residue_grid_encoder_layers: int = 1
    residue_grid_encoder_heads: int = 4
    residue_grid_context_gate_init: float = 0.2
    enable_residue_grid_family_gates: bool = False
    residue_grid_encoder_masked_only: bool = True
    residue_grid_encoder_target_families: list[str] = field(default_factory=list)
    local_evidence_window: int = 3
    enable_residue_anchor_evidence_context: bool = False
    residue_anchor_evidence_offsets: list[int] = field(
        default_factory=lambda: [-1, 0, 1]
    )
    residue_anchor_evidence_target_families: list[str] = field(default_factory=list)
    enable_backbone_evidence_context: bool = False
    backbone_evidence_target_families: list[str] = field(default_factory=list)
    enable_all_family_evidence_affine_calibration: bool = False
    evidence_affine_calibration_families: list[str] = field(default_factory=list)
    evidence_affine_scale_only_families: list[str] = field(default_factory=list)
    enable_hn_variance_calibration: bool = False
    enable_cprime_robust_likelihood: bool = False
    state_entropy_floor: float = 0.0
    state_repulsion_weight: float = 0.0
    basin_coverage_weight: float = 0.0
    nmr_energy_guidance_weight: float = 1.0
    hn_variance_loss_weight: float = 0.0
    cprime_outlier_loss_weight: float = 0.0
    tail_calibration_loss_weight: float = 0.0
    tail_calibration_families: list[str] = field(default_factory=list)
    tail_calibration_fraction: float = 0.2
    tail_direction_loss_weight: float = 0.0
    tail_direction_families: list[str] = field(default_factory=list)
    tail_direction_fraction: float = 0.2
    tail_gap_loss_weight: float = 0.0
    tail_gap_families: list[str] = field(default_factory=list)
    tail_gap_fraction: float = 0.2
    bioemu_all_family_outlier_loss_weight: float = 0.0
    bioemu_all_family_outlier_family_names: list[str] = field(
        default_factory=lambda: ["HN", "N", "CA", "CB", "C'"]
    )
    bioemu_all_family_outlier_fraction: float = 0.05
    bioemu_all_family_outlier_abs_error_threshold: float = 0.0
    bioemu_all_family_outlier_abs_error_threshold_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_all_family_outlier_min_rows: int = 8
    bioemu_all_family_outlier_max_rows_per_family: int = 64
    bioemu_all_family_outlier_low_ess_gate_ess_floor: float = 0.0
    bioemu_all_family_outlier_low_ess_gate_min_scale: float = 0.25
    bioemu_all_family_outlier_residue_family_keys: list[str] = field(
        default_factory=list
    )
    bioemu_hn_signed_support_replay_loss_weight: float = 0.0
    bioemu_hn_signed_support_replay_state_keys: list[str] = field(
        default_factory=list
    )
    bioemu_hn_signed_support_replay_fraction: float = 0.25
    bioemu_hn_signed_support_replay_min_rows: int = 1
    bioemu_hn_signed_support_replay_max_rows_per_state: int = 24
    bioemu_hn_signed_support_replay_huber_beta: float = 0.75
    family_pairwise_rank_loss_weight: float = 0.0
    family_pairwise_rank_families: list[str] = field(default_factory=list)
    family_pairwise_rank_temperature: float = 0.5
    family_pairwise_rank_min_delta: float = 0.15
    family_pairwise_rank_max_rows: int = 96
    residual_trend_loss_weight: float = 0.0
    residual_trend_families: list[str] = field(default_factory=list)
    enable_entry_family_calibration: bool = False
    enable_entry_family_quantile_calibration: bool = False
    entry_family_quantile_calibration_families: list[str] = field(
        default_factory=list
    )
    entry_family_quantile_fraction: float = 0.2
    entry_family_quantile_scale_weight: float = 0.0
    entry_family_calibration_scale_max_by_family: dict[str, float] = field(
        default_factory=dict
    )
    enable_entry_family_regression_calibration: bool = False
    entry_family_regression_calibration_families: list[str] = field(
        default_factory=list
    )
    entry_family_regression_strength: float = 0.0
    entry_family_regression_ridge: float = 0.05
    entry_family_regression_use_raw_predictions: bool = False
    entry_family_regression_slope_min_by_family: dict[str, float] = field(
        default_factory=dict
    )
    entry_family_regression_slope_max_by_family: dict[str, float] = field(
        default_factory=dict
    )
    enable_entry_family_isotonic_calibration: bool = False
    entry_family_isotonic_calibration_families: list[str] = field(
        default_factory=list
    )
    entry_family_isotonic_strength: float = 0.0
    entry_family_isotonic_min_rows: int = 8
    entry_family_isotonic_max_abs: float = 1.0
    entry_family_isotonic_max_abs_by_family: dict[str, float] = field(
        default_factory=dict
    )
    enable_entry_family_feature_residual_calibration: bool = False
    entry_family_feature_residual_calibration_families: list[str] = field(
        default_factory=list
    )
    entry_family_feature_residual_weight: float = 0.0
    entry_family_feature_residual_ridge: float = 0.1
    entry_family_feature_residual_min_rows: int = 8
    entry_family_feature_residual_max_abs: float = 0.6
    enable_measure_aware_row_weights: bool = False
    enable_measure_aware_context_masking: bool = False
    measure_aware_context_target_families: list[str] = field(default_factory=list)
    measure_aware_context_neighbor_offsets: list[int] = field(
        default_factory=lambda: [-1, 0, 1]
    )
    enable_ccc_decomposition_loss: bool = False
    enable_joint_nmr_posterior: bool = False
    family_corr_loss_weight: float = 0.0
    family_scale_loss_weight: float = 0.0
    family_bias_loss_weight: float = 0.0
    student_t_nll_weight: float = 0.0
    crps_calibration_loss_weight: float = 0.0
    joint_nmr_loss_weight: float = 0.0
    hn_context_loss_weight: float = 0.0
    paired_evidence_imputation_loss_weight: float = 0.0
    paired_evidence_imputation_fraction: float = 0.15
    paired_evidence_imputation_targets: list[str] = field(default_factory=list)
    paired_evidence_imputation_entity_weights: dict[str, float] = field(
        default_factory=dict
    )
    paired_evidence_imputation_risk_boost: float = 0.0
    paired_evidence_imputation_tail_boost: float = 0.0
    paired_evidence_imputation_max_fraction: float = 0.8
    enable_pair_evidence_mean_blend: bool = False
    pair_evidence_mean_blend_weight: float = 0.0
    enable_same_family_evidence_interpolation: bool = False
    same_family_evidence_interpolation_weight: float = 0.0
    same_family_evidence_interpolation_sigma: float = 6.0
    same_family_evidence_interpolation_families: list[str] = field(
        default_factory=list
    )
    enable_same_family_residual_interpolation: bool = False
    same_family_residual_interpolation_weight: float = 0.0
    same_family_residual_interpolation_sigma: float = 6.0
    same_family_residual_interpolation_families: list[str] = field(
        default_factory=list
    )
    same_family_residual_interpolation_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    enable_same_family_residual_knn: bool = False
    same_family_residual_knn_weight: float = 0.0
    same_family_residual_knn_residue_sigma: float = 12.0
    same_family_residual_knn_shift_sigma: float = 0.6
    same_family_residual_knn_families: list[str] = field(default_factory=list)
    same_family_residual_knn_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    enable_same_family_residual_gp: bool = False
    same_family_residual_gp_weight: float = 0.0
    same_family_residual_gp_residue_sigma: float = 8.0
    same_family_residual_gp_shift_sigma: float = 100.0
    same_family_residual_gp_ridge: float = 0.05
    same_family_residual_gp_max_observed: int = 128
    same_family_residual_gp_max_abs: float = 1.0
    same_family_residual_gp_families: list[str] = field(default_factory=list)
    same_family_residual_gp_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    same_family_residual_gp_max_abs_by_family: dict[str, float] = field(
        default_factory=dict
    )
    same_family_residual_gp_min_sign_consensus: float = 0.0
    same_family_residual_gp_min_kernel_support: float = 0.0
    same_family_residual_gp_terminal_suppression_distance: float = 0.0
    same_family_residual_gp_risk_gate_min: float = 0.0
    enable_hn_risk_scalar_calibration: bool = False
    hn_risk_scalar_calibration_weight: float = 0.0
    hn_risk_scalar_calibration_min_rows: int = 4
    hn_risk_scalar_calibration_min_risk: float = 0.25
    hn_risk_scalar_calibration_risk_sigma: float = 0.25
    hn_risk_scalar_calibration_shift_sigma: float = 0.25
    hn_risk_scalar_calibration_max_abs: float = 0.20
    hn_risk_scalar_calibration_min_sign_consensus: float = 0.0
    hn_risk_scalar_calibration_terminal_suppression_distance: float = 0.0
    enable_hn_risk_vector_calibration: bool = False
    hn_risk_vector_calibration_feature_sigma: float = 0.35
    enable_hn_entry_preservation_gate: bool = False
    hn_entry_preservation_gate_strength: float = 0.0
    hn_entry_preservation_mae_threshold: float = 0.055
    hn_entry_preservation_ccc_threshold: float = 0.9
    hn_entry_preservation_min_rows: int = 4
    enable_hn_observed_correction_selector: bool = False
    hn_observed_correction_selector_strength: float = 0.0
    hn_observed_correction_selector_min_rows: int = 5
    hn_observed_correction_selector_mae_margin: float = 0.0
    hn_observed_correction_selector_ccc_margin: float = 0.0
    enable_hn_raw_prediction_selector: bool = False
    hn_raw_prediction_selector_strength: float = 0.0
    hn_raw_prediction_selector_min_rows: int = 5
    hn_raw_prediction_selector_mae_margin: float = 0.0
    hn_raw_prediction_selector_ccc_margin: float = 0.0
    enable_hn_pseudomask_raw_prediction_selector: bool = False
    hn_pseudomask_raw_prediction_selector_strength: float = 0.0
    hn_pseudomask_raw_prediction_selector_min_rows: int = 10
    hn_pseudomask_raw_prediction_selector_query_fraction: float = 0.25
    hn_pseudomask_raw_prediction_selector_min_query_rows: int = 3
    hn_pseudomask_raw_prediction_selector_mae_margin: float = 0.0
    hn_pseudomask_raw_prediction_selector_ccc_margin: float = 0.0
    enable_hn_feature_spread_preservation_selector: bool = False
    hn_feature_spread_preservation_selector_strength: float = 0.0
    hn_feature_spread_preservation_selector_min_rows: int = 12
    hn_feature_spread_preservation_selector_std_ratio: float = 0.80
    hn_feature_spread_preservation_selector_delta_margin: float = 0.005
    enable_hn_uncertainty_tail_anchor_selector: bool = False
    hn_uncertainty_tail_anchor_selector_strength: float = 0.0
    hn_uncertainty_tail_anchor_selector_min_rows: int = 12
    hn_uncertainty_tail_anchor_selector_min_masked_rows: int = 8
    hn_uncertainty_tail_anchor_selector_observed_ccc_min: float = 0.88
    hn_uncertainty_tail_anchor_selector_observed_bias_max: float = 0.03
    hn_uncertainty_tail_anchor_selector_observed_std_ratio_min: float = 0.90
    hn_uncertainty_tail_anchor_selector_observed_std_ratio_max: float = 1.18
    hn_uncertainty_tail_anchor_selector_mean_gap_min: float = 0.025
    hn_uncertainty_tail_anchor_selector_sigma_min: float = 0.07
    hn_uncertainty_tail_anchor_selector_low_gap: float = 0.15
    hn_uncertainty_tail_anchor_selector_graph_delta_min: float = 0.08
    hn_uncertainty_tail_anchor_selector_entry_shift_max: float = 0.035
    hn_uncertainty_tail_anchor_selector_row_shift_max: float = 0.18
    hn_uncertainty_tail_anchor_selector_anchor_margin: float = 0.08
    hn_uncertainty_tail_anchor_selector_min_structure_ring: float = 0.04
    hn_uncertainty_tail_anchor_selector_min_anchor_rows: int = 2
    enable_hn_alignment_mean_anchor_selector: bool = False
    hn_alignment_mean_anchor_selector_strength: float = 0.0
    hn_alignment_mean_anchor_selector_min_rows: int = 12
    hn_alignment_mean_anchor_selector_min_masked_rows: int = 8
    hn_alignment_mean_anchor_selector_observed_ccc_min: float = 0.90
    hn_alignment_mean_anchor_selector_observed_bias_max: float = 0.02
    hn_alignment_mean_anchor_selector_observed_std_ratio_min: float = 0.90
    hn_alignment_mean_anchor_selector_observed_std_ratio_max: float = 1.15
    hn_alignment_mean_anchor_selector_min_alignment_risk: float = 0.50
    hn_alignment_mean_anchor_selector_max_structure_ring: float = 0.02
    hn_alignment_mean_anchor_selector_mean_gap_min: float = 0.004
    hn_alignment_mean_anchor_selector_mean_gap_max: float = 0.04
    hn_alignment_mean_anchor_selector_min_shift: float = 0.02
    hn_alignment_mean_anchor_selector_max_shift: float = 0.02
    hn_alignment_mean_anchor_selector_max_abs_graph_delta: float = 0.025
    hn_alignment_mean_anchor_selector_allow_negative_shift: bool = False
    enable_hn_observed_variance_floor_selector: bool = False
    hn_observed_variance_floor_selector_strength: float = 0.0
    hn_observed_variance_floor_selector_min_rows: int = 5
    hn_observed_variance_floor_selector_std_ratio: float = 0.85
    hn_observed_variance_floor_selector_softness: float = 0.05
    hn_observed_variance_floor_selector_raw_std_margin: float = 0.02
    hn_observed_variance_floor_selector_min_risk: float = 0.0
    enable_hn_local_variance_shrink_selector: bool = False
    hn_local_variance_shrink_selector_strength: float = 0.0
    hn_local_variance_shrink_selector_min_rows: int = 10
    hn_local_variance_shrink_selector_query_fraction: float = 0.25
    hn_local_variance_shrink_selector_min_query_rows: int = 3
    hn_local_variance_shrink_selector_residue_sigma: float = 8.0
    hn_local_variance_shrink_selector_feature_sigma: float = 0.35
    hn_local_variance_shrink_selector_std_ratio: float = 1.12
    hn_local_variance_shrink_selector_softness: float = 0.08
    hn_local_variance_shrink_selector_min_support: float = 1.5
    hn_local_variance_shrink_selector_mae_margin: float = 0.002
    hn_local_variance_shrink_selector_ccc_margin: float = 0.004
    hn_local_variance_shrink_selector_max_abs: float = 0.10
    hn_local_variance_shrink_selector_min_risk: float = 0.0
    enable_hn_pseudomask_correction_selector: bool = False
    hn_pseudomask_correction_selector_strength: float = 0.0
    hn_pseudomask_correction_selector_min_rows: int = 8
    hn_pseudomask_correction_selector_query_fraction: float = 0.25
    hn_pseudomask_correction_selector_min_query_rows: int = 3
    hn_pseudomask_correction_selector_mae_margin: float = 0.0
    hn_pseudomask_correction_selector_ccc_margin: float = 0.0
    hn_pseudomask_correction_selector_residue_sigma: float = 8.0
    hn_pseudomask_correction_selector_shift_sigma: float = 0.35
    hn_pseudomask_correction_selector_feature_sigma: float = 0.45
    hn_pseudomask_correction_selector_min_support: float = 0.0
    hn_pseudomask_correction_selector_min_sign_consensus: float = 0.0
    hn_pseudomask_correction_selector_max_abs: float = 0.18
    enable_hn_structure_pseudomask_correction_selector: bool = False
    hn_structure_pseudomask_correction_selector_strength: float = 0.0
    hn_structure_pseudomask_correction_selector_min_rows: int = 8
    hn_structure_pseudomask_correction_selector_query_fraction: float = 0.25
    hn_structure_pseudomask_correction_selector_min_query_rows: int = 3
    hn_structure_pseudomask_correction_selector_mae_margin: float = 0.0
    hn_structure_pseudomask_correction_selector_ccc_margin: float = 0.0
    hn_structure_pseudomask_correction_selector_residue_sigma: float = 8.0
    hn_structure_pseudomask_correction_selector_shift_sigma: float = 0.35
    hn_structure_pseudomask_correction_selector_risk_feature_sigma: float = 0.45
    hn_structure_pseudomask_correction_selector_structure_sigma: float = 0.30
    hn_structure_pseudomask_correction_selector_min_support: float = 0.0
    hn_structure_pseudomask_correction_selector_min_sign_consensus: float = 0.0
    hn_structure_pseudomask_correction_selector_max_abs: float = 0.12
    hn_structure_pseudomask_correction_selector_min_structure_match: float = 0.70
    hn_structure_pseudomask_correction_selector_min_tail_focus: float = 0.0
    hn_structure_pseudomask_correction_selector_tail_softness: float = 0.08
    enable_hn_structure_ridge_residual_selector: bool = False
    hn_structure_ridge_residual_selector_strength: float = 0.0
    hn_structure_ridge_residual_selector_min_rows: int = 12
    hn_structure_ridge_residual_selector_min_support_rows: int = 8
    hn_structure_ridge_residual_selector_query_fraction: float = 0.35
    hn_structure_ridge_residual_selector_min_query_rows: int = 4
    hn_structure_ridge_residual_selector_mae_margin: float = 0.001
    hn_structure_ridge_residual_selector_ccc_margin: float = 0.003
    hn_structure_ridge_residual_selector_ridge: float = 10.0
    hn_structure_ridge_residual_selector_max_abs: float = 0.08
    hn_structure_ridge_residual_selector_min_structure_match: float = 0.80
    hn_structure_ridge_residual_selector_min_tail_focus: float = 0.45
    hn_structure_ridge_residual_selector_exclude_alignment_zero_ring: bool = False
    hn_structure_ridge_residual_selector_exclude_alignment_min: float = 0.50
    hn_structure_ridge_residual_selector_exclude_ring_max: float = 0.02
    hn_structure_ridge_residual_selector_max_masked_mean_ring: float = 0.0
    hn_structure_ridge_residual_selector_max_masked_pred_raw_std_ratio: float = 0.0
    enable_hn_pseudomask_affine_selector: bool = False
    hn_pseudomask_affine_selector_strength: float = 0.0
    hn_pseudomask_affine_selector_min_rows: int = 10
    hn_pseudomask_affine_selector_query_fraction: float = 0.25
    hn_pseudomask_affine_selector_min_query_rows: int = 3
    hn_pseudomask_affine_selector_mae_margin: float = 0.0
    hn_pseudomask_affine_selector_ccc_margin: float = 0.0
    hn_pseudomask_affine_selector_scale_min: float = 0.55
    hn_pseudomask_affine_selector_scale_max: float = 1.35
    hn_pseudomask_affine_selector_max_abs: float = 0.12
    hn_pseudomask_affine_selector_min_observed_std_ratio: float = 0.0
    hn_pseudomask_affine_selector_max_observed_abs_bias: float = 0.0
    hn_pseudomask_affine_selector_min_validation_gate: float = 0.0
    hn_pseudomask_affine_selector_min_masked_raw_observed_std_ratio: float = 0.0
    hn_pseudomask_affine_selector_max_masked_pred_raw_std_ratio: float = 0.0
    hn_pseudomask_affine_selector_min_masked_ring_mean: float = 0.0
    enable_hn_regime_gate: bool = False
    hn_regime_loss_weight: float = 0.0
    hn_regime_delta_weight: float = 0.0
    hn_regime_feature_blend: float = 0.0
    enable_hn_state_coordinate_head: bool = False
    hn_state_coordinate_dim: int = 7
    hn_state_coordinate_names: list[str] = field(
        default_factory=lambda: [
            "ring_current",
            "exchange",
            "alignment_mismatch",
            "sulfur_electrostatic",
            "terminal_disorder",
            "spread_collapse",
            "graph_fragility",
        ]
    )
    enable_hn_tangent_residual_adapter: bool = False
    hn_tangent_residual_weight: float = 0.0
    hn_tangent_residual_max_abs: float = 0.04
    bioemu_tangent_residual_families: list[str] = field(
        default_factory=lambda: ["HN"]
    )
    hn_tangent_allowed_state_names: list[str] = field(default_factory=list)
    hn_tangent_state_chart_weights: dict[str, float] = field(default_factory=dict)
    hn_tangent_state_gate_mode: str = "prior_overlap"
    hn_tangent_state_gate_margin: float = 0.10
    hn_tangent_state_gate_softness: float = 0.06
    enable_hn_tangent_utility_gate: bool = False
    hn_tangent_utility_gate_bias_init: float = 6.0
    hn_tangent_utility_gate_min: float = 0.0
    hn_tangent_utility_gate_max: float = 1.0
    enable_hn_tangent_chart_utility_gate: bool = False
    hn_tangent_chart_utility_gate_init: dict[str, float] = field(default_factory=dict)
    enable_hn_tangent_feature_chart_gate: bool = False
    hn_tangent_feature_chart_gate_state_names: list[str] = field(
        default_factory=list
    )
    hn_tangent_feature_chart_gate_require_dominant_state: bool = False
    hn_tangent_feature_chart_gate_min: float = 0.50
    hn_tangent_feature_chart_gate_strength: float = 1.0
    hn_tangent_utility_supervision_loss_weight: float = 0.0
    hn_tangent_utility_supervision_min_abs: float = 1e-5
    enable_hn_tangent_signed_utility_gate: bool = False
    hn_tangent_signed_utility_gate_bias_init: float = 6.0
    hn_tangent_signed_utility_supervision_loss_weight: float = 0.0
    hn_tangent_signed_utility_supervision_min_abs: float = 1e-5
    hn_tangent_direction_loss_weight: float = 0.0
    hn_tangent_direction_loss_max_abs: float = 0.006
    hn_manifold_contrastive_loss_weight: float = 0.0
    hn_topology_laplacian_loss_weight: float = 0.0
    hn_state_reliability_loss_weight: float = 0.0
    hn_state_abstain_uncertainty_threshold: float = 0.65
    enable_hn_evidence_graph_residual: bool = False
    hn_graph_residual_weight: float = 0.0
    hn_graph_residual_max_abs: float = 0.10
    hn_graph_residual_min_support: float = 0.45
    hn_graph_residual_min_sign_consensus: float = 0.35
    hn_graph_residual_residue_sigma: float = 6.0
    hn_graph_residual_shift_sigma: float = 0.35
    hn_graph_residual_feature_sigma: float = 0.45
    hn_graph_residual_risk_gate_min: float = 0.0
    hn_graph_residual_terminal_suppression_distance: float = 0.0
    hn_graph_residual_target_families: list[str] = field(
        default_factory=lambda: ["HN"]
    )
    hn_graph_residual_evidence_families: list[str] = field(
        default_factory=lambda: ["HN", "N", "CA", "C'"]
    )
    hn95_loss_weight: float = 0.0
    hn95_pairwise_rank_loss_weight: float = 0.0
    hn95_highrisk_tail_loss_weight: float = 0.0
    hn95_regime_balanced_loss_weight: float = 0.0
    hn95_graph_consistency_loss_weight: float = 0.0
    hn95_reconstruction_loss_weight: float = 0.0
    cprime_guardrail_loss_weight: float = 0.0
    cprime_guardrail_ccc_min: float = 0.955
    enable_hn_local_interpolation: bool = True
    hn_local_interpolation_blend: float = 0.25
    hn_local_interpolation_max_distance: int = 4
    enable_bioemu_hn_evidence_local_residual_readout: bool = False
    bioemu_hn_evidence_local_residual_strength: float = 0.50
    bioemu_hn_evidence_local_residual_max_abs: float = 0.08
    bioemu_hn_evidence_local_residual_min_support: float = 0.25
    bioemu_hn_evidence_local_residual_residue_sigma: float = 12.0
    bioemu_hn_evidence_local_residual_shift_sigma: float = 0.12
    bioemu_hn_evidence_local_residual_min_sign_consensus: float = 0.0
    enable_bioemu_hn_evidence_local_residual_pseudomask_guard: bool = False
    bioemu_hn_evidence_local_residual_pseudomask_min_rows: int = 12
    bioemu_hn_evidence_local_residual_pseudomask_query_fraction: float = 0.30
    bioemu_hn_evidence_local_residual_pseudomask_min_query_rows: int = 4
    bioemu_hn_evidence_local_residual_pseudomask_mae_margin: float = 0.0
    bioemu_hn_evidence_local_residual_pseudomask_ccc_margin: float = 0.0
    bioemu_hn_evidence_local_residual_pseudomask_min_gate: float = 0.35
    enable_bioemu_hn_x2d_support_counter_guard: bool = False
    bioemu_hn_x2d_support_counter_guard_max_abs: float = 0.0
    bioemu_hn_x2d_support_counter_guard_pair_center: float = 3.935
    bioemu_hn_x2d_support_counter_guard_pair_softness: float = 0.006
    bioemu_hn_x2d_support_counter_guard_sidecar_center: float = 1.60
    bioemu_hn_x2d_support_counter_guard_sidecar_softness: float = 0.08
    bioemu_hn_x2d_support_counter_guard_direction_center: float = 0.018
    bioemu_hn_x2d_support_counter_guard_direction_softness: float = 0.010
    bioemu_hn_x2d_support_counter_guard_min_gate: float = 0.0
    bioemu_hn_x2d_support_counter_guard_suppressed_residue_names: list[str] = field(
        default_factory=list
    )
    bioemu_hn_x2d_support_counter_guard_relief_residue_names: list[str] = field(
        default_factory=list
    )
    bioemu_hn_x2d_support_counter_guard_relief_max_abs: float = 0.0
    bioemu_hn_x2d_support_counter_guard_relief_min_gate: float = 0.0
    bioemu_hn_x2d_support_counter_guard_rescue_residue_names: list[str] = field(
        default_factory=list
    )
    bioemu_hn_x2d_support_counter_guard_rescue_max_abs: float = 0.0
    bioemu_hn_x2d_support_counter_guard_rescue_pair_center: float = 3.932
    bioemu_hn_x2d_support_counter_guard_rescue_sidecar_center: float = 1.30
    bioemu_hn_x2d_support_counter_guard_rescue_min_gate: float = 0.0
    enable_bioemu_family_specific_evidence_context: bool = False
    bioemu_family_specific_evidence_source_families: dict[str, list[str]] = field(
        default_factory=lambda: {
            "HN": ["HN", "N", "C'"],
            "C'": ["C'", "CA", "N", "HN"],
        }
    )
    bioemu_family_specific_evidence_context_strength_by_family: dict[
        str, float
    ] = field(default_factory=lambda: {"HN": 0.35, "C'": 0.18})
    enable_bioemu_hn_low_ess_local_residual_gate: bool = False
    bioemu_hn_low_ess_local_residual_ess_floor: float = 3.0
    bioemu_hn_low_ess_local_residual_min_scale: float = 0.50
    enable_bioemu_hn_entity_residue_local_residual_gate: bool = False
    bioemu_hn_entity_residue_local_residual_gate_entities: list[str] = field(
        default_factory=list
    )
    bioemu_hn_entity_residue_local_residual_gate_residues: list[str] = field(
        default_factory=list
    )
    bioemu_hn_entity_residue_local_residual_gate_target_ids: list[str] = field(
        default_factory=list
    )
    bioemu_hn_entity_residue_local_residual_gate_factor: float = 0.0
    enable_bioemu_hn_evidence_local_variance_restore: bool = False
    bioemu_hn_evidence_local_variance_restore_strength: float = 1.0
    bioemu_hn_evidence_local_variance_restore_min_rows: int = 4
    bioemu_hn_evidence_local_variance_restore_scale_min: float = 0.75
    bioemu_hn_evidence_local_variance_restore_scale_max: float = 1.35
    bioemu_hn_evidence_local_variance_restore_scale_floor: float = 1.0
    enable_bioemu_hn_post_spread_saturated_ceiling: bool = False
    bioemu_hn_post_spread_saturated_ceiling_min_support: float = 3.0
    bioemu_hn_post_spread_saturated_ceiling_min_local_delta_abs: float = 0.085
    bioemu_hn_post_spread_saturated_ceiling_max_abs: float = 0.025
    bioemu_hn_post_spread_saturated_ceiling_support_softness: float = 3.0
    bioemu_hn_post_spread_saturated_ceiling_direction: str = "negative"
    bioemu_hn_post_spread_saturated_ceiling_high_side_min_mu: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_high_side_softness: float = 0.25
    bioemu_hn_post_spread_saturated_ceiling_high_side_floor: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_risk_boost_scale: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_risk_min_mu: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_risk_mu_softness: float = 0.25
    bioemu_hn_post_spread_saturated_ceiling_risk_min_local_delta_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_risk_local_delta_softness: float = 0.05
    bioemu_hn_post_spread_saturated_ceiling_risk_min_support: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_risk_support_softness: float = 2.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_scale: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_min_mu: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_mu_softness: float = 0.35
    bioemu_hn_post_spread_saturated_ceiling_active_boost_min_support: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_support_softness: float = 2.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_min_local_delta_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_local_delta_softness: float = 0.05
    bioemu_hn_post_spread_saturated_ceiling_active_boost_nonlocal_floor: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_min_norm: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_softness: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_min_norm: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_softness: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_mu: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_mu_softness: float = 0.35
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_support: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_support_softness: float = 3.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_require_negative_local: bool = False
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_local_delta_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_weak_local_delta_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_nonlocal_floor: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_min_norm: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_softness: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_min_norm: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_softness: float = 1.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_abs: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_names: list[str] = field(
        default_factory=list
    )
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_support_bins: list[str] = field(
        default_factory=list
    )
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_mu: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_mu_softness: float = 0.25
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_support: float = 0.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_support_softness: float = 2.0
    bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_local_delta_abs: float = 0.02
    enable_bioemu_hn_ring_coordinate_delta_gate: bool = False
    bioemu_hn_ring_coordinate_delta_gate_coordinate_name: str = "ring_orientation"
    bioemu_hn_ring_coordinate_delta_gate_quantile: float = 0.10
    bioemu_hn_ring_coordinate_delta_gate_side: str = "lo"
    bioemu_hn_ring_coordinate_delta_gate_factor: float = 1.0
    enable_bioemu_same_family_hn_ring_coordinate_delta_gate: bool = False
    bioemu_same_family_hn_ring_coordinate_delta_gate_coordinate_name: str = (
        "ring_orientation"
    )
    bioemu_same_family_hn_ring_coordinate_delta_gate_quantile: float = 0.25
    bioemu_same_family_hn_ring_coordinate_delta_gate_side: str = "lo"
    bioemu_same_family_hn_ring_coordinate_delta_gate_factor: float = 0.35
    enable_bioemu_hn_ring_reliability_delta_gate: bool = False
    bioemu_hn_ring_reliability_delta_gate_coordinate_name: str = "ring_orientation"
    bioemu_hn_ring_reliability_delta_gate_quantile: float = 0.50
    bioemu_hn_ring_reliability_delta_gate_side: str = "hi"
    bioemu_hn_ring_reliability_delta_gate_factor: float = 1.0
    enable_bioemu_cprime_evidence_local_residual_readout: bool = False
    bioemu_cprime_evidence_local_residual_strength: float = 1.25
    bioemu_cprime_evidence_local_residual_max_abs: float = 0.25
    bioemu_cprime_evidence_local_residual_negative_max_abs: float = 0.25
    bioemu_cprime_evidence_local_residual_positive_max_abs: float = 0.25
    bioemu_cprime_evidence_local_residual_min_support: float = 0.25
    bioemu_cprime_evidence_local_residual_residue_sigma: float = 4.0
    bioemu_cprime_evidence_local_residual_shift_sigma: float = 0.0
    bioemu_cprime_evidence_local_residual_min_sign_consensus: float = 0.0
    enable_bioemu_cprime_evidence_local_variance_restore: bool = False
    bioemu_cprime_evidence_local_variance_restore_strength: float = 1.0
    bioemu_cprime_evidence_local_variance_restore_min_rows: int = 4
    bioemu_cprime_evidence_local_variance_restore_scale_min: float = 0.75
    bioemu_cprime_evidence_local_variance_restore_scale_max: float = 1.35
    bioemu_cprime_evidence_local_variance_restore_scale_floor: float = 1.0
    enable_bioemu_cprime_carbonyl_uncertainty_delta_gate: bool = False
    bioemu_cprime_carbonyl_uncertainty_delta_gate_quantile: float = 0.67
    bioemu_cprime_carbonyl_uncertainty_delta_gate_factor: float = 0.0
    enable_bioemu_cprime_carbonyl_uncertainty_delta_low_boost: bool = False
    bioemu_cprime_carbonyl_uncertainty_delta_low_boost_quantile: float = 0.60
    bioemu_cprime_carbonyl_uncertainty_delta_low_boost_factor: float = 1.0
    enable_hn_extended_evidence_context: bool = False
    enable_candidate_free_entry_condition_context: bool = False
    candidate_free_entry_condition_gate_init: float = 0.15
    enable_hn_residual_moment_head: bool = False
    hn_residual_moment_weight: float = 0.35
    hn_risk_residual_moment_weight: float = 0.0
    enable_hn_physics_specialist_head: bool = False
    hn_physics_specialist_weight: float = 0.0
    hn_physics_specialist_min_risk: float = 0.35
    enable_cprime_residual_moment_head: bool = False
    cprime_residual_moment_weight: float = 0.45
    cprime_context_residual_moment_weight: float = 0.0
    enable_hn_n_paired_residual_correction: bool = False
    hn_n_paired_residual_weight: float = 0.04
    hn_n_paired_residual_max_abs: float = 0.20
    hn_n_paired_residual_detach: bool = True
    hn_n_paired_residual_neighbor_sigma: float = 0.0
    hn_n_paired_residual_max_distance: int = 0
    enable_candidate_observable_context: bool = False
    candidate_observable_context_gate_init: float = 1.0
    candidate_observable_context_target_families: list[str] = field(
        default_factory=list
    )
    training_input_mode: str = "legacy_cs_fit"
    enable_bioemu_latent_nmr: bool = False
    bioemu_latent_provider: str = "fixture"
    bioemu_x0_required_provider_class: str | None = None
    bioemu_x0_allow_fixture_provider_for_diagnostic: bool = False
    bioemu_latent_fixture_path: str | None = None
    bioemu_repo_root: str | None = None
    bioemu_checkpoint_path: str | None = None
    base_bioemu_checkpoint_path: str | None = None
    bioemu_score_checkpoint_path: str | None = None
    bioemu_score_model_config_path: str | None = None
    bioemu_score_checkpoint_is_online_generator: bool = False
    bioemu_score_adapter_checkpoint_path: str | None = None
    bioemu_score_adapter_checkpoint_kind: str | None = None
    bioemu_score_adapter_runtime_applicable: bool = False
    bioemu_score_adapter_handoff_summary_path: str | None = None
    bioemu_score_adapter_handoff_ready: bool = False
    bioemu_score_adapter_handoff_teacher_probe_only: bool = False
    bioemu_score_adapter_handoff_final_acceptance_eligible: bool = False
    bioemu_score_adapter_handoff_writes_latest_accepted: bool = False
    bioemu_score_adapter_handoff_use_for_latest_accepted_teacher: bool = False
    bioemu_score_adapter_handoff_guard_round_id: str | None = None
    bioemu_score_adapter_handoff_artifact_kind: str | None = None
    bioemu_score_adapter_handoff_shadow_equivalence_student_summary_path: str | None = None
    bioemu_official_x1d_adapter_enabled: bool = False
    bioemu_official_x1d_adapter_train_through_structure: bool = False
    bioemu_official_x1d_adapter_latent_bridge_trainable: bool = False
    bioemu_official_x1d_adapter_rank: int = 8
    bioemu_official_x1d_adapter_scale: float = 1.0
    bioemu_official_x1d_adapter_max_abs: float = 0.02
    bioemu_official_x2d_adapter_enabled: bool = False
    bioemu_official_x2d_adapter_train_through_structure: bool = False
    bioemu_official_x2d_adapter_rank: int = 8
    bioemu_official_x2d_adapter_scale: float = 1.0
    bioemu_official_x2d_adapter_max_abs: float = 0.01
    bioemu_official_x2d_adapter_wakeup_init_std: float = 0.0
    bioemu_official_x2d_adapter_wakeup_condition_output_std: float = 0.0
    bioemu_official_x2d_adapter_wakeup_zero_threshold: float = 1.0e-12
    bioemu_official_x2d_adapter_wakeup_seed: int = 1009
    bioemu_official_x2d_adapter_wakeup_force: bool = False
    bioemu_official_adapter_conditioning_enabled: bool = False
    bioemu_official_adapter_conditioning_dim: int = 40
    bioemu_official_adapter_conditioning_scale: float = 1.0
    bioemu_official_adapter_gradient_sample_limit: int = 0
    bioemu_provider_lr_multiplier: float = 1.0
    bioemu_provider_gradient_audit_enabled: bool = False
    bioemu_x0_conditioning_guidance_enabled: bool = False
    bioemu_x0_conditioning_guidance_example_limit: int = 64
    bioemu_x0_conditioning_guidance_sample_count: int = 0
    bioemu_x0_conditioning_guidance_anchor_ensemble_size: int = 1
    bioemu_x0_conditioning_guidance_seed_offset: int = 911_000
    bioemu_x0_conditioning_guidance_anchor_scale_nm: float = 1.0
    bioemu_x0_conditioning_guidance_max_abs_nm: float = 5.0
    bioemu_x0_conditioning_guidance_promotable: bool = False
    bioemu_x0_conditioning_guidance_denoising_scale: float = 0.0
    bioemu_x0_conditioning_guidance_denoising_max_norm_nm: float = 0.25
    bioemu_x0_conditioning_guidance_fail_on_error: bool = False
    bioemu_x0_generated_support_transfer_gap_mitigation: bool = False
    bioemu_x0_generated_support_transfer_gap_source: str | None = None
    bioemu_x0_generated_support_transfer_gap_summary_path: str | None = None
    bioemu_x0_epoch_number_offset: int = 0
    bioemu_x0_support_face_contract_enabled: bool = False
    bioemu_x0_support_face_baseline_history_path: str | None = None
    bioemu_x0_support_face_target_baseline_label: str | None = None
    bioemu_x0_support_face_target_baseline_epoch: int = 0
    bioemu_x0_support_face_epoch_matched_control: bool = True
    bioemu_x0_support_face_promote_only_if_target_held: bool = False
    bioemu_x0_support_face_stop_on_target_failure: bool = False
    bioemu_latent_dim: int = 64
    bioemu_projection_dim: int = 64
    bioemu_sample_count: int = 512
    bioemu_allow_sub512_sample_count_for_diagnostic: bool = False
    bioemu_finetune_mode: str = "head_then_adapter"
    bioemu_structure_feature_policy: str = "benchmark_only"
    enable_bioemu_observation_chain_atlas: bool = False
    bioemu_coordinate_layers: list[str] = field(
        default_factory=lambda: [
            "sequence",
            "structural_prior",
            "solution_ensemble",
            "nmr_observation",
        ]
    )
    bioemu_nmr_aggregator_invariance: str = "permutation"
    bioemu_nmr_structure_probe_gradient_policy: str = "detached_benchmark_only"
    bioemu_nmr_sequence_shortcut_penalty_weight: float = 0.0
    bioemu_nmr_component_identifiability_loss_weight: float = 0.0
    bioemu_sequence_embedding_provider: str = "learned"
    bioemu_sequence_embedding_model: str = "esm3_sm_open_v1"
    bioemu_sequence_embedding_cache_dir: str | None = None
    bioemu_sequence_embedding_weight: float = 1.0
    bioemu_sequence_embedding_fail_on_missing: bool = True
    bioemu_nmr_train_sample_count: int = 512
    bioemu_nmr_val_sample_count: int = 512
    bioemu_nmr_benchmark_sample_count: int = 512
    bioemu_cs_reweighting_teacher_predictions_path: str | None = None
    bioemu_cs_reweighting_teacher_guard_path: str | None = None
    bioemu_cs_reweighting_teacher_mean_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_prior_mode: str = "uniform"
    bioemu_cs_reweighting_teacher_prior_source: str = "uniform_bioemu_conformer_pool"
    bioemu_cs_reweighting_teacher_fit_atom_families: list[str] = field(
        default_factory=lambda: ["H", "N", "CA", "CB"]
    )
    bioemu_cs_reweighting_teacher_prior_kl_weight: float = 0.0
    bioemu_cs_reweighting_teacher_prior_kl_temperature: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_kl_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_kl_temperature: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_min_matched_samples: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_min_support_coverage: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_cosat_blend: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_cosat_blend_mode: str = "linear"
    bioemu_cs_reweighting_teacher_generated_prior_cosat_power: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_cosat_scale: float = 256.0
    bioemu_cs_reweighting_teacher_generated_prior_cosat_focus_families: list[
        str
    ] = field(default_factory=list)
    bioemu_cs_reweighting_teacher_generated_prior_cosat_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_cs_reweighting_teacher_generated_prior_cosat_fallback_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_cs_reweighting_teacher_generated_prior_cosat_fallback_min_rows: int = 0
    bioemu_cs_reweighting_teacher_generated_prior_cosat_min_family_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_cosat_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_cs_reweighting_teacher_generated_prior_topk_mass_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_topk_fraction: float = 0.125
    bioemu_cs_reweighting_teacher_generated_prior_topk_min_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_topk_target_mass_scale: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_topk_listwise_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top1_nll_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_tail_mass_contract_required: bool = (
        False
    )
    bioemu_cs_reweighting_teacher_generated_prior_tail_mass_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_tail_fraction: float = 0.125
    bioemu_cs_reweighting_teacher_generated_prior_tail_min_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_tail_mass_cap_scale: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_sharpness_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_sharpness_ess_scale: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_sharpness_top_mass_scale: float = (
        1.0
    )
    bioemu_cs_reweighting_teacher_generated_prior_sharpness_ess_loss_weight: float = (
        1.0
    )
    bioemu_cs_reweighting_teacher_generated_prior_sharpness_top_mass_loss_weight: float = (
        1.0
    )
    bioemu_cs_reweighting_teacher_generated_prior_student_ess_floor: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_student_ess_floor_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_student_top_mass_cap: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_student_top_mass_cap_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_ccc_floor_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_ccc_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_cs_reweighting_teacher_generated_prior_ccc_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_cs_reweighting_teacher_generated_prior_macro_ccc_floor: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_macro_ccc_floor_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_energy_std_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_energy_shape_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_energy_rank_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_energy_corr_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_energy_rank_margin: float = 0.15
    bioemu_cs_reweighting_teacher_generated_prior_energy_rank_top_fraction: float = 0.125
    bioemu_cs_reweighting_teacher_generated_prior_energy_huber_beta: float = 0.5
    bioemu_cs_reweighting_teacher_generated_prior_energy_std_huber_beta: float = 0.25
    bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_huber_beta: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_target_clip: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_weight_power: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_margin_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_margin: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_fraction: float = 0.125
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_min_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_fraction: float = 0.25
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_min_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_mode: str = "teacher_tail"
    bioemu_cs_reweighting_teacher_generated_prior_top_rank_weight_power: float = 0.5
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_margin_loss_weight: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_margin: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_fraction: float = 0.25
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_min_count: int = 1
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_min_teacher_mass: float = 0.0
    bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_teacher_mass_power: float = 0.5
    bioemu_cs_reweighting_teacher_generated_prior_ramp_examples: int = 0
    bioemu_cs_reweighting_teacher_generated_prior_ramp_start_scale: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_ramp_power: float = 1.0
    bioemu_cs_reweighting_teacher_generated_prior_log_prob_source: str = "raw"
    bioemu_cs_reweighting_teacher_require_generated_prior_grad: bool = False
    bioemu_cs_reweighting_teacher_generated_prior_online_proxy_teacher_enabled: bool = False
    bioemu_ucbshift_cnnls_teacher_weights_path: str | None = None
    bioemu_ucbshift_cnnls_teacher_predictions_path: str | None = None
    bioemu_ucbshift_cnnls_teacher_alignment_mode: str = "sample_index"
    bioemu_ucbshift_cnnls_teacher_min_weight_count: int = 1
    bioemu_ucbshift_cnnls_teacher_kl_weight: float = 0.0
    bioemu_ucbshift_cnnls_teacher_prior_kl_weight: float = 0.0
    bioemu_ucbshift_cnnls_teacher_mean_loss_weight: float = 0.0
    bioemu_ucbshift_cnnls_teacher_mean_loss_mode: str = "huber"
    bioemu_ucbshift_cnnls_teacher_mean_family_names: list[str] = field(
        default_factory=list
    )
    bioemu_ucbshift_cnnls_teacher_mean_start_epoch: int = 1
    bioemu_ucbshift_cnnls_teacher_mean_ramp_epochs: int = 1
    bioemu_ucbshift_cnnls_teacher_reliability_weighting: bool = False
    bioemu_ucbshift_cnnls_teacher_reliability_min_weight: float = 0.20
    bioemu_ucbshift_cnnls_teacher_reliability_ess_floor: float = 20.0
    bioemu_ucbshift_cnnls_teacher_reliability_entropy_floor: float = 1.0
    bioemu_ucbshift_cnnls_teacher_reliability_non_bioemu_source_threshold: float = 0.90
    bioemu_ucbshift_cnnls_teacher_reliability_support_gap_floor: float = 0.25
    bioemu_ucbshift_cnnls_teacher_reliability_adaptive_floor_unmet_weight: float = 0.70
    bioemu_ucbshift_cnnls_teacher_guard_path: str | None = None
    bioemu_detached_support_guard_path: str | None = None
    bioemu_detached_support_guard_levels: list[str] = field(
        default_factory=lambda: ["critical_support_gap", "geometry_spread_guard"]
    )
    bioemu_detached_support_guard_min_priority: float = 0.0
    bioemu_ucbshift_cnnls_teacher_guard_families: list[str] = field(
        default_factory=lambda: ["N"]
    )
    bioemu_ucbshift_cnnls_teacher_guard_min_outlier_fraction: float = 0.5
    bioemu_ucbshift_cnnls_teacher_guard_exclude_kl: bool = True
    bioemu_ucbshift_cnnls_teacher_guard_exclude_mean_loss: bool = True
    bioemu_nmr_sampling_method: str = "sobol_antithetic"
    bioemu_nmr_enable_stratified_chart_sampling: bool = True
    bioemu_nmr_min_rare_chart_samples: int = 1
    bioemu_nmr_posterior_weight_temperature: float = 1.0
    bioemu_support_mode_occupancy_guard_enabled: bool = False
    bioemu_support_mode_occupancy_guard_entity: str = ""
    bioemu_support_mode_occupancy_guard_mode: str = ""
    bioemu_support_mode_occupancy_guard_cap: float = 1.0
    bioemu_support_mode_occupancy_guard_mean_weight: float = 0.0
    bioemu_support_mode_occupancy_guard_label_path: str | None = None
    bioemu_inference_readout_overlay_enabled: bool = False
    bioemu_inference_readout_overlay_path: str | None = None
    bioemu_inference_readout_overlay_value_column: str = (
        "inference_overlay_predicted_value"
    )
    bioemu_inference_readout_overlay_delta_column: str = "inference_overlay_delta"
    bioemu_inference_readout_overlay_source_column: str = "inference_overlay_source"
    bioemu_inference_readout_overlay_output_prefix: str = "runtime_overlay"
    bioemu_inference_readout_overlay_active_families: list[str] = field(
        default_factory=lambda: ["HN", "N"]
    )
    bioemu_inference_readout_overlay_require_exact_rows: bool = False
    bioemu_bridge_audit_signal_enabled: bool = False
    bioemu_bridge_uncertainty_guard_only: bool = False
    bioemu_bridge_audit_signal_entity: str = ""
    bioemu_bridge_audit_signal_mode: str = ""
    bioemu_bridge_audit_signal_family: str = ""
    bioemu_bridge_audit_signal_sign: str = ""
    bioemu_bridge_audit_signal_weight: float = 0.0
    bioemu_bridge_audit_signal_mean_weight: float = 0.0
    bioemu_bridge_audit_signal_cap_change: float = 0.0
    bioemu_bridge_audit_signal_residues: list[str] = field(default_factory=list)
    bioemu_bridge_audit_signal_residue_aware_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sidecar_path: str | None = None
    bioemu_bridge_audit_signal_cs_energy_enabled: bool = False
    bioemu_bridge_audit_signal_cs_energy_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_cap: float = 0.02
    bioemu_bridge_audit_signal_cs_energy_mode_only: bool = False
    bioemu_bridge_audit_signal_cs_energy_profile_kind: str = "squared_error"
    bioemu_bridge_audit_signal_cs_energy_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_bridge_audit_signal_cs_energy_tail_fraction: float = 0.25
    bioemu_bridge_audit_signal_cs_energy_signed_consensus_weight: float = 0.35
    bioemu_bridge_audit_signal_cs_energy_learned_scale_enabled: bool = False
    bioemu_bridge_audit_signal_cs_energy_learned_scale_max: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_sample_head_enabled: bool = False
    bioemu_bridge_audit_signal_cs_energy_sample_head_max_abs: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_sample_head_matrix_features_enabled: bool = False
    bioemu_bridge_audit_signal_cs_energy_pairwise_rank_loss_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_pairwise_rank_margin: float = 0.25
    bioemu_bridge_audit_signal_cs_energy_pairwise_rank_temperature: float = 0.1
    bioemu_bridge_audit_signal_cs_energy_pairwise_rank_topk: int = 64
    bioemu_bridge_audit_signal_cs_energy_pairwise_rank_min_gap: float = 0.02
    bioemu_bridge_audit_signal_cs_energy_aux_family_names: list[str] = field(
        default_factory=list
    )
    bioemu_bridge_audit_signal_cs_energy_aux_pairwise_rank_loss_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_aux_logit_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_aux_logit_cap: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_oracle_beta: float = 8.0
    bioemu_bridge_audit_signal_cs_energy_oracle_kl_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_energy_oracle_temperature: float = 1.0
    bioemu_bridge_audit_signal_cs_energy_oracle_min_samples: int = 16
    bioemu_bridge_audit_signal_cs_measure_readout_enabled: bool = False
    bioemu_bridge_audit_signal_cs_measure_readout_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_measure_readout_max_abs_delta: float = 1.0
    bioemu_bridge_audit_signal_cs_measure_distill_loss_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_measure_distill_trainable: bool = True
    bioemu_bridge_audit_signal_cs_measure_distill_family_names: list[str] = field(
        default_factory=list
    )
    bioemu_bridge_audit_signal_cs_measure_distill_family_loss_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_measure_distill_huber_delta: float = 1.0
    bioemu_bridge_audit_signal_cs_measure_distill_min_rows: int = 1
    bioemu_bridge_audit_signal_cs_measure_distill_start_epoch: int = 1
    bioemu_bridge_audit_signal_cs_measure_distill_ramp_epochs: int = 1
    bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_enabled: bool = False
    bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_ess_floor: float = 0.0
    bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_entropy_floor: float = 0.0
    bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_min_scale: float = 0.75
    bioemu_bridge_audit_signal_cs_measure_distill_family_confidence_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_measure_distill_mechanism_mask_focus_enabled: bool = (
        False
    )
    bioemu_bridge_audit_signal_cs_measure_distill_mechanism_match_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_measure_distill_mechanism_nonmatch_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_loss_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_names: list[str] = field(
        default_factory=list
    )
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_loss_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_weight_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_min_rows: int = 1
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_start_epoch: int = 1
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_ramp_epochs: int = 1
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_gate_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_consensus: float = 0.62
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_low_scale: float = 0.80
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_high_scale: float = 1.25
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_valid_samples: int = 16
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_focus_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_min_proxy: float = 0.35
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_low_proxy_scale: float = 0.35
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_terminal_scale: float = 0.50
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_signed_gate_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_center: float = 2.0e-4
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_softness: float = 1.0e-4
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sign_consensus_min: float = 0.60
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_low_consensus_scale: float = 0.45
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_plane_min_proxy: float = 0.30
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_gate_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_min_scale: float = 0.55
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_peptide_plane_weight: float = 1.0
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_weight: float = 1.0
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_min_proxy: float = 0.25
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_use_neighborhood_sidecar: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_weight: float = 0.70
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_weight: float = 0.30
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_min_score: float = 0.05
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_gate_enabled: bool = False
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_min_score: float = 0.08
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_min_score: float = 0.06
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_min_score: float = 0.12
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_weight: float = 0.20
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_loss_weight: float = 0.0
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_hbond_weight: float = 1.0
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_peptide_weight: float = 0.8
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_contact_weight: float = 0.25
    bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_min_rows: int = 1
    enable_bioemu_neighborhood_aware_nmr: bool = False
    bioemu_neighborhood_edge_offsets: list[int] = field(
        default_factory=lambda: [-2, -1, 0, 1, 2]
    )
    bioemu_neighborhood_family_names: list[str] = field(
        default_factory=lambda: ["HN", "C'"]
    )
    bioemu_neighborhood_hidden_delta_max_abs: float = 0.04
    bioemu_neighborhood_energy_weight: float = 0.0
    bioemu_neighborhood_train_with_local_adapters: bool = True
    bioemu_neighborhood_use_row_physics_proxy: bool = True
    bioemu_neighborhood_apply_after_score_aware: bool = True
    bioemu_neighborhood_energy_after_score_aware: bool = False
    bioemu_neighborhood_sidecar_prior_energy_scale: float = 0.0
    bioemu_neighborhood_sidecar_prior_centered: bool = True
    bioemu_neighborhood_sidecar_path: str | None = None
    enable_bioemu_family_split_capacity_readout: bool = False
    bioemu_family_split_capacity_readout_family_names: list[str] = field(
        default_factory=lambda: ["C'", "N", "CA", "CB"]
    )
    bioemu_family_split_capacity_readout_max_abs: float = 4.0
    bioemu_family_split_capacity_readout_width_multiplier: float = 2.0
    bioemu_family_split_capacity_readout_bottleneck_multiplier: float = 1.0
    bioemu_family_split_capacity_readout_depth: int = 2
    bioemu_family_split_capacity_readout_gate_width_multiplier: float = 1.0
    bioemu_family_split_capacity_readout_gate_depth: int = 1
    bioemu_family_split_capacity_readout_use_diagnostics: bool = False
    bioemu_family_split_capacity_readout_diagnostic_scale: float = 1.0
    bioemu_train_only_family_split_capacity_readout: bool = False
    bioemu_capture_x2d_pair_latent: bool = False
    bioemu_x2d_pair_latent_weight: float = 0.0
    bioemu_x2d_pair_latent_family_names: list[str] = field(default_factory=list)
    bioemu_x2d_pair_local_only_family_names: list[str] = field(default_factory=list)
    bioemu_x2d_pair_local_only_weight_scale: float = 1.0
    bioemu_x2d_pair_global_scale_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x2d_pair_missing_fallback: str = "none"
    bioemu_x2d_pair_summary_offsets: list[int] = field(
        default_factory=lambda: [-2, -1, 0, 1, 2]
    )
    bioemu_x2d_pair_global_topk: int = 0
    bioemu_x2d_pair_global_weight: float = 1.0
    bioemu_x2d_pair_global_min_sequence_separation: int = 0
    bioemu_x2d_pair_global_max_sequence_separation: int = 0
    bioemu_x2d_pair_global_extra_bands: list[dict[str, float]] = field(
        default_factory=list
    )
    bioemu_x2d_pair_global_band_name: str = "global"
    bioemu_x2d_pair_split_global_bands: bool = False
    bioemu_x2d_pair_global_band_scale_by_family: dict[
        str, dict[str, float]
    ] = field(default_factory=dict)
    bioemu_x2d_pair_global_band_reliability_feature_weights_by_family: dict[
        str, dict[str, dict[str, float]]
    ] = field(default_factory=dict)
    bioemu_x2d_pair_global_band_reliability_min_scale_by_family: dict[
        str, dict[str, float]
    ] = field(default_factory=dict)
    bioemu_x2d_pair_reliability_gate_enabled: bool = False
    bioemu_x2d_pair_reliability_min_scale_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x2d_pair_reliability_feature_weights_by_family: dict[
        str, dict[str, float]
    ] = field(default_factory=dict)
    bioemu_x2d_pair_reliability_apply_to_global_only: bool = False
    bioemu_x2d_pair_context_norm_cap: float = 0.0
    bioemu_x2d_pair_context_norm_cap_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x2d_pair_update_full_context: bool = True
    bioemu_x2d_pair_projection_init_std: float = 0.0
    bioemu_neighborhood_edge_classes: list[str] = field(
        default_factory=lambda: [
            "sequence",
            "peptide_plane",
            "spatial_contact",
            "hbond",
            "ring",
            "electrostatic",
        ]
    )
    bioemu_neighborhood_topk_by_class: dict[str, int] = field(
        default_factory=lambda: {
            "sequence": 5,
            "peptide_plane": 3,
            "spatial_contact": 16,
            "hbond": 8,
            "ring": 8,
            "electrostatic": 8,
        }
    )
    bioemu_nmr_decode_structures_during_training: bool = False
    bioemu_nmr_enable_control_variate: bool = True
    enable_bioemu_x0_posterior_ensemble: bool = False
    bioemu_x0_posterior_stage: str = "decoder_pretrain"
    bioemu_x0_posterior_support_k: int = 512
    bioemu_x0_target_family_ccc: float = 0.95
    bioemu_x0_stage_b_supervised_fraction: float = 0.25
    bioemu_x0_stage_a_train_capacity_adapter_only: bool = False
    bioemu_x0_stage_c_train_posterior_heads: bool = True
    bioemu_x0_stage_c_train_evidence_heads: bool = True
    bioemu_x0_stage_c_train_decoder_heads: bool = True
    bioemu_x0_stage_c_train_hn_local_decoder_only: bool = False
    bioemu_x0_stage_c_train_hn_local_capacity_adapter: bool = False
    bioemu_x0_stage_c_train_support_basis_only: bool = False
    bioemu_x0_stage_c_train_support_basis_and_family_affine: bool = False
    bioemu_x0_stage_c_train_hn_cprime_n_support_only: bool = False
    bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_only: bool = False
    bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_capacity_adapters: bool = True
    bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_hn_cprime_basis: bool = False
    bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_posterior_energy_extra: bool = False
    bioemu_x0_stage_c_train_cprime_isolated_basis_only: bool = False
    bioemu_x0_stage_c_train_capacity_adapter_only: bool = False
    bioemu_x0_stage_c_train_capacity_adapter_gate_only: bool = False
    bioemu_x0_stage_c_train_capacity_adapter_extra_only: bool = False
    bioemu_x0_stage_c_train_posterior_energy_extra_only: bool = False
    bioemu_x0_stage_b_train_prior_logit_adapter_only: bool = False
    bioemu_x0_decoder_hidden_dim: int = 256
    bioemu_x0_decoder_dropout: float = 0.0
    bioemu_x0_decoder_teacher_loss_weight: float = 1.0
    bioemu_x0_decoder_teacher_family_macro_weight: float = 1.0
    bioemu_x0_decoder_teacher_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_decoder_teacher_support_variance_weight: float = 0.0
    bioemu_x0_decoder_teacher_support_variance_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_decoder_mechanism_support_basis_count: int = 0
    bioemu_x0_decoder_mechanism_support_basis_cap: float = 0.0
    bioemu_x0_decoder_hn_signed_support_basis_count: int = 0
    bioemu_x0_decoder_hn_signed_support_basis_cap: float = 0.0
    bioemu_x0_decoder_hn_direct_support_spread_scale_ppm: float = 0.0
    bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_cap_ppm: float = 0.0
    bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_init_ppm: float = 0.0
    bioemu_x0_decoder_family_direct_support_spread_scale_ppm_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_family_direct_support_spread_trainable_scale_cap_ppm_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_family_direct_support_spread_trainable_scale_init_ppm_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_cprime_isolated_support_basis_count: int = 0
    bioemu_x0_decoder_cprime_isolated_support_basis_cap: float = 0.0
    bioemu_x0_decoder_n_ca_cb_support_deviation_enabled: bool = True
    bioemu_x0_decoder_emit_support_basis_diagnostics: bool = True
    bioemu_x0_support_basis_lr_multiplier: float = 1.0
    bioemu_x0_mechanism_decoder_lr_multiplier: float = 1.0
    bioemu_x0_decoder_phi_trust_region_loss_weight: float = 0.0
    bioemu_x0_decoder_phi_trust_region_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_decoder_phi_trust_region_huber_beta: float = 0.05
    bioemu_x0_support_basis_delta_ceiling_loss_weight: float = 0.0
    bioemu_x0_support_basis_delta_ceiling_ppm: float = 0.0
    bioemu_x0_support_basis_delta_ceiling_huber_beta: float = 0.05
    bioemu_x0_support_basis_delta_ceiling_path_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_use_checkpoint: bool = False
    bioemu_x0_decoder_support_head_chunk_size: int = 0
    bioemu_x0_compact_edge_encoder_sample_chunk_size: int = 0
    bioemu_x0_decoder_capacity_adapter_hidden_dim: int = 0
    bioemu_x0_decoder_capacity_adapter_depth: int = 3
    bioemu_x0_decoder_capacity_adapter_cap_ppm: float = 0.0
    bioemu_x0_decoder_capacity_adapter_scale: float = 1.0
    bioemu_x0_decoder_capacity_adapter_trainable_scale_cap: float = 0.0
    bioemu_x0_decoder_capacity_adapter_trainable_scale_init: float = 0.0
    bioemu_x0_decoder_capacity_adapter_checkpoint_path: str | None = None
    bioemu_x0_decoder_capacity_adapter_extra_hidden_dim: int = 0
    bioemu_x0_decoder_capacity_adapter_extra_depth: int = 3
    bioemu_x0_decoder_capacity_adapter_extra_cap_ppm: float = 0.0
    bioemu_x0_decoder_capacity_adapter_extra_scale: float = 1.0
    bioemu_x0_decoder_capacity_adapter_extra_expert_count: int = 1
    bioemu_x0_decoder_capacity_adapter_extra_factorized_rank: int = 0
    bioemu_x0_decoder_capacity_adapter_extra_effective_hidden_dim: int = 0
    bioemu_x0_decoder_capacity_adapter_extra_effective_factorized_rank: int = 0
    bioemu_x0_decoder_capacity_adapter_extra_max_factorized_params: int = 450_000_000
    bioemu_x0_decoder_capacity_adapter_extra_output_init_std: float = 0.0
    bioemu_x0_decoder_capacity_adapter_extra_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_capacity_adapter_extra_family_gated: bool = False
    bioemu_x0_decoder_capacity_adapter_extra_family_specific: bool = False
    bioemu_x0_decoder_capacity_adapter_extra_family_headed: bool = False
    bioemu_x0_decoder_capacity_adapter_aux_hidden_dim: int = 0
    bioemu_x0_decoder_capacity_adapter_aux_depth: int = 2
    bioemu_x0_decoder_capacity_adapter_aux_cap_ppm: float = 0.0
    bioemu_x0_decoder_capacity_adapter_aux_scale: float = 1.0
    bioemu_x0_decoder_capacity_adapter_aux_expert_count: int = 1
    bioemu_x0_decoder_capacity_adapter_aux_output_init_std: float = 0.0
    bioemu_x0_decoder_capacity_adapter_aux_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_decoder_capacity_adapter_aux_family_gated: bool = False
    bioemu_x0_decoder_capacity_adapter_aux_family_specific: bool = False
    bioemu_x0_decoder_capacity_adapter_aux_family_headed: bool = False
    bioemu_x0_decoder_family_affine_scale_cap: float = 0.0
    bioemu_x0_decoder_family_affine_shift_cap_ppm: float = 0.0
    bioemu_x0_decoder_family_affine_family_scales: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_stage_c_train_family_affine_only: bool = False
    bioemu_x0_family_affine_lr_multiplier: float = 1.0
    bioemu_x0_sidecar_family_affine_calibration_enabled: bool = False
    bioemu_x0_sidecar_evidence_reference_offset_enabled: bool = False
    bioemu_x0_sidecar_evidence_reference_offset_max_abs_ppm: float = 0.0
    bioemu_x0_sidecar_evidence_reference_offset_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_evidence_reference_offset_strength: float = 1.0
    bioemu_x0_sidecar_evidence_reference_offset_min_rows: int = 2
    bioemu_x0_capacity_adapter_lr_multiplier: float = 1.0
    bioemu_x0_capacity_adapter_extra_lr_multiplier: float = 1.0
    bioemu_x0_capacity_adapter_aux_lr_multiplier: float = 1.0
    bioemu_x0_posterior_energy_lr_multiplier: float = 1.0
    bioemu_x0_posterior_mode_lr_multiplier: float = 1.0
    bioemu_x0_evidence_head_lr_multiplier: float = 1.0
    bioemu_x0_prior_logit_adapter_hidden_dim: int = 0
    bioemu_x0_prior_logit_adapter_depth: int = 2
    bioemu_x0_prior_logit_adapter_scale: float = 1.0
    bioemu_x0_prior_logit_adapter_max_abs: float = 0.0
    bioemu_x0_prior_logit_adapter_output_init_std: float = 0.0
    bioemu_x0_prior_logit_adapter_set_context_dim: int = 0
    bioemu_x0_prior_logit_adapter_set_context_depth: int = 1
    bioemu_x0_prior_logit_adapter_include_base_log_prob_features: bool = False
    bioemu_x0_prior_logit_adapter_lr_multiplier: float = 1.0
    bioemu_x0_capacity_adapter_residual_oracle_loss_weight: float = 0.0
    bioemu_x0_capacity_adapter_residual_oracle_scale: float = 256.0
    bioemu_x0_capacity_adapter_residual_oracle_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_capacity_adapter_residual_oracle_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_capacity_adapter_residual_oracle_target_clip_ppm: float = 0.0
    bioemu_x0_capacity_adapter_residual_oracle_min_teacher_ess: float = 0.0
    bioemu_x0_capacity_adapter_residual_oracle_max_teacher_prior_blend: float = 0.0
    bioemu_x0_capacity_adapter_residual_oracle_warmup_epochs: int = 0
    bioemu_x0_capacity_adapter_residual_oracle_start_scale: float = 1.0
    bioemu_x0_support_basis_residual_oracle_loss_weight: float = 0.0
    bioemu_x0_support_basis_residual_oracle_scale: float = 256.0
    bioemu_x0_support_basis_residual_oracle_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_support_basis_residual_oracle_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_support_basis_residual_oracle_path_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_support_basis_residual_oracle_path_family_weights: dict[
        str, dict[str, float]
    ] = field(default_factory=dict)
    bioemu_x0_support_basis_residual_oracle_target_clip_ppm: float = 0.0
    bioemu_x0_support_basis_residual_oracle_min_teacher_ess: float = 0.0
    bioemu_x0_support_basis_residual_oracle_max_teacher_prior_blend: float = 0.0
    bioemu_x0_support_basis_residual_oracle_warmup_epochs: int = 0
    bioemu_x0_support_basis_residual_oracle_start_scale: float = 1.0
    bioemu_x0_hn_shared_q_energy_assignment_loss_weight: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_scale: float = 256.0
    bioemu_x0_hn_shared_q_energy_assignment_temperature: float = 1.0
    bioemu_x0_hn_shared_q_energy_assignment_target_clip_ppm: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_min_residual_ppm: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_residual_power: float = 1.0
    bioemu_x0_hn_shared_q_energy_assignment_min_teacher_ess: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_max_teacher_prior_blend: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_positive_mass_weight: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_target_side_margin_ppm: float = 0.0
    bioemu_x0_hn_shared_q_energy_assignment_warmup_epochs: int = 0
    bioemu_x0_hn_shared_q_energy_assignment_start_scale: float = 1.0
    bioemu_x0_hn_shared_q_generated_prior_loss_weight: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_scale: float = 256.0
    bioemu_x0_hn_shared_q_generated_prior_temperature: float = 1.0
    bioemu_x0_hn_shared_q_generated_prior_target_clip_ppm: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_min_residual_ppm: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_residual_power: float = 1.0
    bioemu_x0_hn_shared_q_generated_prior_min_teacher_ess: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_max_teacher_prior_blend: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_positive_mass_weight: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_target_side_margin_ppm: float = 0.0
    bioemu_x0_hn_shared_q_generated_prior_warmup_epochs: int = 0
    bioemu_x0_hn_shared_q_generated_prior_start_scale: float = 1.0
    bioemu_x0_freeze_eval_support_seed: bool = False
    bioemu_x0_eval_support_seed_epoch: int = 4
    bioemu_x0_freeze_train_support_seed: bool = False
    bioemu_x0_train_support_seed_epoch: int = 1
    bioemu_x0_train_support_seed_epochs: list[int] = field(default_factory=list)
    bioemu_x0_train_support_seed_offset: int = 0
    bioemu_x0_active_subset_initial_epochs: int = 0
    bioemu_x0_active_subset_train_examples: int = 0
    bioemu_x0_active_subset_val_examples: int = 0
    bioemu_x0_active_subset_strategy: str = "family_target_coverage"
    bioemu_x0_active_subset_seed_offset: int = 0
    bioemu_x0_active_subset_protect_replay_entities: bool = False
    bioemu_x0_active_subset_protected_entity_uids: list[str] = field(
        default_factory=list
    )
    bioemu_x0_training_only_teacher_entity_uids: list[str] = field(
        default_factory=list
    )
    bioemu_x0_active_subset_filter_to_bridge_audit_sidecar_entities: bool = False
    bioemu_x0_active_subset_final_acceptance_eligible: bool = False
    bioemu_x0_active_subset_full_dataset_resume_required: bool = True
    bioemu_x0_active_subset_triage_gate_enabled: bool = False
    bioemu_x0_active_subset_triage_stop_on_failure: bool = False
    bioemu_x0_active_subset_triage_min_macro_ccc: float = 0.0
    bioemu_x0_active_subset_triage_min_family_ccc: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_active_subset_triage_use_mean_family_ccc: bool = True
    bioemu_x0_active_subset_triage_floor_tolerance: float = 0.0
    bioemu_x0_active_subset_triage_max_sidecar_to_generated_macro_gap: float = 0.0
    bioemu_x0_active_subset_triage_max_mean_sidecar_to_generated_macro_gap: float = 0.0
    bioemu_x0_same_conformer_row_gate_enabled: bool = False
    bioemu_x0_same_conformer_row_gate_focus_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_same_conformer_row_gate_required_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_same_conformer_row_gate_min_family_count: int = 2
    bioemu_x0_same_conformer_row_gate_min_rows: int = 1
    bioemu_x0_same_conformer_row_gate_max_scan_examples: int = 0
    bioemu_x0_same_conformer_supervision_coherence_enabled: bool = False
    bioemu_x0_same_conformer_supervision_focus_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_same_conformer_supervision_required_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_same_conformer_supervision_min_family_count: int = 2
    bioemu_x0_same_conformer_supervision_max_residue_groups: int = 0
    bioemu_x0_widen_checkpoint_tensors: bool = False
    bioemu_x0_optimizer_name: str = "adamw"
    bioemu_x0_optimizer_foreach: bool | None = None
    bioemu_x0_optimizer_finite_update_guard_enabled: bool = True
    bioemu_x0_joint_teacher_consistency_weight: float = 0.05
    bioemu_x0_full_sidecar_teacher_loss_weight: float = 0.0
    bioemu_x0_full_sidecar_teacher_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_full_sidecar_teacher_family_macro_weight: float = 1.0
    bioemu_x0_full_sidecar_support_delta_teacher_loss_weight: float = 0.0
    bioemu_x0_full_sidecar_support_delta_teacher_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_full_sidecar_support_delta_teacher_family_macro_weight: float = 1.0
    bioemu_x0_full_sidecar_support_delta_teacher_min_valid_samples: int = 2
    bioemu_x0_posterior_family_loss_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_family_ccc_loss_weight: float = 0.0
    bioemu_x0_posterior_family_ccc_min_points: int = 3
    bioemu_x0_posterior_family_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_posterior_family_ccc_floors: dict[str, float] = field(default_factory=dict)
    bioemu_x0_online_simplex_proxy_loss_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_online_simplex_proxy_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_online_simplex_proxy_auto_gap_weight_scale: float = 0.0
    bioemu_x0_online_simplex_proxy_auto_gap_max_multiplier: float = 3.0
    bioemu_x0_online_simplex_proxy_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_ccc_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_online_simplex_proxy_macro_ccc_floor: float = 0.0
    bioemu_x0_online_simplex_proxy_macro_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_target_ccc_loss_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_target_ccc_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_online_simplex_proxy_nonregression_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_nonregression_ccc_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_multiseed_active_set_floor_loss_weight: float = 0.0
    bioemu_x0_multiseed_active_set_family_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_multiseed_active_set_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_multiseed_active_set_seed_family_weights: dict[
        str, dict[str, float]
    ] = field(default_factory=dict)
    bioemu_x0_multiseed_active_set_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_multiseed_active_set_auto_gap_weight_scale: float = 0.0
    bioemu_x0_multiseed_active_set_auto_gap_max_multiplier: float = 3.0
    bioemu_x0_multiseed_active_set_include_gradient_projection: bool = False
    bioemu_x0_multiseed_active_set_min_points: int = 3
    bioemu_x0_online_simplex_proxy_uncertainty_weight: float = 0.0
    bioemu_x0_online_simplex_proxy_guard_weight: float = 0.0
    bioemu_x0_goal_ccc_ladder_contract: bool = False
    bioemu_x0_goal_ccc_floor_schedule_contract: bool = False
    bioemu_x0_goal_ccc_cosatisfaction_contract: bool = False
    bioemu_x0_goal_ccc_fullpath_contract: bool = False
    bioemu_x0_goal_ccc_worst_gap_contract: bool = False
    bioemu_x0_goal_ccc_floor_gap_power: float = 1.0
    bioemu_x0_goal_ccc_floor_worst_family_mix: float = 0.0
    bioemu_x0_goal_ccc_family_gap_priority_contract: bool = False
    bioemu_x0_goal_ccc_floor_auto_gap_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_goal_ccc_floor_auto_gap_weight_scale: float = 0.0
    bioemu_x0_goal_ccc_floor_auto_gap_max_multiplier: float = 1.0
    bioemu_x0_goal_ccc_macro_floor_contract: bool = False
    bioemu_x0_goal_ccc_macro_floor: float = 0.0
    bioemu_x0_goal_ccc_macro_floor_mix: float = 0.0
    bioemu_x0_goal_ccc_macro_health_contract: bool = False
    bioemu_x0_goal_ccc_macro_health_ess_floor: float = 0.0
    bioemu_x0_goal_ccc_macro_health_top_mass_cap: float = 0.0
    bioemu_x0_goal_ccc_macro_unweighted_contract: bool = False
    bioemu_x0_goal_ccc_macro_additive_contract: bool = False
    bioemu_x0_goal_ccc_low_tail_contract: bool = False
    bioemu_x0_goal_ccc_low_tail_floor: float = 0.0
    bioemu_x0_goal_ccc_low_tail_fraction: float = 0.4
    bioemu_x0_goal_ccc_low_tail_mix: float = 0.0
    bioemu_x0_goal_ccc_low_tail_additive: bool = True
    bioemu_x0_goal_ccc_low_tail_checkpoint_contract: bool = False
    bioemu_x0_goal_ccc_source_reference_nonregression_contract: bool = False
    bioemu_x0_goal_ccc_source_reference_nonregression_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_goal_ccc_all_family_cosatisfaction_contract: bool = False
    bioemu_x0_goal_ccc_coverage_aux_cosatisfaction_contract: bool = False
    bioemu_x0_goal_ccc_coverage_weighted_aux_cosatisfaction_contract: bool = False
    bioemu_x0_goal_ccc_floor_schedule_start_epoch: int = 0
    bioemu_x0_goal_ccc_floor_schedule_ramp_epochs: int = 0
    bioemu_x0_goal_ccc_floor_schedule_start_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_protected_family_ensemble_loss_weight: float = 0.0
    bioemu_x0_protected_family_ensemble_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_protected_family_ensemble_warmup_epochs: int = 0
    bioemu_x0_protected_family_ensemble_start_scale: float = 1.0
    bioemu_x0_checkpoint_metric: str = "loss"
    bioemu_x0_checkpoint_metric_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_checkpoint_metric_target_gap_max_blend: float = 0.0
    bioemu_x0_checkpoint_metric_target_gap_low_tail_fraction: float = 0.0
    bioemu_x0_checkpoint_metric_target_gap_low_tail_blend: float = 0.0
    bioemu_x0_checkpoint_metric_nonregression_ccc_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_checkpoint_metric_nonregression_penalty_weight: float = 0.0
    bioemu_x0_checkpoint_metric_nonregression_hard_gate: bool = False
    bioemu_x0_checkpoint_metric_require_model_dependent_source: bool = False
    bioemu_x0_checkpoint_metric_support_within_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_checkpoint_metric_support_within_penalty_weight: float = 0.0
    bioemu_x0_checkpoint_metric_support_within_hard_gate: bool = False
    bioemu_x0_checkpoint_metric_health_gates: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_checkpoint_metric_health_penalty_weight: float = 0.0
    bioemu_x0_checkpoint_metric_health_hard_gate: bool = False
    bioemu_x0_mid_epoch_checkpoint_interval_examples: int = 0
    bioemu_x0_multiseed_checkpoint_enabled: bool = False
    bioemu_x0_multiseed_checkpoint_seed_epochs: list[int] = field(default_factory=list)
    bioemu_x0_multiseed_checkpoint_summary_path: str | None = None
    bioemu_x0_emit_initial_multiseed_checkpoint_summary: bool = False
    bioemu_x0_restore_best_checkpoint_on_metric_regression: bool = False
    bioemu_x0_emit_ready_only_for_checkpoint_metric_acceptance: bool = False
    bioemu_x0_disable_async_offline_ready_emission: bool = False
    bioemu_x0_nonregression_gradient_projection_enabled: bool = False
    bioemu_x0_nonregression_gradient_projection_strength: float = 1.0
    bioemu_x0_nonregression_gradient_projection_min_guard_norm: float = 1.0e-12
    bioemu_x0_nonregression_gradient_projection_skip_on_oom: bool = True
    bioemu_x0_backward_oom_skip_enabled: bool = False
    bioemu_x0_compact_geometry_runtime_diagnostics_required: bool = False
    bioemu_x0_compact_geometry_oom_fallback_allowed: bool = True
    bioemu_x0_compact_geometry_force_empty_edges: bool = False
    bioemu_x0_provider_sample_oom_retry_enabled: bool = False
    bioemu_x0_provider_sample_oom_retry_sample_count: int = 0
    bioemu_x0_cuda_empty_cache_each_example: bool = False
    bioemu_x0_nonregression_gradient_projection_include_decoder_trust_region: bool = False
    bioemu_x0_nonregression_gradient_projection_include_online_simplex: bool = False
    bioemu_x0_nonregression_gradient_projection_include_shared_oracle_decoder: bool = False
    bioemu_x0_nonregression_gradient_projection_include_same_conformer_generated_prior: bool = False
    bioemu_x0_family_expert_aggregate_mode: str = "sum"
    bioemu_x0_family_expert_residual_cap: float = 0.0
    bioemu_x0_family_expert_conflict_penalty_weight: float = 0.0
    bioemu_x0_detach_evidence_head_inputs: bool = True
    bioemu_x0_evidence_head_chunk_size: int = 4
    bioemu_x0_evidence_head_use_checkpoint: bool = False
    bioemu_x0_evidence_head_hidden_dim: int = 0
    bioemu_x0_evidence_head_depth: int = 2
    bioemu_x0_supervised_support_envelope_loss_weight: float = 0.0
    bioemu_x0_supervised_support_envelope_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_supervised_support_envelope_softness: float = 0.05
    bioemu_x0_supervised_support_envelope_warmup_epochs: int = 0
    bioemu_x0_supervised_support_envelope_start_scale: float = 1.0
    bioemu_x0_supervised_support_center_loss_weight: float = 0.0
    bioemu_x0_supervised_support_center_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_supervised_support_center_softness: float = 0.05
    bioemu_x0_supervised_support_center_gap_focus_scale: float = 0.0
    bioemu_x0_supervised_support_center_warmup_epochs: int = 0
    bioemu_x0_supervised_support_center_start_scale: float = 1.0
    bioemu_x0_raw_sample_support_envelope_loss_weight: float = 0.0
    bioemu_x0_raw_sample_support_envelope_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_raw_sample_support_envelope_softness: float = 0.02
    bioemu_x0_raw_sample_support_envelope_margin_sigma_scale: float = 1.0
    bioemu_x0_raw_sample_support_envelope_margin_cap_by_family: dict[
        str, float
    ] = field(
        default_factory=lambda: {
            "HN": 0.05,
            "N": 0.35,
            "CA": 0.25,
            "CB": 0.35,
            "C'": 0.18,
        }
    )
    bioemu_x0_raw_sample_support_envelope_min_rows: int = 1
    bioemu_x0_raw_sample_support_envelope_warmup_epochs: int = 0
    bioemu_x0_raw_sample_support_envelope_start_scale: float = 1.0
    bioemu_x0_raw_sample_support_width_floor_loss_weight: float = 0.0
    bioemu_x0_raw_sample_support_width_floor_min_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_raw_sample_support_width_floor_max_by_family: dict[
        str, float
    ] = field(
        default_factory=lambda: {
            "HN": 0.60,
            "N": 4.00,
            "CA": 3.00,
            "CB": 6.00,
            "C'": 1.50,
        }
    )
    bioemu_x0_raw_sample_support_width_floor_gap_scale: float = 0.5
    bioemu_x0_support_width_floor_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_support_width_floor_auto_gap_scale: float = 0.0
    bioemu_x0_support_width_floor_auto_gap_max_multiplier: float = 2.0
    bioemu_x0_raw_sample_support_tail_loss_weight: float = 0.0
    bioemu_x0_raw_sample_support_tail_fraction: float = 0.125
    bioemu_x0_raw_sample_support_ranked_tail_loss_weight: float = 0.0
    bioemu_x0_raw_sample_support_ranked_tail_fraction: float = 0.125
    bioemu_x0_tail_risk_dual_loss_weight: float = 0.0
    bioemu_x0_tail_risk_dual_target_fraction: float = 0.02
    bioemu_x0_tail_risk_dual_loss_kind: str = "quadratic"
    bioemu_x0_tail_risk_dual_min_active_samples: int = 1
    bioemu_x0_sample_support_delta_enabled: bool = False
    bioemu_x0_sample_support_delta_max_abs_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sample_support_delta_scale: float = 1.0
    bioemu_x0_sample_support_delta_scale_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sample_support_delta_centered: bool = True
    bioemu_x0_sample_support_delta_init_std: float = 0.0
    bioemu_x0_sample_support_delta_lr_multiplier: float = -1.0
    bioemu_x0_sample_support_delta_width_floor_loss_weight: float = 0.0
    bioemu_x0_sample_support_delta_width_floor_min_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_mechanism_support_envelope_loss_weight: float = 0.0
    bioemu_x0_mechanism_support_envelope_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_mechanism_support_envelope_softness: float = 0.05
    bioemu_x0_mechanism_support_envelope_risk_scale: float = 1.0
    bioemu_x0_mechanism_support_envelope_min_context_weight: float = 1.0
    bioemu_x0_mechanism_support_envelope_warmup_epochs: int = 0
    bioemu_x0_mechanism_support_envelope_start_scale: float = 1.0
    bioemu_x0_conditional_support_envelope_loss_weight: float = 0.0
    bioemu_x0_conditional_support_envelope_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_conditional_support_envelope_softness: float = 0.05
    bioemu_x0_conditional_support_envelope_risk_scale: float = 1.0
    bioemu_x0_conditional_support_envelope_min_context_weight: float = 1.0
    bioemu_x0_conditional_support_envelope_gap_margin: float = 0.0
    bioemu_x0_conditional_support_envelope_gap_focus_scale: float = 0.0
    bioemu_x0_conditional_support_envelope_warmup_epochs: int = 0
    bioemu_x0_conditional_support_envelope_start_scale: float = 1.0
    bioemu_x0_directional_support_expansion_loss_weight: float = 0.0
    bioemu_x0_directional_support_expansion_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_directional_support_expansion_softness: float = 0.05
    bioemu_x0_directional_support_expansion_risk_scale: float = 1.0
    bioemu_x0_directional_support_expansion_min_context_weight: float = 1.0
    bioemu_x0_directional_support_expansion_margin: float = 0.0
    bioemu_x0_directional_support_expansion_warmup_epochs: int = 0
    bioemu_x0_directional_support_expansion_start_scale: float = 1.0
    bioemu_x0_directional_support_expansion_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_directional_support_expansion_auto_gap_weight_scale: float = 0.0
    bioemu_x0_directional_support_expansion_auto_gap_max_multiplier: float = 2.5
    bioemu_x0_shared_oracle_decoder_loss_weight: float = 0.0
    bioemu_x0_shared_oracle_decoder_scale: float = 256.0
    bioemu_x0_shared_oracle_decoder_oracle_solver: str = ""
    bioemu_x0_shared_oracle_decoder_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_shared_oracle_decoder_ccc_loss_weight: float = 0.0
    bioemu_x0_shared_oracle_decoder_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_shared_oracle_decoder_ccc_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_shared_oracle_decoder_energy_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_shared_oracle_decoder_use_sidecar_oracle: bool = False
    bioemu_x0_shared_oracle_decoder_metric_contract: bool = False
    bioemu_x0_shared_oracle_decoder_metric_contract_version: str = ""
    bioemu_x0_shared_oracle_decoder_required_diagnostics: list[str] = field(
        default_factory=list
    )
    bioemu_x0_shared_oracle_decoder_metric_target_ccc: float = 0.95
    bioemu_x0_shared_oracle_decoder_metric_health_gates: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_shared_oracle_decoder_uncertainty_penalty_weight: float = 0.0
    bioemu_x0_shared_oracle_decoder_uncertainty_threshold: float = 0.0
    bioemu_x0_shared_oracle_decoder_ood_penalty_weight: float = 0.0
    bioemu_x0_shared_oracle_decoder_ood_threshold: float = 0.0
    bioemu_x0_leave_family_out_oracle_decoder_loss_weight: float = 0.0
    bioemu_x0_leave_family_out_oracle_decoder_scale: float = 256.0
    bioemu_x0_leave_family_out_oracle_decoder_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_leave_family_out_oracle_decoder_warmup_epochs: int = 0
    bioemu_x0_leave_family_out_oracle_decoder_start_scale: float = 1.0
    bioemu_x0_pairwise_cooccurrence_decoder_loss_weight: float = 0.0
    bioemu_x0_pairwise_cooccurrence_decoder_scale: float = 256.0
    bioemu_x0_pairwise_cooccurrence_decoder_pair_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_pairwise_cooccurrence_decoder_warmup_epochs: int = 0
    bioemu_x0_pairwise_cooccurrence_decoder_start_scale: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_scale: float = 256.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_focus_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_decoder_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_decoder_fallback_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_decoder_fallback_min_rows: int = 0
    bioemu_x0_same_conformer_cosatisfaction_decoder_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_same_conformer_cosatisfaction_decoder_min_family_count: int = 2
    bioemu_x0_same_conformer_cosatisfaction_decoder_gap_focus_scale: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_worst_family_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_family_balanced_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_weight_power: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_enabled: bool = False
    bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_min_family_count: int = 0
    bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_row_min_family_count: int = 1
    bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_min_rows: int = 1
    bioemu_x0_same_conformer_cosatisfaction_decoder_warmup_epochs: int = 0
    bioemu_x0_same_conformer_cosatisfaction_decoder_start_scale: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_scale: float = 256.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_temperature: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_focus_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_fallback_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_fallback_min_rows: int = 0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_min_family_count: int = 2
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_min_teacher_ess: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_max_teacher_prior_blend: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_max_teacher_top_mass: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_gap_focus_scale: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_worst_family_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_family_balanced_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_enabled: bool = False
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_min_family_count: int = 0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_row_min_family_count: int = 1
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_min_rows: int = 1
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_mass_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_fraction: float = 0.125
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_min_count: int = 1
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_target_mass_scale: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_listwise_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_top1_nll_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_ess_floor: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_ess_floor_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_top_mass_cap: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_top_mass_cap_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_prior_blend: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_gap_power: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_std_cap: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_std_cap_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_range_cap: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_range_cap_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_warmup_epochs: int = 0
    bioemu_x0_same_conformer_cosatisfaction_energy_assignment_start_scale: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_scale: float = 256.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_temperature: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_focus_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_fallback_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_fallback_min_rows: int = 0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_min_family_count: int = 2
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_min_teacher_ess: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_max_teacher_prior_blend: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_max_teacher_top_mass: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_gap_focus_scale: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_worst_family_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_family_balanced_error_mix: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_weight_power: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_enabled: bool = False
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_min_family_count: int = 0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_row_min_family_count: int = 1
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_min_rows: int = 1
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_mass_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_fraction: float = 0.125
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_min_count: int = 1
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_target_mass_scale: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_listwise_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_top1_nll_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_ess_floor: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_ess_floor_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_top_mass_cap: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_top_mass_cap_loss_weight: float = 0.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_log_prob_source: str = "raw"
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_enabled: bool = False
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_separate_clip_enabled: bool = False
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_clip_norm_multiplier: float = 1.0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_warmup_epochs: int = 0
    bioemu_x0_same_conformer_cosatisfaction_generated_prior_start_scale: float = 1.0
    bioemu_x0_same_conformer_coverage_aux_decoder_loss_weight: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_decoder_scale: float = 256.0
    bioemu_x0_same_conformer_coverage_aux_decoder_focus_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_same_conformer_coverage_aux_decoder_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_coverage_aux_decoder_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_same_conformer_coverage_aux_decoder_min_family_count: int = 2
    bioemu_x0_same_conformer_coverage_aux_decoder_gap_focus_scale: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_decoder_worst_family_error_mix: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_decoder_family_balanced_error_mix: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_decoder_coverage_weight_power: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_decoder_warmup_epochs: int = 0
    bioemu_x0_same_conformer_coverage_aux_decoder_start_scale: float = 1.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_loss_weight: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_scale: float = 256.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_temperature: float = 1.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_focus_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_required_families: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_min_family_count: int = 2
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_min_teacher_ess: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_max_teacher_prior_blend: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_max_teacher_top_mass: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_gap_focus_scale: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_worst_family_error_mix: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_family_balanced_error_mix: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_coverage_weight_power: float = 0.0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_warmup_epochs: int = 0
    bioemu_x0_same_conformer_coverage_aux_energy_assignment_start_scale: float = 1.0
    bioemu_x0_phase_aware_co_support_report_path: str = ""
    bioemu_x0_phase_aware_co_support_pair_multiplier: float = 1.75
    bioemu_x0_phase_aware_co_support_pairs: list[str] = field(default_factory=list)
    bioemu_x0_phase_aware_co_support_auto_from_gap_enabled: bool = False
    bioemu_x0_phase_aware_co_support_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_phase_aware_co_support_auto_gap_threshold: float = 0.05
    bioemu_x0_phase_aware_co_support_auto_gap_scale: float = 1.0
    bioemu_x0_rowwise_oracle_decoder_loss_weight: float = 0.0
    bioemu_x0_rowwise_oracle_decoder_scale: float = 256.0
    bioemu_x0_rowwise_oracle_decoder_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_rowwise_oracle_decoder_family_scales: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_rowwise_oracle_decoder_min_teacher_ess: float = 0.0
    bioemu_x0_rowwise_oracle_decoder_max_teacher_prior_blend: float = 0.0
    bioemu_x0_rowwise_oracle_decoder_warmup_epochs: int = 0
    bioemu_x0_rowwise_oracle_decoder_start_scale: float = 1.0
    bioemu_x0_rowwise_support_rank_loss_weight: float = 0.0
    bioemu_x0_rowwise_support_rank_scale: float = 256.0
    bioemu_x0_rowwise_support_rank_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_rowwise_support_rank_family_scales: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_rowwise_support_rank_margin: float = 0.05
    bioemu_x0_rowwise_support_rank_min_teacher_ess: float = 0.0
    bioemu_x0_rowwise_support_rank_max_teacher_prior_blend: float = 0.0
    bioemu_x0_rowwise_support_rank_risk_scale: float = 0.0
    bioemu_x0_rowwise_support_rank_min_context_weight: float = 1.0
    bioemu_x0_rowwise_support_rank_gap_focus_scale: float = 0.0
    bioemu_x0_rowwise_support_rank_warmup_epochs: int = 0
    bioemu_x0_rowwise_support_rank_start_scale: float = 1.0
    bioemu_x0_rowwise_support_rank_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_rowwise_support_rank_auto_gap_weight_scale: float = 0.0
    bioemu_x0_rowwise_support_rank_auto_gap_max_multiplier: float = 2.5
    bioemu_x0_sidecar_observed_support_rank_loss_weight: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_scale: float = 256.0
    bioemu_x0_sidecar_observed_support_rank_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_observed_support_rank_family_scales: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_observed_support_rank_margin: float = 0.05
    bioemu_x0_sidecar_observed_support_rank_min_teacher_ess: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_max_teacher_prior_blend: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_risk_scale: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_min_context_weight: float = 1.0
    bioemu_x0_sidecar_observed_support_rank_gap_focus_scale: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_warmup_epochs: int = 0
    bioemu_x0_sidecar_observed_support_rank_start_scale: float = 1.0
    bioemu_x0_sidecar_observed_support_rank_auto_gap_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_observed_support_rank_auto_gap_weight_scale: float = 0.0
    bioemu_x0_sidecar_observed_support_rank_auto_gap_max_multiplier: float = 2.5
    bioemu_x0_sidecar_shared_q_oracle_distill_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_distill_scale: float = 256.0
    bioemu_x0_sidecar_shared_q_oracle_distill_energy_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_external_global_family_teacher_enabled: bool = False
    bioemu_x0_external_global_family_teacher_summary_path: str = ""
    bioemu_x0_external_global_family_teacher_weights_path: str = ""
    bioemu_x0_external_global_family_teacher_predictions_path: str = ""
    bioemu_x0_external_global_family_teacher_registry_latest_path: str = ""
    bioemu_x0_external_global_family_teacher_reload_at_epoch_boundary: bool = False
    bioemu_x0_external_global_family_teacher_min_ess_mean: float = 0.0
    bioemu_x0_external_global_family_teacher_max_top_mass: float = 1.0
    bioemu_x0_external_global_family_teacher_min_ucbshift_pass_rate: float = 0.999
    bioemu_x0_external_global_family_teacher_distill_weight: float = 0.0
    bioemu_x0_external_global_family_teacher_energy_target_weight: float = 0.0
    bioemu_x0_external_global_family_teacher_contrastive_energy_weight: float = 0.0
    bioemu_x0_external_global_family_teacher_contrastive_margin: float = 0.04
    bioemu_x0_external_global_family_teacher_contrastive_hard_negative_fraction: float = 0.02
    bioemu_x0_external_global_family_teacher_contrastive_positive_mass_weight: float = 0.0
    bioemu_x0_external_global_family_teacher_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_external_global_family_teacher_require_required_family_complete: bool = False
    bioemu_x0_external_global_family_teacher_coverage_weight_power: float = 0.0
    bioemu_x0_external_global_family_teacher_allow_diagnostic_active_subset_probe: bool = False
    bioemu_x0_external_global_family_teacher_require_paper_cs_reweighting_lineage: bool = False
    bioemu_x0_external_global_family_teacher_assert_vanilla_source_index_identity: bool = False
    bioemu_x0_no_oracle_response_envelope_gate_required: bool = False
    bioemu_x0_no_oracle_response_envelope_gate_metric: str = ""
    bioemu_x0_no_oracle_response_envelope_focus_families: list[str] = field(
        default_factory=list
    )
    bioemu_x0_no_oracle_response_envelope_baseline_family_ccc: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_no_oracle_response_envelope_gap_to_target: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_no_oracle_response_envelope_required_family_ccc: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_no_oracle_response_envelope_min_improvement: float = 0.0
    bioemu_x0_block_new_ucbshift_until_response_envelope_gate: bool = False
    bioemu_x0_block_new_ucbshift_reason: str = ""
    bioemu_x0_sidecar_shared_q_oracle_solver: str = "energy_softmax"
    bioemu_x0_sidecar_shared_q_oracle_distill_oracle_solver: str = ""
    bioemu_x0_sidecar_shared_q_energy_target_oracle_solver: str = ""
    bioemu_x0_sidecar_shared_q_oracle_bme_steps: int = 64
    bioemu_x0_sidecar_shared_q_oracle_bme_learning_rate: float = 0.20
    bioemu_x0_sidecar_shared_q_oracle_bme_kl_weight: float = 0.01
    bioemu_x0_sidecar_shared_q_oracle_bme_likelihood_kind: str = "gaussian"
    bioemu_x0_sidecar_shared_q_oracle_bme_student_df: float = 4.0
    bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_outlier_prob: float = 0.05
    bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_bad_scale: float = 8.0
    bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_l2: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_learning_rate: float = 0.05
    bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_max_abs_ppm: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_restart_count: int = 0
    bioemu_x0_sidecar_shared_q_oracle_bme_restart_noise: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_restart_seed: int = 0
    bioemu_x0_sidecar_shared_q_oracle_bme_energy_warmstart_scales: list[
        float
    ] = field(default_factory=list)
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_floor: float = 0.95
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_min_points: int = 3
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_family_weight_mode: str = "cell"
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_learning_rate: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_max_multiplier: float = 1.0
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_update_every: int = 1
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_warmup_steps: int = 0
    bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_loss_kind: str = "quadratic"
    bioemu_x0_sidecar_shared_q_oracle_bme_selection_metric: str = "objective"
    bioemu_x0_sidecar_shared_q_oracle_bme_ess_floor: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_ess_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_cap: float = 0.0
    bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_scale: float = 256.0
    bioemu_x0_sidecar_shared_q_energy_target_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_shared_q_energy_target_clip: float = 6.0
    bioemu_x0_sidecar_shared_q_energy_target_huber_beta: float = 0.5
    bioemu_x0_sidecar_shared_q_energy_target_std_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_shape_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_rank_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_corr_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_energy_target_std_huber_beta: float = 0.25
    bioemu_x0_sidecar_shared_q_energy_target_shape_huber_beta: float = 0.5
    bioemu_x0_sidecar_shared_q_energy_target_rank_margin: float = 0.25
    bioemu_x0_sidecar_shared_q_energy_target_rank_top_fraction: float = 0.10
    bioemu_x0_sidecar_pair_rank_energy_loss_weight: float = 0.0
    bioemu_x0_sidecar_pair_rank_energy_pair_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_pair_rank_energy_rank_top_fraction: float = 0.10
    bioemu_x0_sidecar_pair_rank_energy_rank_margin: float = 0.15
    bioemu_x0_sidecar_pair_rank_energy_imbalance_weight: float = 0.25
    bioemu_x0_sidecar_pair_rank_energy_min_cells_per_family: int = 1
    bioemu_x0_response_ceiling_pair_manifest_path: str = ""
    bioemu_x0_response_ceiling_pair_manifest_summary_path: str = ""
    bioemu_x0_response_ceiling_pair_manifest_pair_count: int = 0
    bioemu_x0_response_ceiling_pair_manifest_pair_count_by_family: dict[
        str, int
    ] = field(default_factory=dict)
    bioemu_x0_response_ceiling_pair_manifest_selection_policy: str = ""
    bioemu_x0_response_ceiling_pair_manifest_teacher_probe_only: bool = True
    bioemu_x0_response_ceiling_pair_manifest_final_acceptance_eligible: bool = False
    bioemu_x0_response_ceiling_pair_manifest_recommended_use: str = ""
    bioemu_x0_failed_active_face_replay_bundle_path: str = ""
    bioemu_x0_failed_active_face_replay_summary_path: str = ""
    bioemu_x0_failed_active_face_replay_row_count: int = 0
    bioemu_x0_failed_active_face_replay_row_count_by_family: dict[
        str, int
    ] = field(default_factory=dict)
    bioemu_x0_failed_active_face_replay_response_ceiling_pair_attached_rows: int = 0
    bioemu_x0_failed_active_face_replay_mean_dual_multiplier: float = 0.0
    bioemu_x0_failed_active_face_replay_max_dual_multiplier: float = 0.0
    bioemu_x0_failed_active_face_replay_phi_theta_supervision_heads: list[
        str
    ] = field(default_factory=list)
    bioemu_x0_failed_active_face_replay_teacher_probe_only: bool = True
    bioemu_x0_failed_active_face_replay_final_acceptance_eligible: bool = False
    bioemu_x0_failed_active_face_replay_use_for_latest_accepted_teacher: bool = False
    bioemu_x0_failed_active_face_replay_must_not_update_latest_accepted: bool = True
    bioemu_x0_failed_active_face_replay_recommended_use: str = ""
    bioemu_x0_sidecar_evidence_likelihood_energy_scale: float = 0.0
    bioemu_x0_sidecar_evidence_likelihood_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_evidence_likelihood_cell_count_power: float = 0.0
    bioemu_x0_sidecar_evidence_likelihood_kind: str = "gaussian"
    bioemu_x0_sidecar_evidence_likelihood_student_df: float = 4.0
    bioemu_x0_sidecar_evidence_likelihood_good_bad_outlier_prob: float = 0.05
    bioemu_x0_sidecar_evidence_likelihood_good_bad_bad_scale: float = 8.0
    bioemu_x0_evidence_likelihood_cell_count_power: float = 0.0
    bioemu_x0_evidence_likelihood_base_energy_scale: float = 1.0
    bioemu_x0_sidecar_weighted_observable_loss_weight: float = 0.0
    bioemu_x0_sidecar_weighted_observable_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_weighted_observable_ccc_loss_weight: float = 0.0
    bioemu_x0_sidecar_weighted_observable_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_sidecar_weighted_observable_ccc_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floors: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_sidecar_shared_q_witness_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_witness_ccc_floors: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_shared_q_witness_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor: float = 0.0
    bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_sidecar_shared_q_witness_use_gradient_projection: bool = True
    bioemu_x0_posterior_mode_count: int = 1
    bioemu_x0_posterior_mode_energy_scale: float = 0.0
    bioemu_x0_posterior_mode_energy_init_std: float = 0.0
    bioemu_x0_posterior_mode_logit_scale: float = 1.0
    bioemu_x0_posterior_mode_logit_temperature: float = 1.0
    bioemu_x0_posterior_mode_logit_prior: list[float] = field(default_factory=list)
    bioemu_x0_posterior_mode_reinit_std_on_load: float = 0.0
    bioemu_x0_posterior_mode_reinit_reset_logits: bool = False
    bioemu_x0_posterior_mode_family_oracle_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_oracle_scale: float = 256.0
    bioemu_x0_posterior_mode_family_oracle_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_mode_family_oracle_map: dict[str, int] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_mode_family_energy_target_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_energy_target_scale: float = 256.0
    bioemu_x0_posterior_mode_family_energy_target_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_posterior_mode_family_energy_target_map: dict[str, int] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_mode_family_energy_target_clip: float = 6.0
    bioemu_x0_posterior_mode_family_energy_target_huber_beta: float = 0.5
    bioemu_x0_posterior_mode_family_energy_target_std_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_energy_target_corr_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_energy_target_rank_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_energy_target_rank_margin: float = 0.05
    bioemu_x0_posterior_mode_family_energy_target_rank_top_fraction: float = 0.10
    bioemu_x0_posterior_mode_diversity_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_diversity_min_js: float = 0.01
    bioemu_x0_posterior_mode_family_prediction_contrast_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_prediction_contrast_temperature: float = 0.25
    bioemu_x0_posterior_mode_family_prediction_contrast_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_posterior_mode_family_prediction_contrast_map: dict[str, int] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_mode_family_mixture_teacher_loss_weight: float = 0.0
    bioemu_x0_posterior_mode_family_mixture_teacher_scale: float = 256.0
    bioemu_x0_posterior_mode_family_mixture_teacher_mode_sharpness: float = 1.0
    bioemu_x0_posterior_mode_family_mixture_teacher_mode_kl_weight: float = 1.0
    bioemu_x0_posterior_mode_family_mixture_teacher_component_kl_weight: float = 1.0
    bioemu_x0_posterior_mode_family_mixture_teacher_mixture_kl_weight: float = 0.5
    bioemu_x0_posterior_mode_family_mixture_teacher_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_posterior_mode_family_mixture_teacher_map: dict[str, int] = field(
        default_factory=dict
    )
    bioemu_x0_family_mode_observable_enabled: bool = False
    bioemu_x0_family_mode_observable_blend: float = 0.0
    bioemu_x0_family_mode_observable_map: dict[str, int] = field(default_factory=dict)
    bioemu_x0_residue_local_mode_observable_enabled: bool = False
    bioemu_x0_residue_local_mode_observable_blend: float = 0.0
    bioemu_x0_residue_local_mode_gate_hidden_dim: int = 0
    bioemu_x0_residue_local_mode_gate_temperature: float = 1.0
    bioemu_x0_residue_local_mode_gate_teacher_loss_weight: float = 0.0
    bioemu_x0_residue_local_mode_gate_teacher_scale: float = 256.0
    bioemu_x0_residue_local_mode_gate_teacher_hard_loss_weight: float = 0.0
    bioemu_x0_residue_local_mode_gate_teacher_entropy_loss_weight: float = 0.0
    bioemu_x0_residue_local_mode_gate_teacher_entropy_target: float = 0.0
    bioemu_x0_residue_local_mode_gate_teacher_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_residue_local_mode_gate_teacher_family_scales: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_residue_local_mode_gate_teacher_use_sidecar: bool = True
    bioemu_x0_sidecar_family_mode_observable_loss_weight: float = 0.0
    bioemu_x0_sidecar_family_mode_observable_ccc_loss_weight: float = 0.0
    bioemu_x0_sidecar_family_mode_observable_ccc_floor_loss_weight: float = 0.0
    bioemu_x0_posterior_oracle_distill_weight: float = 0.0
    bioemu_x0_posterior_oracle_distill_scale: float = 256.0
    bioemu_x0_posterior_oracle_distill_energy_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_x0_posterior_consensus_oracle_distill_weight: float = 0.0
    bioemu_x0_posterior_consensus_oracle_distill_scale: float = 256.0
    bioemu_x0_posterior_consensus_oracle_conflict_kl_threshold: float = 0.75
    bioemu_x0_posterior_consensus_oracle_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_evidence_conflict_gate_enabled: bool = False
    bioemu_x0_evidence_conflict_gate_kl_threshold: float = 0.75
    bioemu_x0_evidence_conflict_gate_min_scale: float = 0.25
    bioemu_x0_evidence_likelihood_energy_scale: float = 0.0
    bioemu_x0_evidence_likelihood_family_weights: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_hn_learned_evidence_energy_scale: float = 0.0
    bioemu_x0_cprime_learned_evidence_energy_scale: float = 0.0
    bioemu_x0_learned_evidence_energy_family_scales: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_x0_posterior_energy_hidden_dim: int = 256
    bioemu_x0_posterior_energy_init_std: float = 0.0
    bioemu_x0_posterior_energy_extra_hidden_dim: int = 0
    bioemu_x0_posterior_energy_extra_depth: int = 2
    bioemu_x0_posterior_energy_extra_scale: float = 1.0
    bioemu_x0_posterior_energy_extra_init_std: float = 0.0
    bioemu_x0_posterior_energy_extra_factorized_rank: int = 0
    bioemu_x0_posterior_energy_extra_effective_hidden_dim: int = 0
    bioemu_x0_posterior_energy_extra_effective_factorized_rank: int = 0
    bioemu_x0_posterior_energy_extra_max_factorized_params: int = 450_000_000
    bioemu_x0_posterior_energy_extra_shared_block_count: int = 0
    bioemu_x0_posterior_energy_extra_chunk_size: int = 0
    bioemu_x0_posterior_energy_extra_use_checkpoint: bool = True
    bioemu_x0_posterior_energy_extra_lr_multiplier: float = 1.0
    bioemu_x0_posterior_energy_support_attention_hidden_dim: int = 0
    bioemu_x0_posterior_energy_support_attention_head_count: int = 4
    bioemu_x0_posterior_energy_support_attention_depth: int = 1
    bioemu_x0_posterior_energy_support_attention_scale: float = 0.0
    bioemu_x0_posterior_energy_support_attention_init_std: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_hidden_dim: int = 0
    bioemu_x0_cs_latent_field_conditioning_depth: int = 1
    bioemu_x0_cs_latent_field_conditioning_residue_scale: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_energy_scale: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_energy_pooling: str = "mean"
    bioemu_x0_cs_latent_field_conditioning_energy_attention_temperature: float = 1.0
    bioemu_x0_cs_latent_field_conditioning_energy_attention_init_std: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_energy_salience_attention_weight: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_energy_std_floor: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_energy_std_floor_scale_cap: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_family_token_scale: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_family_token_temperature: float = 1.0
    bioemu_x0_cs_latent_field_conditioning_family_token_init_std: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_film_scale: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_film_init_std: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_dropout: float = 0.0
    bioemu_x0_cs_latent_field_conditioning_init_std: float = 0.0
    bioemu_x0_sequence_embedding_residue_feature_dim: int = 0
    bioemu_x0_require_sequence_embedding_active_metric: bool = False
    bioemu_x0_static_context_scale: float = 1.0
    bioemu_x0_mechanism_context_scale: float = 1.0
    bioemu_x0_posterior_energy_temperature: float = 1.0
    bioemu_x0_bounded_delta_max_abs: float = 0.0
    bioemu_x0_bounded_delta_penalty_weight: float = 1.0
    bioemu_x0_posterior_prior_kl_weight: float = 0.01
    bioemu_x0_posterior_ess_floor: float = 32.0
    bioemu_x0_posterior_entropy_floor: float = 3.0
    bioemu_x0_posterior_top_mass_cap: float = 0.10
    bioemu_x0_posterior_diversity_floor: float = 0.0
    bioemu_x0_posterior_diversity_weight: float = 0.0
    bioemu_x0_mechanism_edge_classes: list[str] = field(
        default_factory=lambda: [
            "sequence",
            "peptide_plane",
            "spatial_contact",
            "hbond",
            "ring",
            "electrostatic",
        ]
    )
    bioemu_x0_hn_mechanism_channels: list[str] = field(
        default_factory=lambda: [
            "exchange_hbond",
            "ring_orientation",
            "electrostatic",
            "terminal_disorder",
        ]
    )
    bioemu_x0_cprime_mechanism_channels: list[str] = field(
        default_factory=lambda: [
            "peptide_plane",
            "carbonyl_backbone",
            "carbonyl_hbond",
            "local_strain",
        ]
    )
    enable_bioemu_posterior_moment_diffusion: bool = False
    bioemu_posterior_inference_mode: str = "sample_reweighting"
    bioemu_posterior_chart_count: int = 8
    bioemu_posterior_covariance_mode: str = "diag_low_rank"
    bioemu_posterior_covariance_rank: int = 8
    bioemu_posterior_covariance_floor: float = 1.0e-4
    bioemu_posterior_moment_shift_mode: str = "curvature_trace_proxy"
    bioemu_posterior_sigma_point_count: int = 0
    bioemu_posterior_diffusion_steps_train: int = 1
    bioemu_posterior_diffusion_steps_val: int = 1
    bioemu_posterior_flow_teacher_steps: int = 4
    bioemu_posterior_enable_consistency_distillation: bool = False
    bioemu_posterior_likelihood_score_loss_weight: float = 0.0
    bioemu_posterior_moment_nll_weight: float = 1.0
    bioemu_posterior_covariance_calibration_weight: float = 0.0
    bioemu_posterior_covariance_floor_loss_weight: float = 0.0
    bioemu_posterior_chart_entropy_floor: float = 0.05
    bioemu_posterior_chart_entropy_loss_weight: float = 0.0
    bioemu_posterior_chart_entropy_row_loss_weight: float = 0.0
    bioemu_posterior_chart_entropy_row_floor_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_posterior_chart_entropy_row_weight_by_family: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_posterior_ess_floor: float = 0.0
    bioemu_posterior_ess_loss_weight: float = 0.0
    bioemu_checkpoint_support_safe_enabled: bool = False
    bioemu_checkpoint_support_safe_ess_floor: float = 0.0
    bioemu_checkpoint_support_safe_entropy_floor: float = 0.0
    bioemu_checkpoint_support_safe_ess_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_entropy_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_hn_abs_residual_p95_ceiling: float = 0.0
    bioemu_checkpoint_support_safe_hn_abs_residual_p95_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_hn_abs_residual_max_ceiling: float = 0.0
    bioemu_checkpoint_support_safe_hn_abs_residual_max_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_cprime_ccc_floor: float = 0.0
    bioemu_checkpoint_support_safe_cprime_ccc_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_cprime_abs_residual_p95_ceiling: float = 0.0
    bioemu_checkpoint_support_safe_cprime_abs_residual_p95_penalty_weight: float = 0.0
    bioemu_checkpoint_support_safe_cprime_abs_residual_max_ceiling: float = 0.0
    bioemu_checkpoint_support_safe_cprime_abs_residual_max_penalty_weight: float = 0.0
    bioemu_posterior_occupancy_trust_region_enabled: bool = False
    bioemu_posterior_occupancy_trust_region_weight: float = 0.0
    bioemu_posterior_occupancy_trust_region_ess_floor: float = 0.0
    bioemu_posterior_occupancy_trust_region_entropy_floor: float = 0.0
    bioemu_posterior_occupancy_trust_region_deficit_multiplier: float = 0.0
    bioemu_posterior_top_mass_cap_enabled: bool = False
    bioemu_posterior_top_mass_cap: float = 1.0
    bioemu_posterior_top_mass_cap_loss_weight: float = 0.0
    bioemu_posterior_chart_temperature_enabled: bool = False
    bioemu_posterior_chart_temperature: float = 1.0
    bioemu_low_support_loss_freeze_enabled: bool = False
    bioemu_low_support_loss_freeze_ess_floor: float = 0.0
    bioemu_low_support_loss_freeze_entropy_floor: float = 0.0
    bioemu_low_support_loss_freeze_min_scale: float = 1.0
    bioemu_low_support_loss_freeze_power: float = 2.0
    bioemu_low_support_loss_freeze_hard_raw_threshold: float = 0.0
    bioemu_low_support_loss_freeze_apply_to_measure_distill: bool = True
    bioemu_low_support_loss_freeze_apply_to_mechanism_override: bool = True
    bioemu_posterior_prior_kl_weight: float = 0.0
    bioemu_posterior_bridge_norm_weight: float = 0.0
    bioemu_posterior_tangent_drift_cap: float = 0.04
    bioemu_posterior_chart_symmetry_break_scale: float = 0.01
    bioemu_hn_ccc_loss_weight: float = 0.0
    bioemu_hn_ccc_entity_family_weight_scale: float = 0.0
    bioemu_hn_ccc_entity_family_weight_max: float = 3.0
    bioemu_hn_scale_loss_weight: float = 0.0
    bioemu_hn_bias_loss_weight: float = 0.0
    bioemu_cprime_scale_loss_weight: float = 0.0
    bioemu_cprime_bias_loss_weight: float = 0.0
    enable_bioemu_cprime_chart_bias_adapter: bool = False
    bioemu_cprime_chart_bias_max_abs: float = 0.45
    bioemu_cprime_chart_bias_family_names: list[str] = field(
        default_factory=lambda: ["C'"]
    )
    bioemu_cprime_chart_bias_uncertainty_threshold: float = 0.65
    bioemu_cprime_chart_bias_center_loss_weight: float = 0.0
    bioemu_cprime_chart_bias_l2_loss_weight: float = 0.0
    bioemu_aux_family_ccc_loss_weight: float = 0.0
    bioemu_aux_family_ccc_names: list[str] = field(default_factory=list)
    bioemu_family_ccc_target_loss_weight: float = 0.0
    bioemu_family_ccc_target: float = 0.95
    bioemu_family_ccc_target_names: list[str] = field(
        default_factory=lambda: ["HN", "N", "CA", "CB", "C'"]
    )
    bioemu_family_ccc_target_min_rows: int = 2
    bioemu_family_ccc_target_start_epoch: int = 1
    bioemu_family_ccc_target_ramp_epochs: int = 1
    bioemu_family_orthogonal_loss_weight: float = 0.0
    bioemu_family_orthogonal_target_names: list[str] = field(
        default_factory=lambda: ["HN", "C'"]
    )
    bioemu_family_orthogonal_support_names: list[str] = field(
        default_factory=lambda: ["N", "CA"]
    )
    bioemu_family_orthogonal_detach_support: bool = True
    bioemu_family_orthogonal_start_epoch: int = 1
    bioemu_family_orthogonal_ramp_epochs: int = 1
    enable_bioemu_physics_atlas_v36: bool = False
    bioemu_physics_atlas_coordinate_names: list[str] = field(
        default_factory=lambda: [
            "ring_orientation",
            "ring_occupancy",
            "exchange_hbond",
            "carbonyl_backbone",
            "peptide_plane_coupling",
            "hn_local_manifold",
            "cprime_carbonyl_manifold",
            "ring_hbond_competition",
            "solvent_exchange_risk",
            "transfer_memory_confidence",
            "transfer_support_mismatch",
            "transfer_abstain_risk",
            "electrostatic_sulfur",
            "sidechain_rotamer",
            "ensemble_component",
            "uncertainty_abstain",
        ]
    )
    bioemu_physics_sidecar_path: str | None = None
    rare_regime_atlas_path: str | None = None
    rare_regime_atlas_auto_config: bool = False
    rare_regime_atlas_auto_max_router_keys: int = 12
    rare_regime_atlas_auto_max_residue_family_keys: int = 10
    rare_regime_atlas_auto_min_rows: int = 8
    regime_router_enabled: bool = True
    mechanism_tangent_adapter_rank: int = 8
    bayesian_shrinkage_enabled: bool = True
    direction_conflict_shared_adapter_threshold: float = 0.08
    teacher_mean_policy: str = "guard_or_low_weight"
    enable_bioemu_family_tangent_adapter_bank: bool = False
    bioemu_family_tangent_max_abs_by_family: dict[str, float] = field(
        default_factory=lambda: {
            "HN": 0.04,
            "N": 0.12,
            "CA": 0.35,
            "CB": 0.45,
            "C'": 0.28,
        }
    )
    bioemu_family_chart_names_by_family: dict[str, list[str]] = field(
        default_factory=lambda: {
            "HN": [
                "ring_orientation",
                "ring_occupancy",
                "exchange_hbond",
                "hn_local_manifold",
                "ring_hbond_competition",
                "solvent_exchange_risk",
                "transfer_support_mismatch",
                "transfer_abstain_risk",
                "electrostatic_sulfur",
            ],
            "C'": [
                "carbonyl_backbone",
                "cprime_carbonyl_manifold",
                "peptide_plane_coupling",
                "exchange_hbond",
                "transfer_support_mismatch",
            ],
            "N": [
                "carbonyl_backbone",
                "peptide_plane_coupling",
                "hn_local_manifold",
                "exchange_hbond",
                "transfer_support_mismatch",
                "transfer_abstain_risk",
                "ensemble_component",
            ],
            "CA": ["carbonyl_backbone", "sidechain_rotamer", "ensemble_component"],
            "CB": ["sidechain_rotamer", "electrostatic_sulfur", "ensemble_component"],
        }
    )
    bioemu_family_gradient_conflict_loss_weight: float = 0.0
    bioemu_family_tangent_orthogonality_loss_weight: float = 0.0
    enable_bioemu_residue_family_rare_adapter: bool = False
    bioemu_residue_family_rare_adapter_keys: list[str] = field(default_factory=list)
    bioemu_residue_family_rare_adapter_max_abs_by_family: dict[str, float] = field(
        default_factory=lambda: {
            "HN": 0.08,
            "N": 6.0,
            "CA": 5.0,
            "CB": 15.0,
            "C'": 1.2,
        }
    )
    bioemu_residue_family_rare_adapter_max_abs_by_pair: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_rare_adapter_uncertainty_threshold: float = 0.75
    bioemu_residue_family_rare_adapter_uncertainty_threshold_by_pair: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_residue_family_rare_adapter_loss_weight: float = 0.0
    bioemu_residue_family_rare_adapter_delta_loss_weight: float = 0.0
    bioemu_residue_family_rare_adapter_direction_loss_weight: float = 0.0
    bioemu_residue_family_rare_adapter_hn_signed_delta_loss_weight: float = 0.0
    bioemu_residue_family_rare_adapter_hn_signed_delta_min_abs: float = 0.006
    bioemu_residue_family_rare_adapter_hn_signed_delta_max_abs: float = 0.14
    bioemu_residue_family_rare_adapter_hn_signed_delta_low_scale: float = 0.70
    bioemu_residue_family_rare_adapter_hn_signed_delta_high_scale: float = 1.25
    bioemu_residue_family_rare_adapter_hn_signed_delta_require_matrix_consensus: bool = False
    bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_consensus_min: float = 0.82
    bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_min_valid_samples: int = 32
    bioemu_residue_family_rare_adapter_delta_target_scale: float = 1.0
    bioemu_residue_family_rare_adapter_delta_target_max_by_family: dict[
        str, float
    ] = field(
        default_factory=lambda: {
            "HN": 0.08,
            "N": 4.0,
            "CA": 3.0,
            "CB": 8.0,
            "C'": 0.8,
        }
    )
    bioemu_residue_family_rare_adapter_start_epoch: int = 1
    bioemu_residue_family_rare_adapter_ramp_epochs: int = 2
    bioemu_train_only_residue_family_rare_adapter: bool = False
    bioemu_residue_family_rare_adapter_train_keys: list[str] = field(default_factory=list)
    bioemu_residue_family_rare_adapter_train_shared_heads: bool = True
    enable_bioemu_factor_bin_rare_adapter: bool = False
    bioemu_factor_bin_rare_adapter_coordinate_source: str = "physics"
    bioemu_factor_bin_rare_adapter_keys: list[str] = field(default_factory=list)
    bioemu_factor_bin_rare_adapter_max_abs_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_factor_bin_rare_adapter_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_factor_bin_rare_adapter_width_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_factor_bin_rare_adapter_uncertainty_threshold: float = 0.85
    bioemu_factor_bin_rare_adapter_uncertainty_threshold_by_key: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_factor_bin_rare_adapter_loss_weight: float = 0.0
    bioemu_factor_bin_rare_adapter_start_epoch: int = 1
    bioemu_factor_bin_rare_adapter_ramp_epochs: int = 2
    bioemu_train_only_factor_bin_rare_adapter: bool = False
    bioemu_factor_bin_rare_adapter_train_keys: list[str] = field(default_factory=list)
    bioemu_factor_bin_rare_adapter_train_shared_heads: bool = True
    enable_bioemu_mechanism_signed_router: bool = False
    bioemu_mechanism_signed_router_coordinate_source: str = "physics"
    bioemu_mechanism_signed_router_keys: list[str] = field(default_factory=list)
    bioemu_mechanism_signed_router_direction_by_key: dict[str, str | float] = field(
        default_factory=dict
    )
    bioemu_mechanism_signed_router_max_abs_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_mechanism_signed_router_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_mechanism_signed_router_width_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_mechanism_signed_router_uncertainty_threshold: float = 0.85
    bioemu_mechanism_signed_router_uncertainty_threshold_by_key: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_mechanism_signed_router_loss_weight: float = 0.0
    bioemu_mechanism_signed_router_delta_loss_weight: float = 0.0
    bioemu_mechanism_signed_router_delta_target_scale: float = 1.0
    bioemu_mechanism_signed_router_delta_target_max_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    enable_bioemu_mechanism_signed_cancellation: bool = False
    bioemu_mechanism_signed_cancellation_keys: list[str] = field(default_factory=list)
    bioemu_mechanism_signed_cancellation_strength: float = 1.0
    enable_bioemu_mechanism_signed_bias_conflict_gate: bool = False
    bioemu_mechanism_signed_bias_conflict_gate_keys: list[str] = field(
        default_factory=list
    )
    bioemu_mechanism_signed_bias_conflict_gate_strength: float = 0.0
    enable_bioemu_mechanism_signed_override_adapter: bool = False
    bioemu_mechanism_signed_override_keys: list[str] = field(default_factory=list)
    bioemu_mechanism_signed_override_force_direction: bool = False
    bioemu_mechanism_signed_override_centered_component: bool = False
    bioemu_mechanism_signed_override_loss_weight: float = 0.0
    bioemu_mechanism_signed_override_direction_loss_weight: float = 0.0
    bioemu_mechanism_signed_override_support_confidence_enabled: bool = False
    bioemu_mechanism_signed_override_support_confidence_ess_floor: float = 0.0
    bioemu_mechanism_signed_override_support_confidence_entropy_floor: float = 0.0
    bioemu_mechanism_signed_override_support_confidence_min_scale: float = 0.70
    bioemu_mechanism_signed_override_target_scale: float = 1.0
    bioemu_mechanism_signed_override_target_max_by_family: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_mechanism_signed_router_start_epoch: int = 1
    bioemu_mechanism_signed_router_ramp_epochs: int = 2
    bioemu_train_only_mechanism_signed_router: bool = False
    bioemu_mechanism_signed_router_train_keys: list[str] = field(default_factory=list)
    bioemu_mechanism_signed_router_train_shared_heads: bool = True
    bioemu_mechanism_signed_router_preserve_locked_key_outputs: bool = False
    bioemu_train_only_mechanism_signed_override_adapter: bool = False
    bioemu_mechanism_signed_override_train_keys: list[str] = field(default_factory=list)
    bioemu_mechanism_signed_override_train_shared_heads: bool = True
    bioemu_mechanism_signed_override_preserve_locked_key_outputs: bool = False
    bioemu_train_only_bridge_cs_energy_head: bool = False
    enable_bioemu_chart_bin_residual_readout: bool = False
    bioemu_chart_bin_residual_readout_coordinate_source: str = "latent"
    bioemu_chart_bin_residual_readout_keys: list[str] = field(default_factory=list)
    bioemu_chart_bin_residual_readout_delta_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_max_abs_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_width_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_gate_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_lower_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_chart_bin_residual_readout_upper_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_fixed_family_readout_offset_by_family: dict[str, float] = field(
        default_factory=dict
    )
    enable_bioemu_residue_family_chart_bin_readout: bool = False
    bioemu_residue_family_chart_bin_readout_coordinate_source: str = "physics"
    bioemu_residue_family_chart_bin_readout_keys: list[str] = field(default_factory=list)
    bioemu_residue_family_chart_bin_readout_delta_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_chart_bin_readout_max_abs_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_chart_bin_readout_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_chart_bin_readout_width_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_chart_bin_readout_gate_by_key: dict[str, float] = field(
        default_factory=dict
    )
    enable_bioemu_residue_family_directional_readout: bool = False
    bioemu_residue_family_directional_readout_keys: list[str] = field(default_factory=list)
    bioemu_residue_family_directional_readout_delta_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_directional_readout_max_abs_by_key: dict[
        str, float
    ] = field(default_factory=dict)
    bioemu_residue_family_directional_readout_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_directional_readout_margin_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_directional_readout_gate_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_directional_readout_temperature: float = 0.25
    enable_bioemu_residue_family_anchor_readout: bool = False
    bioemu_residue_family_anchor_readout_keys: list[str] = field(default_factory=list)
    bioemu_residue_family_anchor_readout_center_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_anchor_readout_max_abs_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_anchor_readout_margin_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_anchor_readout_gate_by_key: dict[str, float] = field(
        default_factory=dict
    )
    bioemu_residue_family_anchor_readout_temperature: float = 1.0
    bioemu_physics_sidecar_supervision_weight: float = 0.0
    bioemu_ring_signed_consistency_loss_weight: float = 0.0
    bioemu_carbonyl_backbone_consistency_loss_weight: float = 0.0
    bioemu_nmr_sampling_role: str = "posterior_estimator"
    enable_bioemu_latent_physical_atlas: bool = False
    bioemu_latent_atlas_coordinate_names: list[str] = field(
        default_factory=lambda: [
            "ring_current",
            "exchange",
            "alignment_mismatch",
            "sulfur_electrostatic",
            "terminal_disorder",
            "spread_collapse",
            "graph_fragility",
        ]
    )
    bioemu_latent_atlas_dim: int = 7
    bioemu_latent_metric_mode: str = "low_rank_pullback"
    bioemu_latent_metric_rank: int = 16
    bioemu_latent_contrastive_loss_weight: float = 0.0
    bioemu_latent_topology_loss_weight: float = 0.0
    bioemu_latent_normal_drift_loss_weight: float = 0.0
    enable_hn_ring_subchart_gate: bool = False
    hn_ring_subchart_names: list[str] = field(
        default_factory=lambda: [
            "ring_stable_helpful",
            "ring_transient_helpful",
            "ring_conflict_abstain",
            "ring_reverse_harmful",
        ]
    )
    hn_ring_subchart_confidence_margin: float = 0.15
    hn_ring_subchart_reliability_threshold: float = 0.65
    hn_ring_signed_gate_loss_weight: float = 0.0
    enable_hn_rare_regime_subfeature_bundles: bool = False
    hn_rare_regime_bundle_names: list[str] = field(
        default_factory=lambda: [
            "ring_current",
            "exchange_hbond",
            "alignment_evidence",
            "electrostatic_sulfur",
            "disorder_spread",
        ]
    )
    hn_subfeature_source_policy: str = "bioemu_latent_and_nmr_evidence_only"
    hn_structure_probe_policy: str = "benchmark_only"
    enable_hn_cross_fitted_chart_utility_labels: bool = False
    hn_chart_utility_label_epsilon: float = 0.002
    hn_chart_utility_label_source: str = "train_split_cross_fit_only"
    enable_hn_evidential_subchart_gate: bool = False
    hn_subchart_gate_prior_strength: float = 1.0
    hn_subchart_gate_abstain_margin: float = 0.15
    enable_hn_subchart_sparse_moe: bool = False
    hn_subchart_sparse_moe_mode: str = "entmax"
    hn_subchart_sparse_moe_top_k: int = 2
    enable_hn_counterfactual_chart_dropout: bool = False
    hn_counterfactual_chart_dropout_rate: float = 0.10
    enable_bioemu_latent_curvature_uncertainty: bool = False
    bioemu_latent_jvp_probe_count: int = 2
    bioemu_latent_curvature_loss_weight: float = 0.0
    bioemu_latent_high_curvature_abstain_threshold: float = 0.65
    enable_bioemu_latent_component_persistence_audit: bool = False
    bioemu_nmr_enable_ema_teacher: bool = True
    bioemu_nmr_ema_decay: float = 0.995
    bioemu_nmr_kl_anneal_epochs: int = 3
    bioemu_nmr_weight_temperature_warmup_epochs: int = 2
    bioemu_nmr_initial_weight_detach_epochs: int = 1
    bioemu_nmr_ring_correction_cap_warmup_epochs: int = 2
    bioemu_nmr_enable_per_entry_balanced_batching: bool = True
    bioemu_nmr_rare_regime_train_oversample_factor: float = 2.0
    j_coupling_loss_weight: float = 0.1
    noe_loss_weight: float = 0.1
    entropy_regularization_weight: float = 1e-3
    sequence_embedding_dim: int = 64
    source_embedding_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.1
    use_candidate_set_encoder: bool = False
    set_encoder_layers: int = 0
    set_encoder_heads: int = 4
    logit_temperature: float = 1.0
    enable_forward_residual_head: bool = False
    forward_residual_loss_weight: float = 0.0
    forward_residual_hidden_dim: int | None = None
    forward_residual_regularization_weight: float = 1e-4
    enable_residue_atom_residual_head: bool = False
    reconstruction_ccc_loss_weight: float = 0.0
    masked_holdout_ccc_loss_weight: float = 0.0
    evidence_likelihood_nll_weight: float = 0.0
    enable_secondary_shift_targets: bool = False
    enable_residue_geometry_features: bool = False
    secondary_reconstruction_ccc_loss_weight: float = 0.0
    secondary_masked_holdout_ccc_loss_weight: float = 0.0
    raw_family_ccc_guardrail_weight: float = 0.0
    whitened_masked_ccc_loss_weight: float = 0.0
    whitened_reconstruction_ccc_loss_weight: float = 0.0
    normalized_evidence_nll_weight: float = 0.0
    sequence_smoothness_loss_weight: float = 0.0
    residue_atom_residual_loss_weight: float = 0.0
    residue_atom_residual_regularization_weight: float = 1e-4
    bioemu_residue_atom_mean_scale_by_family: dict[str, float] = field(default_factory=dict)
    bioemu_residue_atom_mean_weight_by_family: dict[str, float] = field(default_factory=dict)
    bioemu_residue_atom_mean_family_names: list[str] = field(default_factory=list)
    bioemu_residue_atom_mean_loss_mode: str = "mse"
    bioemu_residue_atom_mean_huber_delta_by_family: dict[str, float] = field(default_factory=dict)
    bioemu_residue_atom_mean_start_epoch: int = 1
    bioemu_residue_atom_mean_ramp_epochs: int = 1
    residue_atom_mask_fraction: float = 0.2
    residue_atom_block_mask_fraction: float = 0.0
    residue_atom_eval_mask_fraction: float | None = None
    residue_atom_eval_block_mask_fraction: float | None = None
    residue_atom_mask_fraction_by_family: dict[str, float] = field(
        default_factory=dict
    )
    residue_atom_eval_mask_fraction_by_family: dict[str, float] = field(
        default_factory=dict
    )
    residue_atom_max_residue_index: int = 4096
    residue_atom_artifact_top_k: int = 16
    ccc_reconstruction_target: float = 0.95
    inference_top_k: int | None = None
    candidate_cap: int | None = None
    teacher_kl_guardrail: float | None = None
    seed: int = 7
    device: str = "auto"
    chemical_shift_baseline_path: str | None = None
    chemical_shift_reference_corpus_path: str | None = None
    initial_checkpoint_path: str | None = None
    initial_checkpoint_strict: bool = True
    initial_checkpoint_load_optimizer: bool = True
    initial_checkpoint_exclude_prefixes: list[str] = field(default_factory=list)
    bioemu_x0_eval_initial_checkpoint_before_training: bool = False
    trainable_parameter_name_patterns: list[str] = field(default_factory=list)
    candidate_free_evidence_encoder_layers: int = 2
    candidate_free_evidence_encoder_heads: int = 4
    candidate_free_state_count: int = 16
    candidate_free_hn_cprime_loss_weight: float = 8.0
    candidate_free_worst_hn_cprime_loss_weight: float = 0.0
    candidate_free_family_loss_weight: float = 2.0
    candidate_free_reconstruction_loss_weight: float = 3.0
    candidate_free_student_t_nll_weight: float = 0.25
    candidate_free_state_diversity_weight: float = 0.05
    candidate_free_moment_state_consistency_weight: float = 0.05
    enable_residue_numbering_repair: bool = True
    residue_numbering_min_confidence: float = 0.35
    enable_candidate_free_physics_features: bool = True
    candidate_free_physics_context_gate_init: float = 0.10
    candidate_free_physics_target_families: list[str] = field(
        default_factory=lambda: ["HN", "C'"]
    )
    enable_candidate_free_structure_proxy_features: bool = False
    candidate_free_structure_proxy_sources: list[str] = field(
        default_factory=lambda: ["BioEmu", "AF"]
    )
    candidate_free_structure_proxy_max_models: int = 12
    candidate_free_structure_proxy_blend: float = 0.45
    candidate_free_alignment_loss_min_weight: float = 0.25
    candidate_free_physics_risk_loss_boost: float = 0.0
    candidate_free_spectral_tail_loss_boost: float = 0.0
    candidate_free_spectral_tail_fraction: float = 0.2
    candidate_free_spectral_tail_families: list[str] = field(default_factory=list)
    candidate_free_hn_gly_tail_loss_boost: float = 0.0
    candidate_free_hn_gly_tail_fraction: float = 0.25
    candidate_free_hn_context_tail_loss_boost: float = 0.0
    candidate_free_hn_context_tail_fraction: float = 0.25
    candidate_free_hn_context_tail_min_risk: float = 0.20
    hn95_context_tail_residual_loss_weight: float = 0.0
    hn95_context_tail_residual_min_risk: float = 0.20
    candidate_free_reliability_regularization_weight: float = 0.01
    candidate_free_forced_train_entity_uids: list[str] = field(default_factory=list)
    candidate_free_forced_train_entities_path: str | None = None
    candidate_free_exclude_forced_train_from_val: bool = True
    candidate_free_forced_train_val_include_entity_uids: list[str] = field(
        default_factory=list
    )
    candidate_free_forced_train_val_include_entities_path: str | None = None
    candidate_free_replay_entity_uids: list[str] = field(default_factory=list)
    candidate_free_replay_entities_path: str | None = None
    candidate_free_replay_entity_oversample_factor: int = 1
    candidate_free_val_include_entity_uids: list[str] = field(default_factory=list)
    candidate_free_val_exclude_entity_uids: list[str] = field(default_factory=list)
    candidate_free_entity_family_replay_loss_weight: float = 0.0
    candidate_free_entity_family_replay_weight_threshold: float = 1.0
    candidate_free_failed_active_face_support_center_loss_weight: float = 0.0
    candidate_free_failed_active_face_support_center_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    candidate_free_failed_active_face_support_center_softness: float = 0.05
    candidate_free_failed_active_face_support_center_gap_focus_scale: float = 0.0
    candidate_free_failed_active_face_support_center_target_source: str = (
        "target_value"
    )
    candidate_free_failed_active_face_support_center_dual_weight_power: float = 1.0
    candidate_free_failed_active_face_support_center_response_deficit_weight_scale: float = 0.0
    candidate_free_failed_active_face_support_center_min_row_weight: float = 1.0
    candidate_free_failed_active_face_support_center_max_row_weight: float = 10.0
    candidate_free_failed_active_face_support_center_warmup_epochs: int = 0
    candidate_free_failed_active_face_support_center_start_scale: float = 1.0
    candidate_free_failed_active_face_support_tail_loss_weight: float = 0.0
    candidate_free_failed_active_face_support_tail_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    candidate_free_failed_active_face_support_tail_softness: float = 0.05
    candidate_free_failed_active_face_support_tail_direction: str = "auto"
    candidate_free_failed_active_face_support_tail_target_source: str = "target_value"
    candidate_free_failed_active_face_support_tail_dual_weight_power: float = 1.0
    candidate_free_failed_active_face_support_tail_response_deficit_weight_scale: float = 0.0
    candidate_free_failed_active_face_support_tail_min_row_weight: float = 1.0
    candidate_free_failed_active_face_support_tail_max_row_weight: float = 10.0
    candidate_free_failed_active_face_support_tail_warmup_epochs: int = 0
    candidate_free_failed_active_face_support_tail_start_scale: float = 1.0
    candidate_free_failed_active_face_energy_assignment_loss_weight: float = 0.0
    candidate_free_failed_active_face_energy_assignment_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    candidate_free_failed_active_face_energy_assignment_target_source: str = (
        "target_value"
    )
    candidate_free_failed_active_face_energy_assignment_margin: float = 0.04
    candidate_free_failed_active_face_energy_assignment_positive_fraction: float = 0.08
    candidate_free_failed_active_face_energy_assignment_negative_fraction: float = 0.12
    candidate_free_failed_active_face_energy_assignment_positive_mass_weight: float = 0.0
    candidate_free_failed_active_face_energy_assignment_temperature: float = 1.0
    candidate_free_failed_active_face_energy_assignment_dual_weight_power: float = 1.0
    candidate_free_failed_active_face_energy_assignment_response_deficit_weight_scale: float = 0.0
    candidate_free_failed_active_face_energy_assignment_min_row_weight: float = 1.0
    candidate_free_failed_active_face_energy_assignment_max_row_weight: float = 10.0
    candidate_free_failed_active_face_energy_assignment_warmup_epochs: int = 0
    candidate_free_failed_active_face_energy_assignment_start_scale: float = 1.0
    candidate_free_failed_active_face_shared_q_response_loss_weight: float = 0.0
    candidate_free_failed_active_face_shared_q_response_family_weights: dict[
        str, float
    ] = field(default_factory=dict)
    candidate_free_failed_active_face_shared_q_response_target_source: str = (
        "target_value"
    )
    candidate_free_failed_active_face_shared_q_response_temperature: float = 1.0
    candidate_free_failed_active_face_shared_q_response_huber_beta: float = 0.15
    candidate_free_failed_active_face_shared_q_response_support_gap_weight: float = 0.0
    candidate_free_failed_active_face_shared_q_response_dual_weight_power: float = 1.0
    candidate_free_failed_active_face_shared_q_response_deficit_weight_scale: float = 0.0
    candidate_free_failed_active_face_shared_q_response_min_row_weight: float = 1.0
    candidate_free_failed_active_face_shared_q_response_max_row_weight: float = 10.0
    candidate_free_failed_active_face_shared_q_response_warmup_epochs: int = 0
    candidate_free_failed_active_face_shared_q_response_start_scale: float = 1.0
    candidate_free_entity_loss_weights: dict[str, float] = field(default_factory=dict)
    candidate_free_entity_family_loss_weights: dict[str, float] = field(
        default_factory=dict
    )
    candidate_free_replay_ucbshift_residue_offset_contract: dict[str, Any] = field(
        default_factory=dict
    )
    candidate_free_replay_require_ucbshift_residue_offset_contract: bool = False
    legacy_candidate_diagnostic: bool = False
    enable_state_mixture: bool = False
    state_mixture_count: int = 8
    mirror_descent_steps: int = 0
    mirror_descent_step_size: float = 0.0
    robust_likelihood_mode: str = "gaussian"
    student_t_degrees_of_freedom: float = 4.0
    benchmark_render_config: dict[str, Any] = field(default_factory=dict)
    render_best_preview: bool = False
    best_preview_max_train_examples: int = 32
    best_preview_max_val_examples: int = 50
    history_window_epochs: int | None = None
    last_checkpoint_interval: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to one JSON file."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudentTrainingConfig":
        """Create one configuration object from a dictionary."""
        return cls(
            train_splits=list(payload.get("train_splits", ["train"])),
            val_splits=list(payload.get("val_splits", ["val"])),
            max_train_examples=(
                None
                if payload.get("max_train_examples") is None
                else int(payload["max_train_examples"])
            ),
            max_val_examples=(
                None
                if payload.get("max_val_examples") is None
                else int(payload["max_val_examples"])
            ),
            checkpoint_metric=str(payload.get("checkpoint_metric", "teacher_kl_macro")),
            checkpoint_tie_breakers=list(
                payload.get(
                    "checkpoint_tie_breakers",
                    [
                        "teacher_js_macro",
                        "cs_rmse_z_macro",
                        "noe_violation_rate_macro",
                    ],
                )
            ),
            report_micro_metrics=bool(payload.get("report_micro_metrics", True)),
            epochs=int(payload.get("epochs", 25)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            min_learning_rate=float(payload.get("min_learning_rate", 0.0)),
            learning_rate_schedule=str(
                payload.get("learning_rate_schedule", "constant")
            ),
            warmup_epochs=int(payload.get("warmup_epochs", 0)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            gradient_clip_norm=float(payload.get("gradient_clip_norm", 1.0)),
            weight_kl_weight=float(payload.get("weight_kl_weight", 1.0)),
            chemical_shift_loss_weight=float(
                payload.get("chemical_shift_loss_weight", 0.5)
            ),
            chemical_shift_loss_mode=str(
                payload.get("chemical_shift_loss_mode", "mse")
            ),
            chemical_shift_huber_delta=float(
                payload.get("chemical_shift_huber_delta", 5.0)
            ),
            chemical_shift_ccc_loss_weight=float(
                payload.get("chemical_shift_ccc_loss_weight", 0.0)
            ),
            chemical_shift_family_balance=bool(
                payload.get("chemical_shift_family_balance", False)
            ),
            chemical_shift_family_weights=dict(
                payload.get("chemical_shift_family_weights", {})
            ),
            training_task=payload.get("training_task"),
            enable_ccc_support_oracle=bool(
                payload.get("enable_ccc_support_oracle", False)
            ),
            ccc_oracle_solver=str(
                payload.get("ccc_oracle_solver", "projected_simplex")
            ),
            ccc_oracle_steps=int(payload.get("ccc_oracle_steps", 80)),
            ccc_oracle_distillation_weight=float(
                payload.get("ccc_oracle_distillation_weight", 0.0)
            ),
            ccc_geometry_loss_weight=float(
                payload.get("ccc_geometry_loss_weight", 0.0)
            ),
            posterior_evidence_kl_weight=float(
                payload.get("posterior_evidence_kl_weight", 0.0)
            ),
            evidence_temperature=float(payload.get("evidence_temperature", 1.0)),
            ccc_oracle_kl_weight=float(payload.get("ccc_oracle_kl_weight", 0.0)),
            ccc_oracle_temperature=float(payload.get("ccc_oracle_temperature", 0.5)),
            enable_observable_adapters=bool(
                payload.get("enable_observable_adapters", False)
            ),
            observable_oracle_distillation_weight=float(
                payload.get("observable_oracle_distillation_weight", 0.0)
            ),
            observable_oracle_temperature=float(
                payload.get("observable_oracle_temperature", 0.5)
            ),
            observable_channel_weights=dict(
                payload.get(
                    "observable_channel_weights",
                    {
                        "chemical_shifts": 1.0,
                        "j_couplings": 0.10,
                        "noe_restraints": 0.10,
                        "saxs": 0.0,
                    },
                )
            ),
            enable_moment_head=bool(payload.get("enable_moment_head", False)),
            enable_posterior_state_tokens=bool(
                payload.get("enable_posterior_state_tokens", False)
            ),
            posterior_state_count=int(payload.get("posterior_state_count", 16)),
            enable_latent_posterior_flow=bool(
                payload.get("enable_latent_posterior_flow", False)
            ),
            latent_flow_steps=int(payload.get("latent_flow_steps", 4)),
            moment_oracle_distillation_weight=float(
                payload.get("moment_oracle_distillation_weight", 0.0)
            ),
            moment_sample_consistency_weight=float(
                payload.get("moment_sample_consistency_weight", 0.0)
            ),
            state_occupancy_distillation_weight=float(
                payload.get("state_occupancy_distillation_weight", 0.0)
            ),
            state_diversity_weight=float(payload.get("state_diversity_weight", 0.0)),
            enable_family_specific_moment_head=bool(
                payload.get("enable_family_specific_moment_head", False)
            ),
            enable_nmr_structural_features=bool(
                payload.get("enable_nmr_structural_features", False)
            ),
            enable_local_evidence_context=bool(
                payload.get("enable_local_evidence_context", False)
            ),
            enable_same_residue_evidence_context=bool(
                payload.get("enable_same_residue_evidence_context", False)
            ),
            same_residue_evidence_target_families=list(
                payload.get("same_residue_evidence_target_families", [])
            ),
            enable_target_set_evidence_encoder=bool(
                payload.get("enable_target_set_evidence_encoder", False)
            ),
            enable_target_evidence_token=bool(
                payload.get("enable_target_evidence_token", False)
            ),
            target_set_encoder_layers=int(payload.get("target_set_encoder_layers", 1)),
            target_set_encoder_heads=int(payload.get("target_set_encoder_heads", 4)),
            target_set_encoder_max_targets=int(
                payload.get("target_set_encoder_max_targets", 1024)
            ),
            target_set_context_gate_init=float(
                payload.get("target_set_context_gate_init", 1.0)
            ),
            target_set_encoder_masked_only=bool(
                payload.get("target_set_encoder_masked_only", False)
            ),
            target_set_encoder_target_families=list(
                payload.get("target_set_encoder_target_families", [])
            ),
            target_set_encoder_evidence_families=list(
                payload.get("target_set_encoder_evidence_families", [])
            ),
            enable_residue_grid_evidence_encoder=bool(
                payload.get("enable_residue_grid_evidence_encoder", False)
            ),
            residue_grid_encoder_layers=int(
                payload.get("residue_grid_encoder_layers", 1)
            ),
            residue_grid_encoder_heads=int(
                payload.get("residue_grid_encoder_heads", 4)
            ),
            residue_grid_context_gate_init=float(
                payload.get("residue_grid_context_gate_init", 0.2)
            ),
            enable_residue_grid_family_gates=bool(
                payload.get("enable_residue_grid_family_gates", False)
            ),
            residue_grid_encoder_masked_only=bool(
                payload.get("residue_grid_encoder_masked_only", True)
            ),
            residue_grid_encoder_target_families=list(
                payload.get("residue_grid_encoder_target_families", [])
            ),
            local_evidence_window=int(payload.get("local_evidence_window", 3)),
            enable_residue_anchor_evidence_context=bool(
                payload.get("enable_residue_anchor_evidence_context", False)
            ),
            residue_anchor_evidence_offsets=[
                int(offset)
                for offset in payload.get(
                    "residue_anchor_evidence_offsets",
                    [-1, 0, 1],
                )
            ],
            residue_anchor_evidence_target_families=list(
                payload.get("residue_anchor_evidence_target_families", [])
            ),
            enable_backbone_evidence_context=bool(
                payload.get("enable_backbone_evidence_context", False)
            ),
            backbone_evidence_target_families=list(
                payload.get("backbone_evidence_target_families", [])
            ),
            enable_all_family_evidence_affine_calibration=bool(
                payload.get("enable_all_family_evidence_affine_calibration", False)
            ),
            evidence_affine_calibration_families=list(
                payload.get("evidence_affine_calibration_families", [])
            ),
            evidence_affine_scale_only_families=list(
                payload.get("evidence_affine_scale_only_families", [])
            ),
            enable_hn_variance_calibration=bool(
                payload.get("enable_hn_variance_calibration", False)
            ),
            enable_cprime_robust_likelihood=bool(
                payload.get("enable_cprime_robust_likelihood", False)
            ),
            state_entropy_floor=float(payload.get("state_entropy_floor", 0.0)),
            state_repulsion_weight=float(payload.get("state_repulsion_weight", 0.0)),
            basin_coverage_weight=float(payload.get("basin_coverage_weight", 0.0)),
            nmr_energy_guidance_weight=float(
                payload.get("nmr_energy_guidance_weight", 1.0)
            ),
            hn_variance_loss_weight=float(payload.get("hn_variance_loss_weight", 0.0)),
            cprime_outlier_loss_weight=float(
                payload.get("cprime_outlier_loss_weight", 0.0)
            ),
            tail_calibration_loss_weight=float(
                payload.get("tail_calibration_loss_weight", 0.0)
            ),
            tail_calibration_families=list(
                payload.get("tail_calibration_families", [])
            ),
            tail_calibration_fraction=float(
                payload.get("tail_calibration_fraction", 0.2)
            ),
            tail_direction_loss_weight=float(
                payload.get("tail_direction_loss_weight", 0.0)
            ),
            tail_direction_families=list(payload.get("tail_direction_families", [])),
            tail_direction_fraction=float(
                payload.get("tail_direction_fraction", 0.2)
            ),
            tail_gap_loss_weight=float(payload.get("tail_gap_loss_weight", 0.0)),
            tail_gap_families=list(payload.get("tail_gap_families", [])),
            tail_gap_fraction=float(payload.get("tail_gap_fraction", 0.2)),
            bioemu_all_family_outlier_loss_weight=float(
                payload.get("bioemu_all_family_outlier_loss_weight", 0.0)
            ),
            bioemu_all_family_outlier_family_names=list(
                payload.get(
                    "bioemu_all_family_outlier_family_names",
                    ["HN", "N", "CA", "CB", "C'"],
                )
            ),
            bioemu_all_family_outlier_fraction=float(
                payload.get("bioemu_all_family_outlier_fraction", 0.05)
            ),
            bioemu_all_family_outlier_abs_error_threshold=float(
                payload.get("bioemu_all_family_outlier_abs_error_threshold", 0.0)
            ),
            bioemu_all_family_outlier_abs_error_threshold_by_family=dict(
                payload.get(
                    "bioemu_all_family_outlier_abs_error_threshold_by_family",
                    {},
                )
            ),
            bioemu_all_family_outlier_min_rows=int(
                payload.get("bioemu_all_family_outlier_min_rows", 8)
            ),
            bioemu_all_family_outlier_max_rows_per_family=int(
                payload.get("bioemu_all_family_outlier_max_rows_per_family", 64)
            ),
            bioemu_all_family_outlier_low_ess_gate_ess_floor=float(
                payload.get("bioemu_all_family_outlier_low_ess_gate_ess_floor", 0.0)
            ),
            bioemu_all_family_outlier_low_ess_gate_min_scale=float(
                payload.get("bioemu_all_family_outlier_low_ess_gate_min_scale", 0.25)
            ),
            bioemu_all_family_outlier_residue_family_keys=list(
                payload.get("bioemu_all_family_outlier_residue_family_keys", [])
            ),
            bioemu_hn_signed_support_replay_loss_weight=float(
                payload.get("bioemu_hn_signed_support_replay_loss_weight", 0.0)
            ),
            bioemu_hn_signed_support_replay_state_keys=list(
                payload.get("bioemu_hn_signed_support_replay_state_keys", [])
            ),
            bioemu_hn_signed_support_replay_fraction=float(
                payload.get("bioemu_hn_signed_support_replay_fraction", 0.25)
            ),
            bioemu_hn_signed_support_replay_min_rows=int(
                payload.get("bioemu_hn_signed_support_replay_min_rows", 1)
            ),
            bioemu_hn_signed_support_replay_max_rows_per_state=int(
                payload.get("bioemu_hn_signed_support_replay_max_rows_per_state", 24)
            ),
            bioemu_hn_signed_support_replay_huber_beta=float(
                payload.get("bioemu_hn_signed_support_replay_huber_beta", 0.75)
            ),
            family_pairwise_rank_loss_weight=float(
                payload.get("family_pairwise_rank_loss_weight", 0.0)
            ),
            family_pairwise_rank_families=list(
                payload.get("family_pairwise_rank_families", [])
            ),
            family_pairwise_rank_temperature=float(
                payload.get("family_pairwise_rank_temperature", 0.5)
            ),
            family_pairwise_rank_min_delta=float(
                payload.get("family_pairwise_rank_min_delta", 0.15)
            ),
            family_pairwise_rank_max_rows=int(
                payload.get("family_pairwise_rank_max_rows", 96)
            ),
            residual_trend_loss_weight=float(
                payload.get(
                    "residual_trend_loss_weight",
                    payload.get("cprime_residual_trend_loss_weight", 0.0),
                )
            ),
            residual_trend_families=list(
                payload.get("residual_trend_families", [])
            ),
            enable_entry_family_calibration=bool(
                payload.get("enable_entry_family_calibration", False)
            ),
            enable_entry_family_quantile_calibration=bool(
                payload.get("enable_entry_family_quantile_calibration", False)
            ),
            entry_family_quantile_calibration_families=list(
                payload.get("entry_family_quantile_calibration_families", [])
            ),
            entry_family_quantile_fraction=float(
                payload.get("entry_family_quantile_fraction", 0.2)
            ),
            entry_family_quantile_scale_weight=float(
                payload.get("entry_family_quantile_scale_weight", 0.0)
            ),
            entry_family_calibration_scale_max_by_family=dict(
                payload.get("entry_family_calibration_scale_max_by_family", {})
            ),
            enable_entry_family_regression_calibration=bool(
                payload.get("enable_entry_family_regression_calibration", False)
            ),
            entry_family_regression_calibration_families=list(
                payload.get("entry_family_regression_calibration_families", [])
            ),
            entry_family_regression_strength=float(
                payload.get("entry_family_regression_strength", 0.0)
            ),
            entry_family_regression_ridge=float(
                payload.get("entry_family_regression_ridge", 0.05)
            ),
            entry_family_regression_use_raw_predictions=bool(
                payload.get("entry_family_regression_use_raw_predictions", False)
            ),
            entry_family_regression_slope_min_by_family=dict(
                payload.get("entry_family_regression_slope_min_by_family", {})
            ),
            entry_family_regression_slope_max_by_family=dict(
                payload.get("entry_family_regression_slope_max_by_family", {})
            ),
            enable_entry_family_isotonic_calibration=bool(
                payload.get("enable_entry_family_isotonic_calibration", False)
            ),
            entry_family_isotonic_calibration_families=list(
                payload.get("entry_family_isotonic_calibration_families", [])
            ),
            entry_family_isotonic_strength=float(
                payload.get("entry_family_isotonic_strength", 0.0)
            ),
            entry_family_isotonic_min_rows=int(
                payload.get("entry_family_isotonic_min_rows", 8)
            ),
            entry_family_isotonic_max_abs=float(
                payload.get("entry_family_isotonic_max_abs", 1.0)
            ),
            entry_family_isotonic_max_abs_by_family=dict(
                payload.get("entry_family_isotonic_max_abs_by_family", {})
            ),
            enable_entry_family_feature_residual_calibration=bool(
                payload.get(
                    "enable_entry_family_feature_residual_calibration",
                    False,
                )
            ),
            entry_family_feature_residual_calibration_families=list(
                payload.get(
                    "entry_family_feature_residual_calibration_families",
                    [],
                )
            ),
            entry_family_feature_residual_weight=float(
                payload.get("entry_family_feature_residual_weight", 0.0)
            ),
            entry_family_feature_residual_ridge=float(
                payload.get("entry_family_feature_residual_ridge", 0.1)
            ),
            entry_family_feature_residual_min_rows=int(
                payload.get("entry_family_feature_residual_min_rows", 8)
            ),
            entry_family_feature_residual_max_abs=float(
                payload.get("entry_family_feature_residual_max_abs", 0.6)
            ),
            enable_measure_aware_row_weights=bool(
                payload.get("enable_measure_aware_row_weights", False)
            ),
            enable_measure_aware_context_masking=bool(
                payload.get("enable_measure_aware_context_masking", False)
            ),
            measure_aware_context_target_families=list(
                payload.get("measure_aware_context_target_families", [])
            ),
            measure_aware_context_neighbor_offsets=list(
                payload.get("measure_aware_context_neighbor_offsets", [-1, 0, 1])
            ),
            enable_ccc_decomposition_loss=bool(
                payload.get("enable_ccc_decomposition_loss", False)
            ),
            enable_joint_nmr_posterior=bool(
                payload.get("enable_joint_nmr_posterior", False)
            ),
            family_corr_loss_weight=float(
                payload.get("family_corr_loss_weight", 0.0)
            ),
            family_scale_loss_weight=float(
                payload.get("family_scale_loss_weight", 0.0)
            ),
            family_bias_loss_weight=float(
                payload.get("family_bias_loss_weight", 0.0)
            ),
            student_t_nll_weight=float(payload.get("student_t_nll_weight", 0.0)),
            crps_calibration_loss_weight=float(
                payload.get("crps_calibration_loss_weight", 0.0)
            ),
            joint_nmr_loss_weight=float(payload.get("joint_nmr_loss_weight", 0.0)),
            hn_context_loss_weight=float(payload.get("hn_context_loss_weight", 0.0)),
            paired_evidence_imputation_loss_weight=float(
                payload.get("paired_evidence_imputation_loss_weight", 0.0)
            ),
            paired_evidence_imputation_fraction=float(
                payload.get("paired_evidence_imputation_fraction", 0.15)
            ),
            paired_evidence_imputation_targets=list(
                payload.get("paired_evidence_imputation_targets", [])
            ),
            paired_evidence_imputation_entity_weights=dict(
                payload.get("paired_evidence_imputation_entity_weights", {})
            ),
            paired_evidence_imputation_risk_boost=float(
                payload.get("paired_evidence_imputation_risk_boost", 0.0)
            ),
            paired_evidence_imputation_tail_boost=float(
                payload.get("paired_evidence_imputation_tail_boost", 0.0)
            ),
            paired_evidence_imputation_max_fraction=float(
                payload.get("paired_evidence_imputation_max_fraction", 0.8)
            ),
            enable_pair_evidence_mean_blend=bool(
                payload.get("enable_pair_evidence_mean_blend", False)
            ),
            pair_evidence_mean_blend_weight=float(
                payload.get("pair_evidence_mean_blend_weight", 0.0)
            ),
            enable_same_family_evidence_interpolation=bool(
                payload.get("enable_same_family_evidence_interpolation", False)
            ),
            same_family_evidence_interpolation_weight=float(
                payload.get("same_family_evidence_interpolation_weight", 0.0)
            ),
            same_family_evidence_interpolation_sigma=float(
                payload.get("same_family_evidence_interpolation_sigma", 6.0)
            ),
            same_family_evidence_interpolation_families=list(
                payload.get("same_family_evidence_interpolation_families", [])
            ),
            enable_same_family_residual_interpolation=bool(
                payload.get("enable_same_family_residual_interpolation", False)
            ),
            same_family_residual_interpolation_weight=float(
                payload.get("same_family_residual_interpolation_weight", 0.0)
            ),
            same_family_residual_interpolation_sigma=float(
                payload.get("same_family_residual_interpolation_sigma", 6.0)
            ),
            same_family_residual_interpolation_families=list(
                payload.get("same_family_residual_interpolation_families", [])
            ),
            same_family_residual_interpolation_family_weights=dict(
                payload.get("same_family_residual_interpolation_family_weights", {})
            ),
            enable_same_family_residual_knn=bool(
                payload.get("enable_same_family_residual_knn", False)
            ),
            same_family_residual_knn_weight=float(
                payload.get("same_family_residual_knn_weight", 0.0)
            ),
            same_family_residual_knn_residue_sigma=float(
                payload.get("same_family_residual_knn_residue_sigma", 12.0)
            ),
            same_family_residual_knn_shift_sigma=float(
                payload.get("same_family_residual_knn_shift_sigma", 0.6)
            ),
            same_family_residual_knn_families=list(
                payload.get("same_family_residual_knn_families", [])
            ),
            same_family_residual_knn_family_weights=dict(
                payload.get("same_family_residual_knn_family_weights", {})
            ),
            enable_same_family_residual_gp=bool(
                payload.get("enable_same_family_residual_gp", False)
            ),
            same_family_residual_gp_weight=float(
                payload.get("same_family_residual_gp_weight", 0.0)
            ),
            same_family_residual_gp_residue_sigma=float(
                payload.get("same_family_residual_gp_residue_sigma", 8.0)
            ),
            same_family_residual_gp_shift_sigma=float(
                payload.get("same_family_residual_gp_shift_sigma", 100.0)
            ),
            same_family_residual_gp_ridge=float(
                payload.get("same_family_residual_gp_ridge", 0.05)
            ),
            same_family_residual_gp_max_observed=int(
                payload.get("same_family_residual_gp_max_observed", 128)
            ),
            same_family_residual_gp_max_abs=float(
                payload.get("same_family_residual_gp_max_abs", 1.0)
            ),
            same_family_residual_gp_families=list(
                payload.get("same_family_residual_gp_families", [])
            ),
            same_family_residual_gp_family_weights=dict(
                payload.get("same_family_residual_gp_family_weights", {})
            ),
            same_family_residual_gp_max_abs_by_family=dict(
                payload.get("same_family_residual_gp_max_abs_by_family", {})
            ),
            same_family_residual_gp_min_sign_consensus=float(
                payload.get("same_family_residual_gp_min_sign_consensus", 0.0)
            ),
            same_family_residual_gp_min_kernel_support=float(
                payload.get("same_family_residual_gp_min_kernel_support", 0.0)
            ),
            same_family_residual_gp_terminal_suppression_distance=float(
                payload.get("same_family_residual_gp_terminal_suppression_distance", 0.0)
            ),
            same_family_residual_gp_risk_gate_min=float(
                payload.get("same_family_residual_gp_risk_gate_min", 0.0)
            ),
            enable_hn_risk_scalar_calibration=bool(
                payload.get("enable_hn_risk_scalar_calibration", False)
            ),
            hn_risk_scalar_calibration_weight=float(
                payload.get("hn_risk_scalar_calibration_weight", 0.0)
            ),
            hn_risk_scalar_calibration_min_rows=int(
                payload.get("hn_risk_scalar_calibration_min_rows", 4)
            ),
            hn_risk_scalar_calibration_min_risk=float(
                payload.get("hn_risk_scalar_calibration_min_risk", 0.25)
            ),
            hn_risk_scalar_calibration_risk_sigma=float(
                payload.get("hn_risk_scalar_calibration_risk_sigma", 0.25)
            ),
            hn_risk_scalar_calibration_shift_sigma=float(
                payload.get("hn_risk_scalar_calibration_shift_sigma", 0.25)
            ),
            hn_risk_scalar_calibration_max_abs=float(
                payload.get("hn_risk_scalar_calibration_max_abs", 0.20)
            ),
            hn_risk_scalar_calibration_min_sign_consensus=float(
                payload.get("hn_risk_scalar_calibration_min_sign_consensus", 0.0)
            ),
            hn_risk_scalar_calibration_terminal_suppression_distance=float(
                payload.get(
                    "hn_risk_scalar_calibration_terminal_suppression_distance",
                    0.0,
                )
            ),
            enable_hn_risk_vector_calibration=bool(
                payload.get("enable_hn_risk_vector_calibration", False)
            ),
            hn_risk_vector_calibration_feature_sigma=float(
                payload.get("hn_risk_vector_calibration_feature_sigma", 0.35)
            ),
            enable_hn_entry_preservation_gate=bool(
                payload.get("enable_hn_entry_preservation_gate", False)
            ),
            hn_entry_preservation_gate_strength=float(
                payload.get("hn_entry_preservation_gate_strength", 0.0)
            ),
            hn_entry_preservation_mae_threshold=float(
                payload.get("hn_entry_preservation_mae_threshold", 0.055)
            ),
            hn_entry_preservation_ccc_threshold=float(
                payload.get("hn_entry_preservation_ccc_threshold", 0.9)
            ),
            hn_entry_preservation_min_rows=int(
                payload.get("hn_entry_preservation_min_rows", 4)
            ),
            enable_hn_observed_correction_selector=bool(
                payload.get("enable_hn_observed_correction_selector", False)
            ),
            hn_observed_correction_selector_strength=float(
                payload.get("hn_observed_correction_selector_strength", 0.0)
            ),
            hn_observed_correction_selector_min_rows=int(
                payload.get("hn_observed_correction_selector_min_rows", 5)
            ),
            hn_observed_correction_selector_mae_margin=float(
                payload.get("hn_observed_correction_selector_mae_margin", 0.0)
            ),
            hn_observed_correction_selector_ccc_margin=float(
                payload.get("hn_observed_correction_selector_ccc_margin", 0.0)
            ),
            enable_hn_raw_prediction_selector=bool(
                payload.get("enable_hn_raw_prediction_selector", False)
            ),
            hn_raw_prediction_selector_strength=float(
                payload.get("hn_raw_prediction_selector_strength", 0.0)
            ),
            hn_raw_prediction_selector_min_rows=int(
                payload.get("hn_raw_prediction_selector_min_rows", 5)
            ),
            hn_raw_prediction_selector_mae_margin=float(
                payload.get("hn_raw_prediction_selector_mae_margin", 0.0)
            ),
            hn_raw_prediction_selector_ccc_margin=float(
                payload.get("hn_raw_prediction_selector_ccc_margin", 0.0)
            ),
            enable_hn_pseudomask_raw_prediction_selector=bool(
                payload.get("enable_hn_pseudomask_raw_prediction_selector", False)
            ),
            hn_pseudomask_raw_prediction_selector_strength=float(
                payload.get("hn_pseudomask_raw_prediction_selector_strength", 0.0)
            ),
            hn_pseudomask_raw_prediction_selector_min_rows=int(
                payload.get("hn_pseudomask_raw_prediction_selector_min_rows", 10)
            ),
            hn_pseudomask_raw_prediction_selector_query_fraction=float(
                payload.get(
                    "hn_pseudomask_raw_prediction_selector_query_fraction",
                    0.25,
                )
            ),
            hn_pseudomask_raw_prediction_selector_min_query_rows=int(
                payload.get("hn_pseudomask_raw_prediction_selector_min_query_rows", 3)
            ),
            hn_pseudomask_raw_prediction_selector_mae_margin=float(
                payload.get("hn_pseudomask_raw_prediction_selector_mae_margin", 0.0)
            ),
            hn_pseudomask_raw_prediction_selector_ccc_margin=float(
                payload.get("hn_pseudomask_raw_prediction_selector_ccc_margin", 0.0)
            ),
            enable_hn_feature_spread_preservation_selector=bool(
                payload.get("enable_hn_feature_spread_preservation_selector", False)
            ),
            hn_feature_spread_preservation_selector_strength=float(
                payload.get("hn_feature_spread_preservation_selector_strength", 0.0)
            ),
            hn_feature_spread_preservation_selector_min_rows=int(
                payload.get("hn_feature_spread_preservation_selector_min_rows", 12)
            ),
            hn_feature_spread_preservation_selector_std_ratio=float(
                payload.get("hn_feature_spread_preservation_selector_std_ratio", 0.80)
            ),
            hn_feature_spread_preservation_selector_delta_margin=float(
                payload.get(
                    "hn_feature_spread_preservation_selector_delta_margin",
                    0.005,
                )
            ),
            enable_hn_uncertainty_tail_anchor_selector=bool(
                payload.get("enable_hn_uncertainty_tail_anchor_selector", False)
            ),
            hn_uncertainty_tail_anchor_selector_strength=float(
                payload.get("hn_uncertainty_tail_anchor_selector_strength", 0.0)
            ),
            hn_uncertainty_tail_anchor_selector_min_rows=int(
                payload.get("hn_uncertainty_tail_anchor_selector_min_rows", 12)
            ),
            hn_uncertainty_tail_anchor_selector_min_masked_rows=int(
                payload.get("hn_uncertainty_tail_anchor_selector_min_masked_rows", 8)
            ),
            hn_uncertainty_tail_anchor_selector_observed_ccc_min=float(
                payload.get(
                    "hn_uncertainty_tail_anchor_selector_observed_ccc_min",
                    0.88,
                )
            ),
            hn_uncertainty_tail_anchor_selector_observed_bias_max=float(
                payload.get(
                    "hn_uncertainty_tail_anchor_selector_observed_bias_max",
                    0.03,
                )
            ),
            hn_uncertainty_tail_anchor_selector_observed_std_ratio_min=float(
                payload.get(
                    "hn_uncertainty_tail_anchor_selector_observed_std_ratio_min",
                    0.90,
                )
            ),
            hn_uncertainty_tail_anchor_selector_observed_std_ratio_max=float(
                payload.get(
                    "hn_uncertainty_tail_anchor_selector_observed_std_ratio_max",
                    1.18,
                )
            ),
            hn_uncertainty_tail_anchor_selector_mean_gap_min=float(
                payload.get("hn_uncertainty_tail_anchor_selector_mean_gap_min", 0.025)
            ),
            hn_uncertainty_tail_anchor_selector_sigma_min=float(
                payload.get("hn_uncertainty_tail_anchor_selector_sigma_min", 0.07)
            ),
            hn_uncertainty_tail_anchor_selector_low_gap=float(
                payload.get("hn_uncertainty_tail_anchor_selector_low_gap", 0.15)
            ),
            hn_uncertainty_tail_anchor_selector_graph_delta_min=float(
                payload.get("hn_uncertainty_tail_anchor_selector_graph_delta_min", 0.08)
            ),
            hn_uncertainty_tail_anchor_selector_entry_shift_max=float(
                payload.get("hn_uncertainty_tail_anchor_selector_entry_shift_max", 0.035)
            ),
            hn_uncertainty_tail_anchor_selector_row_shift_max=float(
                payload.get("hn_uncertainty_tail_anchor_selector_row_shift_max", 0.18)
            ),
            hn_uncertainty_tail_anchor_selector_anchor_margin=float(
                payload.get("hn_uncertainty_tail_anchor_selector_anchor_margin", 0.08)
            ),
            hn_uncertainty_tail_anchor_selector_min_structure_ring=float(
                payload.get(
                    "hn_uncertainty_tail_anchor_selector_min_structure_ring",
                    0.04,
                )
            ),
            hn_uncertainty_tail_anchor_selector_min_anchor_rows=int(
                payload.get("hn_uncertainty_tail_anchor_selector_min_anchor_rows", 2)
            ),
            enable_hn_alignment_mean_anchor_selector=bool(
                payload.get("enable_hn_alignment_mean_anchor_selector", False)
            ),
            hn_alignment_mean_anchor_selector_strength=float(
                payload.get("hn_alignment_mean_anchor_selector_strength", 0.0)
            ),
            hn_alignment_mean_anchor_selector_min_rows=int(
                payload.get("hn_alignment_mean_anchor_selector_min_rows", 12)
            ),
            hn_alignment_mean_anchor_selector_min_masked_rows=int(
                payload.get("hn_alignment_mean_anchor_selector_min_masked_rows", 8)
            ),
            hn_alignment_mean_anchor_selector_observed_ccc_min=float(
                payload.get("hn_alignment_mean_anchor_selector_observed_ccc_min", 0.90)
            ),
            hn_alignment_mean_anchor_selector_observed_bias_max=float(
                payload.get("hn_alignment_mean_anchor_selector_observed_bias_max", 0.02)
            ),
            hn_alignment_mean_anchor_selector_observed_std_ratio_min=float(
                payload.get(
                    "hn_alignment_mean_anchor_selector_observed_std_ratio_min",
                    0.90,
                )
            ),
            hn_alignment_mean_anchor_selector_observed_std_ratio_max=float(
                payload.get(
                    "hn_alignment_mean_anchor_selector_observed_std_ratio_max",
                    1.15,
                )
            ),
            hn_alignment_mean_anchor_selector_min_alignment_risk=float(
                payload.get("hn_alignment_mean_anchor_selector_min_alignment_risk", 0.50)
            ),
            hn_alignment_mean_anchor_selector_max_structure_ring=float(
                payload.get("hn_alignment_mean_anchor_selector_max_structure_ring", 0.02)
            ),
            hn_alignment_mean_anchor_selector_mean_gap_min=float(
                payload.get("hn_alignment_mean_anchor_selector_mean_gap_min", 0.004)
            ),
            hn_alignment_mean_anchor_selector_mean_gap_max=float(
                payload.get("hn_alignment_mean_anchor_selector_mean_gap_max", 0.04)
            ),
            hn_alignment_mean_anchor_selector_min_shift=float(
                payload.get("hn_alignment_mean_anchor_selector_min_shift", 0.02)
            ),
            hn_alignment_mean_anchor_selector_max_shift=float(
                payload.get("hn_alignment_mean_anchor_selector_max_shift", 0.02)
            ),
            hn_alignment_mean_anchor_selector_max_abs_graph_delta=float(
                payload.get(
                    "hn_alignment_mean_anchor_selector_max_abs_graph_delta",
                    0.025,
                )
            ),
            hn_alignment_mean_anchor_selector_allow_negative_shift=bool(
                payload.get(
                    "hn_alignment_mean_anchor_selector_allow_negative_shift",
                    False,
                )
            ),
            enable_hn_observed_variance_floor_selector=bool(
                payload.get("enable_hn_observed_variance_floor_selector", False)
            ),
            hn_observed_variance_floor_selector_strength=float(
                payload.get("hn_observed_variance_floor_selector_strength", 0.0)
            ),
            hn_observed_variance_floor_selector_min_rows=int(
                payload.get("hn_observed_variance_floor_selector_min_rows", 5)
            ),
            hn_observed_variance_floor_selector_std_ratio=float(
                payload.get("hn_observed_variance_floor_selector_std_ratio", 0.85)
            ),
            hn_observed_variance_floor_selector_softness=float(
                payload.get("hn_observed_variance_floor_selector_softness", 0.05)
            ),
            hn_observed_variance_floor_selector_raw_std_margin=float(
                payload.get("hn_observed_variance_floor_selector_raw_std_margin", 0.02)
            ),
            hn_observed_variance_floor_selector_min_risk=float(
                payload.get("hn_observed_variance_floor_selector_min_risk", 0.0)
            ),
            enable_hn_local_variance_shrink_selector=bool(
                payload.get("enable_hn_local_variance_shrink_selector", False)
            ),
            hn_local_variance_shrink_selector_strength=float(
                payload.get("hn_local_variance_shrink_selector_strength", 0.0)
            ),
            hn_local_variance_shrink_selector_min_rows=int(
                payload.get("hn_local_variance_shrink_selector_min_rows", 10)
            ),
            hn_local_variance_shrink_selector_query_fraction=float(
                payload.get("hn_local_variance_shrink_selector_query_fraction", 0.25)
            ),
            hn_local_variance_shrink_selector_min_query_rows=int(
                payload.get("hn_local_variance_shrink_selector_min_query_rows", 3)
            ),
            hn_local_variance_shrink_selector_residue_sigma=float(
                payload.get("hn_local_variance_shrink_selector_residue_sigma", 8.0)
            ),
            hn_local_variance_shrink_selector_feature_sigma=float(
                payload.get("hn_local_variance_shrink_selector_feature_sigma", 0.35)
            ),
            hn_local_variance_shrink_selector_std_ratio=float(
                payload.get("hn_local_variance_shrink_selector_std_ratio", 1.12)
            ),
            hn_local_variance_shrink_selector_softness=float(
                payload.get("hn_local_variance_shrink_selector_softness", 0.08)
            ),
            hn_local_variance_shrink_selector_min_support=float(
                payload.get("hn_local_variance_shrink_selector_min_support", 1.5)
            ),
            hn_local_variance_shrink_selector_mae_margin=float(
                payload.get("hn_local_variance_shrink_selector_mae_margin", 0.002)
            ),
            hn_local_variance_shrink_selector_ccc_margin=float(
                payload.get("hn_local_variance_shrink_selector_ccc_margin", 0.004)
            ),
            hn_local_variance_shrink_selector_max_abs=float(
                payload.get("hn_local_variance_shrink_selector_max_abs", 0.10)
            ),
            hn_local_variance_shrink_selector_min_risk=float(
                payload.get("hn_local_variance_shrink_selector_min_risk", 0.0)
            ),
            enable_hn_pseudomask_correction_selector=bool(
                payload.get("enable_hn_pseudomask_correction_selector", False)
            ),
            hn_pseudomask_correction_selector_strength=float(
                payload.get("hn_pseudomask_correction_selector_strength", 0.0)
            ),
            hn_pseudomask_correction_selector_min_rows=int(
                payload.get("hn_pseudomask_correction_selector_min_rows", 8)
            ),
            hn_pseudomask_correction_selector_query_fraction=float(
                payload.get("hn_pseudomask_correction_selector_query_fraction", 0.25)
            ),
            hn_pseudomask_correction_selector_min_query_rows=int(
                payload.get("hn_pseudomask_correction_selector_min_query_rows", 3)
            ),
            hn_pseudomask_correction_selector_mae_margin=float(
                payload.get("hn_pseudomask_correction_selector_mae_margin", 0.0)
            ),
            hn_pseudomask_correction_selector_ccc_margin=float(
                payload.get("hn_pseudomask_correction_selector_ccc_margin", 0.0)
            ),
            hn_pseudomask_correction_selector_residue_sigma=float(
                payload.get("hn_pseudomask_correction_selector_residue_sigma", 8.0)
            ),
            hn_pseudomask_correction_selector_shift_sigma=float(
                payload.get("hn_pseudomask_correction_selector_shift_sigma", 0.35)
            ),
            hn_pseudomask_correction_selector_feature_sigma=float(
                payload.get("hn_pseudomask_correction_selector_feature_sigma", 0.45)
            ),
            hn_pseudomask_correction_selector_min_support=float(
                payload.get("hn_pseudomask_correction_selector_min_support", 0.0)
            ),
            hn_pseudomask_correction_selector_min_sign_consensus=float(
                payload.get("hn_pseudomask_correction_selector_min_sign_consensus", 0.0)
            ),
            hn_pseudomask_correction_selector_max_abs=float(
                payload.get("hn_pseudomask_correction_selector_max_abs", 0.18)
            ),
            enable_hn_structure_pseudomask_correction_selector=bool(
                payload.get("enable_hn_structure_pseudomask_correction_selector", False)
            ),
            hn_structure_pseudomask_correction_selector_strength=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_strength",
                    0.0,
                )
            ),
            hn_structure_pseudomask_correction_selector_min_rows=int(
                payload.get("hn_structure_pseudomask_correction_selector_min_rows", 8)
            ),
            hn_structure_pseudomask_correction_selector_query_fraction=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_query_fraction",
                    0.25,
                )
            ),
            hn_structure_pseudomask_correction_selector_min_query_rows=int(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_min_query_rows",
                    3,
                )
            ),
            hn_structure_pseudomask_correction_selector_mae_margin=float(
                payload.get("hn_structure_pseudomask_correction_selector_mae_margin", 0.0)
            ),
            hn_structure_pseudomask_correction_selector_ccc_margin=float(
                payload.get("hn_structure_pseudomask_correction_selector_ccc_margin", 0.0)
            ),
            hn_structure_pseudomask_correction_selector_residue_sigma=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_residue_sigma",
                    8.0,
                )
            ),
            hn_structure_pseudomask_correction_selector_shift_sigma=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_shift_sigma",
                    0.35,
                )
            ),
            hn_structure_pseudomask_correction_selector_risk_feature_sigma=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_risk_feature_sigma",
                    0.45,
                )
            ),
            hn_structure_pseudomask_correction_selector_structure_sigma=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_structure_sigma",
                    0.30,
                )
            ),
            hn_structure_pseudomask_correction_selector_min_support=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_min_support",
                    0.0,
                )
            ),
            hn_structure_pseudomask_correction_selector_min_sign_consensus=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_min_sign_consensus",
                    0.0,
                )
            ),
            hn_structure_pseudomask_correction_selector_max_abs=float(
                payload.get("hn_structure_pseudomask_correction_selector_max_abs", 0.12)
            ),
            hn_structure_pseudomask_correction_selector_min_structure_match=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_min_structure_match",
                    0.70,
                )
            ),
            hn_structure_pseudomask_correction_selector_min_tail_focus=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_min_tail_focus",
                    0.0,
                )
            ),
            hn_structure_pseudomask_correction_selector_tail_softness=float(
                payload.get(
                    "hn_structure_pseudomask_correction_selector_tail_softness",
                    0.08,
                )
            ),
            enable_hn_structure_ridge_residual_selector=bool(
                payload.get("enable_hn_structure_ridge_residual_selector", False)
            ),
            hn_structure_ridge_residual_selector_strength=float(
                payload.get("hn_structure_ridge_residual_selector_strength", 0.0)
            ),
            hn_structure_ridge_residual_selector_min_rows=int(
                payload.get("hn_structure_ridge_residual_selector_min_rows", 12)
            ),
            hn_structure_ridge_residual_selector_min_support_rows=int(
                payload.get("hn_structure_ridge_residual_selector_min_support_rows", 8)
            ),
            hn_structure_ridge_residual_selector_query_fraction=float(
                payload.get("hn_structure_ridge_residual_selector_query_fraction", 0.35)
            ),
            hn_structure_ridge_residual_selector_min_query_rows=int(
                payload.get("hn_structure_ridge_residual_selector_min_query_rows", 4)
            ),
            hn_structure_ridge_residual_selector_mae_margin=float(
                payload.get("hn_structure_ridge_residual_selector_mae_margin", 0.001)
            ),
            hn_structure_ridge_residual_selector_ccc_margin=float(
                payload.get("hn_structure_ridge_residual_selector_ccc_margin", 0.003)
            ),
            hn_structure_ridge_residual_selector_ridge=float(
                payload.get("hn_structure_ridge_residual_selector_ridge", 10.0)
            ),
            hn_structure_ridge_residual_selector_max_abs=float(
                payload.get("hn_structure_ridge_residual_selector_max_abs", 0.08)
            ),
            hn_structure_ridge_residual_selector_min_structure_match=float(
                payload.get(
                    "hn_structure_ridge_residual_selector_min_structure_match",
                    0.80,
                )
            ),
            hn_structure_ridge_residual_selector_min_tail_focus=float(
                payload.get("hn_structure_ridge_residual_selector_min_tail_focus", 0.45)
            ),
            hn_structure_ridge_residual_selector_exclude_alignment_zero_ring=bool(
                payload.get(
                    "hn_structure_ridge_residual_selector_exclude_alignment_zero_ring",
                    False,
                )
            ),
            hn_structure_ridge_residual_selector_exclude_alignment_min=float(
                payload.get(
                    "hn_structure_ridge_residual_selector_exclude_alignment_min",
                    0.50,
                )
            ),
            hn_structure_ridge_residual_selector_exclude_ring_max=float(
                payload.get(
                    "hn_structure_ridge_residual_selector_exclude_ring_max",
                    0.02,
                )
            ),
            hn_structure_ridge_residual_selector_max_masked_mean_ring=float(
                payload.get(
                    "hn_structure_ridge_residual_selector_max_masked_mean_ring",
                    0.0,
                )
            ),
            hn_structure_ridge_residual_selector_max_masked_pred_raw_std_ratio=float(
                payload.get(
                    "hn_structure_ridge_residual_selector_max_masked_pred_raw_std_ratio",
                    0.0,
                )
            ),
            enable_hn_pseudomask_affine_selector=bool(
                payload.get("enable_hn_pseudomask_affine_selector", False)
            ),
            hn_pseudomask_affine_selector_strength=float(
                payload.get("hn_pseudomask_affine_selector_strength", 0.0)
            ),
            hn_pseudomask_affine_selector_min_rows=int(
                payload.get("hn_pseudomask_affine_selector_min_rows", 10)
            ),
            hn_pseudomask_affine_selector_query_fraction=float(
                payload.get("hn_pseudomask_affine_selector_query_fraction", 0.25)
            ),
            hn_pseudomask_affine_selector_min_query_rows=int(
                payload.get("hn_pseudomask_affine_selector_min_query_rows", 3)
            ),
            hn_pseudomask_affine_selector_mae_margin=float(
                payload.get("hn_pseudomask_affine_selector_mae_margin", 0.0)
            ),
            hn_pseudomask_affine_selector_ccc_margin=float(
                payload.get("hn_pseudomask_affine_selector_ccc_margin", 0.0)
            ),
            hn_pseudomask_affine_selector_scale_min=float(
                payload.get("hn_pseudomask_affine_selector_scale_min", 0.55)
            ),
            hn_pseudomask_affine_selector_scale_max=float(
                payload.get("hn_pseudomask_affine_selector_scale_max", 1.35)
            ),
            hn_pseudomask_affine_selector_max_abs=float(
                payload.get("hn_pseudomask_affine_selector_max_abs", 0.12)
            ),
            hn_pseudomask_affine_selector_min_observed_std_ratio=float(
                payload.get(
                    "hn_pseudomask_affine_selector_min_observed_std_ratio",
                    0.0,
                )
            ),
            hn_pseudomask_affine_selector_max_observed_abs_bias=float(
                payload.get(
                    "hn_pseudomask_affine_selector_max_observed_abs_bias",
                    0.0,
                )
            ),
            hn_pseudomask_affine_selector_min_validation_gate=float(
                payload.get("hn_pseudomask_affine_selector_min_validation_gate", 0.0)
            ),
            hn_pseudomask_affine_selector_min_masked_raw_observed_std_ratio=float(
                payload.get(
                    "hn_pseudomask_affine_selector_min_masked_raw_observed_std_ratio",
                    0.0,
                )
            ),
            hn_pseudomask_affine_selector_max_masked_pred_raw_std_ratio=float(
                payload.get(
                    "hn_pseudomask_affine_selector_max_masked_pred_raw_std_ratio",
                    0.0,
                )
            ),
            hn_pseudomask_affine_selector_min_masked_ring_mean=float(
                payload.get(
                    "hn_pseudomask_affine_selector_min_masked_ring_mean",
                    0.0,
                )
            ),
            enable_hn_regime_gate=bool(
                payload.get("enable_hn_regime_gate", False)
            ),
            hn_regime_loss_weight=float(
                payload.get("hn_regime_loss_weight", 0.0)
            ),
            hn_regime_delta_weight=float(
                payload.get("hn_regime_delta_weight", 0.0)
            ),
            hn_regime_feature_blend=float(
                payload.get("hn_regime_feature_blend", 0.0)
            ),
            enable_hn_state_coordinate_head=bool(
                payload.get("enable_hn_state_coordinate_head", False)
            ),
            hn_state_coordinate_dim=int(payload.get("hn_state_coordinate_dim", 7)),
            hn_state_coordinate_names=list(
                payload.get(
                    "hn_state_coordinate_names",
                    [
                        "ring_current",
                        "exchange",
                        "alignment_mismatch",
                        "sulfur_electrostatic",
                        "terminal_disorder",
                        "spread_collapse",
                        "graph_fragility",
                    ],
                )
            ),
            enable_hn_tangent_residual_adapter=bool(
                payload.get("enable_hn_tangent_residual_adapter", False)
            ),
            hn_tangent_residual_weight=float(
                payload.get("hn_tangent_residual_weight", 0.0)
            ),
            hn_tangent_residual_max_abs=float(
                payload.get("hn_tangent_residual_max_abs", 0.04)
            ),
            bioemu_tangent_residual_families=list(
                payload.get("bioemu_tangent_residual_families", ["HN"])
            ),
            hn_tangent_allowed_state_names=list(
                payload.get("hn_tangent_allowed_state_names", [])
            ),
            hn_tangent_state_chart_weights=dict(
                payload.get("hn_tangent_state_chart_weights", {})
            ),
            hn_tangent_state_gate_mode=str(
                payload.get("hn_tangent_state_gate_mode", "prior_overlap")
            ),
            hn_tangent_state_gate_margin=float(
                payload.get("hn_tangent_state_gate_margin", 0.10)
            ),
            hn_tangent_state_gate_softness=float(
                payload.get("hn_tangent_state_gate_softness", 0.06)
            ),
            enable_hn_tangent_utility_gate=bool(
                payload.get("enable_hn_tangent_utility_gate", False)
            ),
            hn_tangent_utility_gate_bias_init=float(
                payload.get("hn_tangent_utility_gate_bias_init", 6.0)
            ),
            hn_tangent_utility_gate_min=float(
                payload.get("hn_tangent_utility_gate_min", 0.0)
            ),
            hn_tangent_utility_gate_max=float(
                payload.get("hn_tangent_utility_gate_max", 1.0)
            ),
            enable_hn_tangent_chart_utility_gate=bool(
                payload.get("enable_hn_tangent_chart_utility_gate", False)
            ),
            hn_tangent_chart_utility_gate_init=dict(
                payload.get("hn_tangent_chart_utility_gate_init", {})
            ),
            enable_hn_tangent_feature_chart_gate=bool(
                payload.get("enable_hn_tangent_feature_chart_gate", False)
            ),
            hn_tangent_feature_chart_gate_state_names=list(
                payload.get("hn_tangent_feature_chart_gate_state_names", [])
            ),
            hn_tangent_feature_chart_gate_require_dominant_state=bool(
                payload.get(
                    "hn_tangent_feature_chart_gate_require_dominant_state",
                    False,
                )
            ),
            hn_tangent_feature_chart_gate_min=float(
                payload.get("hn_tangent_feature_chart_gate_min", 0.50)
            ),
            hn_tangent_feature_chart_gate_strength=float(
                payload.get("hn_tangent_feature_chart_gate_strength", 1.0)
            ),
            hn_tangent_utility_supervision_loss_weight=float(
                payload.get("hn_tangent_utility_supervision_loss_weight", 0.0)
            ),
            hn_tangent_utility_supervision_min_abs=float(
                payload.get("hn_tangent_utility_supervision_min_abs", 1e-5)
            ),
            enable_hn_tangent_signed_utility_gate=bool(
                payload.get("enable_hn_tangent_signed_utility_gate", False)
            ),
            hn_tangent_signed_utility_gate_bias_init=float(
                payload.get("hn_tangent_signed_utility_gate_bias_init", 6.0)
            ),
            hn_tangent_signed_utility_supervision_loss_weight=float(
                payload.get(
                    "hn_tangent_signed_utility_supervision_loss_weight",
                    0.0,
                )
            ),
            hn_tangent_signed_utility_supervision_min_abs=float(
                payload.get("hn_tangent_signed_utility_supervision_min_abs", 1e-5)
            ),
            hn_tangent_direction_loss_weight=float(
                payload.get("hn_tangent_direction_loss_weight", 0.0)
            ),
            hn_tangent_direction_loss_max_abs=float(
                payload.get("hn_tangent_direction_loss_max_abs", 0.006)
            ),
            hn_manifold_contrastive_loss_weight=float(
                payload.get("hn_manifold_contrastive_loss_weight", 0.0)
            ),
            hn_topology_laplacian_loss_weight=float(
                payload.get("hn_topology_laplacian_loss_weight", 0.0)
            ),
            hn_state_reliability_loss_weight=float(
                payload.get("hn_state_reliability_loss_weight", 0.0)
            ),
            hn_state_abstain_uncertainty_threshold=float(
                payload.get("hn_state_abstain_uncertainty_threshold", 0.65)
            ),
            enable_hn_evidence_graph_residual=bool(
                payload.get("enable_hn_evidence_graph_residual", False)
            ),
            hn_graph_residual_weight=float(
                payload.get("hn_graph_residual_weight", 0.0)
            ),
            hn_graph_residual_max_abs=float(
                payload.get("hn_graph_residual_max_abs", 0.10)
            ),
            hn_graph_residual_min_support=float(
                payload.get("hn_graph_residual_min_support", 0.45)
            ),
            hn_graph_residual_min_sign_consensus=float(
                payload.get("hn_graph_residual_min_sign_consensus", 0.35)
            ),
            hn_graph_residual_residue_sigma=float(
                payload.get("hn_graph_residual_residue_sigma", 6.0)
            ),
            hn_graph_residual_shift_sigma=float(
                payload.get("hn_graph_residual_shift_sigma", 0.35)
            ),
            hn_graph_residual_feature_sigma=float(
                payload.get("hn_graph_residual_feature_sigma", 0.45)
            ),
            hn_graph_residual_risk_gate_min=float(
                payload.get("hn_graph_residual_risk_gate_min", 0.0)
            ),
            hn_graph_residual_terminal_suppression_distance=float(
                payload.get("hn_graph_residual_terminal_suppression_distance", 0.0)
            ),
            hn_graph_residual_target_families=list(
                payload.get("hn_graph_residual_target_families", ["HN"])
            ),
            hn_graph_residual_evidence_families=list(
                payload.get(
                    "hn_graph_residual_evidence_families",
                    ["HN", "N", "CA", "C'"],
                )
            ),
            hn95_loss_weight=float(payload.get("hn95_loss_weight", 0.0)),
            hn95_pairwise_rank_loss_weight=float(
                payload.get("hn95_pairwise_rank_loss_weight", 0.0)
            ),
            hn95_highrisk_tail_loss_weight=float(
                payload.get("hn95_highrisk_tail_loss_weight", 0.0)
            ),
            hn95_regime_balanced_loss_weight=float(
                payload.get("hn95_regime_balanced_loss_weight", 0.0)
            ),
            hn95_graph_consistency_loss_weight=float(
                payload.get("hn95_graph_consistency_loss_weight", 0.0)
            ),
            hn95_reconstruction_loss_weight=float(
                payload.get("hn95_reconstruction_loss_weight", 0.0)
            ),
            cprime_guardrail_loss_weight=float(
                payload.get("cprime_guardrail_loss_weight", 0.0)
            ),
            cprime_guardrail_ccc_min=float(
                payload.get("cprime_guardrail_ccc_min", 0.955)
            ),
            enable_hn_local_interpolation=bool(
                payload.get("enable_hn_local_interpolation", True)
            ),
            hn_local_interpolation_blend=float(
                payload.get("hn_local_interpolation_blend", 0.25)
            ),
            hn_local_interpolation_max_distance=int(
                payload.get("hn_local_interpolation_max_distance", 4)
            ),
            enable_bioemu_hn_evidence_local_residual_readout=bool(
                payload.get(
                    "enable_bioemu_hn_evidence_local_residual_readout",
                    False,
                )
            ),
            bioemu_hn_evidence_local_residual_strength=float(
                payload.get("bioemu_hn_evidence_local_residual_strength", 0.50)
            ),
            bioemu_hn_evidence_local_residual_max_abs=float(
                payload.get("bioemu_hn_evidence_local_residual_max_abs", 0.08)
            ),
            bioemu_hn_evidence_local_residual_min_support=float(
                payload.get("bioemu_hn_evidence_local_residual_min_support", 0.25)
            ),
            bioemu_hn_evidence_local_residual_residue_sigma=float(
                payload.get("bioemu_hn_evidence_local_residual_residue_sigma", 12.0)
            ),
            bioemu_hn_evidence_local_residual_shift_sigma=float(
                payload.get("bioemu_hn_evidence_local_residual_shift_sigma", 0.12)
            ),
            bioemu_hn_evidence_local_residual_min_sign_consensus=float(
                payload.get(
                    "bioemu_hn_evidence_local_residual_min_sign_consensus",
                    0.0,
                )
            ),
            enable_bioemu_hn_evidence_local_residual_pseudomask_guard=bool(
                payload.get(
                    "enable_bioemu_hn_evidence_local_residual_pseudomask_guard",
                    False,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_min_rows=int(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_min_rows",
                    12,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_query_fraction=float(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_query_fraction",
                    0.30,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_min_query_rows=int(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_min_query_rows",
                    4,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_mae_margin=float(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_mae_margin",
                    0.0,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_ccc_margin=float(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_ccc_margin",
                    0.0,
                )
            ),
            bioemu_hn_evidence_local_residual_pseudomask_min_gate=float(
                payload.get(
                    "bioemu_hn_evidence_local_residual_pseudomask_min_gate",
                    0.35,
                )
            ),
            enable_bioemu_hn_x2d_support_counter_guard=bool(
                payload.get(
                    "enable_bioemu_hn_x2d_support_counter_guard",
                    False,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_max_abs=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_max_abs",
                    0.0,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_pair_center=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_pair_center",
                    3.935,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_pair_softness=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_pair_softness",
                    0.006,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_sidecar_center=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_sidecar_center",
                    1.60,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_sidecar_softness=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_sidecar_softness",
                    0.08,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_direction_center=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_direction_center",
                    0.018,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_direction_softness=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_direction_softness",
                    0.010,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_min_gate=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_min_gate",
                    0.0,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_suppressed_residue_names=list(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_suppressed_residue_names",
                    [],
                )
            ),
            bioemu_hn_x2d_support_counter_guard_relief_residue_names=list(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_relief_residue_names",
                    [],
                )
            ),
            bioemu_hn_x2d_support_counter_guard_relief_max_abs=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_relief_max_abs",
                    0.0,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_relief_min_gate=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_relief_min_gate",
                    0.0,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_rescue_residue_names=list(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_rescue_residue_names",
                    [],
                )
            ),
            bioemu_hn_x2d_support_counter_guard_rescue_max_abs=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_rescue_max_abs",
                    0.0,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_rescue_pair_center=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_rescue_pair_center",
                    3.932,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_rescue_sidecar_center=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_rescue_sidecar_center",
                    1.30,
                )
            ),
            bioemu_hn_x2d_support_counter_guard_rescue_min_gate=float(
                payload.get(
                    "bioemu_hn_x2d_support_counter_guard_rescue_min_gate",
                    0.0,
                )
            ),
            enable_bioemu_family_specific_evidence_context=bool(
                payload.get(
                    "enable_bioemu_family_specific_evidence_context",
                    False,
                )
            ),
            bioemu_family_specific_evidence_source_families=dict(
                payload.get(
                    "bioemu_family_specific_evidence_source_families",
                    {
                        "HN": ["HN", "N", "C'"],
                        "C'": ["C'", "CA", "N", "HN"],
                    },
                )
            ),
            bioemu_family_specific_evidence_context_strength_by_family=dict(
                payload.get(
                    "bioemu_family_specific_evidence_context_strength_by_family",
                    {"HN": 0.35, "C'": 0.18},
                )
            ),
            enable_bioemu_hn_low_ess_local_residual_gate=bool(
                payload.get(
                    "enable_bioemu_hn_low_ess_local_residual_gate",
                    False,
                )
            ),
            bioemu_hn_low_ess_local_residual_ess_floor=float(
                payload.get("bioemu_hn_low_ess_local_residual_ess_floor", 3.0)
            ),
            bioemu_hn_low_ess_local_residual_min_scale=float(
                payload.get("bioemu_hn_low_ess_local_residual_min_scale", 0.50)
            ),
            enable_bioemu_hn_entity_residue_local_residual_gate=bool(
                payload.get(
                    "enable_bioemu_hn_entity_residue_local_residual_gate",
                    False,
                )
            ),
            bioemu_hn_entity_residue_local_residual_gate_entities=list(
                payload.get(
                    "bioemu_hn_entity_residue_local_residual_gate_entities",
                    [],
                )
            ),
            bioemu_hn_entity_residue_local_residual_gate_residues=list(
                payload.get(
                    "bioemu_hn_entity_residue_local_residual_gate_residues",
                    [],
                )
            ),
            bioemu_hn_entity_residue_local_residual_gate_target_ids=list(
                payload.get(
                    "bioemu_hn_entity_residue_local_residual_gate_target_ids",
                    [],
                )
            ),
            bioemu_hn_entity_residue_local_residual_gate_factor=float(
                payload.get(
                    "bioemu_hn_entity_residue_local_residual_gate_factor",
                    0.0,
                )
            ),
            enable_bioemu_hn_evidence_local_variance_restore=bool(
                payload.get(
                    "enable_bioemu_hn_evidence_local_variance_restore",
                    False,
                )
            ),
            bioemu_hn_evidence_local_variance_restore_strength=float(
                payload.get(
                    "bioemu_hn_evidence_local_variance_restore_strength",
                    1.0,
                )
            ),
            bioemu_hn_evidence_local_variance_restore_min_rows=int(
                payload.get("bioemu_hn_evidence_local_variance_restore_min_rows", 4)
            ),
            bioemu_hn_evidence_local_variance_restore_scale_min=float(
                payload.get(
                    "bioemu_hn_evidence_local_variance_restore_scale_min",
                    0.75,
                )
            ),
            bioemu_hn_evidence_local_variance_restore_scale_max=float(
                payload.get(
                    "bioemu_hn_evidence_local_variance_restore_scale_max",
                    1.35,
                )
            ),
            bioemu_hn_evidence_local_variance_restore_scale_floor=float(
                payload.get(
                    "bioemu_hn_evidence_local_variance_restore_scale_floor",
                    1.0,
                )
            ),
            enable_bioemu_hn_post_spread_saturated_ceiling=bool(
                payload.get(
                    "enable_bioemu_hn_post_spread_saturated_ceiling",
                    False,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_min_support=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_min_support",
                    3.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_min_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_min_local_delta_abs",
                    0.085,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_max_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_max_abs",
                    0.025,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_support_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_support_softness",
                    3.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_direction=str(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_direction",
                    "negative",
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_high_side_min_mu=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_high_side_min_mu",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_high_side_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_high_side_softness",
                    0.25,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_high_side_floor=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_high_side_floor",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_boost_scale=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_boost_scale",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_min_mu=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_min_mu",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_mu_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_mu_softness",
                    0.25,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_min_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_min_local_delta_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_local_delta_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_local_delta_softness",
                    0.05,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_min_support=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_min_support",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_risk_support_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_risk_support_softness",
                    2.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_scale=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_scale",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_min_mu=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_min_mu",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_mu_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_mu_softness",
                    0.35,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_min_support=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_min_support",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_support_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_support_softness",
                    2.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_min_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_min_local_delta_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_local_delta_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_local_delta_softness",
                    0.05,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_nonlocal_floor=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_nonlocal_floor",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_min_norm=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_min_norm",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_pair_softness",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_min_norm=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_min_norm",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_active_boost_sidecar_softness",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_mu=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_mu",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_mu_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_mu_softness",
                    0.35,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_support=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_support",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_support_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_support_softness",
                    3.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_require_negative_local=bool(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_require_negative_local",
                    False,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_min_local_delta_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_weak_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_max_weak_local_delta_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_nonlocal_floor=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_nonlocal_floor",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_min_norm=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_min_norm",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_pair_softness",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_min_norm=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_min_norm",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_sidecar_softness",
                    1.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_abs",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_names=list(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_names",
                    [],
                )
                or []
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_support_bins=list(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_residue_support_bins",
                    [],
                )
                or []
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_mu=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_mu",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_mu_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_mu_softness",
                    0.25,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_support=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_min_support",
                    0.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_support_softness=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_support_softness",
                    2.0,
                )
            ),
            bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_local_delta_abs=float(
                payload.get(
                    "bioemu_hn_post_spread_saturated_ceiling_missed_rescue_neutral_fallback_max_local_delta_abs",
                    0.02,
                )
            ),
            enable_bioemu_hn_ring_coordinate_delta_gate=bool(
                payload.get("enable_bioemu_hn_ring_coordinate_delta_gate", False)
            ),
            bioemu_hn_ring_coordinate_delta_gate_coordinate_name=str(
                payload.get(
                    "bioemu_hn_ring_coordinate_delta_gate_coordinate_name",
                    "ring_orientation",
                )
            ),
            bioemu_hn_ring_coordinate_delta_gate_quantile=float(
                payload.get("bioemu_hn_ring_coordinate_delta_gate_quantile", 0.10)
            ),
            bioemu_hn_ring_coordinate_delta_gate_side=str(
                payload.get("bioemu_hn_ring_coordinate_delta_gate_side", "lo")
            ),
            bioemu_hn_ring_coordinate_delta_gate_factor=float(
                payload.get("bioemu_hn_ring_coordinate_delta_gate_factor", 1.0)
            ),
            enable_bioemu_same_family_hn_ring_coordinate_delta_gate=bool(
                payload.get(
                    "enable_bioemu_same_family_hn_ring_coordinate_delta_gate",
                    False,
                )
            ),
            bioemu_same_family_hn_ring_coordinate_delta_gate_coordinate_name=str(
                payload.get(
                    "bioemu_same_family_hn_ring_coordinate_delta_gate_coordinate_name",
                    "ring_orientation",
                )
            ),
            bioemu_same_family_hn_ring_coordinate_delta_gate_quantile=float(
                payload.get(
                    "bioemu_same_family_hn_ring_coordinate_delta_gate_quantile",
                    0.25,
                )
            ),
            bioemu_same_family_hn_ring_coordinate_delta_gate_side=str(
                payload.get(
                    "bioemu_same_family_hn_ring_coordinate_delta_gate_side",
                    "lo",
                )
            ),
            bioemu_same_family_hn_ring_coordinate_delta_gate_factor=float(
                payload.get(
                    "bioemu_same_family_hn_ring_coordinate_delta_gate_factor",
                    0.35,
                )
            ),
            enable_bioemu_hn_ring_reliability_delta_gate=bool(
                payload.get("enable_bioemu_hn_ring_reliability_delta_gate", False)
            ),
            bioemu_hn_ring_reliability_delta_gate_coordinate_name=str(
                payload.get(
                    "bioemu_hn_ring_reliability_delta_gate_coordinate_name",
                    "ring_orientation",
                )
            ),
            bioemu_hn_ring_reliability_delta_gate_quantile=float(
                payload.get("bioemu_hn_ring_reliability_delta_gate_quantile", 0.50)
            ),
            bioemu_hn_ring_reliability_delta_gate_side=str(
                payload.get("bioemu_hn_ring_reliability_delta_gate_side", "hi")
            ),
            bioemu_hn_ring_reliability_delta_gate_factor=float(
                payload.get("bioemu_hn_ring_reliability_delta_gate_factor", 1.0)
            ),
            enable_bioemu_cprime_evidence_local_residual_readout=bool(
                payload.get(
                    "enable_bioemu_cprime_evidence_local_residual_readout",
                    False,
                )
            ),
            bioemu_cprime_evidence_local_residual_strength=float(
                payload.get("bioemu_cprime_evidence_local_residual_strength", 1.25)
            ),
            bioemu_cprime_evidence_local_residual_max_abs=float(
                payload.get("bioemu_cprime_evidence_local_residual_max_abs", 0.25)
            ),
            bioemu_cprime_evidence_local_residual_negative_max_abs=float(
                payload.get(
                    "bioemu_cprime_evidence_local_residual_negative_max_abs",
                    payload.get("bioemu_cprime_evidence_local_residual_max_abs", 0.25),
                )
            ),
            bioemu_cprime_evidence_local_residual_positive_max_abs=float(
                payload.get(
                    "bioemu_cprime_evidence_local_residual_positive_max_abs",
                    payload.get("bioemu_cprime_evidence_local_residual_max_abs", 0.25),
                )
            ),
            bioemu_cprime_evidence_local_residual_min_support=float(
                payload.get("bioemu_cprime_evidence_local_residual_min_support", 0.25)
            ),
            bioemu_cprime_evidence_local_residual_residue_sigma=float(
                payload.get(
                    "bioemu_cprime_evidence_local_residual_residue_sigma",
                    4.0,
                )
            ),
            bioemu_cprime_evidence_local_residual_shift_sigma=float(
                payload.get("bioemu_cprime_evidence_local_residual_shift_sigma", 0.0)
            ),
            bioemu_cprime_evidence_local_residual_min_sign_consensus=float(
                payload.get(
                    "bioemu_cprime_evidence_local_residual_min_sign_consensus",
                    0.0,
                )
            ),
            enable_bioemu_cprime_evidence_local_variance_restore=bool(
                payload.get(
                    "enable_bioemu_cprime_evidence_local_variance_restore",
                    False,
                )
            ),
            bioemu_cprime_evidence_local_variance_restore_strength=float(
                payload.get(
                    "bioemu_cprime_evidence_local_variance_restore_strength",
                    1.0,
                )
            ),
            bioemu_cprime_evidence_local_variance_restore_min_rows=int(
                payload.get(
                    "bioemu_cprime_evidence_local_variance_restore_min_rows",
                    4,
                )
            ),
            bioemu_cprime_evidence_local_variance_restore_scale_min=float(
                payload.get(
                    "bioemu_cprime_evidence_local_variance_restore_scale_min",
                    0.75,
                )
            ),
            bioemu_cprime_evidence_local_variance_restore_scale_max=float(
                payload.get(
                    "bioemu_cprime_evidence_local_variance_restore_scale_max",
                    1.35,
                )
            ),
            bioemu_cprime_evidence_local_variance_restore_scale_floor=float(
                payload.get(
                    "bioemu_cprime_evidence_local_variance_restore_scale_floor",
                    1.0,
                )
            ),
            enable_bioemu_cprime_carbonyl_uncertainty_delta_gate=bool(
                payload.get(
                    "enable_bioemu_cprime_carbonyl_uncertainty_delta_gate",
                    False,
                )
            ),
            bioemu_cprime_carbonyl_uncertainty_delta_gate_quantile=float(
                payload.get(
                    "bioemu_cprime_carbonyl_uncertainty_delta_gate_quantile",
                    0.67,
                )
            ),
            bioemu_cprime_carbonyl_uncertainty_delta_gate_factor=float(
                payload.get(
                    "bioemu_cprime_carbonyl_uncertainty_delta_gate_factor",
                    0.0,
                )
            ),
            enable_bioemu_cprime_carbonyl_uncertainty_delta_low_boost=bool(
                payload.get(
                    "enable_bioemu_cprime_carbonyl_uncertainty_delta_low_boost",
                    False,
                )
            ),
            bioemu_cprime_carbonyl_uncertainty_delta_low_boost_quantile=float(
                payload.get(
                    "bioemu_cprime_carbonyl_uncertainty_delta_low_boost_quantile",
                    0.60,
                )
            ),
            bioemu_cprime_carbonyl_uncertainty_delta_low_boost_factor=float(
                payload.get(
                    "bioemu_cprime_carbonyl_uncertainty_delta_low_boost_factor",
                    1.0,
                )
            ),
            enable_hn_extended_evidence_context=bool(
                payload.get("enable_hn_extended_evidence_context", False)
            ),
            enable_candidate_free_entry_condition_context=bool(
                payload.get(
                    "enable_candidate_free_entry_condition_context",
                    False,
                )
            ),
            candidate_free_entry_condition_gate_init=float(
                payload.get("candidate_free_entry_condition_gate_init", 0.15)
            ),
            enable_hn_residual_moment_head=bool(
                payload.get("enable_hn_residual_moment_head", False)
            ),
            hn_residual_moment_weight=float(
                payload.get("hn_residual_moment_weight", 0.35)
            ),
            hn_risk_residual_moment_weight=float(
                payload.get("hn_risk_residual_moment_weight", 0.0)
            ),
            enable_hn_physics_specialist_head=bool(
                payload.get("enable_hn_physics_specialist_head", False)
            ),
            hn_physics_specialist_weight=float(
                payload.get("hn_physics_specialist_weight", 0.0)
            ),
            hn_physics_specialist_min_risk=float(
                payload.get("hn_physics_specialist_min_risk", 0.35)
            ),
            enable_cprime_residual_moment_head=bool(
                payload.get("enable_cprime_residual_moment_head", False)
            ),
            cprime_residual_moment_weight=float(
                payload.get("cprime_residual_moment_weight", 0.45)
            ),
            cprime_context_residual_moment_weight=float(
                payload.get("cprime_context_residual_moment_weight", 0.0)
            ),
            enable_hn_n_paired_residual_correction=bool(
                payload.get("enable_hn_n_paired_residual_correction", False)
            ),
            hn_n_paired_residual_weight=float(
                payload.get("hn_n_paired_residual_weight", 0.04)
            ),
            hn_n_paired_residual_max_abs=float(
                payload.get("hn_n_paired_residual_max_abs", 0.20)
            ),
            hn_n_paired_residual_detach=bool(
                payload.get("hn_n_paired_residual_detach", True)
            ),
            hn_n_paired_residual_neighbor_sigma=float(
                payload.get("hn_n_paired_residual_neighbor_sigma", 0.0)
            ),
            hn_n_paired_residual_max_distance=int(
                payload.get("hn_n_paired_residual_max_distance", 0)
            ),
            enable_candidate_observable_context=bool(
                payload.get("enable_candidate_observable_context", False)
            ),
            candidate_observable_context_gate_init=float(
                payload.get("candidate_observable_context_gate_init", 1.0)
            ),
            candidate_observable_context_target_families=list(
                payload.get("candidate_observable_context_target_families", [])
            ),
            training_input_mode=str(
                payload.get("training_input_mode", "legacy_cs_fit")
            ),
            enable_bioemu_latent_nmr=bool(
                payload.get("enable_bioemu_latent_nmr", False)
            ),
            bioemu_latent_provider=str(
                payload.get("bioemu_latent_provider", "fixture")
            ),
            bioemu_x0_required_provider_class=payload.get(
                "bioemu_x0_required_provider_class"
            ),
            bioemu_x0_allow_fixture_provider_for_diagnostic=bool(
                payload.get("bioemu_x0_allow_fixture_provider_for_diagnostic", False)
            ),
            bioemu_latent_fixture_path=payload.get("bioemu_latent_fixture_path"),
            bioemu_repo_root=payload.get("bioemu_repo_root"),
            bioemu_checkpoint_path=payload.get("bioemu_checkpoint_path"),
            base_bioemu_checkpoint_path=payload.get("base_bioemu_checkpoint_path"),
            bioemu_score_checkpoint_path=payload.get("bioemu_score_checkpoint_path"),
            bioemu_score_model_config_path=payload.get(
                "bioemu_score_model_config_path"
            ),
            bioemu_score_checkpoint_is_online_generator=bool(
                payload.get("bioemu_score_checkpoint_is_online_generator", False)
            ),
            bioemu_score_adapter_checkpoint_path=payload.get(
                "bioemu_score_adapter_checkpoint_path"
            ),
            bioemu_score_adapter_checkpoint_kind=payload.get(
                "bioemu_score_adapter_checkpoint_kind"
            ),
            bioemu_score_adapter_runtime_applicable=bool(
                payload.get("bioemu_score_adapter_runtime_applicable", False)
            ),
            bioemu_score_adapter_handoff_summary_path=payload.get(
                "bioemu_score_adapter_handoff_summary_path"
            ),
            bioemu_score_adapter_handoff_ready=bool(
                payload.get("bioemu_score_adapter_handoff_ready", False)
            ),
            bioemu_score_adapter_handoff_teacher_probe_only=bool(
                payload.get("bioemu_score_adapter_handoff_teacher_probe_only", False)
            ),
            bioemu_score_adapter_handoff_final_acceptance_eligible=bool(
                payload.get(
                    "bioemu_score_adapter_handoff_final_acceptance_eligible",
                    False,
                )
            ),
            bioemu_score_adapter_handoff_writes_latest_accepted=bool(
                payload.get(
                    "bioemu_score_adapter_handoff_writes_latest_accepted",
                    False,
                )
            ),
            bioemu_score_adapter_handoff_use_for_latest_accepted_teacher=bool(
                payload.get(
                    "bioemu_score_adapter_handoff_use_for_latest_accepted_teacher",
                    False,
                )
            ),
            bioemu_score_adapter_handoff_guard_round_id=payload.get(
                "bioemu_score_adapter_handoff_guard_round_id"
            ),
            bioemu_score_adapter_handoff_artifact_kind=payload.get(
                "bioemu_score_adapter_handoff_artifact_kind"
            ),
            bioemu_score_adapter_handoff_shadow_equivalence_student_summary_path=payload.get(
                "bioemu_score_adapter_handoff_shadow_equivalence_student_summary_path"
            ),
            bioemu_official_x1d_adapter_enabled=bool(
                payload.get("bioemu_official_x1d_adapter_enabled", False)
            ),
            bioemu_official_x1d_adapter_train_through_structure=bool(
                payload.get(
                    "bioemu_official_x1d_adapter_train_through_structure",
                    False,
                )
            ),
            bioemu_official_x1d_adapter_latent_bridge_trainable=bool(
                payload.get(
                    "bioemu_official_x1d_adapter_latent_bridge_trainable",
                    False,
                )
            ),
            bioemu_official_x1d_adapter_rank=int(
                payload.get("bioemu_official_x1d_adapter_rank", 8)
            ),
            bioemu_official_x1d_adapter_scale=float(
                payload.get("bioemu_official_x1d_adapter_scale", 1.0)
            ),
            bioemu_official_x1d_adapter_max_abs=float(
                payload.get("bioemu_official_x1d_adapter_max_abs", 0.02)
            ),
            bioemu_official_x2d_adapter_enabled=bool(
                payload.get("bioemu_official_x2d_adapter_enabled", False)
            ),
            bioemu_official_x2d_adapter_train_through_structure=bool(
                payload.get(
                    "bioemu_official_x2d_adapter_train_through_structure",
                    False,
                )
            ),
            bioemu_official_x2d_adapter_rank=int(
                payload.get("bioemu_official_x2d_adapter_rank", 8)
            ),
            bioemu_official_x2d_adapter_scale=float(
                payload.get("bioemu_official_x2d_adapter_scale", 1.0)
            ),
            bioemu_official_x2d_adapter_max_abs=float(
                payload.get("bioemu_official_x2d_adapter_max_abs", 0.01)
            ),
            bioemu_official_x2d_adapter_wakeup_init_std=float(
                payload.get("bioemu_official_x2d_adapter_wakeup_init_std", 0.0)
            ),
            bioemu_official_x2d_adapter_wakeup_condition_output_std=float(
                payload.get(
                    "bioemu_official_x2d_adapter_wakeup_condition_output_std",
                    0.0,
                )
            ),
            bioemu_official_x2d_adapter_wakeup_zero_threshold=float(
                payload.get(
                    "bioemu_official_x2d_adapter_wakeup_zero_threshold",
                    1.0e-12,
                )
            ),
            bioemu_official_x2d_adapter_wakeup_seed=int(
                payload.get("bioemu_official_x2d_adapter_wakeup_seed", 1009)
            ),
            bioemu_official_x2d_adapter_wakeup_force=bool(
                payload.get("bioemu_official_x2d_adapter_wakeup_force", False)
            ),
            bioemu_official_adapter_conditioning_enabled=bool(
                payload.get("bioemu_official_adapter_conditioning_enabled", False)
            ),
            bioemu_official_adapter_conditioning_dim=int(
                payload.get("bioemu_official_adapter_conditioning_dim", 40)
            ),
            bioemu_official_adapter_conditioning_scale=float(
                payload.get("bioemu_official_adapter_conditioning_scale", 1.0)
            ),
            bioemu_official_adapter_gradient_sample_limit=int(
                payload.get("bioemu_official_adapter_gradient_sample_limit", 0)
            ),
            bioemu_provider_lr_multiplier=float(
                payload.get("bioemu_provider_lr_multiplier", 1.0)
            ),
            bioemu_provider_gradient_audit_enabled=bool(
                payload.get("bioemu_provider_gradient_audit_enabled", False)
            ),
            bioemu_x0_conditioning_guidance_enabled=bool(
                payload.get("bioemu_x0_conditioning_guidance_enabled", False)
            ),
            bioemu_x0_conditioning_guidance_example_limit=int(
                payload.get("bioemu_x0_conditioning_guidance_example_limit", 64)
            ),
            bioemu_x0_conditioning_guidance_sample_count=int(
                payload.get("bioemu_x0_conditioning_guidance_sample_count", 0)
            ),
            bioemu_x0_conditioning_guidance_anchor_ensemble_size=int(
                payload.get("bioemu_x0_conditioning_guidance_anchor_ensemble_size", 1)
            ),
            bioemu_x0_conditioning_guidance_seed_offset=int(
                payload.get("bioemu_x0_conditioning_guidance_seed_offset", 911_000)
            ),
            bioemu_x0_conditioning_guidance_anchor_scale_nm=float(
                payload.get("bioemu_x0_conditioning_guidance_anchor_scale_nm", 1.0)
            ),
            bioemu_x0_conditioning_guidance_max_abs_nm=float(
                payload.get("bioemu_x0_conditioning_guidance_max_abs_nm", 5.0)
            ),
            bioemu_x0_conditioning_guidance_promotable=bool(
                payload.get("bioemu_x0_conditioning_guidance_promotable", False)
            ),
            bioemu_x0_conditioning_guidance_denoising_scale=float(
                payload.get("bioemu_x0_conditioning_guidance_denoising_scale", 0.0)
            ),
            bioemu_x0_conditioning_guidance_denoising_max_norm_nm=float(
                payload.get(
                    "bioemu_x0_conditioning_guidance_denoising_max_norm_nm",
                    0.25,
                )
            ),
            bioemu_x0_conditioning_guidance_fail_on_error=bool(
                payload.get("bioemu_x0_conditioning_guidance_fail_on_error", False)
            ),
            bioemu_x0_generated_support_transfer_gap_mitigation=bool(
                payload.get(
                    "bioemu_x0_generated_support_transfer_gap_mitigation",
                    False,
                )
            ),
            bioemu_x0_generated_support_transfer_gap_source=payload.get(
                "bioemu_x0_generated_support_transfer_gap_source"
            ),
            bioemu_x0_generated_support_transfer_gap_summary_path=payload.get(
                "bioemu_x0_generated_support_transfer_gap_summary_path"
            ),
            bioemu_x0_epoch_number_offset=int(
                payload.get("bioemu_x0_epoch_number_offset", 0)
            ),
            bioemu_x0_support_face_contract_enabled=bool(
                payload.get("bioemu_x0_support_face_contract_enabled", False)
            ),
            bioemu_x0_support_face_baseline_history_path=payload.get(
                "bioemu_x0_support_face_baseline_history_path"
            ),
            bioemu_x0_support_face_target_baseline_label=payload.get(
                "bioemu_x0_support_face_target_baseline_label"
            ),
            bioemu_x0_support_face_target_baseline_epoch=int(
                payload.get("bioemu_x0_support_face_target_baseline_epoch", 0)
            ),
            bioemu_x0_support_face_epoch_matched_control=bool(
                payload.get("bioemu_x0_support_face_epoch_matched_control", True)
            ),
            bioemu_x0_support_face_promote_only_if_target_held=bool(
                payload.get("bioemu_x0_support_face_promote_only_if_target_held", False)
            ),
            bioemu_x0_support_face_stop_on_target_failure=bool(
                payload.get("bioemu_x0_support_face_stop_on_target_failure", False)
            ),
            bioemu_latent_dim=int(payload.get("bioemu_latent_dim", 64)),
            bioemu_projection_dim=int(payload.get("bioemu_projection_dim", 64)),
            bioemu_sample_count=int(payload.get("bioemu_sample_count", 512)),
            bioemu_allow_sub512_sample_count_for_diagnostic=bool(
                payload.get("bioemu_allow_sub512_sample_count_for_diagnostic", False)
            ),
            bioemu_finetune_mode=str(
                payload.get("bioemu_finetune_mode", "head_then_adapter")
            ),
            bioemu_structure_feature_policy=str(
                payload.get("bioemu_structure_feature_policy", "benchmark_only")
            ),
            enable_bioemu_observation_chain_atlas=bool(
                payload.get("enable_bioemu_observation_chain_atlas", False)
            ),
            bioemu_coordinate_layers=list(
                payload.get(
                    "bioemu_coordinate_layers",
                    [
                        "sequence",
                        "structural_prior",
                        "solution_ensemble",
                        "nmr_observation",
                    ],
                )
            ),
            bioemu_nmr_aggregator_invariance=str(
                payload.get("bioemu_nmr_aggregator_invariance", "permutation")
            ),
            bioemu_nmr_structure_probe_gradient_policy=str(
                payload.get(
                    "bioemu_nmr_structure_probe_gradient_policy",
                    "detached_benchmark_only",
                )
            ),
            bioemu_nmr_sequence_shortcut_penalty_weight=float(
                payload.get("bioemu_nmr_sequence_shortcut_penalty_weight", 0.0)
            ),
            bioemu_nmr_component_identifiability_loss_weight=float(
                payload.get(
                    "bioemu_nmr_component_identifiability_loss_weight",
                    0.0,
                )
            ),
            bioemu_sequence_embedding_provider=str(
                payload.get("bioemu_sequence_embedding_provider", "learned")
            ),
            bioemu_sequence_embedding_model=str(
                payload.get("bioemu_sequence_embedding_model", "esm3_sm_open_v1")
            ),
            bioemu_sequence_embedding_cache_dir=payload.get(
                "bioemu_sequence_embedding_cache_dir"
            ),
            bioemu_sequence_embedding_weight=float(
                payload.get("bioemu_sequence_embedding_weight", 1.0)
            ),
            bioemu_sequence_embedding_fail_on_missing=bool(
                payload.get("bioemu_sequence_embedding_fail_on_missing", True)
            ),
            bioemu_nmr_train_sample_count=int(
                payload.get("bioemu_nmr_train_sample_count", 512)
            ),
            bioemu_nmr_val_sample_count=int(
                payload.get("bioemu_nmr_val_sample_count", 512)
            ),
            bioemu_nmr_benchmark_sample_count=int(
                payload.get("bioemu_nmr_benchmark_sample_count", 512)
            ),
            bioemu_cs_reweighting_teacher_predictions_path=payload.get(
                "bioemu_cs_reweighting_teacher_predictions_path"
            ),
            bioemu_cs_reweighting_teacher_guard_path=payload.get(
                "bioemu_cs_reweighting_teacher_guard_path"
            ),
            bioemu_cs_reweighting_teacher_mean_loss_weight=float(
                payload.get("bioemu_cs_reweighting_teacher_mean_loss_weight", 0.0)
            ),
            bioemu_cs_reweighting_teacher_prior_mode=str(
                payload.get("bioemu_cs_reweighting_teacher_prior_mode", "uniform")
            ),
            bioemu_cs_reweighting_teacher_prior_source=str(
                payload.get(
                    "bioemu_cs_reweighting_teacher_prior_source",
                    "uniform_bioemu_conformer_pool",
                )
            ),
            bioemu_cs_reweighting_teacher_fit_atom_families=list(
                payload.get(
                    "bioemu_cs_reweighting_teacher_fit_atom_families",
                    ["H", "N", "CA", "CB"],
                )
            ),
            bioemu_cs_reweighting_teacher_prior_kl_weight=float(
                payload.get("bioemu_cs_reweighting_teacher_prior_kl_weight", 0.0)
            ),
            bioemu_cs_reweighting_teacher_prior_kl_temperature=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_prior_kl_temperature",
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_kl_weight=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_kl_weight",
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_kl_temperature=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_kl_temperature",
                    payload.get("bioemu_cs_reweighting_teacher_prior_kl_temperature", 1.0),
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_min_matched_samples=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "min_matched_samples"
                    ),
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_min_support_coverage=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "min_support_coverage"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_blend=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_cosat_blend",
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_blend_mode=str(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_blend_mode"
                    ),
                    "linear",
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_power=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_cosat_power",
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_scale=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_cosat_scale",
                    payload.get("bioemu_x0_posterior_oracle_distill_scale", 256.0),
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_focus_families=list(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_focus_families"
                    ),
                    [],
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_required_families=list(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_required_families"
                    ),
                    [],
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_fallback_required_families=list(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_fallback_required_families"
                    ),
                    [],
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_fallback_min_rows"
                    ),
                    0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_min_family_count=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_min_family_count"
                    ),
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_cosat_family_weights=dict(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "cosat_family_weights"
                    ),
                    {},
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_topk_mass_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "topk_mass_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_topk_fraction=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_topk_fraction",
                    0.125,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_topk_min_count=int(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_topk_min_count",
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_topk_target_mass_scale=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "topk_target_mass_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_topk_listwise_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "topk_listwise_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_nll_weight=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_top1_nll_weight",
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_tail_mass_contract_required=bool(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "tail_mass_contract_required"
                    ),
                    False,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_tail_mass_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "tail_mass_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_tail_fraction=float(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_tail_fraction",
                    0.125,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_tail_min_count=int(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_tail_min_count",
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_tail_mass_cap_scale=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "tail_mass_cap_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_sharpness_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "sharpness_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_sharpness_ess_scale=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "sharpness_ess_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_sharpness_top_mass_scale=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "sharpness_top_mass_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_sharpness_ess_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "sharpness_ess_loss_weight"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_sharpness_top_mass_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "sharpness_top_mass_loss_weight"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_student_ess_floor=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "student_ess_floor"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_student_ess_floor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "student_ess_floor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_student_top_mass_cap=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "student_top_mass_cap"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_student_top_mass_cap_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "student_top_mass_cap_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ccc_floor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "ccc_floor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ccc_floors=dict(
                payload.get(
                    "bioemu_cs_reweighting_teacher_generated_prior_ccc_floors",
                    {},
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ccc_family_weights=dict(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "ccc_family_weights"
                    ),
                    {},
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_macro_ccc_floor=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "macro_ccc_floor"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_macro_ccc_floor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "macro_ccc_floor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_std_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_std_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_shape_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_shape_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_rank_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_rank_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_corr_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_corr_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_rank_margin=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_rank_margin"
                    ),
                    0.15,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_rank_top_fraction=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_rank_top_fraction"
                    ),
                    0.125,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_huber_beta=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_huber_beta"
                    ),
                    0.5,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_energy_std_huber_beta=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "energy_std_huber_beta"
                    ),
                    0.25,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "logit_ratio_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_huber_beta=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "logit_ratio_huber_beta"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_target_clip=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "logit_ratio_target_clip"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_logit_ratio_weight_power=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "logit_ratio_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_margin_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_margin_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_margin=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_margin"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_fraction=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_fraction"
                    ),
                    0.125,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_min_count=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_min_count"
                    ),
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_fraction=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_negative_fraction"
                    ),
                    0.25,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_min_count=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_negative_min_count"
                    ),
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_negative_mode=str(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_negative_mode"
                    ),
                    "teacher_tail",
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top_rank_weight_power=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top_rank_weight_power"
                    ),
                    0.5,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_margin_loss_weight=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_margin_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_margin=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_margin"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_fraction=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_fraction"
                    ),
                    0.25,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_min_count=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_min_count"
                    ),
                    1,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_min_teacher_mass=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_min_teacher_mass"
                    ),
                    0.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_top1_hard_negative_teacher_mass_power=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "top1_hard_negative_teacher_mass_power"
                    ),
                    0.5,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ramp_examples=int(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "ramp_examples"
                    ),
                    0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ramp_start_scale=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "ramp_start_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_ramp_power=float(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "ramp_power"
                    ),
                    1.0,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_log_prob_source=str(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "log_prob_source"
                    ),
                    "raw",
                )
            ),
            bioemu_cs_reweighting_teacher_require_generated_prior_grad=bool(
                payload.get(
                    "bioemu_cs_reweighting_teacher_require_generated_prior_grad",
                    False,
                )
            ),
            bioemu_cs_reweighting_teacher_generated_prior_online_proxy_teacher_enabled=bool(
                payload.get(
                    (
                        "bioemu_cs_reweighting_teacher_generated_prior_"
                        "online_proxy_teacher_enabled"
                    ),
                    False,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_weights_path=payload.get(
                "bioemu_ucbshift_cnnls_teacher_weights_path"
            ),
            bioemu_ucbshift_cnnls_teacher_predictions_path=(
                payload.get("bioemu_ucbshift_cnnls_teacher_predictions_path")
                or payload.get("bioemu_cs_reweighting_teacher_predictions_path")
            ),
            bioemu_ucbshift_cnnls_teacher_alignment_mode=str(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_alignment_mode",
                    "sample_index",
                )
            ),
            bioemu_ucbshift_cnnls_teacher_min_weight_count=int(
                payload.get("bioemu_ucbshift_cnnls_teacher_min_weight_count", 1)
            ),
            bioemu_ucbshift_cnnls_teacher_kl_weight=float(
                payload.get("bioemu_ucbshift_cnnls_teacher_kl_weight", 0.0)
            ),
            bioemu_ucbshift_cnnls_teacher_prior_kl_weight=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_prior_kl_weight",
                    payload.get(
                        "bioemu_cs_reweighting_teacher_prior_kl_weight",
                        0.0,
                    ),
                )
            ),
            bioemu_ucbshift_cnnls_teacher_mean_loss_weight=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_mean_loss_weight",
                    payload.get(
                        "bioemu_cs_reweighting_teacher_mean_loss_weight",
                        0.0,
                    ),
                )
            ),
            bioemu_ucbshift_cnnls_teacher_mean_loss_mode=str(
                payload.get("bioemu_ucbshift_cnnls_teacher_mean_loss_mode", "huber")
            ),
            bioemu_ucbshift_cnnls_teacher_mean_family_names=list(
                payload.get("bioemu_ucbshift_cnnls_teacher_mean_family_names", [])
            ),
            bioemu_ucbshift_cnnls_teacher_mean_start_epoch=int(
                payload.get("bioemu_ucbshift_cnnls_teacher_mean_start_epoch", 1)
            ),
            bioemu_ucbshift_cnnls_teacher_mean_ramp_epochs=int(
                payload.get("bioemu_ucbshift_cnnls_teacher_mean_ramp_epochs", 1)
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_weighting=bool(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_weighting",
                    False,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_min_weight=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_min_weight",
                    0.20,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_ess_floor=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_ess_floor",
                    20.0,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_entropy_floor=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_entropy_floor",
                    1.0,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_non_bioemu_source_threshold=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_non_bioemu_source_threshold",
                    0.90,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_support_gap_floor=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_support_gap_floor",
                    0.25,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_reliability_adaptive_floor_unmet_weight=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_reliability_adaptive_floor_unmet_weight",
                    0.70,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_guard_path=(
                payload.get("bioemu_ucbshift_cnnls_teacher_guard_path")
                or payload.get("bioemu_cs_reweighting_teacher_guard_path")
            ),
            bioemu_detached_support_guard_path=payload.get(
                "bioemu_detached_support_guard_path"
            ),
            bioemu_detached_support_guard_levels=list(
                payload.get(
                    "bioemu_detached_support_guard_levels",
                    ["critical_support_gap", "geometry_spread_guard"],
                )
            ),
            bioemu_detached_support_guard_min_priority=float(
                payload.get("bioemu_detached_support_guard_min_priority", 0.0)
            ),
            bioemu_ucbshift_cnnls_teacher_guard_families=list(
                payload.get("bioemu_ucbshift_cnnls_teacher_guard_families", ["N"])
            ),
            bioemu_ucbshift_cnnls_teacher_guard_min_outlier_fraction=float(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_guard_min_outlier_fraction",
                    0.5,
                )
            ),
            bioemu_ucbshift_cnnls_teacher_guard_exclude_kl=bool(
                payload.get("bioemu_ucbshift_cnnls_teacher_guard_exclude_kl", True)
            ),
            bioemu_ucbshift_cnnls_teacher_guard_exclude_mean_loss=bool(
                payload.get(
                    "bioemu_ucbshift_cnnls_teacher_guard_exclude_mean_loss",
                    True,
                )
            ),
            bioemu_nmr_sampling_method=str(
                payload.get("bioemu_nmr_sampling_method", "sobol_antithetic")
            ),
            bioemu_nmr_enable_stratified_chart_sampling=bool(
                payload.get("bioemu_nmr_enable_stratified_chart_sampling", True)
            ),
            bioemu_nmr_min_rare_chart_samples=int(
                payload.get("bioemu_nmr_min_rare_chart_samples", 1)
            ),
            bioemu_nmr_posterior_weight_temperature=float(
                payload.get("bioemu_nmr_posterior_weight_temperature", 1.0)
            ),
            bioemu_support_mode_occupancy_guard_enabled=bool(
                payload.get("bioemu_support_mode_occupancy_guard_enabled", False)
            ),
            bioemu_support_mode_occupancy_guard_entity=str(
                payload.get("bioemu_support_mode_occupancy_guard_entity", "")
            ),
            bioemu_support_mode_occupancy_guard_mode=str(
                payload.get("bioemu_support_mode_occupancy_guard_mode", "")
            ),
            bioemu_support_mode_occupancy_guard_cap=float(
                payload.get("bioemu_support_mode_occupancy_guard_cap", 1.0)
            ),
            bioemu_support_mode_occupancy_guard_mean_weight=float(
                payload.get("bioemu_support_mode_occupancy_guard_mean_weight", 0.0)
            ),
            bioemu_support_mode_occupancy_guard_label_path=(
                None
                if payload.get("bioemu_support_mode_occupancy_guard_label_path")
                in {None, ""}
                else str(payload.get("bioemu_support_mode_occupancy_guard_label_path"))
            ),
            bioemu_inference_readout_overlay_enabled=bool(
                payload.get("bioemu_inference_readout_overlay_enabled", False)
            ),
            bioemu_inference_readout_overlay_path=(
                None
                if payload.get("bioemu_inference_readout_overlay_path") in {None, ""}
                else str(payload.get("bioemu_inference_readout_overlay_path"))
            ),
            bioemu_inference_readout_overlay_value_column=str(
                payload.get(
                    "bioemu_inference_readout_overlay_value_column",
                    "inference_overlay_predicted_value",
                )
            ),
            bioemu_inference_readout_overlay_delta_column=str(
                payload.get(
                    "bioemu_inference_readout_overlay_delta_column",
                    "inference_overlay_delta",
                )
            ),
            bioemu_inference_readout_overlay_source_column=str(
                payload.get(
                    "bioemu_inference_readout_overlay_source_column",
                    "inference_overlay_source",
                )
            ),
            bioemu_inference_readout_overlay_output_prefix=str(
                payload.get(
                    "bioemu_inference_readout_overlay_output_prefix",
                    "runtime_overlay",
                )
            ),
            bioemu_inference_readout_overlay_active_families=list(
                payload.get("bioemu_inference_readout_overlay_active_families", ["HN", "N"])
            ),
            bioemu_inference_readout_overlay_require_exact_rows=bool(
                payload.get("bioemu_inference_readout_overlay_require_exact_rows", False)
            ),
            bioemu_bridge_audit_signal_enabled=bool(
                payload.get("bioemu_bridge_audit_signal_enabled", False)
            ),
            bioemu_bridge_uncertainty_guard_only=bool(
                payload.get("bioemu_bridge_uncertainty_guard_only", False)
            ),
            bioemu_bridge_audit_signal_entity=str(
                payload.get(
                    "bioemu_bridge_audit_signal_entity",
                    payload.get("entity", ""),
                )
            ),
            bioemu_bridge_audit_signal_mode=str(
                payload.get("bioemu_bridge_audit_signal_mode", payload.get("mode", ""))
            ),
            bioemu_bridge_audit_signal_family=str(
                payload.get(
                    "bioemu_bridge_audit_signal_family",
                    payload.get("family", ""),
                )
            ),
            bioemu_bridge_audit_signal_sign=str(
                payload.get("bioemu_bridge_audit_signal_sign", payload.get("sign", ""))
            ),
            bioemu_bridge_audit_signal_weight=float(
                payload.get("bioemu_bridge_audit_signal_weight", payload.get("weight", 0.0))
            ),
            bioemu_bridge_audit_signal_mean_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_mean_weight",
                    payload.get("mean_weight", 0.0),
                )
            ),
            bioemu_bridge_audit_signal_cap_change=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cap_change",
                    payload.get("cap_change", 0.0),
                )
            ),
            bioemu_bridge_audit_signal_residues=list(
                payload.get("bioemu_bridge_audit_signal_residues", [])
            ),
            bioemu_bridge_audit_signal_residue_aware_enabled=bool(
                payload.get("bioemu_bridge_audit_signal_residue_aware_enabled", False)
            ),
            bioemu_bridge_audit_signal_cs_sidecar_path=(
                None
                if payload.get("bioemu_bridge_audit_signal_cs_sidecar_path") in {None, ""}
                else str(payload.get("bioemu_bridge_audit_signal_cs_sidecar_path"))
            ),
            bioemu_bridge_audit_signal_cs_energy_enabled=bool(
                payload.get("bioemu_bridge_audit_signal_cs_energy_enabled", False)
            ),
            bioemu_bridge_audit_signal_cs_energy_weight=float(
                payload.get("bioemu_bridge_audit_signal_cs_energy_weight", 0.0)
            ),
            bioemu_bridge_audit_signal_cs_energy_cap=float(
                payload.get("bioemu_bridge_audit_signal_cs_energy_cap", 0.02)
            ),
            bioemu_bridge_audit_signal_cs_energy_mode_only=bool(
                payload.get("bioemu_bridge_audit_signal_cs_energy_mode_only", False)
            ),
            bioemu_bridge_audit_signal_cs_energy_profile_kind=str(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_profile_kind",
                    "squared_error",
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_family_weights=dict(
                payload.get("bioemu_bridge_audit_signal_cs_energy_family_weights", {})
            ),
            bioemu_bridge_audit_signal_cs_energy_tail_fraction=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_tail_fraction",
                    0.25,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_signed_consensus_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_signed_consensus_weight",
                    0.35,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_learned_scale_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_learned_scale_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_learned_scale_max=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_learned_scale_max",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_sample_head_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_sample_head_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_sample_head_max_abs=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_sample_head_max_abs",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_sample_head_matrix_features_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_sample_head_matrix_features_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_pairwise_rank_loss_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_pairwise_rank_loss_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_pairwise_rank_margin=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_pairwise_rank_margin",
                    0.25,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_pairwise_rank_temperature=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_pairwise_rank_temperature",
                    0.1,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_pairwise_rank_topk=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_pairwise_rank_topk",
                    64,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_pairwise_rank_min_gap=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_pairwise_rank_min_gap",
                    0.02,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_aux_family_names=list(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_aux_family_names",
                    [],
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_aux_pairwise_rank_loss_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_aux_pairwise_rank_loss_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_aux_logit_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_aux_logit_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_aux_logit_cap=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_aux_logit_cap",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_oracle_beta=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_oracle_beta",
                    8.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_oracle_kl_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_oracle_kl_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_oracle_temperature=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_oracle_temperature",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_energy_oracle_min_samples=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_energy_oracle_min_samples",
                    16,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_readout_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_readout_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_readout_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_readout_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_readout_max_abs_delta=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_readout_max_abs_delta",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_loss_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_loss_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_trainable=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_trainable",
                    True,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_family_names=list(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_family_names",
                    [],
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_family_loss_weights={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_measure_distill_family_loss_weights",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_measure_distill_huber_delta=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_huber_delta",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_min_rows=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_min_rows",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_start_epoch=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_start_epoch",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_ramp_epochs=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_ramp_epochs",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_ess_floor=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_ess_floor",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_entropy_floor=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_entropy_floor",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_min_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_support_confidence_min_scale",
                    0.75,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_family_confidence_scales={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_measure_distill_family_confidence_scales",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_measure_distill_mechanism_mask_focus_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_measure_distill_mechanism_mask_focus_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_measure_distill_mechanism_match_scales={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_measure_distill_mechanism_match_scales",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_measure_distill_mechanism_nonmatch_scales={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_measure_distill_mechanism_nonmatch_scales",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_loss_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_loss_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_names=list(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_names",
                    [],
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_loss_weights={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_sample_matrix_distill_family_loss_weights",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_weight_by_family={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_bridge_audit_signal_cs_sample_matrix_distill_weight_by_family",
                        {},
                    )
                ).items()
            },
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_min_rows=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_min_rows",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_start_epoch=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_start_epoch",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_ramp_epochs=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_ramp_epochs",
                    1,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_gate_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_gate_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_consensus=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_consensus",
                    0.62,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_low_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_low_scale",
                    0.80,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_high_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_high_scale",
                    1.25,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_valid_samples=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_hn_signed_min_valid_samples",
                    16,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_focus_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_focus_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_min_proxy=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_min_proxy",
                    0.35,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_low_proxy_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_low_proxy_scale",
                    0.35,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_terminal_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_plane_terminal_scale",
                    0.50,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_signed_gate_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_signed_gate_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_center=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_center",
                    2.0e-4,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_softness=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_pair_norm_softness",
                    1.0e-4,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sign_consensus_min=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sign_consensus_min",
                    0.60,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_low_consensus_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_low_consensus_scale",
                    0.45,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_plane_min_proxy=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_plane_min_proxy",
                    0.30,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_gate_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_gate_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_min_scale=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_min_scale",
                    0.55,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_peptide_plane_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_peptide_plane_weight",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_weight",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_min_proxy=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_carbonyl_hbond_min_proxy",
                    0.25,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_use_neighborhood_sidecar=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_edge_class_use_neighborhood_sidecar",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_weight",
                    0.70,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_weight",
                    0.30,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_min_score=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_min_score",
                    0.05,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_gate_enabled=bool(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_gate_enabled",
                    False,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_min_score=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_hbond_min_score",
                    0.08,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_min_score=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_peptide_min_score",
                    0.06,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_min_score=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_min_score",
                    0.12,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_contact_weight",
                    0.20,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_loss_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_loss_weight",
                    0.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_hbond_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_hbond_weight",
                    1.0,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_peptide_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_peptide_weight",
                    0.8,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_contact_weight=float(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_contact_weight",
                    0.25,
                )
            ),
            bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_min_rows=int(
                payload.get(
                    "bioemu_bridge_audit_signal_cs_sample_matrix_distill_cprime_x2d_sidecar_channel_matrix_min_rows",
                    1,
                )
            ),
            enable_bioemu_neighborhood_aware_nmr=bool(
                payload.get("enable_bioemu_neighborhood_aware_nmr", False)
            ),
            bioemu_neighborhood_edge_offsets=list(
                payload.get("bioemu_neighborhood_edge_offsets", [-2, -1, 0, 1, 2])
            ),
            bioemu_neighborhood_family_names=list(
                payload.get("bioemu_neighborhood_family_names", ["HN", "C'"])
            ),
            bioemu_neighborhood_hidden_delta_max_abs=float(
                payload.get("bioemu_neighborhood_hidden_delta_max_abs", 0.04)
            ),
            bioemu_neighborhood_energy_weight=float(
                payload.get("bioemu_neighborhood_energy_weight", 0.0)
            ),
            bioemu_neighborhood_train_with_local_adapters=bool(
                payload.get("bioemu_neighborhood_train_with_local_adapters", True)
            ),
            bioemu_neighborhood_use_row_physics_proxy=bool(
                payload.get("bioemu_neighborhood_use_row_physics_proxy", True)
            ),
            bioemu_neighborhood_apply_after_score_aware=bool(
                payload.get("bioemu_neighborhood_apply_after_score_aware", True)
            ),
            bioemu_neighborhood_energy_after_score_aware=bool(
                payload.get("bioemu_neighborhood_energy_after_score_aware", False)
            ),
            bioemu_neighborhood_sidecar_prior_energy_scale=float(
                payload.get("bioemu_neighborhood_sidecar_prior_energy_scale", 0.0)
            ),
            bioemu_neighborhood_sidecar_prior_centered=bool(
                payload.get("bioemu_neighborhood_sidecar_prior_centered", True)
            ),
            bioemu_neighborhood_sidecar_path=(
                None
                if payload.get("bioemu_neighborhood_sidecar_path") in {None, ""}
                else str(payload.get("bioemu_neighborhood_sidecar_path"))
            ),
            enable_bioemu_family_split_capacity_readout=bool(
                payload.get("enable_bioemu_family_split_capacity_readout", False)
            ),
            bioemu_family_split_capacity_readout_family_names=list(
                payload.get(
                    "bioemu_family_split_capacity_readout_family_names",
                    ["C'", "N", "CA", "CB"],
                )
            ),
            bioemu_family_split_capacity_readout_max_abs=float(
                payload.get("bioemu_family_split_capacity_readout_max_abs", 4.0)
            ),
            bioemu_family_split_capacity_readout_width_multiplier=float(
                payload.get(
                    "bioemu_family_split_capacity_readout_width_multiplier",
                    2.0,
                )
            ),
            bioemu_family_split_capacity_readout_bottleneck_multiplier=float(
                payload.get(
                    "bioemu_family_split_capacity_readout_bottleneck_multiplier",
                    1.0,
                )
            ),
            bioemu_family_split_capacity_readout_depth=int(
                payload.get("bioemu_family_split_capacity_readout_depth", 2)
            ),
            bioemu_family_split_capacity_readout_gate_width_multiplier=float(
                payload.get(
                    "bioemu_family_split_capacity_readout_gate_width_multiplier",
                    1.0,
                )
            ),
            bioemu_family_split_capacity_readout_gate_depth=int(
                payload.get("bioemu_family_split_capacity_readout_gate_depth", 1)
            ),
            bioemu_family_split_capacity_readout_use_diagnostics=bool(
                payload.get(
                    "bioemu_family_split_capacity_readout_use_diagnostics",
                    False,
                )
            ),
            bioemu_family_split_capacity_readout_diagnostic_scale=float(
                payload.get(
                    "bioemu_family_split_capacity_readout_diagnostic_scale",
                    1.0,
                )
            ),
            bioemu_train_only_family_split_capacity_readout=bool(
                payload.get("bioemu_train_only_family_split_capacity_readout", False)
            ),
            bioemu_capture_x2d_pair_latent=bool(
                payload.get("bioemu_capture_x2d_pair_latent", False)
            ),
            bioemu_x2d_pair_latent_weight=float(
                payload.get("bioemu_x2d_pair_latent_weight", 0.0)
            ),
            bioemu_x2d_pair_latent_family_names=list(
                payload.get("bioemu_x2d_pair_latent_family_names", [])
            ),
            bioemu_x2d_pair_local_only_family_names=list(
                payload.get("bioemu_x2d_pair_local_only_family_names", [])
            ),
            bioemu_x2d_pair_local_only_weight_scale=float(
                payload.get("bioemu_x2d_pair_local_only_weight_scale", 1.0)
            ),
            bioemu_x2d_pair_global_scale_by_family=dict(
                payload.get("bioemu_x2d_pair_global_scale_by_family", {})
            ),
            bioemu_x2d_pair_missing_fallback=str(
                payload.get("bioemu_x2d_pair_missing_fallback", "none")
            ),
            bioemu_x2d_pair_summary_offsets=list(
                payload.get("bioemu_x2d_pair_summary_offsets", [-2, -1, 0, 1, 2])
            ),
            bioemu_x2d_pair_global_topk=int(
                payload.get("bioemu_x2d_pair_global_topk", 0)
            ),
            bioemu_x2d_pair_global_weight=float(
                payload.get("bioemu_x2d_pair_global_weight", 1.0)
            ),
            bioemu_x2d_pair_global_min_sequence_separation=int(
                payload.get("bioemu_x2d_pair_global_min_sequence_separation", 0)
            ),
            bioemu_x2d_pair_global_max_sequence_separation=int(
                payload.get("bioemu_x2d_pair_global_max_sequence_separation", 0)
            ),
            bioemu_x2d_pair_global_extra_bands=[
                dict(item)
                for item in list(
                    payload.get("bioemu_x2d_pair_global_extra_bands", [])
                )
            ],
            bioemu_x2d_pair_global_band_name=str(
                payload.get("bioemu_x2d_pair_global_band_name", "global")
            ),
            bioemu_x2d_pair_split_global_bands=bool(
                payload.get("bioemu_x2d_pair_split_global_bands", False)
            ),
            bioemu_x2d_pair_global_band_scale_by_family={
                str(family): {
                    str(band): float(scale)
                    for band, scale in dict(scales).items()
                }
                for family, scales in dict(
                    payload.get("bioemu_x2d_pair_global_band_scale_by_family", {})
                ).items()
            },
            bioemu_x2d_pair_global_band_reliability_feature_weights_by_family={
                str(family): {
                    str(band): {
                        str(feature): float(weight)
                        for feature, weight in dict(weights).items()
                    }
                    for band, weights in dict(bands).items()
                }
                for family, bands in dict(
                    payload.get(
                        "bioemu_x2d_pair_global_band_reliability_feature_weights_by_family",
                        {},
                    )
                ).items()
            },
            bioemu_x2d_pair_global_band_reliability_min_scale_by_family={
                str(family): {
                    str(band): float(scale)
                    for band, scale in dict(scales).items()
                }
                for family, scales in dict(
                    payload.get(
                        "bioemu_x2d_pair_global_band_reliability_min_scale_by_family",
                        {},
                    )
                ).items()
            },
            bioemu_x2d_pair_reliability_gate_enabled=bool(
                payload.get("bioemu_x2d_pair_reliability_gate_enabled", False)
            ),
            bioemu_x2d_pair_reliability_min_scale_by_family={
                str(key): float(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x2d_pair_reliability_min_scale_by_family",
                        {},
                    )
                ).items()
            },
            bioemu_x2d_pair_reliability_feature_weights_by_family={
                str(family): {
                    str(feature): float(weight)
                    for feature, weight in dict(weights).items()
                }
                for family, weights in dict(
                    payload.get(
                        "bioemu_x2d_pair_reliability_feature_weights_by_family",
                        {},
                    )
                ).items()
            },
            bioemu_x2d_pair_reliability_apply_to_global_only=bool(
                payload.get(
                    "bioemu_x2d_pair_reliability_apply_to_global_only",
                    False,
                )
            ),
            bioemu_x2d_pair_context_norm_cap=float(
                payload.get("bioemu_x2d_pair_context_norm_cap", 0.0)
            ),
            bioemu_x2d_pair_context_norm_cap_by_family=dict(
                payload.get("bioemu_x2d_pair_context_norm_cap_by_family", {})
            ),
            bioemu_x2d_pair_update_full_context=bool(
                payload.get("bioemu_x2d_pair_update_full_context", True)
            ),
            bioemu_x2d_pair_projection_init_std=float(
                payload.get("bioemu_x2d_pair_projection_init_std", 0.0)
            ),
            bioemu_neighborhood_edge_classes=list(
                payload.get(
                    "bioemu_neighborhood_edge_classes",
                    [
                        "sequence",
                        "peptide_plane",
                        "spatial_contact",
                        "hbond",
                        "ring",
                        "electrostatic",
                    ],
                )
            ),
            bioemu_neighborhood_topk_by_class={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_neighborhood_topk_by_class",
                        {
                            "sequence": 5,
                            "peptide_plane": 3,
                            "spatial_contact": 16,
                            "hbond": 8,
                            "ring": 8,
                            "electrostatic": 8,
                        },
                    )
                ).items()
            },
            bioemu_nmr_decode_structures_during_training=bool(
                payload.get("bioemu_nmr_decode_structures_during_training", False)
            ),
            bioemu_nmr_enable_control_variate=bool(
                payload.get("bioemu_nmr_enable_control_variate", True)
            ),
            enable_bioemu_x0_posterior_ensemble=bool(
                payload.get("enable_bioemu_x0_posterior_ensemble", False)
            ),
            bioemu_x0_posterior_stage=str(
                payload.get("bioemu_x0_posterior_stage", "decoder_pretrain")
            ),
            bioemu_x0_posterior_support_k=int(
                payload.get("bioemu_x0_posterior_support_k", 512)
            ),
            bioemu_x0_target_family_ccc=float(
                payload.get("bioemu_x0_target_family_ccc", 0.95)
            ),
            bioemu_x0_stage_b_supervised_fraction=float(
                payload.get("bioemu_x0_stage_b_supervised_fraction", 0.25)
            ),
            bioemu_x0_stage_a_train_capacity_adapter_only=bool(
                payload.get("bioemu_x0_stage_a_train_capacity_adapter_only", False)
            ),
            bioemu_x0_stage_c_train_posterior_heads=bool(
                payload.get("bioemu_x0_stage_c_train_posterior_heads", True)
            ),
            bioemu_x0_stage_c_train_evidence_heads=bool(
                payload.get("bioemu_x0_stage_c_train_evidence_heads", True)
            ),
            bioemu_x0_stage_c_train_decoder_heads=bool(
                payload.get("bioemu_x0_stage_c_train_decoder_heads", True)
            ),
            bioemu_x0_stage_c_train_hn_local_decoder_only=bool(
                payload.get("bioemu_x0_stage_c_train_hn_local_decoder_only", False)
            ),
            bioemu_x0_stage_c_train_hn_local_capacity_adapter=bool(
                payload.get("bioemu_x0_stage_c_train_hn_local_capacity_adapter", False)
            ),
            bioemu_x0_stage_c_train_support_basis_only=bool(
                payload.get("bioemu_x0_stage_c_train_support_basis_only", False)
            ),
            bioemu_x0_stage_c_train_support_basis_and_family_affine=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_support_basis_and_family_affine",
                    False,
                )
            ),
            bioemu_x0_stage_c_train_hn_cprime_n_support_only=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_hn_cprime_n_support_only",
                    False,
                )
            ),
            bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_only=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_only",
                    False,
                )
            ),
            bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_capacity_adapters=bool(
                payload.get(
                    (
                        "bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_"
                        "capacity_adapters"
                    ),
                    True,
                )
            ),
            bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_hn_cprime_basis=bool(
                payload.get(
                    (
                        "bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_"
                        "hn_cprime_basis"
                    ),
                    False,
                )
            ),
            bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_posterior_energy_extra=bool(
                payload.get(
                    (
                        "bioemu_x0_stage_c_train_hn_ca_cb_preserve_cprime_"
                        "posterior_energy_extra"
                    ),
                    False,
                )
            ),
            bioemu_x0_stage_c_train_cprime_isolated_basis_only=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_cprime_isolated_basis_only",
                    False,
                )
            ),
            bioemu_x0_stage_c_train_capacity_adapter_only=bool(
                payload.get("bioemu_x0_stage_c_train_capacity_adapter_only", False)
            ),
            bioemu_x0_stage_c_train_capacity_adapter_gate_only=bool(
                payload.get("bioemu_x0_stage_c_train_capacity_adapter_gate_only", False)
            ),
            bioemu_x0_stage_c_train_capacity_adapter_extra_only=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_capacity_adapter_extra_only",
                    False,
                )
            ),
            bioemu_x0_stage_c_train_posterior_energy_extra_only=bool(
                payload.get(
                    "bioemu_x0_stage_c_train_posterior_energy_extra_only",
                    False,
                )
            ),
            bioemu_x0_stage_b_train_prior_logit_adapter_only=bool(
                payload.get("bioemu_x0_stage_b_train_prior_logit_adapter_only", False)
            ),
            bioemu_x0_decoder_hidden_dim=int(
                payload.get("bioemu_x0_decoder_hidden_dim", 256)
            ),
            bioemu_x0_decoder_dropout=float(
                payload.get("bioemu_x0_decoder_dropout", 0.0)
            ),
            bioemu_x0_decoder_teacher_loss_weight=float(
                payload.get("bioemu_x0_decoder_teacher_loss_weight", 1.0)
            ),
            bioemu_x0_decoder_teacher_family_macro_weight=float(
                payload.get("bioemu_x0_decoder_teacher_family_macro_weight", 1.0)
            ),
            bioemu_x0_decoder_teacher_family_weights=dict(
                payload.get("bioemu_x0_decoder_teacher_family_weights", {})
            ),
            bioemu_x0_decoder_teacher_support_variance_weight=float(
                payload.get("bioemu_x0_decoder_teacher_support_variance_weight", 0.0)
            ),
            bioemu_x0_decoder_teacher_support_variance_family_weights=dict(
                payload.get("bioemu_x0_decoder_teacher_support_variance_family_weights", {})
            ),
            bioemu_x0_decoder_mechanism_support_basis_count=int(
                payload.get("bioemu_x0_decoder_mechanism_support_basis_count", 0)
            ),
            bioemu_x0_decoder_mechanism_support_basis_cap=float(
                payload.get("bioemu_x0_decoder_mechanism_support_basis_cap", 0.0)
            ),
            bioemu_x0_decoder_hn_signed_support_basis_count=int(
                payload.get("bioemu_x0_decoder_hn_signed_support_basis_count", 0)
            ),
            bioemu_x0_decoder_hn_signed_support_basis_cap=float(
                payload.get("bioemu_x0_decoder_hn_signed_support_basis_cap", 0.0)
            ),
            bioemu_x0_decoder_hn_direct_support_spread_scale_ppm=float(
                payload.get("bioemu_x0_decoder_hn_direct_support_spread_scale_ppm", 0.0)
            ),
            bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_cap_ppm=float(
                payload.get(
                    "bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_cap_ppm",
                    0.0,
                )
            ),
            bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_init_ppm=float(
                payload.get(
                    "bioemu_x0_decoder_hn_direct_support_spread_trainable_scale_init_ppm",
                    0.0,
                )
            ),
            bioemu_x0_decoder_family_direct_support_spread_scale_ppm_by_family=dict(
                payload.get(
                    (
                        "bioemu_x0_decoder_family_direct_support_spread_"
                        "scale_ppm_by_family"
                    ),
                    {},
                )
                or {}
            ),
            bioemu_x0_decoder_family_direct_support_spread_trainable_scale_cap_ppm_by_family=dict(
                payload.get(
                    (
                        "bioemu_x0_decoder_family_direct_support_spread_"
                        "trainable_scale_cap_ppm_by_family"
                    ),
                    {},
                )
                or {}
            ),
            bioemu_x0_decoder_family_direct_support_spread_trainable_scale_init_ppm_by_family=dict(
                payload.get(
                    (
                        "bioemu_x0_decoder_family_direct_support_spread_"
                        "trainable_scale_init_ppm_by_family"
                    ),
                    {},
                )
                or {}
            ),
            bioemu_x0_decoder_cprime_isolated_support_basis_count=int(
                payload.get(
                    "bioemu_x0_decoder_cprime_isolated_support_basis_count",
                    0,
                )
            ),
            bioemu_x0_decoder_cprime_isolated_support_basis_cap=float(
                payload.get(
                    "bioemu_x0_decoder_cprime_isolated_support_basis_cap",
                    0.0,
                )
            ),
            bioemu_x0_decoder_n_ca_cb_support_deviation_enabled=bool(
                payload.get(
                    "bioemu_x0_decoder_n_ca_cb_support_deviation_enabled",
                    True,
                )
            ),
            bioemu_x0_decoder_emit_support_basis_diagnostics=bool(
                payload.get(
                    "bioemu_x0_decoder_emit_support_basis_diagnostics",
                    True,
                )
            ),
            bioemu_x0_support_basis_lr_multiplier=float(
                payload.get("bioemu_x0_support_basis_lr_multiplier", 1.0)
            ),
            bioemu_x0_mechanism_decoder_lr_multiplier=float(
                payload.get("bioemu_x0_mechanism_decoder_lr_multiplier", 1.0)
            ),
            bioemu_x0_decoder_phi_trust_region_loss_weight=float(
                payload.get("bioemu_x0_decoder_phi_trust_region_loss_weight", 0.0)
            ),
            bioemu_x0_decoder_phi_trust_region_family_weights=dict(
                payload.get("bioemu_x0_decoder_phi_trust_region_family_weights", {})
            ),
            bioemu_x0_decoder_phi_trust_region_huber_beta=float(
                payload.get("bioemu_x0_decoder_phi_trust_region_huber_beta", 0.05)
            ),
            bioemu_x0_support_basis_delta_ceiling_loss_weight=float(
                payload.get(
                    "bioemu_x0_support_basis_delta_ceiling_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_support_basis_delta_ceiling_ppm=float(
                payload.get("bioemu_x0_support_basis_delta_ceiling_ppm", 0.0)
            ),
            bioemu_x0_support_basis_delta_ceiling_huber_beta=float(
                payload.get(
                    "bioemu_x0_support_basis_delta_ceiling_huber_beta",
                    0.05,
                )
            ),
            bioemu_x0_support_basis_delta_ceiling_path_weights=dict(
                payload.get(
                    "bioemu_x0_support_basis_delta_ceiling_path_weights",
                    {},
                )
            ),
            bioemu_x0_decoder_use_checkpoint=bool(
                payload.get("bioemu_x0_decoder_use_checkpoint", False)
            ),
            bioemu_x0_decoder_support_head_chunk_size=int(
                payload.get("bioemu_x0_decoder_support_head_chunk_size", 0)
            ),
            bioemu_x0_compact_edge_encoder_sample_chunk_size=int(
                payload.get("bioemu_x0_compact_edge_encoder_sample_chunk_size", 0)
            ),
            bioemu_x0_decoder_capacity_adapter_hidden_dim=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_hidden_dim", 0)
            ),
            bioemu_x0_decoder_capacity_adapter_depth=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_depth", 3)
            ),
            bioemu_x0_decoder_capacity_adapter_cap_ppm=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_cap_ppm", 0.0)
            ),
            bioemu_x0_decoder_capacity_adapter_scale=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_scale", 1.0)
            ),
            bioemu_x0_decoder_capacity_adapter_trainable_scale_cap=float(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_trainable_scale_cap",
                    0.0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_trainable_scale_init=float(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_trainable_scale_init",
                    0.0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_checkpoint_path=(
                str(payload["bioemu_x0_decoder_capacity_adapter_checkpoint_path"])
                if payload.get("bioemu_x0_decoder_capacity_adapter_checkpoint_path")
                is not None
                else None
            ),
            bioemu_x0_decoder_capacity_adapter_extra_hidden_dim=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_extra_hidden_dim", 0)
            ),
            bioemu_x0_decoder_capacity_adapter_extra_depth=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_extra_depth", 3)
            ),
            bioemu_x0_decoder_capacity_adapter_extra_cap_ppm=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_extra_cap_ppm", 0.0)
            ),
            bioemu_x0_decoder_capacity_adapter_extra_scale=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_extra_scale", 1.0)
            ),
            bioemu_x0_decoder_capacity_adapter_extra_expert_count=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_expert_count",
                    1,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_factorized_rank=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_factorized_rank",
                    0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_effective_hidden_dim=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_effective_hidden_dim",
                    0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_effective_factorized_rank=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_effective_factorized_rank",
                    0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_max_factorized_params=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_max_factorized_params",
                    450_000_000,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_output_init_std=float(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_output_init_std",
                    0.0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_family_scales=dict(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_family_scales",
                    {},
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_family_gated=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_family_gated",
                    False,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_family_specific=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_family_specific",
                    False,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_extra_family_headed=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_extra_family_headed",
                    False,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_hidden_dim=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_aux_hidden_dim", 0)
            ),
            bioemu_x0_decoder_capacity_adapter_aux_depth=int(
                payload.get("bioemu_x0_decoder_capacity_adapter_aux_depth", 2)
            ),
            bioemu_x0_decoder_capacity_adapter_aux_cap_ppm=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_aux_cap_ppm", 0.0)
            ),
            bioemu_x0_decoder_capacity_adapter_aux_scale=float(
                payload.get("bioemu_x0_decoder_capacity_adapter_aux_scale", 1.0)
            ),
            bioemu_x0_decoder_capacity_adapter_aux_expert_count=int(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_expert_count",
                    1,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_output_init_std=float(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_output_init_std",
                    0.0,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_family_scales=dict(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_family_scales",
                    {},
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_family_gated=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_family_gated",
                    False,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_family_specific=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_family_specific",
                    False,
                )
            ),
            bioemu_x0_decoder_capacity_adapter_aux_family_headed=bool(
                payload.get(
                    "bioemu_x0_decoder_capacity_adapter_aux_family_headed",
                    False,
                )
            ),
            bioemu_x0_decoder_family_affine_scale_cap=float(
                payload.get("bioemu_x0_decoder_family_affine_scale_cap", 0.0)
            ),
            bioemu_x0_decoder_family_affine_shift_cap_ppm=float(
                payload.get("bioemu_x0_decoder_family_affine_shift_cap_ppm", 0.0)
            ),
            bioemu_x0_decoder_family_affine_family_scales=dict(
                payload.get("bioemu_x0_decoder_family_affine_family_scales", {})
            ),
            bioemu_x0_stage_c_train_family_affine_only=bool(
                payload.get("bioemu_x0_stage_c_train_family_affine_only", False)
            ),
            bioemu_x0_family_affine_lr_multiplier=float(
                payload.get("bioemu_x0_family_affine_lr_multiplier", 1.0)
            ),
            bioemu_x0_sidecar_family_affine_calibration_enabled=bool(
                payload.get(
                    "bioemu_x0_sidecar_family_affine_calibration_enabled",
                    False,
                )
            ),
            bioemu_x0_sidecar_evidence_reference_offset_enabled=bool(
                payload.get(
                    "bioemu_x0_sidecar_evidence_reference_offset_enabled",
                    False,
                )
            ),
            bioemu_x0_sidecar_evidence_reference_offset_max_abs_ppm=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_reference_offset_max_abs_ppm",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_evidence_reference_offset_family_scales=dict(
                payload.get(
                    "bioemu_x0_sidecar_evidence_reference_offset_family_scales",
                    {},
                )
            ),
            bioemu_x0_sidecar_evidence_reference_offset_strength=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_reference_offset_strength",
                    1.0,
                )
            ),
            bioemu_x0_sidecar_evidence_reference_offset_min_rows=int(
                payload.get(
                    "bioemu_x0_sidecar_evidence_reference_offset_min_rows",
                    2,
                )
            ),
            bioemu_x0_capacity_adapter_lr_multiplier=float(
                payload.get("bioemu_x0_capacity_adapter_lr_multiplier", 1.0)
            ),
            bioemu_x0_capacity_adapter_extra_lr_multiplier=float(
                payload.get("bioemu_x0_capacity_adapter_extra_lr_multiplier", 1.0)
            ),
            bioemu_x0_capacity_adapter_aux_lr_multiplier=float(
                payload.get("bioemu_x0_capacity_adapter_aux_lr_multiplier", 1.0)
            ),
            bioemu_x0_posterior_energy_lr_multiplier=float(
                payload.get("bioemu_x0_posterior_energy_lr_multiplier", 1.0)
            ),
            bioemu_x0_posterior_mode_lr_multiplier=float(
                payload.get("bioemu_x0_posterior_mode_lr_multiplier", 1.0)
            ),
            bioemu_x0_evidence_head_lr_multiplier=float(
                payload.get("bioemu_x0_evidence_head_lr_multiplier", 1.0)
            ),
            bioemu_x0_prior_logit_adapter_hidden_dim=int(
                payload.get("bioemu_x0_prior_logit_adapter_hidden_dim", 0)
            ),
            bioemu_x0_prior_logit_adapter_depth=int(
                payload.get("bioemu_x0_prior_logit_adapter_depth", 2)
            ),
            bioemu_x0_prior_logit_adapter_scale=float(
                payload.get("bioemu_x0_prior_logit_adapter_scale", 1.0)
            ),
            bioemu_x0_prior_logit_adapter_max_abs=float(
                payload.get("bioemu_x0_prior_logit_adapter_max_abs", 0.0)
            ),
            bioemu_x0_prior_logit_adapter_output_init_std=float(
                payload.get("bioemu_x0_prior_logit_adapter_output_init_std", 0.0)
            ),
            bioemu_x0_prior_logit_adapter_set_context_dim=int(
                payload.get("bioemu_x0_prior_logit_adapter_set_context_dim", 0)
            ),
            bioemu_x0_prior_logit_adapter_set_context_depth=int(
                payload.get("bioemu_x0_prior_logit_adapter_set_context_depth", 1)
            ),
            bioemu_x0_prior_logit_adapter_include_base_log_prob_features=bool(
                payload.get(
                    "bioemu_x0_prior_logit_adapter_include_base_log_prob_features",
                    False,
                )
            ),
            bioemu_x0_prior_logit_adapter_lr_multiplier=float(
                payload.get("bioemu_x0_prior_logit_adapter_lr_multiplier", 1.0)
            ),
            bioemu_x0_capacity_adapter_residual_oracle_loss_weight=float(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_scale=float(
                payload.get("bioemu_x0_capacity_adapter_residual_oracle_scale", 256.0)
            ),
            bioemu_x0_capacity_adapter_residual_oracle_family_weights=dict(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_family_weights",
                    {},
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_family_scales=dict(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_family_scales",
                    {},
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_target_clip_ppm=float(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_target_clip_ppm",
                    0.0,
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_min_teacher_ess=float(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_min_teacher_ess",
                    0.0,
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_capacity_adapter_residual_oracle_start_scale=float(
                payload.get(
                    "bioemu_x0_capacity_adapter_residual_oracle_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_loss_weight=float(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_scale=float(
                payload.get("bioemu_x0_support_basis_residual_oracle_scale", 256.0)
            ),
            bioemu_x0_support_basis_residual_oracle_family_weights=dict(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_family_weights",
                    {},
                )
            ),
            bioemu_x0_support_basis_residual_oracle_family_scales=dict(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_family_scales",
                    {},
                )
            ),
            bioemu_x0_support_basis_residual_oracle_path_weights=dict(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_path_weights",
                    {},
                )
            ),
            bioemu_x0_support_basis_residual_oracle_path_family_weights=dict(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_path_family_weights",
                    {},
                )
            ),
            bioemu_x0_support_basis_residual_oracle_target_clip_ppm=float(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_target_clip_ppm",
                    0.0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_min_teacher_ess=float(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_min_teacher_ess",
                    0.0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_support_basis_residual_oracle_start_scale=float(
                payload.get(
                    "bioemu_x0_support_basis_residual_oracle_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_loss_weight=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_scale=float(
                payload.get("bioemu_x0_hn_shared_q_energy_assignment_scale", 256.0)
            ),
            bioemu_x0_hn_shared_q_energy_assignment_temperature=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_temperature",
                    1.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_target_clip_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_target_clip_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_min_residual_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_min_residual_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_residual_power=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_residual_power",
                    1.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_min_teacher_ess=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_min_teacher_ess",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_positive_mass_weight=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_positive_mass_weight",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_target_side_margin_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_target_side_margin_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_hn_shared_q_energy_assignment_start_scale=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_energy_assignment_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_loss_weight=float(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_loss_weight", 0.0)
            ),
            bioemu_x0_hn_shared_q_generated_prior_scale=float(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_scale", 256.0)
            ),
            bioemu_x0_hn_shared_q_generated_prior_temperature=float(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_temperature", 1.0)
            ),
            bioemu_x0_hn_shared_q_generated_prior_target_clip_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_target_clip_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_min_residual_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_min_residual_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_residual_power=float(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_residual_power", 1.0)
            ),
            bioemu_x0_hn_shared_q_generated_prior_min_teacher_ess=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_min_teacher_ess",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_positive_mass_weight=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_positive_mass_weight",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_target_side_margin_ppm=float(
                payload.get(
                    "bioemu_x0_hn_shared_q_generated_prior_target_side_margin_ppm",
                    0.0,
                )
            ),
            bioemu_x0_hn_shared_q_generated_prior_warmup_epochs=int(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_warmup_epochs", 0)
            ),
            bioemu_x0_hn_shared_q_generated_prior_start_scale=float(
                payload.get("bioemu_x0_hn_shared_q_generated_prior_start_scale", 1.0)
            ),
            bioemu_x0_freeze_eval_support_seed=bool(
                payload.get("bioemu_x0_freeze_eval_support_seed", False)
            ),
            bioemu_x0_eval_support_seed_epoch=int(
                payload.get("bioemu_x0_eval_support_seed_epoch", 4)
            ),
            bioemu_x0_freeze_train_support_seed=bool(
                payload.get("bioemu_x0_freeze_train_support_seed", False)
            ),
            bioemu_x0_train_support_seed_epoch=int(
                payload.get("bioemu_x0_train_support_seed_epoch", 1)
            ),
            bioemu_x0_train_support_seed_epochs=_int_list_payload(
                payload.get("bioemu_x0_train_support_seed_epochs", [])
            ),
            bioemu_x0_train_support_seed_offset=int(
                payload.get("bioemu_x0_train_support_seed_offset", 0)
            ),
            bioemu_x0_active_subset_initial_epochs=int(
                payload.get("bioemu_x0_active_subset_initial_epochs", 0)
            ),
            bioemu_x0_active_subset_train_examples=int(
                payload.get("bioemu_x0_active_subset_train_examples", 0)
            ),
            bioemu_x0_active_subset_val_examples=int(
                payload.get("bioemu_x0_active_subset_val_examples", 0)
            ),
            bioemu_x0_active_subset_strategy=str(
                payload.get(
                    "bioemu_x0_active_subset_strategy",
                    "family_target_coverage",
                )
                or "family_target_coverage"
            ),
            bioemu_x0_active_subset_seed_offset=int(
                payload.get("bioemu_x0_active_subset_seed_offset", 0)
            ),
            bioemu_x0_active_subset_protect_replay_entities=bool(
                payload.get("bioemu_x0_active_subset_protect_replay_entities", False)
            ),
            bioemu_x0_active_subset_protected_entity_uids=list(
                payload.get("bioemu_x0_active_subset_protected_entity_uids", [])
            ),
            bioemu_x0_training_only_teacher_entity_uids=list(
                payload.get("bioemu_x0_training_only_teacher_entity_uids", [])
            ),
            bioemu_x0_active_subset_filter_to_bridge_audit_sidecar_entities=bool(
                payload.get(
                    (
                        "bioemu_x0_active_subset_filter_to_"
                        "bridge_audit_sidecar_entities"
                    ),
                    False,
                )
            ),
            bioemu_x0_active_subset_final_acceptance_eligible=bool(
                payload.get(
                    "bioemu_x0_active_subset_final_acceptance_eligible",
                    False,
                )
            ),
            bioemu_x0_active_subset_full_dataset_resume_required=bool(
                payload.get(
                    "bioemu_x0_active_subset_full_dataset_resume_required",
                    True,
                )
            ),
            bioemu_x0_active_subset_triage_gate_enabled=bool(
                payload.get("bioemu_x0_active_subset_triage_gate_enabled", False)
            ),
            bioemu_x0_active_subset_triage_stop_on_failure=bool(
                payload.get("bioemu_x0_active_subset_triage_stop_on_failure", False)
            ),
            bioemu_x0_active_subset_triage_min_macro_ccc=float(
                payload.get("bioemu_x0_active_subset_triage_min_macro_ccc", 0.0)
                or 0.0
            ),
            bioemu_x0_active_subset_triage_min_family_ccc=dict(
                payload.get("bioemu_x0_active_subset_triage_min_family_ccc", {})
                or {}
            ),
            bioemu_x0_active_subset_triage_use_mean_family_ccc=bool(
                payload.get("bioemu_x0_active_subset_triage_use_mean_family_ccc", True)
            ),
            bioemu_x0_active_subset_triage_floor_tolerance=float(
                payload.get("bioemu_x0_active_subset_triage_floor_tolerance", 0.0)
                or 0.0
            ),
            bioemu_x0_active_subset_triage_max_sidecar_to_generated_macro_gap=float(
                payload.get(
                    "bioemu_x0_active_subset_triage_max_sidecar_to_generated_macro_gap",
                    0.0,
                )
                or 0.0
            ),
            bioemu_x0_active_subset_triage_max_mean_sidecar_to_generated_macro_gap=float(
                payload.get(
                    "bioemu_x0_active_subset_triage_max_mean_sidecar_to_generated_macro_gap",
                    0.0,
                )
                or 0.0
            ),
            bioemu_x0_same_conformer_row_gate_enabled=bool(
                payload.get("bioemu_x0_same_conformer_row_gate_enabled", False)
            ),
            bioemu_x0_same_conformer_row_gate_focus_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_row_gate_focus_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_row_gate_required_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_row_gate_required_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_row_gate_min_family_count=int(
                payload.get("bioemu_x0_same_conformer_row_gate_min_family_count", 2)
            ),
            bioemu_x0_same_conformer_row_gate_min_rows=int(
                payload.get("bioemu_x0_same_conformer_row_gate_min_rows", 1)
            ),
            bioemu_x0_same_conformer_row_gate_max_scan_examples=int(
                payload.get("bioemu_x0_same_conformer_row_gate_max_scan_examples", 0)
            ),
            bioemu_x0_same_conformer_supervision_coherence_enabled=bool(
                payload.get(
                    "bioemu_x0_same_conformer_supervision_coherence_enabled",
                    False,
                )
            ),
            bioemu_x0_same_conformer_supervision_focus_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_supervision_focus_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_supervision_required_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_supervision_required_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_supervision_min_family_count=int(
                payload.get(
                    "bioemu_x0_same_conformer_supervision_min_family_count",
                    2,
                )
            ),
            bioemu_x0_same_conformer_supervision_max_residue_groups=int(
                payload.get(
                    "bioemu_x0_same_conformer_supervision_max_residue_groups",
                    0,
                )
            ),
            bioemu_x0_widen_checkpoint_tensors=bool(
                payload.get("bioemu_x0_widen_checkpoint_tensors", False)
            ),
            bioemu_x0_optimizer_name=str(
                payload.get("bioemu_x0_optimizer_name", "adamw") or "adamw"
            ),
            bioemu_x0_optimizer_foreach=(
                None
                if payload.get("bioemu_x0_optimizer_foreach", None) is None
                else bool(payload.get("bioemu_x0_optimizer_foreach"))
            ),
            bioemu_x0_optimizer_finite_update_guard_enabled=bool(
                payload.get("bioemu_x0_optimizer_finite_update_guard_enabled", True)
            ),
            bioemu_x0_joint_teacher_consistency_weight=float(
                payload.get("bioemu_x0_joint_teacher_consistency_weight", 0.05)
            ),
            bioemu_x0_full_sidecar_teacher_loss_weight=float(
                payload.get("bioemu_x0_full_sidecar_teacher_loss_weight", 0.0)
            ),
            bioemu_x0_full_sidecar_teacher_family_weights=dict(
                payload.get("bioemu_x0_full_sidecar_teacher_family_weights", {})
            ),
            bioemu_x0_full_sidecar_teacher_family_macro_weight=float(
                payload.get("bioemu_x0_full_sidecar_teacher_family_macro_weight", 1.0)
            ),
            bioemu_x0_full_sidecar_support_delta_teacher_loss_weight=float(
                payload.get(
                    "bioemu_x0_full_sidecar_support_delta_teacher_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_full_sidecar_support_delta_teacher_family_weights=dict(
                payload.get(
                    "bioemu_x0_full_sidecar_support_delta_teacher_family_weights",
                    {},
                )
            ),
            bioemu_x0_full_sidecar_support_delta_teacher_family_macro_weight=float(
                payload.get(
                    "bioemu_x0_full_sidecar_support_delta_teacher_family_macro_weight",
                    1.0,
                )
            ),
            bioemu_x0_full_sidecar_support_delta_teacher_min_valid_samples=int(
                payload.get(
                    "bioemu_x0_full_sidecar_support_delta_teacher_min_valid_samples",
                    2,
                )
            ),
            bioemu_x0_posterior_family_loss_weights=dict(
                payload.get("bioemu_x0_posterior_family_loss_weights", {})
            ),
            bioemu_x0_posterior_family_ccc_loss_weight=float(
                payload.get("bioemu_x0_posterior_family_ccc_loss_weight", 0.0)
            ),
            bioemu_x0_posterior_family_ccc_min_points=int(
                payload.get("bioemu_x0_posterior_family_ccc_min_points", 3)
            ),
            bioemu_x0_posterior_family_ccc_floor_loss_weight=float(
                payload.get("bioemu_x0_posterior_family_ccc_floor_loss_weight", 0.0)
            ),
            bioemu_x0_posterior_family_ccc_floors=dict(
                payload.get("bioemu_x0_posterior_family_ccc_floors", {})
            ),
            bioemu_x0_online_simplex_proxy_loss_weight=float(
                payload.get("bioemu_x0_online_simplex_proxy_loss_weight", 0.0)
            ),
            bioemu_x0_online_simplex_proxy_family_weights=dict(
                payload.get("bioemu_x0_online_simplex_proxy_family_weights", {})
            ),
            bioemu_x0_online_simplex_proxy_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_auto_gap_by_family",
                    {},
                )
            ),
            bioemu_x0_online_simplex_proxy_auto_gap_weight_scale=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_auto_gap_weight_scale",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_auto_gap_max_multiplier",
                    3.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_ccc_floors=dict(
                payload.get("bioemu_x0_online_simplex_proxy_ccc_floors", {})
            ),
            bioemu_x0_online_simplex_proxy_macro_ccc_floor=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_macro_ccc_floor",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_macro_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_macro_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_target_ccc_loss_weight=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_target_ccc_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_target_ccc_families=list(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_target_ccc_families",
                    [],
                )
                or []
            ),
            bioemu_x0_online_simplex_proxy_nonregression_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_nonregression_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_online_simplex_proxy_nonregression_ccc_floors=dict(
                payload.get(
                    "bioemu_x0_online_simplex_proxy_nonregression_ccc_floors",
                    {},
                )
            ),
            bioemu_x0_multiseed_active_set_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_multiseed_active_set_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_multiseed_active_set_family_floors=dict(
                payload.get("bioemu_x0_multiseed_active_set_family_floors", {})
            ),
            bioemu_x0_multiseed_active_set_family_weights=dict(
                payload.get("bioemu_x0_multiseed_active_set_family_weights", {})
            ),
            bioemu_x0_multiseed_active_set_seed_family_weights=dict(
                payload.get(
                    "bioemu_x0_multiseed_active_set_seed_family_weights",
                    {},
                )
            ),
            bioemu_x0_multiseed_active_set_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_multiseed_active_set_auto_gap_by_family",
                    {},
                )
            ),
            bioemu_x0_multiseed_active_set_auto_gap_weight_scale=float(
                payload.get(
                    "bioemu_x0_multiseed_active_set_auto_gap_weight_scale",
                    0.0,
                )
            ),
            bioemu_x0_multiseed_active_set_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_multiseed_active_set_auto_gap_max_multiplier",
                    3.0,
                )
            ),
            bioemu_x0_multiseed_active_set_include_gradient_projection=bool(
                payload.get(
                    "bioemu_x0_multiseed_active_set_include_gradient_projection",
                    False,
                )
            ),
            bioemu_x0_multiseed_active_set_min_points=int(
                payload.get("bioemu_x0_multiseed_active_set_min_points", 3)
            ),
            bioemu_x0_online_simplex_proxy_uncertainty_weight=float(
                payload.get("bioemu_x0_online_simplex_proxy_uncertainty_weight", 0.0)
            ),
            bioemu_x0_online_simplex_proxy_guard_weight=float(
                payload.get("bioemu_x0_online_simplex_proxy_guard_weight", 0.0)
            ),
            bioemu_x0_goal_ccc_ladder_contract=bool(
                payload.get("bioemu_x0_goal_ccc_ladder_contract", False)
            ),
            bioemu_x0_goal_ccc_floor_schedule_contract=bool(
                payload.get("bioemu_x0_goal_ccc_floor_schedule_contract", False)
            ),
            bioemu_x0_goal_ccc_cosatisfaction_contract=bool(
                payload.get("bioemu_x0_goal_ccc_cosatisfaction_contract", False)
            ),
            bioemu_x0_goal_ccc_fullpath_contract=bool(
                payload.get("bioemu_x0_goal_ccc_fullpath_contract", False)
            ),
            bioemu_x0_goal_ccc_worst_gap_contract=bool(
                payload.get("bioemu_x0_goal_ccc_worst_gap_contract", False)
            ),
            bioemu_x0_goal_ccc_floor_gap_power=float(
                payload.get("bioemu_x0_goal_ccc_floor_gap_power", 1.0)
            ),
            bioemu_x0_goal_ccc_floor_worst_family_mix=float(
                payload.get("bioemu_x0_goal_ccc_floor_worst_family_mix", 0.0)
            ),
            bioemu_x0_goal_ccc_family_gap_priority_contract=bool(
                payload.get("bioemu_x0_goal_ccc_family_gap_priority_contract", False)
            ),
            bioemu_x0_goal_ccc_floor_auto_gap_by_family=dict(
                payload.get("bioemu_x0_goal_ccc_floor_auto_gap_by_family", {})
            ),
            bioemu_x0_goal_ccc_floor_auto_gap_weight_scale=float(
                payload.get("bioemu_x0_goal_ccc_floor_auto_gap_weight_scale", 0.0)
            ),
            bioemu_x0_goal_ccc_floor_auto_gap_max_multiplier=float(
                payload.get("bioemu_x0_goal_ccc_floor_auto_gap_max_multiplier", 1.0)
            ),
            bioemu_x0_goal_ccc_macro_floor_contract=bool(
                payload.get("bioemu_x0_goal_ccc_macro_floor_contract", False)
            ),
            bioemu_x0_goal_ccc_macro_floor=float(
                payload.get("bioemu_x0_goal_ccc_macro_floor", 0.0)
            ),
            bioemu_x0_goal_ccc_macro_floor_mix=float(
                payload.get("bioemu_x0_goal_ccc_macro_floor_mix", 0.0)
            ),
            bioemu_x0_goal_ccc_macro_health_contract=bool(
                payload.get("bioemu_x0_goal_ccc_macro_health_contract", False)
            ),
            bioemu_x0_goal_ccc_macro_health_ess_floor=float(
                payload.get("bioemu_x0_goal_ccc_macro_health_ess_floor", 0.0)
            ),
            bioemu_x0_goal_ccc_macro_health_top_mass_cap=float(
                payload.get("bioemu_x0_goal_ccc_macro_health_top_mass_cap", 0.0)
            ),
            bioemu_x0_goal_ccc_macro_unweighted_contract=bool(
                payload.get("bioemu_x0_goal_ccc_macro_unweighted_contract", False)
            ),
            bioemu_x0_goal_ccc_macro_additive_contract=bool(
                payload.get("bioemu_x0_goal_ccc_macro_additive_contract", False)
            ),
            bioemu_x0_goal_ccc_low_tail_contract=bool(
                payload.get("bioemu_x0_goal_ccc_low_tail_contract", False)
            ),
            bioemu_x0_goal_ccc_low_tail_floor=float(
                payload.get("bioemu_x0_goal_ccc_low_tail_floor", 0.0)
            ),
            bioemu_x0_goal_ccc_low_tail_fraction=float(
                payload.get("bioemu_x0_goal_ccc_low_tail_fraction", 0.4)
            ),
            bioemu_x0_goal_ccc_low_tail_mix=float(
                payload.get("bioemu_x0_goal_ccc_low_tail_mix", 0.0)
            ),
            bioemu_x0_goal_ccc_low_tail_additive=bool(
                payload.get("bioemu_x0_goal_ccc_low_tail_additive", True)
            ),
            bioemu_x0_goal_ccc_low_tail_checkpoint_contract=bool(
                payload.get(
                    "bioemu_x0_goal_ccc_low_tail_checkpoint_contract",
                    False,
                )
            ),
            bioemu_x0_goal_ccc_source_reference_nonregression_contract=bool(
                payload.get(
                    "bioemu_x0_goal_ccc_source_reference_nonregression_contract",
                    False,
                )
            ),
            bioemu_x0_goal_ccc_source_reference_nonregression_floors=dict(
                payload.get(
                    "bioemu_x0_goal_ccc_source_reference_nonregression_floors",
                    {},
                )
            ),
            bioemu_x0_goal_ccc_all_family_cosatisfaction_contract=bool(
                payload.get(
                    "bioemu_x0_goal_ccc_all_family_cosatisfaction_contract",
                    False,
                )
            ),
            bioemu_x0_goal_ccc_coverage_aux_cosatisfaction_contract=bool(
                payload.get(
                    "bioemu_x0_goal_ccc_coverage_aux_cosatisfaction_contract",
                    False,
                )
            ),
            bioemu_x0_goal_ccc_coverage_weighted_aux_cosatisfaction_contract=bool(
                payload.get(
                    (
                        "bioemu_x0_goal_ccc_coverage_weighted_aux_"
                        "cosatisfaction_contract"
                    ),
                    False,
                )
            ),
            bioemu_x0_goal_ccc_floor_schedule_start_epoch=int(
                payload.get("bioemu_x0_goal_ccc_floor_schedule_start_epoch", 0) or 0
            ),
            bioemu_x0_goal_ccc_floor_schedule_ramp_epochs=max(
                0,
                int(
                    payload.get("bioemu_x0_goal_ccc_floor_schedule_ramp_epochs", 0)
                    or 0
                ),
            ),
            bioemu_x0_goal_ccc_floor_schedule_start_floors=dict(
                payload.get("bioemu_x0_goal_ccc_floor_schedule_start_floors", {})
            ),
            bioemu_x0_protected_family_ensemble_loss_weight=float(
                payload.get("bioemu_x0_protected_family_ensemble_loss_weight", 0.0)
            ),
            bioemu_x0_protected_family_ensemble_family_weights=dict(
                payload.get("bioemu_x0_protected_family_ensemble_family_weights", {})
            ),
            bioemu_x0_protected_family_ensemble_warmup_epochs=int(
                payload.get("bioemu_x0_protected_family_ensemble_warmup_epochs", 0)
            ),
            bioemu_x0_protected_family_ensemble_start_scale=float(
                payload.get("bioemu_x0_protected_family_ensemble_start_scale", 1.0)
            ),
            bioemu_x0_checkpoint_metric=str(
                payload.get("bioemu_x0_checkpoint_metric", "loss")
            ),
            bioemu_x0_checkpoint_metric_family_weights=dict(
                payload.get("bioemu_x0_checkpoint_metric_family_weights", {})
            ),
            bioemu_x0_checkpoint_metric_target_gap_max_blend=float(
                payload.get("bioemu_x0_checkpoint_metric_target_gap_max_blend", 0.0)
            ),
            bioemu_x0_checkpoint_metric_target_gap_low_tail_fraction=float(
                payload.get(
                    "bioemu_x0_checkpoint_metric_target_gap_low_tail_fraction",
                    0.0,
                )
            ),
            bioemu_x0_checkpoint_metric_target_gap_low_tail_blend=float(
                payload.get(
                    "bioemu_x0_checkpoint_metric_target_gap_low_tail_blend",
                    0.0,
                )
            ),
            bioemu_x0_checkpoint_metric_nonregression_ccc_floors=dict(
                payload.get(
                    "bioemu_x0_checkpoint_metric_nonregression_ccc_floors",
                    {},
                )
            ),
            bioemu_x0_checkpoint_metric_nonregression_penalty_weight=float(
                payload.get(
                    "bioemu_x0_checkpoint_metric_nonregression_penalty_weight",
                    0.0,
                )
            ),
            bioemu_x0_checkpoint_metric_nonregression_hard_gate=bool(
                payload.get(
                    "bioemu_x0_checkpoint_metric_nonregression_hard_gate",
                    False,
                )
            ),
            bioemu_x0_checkpoint_metric_require_model_dependent_source=bool(
                payload.get(
                    "bioemu_x0_checkpoint_metric_require_model_dependent_source",
                    False,
                )
            ),
            bioemu_x0_checkpoint_metric_support_within_floors=dict(
                payload.get(
                    "bioemu_x0_checkpoint_metric_support_within_floors",
                    payload.get(
                        "bioemu_x0_must_preserve_support_within_fraction_floors",
                        {},
                    ),
                )
            ),
            bioemu_x0_checkpoint_metric_support_within_penalty_weight=float(
                payload.get(
                    "bioemu_x0_checkpoint_metric_support_within_penalty_weight",
                    0.0,
                )
            ),
            bioemu_x0_checkpoint_metric_support_within_hard_gate=bool(
                payload.get(
                    "bioemu_x0_checkpoint_metric_support_within_hard_gate",
                    "bioemu_x0_must_preserve_support_within_fraction_floors" in payload,
                )
            ),
            bioemu_x0_checkpoint_metric_health_gates=dict(
                payload.get("bioemu_x0_checkpoint_metric_health_gates", {})
            ),
            bioemu_x0_checkpoint_metric_health_penalty_weight=float(
                payload.get(
                    "bioemu_x0_checkpoint_metric_health_penalty_weight",
                    0.0,
                )
            ),
            bioemu_x0_checkpoint_metric_health_hard_gate=bool(
                payload.get("bioemu_x0_checkpoint_metric_health_hard_gate", False)
            ),
            bioemu_x0_mid_epoch_checkpoint_interval_examples=max(
                0,
                int(
                    payload.get(
                        "bioemu_x0_mid_epoch_checkpoint_interval_examples",
                        0,
                    )
                    or 0
                ),
            ),
            bioemu_x0_multiseed_checkpoint_enabled=bool(
                payload.get("bioemu_x0_multiseed_checkpoint_enabled", False)
            ),
            bioemu_x0_multiseed_checkpoint_seed_epochs=_int_list_payload(
                payload.get("bioemu_x0_multiseed_checkpoint_seed_epochs", [])
            ),
            bioemu_x0_multiseed_checkpoint_summary_path=payload.get(
                "bioemu_x0_multiseed_checkpoint_summary_path"
            ),
            bioemu_x0_emit_initial_multiseed_checkpoint_summary=bool(
                payload.get(
                    "bioemu_x0_emit_initial_multiseed_checkpoint_summary",
                    False,
                )
            ),
            bioemu_x0_restore_best_checkpoint_on_metric_regression=bool(
                payload.get(
                    "bioemu_x0_restore_best_checkpoint_on_metric_regression",
                    False,
                )
            ),
            bioemu_x0_emit_ready_only_for_checkpoint_metric_acceptance=bool(
                payload.get(
                    "bioemu_x0_emit_ready_only_for_checkpoint_metric_acceptance",
                    False,
                )
            ),
            bioemu_x0_disable_async_offline_ready_emission=bool(
                payload.get("bioemu_x0_disable_async_offline_ready_emission", False)
            ),
            bioemu_x0_nonregression_gradient_projection_enabled=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_enabled",
                    False,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_strength=float(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_strength",
                    1.0,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_min_guard_norm=float(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_min_guard_norm",
                    1.0e-12,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_skip_on_oom=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_skip_on_oom",
                    True,
                )
            ),
            bioemu_x0_backward_oom_skip_enabled=bool(
                payload.get("bioemu_x0_backward_oom_skip_enabled", False)
            ),
            bioemu_x0_compact_geometry_runtime_diagnostics_required=bool(
                payload.get(
                    "bioemu_x0_compact_geometry_runtime_diagnostics_required",
                    False,
                )
            ),
            bioemu_x0_compact_geometry_oom_fallback_allowed=bool(
                payload.get(
                    "bioemu_x0_compact_geometry_oom_fallback_allowed",
                    True,
                )
            ),
            bioemu_x0_compact_geometry_force_empty_edges=bool(
                payload.get("bioemu_x0_compact_geometry_force_empty_edges", False)
            ),
            bioemu_x0_provider_sample_oom_retry_enabled=bool(
                payload.get("bioemu_x0_provider_sample_oom_retry_enabled", False)
            ),
            bioemu_x0_provider_sample_oom_retry_sample_count=int(
                payload.get("bioemu_x0_provider_sample_oom_retry_sample_count", 0)
            ),
            bioemu_x0_cuda_empty_cache_each_example=bool(
                payload.get("bioemu_x0_cuda_empty_cache_each_example", False)
            ),
            bioemu_x0_nonregression_gradient_projection_include_decoder_trust_region=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_include_decoder_trust_region",
                    False,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_include_online_simplex=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_include_online_simplex",
                    False,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_include_shared_oracle_decoder=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_include_shared_oracle_decoder",
                    False,
                )
            ),
            bioemu_x0_nonregression_gradient_projection_include_same_conformer_generated_prior=bool(
                payload.get(
                    "bioemu_x0_nonregression_gradient_projection_include_same_conformer_generated_prior",
                    False,
                )
            ),
            bioemu_x0_family_expert_aggregate_mode=str(
                payload.get("bioemu_x0_family_expert_aggregate_mode", "sum")
            ),
            bioemu_x0_family_expert_residual_cap=float(
                payload.get("bioemu_x0_family_expert_residual_cap", 0.0)
            ),
            bioemu_x0_family_expert_conflict_penalty_weight=float(
                payload.get("bioemu_x0_family_expert_conflict_penalty_weight", 0.0)
            ),
            bioemu_x0_detach_evidence_head_inputs=bool(
                payload.get("bioemu_x0_detach_evidence_head_inputs", True)
            ),
            bioemu_x0_evidence_head_chunk_size=int(
                payload.get("bioemu_x0_evidence_head_chunk_size", 4)
            ),
            bioemu_x0_evidence_head_use_checkpoint=bool(
                payload.get("bioemu_x0_evidence_head_use_checkpoint", False)
            ),
            bioemu_x0_evidence_head_hidden_dim=int(
                payload.get("bioemu_x0_evidence_head_hidden_dim", 0)
            ),
            bioemu_x0_evidence_head_depth=int(
                payload.get("bioemu_x0_evidence_head_depth", 2)
            ),
            bioemu_x0_supervised_support_envelope_loss_weight=float(
                payload.get(
                    "bioemu_x0_supervised_support_envelope_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_supervised_support_envelope_family_weights=dict(
                payload.get("bioemu_x0_supervised_support_envelope_family_weights", {})
            ),
            bioemu_x0_supervised_support_envelope_softness=float(
                payload.get("bioemu_x0_supervised_support_envelope_softness", 0.05)
            ),
            bioemu_x0_supervised_support_envelope_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_supervised_support_envelope_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_supervised_support_envelope_start_scale=float(
                payload.get(
                    "bioemu_x0_supervised_support_envelope_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_supervised_support_center_loss_weight=float(
                payload.get(
                    "bioemu_x0_supervised_support_center_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_supervised_support_center_family_weights=dict(
                payload.get("bioemu_x0_supervised_support_center_family_weights", {})
            ),
            bioemu_x0_supervised_support_center_softness=float(
                payload.get("bioemu_x0_supervised_support_center_softness", 0.05)
            ),
            bioemu_x0_supervised_support_center_gap_focus_scale=float(
                payload.get(
                    "bioemu_x0_supervised_support_center_gap_focus_scale",
                    0.0,
                )
            ),
            bioemu_x0_supervised_support_center_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_supervised_support_center_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_supervised_support_center_start_scale=float(
                payload.get(
                    "bioemu_x0_supervised_support_center_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_raw_sample_support_envelope_loss_weight=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_raw_sample_support_envelope_family_weights=dict(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_family_weights",
                    {},
                )
            ),
            bioemu_x0_raw_sample_support_envelope_softness=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_softness",
                    0.02,
                )
            ),
            bioemu_x0_raw_sample_support_envelope_margin_sigma_scale=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_margin_sigma_scale",
                    1.0,
                )
            ),
            bioemu_x0_raw_sample_support_envelope_margin_cap_by_family=dict(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_margin_cap_by_family",
                    {
                        "HN": 0.05,
                        "N": 0.35,
                        "CA": 0.25,
                        "CB": 0.35,
                        "C'": 0.18,
                    },
                )
            ),
            bioemu_x0_raw_sample_support_envelope_min_rows=int(
                payload.get("bioemu_x0_raw_sample_support_envelope_min_rows", 1)
            ),
            bioemu_x0_raw_sample_support_envelope_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_raw_sample_support_envelope_start_scale=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_envelope_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_raw_sample_support_width_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_width_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_raw_sample_support_width_floor_min_by_family=dict(
                payload.get(
                    "bioemu_x0_raw_sample_support_width_floor_min_by_family",
                    {},
                )
            ),
            bioemu_x0_raw_sample_support_width_floor_max_by_family=dict(
                payload.get(
                    "bioemu_x0_raw_sample_support_width_floor_max_by_family",
                    {
                        "HN": 0.60,
                        "N": 4.00,
                        "CA": 3.00,
                        "CB": 6.00,
                        "C'": 1.50,
                    },
                )
            ),
            bioemu_x0_raw_sample_support_width_floor_gap_scale=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_width_floor_gap_scale",
                    0.5,
                )
            ),
            bioemu_x0_support_width_floor_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_support_width_floor_auto_gap_by_family",
                    {},
                )
                or {}
            ),
            bioemu_x0_support_width_floor_auto_gap_scale=float(
                payload.get(
                    "bioemu_x0_support_width_floor_auto_gap_scale",
                    0.0,
                )
            ),
            bioemu_x0_support_width_floor_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_support_width_floor_auto_gap_max_multiplier",
                    2.0,
                )
            ),
            bioemu_x0_raw_sample_support_tail_loss_weight=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_tail_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_raw_sample_support_tail_fraction=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_tail_fraction",
                    0.125,
                )
            ),
            bioemu_x0_raw_sample_support_ranked_tail_loss_weight=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_ranked_tail_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_raw_sample_support_ranked_tail_fraction=float(
                payload.get(
                    "bioemu_x0_raw_sample_support_ranked_tail_fraction",
                    0.125,
                )
            ),
            bioemu_x0_tail_risk_dual_loss_weight=float(
                payload.get("bioemu_x0_tail_risk_dual_loss_weight", 0.0)
            ),
            bioemu_x0_tail_risk_dual_target_fraction=float(
                payload.get("bioemu_x0_tail_risk_dual_target_fraction", 0.02)
            ),
            bioemu_x0_tail_risk_dual_loss_kind=str(
                payload.get("bioemu_x0_tail_risk_dual_loss_kind", "quadratic")
            ),
            bioemu_x0_tail_risk_dual_min_active_samples=int(
                payload.get("bioemu_x0_tail_risk_dual_min_active_samples", 1)
            ),
            bioemu_x0_sample_support_delta_enabled=bool(
                payload.get("bioemu_x0_sample_support_delta_enabled", False)
            ),
            bioemu_x0_sample_support_delta_max_abs_by_family=dict(
                payload.get(
                    "bioemu_x0_sample_support_delta_max_abs_by_family",
                    {},
                )
            ),
            bioemu_x0_sample_support_delta_scale=float(
                payload.get("bioemu_x0_sample_support_delta_scale", 1.0)
            ),
            bioemu_x0_sample_support_delta_scale_by_family=dict(
                payload.get(
                    "bioemu_x0_sample_support_delta_scale_by_family",
                    {},
                )
                or {}
            ),
            bioemu_x0_sample_support_delta_centered=bool(
                payload.get("bioemu_x0_sample_support_delta_centered", True)
            ),
            bioemu_x0_sample_support_delta_init_std=float(
                payload.get("bioemu_x0_sample_support_delta_init_std", 0.0)
            ),
            bioemu_x0_sample_support_delta_lr_multiplier=float(
                payload.get("bioemu_x0_sample_support_delta_lr_multiplier", -1.0)
            ),
            bioemu_x0_sample_support_delta_width_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sample_support_delta_width_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sample_support_delta_width_floor_min_by_family=dict(
                payload.get(
                    "bioemu_x0_sample_support_delta_width_floor_min_by_family",
                    {},
                )
                or {}
            ),
            bioemu_x0_mechanism_support_envelope_loss_weight=float(
                payload.get(
                    "bioemu_x0_mechanism_support_envelope_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_mechanism_support_envelope_family_weights=dict(
                payload.get("bioemu_x0_mechanism_support_envelope_family_weights", {})
            ),
            bioemu_x0_mechanism_support_envelope_softness=float(
                payload.get("bioemu_x0_mechanism_support_envelope_softness", 0.05)
            ),
            bioemu_x0_mechanism_support_envelope_risk_scale=float(
                payload.get("bioemu_x0_mechanism_support_envelope_risk_scale", 1.0)
            ),
            bioemu_x0_mechanism_support_envelope_min_context_weight=float(
                payload.get(
                    "bioemu_x0_mechanism_support_envelope_min_context_weight",
                    1.0,
                )
            ),
            bioemu_x0_mechanism_support_envelope_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_mechanism_support_envelope_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_mechanism_support_envelope_start_scale=float(
                payload.get(
                    "bioemu_x0_mechanism_support_envelope_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_conditional_support_envelope_loss_weight=float(
                payload.get(
                    "bioemu_x0_conditional_support_envelope_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_conditional_support_envelope_family_weights=dict(
                payload.get("bioemu_x0_conditional_support_envelope_family_weights", {})
            ),
            bioemu_x0_conditional_support_envelope_softness=float(
                payload.get("bioemu_x0_conditional_support_envelope_softness", 0.05)
            ),
            bioemu_x0_conditional_support_envelope_risk_scale=float(
                payload.get("bioemu_x0_conditional_support_envelope_risk_scale", 1.0)
            ),
            bioemu_x0_conditional_support_envelope_min_context_weight=float(
                payload.get(
                    "bioemu_x0_conditional_support_envelope_min_context_weight",
                    1.0,
                )
            ),
            bioemu_x0_conditional_support_envelope_gap_margin=float(
                payload.get("bioemu_x0_conditional_support_envelope_gap_margin", 0.0)
            ),
            bioemu_x0_conditional_support_envelope_gap_focus_scale=float(
                payload.get(
                    "bioemu_x0_conditional_support_envelope_gap_focus_scale",
                    0.0,
                )
            ),
            bioemu_x0_conditional_support_envelope_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_conditional_support_envelope_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_conditional_support_envelope_start_scale=float(
                payload.get(
                    "bioemu_x0_conditional_support_envelope_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_directional_support_expansion_loss_weight=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_directional_support_expansion_family_weights=dict(
                payload.get(
                    "bioemu_x0_directional_support_expansion_family_weights",
                    {},
                )
            ),
            bioemu_x0_directional_support_expansion_softness=float(
                payload.get("bioemu_x0_directional_support_expansion_softness", 0.05)
            ),
            bioemu_x0_directional_support_expansion_risk_scale=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_risk_scale",
                    1.0,
                )
            ),
            bioemu_x0_directional_support_expansion_min_context_weight=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_min_context_weight",
                    1.0,
                )
            ),
            bioemu_x0_directional_support_expansion_margin=float(
                payload.get("bioemu_x0_directional_support_expansion_margin", 0.0)
            ),
            bioemu_x0_directional_support_expansion_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_directional_support_expansion_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_directional_support_expansion_start_scale=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_directional_support_expansion_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_directional_support_expansion_auto_gap_by_family",
                    {},
                )
            ),
            bioemu_x0_directional_support_expansion_auto_gap_weight_scale=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_auto_gap_weight_scale",
                    0.0,
                )
            ),
            bioemu_x0_directional_support_expansion_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_directional_support_expansion_auto_gap_max_multiplier",
                    2.5,
                )
            ),
            bioemu_x0_shared_oracle_decoder_loss_weight=float(
                payload.get("bioemu_x0_shared_oracle_decoder_loss_weight", 0.0)
            ),
            bioemu_x0_shared_oracle_decoder_scale=float(
                payload.get("bioemu_x0_shared_oracle_decoder_scale", 256.0)
            ),
            bioemu_x0_shared_oracle_decoder_oracle_solver=str(
                payload.get("bioemu_x0_shared_oracle_decoder_oracle_solver", "")
                or ""
            ),
            bioemu_x0_shared_oracle_decoder_family_weights=dict(
                payload.get("bioemu_x0_shared_oracle_decoder_family_weights", {})
            ),
            bioemu_x0_shared_oracle_decoder_ccc_loss_weight=float(
                payload.get("bioemu_x0_shared_oracle_decoder_ccc_loss_weight", 0.0)
            ),
            bioemu_x0_shared_oracle_decoder_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_shared_oracle_decoder_ccc_floors=dict(
                payload.get("bioemu_x0_shared_oracle_decoder_ccc_floors", {})
            ),
            bioemu_x0_shared_oracle_decoder_energy_family_weights=dict(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_energy_family_weights",
                    {},
                )
            ),
            bioemu_x0_shared_oracle_decoder_use_sidecar_oracle=bool(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_use_sidecar_oracle",
                    False,
                )
            ),
            bioemu_x0_shared_oracle_decoder_metric_contract=bool(
                payload.get("bioemu_x0_shared_oracle_decoder_metric_contract", False)
            ),
            bioemu_x0_shared_oracle_decoder_metric_contract_version=str(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_metric_contract_version",
                    "",
                )
                or ""
            ),
            bioemu_x0_shared_oracle_decoder_required_diagnostics=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_shared_oracle_decoder_required_diagnostics",
                    [],
                )
            ],
            bioemu_x0_shared_oracle_decoder_metric_target_ccc=float(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_metric_target_ccc",
                    0.95,
                )
            ),
            bioemu_x0_shared_oracle_decoder_metric_health_gates=dict(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_metric_health_gates",
                    {},
                )
            ),
            bioemu_x0_shared_oracle_decoder_uncertainty_penalty_weight=float(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_uncertainty_penalty_weight",
                    0.0,
                )
            ),
            bioemu_x0_shared_oracle_decoder_uncertainty_threshold=float(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_uncertainty_threshold",
                    0.0,
                )
            ),
            bioemu_x0_shared_oracle_decoder_ood_penalty_weight=float(
                payload.get(
                    "bioemu_x0_shared_oracle_decoder_ood_penalty_weight",
                    0.0,
                )
            ),
            bioemu_x0_shared_oracle_decoder_ood_threshold=float(
                payload.get("bioemu_x0_shared_oracle_decoder_ood_threshold", 0.0)
            ),
            bioemu_x0_leave_family_out_oracle_decoder_loss_weight=float(
                payload.get(
                    "bioemu_x0_leave_family_out_oracle_decoder_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_leave_family_out_oracle_decoder_scale=float(
                payload.get("bioemu_x0_leave_family_out_oracle_decoder_scale", 256.0)
            ),
            bioemu_x0_leave_family_out_oracle_decoder_family_weights=dict(
                payload.get(
                    "bioemu_x0_leave_family_out_oracle_decoder_family_weights",
                    {},
                )
            ),
            bioemu_x0_leave_family_out_oracle_decoder_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_leave_family_out_oracle_decoder_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_leave_family_out_oracle_decoder_start_scale=float(
                payload.get(
                    "bioemu_x0_leave_family_out_oracle_decoder_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_pairwise_cooccurrence_decoder_loss_weight=float(
                payload.get(
                    "bioemu_x0_pairwise_cooccurrence_decoder_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_pairwise_cooccurrence_decoder_scale=float(
                payload.get("bioemu_x0_pairwise_cooccurrence_decoder_scale", 256.0)
            ),
            bioemu_x0_pairwise_cooccurrence_decoder_pair_weights=dict(
                payload.get(
                    "bioemu_x0_pairwise_cooccurrence_decoder_pair_weights",
                    {},
                )
            ),
            bioemu_x0_pairwise_cooccurrence_decoder_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_pairwise_cooccurrence_decoder_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_pairwise_cooccurrence_decoder_start_scale=float(
                payload.get(
                    "bioemu_x0_pairwise_cooccurrence_decoder_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_loss_weight=float(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_scale",
                    256.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_focus_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_focus_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_decoder_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_decoder_fallback_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_fallback_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_decoder_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_fallback_min_rows"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_family_weights=dict(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_family_weights",
                    {},
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_min_family_count=int(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_min_family_count",
                    2,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_gap_focus_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_gap_focus_scale",
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_worst_family_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_worst_family_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_family_balanced_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_family_balanced_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_weight_power=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_coverage_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_enabled=bool(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_coverage_fallback_enabled"
                    ),
                    False,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_coverage_fallback_min_family_count"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_row_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_coverage_fallback_row_min_family_count"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_coverage_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "decoder_coverage_fallback_min_rows"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_decoder_start_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_decoder_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_scale",
                    256.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_temperature=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_temperature"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_focus_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_focus_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_fallback_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_fallback_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_fallback_min_rows"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_family_weights=dict(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_family_weights"
                    ),
                    {},
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_min_family_count"
                    ),
                    2,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_min_teacher_ess=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_min_teacher_ess"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_max_teacher_prior_blend=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_max_teacher_prior_blend"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_max_teacher_top_mass=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_max_teacher_top_mass"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_gap_focus_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_gap_focus_scale"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_worst_family_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_worst_family_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_family_balanced_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_family_balanced_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_enabled=bool(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "coverage_fallback_enabled"
                    ),
                    False,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "coverage_fallback_min_family_count"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_row_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "coverage_fallback_row_min_family_count"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_coverage_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "coverage_fallback_min_rows"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_mass_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "topk_mass_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_fraction=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "topk_fraction"
                    ),
                    0.125,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_min_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "topk_min_count"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_target_mass_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "topk_target_mass_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_topk_listwise_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "topk_listwise_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_top1_nll_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "top1_nll_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_ess_floor=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_ess_floor"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_ess_floor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_ess_floor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_top_mass_cap=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_top_mass_cap"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_top_mass_cap_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_top_mass_cap_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_health_anchor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_prior_blend=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_health_anchor_prior_blend"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_health_anchor_gap_power=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_health_anchor_gap_power"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_std_cap=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_logit_std_cap"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_std_cap_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_logit_std_cap_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_range_cap=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_logit_range_cap"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_student_logit_range_cap_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_energy_assignment_"
                        "student_logit_range_cap_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_warmup_epochs=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_warmup_epochs"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_energy_assignment_start_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "energy_assignment_start_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_scale"
                    ),
                    256.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_temperature=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_temperature"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_focus_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_focus_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_fallback_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_fallback_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_fallback_min_rows"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_family_weights=dict(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_family_weights"
                    ),
                    {},
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_min_family_count"
                    ),
                    2,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_min_teacher_ess=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_min_teacher_ess"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_max_teacher_prior_blend=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_max_teacher_prior_blend"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_max_teacher_top_mass=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_max_teacher_top_mass"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_gap_focus_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_gap_focus_scale"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_worst_family_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_worst_family_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_family_balanced_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_family_balanced_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_weight_power=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_coverage_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_enabled=bool(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_generated_prior_"
                        "coverage_fallback_enabled"
                    ),
                    False,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_generated_prior_"
                        "coverage_fallback_min_family_count"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_row_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_generated_prior_"
                        "coverage_fallback_row_min_family_count"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_coverage_fallback_min_rows=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_generated_prior_"
                        "coverage_fallback_min_rows"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_mass_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_topk_mass_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_fraction=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_topk_fraction"
                    ),
                    0.125,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_min_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_topk_min_count"
                    ),
                    1,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_target_mass_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_topk_target_mass_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_topk_listwise_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_topk_listwise_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_top1_nll_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_top1_nll_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_ess_floor=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_student_ess_floor"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_ess_floor_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_student_ess_floor_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_top_mass_cap=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_student_top_mass_cap"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_student_top_mass_cap_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_student_top_mass_cap_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_log_prob_source=str(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_log_prob_source"
                    ),
                    "raw",
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_enabled=bool(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_isolated_prior_restoration_enabled"
                    ),
                    False,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_separate_clip_enabled=bool(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_isolated_prior_restoration_separate_clip_enabled"
                    ),
                    False,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_isolated_prior_restoration_clip_norm_multiplier=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_isolated_prior_restoration_clip_norm_multiplier"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_warmup_epochs=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_warmup_epochs"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_cosatisfaction_generated_prior_start_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_cosatisfaction_"
                        "generated_prior_start_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_loss_weight=float(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_scale",
                    256.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_focus_families=[
                str(value)
                for value in payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_focus_families",
                    [],
                )
            ],
            bioemu_x0_same_conformer_coverage_aux_decoder_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "decoder_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_coverage_aux_decoder_family_weights=dict(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_family_weights",
                    {},
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_min_family_count=int(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_min_family_count",
                    2,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_gap_focus_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_gap_focus_scale",
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_worst_family_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "decoder_worst_family_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_family_balanced_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "decoder_family_balanced_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_coverage_weight_power=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "decoder_coverage_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_warmup_epochs=int(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_warmup_epochs",
                    0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_decoder_start_scale=float(
                payload.get(
                    "bioemu_x0_same_conformer_coverage_aux_decoder_start_scale",
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_loss_weight=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_loss_weight"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_scale"
                    ),
                    256.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_temperature=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_temperature"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_focus_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_focus_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_required_families=[
                str(value)
                for value in payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_required_families"
                    ),
                    [],
                )
            ],
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_family_weights=dict(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_family_weights"
                    ),
                    {},
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_min_family_count=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_min_family_count"
                    ),
                    2,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_min_teacher_ess=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_min_teacher_ess"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_max_teacher_prior_blend=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_max_teacher_prior_blend"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_max_teacher_top_mass=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_max_teacher_top_mass"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_gap_focus_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_gap_focus_scale"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_worst_family_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_worst_family_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_family_balanced_error_mix=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_family_balanced_error_mix"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_coverage_weight_power=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_coverage_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_warmup_epochs=int(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_warmup_epochs"
                    ),
                    0,
                )
            ),
            bioemu_x0_same_conformer_coverage_aux_energy_assignment_start_scale=float(
                payload.get(
                    (
                        "bioemu_x0_same_conformer_coverage_aux_"
                        "energy_assignment_start_scale"
                    ),
                    1.0,
                )
            ),
            bioemu_x0_phase_aware_co_support_report_path=str(
                payload.get("bioemu_x0_phase_aware_co_support_report_path", "")
                or ""
            ),
            bioemu_x0_phase_aware_co_support_pair_multiplier=float(
                payload.get(
                    "bioemu_x0_phase_aware_co_support_pair_multiplier",
                    1.75,
                )
            ),
            bioemu_x0_phase_aware_co_support_pairs=list(
                payload.get("bioemu_x0_phase_aware_co_support_pairs", []) or []
            ),
            bioemu_x0_phase_aware_co_support_auto_from_gap_enabled=bool(
                payload.get(
                    "bioemu_x0_phase_aware_co_support_auto_from_gap_enabled",
                    False,
                )
            ),
            bioemu_x0_phase_aware_co_support_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_phase_aware_co_support_auto_gap_by_family",
                    {},
                )
            ),
            bioemu_x0_phase_aware_co_support_auto_gap_threshold=float(
                payload.get(
                    "bioemu_x0_phase_aware_co_support_auto_gap_threshold",
                    0.05,
                )
            ),
            bioemu_x0_phase_aware_co_support_auto_gap_scale=float(
                payload.get(
                    "bioemu_x0_phase_aware_co_support_auto_gap_scale",
                    1.0,
                )
            ),
            bioemu_x0_rowwise_oracle_decoder_loss_weight=float(
                payload.get("bioemu_x0_rowwise_oracle_decoder_loss_weight", 0.0)
            ),
            bioemu_x0_rowwise_oracle_decoder_scale=float(
                payload.get("bioemu_x0_rowwise_oracle_decoder_scale", 256.0)
            ),
            bioemu_x0_rowwise_oracle_decoder_family_weights=dict(
                payload.get("bioemu_x0_rowwise_oracle_decoder_family_weights", {})
            ),
            bioemu_x0_rowwise_oracle_decoder_family_scales=dict(
                payload.get("bioemu_x0_rowwise_oracle_decoder_family_scales", {})
            ),
            bioemu_x0_rowwise_oracle_decoder_min_teacher_ess=float(
                payload.get("bioemu_x0_rowwise_oracle_decoder_min_teacher_ess", 0.0)
            ),
            bioemu_x0_rowwise_oracle_decoder_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_rowwise_oracle_decoder_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_rowwise_oracle_decoder_warmup_epochs=int(
                payload.get("bioemu_x0_rowwise_oracle_decoder_warmup_epochs", 0)
            ),
            bioemu_x0_rowwise_oracle_decoder_start_scale=float(
                payload.get("bioemu_x0_rowwise_oracle_decoder_start_scale", 1.0)
            ),
            bioemu_x0_rowwise_support_rank_loss_weight=float(
                payload.get("bioemu_x0_rowwise_support_rank_loss_weight", 0.0)
            ),
            bioemu_x0_rowwise_support_rank_scale=float(
                payload.get("bioemu_x0_rowwise_support_rank_scale", 256.0)
            ),
            bioemu_x0_rowwise_support_rank_family_weights=dict(
                payload.get("bioemu_x0_rowwise_support_rank_family_weights", {})
            ),
            bioemu_x0_rowwise_support_rank_family_scales=dict(
                payload.get("bioemu_x0_rowwise_support_rank_family_scales", {})
            ),
            bioemu_x0_rowwise_support_rank_margin=float(
                payload.get("bioemu_x0_rowwise_support_rank_margin", 0.05)
            ),
            bioemu_x0_rowwise_support_rank_min_teacher_ess=float(
                payload.get("bioemu_x0_rowwise_support_rank_min_teacher_ess", 0.0)
            ),
            bioemu_x0_rowwise_support_rank_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_rowwise_support_rank_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_rowwise_support_rank_risk_scale=float(
                payload.get("bioemu_x0_rowwise_support_rank_risk_scale", 0.0)
            ),
            bioemu_x0_rowwise_support_rank_min_context_weight=float(
                payload.get("bioemu_x0_rowwise_support_rank_min_context_weight", 1.0)
            ),
            bioemu_x0_rowwise_support_rank_gap_focus_scale=float(
                payload.get("bioemu_x0_rowwise_support_rank_gap_focus_scale", 0.0)
            ),
            bioemu_x0_rowwise_support_rank_warmup_epochs=int(
                payload.get("bioemu_x0_rowwise_support_rank_warmup_epochs", 0)
            ),
            bioemu_x0_rowwise_support_rank_start_scale=float(
                payload.get("bioemu_x0_rowwise_support_rank_start_scale", 1.0)
            ),
            bioemu_x0_rowwise_support_rank_auto_gap_by_family=dict(
                payload.get("bioemu_x0_rowwise_support_rank_auto_gap_by_family", {})
            ),
            bioemu_x0_rowwise_support_rank_auto_gap_weight_scale=float(
                payload.get(
                    "bioemu_x0_rowwise_support_rank_auto_gap_weight_scale",
                    0.0,
                )
            ),
            bioemu_x0_rowwise_support_rank_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_rowwise_support_rank_auto_gap_max_multiplier",
                    2.5,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_scale=float(
                payload.get("bioemu_x0_sidecar_observed_support_rank_scale", 256.0)
            ),
            bioemu_x0_sidecar_observed_support_rank_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_family_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_family_scales=dict(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_family_scales",
                    {},
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_margin=float(
                payload.get("bioemu_x0_sidecar_observed_support_rank_margin", 0.05)
            ),
            bioemu_x0_sidecar_observed_support_rank_min_teacher_ess=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_min_teacher_ess",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_max_teacher_prior_blend=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_max_teacher_prior_blend",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_risk_scale=float(
                payload.get("bioemu_x0_sidecar_observed_support_rank_risk_scale", 0.0)
            ),
            bioemu_x0_sidecar_observed_support_rank_min_context_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_min_context_weight",
                    1.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_gap_focus_scale=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_gap_focus_scale",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_warmup_epochs=int(
                payload.get("bioemu_x0_sidecar_observed_support_rank_warmup_epochs", 0)
            ),
            bioemu_x0_sidecar_observed_support_rank_start_scale=float(
                payload.get("bioemu_x0_sidecar_observed_support_rank_start_scale", 1.0)
            ),
            bioemu_x0_sidecar_observed_support_rank_auto_gap_by_family=dict(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_auto_gap_by_family",
                    {},
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_auto_gap_weight_scale=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_auto_gap_weight_scale",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_observed_support_rank_auto_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_sidecar_observed_support_rank_auto_gap_max_multiplier",
                    2.5,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_distill_weight=float(
                payload.get("bioemu_x0_sidecar_shared_q_oracle_distill_weight", 0.0)
            ),
            bioemu_x0_sidecar_shared_q_oracle_distill_scale=float(
                payload.get("bioemu_x0_sidecar_shared_q_oracle_distill_scale", 256.0)
            ),
            bioemu_x0_sidecar_shared_q_oracle_distill_energy_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_distill_energy_family_weights",
                    {},
                )
            ),
            bioemu_x0_external_global_family_teacher_enabled=bool(
                payload.get("bioemu_x0_external_global_family_teacher_enabled", False)
            ),
            bioemu_x0_external_global_family_teacher_summary_path=str(
                payload.get("bioemu_x0_external_global_family_teacher_summary_path", "")
            ),
            bioemu_x0_external_global_family_teacher_weights_path=str(
                payload.get("bioemu_x0_external_global_family_teacher_weights_path", "")
            ),
            bioemu_x0_external_global_family_teacher_predictions_path=str(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_predictions_path",
                    "",
                )
            ),
            bioemu_x0_external_global_family_teacher_registry_latest_path=str(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_registry_latest_path",
                    "",
                )
            ),
            bioemu_x0_external_global_family_teacher_reload_at_epoch_boundary=bool(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_reload_at_epoch_boundary",
                    False,
                )
            ),
            bioemu_x0_external_global_family_teacher_min_ess_mean=float(
                payload.get("bioemu_x0_external_global_family_teacher_min_ess_mean", 0.0)
            ),
            bioemu_x0_external_global_family_teacher_max_top_mass=float(
                payload.get("bioemu_x0_external_global_family_teacher_max_top_mass", 1.0)
            ),
            bioemu_x0_external_global_family_teacher_min_ucbshift_pass_rate=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_min_ucbshift_pass_rate",
                    0.999,
                )
            ),
            bioemu_x0_external_global_family_teacher_distill_weight=float(
                payload.get("bioemu_x0_external_global_family_teacher_distill_weight", 0.0)
            ),
            bioemu_x0_external_global_family_teacher_energy_target_weight=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_energy_target_weight",
                    0.0,
                )
            ),
            bioemu_x0_external_global_family_teacher_contrastive_energy_weight=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_contrastive_energy_weight",
                    0.0,
                )
            ),
            bioemu_x0_external_global_family_teacher_contrastive_margin=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_contrastive_margin",
                    0.04,
                )
            ),
            bioemu_x0_external_global_family_teacher_contrastive_hard_negative_fraction=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_contrastive_hard_negative_fraction",
                    0.02,
                )
            ),
            bioemu_x0_external_global_family_teacher_contrastive_positive_mass_weight=float(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_contrastive_positive_mass_weight",
                    0.0,
                )
            ),
            bioemu_x0_external_global_family_teacher_family_weights=dict(
                payload.get(
                    "bioemu_x0_external_global_family_teacher_family_weights",
                    {},
                )
            ),
            bioemu_x0_external_global_family_teacher_require_required_family_complete=bool(
                payload.get(
                    (
                        "bioemu_x0_external_global_family_teacher_"
                        "require_required_family_complete"
                    ),
                    False,
                )
            ),
            bioemu_x0_external_global_family_teacher_coverage_weight_power=float(
                payload.get(
                    (
                        "bioemu_x0_external_global_family_teacher_"
                        "coverage_weight_power"
                    ),
                    0.0,
                )
            ),
            bioemu_x0_external_global_family_teacher_allow_diagnostic_active_subset_probe=bool(
                payload.get(
                    (
                        "bioemu_x0_external_global_family_teacher_"
                        "allow_diagnostic_active_subset_probe"
                    ),
                    False,
                )
            ),
            bioemu_x0_external_global_family_teacher_require_paper_cs_reweighting_lineage=bool(
                payload.get(
                    (
                        "bioemu_x0_external_global_family_teacher_"
                        "require_paper_cs_reweighting_lineage"
                    ),
                    False,
                )
            ),
            bioemu_x0_external_global_family_teacher_assert_vanilla_source_index_identity=bool(
                payload.get(
                    (
                        "bioemu_x0_external_global_family_teacher_"
                        "assert_vanilla_source_index_identity"
                    ),
                    False,
                )
            ),
            bioemu_x0_no_oracle_response_envelope_gate_required=bool(
                payload.get("bioemu_x0_no_oracle_response_envelope_gate_required", False)
            ),
            bioemu_x0_no_oracle_response_envelope_gate_metric=str(
                payload.get("bioemu_x0_no_oracle_response_envelope_gate_metric", "")
                or ""
            ),
            bioemu_x0_no_oracle_response_envelope_focus_families=list(
                payload.get("bioemu_x0_no_oracle_response_envelope_focus_families", [])
                or []
            ),
            bioemu_x0_no_oracle_response_envelope_baseline_family_ccc=dict(
                payload.get(
                    "bioemu_x0_no_oracle_response_envelope_baseline_family_ccc",
                    {},
                )
                or {}
            ),
            bioemu_x0_no_oracle_response_envelope_gap_to_target=dict(
                payload.get(
                    "bioemu_x0_no_oracle_response_envelope_gap_to_target",
                    {},
                )
                or {}
            ),
            bioemu_x0_no_oracle_response_envelope_required_family_ccc=dict(
                payload.get(
                    "bioemu_x0_no_oracle_response_envelope_required_family_ccc",
                    {},
                )
                or {}
            ),
            bioemu_x0_no_oracle_response_envelope_min_improvement=float(
                payload.get("bioemu_x0_no_oracle_response_envelope_min_improvement", 0.0)
            ),
            bioemu_x0_block_new_ucbshift_until_response_envelope_gate=bool(
                payload.get(
                    "bioemu_x0_block_new_ucbshift_until_response_envelope_gate",
                    False,
                )
            ),
            bioemu_x0_block_new_ucbshift_reason=str(
                payload.get("bioemu_x0_block_new_ucbshift_reason", "") or ""
            ),
            bioemu_x0_sidecar_shared_q_oracle_solver=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_solver",
                    "energy_softmax",
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_distill_oracle_solver=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_distill_oracle_solver",
                    "",
                )
                or ""
            ),
            bioemu_x0_sidecar_shared_q_energy_target_oracle_solver=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_oracle_solver",
                    "",
                )
                or ""
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_steps=int(
                payload.get("bioemu_x0_sidecar_shared_q_oracle_bme_steps", 64)
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_learning_rate=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_learning_rate",
                    0.20,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_kl_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_kl_weight",
                    0.01,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_likelihood_kind=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_likelihood_kind",
                    "gaussian",
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_student_df=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_student_df",
                    4.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_outlier_prob=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_outlier_prob",
                    0.05,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_bad_scale=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_good_bad_bad_scale",
                    8.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_l2=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_l2",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_learning_rate=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_learning_rate",
                    0.05,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_max_abs_ppm=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_reference_offset_max_abs_ppm",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_restart_count=int(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_restart_count",
                    0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_restart_noise=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_restart_noise",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_restart_seed=int(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_restart_seed",
                    0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_energy_warmstart_scales=[
                float(value)
                for value in payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_energy_warmstart_scales",
                    [],
                )
            ],
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_floor=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_floor",
                    0.95,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_min_points=int(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_min_points",
                    3,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_family_weight_mode=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_family_weight_mode",
                    "cell",
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_learning_rate=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_learning_rate",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_max_multiplier=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_max_multiplier",
                    1.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_update_every=int(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_update_every",
                    1,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_warmup_steps=int(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_warmup_steps",
                    0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_loss_kind=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ccc_dual_gap_loss_kind",
                    "quadratic",
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_selection_metric=str(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_selection_metric",
                    "objective",
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ess_floor=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ess_floor",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_ess_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_ess_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_cap=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_cap",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_oracle_bme_top_mass_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_scale=float(
                payload.get("bioemu_x0_sidecar_shared_q_energy_target_scale", 256.0)
            ),
            bioemu_x0_sidecar_shared_q_energy_target_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_family_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_clip=float(
                payload.get("bioemu_x0_sidecar_shared_q_energy_target_clip", 6.0)
            ),
            bioemu_x0_sidecar_shared_q_energy_target_huber_beta=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_huber_beta",
                    0.5,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_std_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_std_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_shape_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_shape_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_rank_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_rank_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_corr_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_corr_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_std_huber_beta=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_std_huber_beta",
                    0.25,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_shape_huber_beta=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_shape_huber_beta",
                    0.5,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_rank_margin=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_rank_margin",
                    0.25,
                )
            ),
            bioemu_x0_sidecar_shared_q_energy_target_rank_top_fraction=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_energy_target_rank_top_fraction",
                    0.10,
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_pair_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_pair_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_rank_top_fraction=float(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_rank_top_fraction",
                    0.10,
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_rank_margin=float(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_rank_margin",
                    0.15,
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_imbalance_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_imbalance_weight",
                    0.25,
                )
            ),
            bioemu_x0_sidecar_pair_rank_energy_min_cells_per_family=int(
                payload.get(
                    "bioemu_x0_sidecar_pair_rank_energy_min_cells_per_family",
                    1,
                )
            ),
            bioemu_x0_response_ceiling_pair_manifest_path=str(
                payload.get("bioemu_x0_response_ceiling_pair_manifest_path", "")
                or ""
            ),
            bioemu_x0_response_ceiling_pair_manifest_summary_path=str(
                payload.get(
                    "bioemu_x0_response_ceiling_pair_manifest_summary_path",
                    "",
                )
                or ""
            ),
            bioemu_x0_response_ceiling_pair_manifest_pair_count=int(
                payload.get("bioemu_x0_response_ceiling_pair_manifest_pair_count", 0)
            ),
            bioemu_x0_response_ceiling_pair_manifest_pair_count_by_family={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x0_response_ceiling_pair_manifest_pair_count_by_family",
                        {},
                    )
                    or {}
                ).items()
            },
            bioemu_x0_response_ceiling_pair_manifest_selection_policy=str(
                payload.get(
                    "bioemu_x0_response_ceiling_pair_manifest_selection_policy",
                    "",
                )
                or ""
            ),
            bioemu_x0_response_ceiling_pair_manifest_teacher_probe_only=bool(
                payload.get(
                    "bioemu_x0_response_ceiling_pair_manifest_teacher_probe_only",
                    True,
                )
            ),
            bioemu_x0_response_ceiling_pair_manifest_final_acceptance_eligible=bool(
                payload.get(
                    "bioemu_x0_response_ceiling_pair_manifest_final_acceptance_eligible",
                    False,
                )
            ),
            bioemu_x0_response_ceiling_pair_manifest_recommended_use=str(
                payload.get(
                    "bioemu_x0_response_ceiling_pair_manifest_recommended_use",
                    "",
                )
                or ""
            ),
            bioemu_x0_failed_active_face_replay_bundle_path=str(
                payload.get("bioemu_x0_failed_active_face_replay_bundle_path", "")
                or ""
            ),
            bioemu_x0_failed_active_face_replay_summary_path=str(
                payload.get("bioemu_x0_failed_active_face_replay_summary_path", "")
                or ""
            ),
            bioemu_x0_failed_active_face_replay_row_count=int(
                payload.get("bioemu_x0_failed_active_face_replay_row_count", 0)
            ),
            bioemu_x0_failed_active_face_replay_row_count_by_family={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x0_failed_active_face_replay_row_count_by_family",
                        {},
                    )
                    or {}
                ).items()
            },
            bioemu_x0_failed_active_face_replay_response_ceiling_pair_attached_rows=int(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_response_ceiling_pair_attached_rows",
                    0,
                )
            ),
            bioemu_x0_failed_active_face_replay_mean_dual_multiplier=float(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_mean_dual_multiplier",
                    0.0,
                )
            ),
            bioemu_x0_failed_active_face_replay_max_dual_multiplier=float(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_max_dual_multiplier",
                    0.0,
                )
            ),
            bioemu_x0_failed_active_face_replay_phi_theta_supervision_heads=list(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_phi_theta_supervision_heads",
                    [],
                )
                or []
            ),
            bioemu_x0_failed_active_face_replay_teacher_probe_only=bool(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_teacher_probe_only",
                    True,
                )
            ),
            bioemu_x0_failed_active_face_replay_final_acceptance_eligible=bool(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_final_acceptance_eligible",
                    False,
                )
            ),
            bioemu_x0_failed_active_face_replay_use_for_latest_accepted_teacher=bool(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_use_for_latest_accepted_teacher",
                    False,
                )
            ),
            bioemu_x0_failed_active_face_replay_must_not_update_latest_accepted=bool(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_must_not_update_latest_accepted",
                    True,
                )
            ),
            bioemu_x0_failed_active_face_replay_recommended_use=str(
                payload.get(
                    "bioemu_x0_failed_active_face_replay_recommended_use",
                    "",
                )
                or ""
            ),
            bioemu_x0_sidecar_evidence_likelihood_energy_scale=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_energy_scale",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_family_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_cell_count_power=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_cell_count_power",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_kind=str(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_kind",
                    "gaussian",
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_student_df=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_student_df",
                    4.0,
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_good_bad_outlier_prob=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_good_bad_outlier_prob",
                    0.05,
                )
            ),
            bioemu_x0_sidecar_evidence_likelihood_good_bad_bad_scale=float(
                payload.get(
                    "bioemu_x0_sidecar_evidence_likelihood_good_bad_bad_scale",
                    8.0,
                )
            ),
            bioemu_x0_evidence_likelihood_cell_count_power=float(
                payload.get(
                    "bioemu_x0_evidence_likelihood_cell_count_power",
                    0.0,
                )
            ),
            bioemu_x0_evidence_likelihood_base_energy_scale=float(
                payload.get(
                    "bioemu_x0_evidence_likelihood_base_energy_scale",
                    1.0,
                )
            ),
            bioemu_x0_sidecar_weighted_observable_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_weighted_observable_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_family_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_weighted_observable_ccc_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_ccc_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_weighted_observable_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_weighted_observable_ccc_floors=dict(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_ccc_floors",
                    {},
                )
            ),
            bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floors=dict(
                payload.get(
                    "bioemu_x0_sidecar_weighted_observable_nonregression_ccc_floors",
                    {},
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_ccc_floors=dict(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_ccc_floors",
                    {},
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_family_weights=dict(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_family_weights",
                    {},
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_macro_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_shared_q_witness_use_gradient_projection=bool(
                payload.get(
                    "bioemu_x0_sidecar_shared_q_witness_use_gradient_projection",
                    True,
                )
            ),
            bioemu_x0_posterior_mode_count=int(
                payload.get("bioemu_x0_posterior_mode_count", 1)
            ),
            bioemu_x0_posterior_mode_energy_scale=float(
                payload.get("bioemu_x0_posterior_mode_energy_scale", 0.0)
            ),
            bioemu_x0_posterior_mode_energy_init_std=float(
                payload.get("bioemu_x0_posterior_mode_energy_init_std", 0.0)
            ),
            bioemu_x0_posterior_mode_logit_scale=float(
                payload.get("bioemu_x0_posterior_mode_logit_scale", 1.0)
            ),
            bioemu_x0_posterior_mode_logit_temperature=float(
                payload.get("bioemu_x0_posterior_mode_logit_temperature", 1.0)
            ),
            bioemu_x0_posterior_mode_logit_prior=[
                float(value)
                for value in list(
                    payload.get("bioemu_x0_posterior_mode_logit_prior", [])
                )
            ],
            bioemu_x0_posterior_mode_reinit_std_on_load=float(
                payload.get("bioemu_x0_posterior_mode_reinit_std_on_load", 0.0)
            ),
            bioemu_x0_posterior_mode_reinit_reset_logits=bool(
                payload.get("bioemu_x0_posterior_mode_reinit_reset_logits", False)
            ),
            bioemu_x0_posterior_mode_family_oracle_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_oracle_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_oracle_scale=float(
                payload.get("bioemu_x0_posterior_mode_family_oracle_scale", 256.0)
            ),
            bioemu_x0_posterior_mode_family_oracle_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_mode_family_oracle_family_weights",
                    {},
                )
            ),
            bioemu_x0_posterior_mode_family_oracle_map={
                str(key): int(value)
                for key, value in dict(
                    payload.get("bioemu_x0_posterior_mode_family_oracle_map", {})
                ).items()
            },
            bioemu_x0_posterior_mode_family_energy_target_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_scale=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_scale",
                    256.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_family_weights",
                    {},
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_map={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x0_posterior_mode_family_energy_target_map",
                        {},
                    )
                ).items()
            },
            bioemu_x0_posterior_mode_family_energy_target_clip=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_clip",
                    6.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_huber_beta=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_huber_beta",
                    0.5,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_std_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_std_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_corr_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_corr_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_rank_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_rank_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_rank_margin=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_rank_margin",
                    0.05,
                )
            ),
            bioemu_x0_posterior_mode_family_energy_target_rank_top_fraction=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_energy_target_rank_top_fraction",
                    0.10,
                )
            ),
            bioemu_x0_posterior_mode_diversity_loss_weight=float(
                payload.get("bioemu_x0_posterior_mode_diversity_loss_weight", 0.0)
            ),
            bioemu_x0_posterior_mode_diversity_min_js=float(
                payload.get("bioemu_x0_posterior_mode_diversity_min_js", 0.01)
            ),
            bioemu_x0_posterior_mode_family_prediction_contrast_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_prediction_contrast_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_prediction_contrast_temperature=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_prediction_contrast_temperature",
                    0.25,
                )
            ),
            bioemu_x0_posterior_mode_family_prediction_contrast_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_mode_family_prediction_contrast_family_weights",
                    {},
                )
            ),
            bioemu_x0_posterior_mode_family_prediction_contrast_map={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x0_posterior_mode_family_prediction_contrast_map",
                        {},
                    )
                ).items()
            },
            bioemu_x0_posterior_mode_family_mixture_teacher_loss_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_scale=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_scale",
                    256.0,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_mode_sharpness=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_mode_sharpness",
                    1.0,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_mode_kl_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_mode_kl_weight",
                    1.0,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_component_kl_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_component_kl_weight",
                    1.0,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_mixture_kl_weight=float(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_mixture_kl_weight",
                    0.5,
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_mode_family_mixture_teacher_family_weights",
                    {},
                )
            ),
            bioemu_x0_posterior_mode_family_mixture_teacher_map={
                str(key): int(value)
                for key, value in dict(
                    payload.get(
                        "bioemu_x0_posterior_mode_family_mixture_teacher_map",
                        {},
                    )
                ).items()
            },
            bioemu_x0_family_mode_observable_enabled=bool(
                payload.get("bioemu_x0_family_mode_observable_enabled", False)
            ),
            bioemu_x0_family_mode_observable_blend=float(
                payload.get("bioemu_x0_family_mode_observable_blend", 0.0)
            ),
            bioemu_x0_family_mode_observable_map={
                str(key): int(value)
                for key, value in dict(
                    payload.get("bioemu_x0_family_mode_observable_map", {})
                ).items()
            },
            bioemu_x0_residue_local_mode_observable_enabled=bool(
                payload.get("bioemu_x0_residue_local_mode_observable_enabled", False)
            ),
            bioemu_x0_residue_local_mode_observable_blend=float(
                payload.get("bioemu_x0_residue_local_mode_observable_blend", 0.0)
            ),
            bioemu_x0_residue_local_mode_gate_hidden_dim=int(
                payload.get("bioemu_x0_residue_local_mode_gate_hidden_dim", 0)
            ),
            bioemu_x0_residue_local_mode_gate_temperature=float(
                payload.get("bioemu_x0_residue_local_mode_gate_temperature", 1.0)
            ),
            bioemu_x0_residue_local_mode_gate_teacher_loss_weight=float(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_scale=float(
                payload.get("bioemu_x0_residue_local_mode_gate_teacher_scale", 256.0)
            ),
            bioemu_x0_residue_local_mode_gate_teacher_hard_loss_weight=float(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_hard_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_entropy_loss_weight=float(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_entropy_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_entropy_target=float(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_entropy_target",
                    0.0,
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_family_weights=dict(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_family_weights",
                    {},
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_family_scales=dict(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_family_scales",
                    {},
                )
            ),
            bioemu_x0_residue_local_mode_gate_teacher_use_sidecar=bool(
                payload.get(
                    "bioemu_x0_residue_local_mode_gate_teacher_use_sidecar",
                    True,
                )
            ),
            bioemu_x0_sidecar_family_mode_observable_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_family_mode_observable_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_family_mode_observable_ccc_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_family_mode_observable_ccc_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_sidecar_family_mode_observable_ccc_floor_loss_weight=float(
                payload.get(
                    "bioemu_x0_sidecar_family_mode_observable_ccc_floor_loss_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_oracle_distill_weight=float(
                payload.get("bioemu_x0_posterior_oracle_distill_weight", 0.0)
            ),
            bioemu_x0_posterior_oracle_distill_scale=float(
                payload.get("bioemu_x0_posterior_oracle_distill_scale", 256.0)
            ),
            bioemu_x0_posterior_oracle_distill_energy_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_oracle_distill_energy_family_weights",
                    {},
                )
            ),
            bioemu_x0_posterior_consensus_oracle_distill_weight=float(
                payload.get(
                    "bioemu_x0_posterior_consensus_oracle_distill_weight",
                    0.0,
                )
            ),
            bioemu_x0_posterior_consensus_oracle_distill_scale=float(
                payload.get(
                    "bioemu_x0_posterior_consensus_oracle_distill_scale",
                    256.0,
                )
            ),
            bioemu_x0_posterior_consensus_oracle_conflict_kl_threshold=float(
                payload.get(
                    "bioemu_x0_posterior_consensus_oracle_conflict_kl_threshold",
                    0.75,
                )
            ),
            bioemu_x0_posterior_consensus_oracle_family_weights=dict(
                payload.get(
                    "bioemu_x0_posterior_consensus_oracle_family_weights",
                    {},
                )
            ),
            bioemu_x0_evidence_conflict_gate_enabled=bool(
                payload.get("bioemu_x0_evidence_conflict_gate_enabled", False)
            ),
            bioemu_x0_evidence_conflict_gate_kl_threshold=float(
                payload.get("bioemu_x0_evidence_conflict_gate_kl_threshold", 0.75)
            ),
            bioemu_x0_evidence_conflict_gate_min_scale=float(
                payload.get("bioemu_x0_evidence_conflict_gate_min_scale", 0.25)
            ),
            bioemu_x0_evidence_likelihood_energy_scale=float(
                payload.get("bioemu_x0_evidence_likelihood_energy_scale", 0.0)
            ),
            bioemu_x0_evidence_likelihood_family_weights=dict(
                payload.get("bioemu_x0_evidence_likelihood_family_weights", {})
            ),
            bioemu_x0_hn_learned_evidence_energy_scale=float(
                payload.get("bioemu_x0_hn_learned_evidence_energy_scale", 0.0)
            ),
            bioemu_x0_cprime_learned_evidence_energy_scale=float(
                payload.get("bioemu_x0_cprime_learned_evidence_energy_scale", 0.0)
            ),
            bioemu_x0_learned_evidence_energy_family_scales=dict(
                payload.get("bioemu_x0_learned_evidence_energy_family_scales", {})
            ),
            bioemu_x0_posterior_energy_hidden_dim=int(
                payload.get("bioemu_x0_posterior_energy_hidden_dim", 256)
            ),
            bioemu_x0_posterior_energy_init_std=float(
                payload.get("bioemu_x0_posterior_energy_init_std", 0.0)
            ),
            bioemu_x0_posterior_energy_extra_hidden_dim=int(
                payload.get("bioemu_x0_posterior_energy_extra_hidden_dim", 0)
            ),
            bioemu_x0_posterior_energy_extra_depth=int(
                payload.get("bioemu_x0_posterior_energy_extra_depth", 2)
            ),
            bioemu_x0_posterior_energy_extra_scale=float(
                payload.get("bioemu_x0_posterior_energy_extra_scale", 1.0)
            ),
            bioemu_x0_posterior_energy_extra_init_std=float(
                payload.get("bioemu_x0_posterior_energy_extra_init_std", 0.0)
            ),
            bioemu_x0_posterior_energy_extra_factorized_rank=int(
                payload.get(
                    "bioemu_x0_posterior_energy_extra_factorized_rank",
                    0,
                )
            ),
            bioemu_x0_posterior_energy_extra_effective_hidden_dim=int(
                payload.get(
                    "bioemu_x0_posterior_energy_extra_effective_hidden_dim",
                    0,
                )
            ),
            bioemu_x0_posterior_energy_extra_effective_factorized_rank=int(
                payload.get(
                    "bioemu_x0_posterior_energy_extra_effective_factorized_rank",
                    0,
                )
            ),
            bioemu_x0_posterior_energy_extra_max_factorized_params=int(
                payload.get(
                    "bioemu_x0_posterior_energy_extra_max_factorized_params",
                    450_000_000,
                )
            ),
            bioemu_x0_posterior_energy_extra_shared_block_count=int(
                payload.get(
                    "bioemu_x0_posterior_energy_extra_shared_block_count",
                    0,
                )
            ),
            bioemu_x0_posterior_energy_extra_chunk_size=int(
                payload.get("bioemu_x0_posterior_energy_extra_chunk_size", 0)
            ),
            bioemu_x0_posterior_energy_extra_use_checkpoint=bool(
                payload.get("bioemu_x0_posterior_energy_extra_use_checkpoint", True)
            ),
            bioemu_x0_posterior_energy_extra_lr_multiplier=float(
                payload.get("bioemu_x0_posterior_energy_extra_lr_multiplier", 1.0)
            ),
            bioemu_x0_posterior_energy_support_attention_hidden_dim=int(
                payload.get(
                    "bioemu_x0_posterior_energy_support_attention_hidden_dim",
                    0,
                )
            ),
            bioemu_x0_posterior_energy_support_attention_head_count=int(
                payload.get(
                    "bioemu_x0_posterior_energy_support_attention_head_count",
                    4,
                )
            ),
            bioemu_x0_posterior_energy_support_attention_depth=int(
                payload.get("bioemu_x0_posterior_energy_support_attention_depth", 1)
            ),
            bioemu_x0_posterior_energy_support_attention_scale=float(
                payload.get("bioemu_x0_posterior_energy_support_attention_scale", 0.0)
            ),
            bioemu_x0_posterior_energy_support_attention_init_std=float(
                payload.get(
                    "bioemu_x0_posterior_energy_support_attention_init_std",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_hidden_dim=int(
                payload.get("bioemu_x0_cs_latent_field_conditioning_hidden_dim", 0)
            ),
            bioemu_x0_cs_latent_field_conditioning_depth=int(
                payload.get("bioemu_x0_cs_latent_field_conditioning_depth", 1)
            ),
            bioemu_x0_cs_latent_field_conditioning_residue_scale=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_residue_scale",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_scale=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_scale",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_pooling=str(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_pooling",
                    "mean",
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_attention_temperature=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_attention_temperature",
                    1.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_attention_init_std=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_attention_init_std",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_salience_attention_weight=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_salience_attention_weight",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_std_floor=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_std_floor",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_energy_std_floor_scale_cap=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_energy_std_floor_scale_cap",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_family_token_scale=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_family_token_scale",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_family_token_temperature=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_family_token_temperature",
                    1.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_family_token_init_std=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_family_token_init_std",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_film_scale=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_film_scale",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_film_init_std=float(
                payload.get(
                    "bioemu_x0_cs_latent_field_conditioning_film_init_std",
                    0.0,
                )
            ),
            bioemu_x0_cs_latent_field_conditioning_dropout=float(
                payload.get("bioemu_x0_cs_latent_field_conditioning_dropout", 0.0)
            ),
            bioemu_x0_cs_latent_field_conditioning_init_std=float(
                payload.get("bioemu_x0_cs_latent_field_conditioning_init_std", 0.0)
            ),
            bioemu_x0_sequence_embedding_residue_feature_dim=int(
                payload.get("bioemu_x0_sequence_embedding_residue_feature_dim", 0)
            ),
            bioemu_x0_require_sequence_embedding_active_metric=bool(
                payload.get("bioemu_x0_require_sequence_embedding_active_metric", False)
            ),
            bioemu_x0_static_context_scale=float(
                payload.get("bioemu_x0_static_context_scale", 1.0)
            ),
            bioemu_x0_mechanism_context_scale=float(
                payload.get("bioemu_x0_mechanism_context_scale", 1.0)
            ),
            bioemu_x0_posterior_energy_temperature=float(
                payload.get("bioemu_x0_posterior_energy_temperature", 1.0)
            ),
            bioemu_x0_bounded_delta_max_abs=float(
                payload.get("bioemu_x0_bounded_delta_max_abs", 0.0)
            ),
            bioemu_x0_bounded_delta_penalty_weight=float(
                payload.get("bioemu_x0_bounded_delta_penalty_weight", 1.0)
            ),
            bioemu_x0_posterior_prior_kl_weight=float(
                payload.get("bioemu_x0_posterior_prior_kl_weight", 0.01)
            ),
            bioemu_x0_posterior_ess_floor=float(
                payload.get("bioemu_x0_posterior_ess_floor", 32.0)
            ),
            bioemu_x0_posterior_entropy_floor=float(
                payload.get("bioemu_x0_posterior_entropy_floor", 3.0)
            ),
            bioemu_x0_posterior_top_mass_cap=float(
                payload.get("bioemu_x0_posterior_top_mass_cap", 0.10)
            ),
            bioemu_x0_posterior_diversity_floor=float(
                payload.get("bioemu_x0_posterior_diversity_floor", 0.0)
            ),
            bioemu_x0_posterior_diversity_weight=float(
                payload.get("bioemu_x0_posterior_diversity_weight", 0.0)
            ),
            bioemu_x0_mechanism_edge_classes=list(
                payload.get(
                    "bioemu_x0_mechanism_edge_classes",
                    [
                        "sequence",
                        "peptide_plane",
                        "spatial_contact",
                        "hbond",
                        "ring",
                        "electrostatic",
                    ],
                )
            ),
            bioemu_x0_hn_mechanism_channels=list(
                payload.get(
                    "bioemu_x0_hn_mechanism_channels",
                    [
                        "exchange_hbond",
                        "ring_orientation",
                        "electrostatic",
                        "terminal_disorder",
                    ],
                )
            ),
            bioemu_x0_cprime_mechanism_channels=list(
                payload.get(
                    "bioemu_x0_cprime_mechanism_channels",
                    [
                        "peptide_plane",
                        "carbonyl_backbone",
                        "carbonyl_hbond",
                        "local_strain",
                    ],
                )
            ),
            enable_bioemu_posterior_moment_diffusion=bool(
                payload.get("enable_bioemu_posterior_moment_diffusion", False)
            ),
            bioemu_posterior_inference_mode=str(
                payload.get("bioemu_posterior_inference_mode", "sample_reweighting")
            ),
            bioemu_posterior_chart_count=int(
                payload.get("bioemu_posterior_chart_count", 8)
            ),
            bioemu_posterior_covariance_mode=str(
                payload.get("bioemu_posterior_covariance_mode", "diag_low_rank")
            ),
            bioemu_posterior_covariance_rank=int(
                payload.get("bioemu_posterior_covariance_rank", 8)
            ),
            bioemu_posterior_covariance_floor=float(
                payload.get("bioemu_posterior_covariance_floor", 1.0e-4)
            ),
            bioemu_posterior_moment_shift_mode=str(
                payload.get(
                    "bioemu_posterior_moment_shift_mode",
                    "curvature_trace_proxy",
                )
            ),
            bioemu_posterior_sigma_point_count=int(
                payload.get("bioemu_posterior_sigma_point_count", 0)
            ),
            bioemu_posterior_diffusion_steps_train=int(
                payload.get("bioemu_posterior_diffusion_steps_train", 1)
            ),
            bioemu_posterior_diffusion_steps_val=int(
                payload.get("bioemu_posterior_diffusion_steps_val", 1)
            ),
            bioemu_posterior_flow_teacher_steps=int(
                payload.get("bioemu_posterior_flow_teacher_steps", 4)
            ),
            bioemu_posterior_enable_consistency_distillation=bool(
                payload.get("bioemu_posterior_enable_consistency_distillation", False)
            ),
            bioemu_posterior_likelihood_score_loss_weight=float(
                payload.get("bioemu_posterior_likelihood_score_loss_weight", 0.0)
            ),
            bioemu_posterior_moment_nll_weight=float(
                payload.get("bioemu_posterior_moment_nll_weight", 1.0)
            ),
            bioemu_posterior_covariance_calibration_weight=float(
                payload.get("bioemu_posterior_covariance_calibration_weight", 0.0)
            ),
            bioemu_posterior_covariance_floor_loss_weight=float(
                payload.get("bioemu_posterior_covariance_floor_loss_weight", 0.0)
            ),
            bioemu_posterior_chart_entropy_floor=float(
                payload.get("bioemu_posterior_chart_entropy_floor", 0.05)
            ),
            bioemu_posterior_chart_entropy_loss_weight=float(
                payload.get("bioemu_posterior_chart_entropy_loss_weight", 0.0)
            ),
            bioemu_posterior_chart_entropy_row_loss_weight=float(
                payload.get("bioemu_posterior_chart_entropy_row_loss_weight", 0.0)
            ),
            bioemu_posterior_chart_entropy_row_floor_by_family={
                str(key): float(value)
                for key, value in payload.get(
                    "bioemu_posterior_chart_entropy_row_floor_by_family", {}
                ).items()
            },
            bioemu_posterior_chart_entropy_row_weight_by_family={
                str(key): float(value)
                for key, value in payload.get(
                    "bioemu_posterior_chart_entropy_row_weight_by_family", {}
                ).items()
            },
            bioemu_posterior_ess_floor=float(
                payload.get("bioemu_posterior_ess_floor", 0.0)
            ),
            bioemu_posterior_ess_loss_weight=float(
                payload.get("bioemu_posterior_ess_loss_weight", 0.0)
            ),
            bioemu_checkpoint_support_safe_enabled=bool(
                payload.get("bioemu_checkpoint_support_safe_enabled", False)
            ),
            bioemu_checkpoint_support_safe_ess_floor=float(
                payload.get("bioemu_checkpoint_support_safe_ess_floor", 0.0)
            ),
            bioemu_checkpoint_support_safe_entropy_floor=float(
                payload.get("bioemu_checkpoint_support_safe_entropy_floor", 0.0)
            ),
            bioemu_checkpoint_support_safe_ess_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_ess_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_entropy_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_entropy_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_hn_abs_residual_p95_ceiling=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_hn_abs_residual_p95_ceiling",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_hn_abs_residual_p95_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_hn_abs_residual_p95_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_hn_abs_residual_max_ceiling=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_hn_abs_residual_max_ceiling",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_hn_abs_residual_max_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_hn_abs_residual_max_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_ccc_floor=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_ccc_floor",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_ccc_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_ccc_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_abs_residual_p95_ceiling=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_abs_residual_p95_ceiling",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_abs_residual_p95_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_abs_residual_p95_penalty_weight",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_abs_residual_max_ceiling=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_abs_residual_max_ceiling",
                    0.0,
                )
            ),
            bioemu_checkpoint_support_safe_cprime_abs_residual_max_penalty_weight=float(
                payload.get(
                    "bioemu_checkpoint_support_safe_cprime_abs_residual_max_penalty_weight",
                    0.0,
                )
            ),
            bioemu_posterior_occupancy_trust_region_enabled=bool(
                payload.get(
                    "bioemu_posterior_occupancy_trust_region_enabled",
                    False,
                )
            ),
            bioemu_posterior_occupancy_trust_region_weight=float(
                payload.get(
                    "bioemu_posterior_occupancy_trust_region_weight",
                    0.0,
                )
            ),
            bioemu_posterior_occupancy_trust_region_ess_floor=float(
                payload.get(
                    "bioemu_posterior_occupancy_trust_region_ess_floor",
                    0.0,
                )
            ),
            bioemu_posterior_occupancy_trust_region_entropy_floor=float(
                payload.get(
                    "bioemu_posterior_occupancy_trust_region_entropy_floor",
                    0.0,
                )
            ),
            bioemu_posterior_occupancy_trust_region_deficit_multiplier=float(
                payload.get(
                    "bioemu_posterior_occupancy_trust_region_deficit_multiplier",
                    0.0,
                )
            ),
            bioemu_posterior_top_mass_cap_enabled=bool(
                payload.get("bioemu_posterior_top_mass_cap_enabled", False)
            ),
            bioemu_posterior_top_mass_cap=float(
                payload.get("bioemu_posterior_top_mass_cap", 1.0)
            ),
            bioemu_posterior_top_mass_cap_loss_weight=float(
                payload.get("bioemu_posterior_top_mass_cap_loss_weight", 0.0)
            ),
            bioemu_posterior_chart_temperature_enabled=bool(
                payload.get("bioemu_posterior_chart_temperature_enabled", False)
            ),
            bioemu_posterior_chart_temperature=float(
                payload.get("bioemu_posterior_chart_temperature", 1.0)
            ),
            bioemu_low_support_loss_freeze_enabled=bool(
                payload.get("bioemu_low_support_loss_freeze_enabled", False)
            ),
            bioemu_low_support_loss_freeze_ess_floor=float(
                payload.get("bioemu_low_support_loss_freeze_ess_floor", 0.0)
            ),
            bioemu_low_support_loss_freeze_entropy_floor=float(
                payload.get("bioemu_low_support_loss_freeze_entropy_floor", 0.0)
            ),
            bioemu_low_support_loss_freeze_min_scale=float(
                payload.get("bioemu_low_support_loss_freeze_min_scale", 1.0)
            ),
            bioemu_low_support_loss_freeze_power=float(
                payload.get("bioemu_low_support_loss_freeze_power", 2.0)
            ),
            bioemu_low_support_loss_freeze_hard_raw_threshold=float(
                payload.get(
                    "bioemu_low_support_loss_freeze_hard_raw_threshold",
                    0.0,
                )
            ),
            bioemu_low_support_loss_freeze_apply_to_measure_distill=bool(
                payload.get(
                    "bioemu_low_support_loss_freeze_apply_to_measure_distill",
                    True,
                )
            ),
            bioemu_low_support_loss_freeze_apply_to_mechanism_override=bool(
                payload.get(
                    "bioemu_low_support_loss_freeze_apply_to_mechanism_override",
                    True,
                )
            ),
            bioemu_posterior_prior_kl_weight=float(
                payload.get("bioemu_posterior_prior_kl_weight", 0.0)
            ),
            bioemu_posterior_bridge_norm_weight=float(
                payload.get("bioemu_posterior_bridge_norm_weight", 0.0)
            ),
            bioemu_posterior_tangent_drift_cap=float(
                payload.get("bioemu_posterior_tangent_drift_cap", 0.04)
            ),
            bioemu_posterior_chart_symmetry_break_scale=float(
                payload.get("bioemu_posterior_chart_symmetry_break_scale", 0.01)
            ),
            bioemu_hn_ccc_loss_weight=float(
                payload.get("bioemu_hn_ccc_loss_weight", 0.0)
            ),
            bioemu_hn_ccc_entity_family_weight_scale=float(
                payload.get("bioemu_hn_ccc_entity_family_weight_scale", 0.0)
            ),
            bioemu_hn_ccc_entity_family_weight_max=float(
                payload.get("bioemu_hn_ccc_entity_family_weight_max", 3.0)
            ),
            bioemu_hn_scale_loss_weight=float(
                payload.get("bioemu_hn_scale_loss_weight", 0.0)
            ),
            bioemu_hn_bias_loss_weight=float(
                payload.get("bioemu_hn_bias_loss_weight", 0.0)
            ),
            bioemu_cprime_scale_loss_weight=float(
                payload.get("bioemu_cprime_scale_loss_weight", 0.0)
            ),
            bioemu_cprime_bias_loss_weight=float(
                payload.get("bioemu_cprime_bias_loss_weight", 0.0)
            ),
            enable_bioemu_cprime_chart_bias_adapter=bool(
                payload.get("enable_bioemu_cprime_chart_bias_adapter", False)
            ),
            bioemu_cprime_chart_bias_max_abs=float(
                payload.get("bioemu_cprime_chart_bias_max_abs", 0.45)
            ),
            bioemu_cprime_chart_bias_family_names=list(
                payload.get("bioemu_cprime_chart_bias_family_names", ["C'"])
            ),
            bioemu_cprime_chart_bias_uncertainty_threshold=float(
                payload.get("bioemu_cprime_chart_bias_uncertainty_threshold", 0.65)
            ),
            bioemu_cprime_chart_bias_center_loss_weight=float(
                payload.get("bioemu_cprime_chart_bias_center_loss_weight", 0.0)
            ),
            bioemu_cprime_chart_bias_l2_loss_weight=float(
                payload.get("bioemu_cprime_chart_bias_l2_loss_weight", 0.0)
            ),
            bioemu_aux_family_ccc_loss_weight=float(
                payload.get("bioemu_aux_family_ccc_loss_weight", 0.0)
            ),
            bioemu_aux_family_ccc_names=list(
                payload.get("bioemu_aux_family_ccc_names", [])
            ),
            bioemu_family_ccc_target_loss_weight=float(
                payload.get("bioemu_family_ccc_target_loss_weight", 0.0)
            ),
            bioemu_family_ccc_target=float(
                payload.get("bioemu_family_ccc_target", 0.95)
            ),
            bioemu_family_ccc_target_names=list(
                payload.get(
                    "bioemu_family_ccc_target_names",
                    ["HN", "N", "CA", "CB", "C'"],
                )
            ),
            bioemu_family_ccc_target_min_rows=int(
                payload.get("bioemu_family_ccc_target_min_rows", 2)
            ),
            bioemu_family_ccc_target_start_epoch=int(
                payload.get("bioemu_family_ccc_target_start_epoch", 1)
            ),
            bioemu_family_ccc_target_ramp_epochs=int(
                payload.get("bioemu_family_ccc_target_ramp_epochs", 1)
            ),
            bioemu_family_orthogonal_loss_weight=float(
                payload.get("bioemu_family_orthogonal_loss_weight", 0.0)
            ),
            bioemu_family_orthogonal_target_names=list(
                payload.get("bioemu_family_orthogonal_target_names", ["HN", "C'"])
            ),
            bioemu_family_orthogonal_support_names=list(
                payload.get("bioemu_family_orthogonal_support_names", ["N", "CA"])
            ),
            bioemu_family_orthogonal_detach_support=bool(
                payload.get("bioemu_family_orthogonal_detach_support", True)
            ),
            bioemu_family_orthogonal_start_epoch=int(
                payload.get("bioemu_family_orthogonal_start_epoch", 1)
            ),
            bioemu_family_orthogonal_ramp_epochs=int(
                payload.get("bioemu_family_orthogonal_ramp_epochs", 1)
            ),
            enable_bioemu_physics_atlas_v36=bool(
                payload.get("enable_bioemu_physics_atlas_v36", False)
            ),
            bioemu_physics_atlas_coordinate_names=list(
                payload.get(
                    "bioemu_physics_atlas_coordinate_names",
                    [
                        "ring_orientation",
                        "ring_occupancy",
                        "exchange_hbond",
                        "carbonyl_backbone",
                        "peptide_plane_coupling",
                        "hn_local_manifold",
                        "cprime_carbonyl_manifold",
                        "ring_hbond_competition",
                        "solvent_exchange_risk",
                        "transfer_memory_confidence",
                        "transfer_support_mismatch",
                        "transfer_abstain_risk",
                        "electrostatic_sulfur",
                        "sidechain_rotamer",
                        "ensemble_component",
                        "uncertainty_abstain",
                    ],
                )
            ),
            bioemu_physics_sidecar_path=payload.get("bioemu_physics_sidecar_path"),
            rare_regime_atlas_path=payload.get("rare_regime_atlas_path"),
            rare_regime_atlas_auto_config=bool(
                payload.get("rare_regime_atlas_auto_config", False)
            ),
            rare_regime_atlas_auto_max_router_keys=int(
                payload.get("rare_regime_atlas_auto_max_router_keys", 12)
            ),
            rare_regime_atlas_auto_max_residue_family_keys=int(
                payload.get("rare_regime_atlas_auto_max_residue_family_keys", 10)
            ),
            rare_regime_atlas_auto_min_rows=int(
                payload.get("rare_regime_atlas_auto_min_rows", 8)
            ),
            regime_router_enabled=bool(
                payload.get("regime_router_enabled", True)
            ),
            mechanism_tangent_adapter_rank=int(
                payload.get("mechanism_tangent_adapter_rank", 8)
            ),
            bayesian_shrinkage_enabled=bool(
                payload.get("bayesian_shrinkage_enabled", True)
            ),
            direction_conflict_shared_adapter_threshold=float(
                payload.get("direction_conflict_shared_adapter_threshold", 0.08)
            ),
            teacher_mean_policy=str(
                payload.get("teacher_mean_policy", "guard_or_low_weight")
            ),
            enable_bioemu_family_tangent_adapter_bank=bool(
                payload.get("enable_bioemu_family_tangent_adapter_bank", False)
            ),
            bioemu_family_tangent_max_abs_by_family=dict(
                payload.get(
                    "bioemu_family_tangent_max_abs_by_family",
                    {
                        "HN": 0.04,
                        "N": 0.12,
                        "CA": 0.35,
                        "CB": 0.45,
                        "C'": 0.28,
                    },
                )
            ),
            bioemu_family_chart_names_by_family=dict(
                payload.get(
                    "bioemu_family_chart_names_by_family",
                    {
                        "HN": [
                            "ring_orientation",
                            "ring_occupancy",
                            "exchange_hbond",
                            "hn_local_manifold",
                            "ring_hbond_competition",
                            "solvent_exchange_risk",
                            "transfer_support_mismatch",
                            "transfer_abstain_risk",
                            "electrostatic_sulfur",
                        ],
                        "C'": [
                            "carbonyl_backbone",
                            "cprime_carbonyl_manifold",
                            "peptide_plane_coupling",
                            "exchange_hbond",
                            "transfer_support_mismatch",
                        ],
                        "N": [
                            "carbonyl_backbone",
                            "peptide_plane_coupling",
                            "hn_local_manifold",
                            "exchange_hbond",
                            "transfer_support_mismatch",
                            "transfer_abstain_risk",
                            "ensemble_component",
                        ],
                        "CA": [
                            "carbonyl_backbone",
                            "sidechain_rotamer",
                            "ensemble_component",
                        ],
                        "CB": [
                            "sidechain_rotamer",
                            "electrostatic_sulfur",
                            "ensemble_component",
                        ],
                    },
                )
            ),
            bioemu_family_gradient_conflict_loss_weight=float(
                payload.get("bioemu_family_gradient_conflict_loss_weight", 0.0)
            ),
            bioemu_family_tangent_orthogonality_loss_weight=float(
                payload.get("bioemu_family_tangent_orthogonality_loss_weight", 0.0)
            ),
            enable_bioemu_residue_family_rare_adapter=bool(
                payload.get("enable_bioemu_residue_family_rare_adapter", False)
            ),
            bioemu_residue_family_rare_adapter_keys=list(
                payload.get("bioemu_residue_family_rare_adapter_keys", [])
            ),
            bioemu_residue_family_rare_adapter_max_abs_by_family=dict(
                payload.get(
                    "bioemu_residue_family_rare_adapter_max_abs_by_family",
                    {
                        "HN": 0.08,
                        "N": 6.0,
                        "CA": 5.0,
                        "CB": 15.0,
                        "C'": 1.2,
                    },
                )
            ),
            bioemu_residue_family_rare_adapter_max_abs_by_pair=dict(
                payload.get("bioemu_residue_family_rare_adapter_max_abs_by_pair", {})
            ),
            bioemu_residue_family_rare_adapter_uncertainty_threshold=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_uncertainty_threshold",
                    0.75,
                )
            ),
            bioemu_residue_family_rare_adapter_uncertainty_threshold_by_pair=dict(
                payload.get(
                    "bioemu_residue_family_rare_adapter_uncertainty_threshold_by_pair",
                    {},
                )
            ),
            bioemu_residue_family_rare_adapter_loss_weight=float(
                payload.get("bioemu_residue_family_rare_adapter_loss_weight", 0.0)
            ),
            bioemu_residue_family_rare_adapter_delta_loss_weight=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_delta_loss_weight",
                    0.0,
                )
            ),
            bioemu_residue_family_rare_adapter_direction_loss_weight=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_direction_loss_weight",
                    0.0,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_loss_weight=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_loss_weight",
                    0.0,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_min_abs=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_min_abs",
                    0.006,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_max_abs=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_max_abs",
                    0.14,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_low_scale=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_low_scale",
                    0.70,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_high_scale=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_high_scale",
                    1.25,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_require_matrix_consensus=bool(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_require_matrix_consensus",
                    False,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_consensus_min=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_consensus_min",
                    0.82,
                )
            ),
            bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_min_valid_samples=int(
                payload.get(
                    "bioemu_residue_family_rare_adapter_hn_signed_delta_matrix_min_valid_samples",
                    32,
                )
            ),
            bioemu_residue_family_rare_adapter_delta_target_scale=float(
                payload.get(
                    "bioemu_residue_family_rare_adapter_delta_target_scale",
                    1.0,
                )
            ),
            bioemu_residue_family_rare_adapter_delta_target_max_by_family=dict(
                payload.get(
                    "bioemu_residue_family_rare_adapter_delta_target_max_by_family",
                    {"HN": 0.08, "N": 4.0, "CA": 3.0, "CB": 8.0, "C'": 0.8},
                )
            ),
            bioemu_residue_family_rare_adapter_start_epoch=int(
                payload.get("bioemu_residue_family_rare_adapter_start_epoch", 1)
            ),
            bioemu_residue_family_rare_adapter_ramp_epochs=int(
                payload.get("bioemu_residue_family_rare_adapter_ramp_epochs", 2)
            ),
            bioemu_train_only_residue_family_rare_adapter=bool(
                payload.get("bioemu_train_only_residue_family_rare_adapter", False)
            ),
            bioemu_residue_family_rare_adapter_train_keys=list(
                payload.get("bioemu_residue_family_rare_adapter_train_keys", [])
            ),
            bioemu_residue_family_rare_adapter_train_shared_heads=bool(
                payload.get(
                    "bioemu_residue_family_rare_adapter_train_shared_heads",
                    True,
                )
            ),
            enable_bioemu_factor_bin_rare_adapter=bool(
                payload.get("enable_bioemu_factor_bin_rare_adapter", False)
            ),
            bioemu_factor_bin_rare_adapter_coordinate_source=str(
                payload.get(
                    "bioemu_factor_bin_rare_adapter_coordinate_source",
                    "physics",
                )
            ),
            bioemu_factor_bin_rare_adapter_keys=list(
                payload.get("bioemu_factor_bin_rare_adapter_keys", [])
            ),
            bioemu_factor_bin_rare_adapter_max_abs_by_key=dict(
                payload.get("bioemu_factor_bin_rare_adapter_max_abs_by_key", {})
            ),
            bioemu_factor_bin_rare_adapter_center_by_key=dict(
                payload.get("bioemu_factor_bin_rare_adapter_center_by_key", {})
            ),
            bioemu_factor_bin_rare_adapter_width_by_key=dict(
                payload.get("bioemu_factor_bin_rare_adapter_width_by_key", {})
            ),
            bioemu_factor_bin_rare_adapter_uncertainty_threshold=float(
                payload.get("bioemu_factor_bin_rare_adapter_uncertainty_threshold", 0.85)
            ),
            bioemu_factor_bin_rare_adapter_uncertainty_threshold_by_key=dict(
                payload.get(
                    "bioemu_factor_bin_rare_adapter_uncertainty_threshold_by_key",
                    {},
                )
            ),
            bioemu_factor_bin_rare_adapter_loss_weight=float(
                payload.get("bioemu_factor_bin_rare_adapter_loss_weight", 0.0)
            ),
            bioemu_factor_bin_rare_adapter_start_epoch=int(
                payload.get("bioemu_factor_bin_rare_adapter_start_epoch", 1)
            ),
            bioemu_factor_bin_rare_adapter_ramp_epochs=int(
                payload.get("bioemu_factor_bin_rare_adapter_ramp_epochs", 2)
            ),
            bioemu_train_only_factor_bin_rare_adapter=bool(
                payload.get("bioemu_train_only_factor_bin_rare_adapter", False)
            ),
            bioemu_factor_bin_rare_adapter_train_keys=list(
                payload.get("bioemu_factor_bin_rare_adapter_train_keys", [])
            ),
            bioemu_factor_bin_rare_adapter_train_shared_heads=bool(
                payload.get("bioemu_factor_bin_rare_adapter_train_shared_heads", True)
            ),
            enable_bioemu_mechanism_signed_router=bool(
                payload.get("enable_bioemu_mechanism_signed_router", False)
            ),
            bioemu_mechanism_signed_router_coordinate_source=str(
                payload.get(
                    "bioemu_mechanism_signed_router_coordinate_source",
                    "physics",
                )
            ),
            bioemu_mechanism_signed_router_keys=list(
                payload.get("bioemu_mechanism_signed_router_keys", [])
            ),
            bioemu_mechanism_signed_router_direction_by_key=dict(
                payload.get("bioemu_mechanism_signed_router_direction_by_key", {})
            ),
            bioemu_mechanism_signed_router_max_abs_by_key=dict(
                payload.get("bioemu_mechanism_signed_router_max_abs_by_key", {})
            ),
            bioemu_mechanism_signed_router_center_by_key=dict(
                payload.get("bioemu_mechanism_signed_router_center_by_key", {})
            ),
            bioemu_mechanism_signed_router_width_by_key=dict(
                payload.get("bioemu_mechanism_signed_router_width_by_key", {})
            ),
            bioemu_mechanism_signed_router_uncertainty_threshold=float(
                payload.get(
                    "bioemu_mechanism_signed_router_uncertainty_threshold",
                    0.85,
                )
            ),
            bioemu_mechanism_signed_router_uncertainty_threshold_by_key=dict(
                payload.get(
                    "bioemu_mechanism_signed_router_uncertainty_threshold_by_key",
                    {},
                )
            ),
            bioemu_mechanism_signed_router_loss_weight=float(
                payload.get("bioemu_mechanism_signed_router_loss_weight", 0.0)
            ),
            bioemu_mechanism_signed_router_delta_loss_weight=float(
                payload.get(
                    "bioemu_mechanism_signed_router_delta_loss_weight",
                    0.0,
                )
            ),
            bioemu_mechanism_signed_router_delta_target_scale=float(
                payload.get(
                    "bioemu_mechanism_signed_router_delta_target_scale",
                    1.0,
                )
            ),
            bioemu_mechanism_signed_router_delta_target_max_by_family=dict(
                payload.get(
                    "bioemu_mechanism_signed_router_delta_target_max_by_family",
                    {},
                )
            ),
            enable_bioemu_mechanism_signed_cancellation=bool(
                payload.get("enable_bioemu_mechanism_signed_cancellation", False)
            ),
            bioemu_mechanism_signed_cancellation_keys=list(
                payload.get("bioemu_mechanism_signed_cancellation_keys", [])
            ),
            bioemu_mechanism_signed_cancellation_strength=float(
                payload.get("bioemu_mechanism_signed_cancellation_strength", 1.0)
            ),
            enable_bioemu_mechanism_signed_bias_conflict_gate=bool(
                payload.get(
                    "enable_bioemu_mechanism_signed_bias_conflict_gate",
                    False,
                )
            ),
            bioemu_mechanism_signed_bias_conflict_gate_keys=list(
                payload.get("bioemu_mechanism_signed_bias_conflict_gate_keys", [])
            ),
            bioemu_mechanism_signed_bias_conflict_gate_strength=float(
                payload.get(
                    "bioemu_mechanism_signed_bias_conflict_gate_strength",
                    0.0,
                )
            ),
            enable_bioemu_mechanism_signed_override_adapter=bool(
                payload.get(
                    "enable_bioemu_mechanism_signed_override_adapter",
                    False,
                )
            ),
            bioemu_mechanism_signed_override_keys=list(
                payload.get("bioemu_mechanism_signed_override_keys", [])
            ),
            bioemu_mechanism_signed_override_force_direction=bool(
                payload.get(
                    "bioemu_mechanism_signed_override_force_direction",
                    False,
                )
            ),
            bioemu_mechanism_signed_override_centered_component=bool(
                payload.get(
                    "bioemu_mechanism_signed_override_centered_component",
                    False,
                )
            ),
            bioemu_mechanism_signed_override_loss_weight=float(
                payload.get("bioemu_mechanism_signed_override_loss_weight", 0.0)
            ),
            bioemu_mechanism_signed_override_direction_loss_weight=float(
                payload.get(
                    "bioemu_mechanism_signed_override_direction_loss_weight",
                    0.0,
                )
            ),
            bioemu_mechanism_signed_override_support_confidence_enabled=bool(
                payload.get(
                    "bioemu_mechanism_signed_override_support_confidence_enabled",
                    False,
                )
            ),
            bioemu_mechanism_signed_override_support_confidence_ess_floor=float(
                payload.get(
                    "bioemu_mechanism_signed_override_support_confidence_ess_floor",
                    0.0,
                )
            ),
            bioemu_mechanism_signed_override_support_confidence_entropy_floor=float(
                payload.get(
                    "bioemu_mechanism_signed_override_support_confidence_entropy_floor",
                    0.0,
                )
            ),
            bioemu_mechanism_signed_override_support_confidence_min_scale=float(
                payload.get(
                    "bioemu_mechanism_signed_override_support_confidence_min_scale",
                    0.70,
                )
            ),
            bioemu_mechanism_signed_override_target_scale=float(
                payload.get("bioemu_mechanism_signed_override_target_scale", 1.0)
            ),
            bioemu_mechanism_signed_override_target_max_by_family=dict(
                payload.get(
                    "bioemu_mechanism_signed_override_target_max_by_family",
                    {},
                )
            ),
            bioemu_mechanism_signed_router_start_epoch=int(
                payload.get("bioemu_mechanism_signed_router_start_epoch", 1)
            ),
            bioemu_mechanism_signed_router_ramp_epochs=int(
                payload.get("bioemu_mechanism_signed_router_ramp_epochs", 2)
            ),
            bioemu_train_only_mechanism_signed_router=bool(
                payload.get("bioemu_train_only_mechanism_signed_router", False)
            ),
            bioemu_mechanism_signed_router_train_keys=list(
                payload.get("bioemu_mechanism_signed_router_train_keys", [])
            ),
            bioemu_mechanism_signed_router_train_shared_heads=bool(
                payload.get(
                    "bioemu_mechanism_signed_router_train_shared_heads",
                    True,
                )
            ),
            bioemu_mechanism_signed_router_preserve_locked_key_outputs=bool(
                payload.get(
                    "bioemu_mechanism_signed_router_preserve_locked_key_outputs",
                    False,
                )
            ),
            bioemu_train_only_mechanism_signed_override_adapter=bool(
                payload.get(
                    "bioemu_train_only_mechanism_signed_override_adapter",
                    False,
                )
            ),
            bioemu_mechanism_signed_override_train_keys=list(
                payload.get("bioemu_mechanism_signed_override_train_keys", [])
            ),
            bioemu_mechanism_signed_override_train_shared_heads=bool(
                payload.get(
                    "bioemu_mechanism_signed_override_train_shared_heads",
                    True,
                )
            ),
            bioemu_mechanism_signed_override_preserve_locked_key_outputs=bool(
                payload.get(
                    "bioemu_mechanism_signed_override_preserve_locked_key_outputs",
                    False,
                )
            ),
            bioemu_train_only_bridge_cs_energy_head=bool(
                payload.get("bioemu_train_only_bridge_cs_energy_head", False)
            ),
            enable_bioemu_chart_bin_residual_readout=bool(
                payload.get("enable_bioemu_chart_bin_residual_readout", False)
            ),
            bioemu_chart_bin_residual_readout_coordinate_source=str(
                payload.get(
                    "bioemu_chart_bin_residual_readout_coordinate_source",
                    "latent",
                )
            ),
            bioemu_chart_bin_residual_readout_keys=list(
                payload.get("bioemu_chart_bin_residual_readout_keys", [])
            ),
            bioemu_chart_bin_residual_readout_delta_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_delta_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_max_abs_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_max_abs_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_center_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_center_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_width_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_width_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_gate_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_gate_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_lower_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_lower_by_key", {})
            ),
            bioemu_chart_bin_residual_readout_upper_by_key=dict(
                payload.get("bioemu_chart_bin_residual_readout_upper_by_key", {})
            ),
            bioemu_fixed_family_readout_offset_by_family=dict(
                payload.get("bioemu_fixed_family_readout_offset_by_family", {})
            ),
            enable_bioemu_residue_family_chart_bin_readout=bool(
                payload.get("enable_bioemu_residue_family_chart_bin_readout", False)
            ),
            bioemu_residue_family_chart_bin_readout_coordinate_source=str(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_coordinate_source",
                    "physics",
                )
            ),
            bioemu_residue_family_chart_bin_readout_keys=list(
                payload.get("bioemu_residue_family_chart_bin_readout_keys", [])
            ),
            bioemu_residue_family_chart_bin_readout_delta_by_key=dict(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_delta_by_key",
                    {},
                )
            ),
            bioemu_residue_family_chart_bin_readout_max_abs_by_key=dict(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_max_abs_by_key",
                    {},
                )
            ),
            bioemu_residue_family_chart_bin_readout_center_by_key=dict(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_center_by_key",
                    {},
                )
            ),
            bioemu_residue_family_chart_bin_readout_width_by_key=dict(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_width_by_key",
                    {},
                )
            ),
            bioemu_residue_family_chart_bin_readout_gate_by_key=dict(
                payload.get(
                    "bioemu_residue_family_chart_bin_readout_gate_by_key",
                    {},
                )
            ),
            enable_bioemu_residue_family_directional_readout=bool(
                payload.get("enable_bioemu_residue_family_directional_readout", False)
            ),
            bioemu_residue_family_directional_readout_keys=list(
                payload.get("bioemu_residue_family_directional_readout_keys", [])
            ),
            bioemu_residue_family_directional_readout_delta_by_key=dict(
                payload.get(
                    "bioemu_residue_family_directional_readout_delta_by_key",
                    {},
                )
            ),
            bioemu_residue_family_directional_readout_max_abs_by_key=dict(
                payload.get(
                    "bioemu_residue_family_directional_readout_max_abs_by_key",
                    {},
                )
            ),
            bioemu_residue_family_directional_readout_center_by_key=dict(
                payload.get(
                    "bioemu_residue_family_directional_readout_center_by_key",
                    {},
                )
            ),
            bioemu_residue_family_directional_readout_margin_by_key=dict(
                payload.get(
                    "bioemu_residue_family_directional_readout_margin_by_key",
                    {},
                )
            ),
            bioemu_residue_family_directional_readout_gate_by_key=dict(
                payload.get(
                    "bioemu_residue_family_directional_readout_gate_by_key",
                    {},
                )
            ),
            bioemu_residue_family_directional_readout_temperature=float(
                payload.get(
                    "bioemu_residue_family_directional_readout_temperature",
                    0.25,
                )
            ),
            enable_bioemu_residue_family_anchor_readout=bool(
                payload.get("enable_bioemu_residue_family_anchor_readout", False)
            ),
            bioemu_residue_family_anchor_readout_keys=list(
                payload.get("bioemu_residue_family_anchor_readout_keys", [])
            ),
            bioemu_residue_family_anchor_readout_center_by_key=dict(
                payload.get("bioemu_residue_family_anchor_readout_center_by_key", {})
            ),
            bioemu_residue_family_anchor_readout_max_abs_by_key=dict(
                payload.get("bioemu_residue_family_anchor_readout_max_abs_by_key", {})
            ),
            bioemu_residue_family_anchor_readout_margin_by_key=dict(
                payload.get("bioemu_residue_family_anchor_readout_margin_by_key", {})
            ),
            bioemu_residue_family_anchor_readout_gate_by_key=dict(
                payload.get("bioemu_residue_family_anchor_readout_gate_by_key", {})
            ),
            bioemu_residue_family_anchor_readout_temperature=float(
                payload.get("bioemu_residue_family_anchor_readout_temperature", 1.0)
            ),
            bioemu_physics_sidecar_supervision_weight=float(
                payload.get("bioemu_physics_sidecar_supervision_weight", 0.0)
            ),
            bioemu_ring_signed_consistency_loss_weight=float(
                payload.get("bioemu_ring_signed_consistency_loss_weight", 0.0)
            ),
            bioemu_carbonyl_backbone_consistency_loss_weight=float(
                payload.get("bioemu_carbonyl_backbone_consistency_loss_weight", 0.0)
            ),
            bioemu_nmr_sampling_role=str(
                payload.get("bioemu_nmr_sampling_role", "posterior_estimator")
            ),
            enable_bioemu_latent_physical_atlas=bool(
                payload.get("enable_bioemu_latent_physical_atlas", False)
            ),
            bioemu_latent_atlas_coordinate_names=list(
                payload.get(
                    "bioemu_latent_atlas_coordinate_names",
                    [
                        "ring_current",
                        "exchange",
                        "alignment_mismatch",
                        "sulfur_electrostatic",
                        "terminal_disorder",
                        "spread_collapse",
                        "graph_fragility",
                    ],
                )
            ),
            bioemu_latent_atlas_dim=int(
                payload.get("bioemu_latent_atlas_dim", 7)
            ),
            bioemu_latent_metric_mode=str(
                payload.get("bioemu_latent_metric_mode", "low_rank_pullback")
            ),
            bioemu_latent_metric_rank=int(
                payload.get("bioemu_latent_metric_rank", 16)
            ),
            bioemu_latent_contrastive_loss_weight=float(
                payload.get("bioemu_latent_contrastive_loss_weight", 0.0)
            ),
            bioemu_latent_topology_loss_weight=float(
                payload.get("bioemu_latent_topology_loss_weight", 0.0)
            ),
            bioemu_latent_normal_drift_loss_weight=float(
                payload.get("bioemu_latent_normal_drift_loss_weight", 0.0)
            ),
            enable_hn_ring_subchart_gate=bool(
                payload.get("enable_hn_ring_subchart_gate", False)
            ),
            hn_ring_subchart_names=list(
                payload.get(
                    "hn_ring_subchart_names",
                    [
                        "ring_stable_helpful",
                        "ring_transient_helpful",
                        "ring_conflict_abstain",
                        "ring_reverse_harmful",
                    ],
                )
            ),
            hn_ring_subchart_confidence_margin=float(
                payload.get("hn_ring_subchart_confidence_margin", 0.15)
            ),
            hn_ring_subchart_reliability_threshold=float(
                payload.get("hn_ring_subchart_reliability_threshold", 0.65)
            ),
            hn_ring_signed_gate_loss_weight=float(
                payload.get("hn_ring_signed_gate_loss_weight", 0.0)
            ),
            enable_hn_rare_regime_subfeature_bundles=bool(
                payload.get("enable_hn_rare_regime_subfeature_bundles", False)
            ),
            hn_rare_regime_bundle_names=list(
                payload.get(
                    "hn_rare_regime_bundle_names",
                    [
                        "ring_current",
                        "exchange_hbond",
                        "alignment_evidence",
                        "electrostatic_sulfur",
                        "disorder_spread",
                    ],
                )
            ),
            hn_subfeature_source_policy=str(
                payload.get(
                    "hn_subfeature_source_policy",
                    "bioemu_latent_and_nmr_evidence_only",
                )
            ),
            hn_structure_probe_policy=str(
                payload.get("hn_structure_probe_policy", "benchmark_only")
            ),
            enable_hn_cross_fitted_chart_utility_labels=bool(
                payload.get("enable_hn_cross_fitted_chart_utility_labels", False)
            ),
            hn_chart_utility_label_epsilon=float(
                payload.get("hn_chart_utility_label_epsilon", 0.002)
            ),
            hn_chart_utility_label_source=str(
                payload.get(
                    "hn_chart_utility_label_source",
                    "train_split_cross_fit_only",
                )
            ),
            enable_hn_evidential_subchart_gate=bool(
                payload.get("enable_hn_evidential_subchart_gate", False)
            ),
            hn_subchart_gate_prior_strength=float(
                payload.get("hn_subchart_gate_prior_strength", 1.0)
            ),
            hn_subchart_gate_abstain_margin=float(
                payload.get("hn_subchart_gate_abstain_margin", 0.15)
            ),
            enable_hn_subchart_sparse_moe=bool(
                payload.get("enable_hn_subchart_sparse_moe", False)
            ),
            hn_subchart_sparse_moe_mode=str(
                payload.get("hn_subchart_sparse_moe_mode", "entmax")
            ),
            hn_subchart_sparse_moe_top_k=int(
                payload.get("hn_subchart_sparse_moe_top_k", 2)
            ),
            enable_hn_counterfactual_chart_dropout=bool(
                payload.get("enable_hn_counterfactual_chart_dropout", False)
            ),
            hn_counterfactual_chart_dropout_rate=float(
                payload.get("hn_counterfactual_chart_dropout_rate", 0.10)
            ),
            enable_bioemu_latent_curvature_uncertainty=bool(
                payload.get("enable_bioemu_latent_curvature_uncertainty", False)
            ),
            bioemu_latent_jvp_probe_count=int(
                payload.get("bioemu_latent_jvp_probe_count", 2)
            ),
            bioemu_latent_curvature_loss_weight=float(
                payload.get("bioemu_latent_curvature_loss_weight", 0.0)
            ),
            bioemu_latent_high_curvature_abstain_threshold=float(
                payload.get("bioemu_latent_high_curvature_abstain_threshold", 0.65)
            ),
            enable_bioemu_latent_component_persistence_audit=bool(
                payload.get("enable_bioemu_latent_component_persistence_audit", False)
            ),
            bioemu_nmr_enable_ema_teacher=bool(
                payload.get("bioemu_nmr_enable_ema_teacher", True)
            ),
            bioemu_nmr_ema_decay=float(payload.get("bioemu_nmr_ema_decay", 0.995)),
            bioemu_nmr_kl_anneal_epochs=int(
                payload.get("bioemu_nmr_kl_anneal_epochs", 3)
            ),
            bioemu_nmr_weight_temperature_warmup_epochs=int(
                payload.get("bioemu_nmr_weight_temperature_warmup_epochs", 2)
            ),
            bioemu_nmr_initial_weight_detach_epochs=int(
                payload.get("bioemu_nmr_initial_weight_detach_epochs", 1)
            ),
            bioemu_nmr_ring_correction_cap_warmup_epochs=int(
                payload.get("bioemu_nmr_ring_correction_cap_warmup_epochs", 2)
            ),
            bioemu_nmr_enable_per_entry_balanced_batching=bool(
                payload.get("bioemu_nmr_enable_per_entry_balanced_batching", True)
            ),
            bioemu_nmr_rare_regime_train_oversample_factor=float(
                payload.get("bioemu_nmr_rare_regime_train_oversample_factor", 2.0)
            ),
            j_coupling_loss_weight=float(payload.get("j_coupling_loss_weight", 0.1)),
            noe_loss_weight=float(payload.get("noe_loss_weight", 0.1)),
            entropy_regularization_weight=float(
                payload.get("entropy_regularization_weight", 1e-3)
            ),
            sequence_embedding_dim=int(payload.get("sequence_embedding_dim", 64)),
            source_embedding_dim=int(payload.get("source_embedding_dim", 16)),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            dropout=float(payload.get("dropout", 0.1)),
            use_candidate_set_encoder=bool(
                payload.get("use_candidate_set_encoder", False)
            ),
            set_encoder_layers=int(payload.get("set_encoder_layers", 0)),
            set_encoder_heads=int(payload.get("set_encoder_heads", 4)),
            logit_temperature=float(payload.get("logit_temperature", 1.0)),
            enable_forward_residual_head=bool(
                payload.get("enable_forward_residual_head", False)
            ),
            forward_residual_loss_weight=float(
                payload.get("forward_residual_loss_weight", 0.0)
            ),
            forward_residual_hidden_dim=(
                None
                if payload.get("forward_residual_hidden_dim") is None
                else int(payload["forward_residual_hidden_dim"])
            ),
            forward_residual_regularization_weight=float(
                payload.get("forward_residual_regularization_weight", 1e-4)
            ),
            enable_residue_atom_residual_head=bool(
                payload.get("enable_residue_atom_residual_head", False)
            ),
            reconstruction_ccc_loss_weight=float(
                payload.get("reconstruction_ccc_loss_weight", 0.0)
            ),
            masked_holdout_ccc_loss_weight=float(
                payload.get("masked_holdout_ccc_loss_weight", 0.0)
            ),
            evidence_likelihood_nll_weight=float(
                payload.get("evidence_likelihood_nll_weight", 0.0)
            ),
            enable_secondary_shift_targets=bool(
                payload.get("enable_secondary_shift_targets", False)
            ),
            enable_residue_geometry_features=bool(
                payload.get("enable_residue_geometry_features", False)
            ),
            secondary_reconstruction_ccc_loss_weight=float(
                payload.get("secondary_reconstruction_ccc_loss_weight", 0.0)
            ),
            secondary_masked_holdout_ccc_loss_weight=float(
                payload.get("secondary_masked_holdout_ccc_loss_weight", 0.0)
            ),
            raw_family_ccc_guardrail_weight=float(
                payload.get("raw_family_ccc_guardrail_weight", 0.0)
            ),
            whitened_masked_ccc_loss_weight=float(
                payload.get("whitened_masked_ccc_loss_weight", 0.0)
            ),
            whitened_reconstruction_ccc_loss_weight=float(
                payload.get("whitened_reconstruction_ccc_loss_weight", 0.0)
            ),
            normalized_evidence_nll_weight=float(
                payload.get("normalized_evidence_nll_weight", 0.0)
            ),
            sequence_smoothness_loss_weight=float(
                payload.get("sequence_smoothness_loss_weight", 0.0)
            ),
            residue_atom_residual_loss_weight=float(
                payload.get("residue_atom_residual_loss_weight", 0.0)
            ),
            bioemu_residue_atom_mean_scale_by_family=dict(
                payload.get("bioemu_residue_atom_mean_scale_by_family", {})
            ),
            bioemu_residue_atom_mean_weight_by_family=dict(
                payload.get("bioemu_residue_atom_mean_weight_by_family", {})
            ),
            bioemu_residue_atom_mean_family_names=list(
                payload.get("bioemu_residue_atom_mean_family_names", [])
            ),
            bioemu_residue_atom_mean_loss_mode=str(
                payload.get("bioemu_residue_atom_mean_loss_mode", "mse")
            ),
            bioemu_residue_atom_mean_huber_delta_by_family=dict(
                payload.get("bioemu_residue_atom_mean_huber_delta_by_family", {})
            ),
            bioemu_residue_atom_mean_start_epoch=int(
                payload.get("bioemu_residue_atom_mean_start_epoch", 1)
            ),
            bioemu_residue_atom_mean_ramp_epochs=int(
                payload.get("bioemu_residue_atom_mean_ramp_epochs", 1)
            ),
            residue_atom_residual_regularization_weight=float(
                payload.get("residue_atom_residual_regularization_weight", 1e-4)
            ),
            residue_atom_mask_fraction=float(
                payload.get("residue_atom_mask_fraction", 0.2)
            ),
            residue_atom_block_mask_fraction=float(
                payload.get("residue_atom_block_mask_fraction", 0.0)
            ),
            residue_atom_eval_mask_fraction=(
                None
                if payload.get("residue_atom_eval_mask_fraction") is None
                else float(payload["residue_atom_eval_mask_fraction"])
            ),
            residue_atom_eval_block_mask_fraction=(
                None
                if payload.get("residue_atom_eval_block_mask_fraction") is None
                else float(payload["residue_atom_eval_block_mask_fraction"])
            ),
            residue_atom_mask_fraction_by_family=dict(
                payload.get("residue_atom_mask_fraction_by_family", {})
            ),
            residue_atom_eval_mask_fraction_by_family=dict(
                payload.get("residue_atom_eval_mask_fraction_by_family", {})
            ),
            residue_atom_max_residue_index=int(
                payload.get("residue_atom_max_residue_index", 4096)
            ),
            residue_atom_artifact_top_k=int(
                payload.get("residue_atom_artifact_top_k", 16)
            ),
            ccc_reconstruction_target=float(
                payload.get("ccc_reconstruction_target", 0.95)
            ),
            inference_top_k=(
                None
                if payload.get("inference_top_k") is None
                else int(payload["inference_top_k"])
            ),
            candidate_cap=(
                None
                if payload.get("candidate_cap") is None
                else int(payload["candidate_cap"])
            ),
            teacher_kl_guardrail=(
                None
                if payload.get("teacher_kl_guardrail") is None
                else float(payload["teacher_kl_guardrail"])
            ),
            seed=int(payload.get("seed", 7)),
            device=str(payload.get("device", "auto")),
            chemical_shift_baseline_path=payload.get("chemical_shift_baseline_path"),
            chemical_shift_reference_corpus_path=payload.get(
                "chemical_shift_reference_corpus_path"
            ),
            initial_checkpoint_path=payload.get("initial_checkpoint_path")
            or payload.get("resume_from_checkpoint"),
            initial_checkpoint_strict=bool(
                payload.get("initial_checkpoint_strict", True)
            ),
            initial_checkpoint_load_optimizer=bool(
                payload.get("initial_checkpoint_load_optimizer", True)
            ),
            initial_checkpoint_exclude_prefixes=list(
                payload.get("initial_checkpoint_exclude_prefixes", [])
            ),
            bioemu_x0_eval_initial_checkpoint_before_training=bool(
                payload.get("bioemu_x0_eval_initial_checkpoint_before_training", False)
            ),
            trainable_parameter_name_patterns=list(
                payload.get("trainable_parameter_name_patterns", [])
            ),
            candidate_free_evidence_encoder_layers=int(
                payload.get("candidate_free_evidence_encoder_layers", 2)
            ),
            candidate_free_evidence_encoder_heads=int(
                payload.get("candidate_free_evidence_encoder_heads", 4)
            ),
            candidate_free_state_count=int(
                payload.get("candidate_free_state_count", 16)
            ),
            candidate_free_hn_cprime_loss_weight=float(
                payload.get("candidate_free_hn_cprime_loss_weight", 8.0)
            ),
            candidate_free_worst_hn_cprime_loss_weight=float(
                payload.get("candidate_free_worst_hn_cprime_loss_weight", 0.0)
            ),
            candidate_free_family_loss_weight=float(
                payload.get("candidate_free_family_loss_weight", 2.0)
            ),
            candidate_free_reconstruction_loss_weight=float(
                payload.get("candidate_free_reconstruction_loss_weight", 3.0)
            ),
            candidate_free_student_t_nll_weight=float(
                payload.get("candidate_free_student_t_nll_weight", 0.25)
            ),
            candidate_free_state_diversity_weight=float(
                payload.get("candidate_free_state_diversity_weight", 0.05)
            ),
            candidate_free_moment_state_consistency_weight=float(
                payload.get("candidate_free_moment_state_consistency_weight", 0.05)
            ),
            enable_residue_numbering_repair=bool(
                payload.get("enable_residue_numbering_repair", True)
            ),
            residue_numbering_min_confidence=float(
                payload.get("residue_numbering_min_confidence", 0.35)
            ),
            enable_candidate_free_physics_features=bool(
                payload.get("enable_candidate_free_physics_features", True)
            ),
            candidate_free_physics_context_gate_init=float(
                payload.get("candidate_free_physics_context_gate_init", 0.10)
            ),
            candidate_free_physics_target_families=list(
                payload.get("candidate_free_physics_target_families", ["HN", "C'"])
            ),
            enable_candidate_free_structure_proxy_features=bool(
                payload.get("enable_candidate_free_structure_proxy_features", False)
            ),
            candidate_free_structure_proxy_sources=list(
                payload.get(
                    "candidate_free_structure_proxy_sources",
                    ["BioEmu", "AF"],
                )
            ),
            candidate_free_structure_proxy_max_models=int(
                payload.get("candidate_free_structure_proxy_max_models", 12)
            ),
            candidate_free_structure_proxy_blend=float(
                payload.get("candidate_free_structure_proxy_blend", 0.45)
            ),
            candidate_free_alignment_loss_min_weight=float(
                payload.get("candidate_free_alignment_loss_min_weight", 0.25)
            ),
            candidate_free_physics_risk_loss_boost=float(
                payload.get("candidate_free_physics_risk_loss_boost", 0.0)
            ),
            candidate_free_spectral_tail_loss_boost=float(
                payload.get("candidate_free_spectral_tail_loss_boost", 0.0)
            ),
            candidate_free_spectral_tail_fraction=float(
                payload.get("candidate_free_spectral_tail_fraction", 0.2)
            ),
            candidate_free_spectral_tail_families=list(
                payload.get("candidate_free_spectral_tail_families", [])
            ),
            candidate_free_hn_gly_tail_loss_boost=float(
                payload.get("candidate_free_hn_gly_tail_loss_boost", 0.0)
            ),
            candidate_free_hn_gly_tail_fraction=float(
                payload.get("candidate_free_hn_gly_tail_fraction", 0.25)
            ),
            candidate_free_hn_context_tail_loss_boost=float(
                payload.get("candidate_free_hn_context_tail_loss_boost", 0.0)
            ),
            candidate_free_hn_context_tail_fraction=float(
                payload.get("candidate_free_hn_context_tail_fraction", 0.25)
            ),
            candidate_free_hn_context_tail_min_risk=float(
                payload.get("candidate_free_hn_context_tail_min_risk", 0.20)
            ),
            hn95_context_tail_residual_loss_weight=float(
                payload.get("hn95_context_tail_residual_loss_weight", 0.0)
            ),
            hn95_context_tail_residual_min_risk=float(
                payload.get("hn95_context_tail_residual_min_risk", 0.20)
            ),
            candidate_free_reliability_regularization_weight=float(
                payload.get("candidate_free_reliability_regularization_weight", 0.01)
            ),
            candidate_free_forced_train_entity_uids=list(
                payload.get("candidate_free_forced_train_entity_uids", [])
            ),
            candidate_free_forced_train_entities_path=payload.get(
                "candidate_free_forced_train_entities_path"
            ),
            candidate_free_exclude_forced_train_from_val=bool(
                payload.get("candidate_free_exclude_forced_train_from_val", True)
            ),
            candidate_free_forced_train_val_include_entity_uids=list(
                payload.get("candidate_free_forced_train_val_include_entity_uids", [])
            ),
            candidate_free_forced_train_val_include_entities_path=payload.get(
                "candidate_free_forced_train_val_include_entities_path"
            ),
            candidate_free_replay_entity_uids=list(
                payload.get("candidate_free_replay_entity_uids", [])
            ),
            candidate_free_replay_entities_path=payload.get(
                "candidate_free_replay_entities_path"
            ),
            candidate_free_replay_entity_oversample_factor=int(
                payload.get("candidate_free_replay_entity_oversample_factor", 1)
            ),
            candidate_free_val_include_entity_uids=list(
                payload.get("candidate_free_val_include_entity_uids", [])
            ),
            candidate_free_val_exclude_entity_uids=list(
                payload.get("candidate_free_val_exclude_entity_uids", [])
            ),
            candidate_free_entity_family_replay_loss_weight=float(
                payload.get("candidate_free_entity_family_replay_loss_weight", 0.0)
            ),
            candidate_free_entity_family_replay_weight_threshold=float(
                payload.get(
                    "candidate_free_entity_family_replay_weight_threshold",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_center_loss_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_loss_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_support_center_family_weights=dict(
                payload.get(
                    "candidate_free_failed_active_face_support_center_family_weights",
                    {},
                )
            ),
            candidate_free_failed_active_face_support_center_softness=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_softness",
                    0.05,
                )
            ),
            candidate_free_failed_active_face_support_center_gap_focus_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_gap_focus_scale",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_support_center_target_source=str(
                payload.get(
                    "candidate_free_failed_active_face_support_center_target_source",
                    "target_value",
                )
            ),
            candidate_free_failed_active_face_support_center_dual_weight_power=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_dual_weight_power",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_center_response_deficit_weight_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_response_deficit_weight_scale",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_support_center_min_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_min_row_weight",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_center_max_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_max_row_weight",
                    10.0,
                )
            ),
            candidate_free_failed_active_face_support_center_warmup_epochs=int(
                payload.get(
                    "candidate_free_failed_active_face_support_center_warmup_epochs",
                    0,
                )
            ),
            candidate_free_failed_active_face_support_center_start_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_support_center_start_scale",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_loss_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_loss_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_family_weights=dict(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_family_weights",
                    {},
                )
            ),
            candidate_free_failed_active_face_support_tail_softness=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_softness",
                    0.05,
                )
            ),
            candidate_free_failed_active_face_support_tail_direction=str(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_direction",
                    "auto",
                )
            ),
            candidate_free_failed_active_face_support_tail_target_source=str(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_target_source",
                    "target_value",
                )
            ),
            candidate_free_failed_active_face_support_tail_dual_weight_power=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_dual_weight_power",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_response_deficit_weight_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_response_deficit_weight_scale",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_min_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_min_row_weight",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_max_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_max_row_weight",
                    10.0,
                )
            ),
            candidate_free_failed_active_face_support_tail_warmup_epochs=int(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_warmup_epochs",
                    0,
                )
            ),
            candidate_free_failed_active_face_support_tail_start_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_support_tail_start_scale",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_loss_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_loss_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_family_weights=dict(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_family_weights",
                    {},
                )
            ),
            candidate_free_failed_active_face_energy_assignment_target_source=str(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_target_source",
                    "target_value",
                )
            ),
            candidate_free_failed_active_face_energy_assignment_margin=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_margin",
                    0.04,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_positive_fraction=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_positive_fraction",
                    0.08,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_negative_fraction=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_negative_fraction",
                    0.12,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_positive_mass_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_positive_mass_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_temperature=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_temperature",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_dual_weight_power=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_dual_weight_power",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_response_deficit_weight_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_response_deficit_weight_scale",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_min_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_min_row_weight",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_max_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_max_row_weight",
                    10.0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_warmup_epochs=int(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_warmup_epochs",
                    0,
                )
            ),
            candidate_free_failed_active_face_energy_assignment_start_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_energy_assignment_start_scale",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_loss_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_loss_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_family_weights=dict(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_family_weights",
                    {},
                )
            ),
            candidate_free_failed_active_face_shared_q_response_target_source=str(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_target_source",
                    "target_value",
                )
            ),
            candidate_free_failed_active_face_shared_q_response_temperature=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_temperature",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_huber_beta=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_huber_beta",
                    0.15,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_support_gap_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_support_gap_weight",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_dual_weight_power=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_dual_weight_power",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_deficit_weight_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_deficit_weight_scale",
                    0.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_min_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_min_row_weight",
                    1.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_max_row_weight=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_max_row_weight",
                    10.0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_warmup_epochs=int(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_warmup_epochs",
                    0,
                )
            ),
            candidate_free_failed_active_face_shared_q_response_start_scale=float(
                payload.get(
                    "candidate_free_failed_active_face_shared_q_response_start_scale",
                    1.0,
                )
            ),
            candidate_free_entity_loss_weights=dict(
                payload.get("candidate_free_entity_loss_weights", {})
            ),
            candidate_free_entity_family_loss_weights=dict(
                payload.get("candidate_free_entity_family_loss_weights", {})
            ),
            candidate_free_replay_ucbshift_residue_offset_contract=dict(
                payload.get(
                    "candidate_free_replay_ucbshift_residue_offset_contract",
                    payload.get("rejected_guard_ucbshift_residue_offset_contract", {}),
                )
                or {}
            ),
            candidate_free_replay_require_ucbshift_residue_offset_contract=bool(
                payload.get(
                    "candidate_free_replay_require_ucbshift_residue_offset_contract",
                    False,
                )
            ),
            legacy_candidate_diagnostic=bool(
                payload.get("legacy_candidate_diagnostic", False)
            ),
            enable_state_mixture=bool(payload.get("enable_state_mixture", False)),
            state_mixture_count=int(payload.get("state_mixture_count", 8)),
            mirror_descent_steps=int(payload.get("mirror_descent_steps", 0)),
            mirror_descent_step_size=float(
                payload.get("mirror_descent_step_size", 0.0)
            ),
            robust_likelihood_mode=str(
                payload.get("robust_likelihood_mode", "gaussian")
            ),
            student_t_degrees_of_freedom=float(
                payload.get("student_t_degrees_of_freedom", 4.0)
            ),
            benchmark_render_config=dict(
                payload.get(
                    "benchmark_render_config",
                    payload.get("benchmark_render", {}),
                )
            ),
            render_best_preview=bool(payload.get("render_best_preview", False)),
            best_preview_max_train_examples=int(
                payload.get("best_preview_max_train_examples", 32)
            ),
            best_preview_max_val_examples=int(
                payload.get("best_preview_max_val_examples", 50)
            ),
            history_window_epochs=(
                None
                if payload.get("history_window_epochs") is None
                else int(payload["history_window_epochs"])
            ),
            last_checkpoint_interval=int(payload.get("last_checkpoint_interval", 1)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "StudentTrainingConfig":
        """Load the configuration from one JSON file."""
        path_obj = Path(path).expanduser()

        def load_with_extends(config_path: Path, seen: set[Path]) -> dict[str, Any]:
            resolved_path = config_path.expanduser()
            if not resolved_path.is_absolute():
                resolved_path = resolved_path.resolve()
            cycle_key = resolved_path.resolve()
            if cycle_key in seen:
                raise ValueError(f"cyclic StudentTrainingConfig extends: {cycle_key}")
            payload = json.loads(resolved_path.read_text())
            parent_path = payload.pop("extends", None)
            if not parent_path:
                return payload
            resolved_parent = Path(str(parent_path)).expanduser()
            if not resolved_parent.is_absolute():
                resolved_parent = resolved_path.parent / resolved_parent
            parent_payload = load_with_extends(resolved_parent, seen | {cycle_key})
            parent_payload.update(payload)
            return parent_payload

        return cls.from_dict(load_with_extends(path_obj, set()))


@dataclass(slots=True)
class BenchmarkRenderConfig:
    """Configuration for benchmark rendering and uncertainty artifacts."""

    emit_landscape: bool = True
    emit_structure_uncertainty: bool = True
    emit_nmr_posteriors: bool = True
    emit_rci_adapter: bool = True
    emit_ensemble_states: bool = True
    rci_profile_path: str | None = None
    projection_method: str = "weighted_pca"
    landscape_density_method: str = "weighted_kde"
    representative_structure: str = "weighted_medoid"
    max_pair_feature_residues: int = 48
    max_pair_distance_features: int = 128
    grid_size: int = 64
    max_ensemble_states: int = 6
    min_ensemble_state_mass: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to one JSON file."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkRenderConfig":
        """Create one configuration object from a dictionary."""
        return cls(
            emit_landscape=bool(payload.get("emit_landscape", True)),
            emit_structure_uncertainty=bool(
                payload.get("emit_structure_uncertainty", True)
            ),
            emit_nmr_posteriors=bool(payload.get("emit_nmr_posteriors", True)),
            emit_rci_adapter=bool(payload.get("emit_rci_adapter", True)),
            emit_ensemble_states=bool(payload.get("emit_ensemble_states", True)),
            rci_profile_path=payload.get("rci_profile_path"),
            projection_method=str(payload.get("projection_method", "weighted_pca")),
            landscape_density_method=str(
                payload.get("landscape_density_method", "weighted_kde")
            ),
            representative_structure=str(
                payload.get("representative_structure", "weighted_medoid")
            ),
            max_pair_feature_residues=int(payload.get("max_pair_feature_residues", 48)),
            max_pair_distance_features=int(
                payload.get("max_pair_distance_features", 128)
            ),
            grid_size=int(payload.get("grid_size", 64)),
            max_ensemble_states=int(payload.get("max_ensemble_states", 6)),
            min_ensemble_state_mass=float(payload.get("min_ensemble_state_mass", 0.05)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkRenderConfig":
        """Load the configuration from one JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(slots=True)
class TrainingReadinessConfig:
    """Configuration for pre-training readiness audits."""

    selected_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    validation_splits: list[str] = field(default_factory=lambda: ["val"])
    require_ucbshift_summary: bool = True
    require_zero_failed_examples: bool = True
    required_written_examples: int = 250
    required_validation_examples: int = 50
    minimum_median_chemical_shift_valid_fraction: float = 0.7
    minimum_source_count: int = 2
    minimum_requested_candidate_fraction: float = 0.75
    minimum_ucbshift_written_example_fraction: float = 0.75
    minimum_ucbshift_baseline_example_fraction: float = 0.75
    max_missing_shift_fraction: float = 0.25
    max_no_cs_coverage_fraction: float = 0.25

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to one JSON file."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingReadinessConfig":
        """Create one configuration object from a dictionary."""
        return cls(
            selected_splits=list(payload.get("selected_splits", ["train", "val"])),
            validation_splits=list(payload.get("validation_splits", ["val"])),
            require_ucbshift_summary=bool(
                payload.get("require_ucbshift_summary", True)
            ),
            require_zero_failed_examples=bool(
                payload.get("require_zero_failed_examples", True)
            ),
            required_written_examples=int(
                payload.get("required_written_examples", 250)
            ),
            required_validation_examples=int(
                payload.get("required_validation_examples", 50)
            ),
            minimum_median_chemical_shift_valid_fraction=float(
                payload.get("minimum_median_chemical_shift_valid_fraction", 0.7)
            ),
            minimum_source_count=int(payload.get("minimum_source_count", 2)),
            minimum_requested_candidate_fraction=float(
                payload.get("minimum_requested_candidate_fraction", 0.75)
            ),
            minimum_ucbshift_written_example_fraction=float(
                payload.get("minimum_ucbshift_written_example_fraction", 0.75)
            ),
            minimum_ucbshift_baseline_example_fraction=float(
                payload.get("minimum_ucbshift_baseline_example_fraction", 0.75)
            ),
            max_missing_shift_fraction=float(
                payload.get("max_missing_shift_fraction", 0.25)
            ),
            max_no_cs_coverage_fraction=float(
                payload.get("max_no_cs_coverage_fraction", 0.25)
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingReadinessConfig":
        """Load the configuration from one JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(slots=True)
class UCBShiftGenerationConfig:
    """Configuration for UCBShift2.0 sidecar and baseline generation."""

    selected_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    selected_bmrb_ids: list[str] = field(default_factory=list)
    selected_sources: list[str] = field(
        default_factory=lambda: ["AF3", "BioEmu", "CALVADOS2"]
    )
    output_dir_templates: dict[str, str] = field(default_factory=dict)
    representative_source_priority: list[str] = field(
        default_factory=lambda: ["AF3", "BioEmu", "CALVADOS2"]
    )
    ucbshift_root: str = "external/ucbshift2/CSpred"
    python_executable: str = "python"
    models_dir: str | None = None
    models_url: str = "https://zenodo.org/records/15375968/files/models.zip?download=1"
    prediction_mode: str = "shiftx_only"
    ph: float = 5.0
    worker_count: int = 1
    ucbshift_worker_count: int = 1
    subprocess_timeout_seconds: float | None = 600.0
    sidecar_lock_timeout_seconds: float = 120.0
    sidecar_stale_lock_seconds: float = 900.0
    refresh_outputs: bool = False
    max_examples: int | None = None
    max_candidates_per_example: int | None = None
    shard_count: int = 1
    shard_index: int = 0
    fail_fast: bool = False
    baseline_output_path: str | None = "data/integrated/baselines/ucbshift2_holdout.csv"
    reference_corpus_output_path: str | None = (
        "data/integrated/baselines/ucbshift2_reference_corpus.csv"
    )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write the configuration to one JSON file."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UCBShiftGenerationConfig":
        """Create one configuration object from a dictionary."""
        return cls(
            selected_splits=list(payload.get("selected_splits", ["train", "val"])),
            selected_bmrb_ids=list(payload.get("selected_bmrb_ids", [])),
            selected_sources=list(
                payload.get("selected_sources", ["AF3", "BioEmu", "CALVADOS2"])
            ),
            output_dir_templates=dict(payload.get("output_dir_templates", {})),
            representative_source_priority=list(
                payload.get(
                    "representative_source_priority", ["AF3", "BioEmu", "CALVADOS2"]
                )
            ),
            ucbshift_root=str(
                payload.get("ucbshift_root", "external/ucbshift2/CSpred")
            ),
            python_executable=str(payload.get("python_executable", "python")),
            models_dir=payload.get("models_dir"),
            models_url=str(
                payload.get(
                    "models_url",
                    "https://zenodo.org/records/15375968/files/models.zip?download=1",
                )
            ),
            prediction_mode=str(payload.get("prediction_mode", "shiftx_only")),
            ph=float(payload.get("ph", 5.0)),
            worker_count=int(payload.get("worker_count", 1)),
            ucbshift_worker_count=int(payload.get("ucbshift_worker_count", 1)),
            subprocess_timeout_seconds=(
                None
                if payload.get("subprocess_timeout_seconds") is None
                else float(payload.get("subprocess_timeout_seconds", 600.0))
            ),
            sidecar_lock_timeout_seconds=float(
                payload.get("sidecar_lock_timeout_seconds", 120.0)
            ),
            sidecar_stale_lock_seconds=float(
                payload.get("sidecar_stale_lock_seconds", 900.0)
            ),
            refresh_outputs=bool(payload.get("refresh_outputs", False)),
            max_examples=(
                None
                if payload.get("max_examples") is None
                else int(payload["max_examples"])
            ),
            max_candidates_per_example=(
                None
                if payload.get("max_candidates_per_example") is None
                else int(payload["max_candidates_per_example"])
            ),
            shard_count=int(payload.get("shard_count", 1)),
            shard_index=int(payload.get("shard_index", 0)),
            fail_fast=bool(payload.get("fail_fast", False)),
            baseline_output_path=payload.get("baseline_output_path"),
            reference_corpus_output_path=payload.get("reference_corpus_output_path"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "UCBShiftGenerationConfig":
        """Load the configuration from one JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))
