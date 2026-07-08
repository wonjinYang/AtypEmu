"""Rare-regime physics atlas materialization for BioEmu NMR runs.

The atlas is an analysis-first table: it joins masked chemical-shift residuals
with physics coordinates, posterior reliability, teacher/source audits, and
residue-family expert probes.  The v3 row grain is
``entity_uid x residue x atom_family x mechanism x quantile_bin x ensemble_state_class``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atypemu.training.data import parse_chemical_shift_target_id


FAMILY_ORDER = ["HN", "C'", "N", "CA", "CB"]
SOURCE_COLUMNS = [
    "source_mass_bioemu",
    "source_mass_af3",
    "source_mass_calvados2",
]
RARE_REGIME_V2_COORDINATE_NAMES = [
    "ring_orientation",
    "ring_occupancy",
    "exchange_hbond",
    "terminal_disorder",
    "gly_flexibility",
    "carbonyl_backbone",
    "spread_collapse",
    "peptide_plane_geometry",
    "sulfur_electrostatic",
    "electrostatic_sulfur",
    "sidechain_rotamer",
    "ring_current",
    "alignment_mismatch",
    "graph_fragility",
    "bioemu_component",
    "ensemble_component",
    "uncertainty_abstain",
]
ENSEMBLE_STATE_CLASSES = [
    "fully_flexible",
    "metamorphic_switching",
    "partial_folded_rigid",
    "support_gap",
]
PRIMARY_MECHANISMS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "HN": (
        "exchange",
        "exchange_hbond",
        "ring_orientation",
        "terminal_disorder",
        "gly_flexibility",
    ),
    "C'": ("spread_collapse", "carbonyl_backbone", "exchange_hbond"),
    "N": ("sulfur_electrostatic", "electrostatic_sulfur", "residue_family"),
    "CA": ("ring_current", "sidechain_rotamer", "residue_family"),
    "CB": ("exchange", "sidechain_rotamer", "sulfur_electrostatic", "residue_family"),
}
DEFAULT_MECHANISMS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "HN": (
        "exchange",
        "ring_orientation",
        "ring_occupancy",
        "terminal_disorder",
        "gly_flexibility",
        "alignment_mismatch",
    ),
    "C'": (
        "spread_collapse",
        "carbonyl_backbone",
        "exchange_hbond",
        "alignment_mismatch",
    ),
    "N": (
        "sulfur_electrostatic",
        "carbonyl_backbone",
        "exchange_hbond",
        "residue_family",
    ),
    "CA": ("ring_current", "sidechain_rotamer", "bioemu_component", "residue_family"),
    "CB": ("exchange", "sidechain_rotamer", "sulfur_electrostatic", "residue_family"),
}
TAIL_THRESHOLDS = {
    "HN": 0.30,
    "N": 8.0,
    "CA": 4.5,
    "CB": 12.0,
    "C'": 1.8,
}
RARE_REGIME_V2_MAX_ABS_BY_FAMILY = {
    "HN": 0.035,
    "N": 3.0,
    "CA": 2.5,
    "CB": 5.0,
    "C'": 0.35,
}
RARE_REGIME_V2_RESIDUE_FAMILY_PRIOR_KEYS = [
    "PRO:N",
    "GLY:N",
    "THR:CB",
    "SER:CB",
    "ILE:CA",
    "PRO:CA",
    "VAL:CA",
    "GLY:CA",
    "VAL:CB",
    "THR:C'",
]
ATLAS_COLUMNS = [
    "split",
    "entity_uid",
    "bmrb_id",
    "target_id",
    "residue_index",
    "residue_name",
    "atom_family",
    "mechanism",
    "quantile_bin",
    "ensemble_state_class",
    "coordinate_value",
    "coordinate_abs",
    "target_value",
    "predicted_value",
    "raw_predicted_value",
    "residual",
    "abs_residual",
    "residual_sign",
    "posterior_sigma",
    "posterior_ess",
    "posterior_entropy",
    "posterior_bridge_norm",
    "teacher_ess",
    "teacher_entropy",
    "teacher_reliability",
    "source_mass_bioemu",
    "source_mass_af3",
    "source_mass_calvados2",
    "top_source",
    "top_source_mass",
    "family_conflict_score",
    "hn_cprime_direction_conflict_score",
    "router_confidence",
    "tangent_alignment",
    "expert_policy_gain",
    "primary_physics_role",
    "teacher_action",
    "signed_residual_consensus",
    "support_replay_priority",
    "rare_regime_priority",
]
SUMMARY_COLUMNS = [
    "split",
    "atom_family",
    "mechanism",
    "quantile_bin",
    "ensemble_state_class",
    "row_count",
    "ccc",
    "ccc_target",
    "ccc_gap_to_target",
    "mae",
    "bias",
    "p95_abs_residual",
    "tail_fraction",
    "mean_residual_sign",
    "mean_posterior_ess",
    "mean_posterior_entropy",
    "mean_teacher_reliability",
    "mean_source_mass_bioemu",
    "mean_source_mass_af3",
    "mean_source_mass_calvados2",
    "mean_family_conflict_score",
    "mean_hn_cprime_direction_conflict_score",
    "mean_router_confidence",
    "mean_tangent_alignment",
    "mean_expert_policy_gain",
    "mean_signed_residual_consensus",
    "mean_support_replay_priority",
    "mean_rare_regime_priority",
    "recommended_action",
]
ENSEMBLE_AUDIT_COLUMNS = [
    "split",
    "atom_family",
    "ensemble_state_class",
    "row_count",
    "ccc",
    "mae",
    "p95_abs_residual",
    "mean_posterior_ess",
    "mean_posterior_entropy",
    "mean_source_mass_bioemu",
    "mean_support_replay_priority",
    "state_action",
]
ACCEPTANCE_COLUMNS = [
    "criterion",
    "status",
    "value",
    "threshold",
    "detail",
]


def write_rare_regime_atlas(
    run_dir: str | Path,
    *,
    target_ccc: float = 0.95,
) -> dict[str, Path]:
    """Build and write rare-regime atlas parquet/tsv outputs for one run."""

    run_path = Path(run_dir).expanduser().resolve()
    arrays_dir = run_path / "reports" / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    atlas, summary, acceptance = build_rare_regime_atlas(
        arrays_dir,
        target_ccc=float(target_ccc),
    )
    outputs = {
        "atlas": arrays_dir / "bioemu_rare_regime_atlas.parquet",
        "summary": arrays_dir / "bioemu_rare_regime_summary.parquet",
        "acceptance": arrays_dir / "bioemu_rare_regime_acceptance.parquet",
        "atlas_v3": arrays_dir / "bioemu_rare_regime_atlas_v3.parquet",
        "summary_v3": arrays_dir / "bioemu_rare_regime_summary_v3.parquet",
        "ensemble_state_audit": arrays_dir / "bioemu_ensemble_state_audit.parquet",
    }
    _write_frame(atlas, outputs["atlas"])
    _write_frame(summary, outputs["summary"])
    _write_frame(acceptance, outputs["acceptance"])
    _write_frame(atlas, outputs["atlas_v3"])
    _write_frame(summary, outputs["summary_v3"])
    _write_frame(ensemble_state_audit(atlas), outputs["ensemble_state_audit"])
    return outputs


def derive_rare_regime_training_overrides(
    frame_or_path: pd.DataFrame | str | Path,
    *,
    target_ccc: float = 0.95,
    max_router_keys: int = 12,
    max_residue_family_keys: int = 10,
    min_rows: int = 8,
) -> dict[str, Any]:
    """Derive conservative student-training overrides from a rare-regime atlas.

    The returned payload intentionally configures small local corrections rather
    than broad global pressure.  Residuals in the atlas are defined as
    ``predicted - target``, so a positive mean residual maps to a negative
    correction direction.
    """

    source = _load_policy_source_frame(frame_or_path)
    if source.empty:
        return _empty_rare_regime_training_overrides()
    if _is_summary_frame(source):
        summary = source.copy()
        atlas = pd.DataFrame()
    else:
        atlas = source.copy()
        summary = summarize_rare_regime_atlas(atlas, target_ccc=float(target_ccc))
    candidates = _select_router_candidate_rows(
        summary,
        target_ccc=float(target_ccc),
        min_rows=int(min_rows),
    )
    router_rows = candidates.head(max(int(max_router_keys), 0)).copy()
    router_keys = [_policy_key(row) for _, row in router_rows.iterrows()]
    direction_by_key = {
        key: _policy_direction(row)
        for key, (_, row) in zip(router_keys, router_rows.iterrows(), strict=False)
        if key
    }
    max_abs_by_key = {
        key: _policy_max_abs(row)
        for key, (_, row) in zip(router_keys, router_rows.iterrows(), strict=False)
        if key
    }
    width_by_key = {
        key: _policy_width(row)
        for key, (_, row) in zip(router_keys, router_rows.iterrows(), strict=False)
        if key
    }
    uncertainty_by_key = {
        key: _policy_uncertainty_threshold(row)
        for key, (_, row) in zip(router_keys, router_rows.iterrows(), strict=False)
        if key
    }
    override_keys = [
        key
        for key, row in zip(router_keys, router_rows.to_dict("records"), strict=False)
        if str(row.get("atom_family")) in {"HN", "C'"}
    ][: max(1, min(len(router_keys), max(int(max_router_keys) // 2, 1)))]
    conflict_gate_keys = [
        key
        for key, row in zip(router_keys, router_rows.to_dict("records"), strict=False)
        if _row_conflict_score(row) >= 0.05 or str(row.get("atom_family")) in {"HN", "C'"}
    ][: max(1, min(len(router_keys), 6))]
    residue_keys = _derive_residue_family_keys(
        atlas,
        max_keys=max(int(max_residue_family_keys), 0),
    )
    return {
        "enable_bioemu_physics_atlas_v36": True,
        "bioemu_physics_atlas_coordinate_names": list(RARE_REGIME_V2_COORDINATE_NAMES),
        "regime_router_enabled": True,
        "mechanism_tangent_adapter_rank": 8,
        "bayesian_shrinkage_enabled": True,
        "direction_conflict_shared_adapter_threshold": 0.08,
        "teacher_mean_policy": "guard_or_low_weight",
        "enable_bioemu_neighborhood_aware_nmr": True,
        "bioemu_neighborhood_edge_offsets": [-2, -1, 0, 1, 2],
        "bioemu_neighborhood_family_names": ["HN", "C'"],
        "bioemu_neighborhood_hidden_delta_max_abs": 0.035,
        "bioemu_neighborhood_energy_weight": 0.05,
        "bioemu_neighborhood_train_with_local_adapters": True,
        "bioemu_neighborhood_use_row_physics_proxy": True,
        "enable_bioemu_family_tangent_adapter_bank": True,
        "bioemu_family_tangent_max_abs_by_family": dict(
            RARE_REGIME_V2_MAX_ABS_BY_FAMILY
        ),
        "bioemu_family_chart_names_by_family": {
            "HN": [
                "ring_orientation",
                "ring_occupancy",
                "exchange_hbond",
                "terminal_disorder",
                "gly_flexibility",
            ],
            "C'": [
                "carbonyl_backbone",
                "exchange_hbond",
                "spread_collapse",
                "peptide_plane_geometry",
            ],
            "N": ["sulfur_electrostatic", "exchange_hbond", "residue_family"],
            "CA": ["sidechain_rotamer", "alignment_mismatch", "bioemu_component"],
            "CB": ["sidechain_rotamer", "sulfur_electrostatic", "residue_family"],
        },
        "enable_bioemu_mechanism_signed_router": bool(router_keys),
        "bioemu_mechanism_signed_router_coordinate_source": "hybrid",
        "bioemu_mechanism_signed_router_keys": router_keys,
        "bioemu_mechanism_signed_router_direction_by_key": direction_by_key,
        "bioemu_mechanism_signed_router_max_abs_by_key": max_abs_by_key,
        "bioemu_mechanism_signed_router_width_by_key": width_by_key,
        "bioemu_mechanism_signed_router_uncertainty_threshold_by_key": uncertainty_by_key,
        "bioemu_mechanism_signed_router_uncertainty_threshold": 0.82,
        "bioemu_mechanism_signed_router_loss_weight": 0.0025,
        "bioemu_mechanism_signed_router_delta_loss_weight": 0.0025,
        "bioemu_mechanism_signed_router_delta_target_scale": 0.12,
        "bioemu_mechanism_signed_router_delta_target_max_by_family": {
            "HN": 0.035,
            "C'": 0.35,
            "N": 3.0,
            "CA": 2.5,
            "CB": 5.0,
        },
        "bioemu_mechanism_signed_router_start_epoch": 3,
        "bioemu_mechanism_signed_router_ramp_epochs": 6,
        "enable_bioemu_mechanism_signed_bias_conflict_gate": bool(conflict_gate_keys),
        "bioemu_mechanism_signed_bias_conflict_gate_keys": conflict_gate_keys,
        "bioemu_mechanism_signed_bias_conflict_gate_strength": 0.12,
        "enable_bioemu_mechanism_signed_override_adapter": bool(override_keys),
        "bioemu_mechanism_signed_override_keys": override_keys,
        "bioemu_mechanism_signed_override_force_direction": False,
        "bioemu_mechanism_signed_override_centered_component": True,
        "bioemu_mechanism_signed_override_loss_weight": 0.0018,
        "bioemu_mechanism_signed_override_direction_loss_weight": 0.0006,
        "bioemu_mechanism_signed_override_target_scale": 0.12,
        "bioemu_mechanism_signed_override_target_max_by_family": {
            "HN": 0.035,
            "C'": 0.35,
        },
        "enable_bioemu_residue_family_rare_adapter": bool(residue_keys),
        "bioemu_residue_family_rare_adapter_keys": residue_keys,
        "bioemu_train_only_residue_family_rare_adapter": True,
        "bioemu_residue_family_rare_adapter_train_keys": residue_keys,
        "bioemu_residue_family_rare_adapter_train_shared_heads": False,
        "bioemu_residue_family_rare_adapter_loss_weight": 0.0012,
        "bioemu_residue_family_rare_adapter_delta_loss_weight": 0.0012,
        "bioemu_residue_family_rare_adapter_direction_loss_weight": 0.0006,
        "bioemu_residue_family_rare_adapter_delta_target_scale": 0.18,
        "bioemu_residue_family_rare_adapter_uncertainty_threshold": 0.84,
        "bioemu_residue_family_rare_adapter_start_epoch": 4,
        "bioemu_residue_family_rare_adapter_ramp_epochs": 6,
        "bioemu_family_tangent_orthogonality_loss_weight": 0.025,
        "bioemu_family_gradient_conflict_loss_weight": 0.020,
        "bioemu_posterior_ess_floor": 2.8,
        "bioemu_posterior_ess_loss_weight": 0.18,
        "bioemu_posterior_chart_entropy_floor": 0.28,
        "bioemu_posterior_chart_entropy_loss_weight": 0.20,
        "bioemu_posterior_bridge_norm_weight": 0.034,
        "bioemu_all_family_outlier_loss_weight": 0.012,
        "bioemu_all_family_outlier_family_names": ["N", "CA", "CB"],
        "bioemu_all_family_outlier_fraction": 0.06,
        "bioemu_all_family_outlier_min_rows": 8,
        "bioemu_all_family_outlier_max_rows_per_family": 32,
        "bioemu_all_family_outlier_low_ess_gate_ess_floor": 2.2,
        "bioemu_all_family_outlier_low_ess_gate_min_scale": 0.35,
        "bioemu_ucbshift_cnnls_teacher_kl_weight": 0.0,
        "bioemu_ucbshift_cnnls_teacher_guard_exclude_mean_loss": True,
        "bioemu_ucbshift_cnnls_teacher_reliability_weighting": True,
        "bioemu_ucbshift_cnnls_teacher_mean_family_names": ["N", "CA"],
        "bioemu_cs_reweighting_teacher_mean_loss_weight": 0.012,
    }


def _empty_rare_regime_training_overrides() -> dict[str, Any]:
    return {
        "enable_bioemu_physics_atlas_v36": True,
        "regime_router_enabled": True,
        "bayesian_shrinkage_enabled": True,
        "teacher_mean_policy": "guard_or_low_weight",
        "bioemu_ucbshift_cnnls_teacher_kl_weight": 0.0,
    }


def _load_policy_source_frame(frame_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    path = Path(frame_or_path).expanduser()
    if path.is_dir():
        path = path / "bioemu_rare_regime_atlas.parquet"
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _is_summary_frame(frame: pd.DataFrame) -> bool:
    return {"row_count", "ccc", "ccc_gap_to_target"}.issubset(frame.columns)


def _select_router_candidate_rows(
    summary: pd.DataFrame,
    *,
    target_ccc: float,
    min_rows: int,
) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    frame = summary.copy()
    frame["atom_family"] = frame["atom_family"].map(_normalize_family)
    frame["mechanism"] = frame["mechanism"].map(_normalize_mechanism)
    frame["quantile_bin"] = frame["quantile_bin"].astype(str)
    row_count = pd.to_numeric(frame.get("row_count"), errors="coerce").fillna(0)
    frame = frame.loc[row_count >= max(int(min_rows), 1)].copy()
    frame = frame.loc[frame["quantile_bin"].ne("Q?")].copy()
    if frame.empty:
        return frame
    frame = frame.loc[
        [
            mechanism in DEFAULT_MECHANISMS_BY_FAMILY.get(family, ())
            or mechanism in PRIMARY_MECHANISMS_BY_FAMILY.get(family, ())
            for family, mechanism in zip(frame["atom_family"], frame["mechanism"], strict=False)
        ]
    ].copy()
    priority = pd.to_numeric(
        frame.get("mean_rare_regime_priority"),
        errors="coerce",
    )
    ccc_gap = pd.to_numeric(frame.get("ccc_gap_to_target"), errors="coerce")
    if ccc_gap.isna().all() and "ccc" in frame:
        ccc_gap = float(target_ccc) - pd.to_numeric(frame["ccc"], errors="coerce")
    tail = pd.to_numeric(frame.get("tail_fraction"), errors="coerce").fillna(0.0)
    signed_consensus = pd.to_numeric(
        frame.get("mean_signed_residual_consensus"),
        errors="coerce",
    ).fillna(0.0)
    primary = pd.Series(
        [
            0.5 if mechanism in PRIMARY_MECHANISMS_BY_FAMILY.get(family, ()) else 0.0
            for family, mechanism in zip(frame["atom_family"], frame["mechanism"], strict=False)
        ],
        index=frame.index,
    )
    split_bonus = frame.get("split", pd.Series("", index=frame.index)).astype(str).map(
        {"val": 0.25, "train": 0.10}
    ).fillna(0.0)
    frame["_policy_priority"] = (
        priority.fillna(0.0)
        + ccc_gap.clip(lower=0.0).fillna(0.0)
        + tail.clip(lower=0.0).fillna(0.0)
        + signed_consensus.clip(lower=0.0, upper=1.0) * 0.35
        + primary
        + split_bonus
    )
    frame["_policy_key"] = [_policy_key(row) for _, row in frame.iterrows()]
    frame = frame.loc[frame["_policy_key"].astype(bool)].copy()
    if "mean_signed_residual_consensus" in frame:
        consensus = pd.to_numeric(
            frame["mean_signed_residual_consensus"],
            errors="coerce",
        ).fillna(0.0)
        family = frame["atom_family"].astype(str)
        bias = pd.to_numeric(frame.get("bias"), errors="coerce").fillna(0.0)
        protected = family.isin(["HN", "C'"])
        signed_safe = (consensus >= 0.20) | (bias.abs() > 1.0e-8)
        frame = frame.loc[(~protected) | signed_safe].copy()
    frame = frame.sort_values(
        ["_policy_priority", "row_count"],
        ascending=[False, False],
        kind="stable",
    )
    return frame.drop_duplicates("_policy_key", keep="first").reset_index(drop=True)


def _policy_key(row: pd.Series | dict[str, Any]) -> str:
    family = _normalize_family(row.get("atom_family", ""))
    mechanism = _normalize_mechanism(row.get("mechanism", ""))
    quantile = str(row.get("quantile_bin", "")).strip()
    if not family or not mechanism or not quantile or quantile == "Q?":
        return ""
    return f"{family}:{mechanism}:{quantile}"


def _policy_direction(row: pd.Series | dict[str, Any]) -> str:
    bias = _safe_float(row.get("bias"))
    sign = _safe_float(row.get("mean_residual_sign"))
    signal = bias if math.isfinite(bias) and abs(bias) > 1.0e-8 else sign
    if not math.isfinite(signal) or abs(signal) <= 1.0e-8:
        return "negative"
    return "negative" if signal > 0.0 else "positive"


def _policy_max_abs(row: pd.Series | dict[str, Any]) -> float:
    family = _normalize_family(row.get("atom_family", ""))
    cap = RARE_REGIME_V2_MAX_ABS_BY_FAMILY.get(family, 0.1)
    p95 = abs(_safe_float(row.get("p95_abs_residual")))
    if not math.isfinite(p95) or p95 <= 0.0:
        return float(cap)
    return float(min(cap, max(cap * 0.35, p95 * 0.10)))


def _policy_width(row: pd.Series | dict[str, Any]) -> float:
    family = _normalize_family(row.get("atom_family", ""))
    if family == "HN":
        return 0.28
    if family == "C'":
        return 0.30
    return 0.35


def _policy_uncertainty_threshold(row: pd.Series | dict[str, Any]) -> float:
    reliability = _safe_float(row.get("mean_teacher_reliability"))
    if math.isfinite(reliability) and reliability < 0.55:
        return 0.68
    return 0.84


def _row_conflict_score(row: pd.Series | dict[str, Any]) -> float:
    scores = [
        _safe_float(row.get("mean_family_conflict_score")),
        _safe_float(row.get("mean_hn_cprime_direction_conflict_score")),
    ]
    finite = [abs(score) for score in scores if math.isfinite(score)]
    return max(finite) if finite else 0.0


def _derive_residue_family_keys(atlas: pd.DataFrame, *, max_keys: int) -> list[str]:
    if max_keys <= 0:
        return []
    keys = list(RARE_REGIME_V2_RESIDUE_FAMILY_PRIOR_KEYS)
    if not atlas.empty and {"residue_name", "atom_family"}.issubset(atlas.columns):
        frame = atlas.copy()
        frame["residue_name"] = frame["residue_name"].astype(str).str.upper().str[:3]
        frame["atom_family"] = frame["atom_family"].map(_normalize_family)
        frame["abs_residual"] = pd.to_numeric(
            frame.get("abs_residual"),
            errors="coerce",
        ).fillna(0.0)
        grouped = (
            frame.loc[frame["atom_family"].isin(["N", "CA", "CB", "C'"])]
            .groupby(["residue_name", "atom_family"], dropna=False)
            .agg(
                mean_abs_residual=("abs_residual", "mean"),
                rows=("abs_residual", "size"),
            )
            .reset_index()
            .sort_values(["mean_abs_residual", "rows"], ascending=[False, False])
        )
        for _, row in grouped.iterrows():
            residue = str(row.get("residue_name", "")).strip()
            family = str(row.get("atom_family", "")).strip()
            if len(residue) != 3 or not family:
                continue
            key = f"{residue}:{family}"
            if key not in keys:
                keys.append(key)
            if len(keys) >= max_keys:
                break
    return keys[:max_keys]


def build_rare_regime_atlas(
    arrays_dir: str | Path,
    *,
    target_ccc: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return atlas, grouped summary, and operational acceptance table."""

    arrays = Path(arrays_dir)
    predictions = _prediction_frame(
        _read_frame(arrays / "bioemu_latent_nmr_masked_predictions.parquet"),
        fallback=_read_frame(arrays / "bioemu_latent_nmr_predictions.parquet"),
    )
    coordinates = _coordinate_frame(
        _read_frame(arrays / "bioemu_physics_atlas_coordinates.parquet"),
        fallback=_read_frame(arrays / "bioemu_latent_atlas_coordinates.parquet"),
    )
    atlas = _join_predictions_and_coordinates(predictions, coordinates)
    atlas = _attach_family_conflict(
        atlas,
        _read_frame(arrays / "bioemu_family_conflict_audit.parquet"),
    )
    atlas = _attach_hn_cprime_conflict(
        atlas,
        _read_frame(arrays / "bioemu_hn_cprime_direction_conflict_audit.parquet"),
    )
    atlas = _attach_teacher_reliability(
        atlas,
        _read_frame(arrays / "bioemu_cs_reweighting_teacher_family_summary.parquet"),
        _read_frame(arrays / "bioemu_cs_reweighting_teacher_source_audit.parquet"),
        _read_frame(arrays / "bioemu_ucbshift_cnnls_teacher_predictions.parquet"),
    )
    atlas = _attach_bridge(
        atlas,
        _read_frame(arrays / "bioemu_posterior_bridge_audit.parquet"),
    )
    atlas = _attach_expert_policy(
        atlas,
        _read_frame(arrays / "bioemu_residue_family_expert_policy_probe.parquet"),
        _read_frame(arrays / "bioemu_residue_family_expert_reliability_gate_probe.parquet"),
    )
    atlas = _finalize_atlas(atlas)
    summary = summarize_rare_regime_atlas(atlas, target_ccc=float(target_ccc))
    acceptance = rare_regime_acceptance(summary, atlas, target_ccc=float(target_ccc))
    return atlas, summary, acceptance


def summarize_rare_regime_atlas(
    atlas: pd.DataFrame,
    *,
    target_ccc: float = 0.95,
) -> pd.DataFrame:
    """Group rare-regime rows by family, mechanism, quantile, and ensemble state."""

    if atlas.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "split",
        "atom_family",
        "mechanism",
        "quantile_bin",
        "ensemble_state_class",
    ]
    for keys, group in atlas.groupby(group_cols, dropna=False):
        split, family, mechanism, quantile_bin, ensemble_state_class = keys
        target = _numeric(group, "target_value").to_numpy(float)
        pred = _numeric(group, "predicted_value").to_numpy(float)
        residual = _numeric(group, "residual").to_numpy(float)
        valid = np.isfinite(target) & np.isfinite(pred)
        abs_residual = np.abs(residual[np.isfinite(residual)])
        family_name = str(family)
        threshold = float(TAIL_THRESHOLDS.get(family_name, np.nanquantile(abs_residual, 0.95) if abs_residual.size else np.nan))
        tail_fraction = (
            float(np.mean(abs_residual >= threshold))
            if abs_residual.size and math.isfinite(threshold)
            else float("nan")
        )
        ccc = _ccc(target[valid], pred[valid])
        rows.append(
            {
                "split": str(split),
                "atom_family": family_name,
                "mechanism": str(mechanism),
                "quantile_bin": str(quantile_bin),
                "ensemble_state_class": str(ensemble_state_class),
                "row_count": int(len(group)),
                "ccc": ccc,
                "ccc_target": float(target_ccc),
                "ccc_gap_to_target": (
                    max(float(target_ccc) - ccc, 0.0)
                    if math.isfinite(ccc)
                    else float("nan")
                ),
                "mae": float(np.nanmean(abs_residual)) if abs_residual.size else float("nan"),
                "bias": _mean(residual),
                "p95_abs_residual": _quantile(abs_residual, 0.95),
                "tail_fraction": tail_fraction,
                "mean_residual_sign": _mean(_numeric(group, "residual_sign")),
                "mean_posterior_ess": _mean(_numeric(group, "posterior_ess")),
                "mean_posterior_entropy": _mean(_numeric(group, "posterior_entropy")),
                "mean_teacher_reliability": _mean(_numeric(group, "teacher_reliability")),
                "mean_source_mass_bioemu": _mean(_numeric(group, "source_mass_bioemu")),
                "mean_source_mass_af3": _mean(_numeric(group, "source_mass_af3")),
                "mean_source_mass_calvados2": _mean(_numeric(group, "source_mass_calvados2")),
                "mean_family_conflict_score": _mean(_numeric(group, "family_conflict_score")),
                "mean_hn_cprime_direction_conflict_score": _mean(
                    _numeric(group, "hn_cprime_direction_conflict_score")
                ),
                "mean_router_confidence": _mean(_numeric(group, "router_confidence")),
                "mean_tangent_alignment": _mean(_numeric(group, "tangent_alignment")),
                "mean_expert_policy_gain": _mean(_numeric(group, "expert_policy_gain")),
                "mean_signed_residual_consensus": _mean(
                    _numeric(group, "signed_residual_consensus")
                ),
                "mean_support_replay_priority": _mean(
                    _numeric(group, "support_replay_priority")
                ),
                "mean_rare_regime_priority": _mean(_numeric(group, "rare_regime_priority")),
                "recommended_action": _recommended_action(group),
            }
        )
    result = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not result.empty:
        result = result.sort_values(
            ["mean_rare_regime_priority", "row_count"],
            ascending=[False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return result


def ensemble_state_audit(atlas: pd.DataFrame) -> pd.DataFrame:
    """Summarize flexible/metamorphic/rigid/support-gap state evidence."""

    if atlas.empty:
        return pd.DataFrame(columns=ENSEMBLE_AUDIT_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = ["split", "atom_family", "ensemble_state_class"]
    for keys, group in atlas.groupby(group_cols, dropna=False):
        split, family, ensemble_state_class = keys
        target = _numeric(group, "target_value").to_numpy(float)
        pred = _numeric(group, "predicted_value").to_numpy(float)
        valid = np.isfinite(target) & np.isfinite(pred)
        abs_residual = _numeric(group, "abs_residual").to_numpy(float)
        abs_residual = abs_residual[np.isfinite(abs_residual)]
        rows.append(
            {
                "split": str(split),
                "atom_family": str(family),
                "ensemble_state_class": str(ensemble_state_class),
                "row_count": int(len(group)),
                "ccc": _ccc(target[valid], pred[valid]),
                "mae": (
                    float(np.nanmean(abs_residual))
                    if abs_residual.size
                    else float("nan")
                ),
                "p95_abs_residual": _quantile(abs_residual, 0.95),
                "mean_posterior_ess": _mean(_numeric(group, "posterior_ess")),
                "mean_posterior_entropy": _mean(_numeric(group, "posterior_entropy")),
                "mean_source_mass_bioemu": _mean(_numeric(group, "source_mass_bioemu")),
                "mean_support_replay_priority": _mean(
                    _numeric(group, "support_replay_priority")
                ),
                "state_action": _ensemble_state_action(group),
            }
        )
    result = pd.DataFrame(rows, columns=ENSEMBLE_AUDIT_COLUMNS)
    if not result.empty:
        result = result.sort_values(
            ["mean_support_replay_priority", "row_count"],
            ascending=[False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return result


def rare_regime_acceptance(
    summary: pd.DataFrame,
    atlas: pd.DataFrame,
    *,
    target_ccc: float = 0.95,
) -> pd.DataFrame:
    """Operational acceptance checklist for the rare-regime strategy."""

    if atlas.empty:
        return pd.DataFrame(
            [
                {
                    "criterion": "atlas materialized",
                    "status": "waiting",
                    "value": 0.0,
                    "threshold": 1.0,
                    "detail": "prediction and coordinate artifacts are not available yet",
                }
            ],
            columns=ACCEPTANCE_COLUMNS,
        )
    hn_cp = summary.loc[summary["atom_family"].isin(["HN", "C'"])]
    cb = summary.loc[summary["atom_family"].eq("CB")]
    val_summary = summary.loc[summary["split"].astype(str).eq("val")]
    if val_summary.empty:
        val_summary = summary
    min_local_ccc = _nanmin(val_summary.get("ccc"))
    mean_gap = _mean(val_summary.get("ccc_gap_to_target"))
    ess = _numeric(atlas, "posterior_ess")
    ensemble = ensemble_state_audit(atlas)
    collapsed = atlas.loc[
        atlas["ensemble_state_class"].astype(str).eq("partial_folded_rigid")
        & (_numeric(atlas, "posterior_ess") < 1.5)
    ]
    source_shortcut = atlas.loc[
        (_numeric(atlas, "top_source_mass") > 0.90)
        & atlas["top_source"].astype(str).str.lower().ne("bioemu")
    ]
    rows = [
        {
            "criterion": "CCC target 0.95",
            "status": _pass_fail(min_local_ccc, float(target_ccc), high_is_good=True),
            "value": min_local_ccc,
            "threshold": float(target_ccc),
            "detail": (
                "minimum local chart CCC is compared against the global "
                f"accuracy target {float(target_ccc):.2f}; mean gap={mean_gap:.3f}"
            ),
        },
        {
            "criterion": "HN/Cprime tail protected",
            "status": _pass_fail(_nanmax(hn_cp.get("p95_abs_residual")), 1.8, high_is_good=False),
            "value": _nanmax(hn_cp.get("p95_abs_residual")),
            "threshold": 1.8,
            "detail": "HN/Cprime p95 residual should not expand while rare adapters improve CCC",
        },
        {
            "criterion": "CB tail monitored",
            "status": _pass_fail(_nanmax(cb.get("p95_abs_residual")), 12.0, high_is_good=False),
            "value": _nanmax(cb.get("p95_abs_residual")),
            "threshold": 12.0,
            "detail": "CB is guard/calibration-first and should not become the hidden outlier sink",
        },
        {
            "criterion": "posterior ESS floor",
            "status": _pass_fail(_mean(ess), 2.5, high_is_good=True),
            "value": _mean(ess),
            "threshold": 2.5,
            "detail": "single-structure collapse is rejected even if CCC improves",
        },
        {
            "criterion": "no AF/CALVADOS-only shortcut",
            "status": "pass" if source_shortcut.empty else "warn",
            "value": float(len(source_shortcut)),
            "threshold": 0.0,
            "detail": "rows with non-BioEmu source mass above 0.90 are guard-only targets",
        },
        {
            "criterion": "rare atlas rows",
            "status": "pass" if len(atlas) > 0 else "waiting",
            "value": float(len(atlas)),
            "threshold": 1.0,
            "detail": "entity-residue-family-mechanism bins are available for viewer inspection",
        },
        {
            "criterion": "ensemble state audit",
            "status": "pass" if not ensemble.empty and collapsed.empty else "warn",
            "value": float(len(ensemble)),
            "threshold": 1.0,
            "detail": (
                "solution-state classes are materialized; partial-folded rigid "
                "states are accepted only when they are not single-structure collapse"
            ),
        },
    ]
    return pd.DataFrame(rows, columns=ACCEPTANCE_COLUMNS)


def _prediction_frame(frame: pd.DataFrame, *, fallback: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and not fallback.empty:
        frame = fallback.copy()
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    rename = {}
    if "predicted_value" not in result and "mu" in result:
        rename["mu"] = "predicted_value"
    if "target_value" not in result:
        for alias in ("observed_value", "chemical_shift", "value", "target"):
            if alias in result:
                rename[alias] = "target_value"
                break
    if "raw_predicted_value" not in result:
        for alias in ("raw_mu", "base_mu", "uncalibrated_predicted_value"):
            if alias in result:
                rename[alias] = "raw_predicted_value"
                break
    if rename:
        result = result.rename(columns=rename)
    for column in ("target_value", "predicted_value", "raw_predicted_value"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "raw_predicted_value" not in result or result["raw_predicted_value"].isna().all():
        result["raw_predicted_value"] = result["predicted_value"]
    if "target_id" in result:
        parsed = result["target_id"].map(_parse_target)
        for column in ("residue_index", "residue_name", "atom_family"):
            if column not in result or result[column].isna().all():
                result[column] = parsed.map(lambda item, key=column: item.get(key))
    for column in ("split", "entity_uid", "bmrb_id", "target_id", "residue_name", "atom_family"):
        if column not in result:
            result[column] = ""
    if "residue_index" not in result:
        result["residue_index"] = 0
    result["residue_index"] = pd.to_numeric(result["residue_index"], errors="coerce").fillna(0).astype(int)
    result["atom_family"] = result["atom_family"].map(_normalize_family)
    result["residual"] = result["predicted_value"] - result["target_value"]
    result["abs_residual"] = result["residual"].abs()
    result["residual_sign"] = np.sign(result["residual"].fillna(0.0))
    for column in ("posterior_sigma", "posterior_ess", "posterior_entropy"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    keep = [
        "split",
        "entity_uid",
        "bmrb_id",
        "target_id",
        "residue_index",
        "residue_name",
        "atom_family",
        "target_value",
        "predicted_value",
        "raw_predicted_value",
        "residual",
        "abs_residual",
        "residual_sign",
        "posterior_sigma",
        "posterior_ess",
        "posterior_entropy",
    ]
    return result[[column for column in keep if column in result]].copy()


def _coordinate_frame(frame: pd.DataFrame, *, fallback: pd.DataFrame) -> pd.DataFrame:
    result = _normalise_coordinate_frame(frame)
    if result.empty:
        result = _normalise_coordinate_frame(fallback)
    return result


def _normalise_coordinate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if "mechanism" in result and "coordinate_name" not in result:
        result = result.rename(columns={"mechanism": "coordinate_name"})
    if "factor_name" in result and "coordinate_name" not in result:
        result = result.rename(columns={"factor_name": "coordinate_name"})
    if "value" in result and "coordinate_value" not in result:
        result = result.rename(columns={"value": "coordinate_value"})
    known = sorted(
        {
            mechanism
            for mechanisms in DEFAULT_MECHANISMS_BY_FAMILY.values()
            for mechanism in mechanisms
        }
        | set(RARE_REGIME_V2_COORDINATE_NAMES)
        | {"ring_current", "graph_fragility", "bioemu_component", "uncertainty_abstain"}
    )
    if "coordinate_name" not in result or "coordinate_value" not in result:
        id_cols = [
            column
            for column in (
                "split",
                "entity_uid",
                "bmrb_id",
                "target_id",
                "residue_index",
                "residue_name",
                "atom_family",
                "reliability",
                "uncertainty",
                "chart_probability",
            )
            if column in result
        ]
        value_cols = [column for column in known if column in result]
        if value_cols:
            result = result.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name="coordinate_name",
                value_name="coordinate_value",
            )
    if "coordinate_name" not in result or "coordinate_value" not in result:
        return pd.DataFrame()
    if "target_id" in result:
        parsed = result["target_id"].map(_parse_target)
        for column in ("residue_index", "residue_name", "atom_family"):
            if column not in result or result[column].isna().all():
                result[column] = parsed.map(lambda item, key=column: item.get(key))
    for column in ("split", "entity_uid", "bmrb_id", "target_id", "residue_name", "atom_family"):
        if column not in result:
            result[column] = ""
    if "residue_index" not in result:
        result["residue_index"] = 0
    for column in ("reliability", "uncertainty", "chart_probability"):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["coordinate_value"] = pd.to_numeric(result["coordinate_value"], errors="coerce")
    result["coordinate_name"] = result["coordinate_name"].map(_normalize_mechanism)
    result["atom_family"] = result["atom_family"].map(_normalize_family)
    return result[
        [
            "split",
            "entity_uid",
            "bmrb_id",
            "target_id",
            "residue_index",
            "residue_name",
            "atom_family",
            "coordinate_name",
            "coordinate_value",
            "reliability",
            "uncertainty",
            "chart_probability",
        ]
    ].copy()


def _join_predictions_and_coordinates(
    predictions: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty and coordinates.empty:
        return pd.DataFrame(columns=ATLAS_COLUMNS)
    if predictions.empty:
        base = coordinates.drop_duplicates(
            ["split", "entity_uid", "target_id", "atom_family", "coordinate_name"],
            keep="first",
        ).copy()
        for column in (
            "target_value",
            "predicted_value",
            "raw_predicted_value",
            "residual",
            "abs_residual",
            "residual_sign",
            "posterior_sigma",
            "posterior_ess",
            "posterior_entropy",
        ):
            base[column] = np.nan
        base = base.rename(columns={"coordinate_name": "mechanism"})
        return base
    if coordinates.empty:
        rows: list[dict[str, Any]] = []
        for _, row in predictions.iterrows():
            family = str(row.get("atom_family"))
            for mechanism in DEFAULT_MECHANISMS_BY_FAMILY.get(family, ("residue_family",)):
                item = row.to_dict()
                item["mechanism"] = mechanism
                item["coordinate_value"] = row.get("abs_residual")
                item["reliability"] = np.nan
                item["uncertainty"] = np.nan
                item["chart_probability"] = np.nan
                rows.append(item)
        return pd.DataFrame(rows)
    key_candidates = [
        ["split", "entity_uid", "target_id", "atom_family"],
        ["entity_uid", "target_id", "atom_family"],
        ["target_id", "atom_family"],
    ]
    keys = next(
        (
            candidate
            for candidate in key_candidates
            if set(candidate).issubset(predictions.columns)
            and set(candidate).issubset(coordinates.columns)
        ),
        [],
    )
    if not keys:
        return _join_predictions_and_coordinates(predictions, pd.DataFrame())
    merged = predictions.merge(
        coordinates,
        on=keys,
        how="left",
        suffixes=("", "_coord"),
    )
    for column in ("split", "entity_uid", "bmrb_id", "residue_index", "residue_name"):
        coord_column = f"{column}_coord"
        if coord_column in merged:
            merged[column] = merged[column].where(merged[column].notna() & merged[column].astype(str).ne(""), merged[coord_column])
    missing = merged["coordinate_name"].isna() if "coordinate_name" in merged else pd.Series(True, index=merged.index)
    if missing.any():
        synthetic = _join_predictions_and_coordinates(merged.loc[missing, predictions.columns], pd.DataFrame())
        merged = pd.concat([merged.loc[~missing], synthetic], ignore_index=True, sort=False)
    merged = merged.rename(columns={"coordinate_name": "mechanism"})
    return merged


def _attach_family_conflict(atlas: pd.DataFrame, conflict: pd.DataFrame) -> pd.DataFrame:
    result = atlas.copy()
    result["family_conflict_score"] = np.nan
    if conflict.empty or "atom_family" not in conflict or result.empty:
        return result
    score_col = _pick_col(
        conflict,
        [
            "family_conflict_score",
            "direction_conflict_score",
            "gradient_conflict_score",
            "conflict_score",
            "opposition_rate",
        ],
    )
    if score_col is None:
        return result
    summary = (
        conflict.assign(atom_family=conflict["atom_family"].map(_normalize_family))
        .groupby("atom_family", dropna=False)[score_col]
        .mean()
        .rename("family_conflict_score")
        .reset_index()
    )
    return result.drop(columns=["family_conflict_score"]).merge(
        summary,
        on="atom_family",
        how="left",
    )


def _attach_hn_cprime_conflict(atlas: pd.DataFrame, conflict: pd.DataFrame) -> pd.DataFrame:
    result = atlas.copy()
    result["hn_cprime_direction_conflict_score"] = np.nan
    if conflict.empty or result.empty:
        return result
    frame = conflict.copy()
    if "mechanism" not in frame and "coordinate_name" in frame:
        frame = frame.rename(columns={"coordinate_name": "mechanism"})
    if "mechanism" not in frame:
        return result
    frame["mechanism"] = frame["mechanism"].map(_normalize_mechanism)
    score_col = _pick_col(
        frame,
        [
            "hn_cprime_direction_conflict_score",
            "direction_conflict_score",
            "conflict_score",
            "opposed_fraction",
        ],
    )
    if score_col is None:
        return result
    group_cols = ["mechanism"]
    if "atom_family" in frame:
        frame["atom_family"] = frame["atom_family"].map(_normalize_family)
        group_cols = ["atom_family", "mechanism"]
    summary = frame.groupby(group_cols, dropna=False)[score_col].mean().rename(
        "hn_cprime_direction_conflict_score"
    ).reset_index()
    merge_cols = [column for column in group_cols if column in result]
    return result.drop(columns=["hn_cprime_direction_conflict_score"]).merge(
        summary,
        on=merge_cols,
        how="left",
    )


def _attach_teacher_reliability(
    atlas: pd.DataFrame,
    family_summary: pd.DataFrame,
    source_audit: pd.DataFrame,
    teacher_predictions: pd.DataFrame,
) -> pd.DataFrame:
    result = atlas.copy()
    for column in ("teacher_ess", "teacher_entropy", "teacher_reliability", *SOURCE_COLUMNS):
        result[column] = np.nan
    if not teacher_predictions.empty:
        result = _merge_teacher_table(result, teacher_predictions)
    if not source_audit.empty:
        result = _merge_teacher_table(result, source_audit)
    if not family_summary.empty and "atom_family" in family_summary:
        frame = family_summary.copy()
        frame["atom_family"] = frame["atom_family"].map(_normalize_family)
        rename: dict[str, str] = {}
        if "mean_teacher_ess" in frame and "teacher_ess" not in frame:
            rename["mean_teacher_ess"] = "teacher_ess"
        if "mean_teacher_entropy" in frame and "teacher_entropy" not in frame:
            rename["mean_teacher_entropy"] = "teacher_entropy"
        if "mean_reliability" in frame and "teacher_reliability" not in frame:
            rename["mean_reliability"] = "teacher_reliability"
        frame = frame.rename(columns=rename)
        keep = [
            column
            for column in (
                "atom_family",
                "teacher_ess",
                "teacher_entropy",
                "teacher_reliability",
                *SOURCE_COLUMNS,
            )
            if column in frame
        ]
        summary = frame[keep].groupby("atom_family", dropna=False).mean(numeric_only=True).reset_index()
        result = _coalesce_merge(result, summary, on=["atom_family"])
    return result


def _merge_teacher_table(result: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
    frame = teacher.copy()
    if "atom_family" in frame:
        frame["atom_family"] = frame["atom_family"].map(_normalize_family)
    rename = {}
    if "teacher_mean" in frame and "predicted_value" in frame:
        pass
    if "ess" in frame and "teacher_ess" not in frame:
        rename["ess"] = "teacher_ess"
    if "entropy" in frame and "teacher_entropy" not in frame:
        rename["entropy"] = "teacher_entropy"
    if "reliability" in frame and "teacher_reliability" not in frame:
        rename["reliability"] = "teacher_reliability"
    frame = frame.rename(columns=rename)
    keep = [
        column
        for column in (
            "split",
            "entity_uid",
            "target_id",
            "atom_family",
            "teacher_ess",
            "teacher_entropy",
            "teacher_reliability",
            *SOURCE_COLUMNS,
        )
        if column in frame
    ]
    if "target_id" not in keep:
        return result
    frame = frame[keep].copy()
    key_candidates = [
        ["split", "entity_uid", "target_id", "atom_family"],
        ["entity_uid", "target_id", "atom_family"],
        ["target_id", "atom_family"],
        ["target_id"],
    ]
    keys = next(
        (
            candidate
            for candidate in key_candidates
            if set(candidate).issubset(result.columns) and set(candidate).issubset(frame.columns)
        ),
        [],
    )
    if not keys:
        return result
    summary = frame.groupby(keys, dropna=False).mean(numeric_only=True).reset_index()
    return _coalesce_merge(result, summary, on=keys)


def _attach_bridge(atlas: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    result = atlas.copy()
    if "posterior_bridge_norm" not in result:
        result["posterior_bridge_norm"] = np.nan
    if bridge.empty:
        return result
    frame = bridge.copy()
    if "bridge_norm" in frame and "posterior_bridge_norm" not in frame:
        frame = frame.rename(columns={"bridge_norm": "posterior_bridge_norm"})
    keep = [
        column
        for column in (
            "split",
            "entity_uid",
            "target_id",
            "atom_family",
            "posterior_bridge_norm",
            "chart_entropy",
            "chart_ess",
            "max_chart_probability",
        )
        if column in frame
    ]
    if "target_id" not in keep:
        return result
    if "atom_family" in frame:
        frame["atom_family"] = frame["atom_family"].map(_normalize_family)
    frame = frame[keep].copy()
    if "chart_entropy" in frame and "posterior_entropy" not in result:
        result["posterior_entropy"] = np.nan
    if "chart_ess" in frame and "posterior_ess" not in result:
        result["posterior_ess"] = np.nan
    rename = {"chart_entropy": "posterior_entropy", "chart_ess": "posterior_ess"}
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame})
    keys = [
        column
        for column in ("split", "entity_uid", "target_id", "atom_family")
        if column in result and column in frame
    ]
    if not keys:
        keys = ["target_id"] if "target_id" in result and "target_id" in frame else []
    if not keys:
        return result
    frame = frame.groupby(keys, dropna=False).mean(numeric_only=True).reset_index()
    return _coalesce_merge(result, frame, on=keys)


def _attach_expert_policy(
    atlas: pd.DataFrame,
    policy: pd.DataFrame,
    reliability_gate: pd.DataFrame,
) -> pd.DataFrame:
    result = atlas.copy()
    result["expert_policy_gain"] = np.nan
    probes = [frame for frame in (policy, reliability_gate) if not frame.empty]
    if not probes:
        return result
    rows = []
    for frame in probes:
        data = frame.copy()
        if "atom_family" not in data:
            continue
        data["atom_family"] = data["atom_family"].map(_normalize_family)
        gain_col = _pick_col(
            data,
            [
                "policy_gain",
                "delta_ccc",
                "ccc_gain",
                "policy_minus_guarded_ccc",
                "expert_gain",
            ],
        )
        if gain_col is None:
            ccc_cols = [column for column in data.columns if column.endswith("_ccc") or "ccc" in column]
            if len(ccc_cols) >= 2:
                data["expert_policy_gain"] = (
                    pd.to_numeric(data[ccc_cols[0]], errors="coerce")
                    - pd.to_numeric(data[ccc_cols[1]], errors="coerce")
                )
                gain_col = "expert_policy_gain"
        if gain_col is None:
            continue
        for _, row in data.iterrows():
            rows.append(
                {
                    "atom_family": row.get("atom_family"),
                    "expert_policy_gain": _safe_float(row.get(gain_col)),
                }
            )
    if not rows:
        return result
    summary = pd.DataFrame(rows).groupby("atom_family", dropna=False).mean(numeric_only=True).reset_index()
    return result.drop(columns=["expert_policy_gain"]).merge(summary, on="atom_family", how="left")


def _finalize_atlas(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame(columns=ATLAS_COLUMNS)
    result = atlas.copy()
    if "mechanism" not in result:
        result["mechanism"] = "residue_family"
    result["mechanism"] = result["mechanism"].map(_normalize_mechanism)
    result["coordinate_value"] = pd.to_numeric(result.get("coordinate_value"), errors="coerce")
    if result["coordinate_value"].isna().all() and "abs_residual" in result:
        result["coordinate_value"] = pd.to_numeric(result["abs_residual"], errors="coerce")
    result["coordinate_abs"] = result["coordinate_value"].abs()
    result = _add_quantile_bins(result)
    for column in ATLAS_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result["atom_family"] = result["atom_family"].map(_normalize_family)
    result["primary_physics_role"] = [
        "primary" if _is_primary_role(family, mechanism) else "support"
        for family, mechanism in zip(result["atom_family"], result["mechanism"], strict=False)
    ]
    source_values = result[SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    result["top_source_mass"] = source_values.max(axis=1)
    top_source_index = source_values.to_numpy(dtype=float)
    labels = np.array(["bioemu", "af3", "calvados2"], dtype=object)
    top_labels = []
    for row in top_source_index:
        if np.isfinite(row).any():
            top_labels.append(labels[int(np.nanargmax(row))])
        else:
            top_labels.append("")
    result["top_source"] = top_labels
    result["teacher_reliability"] = _teacher_reliability(result)
    result["router_confidence"] = _router_confidence(result)
    result["tangent_alignment"] = _tangent_alignment(result)
    result["teacher_action"] = [
        _teacher_action(family, reliability, top_source, top_mass, ess)
        for family, reliability, top_source, top_mass, ess in zip(
            result["atom_family"],
            result["teacher_reliability"],
            result["top_source"],
            result["top_source_mass"],
            result["teacher_ess"],
            strict=False,
        )
    ]
    result["ensemble_state_class"] = _ensemble_state_class(result)
    result["signed_residual_consensus"] = _signed_residual_consensus(result)
    result["support_replay_priority"] = _support_replay_priority(result)
    result["rare_regime_priority"] = _rare_priority(result)
    result = result[ATLAS_COLUMNS].copy()
    result = result.sort_values(
        ["rare_regime_priority", "abs_residual"],
        ascending=[False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return result


def _teacher_reliability(frame: pd.DataFrame) -> pd.Series:
    existing = pd.to_numeric(frame.get("teacher_reliability"), errors="coerce")
    ess = pd.to_numeric(frame.get("teacher_ess"), errors="coerce")
    entropy = pd.to_numeric(frame.get("teacher_entropy"), errors="coerce")
    bioemu_mass = pd.to_numeric(frame.get("source_mass_bioemu"), errors="coerce")
    ess_score = np.clip(ess.to_numpy(float) / 20.0, 0.0, 1.0)
    entropy_score = np.clip(entropy.to_numpy(float) / 1.0, 0.0, 1.0)
    source_score = np.where(np.isfinite(bioemu_mass.to_numpy(float)), bioemu_mass.to_numpy(float), 0.5)
    combined = np.nanmean(np.vstack([ess_score, entropy_score, source_score]), axis=0)
    return existing.where(existing.notna(), pd.Series(combined, index=frame.index)).clip(0.0, 1.0)


def _router_confidence(frame: pd.DataFrame) -> pd.Series:
    reliability = pd.to_numeric(frame.get("teacher_reliability"), errors="coerce").fillna(0.5)
    uncertainty = pd.to_numeric(frame.get("uncertainty"), errors="coerce").fillna(0.5)
    chart_prob = pd.to_numeric(frame.get("chart_probability"), errors="coerce").fillna(0.5)
    conflict = pd.to_numeric(frame.get("family_conflict_score"), errors="coerce").fillna(0.0)
    confidence = reliability * (1.0 - uncertainty.clip(0.0, 1.0)) * chart_prob.clip(0.0, 1.0)
    confidence = confidence * (1.0 - conflict.clip(0.0, 1.0) * 0.5)
    return confidence.clip(0.0, 1.0)


def _tangent_alignment(frame: pd.DataFrame) -> pd.Series:
    residual_sign = pd.to_numeric(frame.get("residual_sign"), errors="coerce")
    score = (
        pd.to_numeric(frame.get("expert_policy_gain"), errors="coerce").fillna(0.0)
        - pd.to_numeric(frame.get("family_conflict_score"), errors="coerce").fillna(0.0)
    )
    sign = np.sign(score.to_numpy(float))
    return pd.Series(sign * residual_sign.fillna(0.0).to_numpy(float), index=frame.index)


def _ensemble_state_class(frame: pd.DataFrame) -> pd.Series:
    ess = pd.to_numeric(frame.get("posterior_ess"), errors="coerce")
    entropy = pd.to_numeric(frame.get("posterior_entropy"), errors="coerce")
    bridge = pd.to_numeric(frame.get("posterior_bridge_norm"), errors="coerce")
    top_source = frame.get("top_source", pd.Series("", index=frame.index)).astype(str).str.lower()
    top_mass = pd.to_numeric(frame.get("top_source_mass"), errors="coerce")

    values = np.full(len(frame), "fully_flexible", dtype=object)
    source_gap = top_source.ne("bioemu") & top_mass.gt(0.90)
    partial_rigid = ess.lt(2.5) | entropy.lt(0.35)
    finite_bridge = bridge[np.isfinite(bridge)]
    bridge_q75 = float(finite_bridge.quantile(0.75)) if len(finite_bridge) else float("nan")
    metamorphic = (
        (ess.ge(6.0) & entropy.ge(0.80))
        | (
            bridge.ge(bridge_q75)
            if math.isfinite(bridge_q75)
            else pd.Series(False, index=frame.index)
        )
    )
    values[metamorphic.fillna(False).to_numpy(bool)] = "metamorphic_switching"
    values[partial_rigid.fillna(False).to_numpy(bool)] = "partial_folded_rigid"
    values[source_gap.fillna(False).to_numpy(bool)] = "support_gap"
    return pd.Series(values, index=frame.index)


def _signed_residual_consensus(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "residual_sign" not in frame:
        return pd.Series(dtype=float)
    result = pd.Series(0.0, index=frame.index, dtype=float)
    group_cols = [
        column
        for column in (
            "split",
            "atom_family",
            "mechanism",
            "quantile_bin",
            "ensemble_state_class",
        )
        if column in frame
    ]
    signs = pd.to_numeric(frame["residual_sign"], errors="coerce").fillna(0.0)
    if not group_cols:
        consensus = abs(float(signs.mean())) if len(signs) else 0.0
        result.loc[:] = consensus
        return result.clip(0.0, 1.0)
    for _, index in frame.groupby(group_cols, dropna=False).groups.items():
        group_signs = signs.loc[index]
        result.loc[index] = abs(float(group_signs.mean())) if len(group_signs) else 0.0
    return result.clip(0.0, 1.0)


def _support_replay_priority(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    family = frame["atom_family"].astype(str)
    residue = frame.get("residue_name", pd.Series("", index=frame.index)).astype(str).str.upper().str[:3]
    pair = residue + ":" + family
    pair_boost = pair.isin(RARE_REGIME_V2_RESIDUE_FAMILY_PRIOR_KEYS).astype(float)
    target_family = family.isin(["N", "CA", "CB"]).astype(float)
    abs_res = pd.to_numeric(frame.get("abs_residual"), errors="coerce").fillna(0.0)
    scale = family.map(lambda value: TAIL_THRESHOLDS.get(str(value), np.nan)).astype(float)
    fallback_scale = abs_res.quantile(0.95) if len(abs_res) else 1.0
    scale = scale.replace(0.0, np.nan).fillna(fallback_scale).clip(lower=1.0e-6)
    tail = (abs_res / scale).clip(0.0, 4.0)
    ess = pd.to_numeric(frame.get("posterior_ess"), errors="coerce")
    low_ess = (1.0 - (ess / 8.0).clip(0.0, 1.0)).fillna(0.25)
    support_gap = frame.get(
        "ensemble_state_class",
        pd.Series("", index=frame.index),
    ).astype(str).eq("support_gap").astype(float)
    replay = target_family * (0.5 + pair_boost) * tail * (1.0 + 0.5 * low_ess + support_gap)
    return replay.astype(float)


def _rare_priority(frame: pd.DataFrame) -> pd.Series:
    abs_res = pd.to_numeric(frame.get("abs_residual"), errors="coerce").fillna(0.0)
    by_family_scale = frame["atom_family"].map(lambda family: TAIL_THRESHOLDS.get(str(family), np.nan)).astype(float)
    scale = by_family_scale.replace(0.0, np.nan).fillna(abs_res.quantile(0.95) if len(abs_res) else 1.0)
    tail_score = (abs_res / scale.clip(lower=1.0e-6)).clip(0.0, 4.0)
    conflict = pd.to_numeric(frame.get("family_conflict_score"), errors="coerce").fillna(0.0)
    hn_cprime = pd.to_numeric(frame.get("hn_cprime_direction_conflict_score"), errors="coerce").fillna(0.0)
    router = pd.to_numeric(frame.get("router_confidence"), errors="coerce").fillna(0.5)
    ess = pd.to_numeric(frame.get("posterior_ess"), errors="coerce")
    ess_penalty = (1.0 - (ess / 20.0).clip(0.0, 1.0)).fillna(0.5)
    role_boost = frame["primary_physics_role"].astype(str).eq("primary").astype(float) * 0.4
    state = frame.get("ensemble_state_class", pd.Series("", index=frame.index)).astype(str)
    state_boost = (
        state.map(
            {
                "support_gap": 0.70,
                "partial_folded_rigid": 0.40,
                "metamorphic_switching": 0.25,
                "fully_flexible": 0.0,
            }
        )
        .fillna(0.0)
        .astype(float)
    )
    replay = pd.to_numeric(
        frame.get("support_replay_priority"),
        errors="coerce",
    ).fillna(0.0)
    consensus = pd.to_numeric(
        frame.get("signed_residual_consensus"),
        errors="coerce",
    ).fillna(0.0)
    return (
        tail_score
        + conflict
        + hn_cprime
        + (1.0 - router)
        + ess_penalty
        + role_boost
        + state_boost
        + replay.clip(0.0, 2.0) * 0.25
        + consensus.clip(0.0, 1.0) * 0.15
    )


def _add_quantile_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["quantile_bin"] = "Q?"
    for _, index in result.groupby(["atom_family", "mechanism"], dropna=False).groups.items():
        values = pd.to_numeric(result.loc[index, "coordinate_value"], errors="coerce")
        if values.notna().sum() < 4 or values.nunique(dropna=True) < 2:
            result.loc[index, "quantile_bin"] = "Q?"
            continue
        ranks = values.rank(pct=True, method="average")
        labels = np.where(
            ranks <= 0.25,
            "Q1",
            np.where(ranks <= 0.50, "Q2", np.where(ranks <= 0.75, "Q3", "Q4")),
        )
        result.loc[index, "quantile_bin"] = labels
    return result


def _coalesce_merge(left: pd.DataFrame, right: pd.DataFrame, *, on: list[str]) -> pd.DataFrame:
    merged = left.merge(right, on=on, how="left", suffixes=("", "_new"))
    for column in list(right.columns):
        if column in on:
            continue
        new_column = f"{column}_new"
        if new_column not in merged:
            continue
        if column in merged:
            merged[column] = merged[column].where(merged[column].notna(), merged[new_column])
            merged = merged.drop(columns=[new_column])
        else:
            merged = merged.rename(columns={new_column: column})
    return merged


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".tsv"), sep="\t", index=False)
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        pass


def _read_frame(path: Path) -> pd.DataFrame:
    tsv = path.with_suffix(".tsv")
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    if tsv.exists():
        try:
            return pd.read_csv(tsv, sep="\t")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _parse_target(target_id: str) -> dict[str, Any]:
    return dict(parse_chemical_shift_target_id(str(target_id)))


def _normalize_family(value: Any) -> str:
    family = str(value).strip()
    upper = family.upper().replace(" ", "")
    if upper in {"C", "C'", "CPRIME", "C_PRIME", "C-PRIME"}:
        return "C'"
    if upper in {"H", "HN", "H-N"}:
        return "HN"
    return upper


def _normalize_mechanism(value: Any) -> str:
    name = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "electrostatic_sulfur": "sulfur_electrostatic",
        "sulfur_electrostatics": "sulfur_electrostatic",
        "ensemble_component": "bioemu_component",
        "ring": "ring_orientation",
        "hbond": "exchange_hbond",
    }
    return aliases.get(name, name)


def _is_primary_role(family: Any, mechanism: Any) -> bool:
    family_name = _normalize_family(family)
    mechanism_name = _normalize_mechanism(mechanism)
    return mechanism_name in PRIMARY_MECHANISMS_BY_FAMILY.get(family_name, ())


def _teacher_action(
    family: Any,
    reliability: Any,
    top_source: Any,
    top_mass: Any,
    ess: Any,
) -> str:
    family_name = _normalize_family(family)
    reliability_value = _safe_float(reliability)
    top_mass_value = _safe_float(top_mass)
    ess_value = _safe_float(ess)
    if family_name in {"HN", "C'"}:
        return "guard_or_low_weight"
    if math.isfinite(ess_value) and ess_value < 20.0:
        return "guard_only"
    if str(top_source).lower() != "bioemu" and math.isfinite(top_mass_value) and top_mass_value > 0.90:
        return "guard_only"
    if math.isfinite(reliability_value) and reliability_value < 0.35:
        return "uncertainty_only"
    return "mean_low_weight"


def _recommended_action(group: pd.DataFrame) -> str:
    family = str(group["atom_family"].iloc[0])
    mechanism = str(group["mechanism"].iloc[0])
    ensemble_state = (
        str(group["ensemble_state_class"].mode().iloc[0])
        if "ensemble_state_class" in group and not group["ensemble_state_class"].empty
        else ""
    )
    signed_consensus = _mean(_numeric(group, "signed_residual_consensus"))
    teacher_action = str(group["teacher_action"].mode().iloc[0]) if "teacher_action" in group and not group["teacher_action"].empty else ""
    if ensemble_state == "support_gap":
        return "support-gap guard; replay train analogs, no source shortcut mean"
    if family in {"N", "CA", "CB"}:
        return "support replay + non-regression guard"
    if teacher_action in {"guard_only", "uncertainty_only"}:
        return "guard target; do not distill mean"
    if family in {"HN", "C'"}:
        if math.isfinite(signed_consensus) and signed_consensus < 0.20:
            return "signed guard only until residual direction is stable"
        return "local chart adapter + posterior entropy guard"
    if mechanism == "residue_family":
        return "residue-family expert with shrinkage"
    return "calibrate local mechanism chart"


def _ensemble_state_action(group: pd.DataFrame) -> str:
    state = (
        str(group["ensemble_state_class"].mode().iloc[0])
        if "ensemble_state_class" in group and not group["ensemble_state_class"].empty
        else ""
    )
    if state == "support_gap":
        return "guard source-dominated rows and replay same residue-family patterns in train"
    if state == "partial_folded_rigid":
        return "allow rigid subpopulation only with ESS/diversity non-collapse checks"
    if state == "metamorphic_switching":
        return "route as state-switching ensemble; preserve entropy and occupancy diversity"
    return "preserve flexible ensemble entropy while fitting NMR posterior mean"


def _pick_col(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _mean(values: Any) -> float:
    data = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    if data.size == 0:
        return float("nan")
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    result = float(np.mean(data))
    return result if math.isfinite(result) else float("nan")


def _quantile(values: Any, q: float) -> float:
    data = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.nanquantile(data, q))


def _nanmax(values: Any) -> float:
    if values is None:
        return float("nan")
    data = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.nanmax(data))


def _nanmin(values: Any) -> float:
    if values is None:
        return float("nan")
    data = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.nanmin(data))


def _pass_fail(value: float, threshold: float, *, high_is_good: bool) -> str:
    if not math.isfinite(value):
        return "waiting"
    return "pass" if (value >= threshold if high_is_good else value <= threshold) else "warn"


def _ccc(target: np.ndarray, pred: np.ndarray) -> float:
    if target.size < 2 or pred.size < 2:
        return float("nan")
    target = target.astype(float)
    pred = pred.astype(float)
    mt = float(np.mean(target))
    mp = float(np.mean(pred))
    vt = float(np.var(target))
    vp = float(np.var(pred))
    cov = float(np.mean((target - mt) * (pred - mp)))
    denom = vt + vp + (mt - mp) ** 2
    if denom <= 0:
        return float("nan")
    return 2.0 * cov / denom
