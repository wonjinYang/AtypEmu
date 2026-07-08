"""Streamlit training monitor for AtypEmu runs."""

from __future__ import annotations

import argparse
import base64
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.monitoring.loaders import load_monitor_snapshot
from atypemu.monitoring.plots import (
    ENSEMBLE_LOCAL_CONFIDENCE_LABEL,
    available_ensemble_state_ids,
    available_default_metrics,
    available_joint_atom_sets,
    available_landscape_color_columns,
    build_structure_viewer_html,
    ccc_family_gap_figure,
    ccc_oracle_vs_model_figure,
    chemical_shift_scatter_figure,
    chemical_shift_legend_rows,
    chemical_shift_uncertainty_figure,
    confidence_legend_rows,
    coverage_table,
    ensemble_state_mass_figure,
    ensemble_state_scatter_figure,
    joint_posterior_parallel_figure,
    metric_curve_figure,
    moment_consistency_figure,
    moment_oracle_table,
    moment_target_figure,
    observable_conflict_figure,
    observable_support_figure,
    observable_support_table,
    posterior_state_occupancy_figure,
    projection_scatter_figure,
    pseudo_hsqc_figure,
    rci_legend_rows,
    sample_support_figure,
)


DEFAULT_REMOTE_BASE = Path("/scratch/wkyu514/yang07/atypemu")
DEFAULT_RUN_DIR = (
    DEFAULT_REMOTE_BASE / "results" / "baseline_ucbshift_rebuild_seed7_rerun"
)
DEFAULT_INTEGRATED_ROOT = DEFAULT_REMOTE_BASE / "data" / "integrated"
DEFAULT_PANEL_LABELS = (
    "Metrics",
    "Analysis PNGs",
    "CCC Geometry",
    "Multi-Observable",
    "Moment Posterior",
    "BioEmu Atlas",
    "Energy Landscape",
    "Conformation Uncertainty",
    "NMR Chemical Shifts",
    "Raw Reports",
)
BIOEMU_FAMILY_ORDER = ("HN", "C'", "N", "CA", "CB")
BIOEMU_FAMILY_COLOR = {
    "HN": "#2563eb",
    "C'": "#059669",
    "N": "#7c3aed",
    "CA": "#d97706",
    "CB": "#dc2626",
    "HN/C' combined": "#111827",
}
BIOEMU_LATEST_ANALYSIS_PNG_GROUPS = (
    (
        "Panel-Level Analysis Data",
        (
            "bioemu_panel_ccc_geometry.png",
            "bioemu_panel_multi_observable.png",
            "bioemu_panel_moment_posterior.png",
            "bioemu_panel_bioemu_atlas.png",
            "bioemu_panel_energy_landscape.png",
            "bioemu_panel_conformation_uncertainty.png",
            "bioemu_panel_nmr_chemical_shifts.png",
        ),
    ),
    (
        "Current Training Health",
        (
            "bioemu_section_training_overview.png",
            "bioemu_training_artifact_consistency_audit.png",
            "bioemu_live_epoch_progress.png",
            "bioemu_history_ccc_progress.png",
            "bioemu_history_hn_cprime_tradeoff.png",
            "bioemu_history_loss_guardrail_progress.png",
            "bioemu_history_stability_progress.png",
        ),
    ),
    (
        "Current CCC95 Target",
        (
            "bioemu_section_family_scorecard.png",
            "bioemu_family_ccc95_gap.png",
            "bioemu_entity_macro_action_plan.png",
            "bioemu_guarded_training_roadmap.png",
            "bioemu_guarded_loss_schedule.png",
            "bioemu_next_run_guarded_config_proposal.png",
            "bioemu_guarded_config_readiness.png",
            "bioemu_hn_ca_intervention_targets.png",
            "bioemu_hn_local_chart_targets.png",
            "bioemu_hn_crossfit_local_calibration.png",
            "bioemu_hn_signed_residual_chart_router.png",
            "bioemu_ca_entity_guard_targets.png",
            "bioemu_cprime_low_n_guard_targets.png",
            "bioemu_n_low_ess_guard_targets.png",
            "bioemu_cb_tail_guard_targets.png",
            "bioemu_recommended_family_ccc95_gap.png",
            "bioemu_family_shrinkage_bias.png",
            "bioemu_family_sigma_error_calibration.png",
            "bioemu_family_sigma_error_binned_calibration.png",
            "bioemu_family_residual_histograms.png",
            "bioemu_worst_entry_family_ccc95_gap.png",
            "bioemu_all_family_outlier_tail_summary.png",
            "bioemu_all_family_outlier_residue_heatmap.png",
            "bioemu_all_family_outlier_worst_rows.png",
            "bioemu_all_family_entity_reliability_audit.png",
            "bioemu_all_family_reliability_stratified_metrics.png",
            "bioemu_all_family_reliability_action_plan.png",
            "bioemu_detached_support_expansion_action_plan.png",
            "bioemu_detached_conformer_family_quality_guard.png",
        ),
    ),
    (
        "Current Masked Chemical Shifts",
        (
            "bioemu_section_shift_residual_map.png",
            "bioemu_latent_hn_masked_scatter.png",
            "bioemu_latent_cprime_masked_scatter.png",
            "bioemu_latent_hn_cprime_target_progress.png",
            "bioemu_latent_all_family_progress.png",
            "bioemu_latent_n_masked_scatter.png",
            "bioemu_latent_ca_masked_scatter.png",
            "bioemu_latent_cb_masked_scatter.png",
            "bioemu_latent_val_hn_masked_scatter.png",
            "bioemu_latent_val_cprime_masked_scatter.png",
            "bioemu_latent_val_hn_cprime_target_progress.png",
            "bioemu_latent_val_all_family_progress.png",
        ),
    ),
    (
        "Current v101 Physics Bottlenecks",
        (
            "bioemu_section_physics_bottleneck_map.png",
            "bioemu_mechanism_signed_direction_audit.png",
            "bioemu_hn_ring_signed_subregime_audit.png",
            "bioemu_cprime_carbonyl_subregime_audit.png",
            "bioemu_ucbshift_factor_atlas_summary.png",
            "bioemu_ucbshift_factor_atlas_hn_ring_bins.png",
            "bioemu_ucbshift_factor_atlas_cprime_carbonyl_bins.png",
            "bioemu_ucbshift_factor_atlas_hn_cprime_conflict.png",
            "bioemu_ring_physics_audit.png",
            "bioemu_carbonyl_backbone_audit.png",
            "bioemu_hn_cprime_direction_conflict_audit.png",
        ),
    ),
    (
        "Current Rare-Regime Physics Atlas",
        (
            "bioemu_section_rare_regime_atlas.png",
            "bioemu_rare_regime_conflict_heatmap.png",
            "bioemu_rare_regime_router_confidence.png",
            "bioemu_rare_regime_tangent_alignment.png",
            "bioemu_rare_regime_residue_family_expert_gain.png",
            "bioemu_rare_regime_ensemble_occupancy_shift.png",
            "bioemu_rare_regime_posterior_calibration.png",
        ),
    ),
    (
        "Current Teacher Prior",
        (
            "bioemu_section_teacher_trust_map.png",
            "bioemu_cs_reweighting_teacher_audit.png",
            "bioemu_ucbshift_teacher_sidecar_progress.png",
            "bioemu_ucbshift_export_guard_summary.png",
        ),
    ),
    (
        "Current Posterior Moment",
        (
            "bioemu_section_posterior_geometry_map.png",
            "bioemu_posterior_moment_calibration.png",
            "bioemu_posterior_bridge_vector_field.png",
            "bioemu_sample_weight_entropy.png",
            "bioemu_component_persistence_audit.png",
        ),
    ),
)


def main(argv: list[str] | None = None) -> None:
    """Launch the Streamlit monitor UI."""

    args = parse_args(argv)
    try:
        import streamlit as st
        import streamlit.components.v1 as components
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required for the monitor. Install with "
            "`pip install -e '.[monitor]'` or use the remote helper script."
        ) from exc

    st.set_page_config(
        page_title="AtypEmu Training Monitor",
        page_icon="A",
        layout="wide",
    )
    inject_theme(st)

    json_cache = st.session_state.setdefault("json_cache", {})
    sidebar = build_sidebar(st, args)

    def render_live_dashboard() -> None:
        snapshot = load_monitor_snapshot(
            run_dir=sidebar["run_dir"],
            integrated_root=sidebar["integrated_root"],
            json_cache=json_cache,
            job_id=sidebar["job_id"],
        )
        render_dashboard(st, components, snapshot, sidebar)

    if sidebar["refresh_mode"] == "Silent" and sidebar["refresh_seconds"] > 0:
        if hasattr(st, "fragment"):
            interval = f"{int(sidebar['refresh_seconds'])}s"

            @st.fragment(run_every=interval)
            def render_fragment() -> None:
                render_live_dashboard()

            render_fragment()
        else:
            st.warning(
                "Silent refresh needs a newer Streamlit with fragment support. "
                "Showing a static snapshot instead."
            )
            render_live_dashboard()
    else:
        if (
            sidebar["refresh_mode"] == "Browser reload"
            and sidebar["refresh_seconds"] > 0
        ):
            st.markdown(
                f"<meta http-equiv='refresh' content='{sidebar['refresh_seconds']}'>",
                unsafe_allow_html=True,
            )
        render_live_dashboard()


def render_dashboard(
    st: Any,
    components: Any,
    snapshot: dict[str, Any],
    sidebar: dict[str, Any],
) -> None:
    """Render all dashboard panels for one monitor snapshot."""

    render_overview(st, snapshot, sidebar)
    renderers = {
        "Metrics": lambda: render_metrics_tab(st, snapshot),
        "Analysis PNGs": lambda: render_analysis_pngs_tab(st, snapshot),
        "CCC Geometry": lambda: render_ccc_geometry_tab(st, snapshot),
        "Multi-Observable": lambda: render_multi_observable_tab(st, snapshot),
        "Moment Posterior": lambda: render_moment_posterior_tab(st, snapshot),
        "BioEmu Atlas": lambda: render_bioemu_latent_atlas_tab(st, snapshot),
        "Energy Landscape": lambda: render_energy_landscape_tab(st, snapshot),
        "Conformation Uncertainty": lambda: render_conformation_tab(
            st,
            components,
            snapshot,
        ),
        "NMR Chemical Shifts": lambda: render_nmr_tab(st, snapshot),
        "Raw Reports": lambda: render_raw_reports_tab(st, snapshot),
    }
    selected_labels = _selected_panel_labels(
        sidebar.get("visible_panels"),
        list(renderers.keys()),
    )
    tabs = st.tabs(selected_labels)
    for tab, label in zip(tabs, selected_labels, strict=True):
        with tab:
            renderers[label]()


def _selected_panel_labels(requested: Any, available: list[str]) -> list[str]:
    """Return available dashboard panel labels selected by the sidebar."""

    if not isinstance(requested, (list, tuple, set)):
        return available
    selected = [str(label) for label in requested if str(label) in set(available)]
    return selected or available


def _viewer_panel_status(
    *,
    required_available: int,
    required_total: int,
    optional_available: int,
    applicable: bool = True,
) -> str:
    """Return a compact readiness state for one dashboard panel."""

    if not applicable:
        return "not_applicable"
    if required_total == 0:
        return "ready" if optional_available else "waiting"
    if required_available == required_total:
        return "ready"
    if required_available > 0 or optional_available > 0:
        return "partial"
    return "waiting"


def _viewer_panel_readiness_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return panel-by-panel viewer capabilities and missing-artifact actions."""

    paths = snapshot["paths"]
    is_bioemu = _is_bioemu_latent_snapshot(snapshot)
    is_candidate_free = _is_candidate_free_snapshot(snapshot)
    png_paths = (
        sorted(paths.figures_dir.glob("*.png")) if paths.figures_dir.exists() else []
    )

    def frame(key: str) -> bool:
        return _frame_available(snapshot.get(key))

    def payload(key: str) -> bool:
        value = snapshot.get(key)
        return isinstance(value, dict) and bool(value)

    def path_exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def png_exists(name: str) -> bool:
        return path_exists(paths.figures_dir / name)

    def has_any_prediction() -> bool:
        return (
            frame("bioemu_latent_nmr_predictions")
            or frame("bioemu_latent_nmr_masked_predictions")
            or frame("candidate_free_predictions")
            or frame("candidate_free_masked_predictions")
            or frame("chemical_shift_posteriors")
            or frame("chemical_shift_predictions_calibrated")
        )

    def has_bioemu_live() -> bool:
        return payload("bioemu_live_epoch_progress") or path_exists(
            paths.bioemu_live_epoch_progress
        )

    def add_row(
        rows: list[dict[str, Any]],
        *,
        panel: str,
        scope: str,
        required: list[tuple[str, bool]],
        optional: list[tuple[str, bool]],
        canonical_sources: list[str],
        legacy_sources: list[str],
        next_action: str,
        applicable: bool = True,
    ) -> None:
        required_available = sum(1 for _, ok in required if ok)
        optional_available = sum(1 for _, ok in optional if ok)
        status = _viewer_panel_status(
            required_available=required_available,
            required_total=len(required),
            optional_available=optional_available,
            applicable=applicable,
        )
        missing_required = [name for name, ok in required if not ok]
        missing_optional = [name for name, ok in optional if not ok]
        rows.append(
            {
                "panel": panel,
                "status": status,
                "scope": scope,
                "required_available": required_available,
                "required_total": len(required),
                "optional_available": optional_available,
                "optional_total": len(optional),
                "missing_required": ", ".join(missing_required) or "-",
                "missing_optional": ", ".join(missing_optional[:5]) or "-",
                "canonical_sources": ", ".join(canonical_sources) or "-",
                "legacy_sources": ", ".join(legacy_sources) or "-",
                "next_action": next_action if status != "ready" else "No immediate action.",
            }
        )

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        panel="Metrics",
        scope="all tasks",
        required=[
            (
                "history.json or BioEmu live progress",
                path_exists(paths.history) or has_bioemu_live(),
            )
        ],
        optional=[
            ("summary.json", path_exists(paths.summary) or payload("summary")),
            ("metric_frame rows", _frame_available(snapshot.get("metric_frame"))),
        ],
        canonical_sources=["history.json", "bioemu_live_epoch_progress.json"],
        legacy_sources=["summary.json", "benchmark_report.json"],
        next_action="Wait for the first epoch record or live heartbeat artifact.",
    )
    add_row(
        rows,
        panel="Analysis PNGs",
        scope="all tasks",
        required=[("at least one PNG in reports/figures", bool(png_paths))],
        optional=[
            ("figures directory", path_exists(paths.figures_dir)),
            ("HN/C' target progress PNG", png_exists("bioemu_latent_hn_cprime_target_progress.png")
             or png_exists("candidate_free_hn_cprime_target_progress.png")),
            ("live/history progress PNG", png_exists("bioemu_live_epoch_progress.png")
             or png_exists("bioemu_history_ccc_progress.png")),
        ],
        canonical_sources=["reports/figures/*.png"],
        legacy_sources=["initial checkpoint reference PNG fallback"],
        next_action="Run the PNG updater or wait for best-preview/final analysis artifacts.",
    )
    add_row(
        rows,
        panel="CCC Geometry",
        scope="legacy support geometry plus BioEmu dynamic CCC",
        required=[
            (
                "CCC geometry source",
                _has_ccc_geometry_artifacts(snapshot)
                or frame("bioemu_latent_nmr_masked_predictions")
                or frame("bioemu_latent_nmr_predictions"),
            )
        ],
        optional=[
            ("ccc_support_oracle.parquet", frame("ccc_support_oracle")),
            ("forward residual rows", frame("forward_residual_predictions")),
            ("BioEmu family predictions", frame("bioemu_latent_nmr_masked_predictions")),
        ],
        canonical_sources=["bioemu_*predictions.parquet", "ccc_support_oracle.parquet"],
        legacy_sources=["forward_residual_predictions.parquet"],
        next_action="For BioEmu runs, create prediction parquet; for legacy runs, emit CCC oracle/support tables.",
    )
    add_row(
        rows,
        panel="Multi-Observable",
        scope="multi-observable adapters when emitted by the task",
        required=[
            (
                "observable adapter/support artifact",
                _has_multi_observable_artifacts(snapshot)
                or payload("multi_observable_support_report"),
            )
        ],
        optional=[
            ("observable_adapter_report.json", payload("observable_adapter_report")),
            ("observable_oracle_weights.parquet", frame("observable_oracle_weights")),
        ],
        canonical_sources=["observable_adapter_report.json"],
        legacy_sources=["observable_oracle_weights.parquet"],
        next_action="Only needed for multi-observable runs; current CS-only runs keep this as a scoped legacy panel.",
        applicable=not (is_bioemu or is_candidate_free)
        or _has_multi_observable_artifacts(snapshot),
    )
    add_row(
        rows,
        panel="Moment Posterior",
        scope="BioEmu moment diffusion or legacy posterior moment task",
        required=[
            (
                "posterior moment source",
                frame("bioemu_posterior_moments")
                or frame("bioemu_posterior_bridge_audit")
                or has_bioemu_live()
                or _has_moment_artifacts(snapshot),
            )
        ],
        optional=[
            ("posterior_moments.parquet", frame("bioemu_posterior_moments")),
            ("posterior_bridge_audit.parquet", frame("bioemu_posterior_bridge_audit")),
            ("legacy moment predictions", frame("moment_predictions")),
            ("posterior state tokens", frame("posterior_state_tokens")),
        ],
        canonical_sources=["bioemu_posterior_moments.parquet", "bioemu_posterior_bridge_audit.parquet"],
        legacy_sources=["moment_predictions.parquet", "posterior_state_tokens.parquet"],
        next_action="Wait for moment/bridge artifacts or retain live-progress diagnostics during active epochs.",
    )
    add_row(
        rows,
        panel="BioEmu Atlas",
        scope="BioEmu latent/moment atlas only",
        required=[
            (
                "BioEmu atlas source",
                is_bioemu
                and (
                    frame("bioemu_latent_atlas_coordinates")
                    or frame("bioemu_posterior_moments")
                    or frame("bioemu_posterior_bridge_audit")
                    or has_any_prediction()
                    or has_bioemu_live()
                ),
            )
        ],
        optional=[
            ("atlas_coordinates.parquet", frame("bioemu_latent_atlas_coordinates")),
            ("family conflict audit", frame("bioemu_family_conflict_audit")),
            ("residue-family expert probe", frame("bioemu_residue_family_expert_probe")),
            (
                "mechanism-signed direction audit",
                frame("bioemu_mechanism_signed_direction_audit"),
            ),
            ("component persistence PNG", png_exists("bioemu_component_persistence_audit.png")),
        ],
        canonical_sources=["bioemu_latent_atlas_coordinates.parquet", "bioemu_posterior_moments.parquet"],
        legacy_sources=[],
        next_action="Wait for BioEmu atlas/moment artifacts; for non-BioEmu runs hide this panel.",
        applicable=is_bioemu,
    )
    add_row(
        rows,
        panel="Energy Landscape",
        scope="legacy ensemble landscape plus BioEmu conformer probes",
        required=[
            (
                "landscape or conformer audit source",
                _has_energy_landscape_artifacts(snapshot)
                or frame("bioemu_round_conformer_landscape_audit")
                or frame("bioemu_detached_conformer_ensemble_pca")
                or png_exists("bioemu_round_conformer_landscape_audit.png")
                or png_exists("landscape_teacher_vs_student.png"),
            )
        ],
        optional=[
            ("projection_coordinates.parquet", frame("projection_coordinates")),
            ("round conformer audit", frame("bioemu_round_conformer_landscape_audit")),
            ("detached conformer PCA", frame("bioemu_detached_conformer_ensemble_pca")),
            ("landscape PNG", png_exists("landscape_teacher_vs_student.png")),
        ],
        canonical_sources=["bioemu_round_conformer_landscape_audit.parquet", "projection_coordinates.parquet"],
        legacy_sources=["landscape_teacher_vs_student.png", "ensemble_states.parquet"],
        next_action="Run the detached conformer probe or legacy landscape renderer after a usable checkpoint exists.",
    )
    add_row(
        rows,
        panel="Conformation Uncertainty",
        scope="structure uncertainty and BioEmu detached conformer quality",
        required=[
            (
                "conformation uncertainty source",
                _has_conformation_artifacts(snapshot)
                or frame("bioemu_detached_conformer_ensemble_quality")
                or frame("bioemu_detached_conformer_nmr_coupling")
                or frame("bioemu_detached_conformer_residue_shift_coupling")
                or png_exists("bioemu_detached_conformer_ensemble_quality.png"),
            )
        ],
        optional=[
            ("annotated representative PDB", path_exists(paths.annotated_representative_pdb)),
            ("ensemble states", frame("ensemble_states")),
            ("detached conformer quality", frame("bioemu_detached_conformer_ensemble_quality")),
            ("NMR coupling", frame("bioemu_detached_conformer_nmr_coupling")),
        ],
        canonical_sources=["bioemu_detached_conformer_ensemble_quality.parquet"],
        legacy_sources=["annotated_representative.pdb", "ensemble_states.parquet"],
        next_action="Prepare/update conformer probe outputs to judge whether the ensemble itself is learning.",
    )
    add_row(
        rows,
        panel="NMR Chemical Shifts",
        scope="all atom-family chemical-shift diagnostics",
        required=[
            (
                "chemical-shift prediction source",
                frame("bioemu_latent_nmr_masked_predictions")
                or frame("bioemu_latent_nmr_predictions")
                or frame("candidate_free_masked_predictions")
                or frame("candidate_free_predictions")
                or frame("chemical_shift_posteriors"),
            )
        ],
        optional=[
            ("BioEmu masked predictions", frame("bioemu_latent_nmr_masked_predictions")),
            ("candidate-free masked predictions", frame("candidate_free_masked_predictions")),
            ("legacy CS posteriors", frame("chemical_shift_posteriors")),
            ("family calibration strategy predictions", frame("bioemu_family_calibration_strategy_predictions")),
        ],
        canonical_sources=["bioemu_latent_nmr_masked_predictions.parquet", "candidate_free_masked_predictions.parquet"],
        legacy_sources=["chemical_shift_posteriors.parquet"],
        next_action="Emit masked prediction parquet so family CCC/MAE/bias/std-ratio can be audited.",
    )
    add_row(
        rows,
        panel="Raw Reports",
        scope="all tasks",
        required=[
            (
                "at least one report, history, or loaded table",
                path_exists(paths.history)
                or path_exists(paths.summary)
                or payload("summary")
                or bool(png_paths)
                or has_any_prediction(),
            )
        ],
        optional=[
            ("summary.json", payload("summary") or path_exists(paths.summary)),
            ("BioEmu report", payload("bioemu_latent_nmr_report")),
            ("candidate-free report", payload("candidate_free_report")),
            ("best preview report", payload("best_preview_report")),
        ],
        canonical_sources=["summary.json", "reports/metrics/*.json", "reports/arrays/*.parquet"],
        legacy_sources=["benchmark_report.json", "uncertainty_report.json"],
        next_action="Wait for run artifacts or point the viewer at the correct run directory.",
    )
    return rows


def render_viewer_capability_audit(
    st: Any,
    snapshot: dict[str, Any],
    *,
    expanded: bool = False,
) -> None:
    """Render a panel-level audit of restored and current viewer capabilities."""

    rows = _viewer_panel_readiness_rows(snapshot)
    if not rows:
        return
    table = pd.DataFrame(rows)
    actionable = table.loc[table["status"].ne("not_applicable")]
    usable = actionable.loc[actionable["status"].isin(["ready", "partial"])]
    fraction = len(usable) / max(len(actionable), 1)
    st.progress(
        fraction,
        text=(
            f"Viewer capability readiness: {len(usable)}/{len(actionable)} "
            "applicable panels usable"
        ),
    )
    with st.expander("Viewer capability audit", expanded=expanded):
        st.caption(
            "This table maps each restored viewer panel to its current canonical "
            "BioEmu/candidate-free artifacts, legacy fallbacks, and the next "
            "action when a panel is empty."
        )
        status_order = {"ready": 0, "partial": 1, "waiting": 2, "not_applicable": 3}
        display = table.copy()
        display["_status_order"] = display["status"].map(status_order).fillna(9)
        display = display.sort_values(["_status_order", "panel"], kind="stable").drop(
            columns=["_status_order"]
        )
        st.dataframe(display, width="stretch", hide_index=True)
        next_actions = display.loc[
            display["status"].isin(["partial", "waiting"]),
            ["panel", "status", "missing_required", "next_action"],
        ]
        if not next_actions.empty:
            st.write("Next action buckets")
            st.dataframe(next_actions, width="stretch", hide_index=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Streamlit app arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--integrated-root", default=str(DEFAULT_INTEGRATED_ROOT))
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    return parser.parse_args(argv)


def _sync_sidebar_default(st: Any, key: str, default_value: str) -> None:
    """Update a sidebar widget state when launcher defaults change."""

    marker_key = f"{key}__launcher_default"
    if st.session_state.get(marker_key) != default_value:
        st.session_state[key] = default_value
        st.session_state[marker_key] = default_value


def build_sidebar(st: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Render sidebar inputs and return selected monitor paths."""

    with st.sidebar:
        st.header("Monitor Input")
        _sync_sidebar_default(st, "monitor_run_dir", str(args.run_dir))
        _sync_sidebar_default(
            st,
            "monitor_integrated_root",
            str(args.integrated_root),
        )
        _sync_sidebar_default(
            st,
            "monitor_job_id",
            "" if args.job_id is None else str(args.job_id),
        )
        run_dir = st.text_input("Run directory", key="monitor_run_dir")
        integrated_root = st.text_input(
            "Integrated root",
            key="monitor_integrated_root",
            help="Used for UCBShift and teacher materialization summaries.",
        )
        job_id = st.text_input(
            "Slurm job id",
            key="monitor_job_id",
            help="Optional. The monitor only queries this id; it never edits jobs.",
        )
        refresh_seconds = st.number_input(
            "Refresh seconds",
            min_value=0,
            max_value=600,
            value=int(args.refresh_seconds),
            step=5,
        )
        refresh_mode = st.radio(
            "Refresh mode",
            options=["Silent", "Manual", "Browser reload"],
            index=0,
            help=(
                "Silent uses Streamlit fragment reruns, Manual disables auto "
                "refresh, and Browser reload keeps the old full-page refresh."
            ),
        )
        visible_panels = st.multiselect(
            "Visible panels",
            options=list(DEFAULT_PANEL_LABELS),
            default=list(DEFAULT_PANEL_LABELS),
            key="monitor_visible_panels",
            help=(
                "Select only the dashboard panels you want rendered. This keeps "
                "BioEmu moment-atlas monitoring lightweight during active runs."
            ),
        )
        if refresh_mode == "Manual" and st.button("Refresh now", width="stretch"):
            st.rerun()
        st.caption("SSH tunnel example: " "`ssh -L 8501:127.0.0.1:8501 iremb6-server`")
    args.refresh_seconds = int(refresh_seconds)
    return {
        "run_dir": run_dir,
        "integrated_root": integrated_root or None,
        "job_id": job_id or None,
        "refresh_seconds": int(refresh_seconds),
        "refresh_mode": refresh_mode,
        "visible_panels": visible_panels or list(DEFAULT_PANEL_LABELS),
    }


def render_overview(
    st: Any,
    snapshot: dict[str, Any],
    sidebar: dict[str, Any],
) -> None:
    """Render the run overview cards."""

    paths = snapshot["paths"]
    summary = snapshot["summary"] if isinstance(snapshot["summary"], dict) else {}
    checkpoint = summary.get("best_checkpoint_metric") or {}
    job_status = snapshot.get("job_status") or {}

    render_hero(st, snapshot, sidebar)

    first_row = st.columns([1, 1, 1.3, 1, 1])
    render_metric_card(
        first_row[0],
        "Current epoch",
        _display(snapshot.get("current_epoch")),
        "live history.json",
        tone="blue",
    )
    render_metric_card(
        first_row[1],
        "Best epoch",
        _display(summary.get("best_epoch")),
        "selected checkpoint",
        tone="green",
    )
    render_metric_card(
        first_row[2],
        "Checkpoint metric",
        _display(checkpoint.get("metric_name") or summary.get("best_metric")),
        "teacher-first selection",
        tone="amber",
    )
    render_metric_card(
        first_row[3],
        "Train examples",
        _display(summary.get("train_examples")),
        "materialized",
    )
    render_metric_card(
        first_row[4],
        "Val examples",
        _display(summary.get("val_examples")),
        "materialized",
    )

    second_row = st.columns(4)
    render_metric_card(
        second_row[0],
        "Slurm state",
        _display(job_status.get("state")),
        _display(job_status.get("source")),
        tone=_status_tone(job_status.get("state")),
    )
    render_metric_card(
        second_row[1],
        "Elapsed",
        _display(job_status.get("elapsed")),
        "job wall time",
    )
    render_metric_card(
        second_row[2],
        "Job id",
        _display(job_status.get("job_id") or sidebar["job_id"]),
        _display(job_status.get("job_name")),
    )
    render_metric_card(
        second_row[3],
        "Run directory",
        paths.run_dir.name,
        str(paths.run_dir.parent),
    )

    metric_frame = snapshot.get("metric_frame")
    if metric_frame is not None and not metric_frame.empty:
        if _is_bioemu_latent_snapshot(snapshot):
            render_bioemu_live_metric_cards(st, metric_frame)
        else:
            target = 0.95
            reconstruction = latest_metric_value(
                metric_frame,
                "cs_reconstruction_family_ccc",
                split="val",
            )
            secondary_reconstruction = latest_metric_value(
                metric_frame,
                "cs_reconstruction_secondary_family_ccc",
                split="val",
            )
            masked = latest_metric_value(
                metric_frame,
                "cs_masked_holdout_family_ccc",
                split="val",
            )
            whitened_masked = latest_metric_value(
                metric_frame,
                "cs_masked_holdout_whitened_family_ccc",
                split="val",
            )
            secondary_masked = latest_metric_value(
                metric_frame,
                "cs_masked_holdout_secondary_family_ccc",
                split="val",
            )
            progress_row = st.columns(4)
            render_metric_card(
                progress_row[0],
                "CCC reconstruction / 0.95",
                _display(reconstruction),
                _display(None if reconstruction is None else reconstruction / target),
                tone="green" if reconstruction and reconstruction >= 0.8 else "amber",
            )
            render_metric_card(
                progress_row[1],
                "Secondary CCC / 0.95",
                _display(secondary_reconstruction),
                "random-coil centered",
                tone=(
                    "green"
                    if secondary_reconstruction and secondary_reconstruction >= 0.8
                    else "amber"
                ),
            )
            render_metric_card(
                progress_row[2],
                "Masked holdout CCC",
                _display(masked),
                (
                    f"affine {_display(whitened_masked)}"
                    if whitened_masked is not None
                    else (
                        _display(secondary_masked)
                        if secondary_masked is not None
                        else "target rows hidden from evidence"
                    )
                ),
                tone="blue",
            )
            render_metric_card(
                progress_row[3],
                "CCC target",
                "0.95",
                "NMR posterior reconstruction",
                tone="green",
            )
    elif _is_bioemu_latent_snapshot(snapshot):
        render_bioemu_live_progress_cards(st, snapshot)

    render_top_analysis_pngs(st, snapshot)
    render_artifact_progress(st, snapshot)
    render_viewer_capability_audit(st, snapshot)
    source_vocab = summary.get("source_vocab")
    if source_vocab:
        with st.expander("Source vocabulary", expanded=False):
            st.dataframe(
                _dict_table(source_vocab, "source", "index"),
                width="stretch",
                hide_index=True,
            )

    status_columns = st.columns(3)
    with status_columns[0]:
        render_status_box(st, "UCBShift Sidecar", snapshot.get("sidecar_summary"))
    with status_columns[1]:
        render_status_box(
            st,
            "Teacher Materialization",
            snapshot.get("materialization_summary"),
        )
    with status_columns[2]:
        render_status_box(st, "Training Summary", summary)


def render_metrics_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render metric curves and metric tables."""

    render_panel_header(
        st,
        "TensorBoard-Style Metrics",
        "Live curves from history.json. Best checkpoint selection remains "
        "teacher-first, not total-loss-first.",
    )
    metric_frame = snapshot["metric_frame"]
    if metric_frame.empty:
        st.info("Waiting for history.json metric rows.")
        return

    available = sorted(metric_frame["metric_name"].astype(str).unique().tolist())
    defaults = available_default_metrics(metric_frame)
    selected = st.multiselect(
        "Metrics",
        options=available,
        default=[metric for metric in defaults if metric in available],
    )
    figure = metric_curve_figure(metric_frame, metric_names=selected)
    if figure is None:
        st.warning("Plotly is not available, so only the metric table is shown.")
    else:
        st.plotly_chart(figure, width="stretch")

    render_loss_decomposition_panel(st, metric_frame)

    st.write("Latest metric rows")
    latest_epoch = metric_frame["epoch"].max()
    st.dataframe(
        metric_frame.loc[metric_frame["epoch"] == latest_epoch]
        .sort_values(["split", "metric_name", "aggregation"])
        .reset_index(drop=True),
        width="stretch",
    )

    summary = snapshot.get("summary") or {}
    eligible_counts = summary.get("eligible_counts")
    if eligible_counts:
        st.write("Eligible counts")
        st.json(eligible_counts, expanded=False)

    learnability_report = snapshot.get("learnability_report") or {}
    render_panel_header(
        st,
        "Learnability Audit",
        "Uniform, teacher, CS-fit oracle, and current-model baselines for "
        "checking whether the CS/CCC objective still has room to improve.",
    )
    if learnability_report:
        variant_metrics = learnability_report.get("variant_metrics", {})
        if isinstance(variant_metrics, dict) and variant_metrics:
            split = st.selectbox(
                "Learnability split",
                sorted(variant_metrics.keys()),
            )
            st.dataframe(
                _learnability_table(variant_metrics.get(split, {})),
                width="stretch",
                hide_index=True,
            )
        with st.expander("Gradient summary", expanded=False):
            st.json(learnability_report.get("gradient_summary", {}))
    else:
        st.info("Waiting for learnability_report.json.")


def render_bioemu_live_metric_cards(st: Any, metric_frame: Any) -> None:
    """Render BioEmu-specific live objective cards in the overview."""

    target = 0.95
    hn = latest_metric_value_any(
        metric_frame,
        ["masked_HN_ccc", "masked_HN_ccc_macro", "cs_masked_holdout_HN_ccc"],
        split="val",
    )
    cprime = latest_metric_value_any(
        metric_frame,
        ["masked_C'_ccc", "masked_C'_ccc_macro", "cs_masked_holdout_C'_ccc"],
        split="val",
    )
    hn_cprime = latest_metric_value_any(
        metric_frame,
        ["masked_HN_Cprime_ccc", "masked_HN_Cprime_ccc_macro"],
        split="val",
    )
    balanced = latest_metric_value_any(
        metric_frame,
        [
            "masked_HN_Cprime_family_cprime_balanced_ccc_macro",
            "reconstruction_HN_Cprime_ccc",
        ],
        split="val",
    )
    family = latest_metric_value_any(
        metric_frame,
        ["masked_family_ccc", "masked_family_ccc_macro", "cs_masked_holdout_family_ccc"],
        split="val",
    )
    ess = latest_metric_value(metric_frame, "posterior_ess", split="val")
    bridge = latest_metric_value(metric_frame, "posterior_bridge_norm", split="val")

    progress_row = st.columns(5)
    render_metric_card(
        progress_row[0],
        "HN CCC / 0.95",
        _display(hn),
        _display(None if hn is None else hn / target),
        tone=_ccc_tone(hn),
    )
    render_metric_card(
        progress_row[1],
        "C' CCC / 0.95",
        _display(cprime),
        _display(None if cprime is None else cprime / target),
        tone=_ccc_tone(cprime),
    )
    render_metric_card(
        progress_row[2],
        "HN/C' macro",
        _display(hn_cprime),
        f"balanced {_display(balanced)}",
        tone=_ccc_tone(hn_cprime),
    )
    render_metric_card(
        progress_row[3],
        "Family macro",
        _display(family),
        "HN, N, CA, CB, C'",
        tone=_ccc_tone(family),
    )
    render_metric_card(
        progress_row[4],
        "ESS / bridge",
        _display(ess),
        f"bridge {_display(bridge)}",
        tone="blue" if ess is not None and ess >= 4.0 else "amber",
    )


def render_bioemu_live_progress_cards(st: Any, snapshot: dict[str, Any]) -> None:
    """Render BioEmu heartbeat cards before first epoch artifacts exist."""

    live = snapshot.get("bioemu_live_epoch_progress")
    if not isinstance(live, dict) or not live:
        st.info(
            "BioEmu full run is starting. Waiting for the first live heartbeat or "
            "epoch history artifact."
        )
        return
    losses = live.get("running_loss_summary")
    losses = losses if isinstance(losses, dict) else {}
    processed = live.get("processed_examples")
    total = live.get("total_examples")
    fraction = live.get("fraction")
    if fraction is None and processed is not None and total:
        try:
            fraction = float(processed) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            fraction = None

    row = st.columns(5)
    render_metric_card(
        row[0],
        "Live phase",
        _display(live.get("phase")),
        f"epoch {_display(live.get('epoch'))}",
        tone="blue",
    )
    render_metric_card(
        row[1],
        "Processed",
        f"{_display(processed)} / {_display(total)}",
        _display(fraction),
        tone="blue",
    )
    render_metric_card(
        row[2],
        "Running loss",
        _display(losses.get("loss")),
        f"K={_display(live.get('sample_count'))}",
        tone="amber",
    )
    render_metric_card(
        row[3],
        "ESS / entropy",
        _display(losses.get("posterior_ess")),
        f"entropy {_display(losses.get('posterior_entropy'))}",
        tone="blue",
    )
    render_metric_card(
        row[4],
        "Bridge norm",
        _display(losses.get("posterior_bridge_norm")),
        f"sample disagreement {_display(losses.get('sample_disagreement'))}",
        tone="amber",
    )


def _ccc_tone(value: float | None) -> str:
    if value is None:
        return "muted"
    if value >= 0.9:
        return "green"
    if value >= 0.65:
        return "blue"
    return "amber"


def latest_metric_value_any(
    metric_frame: Any,
    metric_names: list[str],
    *,
    split: str,
    aggregation: str = "macro",
) -> float | None:
    for metric_name in metric_names:
        value = latest_metric_value(
            metric_frame,
            metric_name,
            split=split,
            aggregation=aggregation,
        )
        if value is not None:
            return value
    return None


def render_loss_decomposition_panel(st: Any, metric_frame: Any) -> None:
    """Render compact, interactive loss component curves."""

    bioemu_component_names = [
        "loss",
        "nll",
        "hn95_nll",
        "ccc_loss",
        "aux_family_ccc_loss",
        "family_ccc_target_loss",
        "residue_atom_posterior_mean_loss",
        "ucbshift_cnnls_teacher_mean_loss",
        "cprime_guardrail",
        "hn_ccc_loss",
        "hn_scale_loss",
        "hn_bias_loss",
        "cprime_scale_loss",
        "cprime_bias_loss",
        "family_scale_loss",
        "family_bias_loss",
        "bridge_norm_loss",
        "tail_gap_loss",
    ]
    available = set(metric_frame["metric_name"].astype(str).unique().tolist())
    if any(name in available for name in bioemu_component_names):
        render_panel_header(
            st,
            "BioEmu Loss / Guardrail Decomposition",
            "Loss stays in the interactive Streamlit view so we can inspect "
            "objective alignment without regenerating static PNGs. The PNG "
            "gallery remains an archival snapshot layer.",
        )
        cards = st.columns(6)
        card_specs = [
            ("Val loss", "loss", "overall objective", "blue"),
            ("Family CCC target", "family_ccc_target_loss", "0.95 pressure", "green"),
            (
                "Teacher mean",
                "ucbshift_cnnls_teacher_mean_loss",
                "CS-reweighting teacher",
                "amber",
            ),
            ("C' guardrail", "cprime_guardrail", "late-collapse check", "red"),
            ("Bridge norm", "bridge_norm_loss", "posterior bridge", "amber"),
            ("HN CCC loss", "hn_ccc_loss", "HN target pressure", "muted"),
        ]
        for column, (label, metric_name, caption, tone) in zip(
            cards,
            card_specs,
            strict=False,
        ):
            render_metric_card(
                column,
                label,
                _format_metric_value(
                    latest_metric_value(metric_frame, metric_name, split="val")
                ),
                caption,
                tone=tone,
            )

        selected_components = [
            name for name in bioemu_component_names if name in available
        ]
        figure = metric_curve_figure(metric_frame, metric_names=selected_components)
        if figure is not None:
            st.plotly_chart(figure, width="stretch", key="bioemu_loss_decomposition_curve")
        st.caption(
            "Interpretation hint: if family/HN CCC target losses flatten while "
            "C' guardrail rises, stop increasing family target pressure and "
            "stabilize C' scale/bias. If bridge_norm_loss falls while CCC falls, "
            "the posterior bridge is over-regularized."
        )
        return

    component_names = [
        "total_loss",
        "masked_HN_Cprime_ccc_loss",
        "smooth_worst_HN_Cprime_ccc_loss",
        "family_scale_loss",
        "family_bias_loss",
        "tail_calibration_loss",
        "student_t_nll",
    ]
    available = set(metric_frame["metric_name"].astype(str).unique().tolist())
    if not any(name in available for name in component_names):
        return

    render_panel_header(
        st,
        "Loss Decomposition",
        "Candidate-free HN/C' runs can keep lowering total loss even when CCC "
        "plateaus. This panel separates CCC, scale/bias, tail, and NLL terms.",
    )
    cards = st.columns(5)
    card_specs = [
        ("Val total", "total_loss", "overall objective", "blue"),
        ("HN/C' CCC loss", "masked_HN_Cprime_ccc_loss", "primary shape term", "green"),
        ("Scale loss", "family_scale_loss", "variance matching", "amber"),
        ("Tail loss", "tail_calibration_loss", "extreme-shift rows", "amber"),
        ("Student-t NLL", "student_t_nll", "calibration likelihood", "muted"),
    ]
    for column, (label, metric_name, caption, tone) in zip(
        cards,
        card_specs,
        strict=False,
    ):
        render_metric_card(
            column,
            label,
            _format_metric_value(
                latest_metric_value(metric_frame, metric_name, split="val")
            ),
            caption,
            tone=tone,
        )

    selected_components = [
        name for name in component_names if name in available
    ]
    figure = metric_curve_figure(metric_frame, metric_names=selected_components)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", key="loss_decomposition_curve")
    st.caption(
        "Interpretation hint: if total_loss falls while HN/C' CCC loss stalls, "
        "the next run should reduce NLL weight or increase tail/scale pressure. "
        "If scale/tail losses fall with CCC gains, the new objective is aligned."
    )


def render_top_analysis_pngs(st: Any, snapshot: dict[str, Any]) -> None:
    """Render the primary analysis PNGs near the top of the dashboard."""

    paths = snapshot["paths"]
    if _is_bioemu_latent_snapshot(snapshot):
        section_overview_paths = [
            paths.figures_dir / "bioemu_section_training_overview.png",
            paths.figures_dir / "bioemu_training_artifact_consistency_audit.png",
            paths.figures_dir / "bioemu_section_family_scorecard.png",
            paths.figures_dir / "bioemu_section_teacher_trust_map.png",
        ]
        v101_primary_paths = [
            paths.figures_dir / "bioemu_family_ccc95_gap.png",
            paths.figures_dir / "bioemu_mechanism_signed_direction_audit.png",
            paths.figures_dir / "bioemu_family_shrinkage_bias.png",
        ]
        ccc95_paths = [
            paths.figures_dir / "bioemu_family_ccc95_gap.png",
            paths.figures_dir / "bioemu_family_shrinkage_bias.png",
            paths.figures_dir / "bioemu_worst_entry_family_ccc95_gap.png",
        ]
        fallback_paths = [
            paths.figures_dir / "bioemu_latent_hn_masked_scatter.png",
            paths.figures_dir / "bioemu_latent_cprime_masked_scatter.png",
            paths.figures_dir / "bioemu_latent_all_family_progress.png",
        ]
        live_path = paths.figures_dir / "bioemu_live_epoch_progress.png"
        if any(path.exists() for path in section_overview_paths):
            primary_paths = [path for path in section_overview_paths if path.exists()]
            caption_text = (
                "Section-level BioEmu overview figures: training trajectory and "
                "late drift, training-vs-artifact consistency, family-level "
                "CCC/tail scorecard, and CS-reweighting teacher trust map. "
                "Detailed mechanistic PNGs are grouped below."
            )
        elif all(path.exists() for path in v101_primary_paths):
            primary_paths = v101_primary_paths
            caption_text = (
                "Latest BioEmu analysis only: atom-family CCC gap, v101 "
                "mechanism-signed rare-regime priority, and posterior-mean "
                "shrinkage/bias. Legacy exploratory PNGs are intentionally "
                "kept out of the primary viewer."
            )
        elif any(path.exists() for path in ccc95_paths):
            primary_paths = ccc95_paths
            caption_text = (
                "Primary BioEmu latent atlas figures for the current atom-family "
                "CCC 0.95 target. These make family gaps, posterior-mean shrinkage, "
                "bias, and worst entry-family bottlenecks visible before drilling "
                "into individual HN/C' scatter panels."
            )
        elif live_path.exists():
            primary_paths = [live_path]
            caption_text = (
                "Live BioEmu latent atlas heartbeat. This appears before the first "
                "epoch history and best-checkpoint prediction artifacts are written."
            )
        else:
            primary_paths = fallback_paths
            caption_text = (
                "Primary BioEmu latent atlas figures rendered from the latest best "
                "checkpoint artifacts. HN/C' stay prominent while all-family "
                "progress exposes N, CA, and CB behavior."
            )
    else:
        primary_paths = [
            _current_or_reference_figure(
                snapshot, paths.figures_dir / "candidate_free_hn_masked_scatter.png"
            ),
            _current_or_reference_figure(
                snapshot, paths.figures_dir / "candidate_free_cprime_masked_scatter.png"
            ),
            _current_or_reference_figure(
                snapshot, paths.figures_dir / "candidate_free_hn_cprime_target_progress.png"
            ),
        ]
        caption_text = (
            "Primary best-checkpoint candidate-free figures. These are rendered "
            "with Streamlit's native image component for reliability. If the "
            "current run has not produced a first preview yet, the viewer shows "
            "the initial checkpoint reference figures."
        )
    if not any(path.exists() for path in primary_paths):
        return
    with st.expander("Latest Analysis PNGs", expanded=True):
        st.caption(caption_text)
        columns = st.columns(3)
        for column, path in zip(columns, primary_paths, strict=False):
            caption = _pretty_figure_caption(path.name)
            if _is_reference_figure(snapshot, path):
                caption = f"Reference initial checkpoint: {caption}"
            render_native_image_or_wait(
                column,
                path,
                caption,
                show_path=True,
            )


def render_analysis_pngs_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render a dedicated gallery for static analysis PNG artifacts."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "Analysis PNG Gallery",
        "Static analysis figures from the current run. This tab surfaces PNG "
        "artifacts directly so best-checkpoint analysis is not hidden below "
        "waiting panels in task-specific tabs.",
    )
    png_paths = sorted(paths.figures_dir.glob("*.png")) if paths.figures_dir.exists() else []
    reference_dir = _initial_checkpoint_figures_dir(snapshot)
    reference_png_paths = (
        sorted(reference_dir.glob("*.png"))
        if reference_dir is not None and reference_dir.exists()
        else []
    )
    showing_reference = False
    if not png_paths and reference_png_paths:
        st.info(
            "Current run has not produced PNG artifacts yet. Showing initial "
            "checkpoint reference PNGs until the next best-preview render arrives."
        )
        png_paths = reference_png_paths
        showing_reference = True
    elif not paths.figures_dir.exists():
        st.info(f"Waiting for figures directory: {paths.figures_dir}")
        return
    elif not png_paths:
        st.info(f"No PNG artifacts found in {paths.figures_dir}")
        return

    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    checkpoint = summary.get("best_checkpoint_metric") or {}
    best_values = checkpoint.get("values") or {}
    card_cols = st.columns(5)
    with card_cols[0]:
        st.metric("Best epoch", _display(summary.get("best_epoch")))
    with card_cols[1]:
        st.metric(
            "HN/C' macro",
            _format_metric_value(best_values.get("val_masked_HN_Cprime_ccc_macro")),
        )
    with card_cols[2]:
        st.metric(
            "HN macro",
            _format_metric_value(best_values.get("val_masked_HN_ccc_macro")),
        )
    with card_cols[3]:
        st.metric(
            "C' macro",
            _format_metric_value(best_values.get("val_masked_C'_ccc_macro")),
        )
    with card_cols[4]:
        st.metric(
            "Family macro",
            _format_metric_value(best_values.get("val_masked_family_ccc_macro")),
        )

    if showing_reference and reference_dir is not None:
        st.caption(f"Reference figures directory: `{reference_dir}`")
    else:
        st.caption(f"Figures directory: `{paths.figures_dir}`")
    if _is_bioemu_latent_snapshot(snapshot):
        groups = BIOEMU_LATEST_ANALYSIS_PNG_GROUPS
        shown_names: set[str] = set()
        base_dir = Path(png_paths[0]).parent if showing_reference else paths.figures_dir
        for group_title, names in groups:
            group_paths = [base_dir / name for name in names if (base_dir / name).exists()]
            if not group_paths:
                continue
            shown_names.update(path.name for path in group_paths)
            st.subheader(group_title)
            gallery_cols = st.columns(min(3, max(1, len(group_paths))))
            for index, path in enumerate(group_paths):
                render_native_image_or_wait(
                    gallery_cols[index % len(gallery_cols)],
                    path,
                    _pretty_figure_caption(path.name),
                    show_path=True,
                )
        if not shown_names:
            latest_paths = _latest_png_paths(
                [path for path in png_paths if path.name.startswith("bioemu_")],
                limit=12,
            )
            if latest_paths:
                st.subheader("Newest BioEmu PNGs")
                gallery_cols = st.columns(3)
                for index, path in enumerate(latest_paths):
                    shown_names.add(path.name)
                    render_native_image_or_wait(
                        gallery_cols[index % len(gallery_cols)],
                        path,
                        _pretty_figure_caption(path.name),
                        show_path=True,
                    )
        hidden_count = len([path for path in png_paths if path.name not in shown_names])
        if hidden_count:
            st.caption(
                f"Hidden archived/legacy PNGs: {hidden_count}. "
                "They remain on disk for reproducibility but are no longer "
                "rendered in the BioEmu analysis viewer."
            )
        remaining = []
    else:
        primary_names = [
            "candidate_free_hn_masked_scatter.png",
            "candidate_free_cprime_masked_scatter.png",
            "candidate_free_hn_cprime_target_progress.png",
        ]
        primary_title = "Primary Candidate-Free HN/C' Analysis"
        primary_paths = [
            (
                Path(png_paths[0]).parent / name
                if showing_reference
                else paths.figures_dir / name
            )
            for name in primary_names
        ]
        if any(path.exists() for path in primary_paths):
            st.subheader(primary_title)
            primary_cols = st.columns(3)
            for column, path in zip(primary_cols, primary_paths, strict=False):
                render_native_image_or_wait(
                    column,
                    path,
                    _pretty_figure_caption(path.name),
                    show_path=True,
                )
        remaining = [path for path in png_paths if path.name not in set(primary_names)]
    if remaining:
        st.subheader("Raw Generated PNGs" if _is_bioemu_latent_snapshot(snapshot) else "All Other Analysis PNGs")
        gallery_cols = st.columns(2)
        for index, path in enumerate(remaining):
            render_native_image_or_wait(
                gallery_cols[index % len(gallery_cols)],
                path,
                _pretty_figure_caption(path.name),
                show_path=True,
            )

    with st.expander("PNG artifact inventory", expanded=False):
        inventory_paths = (
            [path for path in png_paths if path.name in shown_names]
            if _is_bioemu_latent_snapshot(snapshot)
            else png_paths
        )
        inventory = [
            {
                "file": path.name,
                "size_kb": round(path.stat().st_size / 1024.0, 1),
                "path": str(path),
            }
            for path in inventory_paths
        ]
        st.dataframe(inventory, width="stretch", hide_index=True)


def _latest_png_paths(paths: list[Path], *, limit: int) -> list[Path]:
    """Return the newest non-empty PNG paths without surfacing old galleries."""

    existing = [path for path in paths if path.exists() and path.stat().st_size > 0]
    return sorted(
        existing,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(int(limit), 0)]


def render_panel_analysis_png(
    st: Any,
    snapshot: dict[str, Any],
    filename: str,
    caption: str,
) -> None:
    """Render a compact panel-level BioEmu analysis PNG when present."""

    path = snapshot["paths"].figures_dir / filename
    if path.exists() and path.stat().st_size > 0:
        render_image_or_wait(st, path, caption)


def render_energy_landscape_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render BioEmu-style ensemble landscape panels."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "BioEmu-Style Energy Landscape",
        "Teacher and student ensembles are compared on the same weighted-PCA "
        "basis and rendered as free-energy surfaces.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_energy_landscape.png",
        "Panel-level energy landscape proxy and readiness analysis",
    )
    round_audit = snapshot.get("bioemu_round_conformer_landscape_audit")
    round_audit_path = paths.figures_dir / "bioemu_round_conformer_landscape_audit.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(round_audit) or round_audit_path.exists()
    ):
        render_image_or_wait(
            st,
            round_audit_path,
            "Round conformer ensemble and energy-landscape readiness",
        )
        if _frame_available(round_audit) and "check_name" in round_audit.columns:
            verdict = round_audit.loc[
                round_audit["check_name"].astype(str).eq(
                    "actual_conformer_learning_verdict"
                )
            ]
            if not verdict.empty and str(verdict.iloc[0].get("status", "")) != "pass":
                st.warning(str(verdict.iloc[0].get("interpretation", "")))
    probe_manifest = snapshot.get("bioemu_detached_conformer_probe_manifest")
    probe_manifest_path = paths.figures_dir / "bioemu_detached_conformer_probe_manifest.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_manifest) or probe_manifest_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_manifest_path,
            "Next-round detached conformer probe queue",
        )
        if _frame_available(probe_manifest):
            st.dataframe(probe_manifest.head(12), width="stretch", hide_index=True)
    probe_inputs = snapshot.get("bioemu_detached_conformer_probe_inputs")
    probe_inputs_path = paths.figures_dir / "bioemu_detached_conformer_probe_inputs.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_inputs) or probe_inputs_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_inputs_path,
            "Prepared detached conformer probe FASTA inputs",
        )
        if _frame_available(probe_inputs):
            display_columns = [
                column
                for column in (
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "sequence_length",
                    "sample_count",
                    "ready",
                    "fasta_path",
                    "output_dir",
                )
                if column in probe_inputs.columns
            ]
            st.dataframe(
                probe_inputs[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    probe_output_audit = snapshot.get("bioemu_detached_conformer_probe_output_audit")
    probe_output_path = (
        paths.figures_dir / "bioemu_detached_conformer_probe_output_audit.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_output_audit) or probe_output_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_output_path,
            "Detached BioEmu conformer probe generated outputs",
        )
        if _frame_available(probe_output_audit):
            display_columns = [
                column
                for column in (
                    "probe_name",
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "requested_samples",
                    "actual_samples",
                    "residue_count",
                    "status",
                    "context_mode",
                    "topology_path",
                    "xtc_path",
                )
                if column in probe_output_audit.columns
            ]
            st.dataframe(
                probe_output_audit[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    ensemble_quality = snapshot.get("bioemu_detached_conformer_ensemble_quality")
    ensemble_quality_path = (
        paths.figures_dir / "bioemu_detached_conformer_ensemble_quality.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(ensemble_quality) or ensemble_quality_path.exists()
    ):
        render_image_or_wait(
            st,
            ensemble_quality_path,
            "Detached BioEmu conformer diversity and PCA landscape",
        )
        if _frame_available(ensemble_quality):
            display_columns = [
                column
                for column in (
                    "probe_name",
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "actual_samples",
                    "residue_count",
                    "quality_flag",
                    "quality_outlier_score",
                    "family_quality_outlier_count",
                    "family_quality_outlier_fraction",
                    "family_repeated_quality_outlier",
                    "mean_pairwise_rmsd",
                    "max_pairwise_rmsd",
                    "rg_mean",
                    "rg_std",
                    "contact_density_mean",
                    "pca_spread",
                )
                if column in ensemble_quality.columns
            ]
            st.dataframe(
                ensemble_quality[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    family_quality_guard = snapshot.get(
        "bioemu_detached_conformer_family_quality_guard"
    )
    family_quality_guard_path = (
        paths.figures_dir / "bioemu_detached_conformer_family_quality_guard.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(family_quality_guard) or family_quality_guard_path.exists()
    ):
        render_image_or_wait(
            st,
            family_quality_guard_path,
            "Family-level detached conformer geometry-spread guard",
        )
        if _frame_available(family_quality_guard):
            display_columns = [
                column
                for column in (
                    "top_atom_family",
                    "ready_entities",
                    "quality_outlier_entities",
                    "quality_outlier_fraction",
                    "max_quality_outlier_score",
                    "outlier_entities",
                    "guard_priority_score",
                    "recommended_guard_action",
                )
                if column in family_quality_guard.columns
            ]
            st.dataframe(
                family_quality_guard[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    support_action = snapshot.get("bioemu_detached_support_expansion_action_plan")
    support_action_path = (
        paths.figures_dir / "bioemu_detached_support_expansion_action_plan.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(support_action) or support_action_path.exists()
    ):
        render_image_or_wait(
            st,
            support_action_path,
            "Detached probe support-expansion action plan",
        )
        if _frame_available(support_action):
            display_columns = [
                column
                for column in (
                    "entity_uid",
                    "top_atom_family",
                    "quality_flag",
                    "support_guard_level",
                    "combined_priority_score",
                    "priority_score",
                    "low_ess_tail_rows",
                    "rg_std",
                    "pca_spread",
                    "recommended_next_action",
                )
                if column in support_action.columns
            ]
            st.dataframe(
                support_action[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    nmr_coupling = snapshot.get("bioemu_detached_conformer_nmr_coupling")
    nmr_coupling_path = (
        paths.figures_dir / "bioemu_detached_conformer_nmr_coupling.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(nmr_coupling) or nmr_coupling_path.exists()
    ):
        render_image_or_wait(
            st,
            nmr_coupling_path,
            "NMR bottleneck and conformer-ensemble coupling",
        )
        if _frame_available(nmr_coupling):
            display_columns = [
                column
                for column in (
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "priority_score",
                    "hn_cprime_mae",
                    "all_family_mae",
                    "mean_pairwise_rmsd",
                    "rg_std",
                    "contact_density_mean",
                    "nmr_structure_coupling_score",
                    "coupling_interpretation",
                )
                if column in nmr_coupling.columns
            ]
            st.dataframe(
                nmr_coupling[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    residue_coupling = snapshot.get(
        "bioemu_detached_conformer_residue_shift_coupling"
    )
    residue_coupling_path = (
        paths.figures_dir / "bioemu_detached_conformer_residue_shift_coupling.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(residue_coupling) or residue_coupling_path.exists()
    ):
        render_image_or_wait(
            st,
            residue_coupling_path,
            "Residue-level NMR residual and conformer-mobility coupling",
        )
        if _frame_available(residue_coupling):
            display_columns = [
                column
                for column in (
                    "probe_rank",
                    "entity_uid",
                    "residue_index",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "predicted_value",
                    "guarded_predicted_value",
                    "active_abs_error",
                    "posterior_sigma",
                    "residue_rmsf",
                    "local_contact_variability",
                    "residue_shift_mobility_score",
                    "selected_strategy",
                    "guard_action",
                )
                if column in residue_coupling.columns
            ]
            st.dataframe(
                residue_coupling[display_columns].head(20),
                width="stretch",
                hide_index=True,
            )
    if _is_bioemu_latent_snapshot(snapshot) and not _has_energy_landscape_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="Actual conformer energy landscape is not ready for this round.",
            text=(
                "This BioEmu latent round has NMR/atlas diagnostics, but no "
                "decoded conformer landscape artifacts. Do not promote the round "
                "on chemical-shift metrics alone until projection coordinates and "
                "free-energy surfaces are emitted."
            ),
            expected=[
                "reports/arrays/projection_coordinates.parquet",
                "reports/figures/landscape_teacher_vs_student.png",
                "reports/figures/landscape_delta_free_energy.png",
            ],
        )
        return
    if _is_candidate_free_snapshot(snapshot) and not _has_energy_landscape_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="Energy landscape is not emitted by this candidate-free run.",
            text=(
                "The current run uses the candidate-free posterior path. It "
                "does not generate BioEmu-style projection grids or landscape "
                "PNGs, so this tab is marked as not applicable rather than "
                "waiting."
                " Candidate-free measure diagnostics are consolidated in the "
                "NMR Chemical Shifts tab."
            ),
            expected=[
                "reports/arrays/projection_coordinates.parquet",
                "reports/figures/landscape_teacher_vs_student.png",
            ],
        )
        return
    metrics = _nested_metrics(
        snapshot.get("benchmark_report"),
        ["ensemble_fidelity", "metrics"],
        fallback_key="ensemble_fidelity",
    )
    render_metric_dict(st, metrics)

    cols = st.columns(2)
    render_image_or_wait(
        cols[0],
        paths.figures_dir / "landscape_teacher_vs_student.png",
        "Teacher vs student free-energy surface",
    )
    render_image_or_wait(
        cols[1],
        paths.figures_dir / "landscape_delta_free_energy.png",
        "Student minus teacher free energy",
    )

    overlay_cols = st.columns(3)
    render_image_or_wait(
        overlay_cols[0],
        paths.figures_dir / "landscape_teacher_score.png",
        "Teacher score overlay",
    )
    render_image_or_wait(
        overlay_cols[1],
        paths.figures_dir / "landscape_cs_error.png",
        "Chemical-shift error overlay",
    )
    render_image_or_wait(
        overlay_cols[2],
        paths.figures_dir / "landscape_rci_mismatch.png",
        "RCI mismatch overlay",
    )

    projection_frame = snapshot["projection_coordinates"]
    color_options = available_landscape_color_columns(projection_frame)
    color_by = (
        st.selectbox(
            "Landscape color",
            options=color_options,
            index=(
                color_options.index("student_minus_teacher")
                if "student_minus_teacher" in color_options
                else 0
            ),
            help="NMR evidence overlays appear when final render artifacts are present.",
        )
        if color_options
        else "student_minus_teacher"
    )
    figure = projection_scatter_figure(projection_frame, color_by=color_by)
    if figure is None:
        st.info("Waiting for projection_coordinates.parquet or Plotly.")
    else:
        st.plotly_chart(figure, width="stretch")


def render_ccc_geometry_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render CCC support geometry and oracle/model gap panels."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "CCC Geometry",
        "Support-ceiling diagnostics separate model gap from candidate/support "
        "limitations. Projected-simplex oracle rows are offline diagnostics, "
        "not training-set claims.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_ccc_geometry.png",
        "Panel-level CCC geometry analysis",
    )
    if _is_candidate_free_snapshot(snapshot) and not _has_ccc_geometry_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="CCC geometry artifacts are not produced for this run.",
            text=(
                "Candidate-free v1 removes the 512-candidate simplex from the "
                "primary path. The CCC geometry tab expects candidate/support "
                "oracle files, so it is intentionally inactive for this run."
                " Candidate-free CCC decomposition is consolidated in the "
                "NMR Chemical Shifts tab."
            ),
            expected=[
                "reports/arrays/ccc_support_oracle.parquet",
                "reports/metrics/support_ceiling_report.json",
                "benchmark_report.json",
            ],
        )
        return
    support = (
        snapshot.get("benchmark_report", {}).get("support_ceiling")
        if isinstance(snapshot.get("benchmark_report"), dict)
        else None
    )
    if isinstance(support, dict):
        ccc_geometry = support.get("ccc_geometry", {})
        if isinstance(ccc_geometry, dict):
            render_metric_dict(st, ccc_geometry)

    metric_frame = snapshot.get("metric_frame")
    if metric_frame is not None and not metric_frame.empty:
        cards = st.columns(4)
        render_metric_card(
            cards[0],
            "Raw masked CCC",
            _display(
                latest_metric_value(
                    metric_frame,
                    "cs_masked_holdout_family_ccc",
                    split="val",
                )
            ),
            "primary v2 objective",
            tone="green",
        )
        render_metric_card(
            cards[1],
            "Affine masked CCC",
            _display(
                latest_metric_value(
                    metric_frame,
                    "cs_masked_holdout_whitened_family_ccc",
                    split="val",
                )
            ),
            "evidence-fit calibration",
            tone="blue",
        )
        render_metric_card(
            cards[2],
            "Raw reconstruction CCC",
            _display(
                latest_metric_value(
                    metric_frame,
                    "cs_reconstruction_family_ccc",
                    split="val",
                )
            ),
            "all observed rows",
            tone="amber",
        )
        render_metric_card(
            cards[3],
            "Teacher KL",
            _display(
                latest_metric_value(metric_frame, "teacher_kl_macro", split="val")
            ),
            "guardrail",
        )

    static_cols = st.columns(2)
    render_image_or_wait(
        static_cols[0],
        paths.figures_dir / "oracle_vs_model_ccc.png",
        "Oracle vs model CCC",
    )
    render_image_or_wait(
        static_cols[1],
        paths.figures_dir / "ccc_support_gap_by_family.png",
        "CCC support gap by atom family",
    )

    oracle_frame = snapshot.get("ccc_support_oracle")
    if oracle_frame is None or oracle_frame.empty:
        st.info("Waiting for ccc_support_oracle.parquet.")
    else:
        figures = [
            ccc_oracle_vs_model_figure(oracle_frame),
            ccc_family_gap_figure(oracle_frame),
        ]
        for figure in figures:
            if figure is not None:
                st.plotly_chart(figure, width="stretch")
        with st.expander("CCC support oracle rows", expanded=False):
            st.dataframe(
                oracle_frame.head(400),
                width="stretch",
                hide_index=True,
            )

    residual_frame = snapshot.get("residue_atom_forward_residuals")
    has_residue_residual = residual_frame is not None and not residual_frame.empty
    if has_residue_residual:
        st.write("Residue-atom forward residual predictions")
        st.dataframe(
            residual_frame.head(400),
            width="stretch",
            hide_index=True,
        )
        render_image_or_wait(
            st,
            paths.figures_dir / "residue_atom_residual_heatmap.png",
            "Residue-atom residual heatmap",
        )

    secondary_frame = snapshot.get("secondary_shift_posteriors")
    if secondary_frame is not None and not secondary_frame.empty:
        st.write("Secondary-shift posterior rows")
        st.dataframe(
            secondary_frame.head(400),
            width="stretch",
            hide_index=True,
        )

    if not has_residue_residual:
        residual_frame = snapshot.get("forward_residual_predictions")
        if residual_frame is None or residual_frame.empty:
            st.info("Waiting for forward residual artifacts.")
        else:
            st.write("Forward residual head predictions")
            st.dataframe(
                residual_frame.head(400),
                width="stretch",
                hide_index=True,
            )


def render_multi_observable_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render multi-observable approximation adapter diagnostics."""

    render_panel_header(
        st,
        "Multi-Observable NMR Approximation",
        "CS remains the primary dense objective. J-coupling and NOE contribute "
        "as low-weight auxiliary evidence; SAXS stays report-only until "
        "candidate profile matrices exist.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_multi_observable.png",
        "Panel-level multi-observable evidence stack",
    )
    if _is_candidate_free_snapshot(snapshot) and not _has_multi_observable_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="Multi-observable adapters are not emitted by this CS-only run.",
            text=(
                "The current candidate-free campaign is chemical-shift focused. "
                "J/NOE/SAXS adapter reports were not requested, so this tab is "
                "not applicable for the current run."
                " Candidate-free CS coverage diagnostics are consolidated in "
                "the NMR Chemical Shifts tab."
            ),
            expected=[
                "reports/metrics/observable_adapter_report.json",
                "reports/arrays/observable_oracle_weights.parquet",
            ],
        )
        return
    report = snapshot.get("observable_adapter_report") or {}
    support_report = snapshot.get("multi_observable_support_report") or {}
    if not report:
        st.info("Waiting for observable_adapter_report.json.")
    else:
        table = observable_support_table(report)
        if table.empty:
            st.info("Observable adapter report is present but has no channel rows yet.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)
        figure = observable_support_figure(report)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")

    summary_frame = snapshot.get("observable_adapter_summary")
    conflict = (
        None if summary_frame is None else observable_conflict_figure(summary_frame)
    )
    if conflict is not None:
        st.plotly_chart(conflict, width="stretch")

    weights_frame = snapshot.get("observable_oracle_weights")
    if weights_frame is None or weights_frame.empty:
        st.info("Waiting for observable_oracle_weights.parquet.")
    else:
        with st.expander("Observable oracle weights", expanded=False):
            st.dataframe(
                weights_frame.head(600),
                width="stretch",
                hide_index=True,
            )
    if support_report:
        with st.expander("Multi-observable support ceiling report", expanded=False):
            st.json(support_report)


def render_moment_posterior_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render MomentHead and BioEmu-style posterior state diagnostics."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "Moment-Consistent Posterior",
        "Fast MomentHead predictions drive chemical-shift CCC, while posterior "
        "state tokens and latent samples provide BioEmu-style ensemble "
        "diagnostics.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_moment_posterior.png",
        "Panel-level posterior moment geometry",
    )
    if _is_bioemu_latent_snapshot(snapshot):
        render_bioemu_moment_posterior_panels(st, snapshot)
        return
    if _is_candidate_free_snapshot(snapshot) and not _has_moment_artifacts(snapshot):
        render_task_scope_notice(
            st,
            snapshot,
            title="Moment-diffusion artifacts are not produced for this run.",
            text=(
                "This tab now shows the candidate-free posterior analysis above. "
                "The NNLS oracle, MomentHead sample consistency, and posterior "
                "latent-flow files belong to the moment-diffusion tasks, not to "
                "posterior_nmr_candidate_free_v1."
                " Candidate-free posterior mean diagnostics are consolidated "
                "in the NMR Chemical Shifts tab."
            ),
            expected=[
                "reports/arrays/moment_predictions.parquet",
                "reports/metrics/moment_oracle_report.json",
                "reports/arrays/posterior_state_tokens.parquet",
            ],
        )
        return
    preview_report = snapshot.get("best_preview_report") or {}
    if preview_report:
        preview_cols = st.columns(3)
        preview_metric = preview_report.get("checkpoint_metric") or {}
        preview_scope = preview_report.get("preview_scope") or {}
        with preview_cols[0]:
            st.metric("Best-preview epoch", preview_report.get("epoch", "n/a"))
        with preview_cols[1]:
            st.metric(
                "Preview train/val",
                f"{preview_scope.get('train_examples', 0)}/"
                f"{preview_scope.get('val_examples', 0)}",
            )
        with preview_cols[2]:
            st.metric(
                "Best metric",
                preview_metric.get("primary", preview_metric.get("metric", "n/a")),
            )
        with st.expander("Current best-preview analysis", expanded=False):
            st.json(preview_report, expanded=False)
    else:
        st.info("Waiting for best_preview_report.json from the next best checkpoint.")

    oracle_report = snapshot.get("moment_oracle_report") or {}
    consistency_report = snapshot.get("moment_consistency_report") or {}
    metric_cols = st.columns(2)
    with metric_cols[0]:
        table = moment_oracle_table(oracle_report)
        if table.empty:
            st.info("Waiting for moment_oracle_report.json.")
        else:
            st.write("NNLS/QP Oracle vs MomentHead")
            st.dataframe(table, width="stretch", hide_index=True)
    with metric_cols[1]:
        if consistency_report:
            st.write("Candidate-sample support diagnostic")
            st.json(consistency_report, expanded=False)
        else:
            st.info("Waiting for moment_consistency_report.json.")

    moment_frame = snapshot.get("moment_predictions")
    state_frame = snapshot.get("posterior_state_tokens")
    render_candidate_free_panels(st, snapshot)
    render_moment_v2_cards(st, moment_frame, state_frame)
    render_moment_v3_panels(st, snapshot)

    moment_main_cols = st.columns(2)
    render_image_or_wait(
        moment_main_cols[0],
        paths.figures_dir / "moment_vs_target.png",
        "MomentHead vs experimental shift",
    )
    render_image_or_wait(
        moment_main_cols[1],
        paths.figures_dir / "sample_vs_target_support.png",
        "Candidate-sample support vs experimental shift",
    )
    figure_cols = st.columns(2)
    render_image_or_wait(
        figure_cols[0],
        paths.figures_dir / "moment_sample_consistency.png",
        "MomentHead vs candidate-sample diagnostic",
    )
    render_image_or_wait(
        figure_cols[1],
        paths.figures_dir / "prior_vs_posterior_landscape.png",
        "Prior vs posterior state occupancy",
    )
    v2_figure_cols = st.columns(3)
    render_image_or_wait(
        v2_figure_cols[0],
        paths.figures_dir / "hn_variance_calibration.png",
        "HN calibration",
    )
    render_image_or_wait(
        v2_figure_cols[1],
        paths.figures_dir / "cprime_outlier_audit.png",
        "C' audit",
    )
    render_image_or_wait(
        v2_figure_cols[2],
        paths.figures_dir / "state_diversity_diagnostics.png",
        "State diversity",
    )
    v3_figure_cols = st.columns(3)
    render_image_or_wait(
        v3_figure_cols[0],
        paths.figures_dir / "ccc_decomposition_by_family.png",
        "Raw vs calibrated CCC",
    )
    render_image_or_wait(
        v3_figure_cols[1],
        paths.figures_dir / "entry_family_calibration_heatmap.png",
        "Entry/family calibration",
    )
    render_image_or_wait(
        v3_figure_cols[2],
        paths.figures_dir / "hn_context_audit.png",
        "HN diagnostics",
    )
    v3_figure_cols_2 = st.columns(3)
    render_image_or_wait(
        v3_figure_cols_2[0],
        paths.figures_dir / "cprime_residual_trend_audit.png",
        "C' residual trend",
    )
    render_image_or_wait(
        v3_figure_cols_2[1],
        paths.figures_dir / "joint_hn_n_density.png",
        "Joint HN-N posterior",
    )
    render_image_or_wait(
        v3_figure_cols_2[2],
        paths.figures_dir / "joint_ca_c_density.png",
        "Joint CA-C posterior",
    )
    v3_pair_cols = st.columns(2)
    render_image_or_wait(
        v3_pair_cols[0],
        paths.figures_dir / "hn_n_pair_evidence_audit.png",
        "HN <- N paired evidence",
    )
    render_image_or_wait(
        v3_pair_cols[1],
        paths.figures_dir / "cprime_ca_pair_evidence_audit.png",
        "C' <- CA paired evidence",
    )
    v3_masked_cols = st.columns(2)
    render_image_or_wait(
        v3_masked_cols[0],
        paths.figures_dir / "hn_masked_holdout_scatter.png",
        "HN masked-holdout posterior mean",
    )
    render_image_or_wait(
        v3_masked_cols[1],
        paths.figures_dir / "cprime_masked_holdout_scatter.png",
        "C' masked-holdout posterior mean",
    )

    moment_target_plot = (
        None if moment_frame is None else moment_target_figure(moment_frame)
    )
    if moment_target_plot is not None:
        st.plotly_chart(moment_target_plot, width="stretch")

    sample_support_plot = (
        None if moment_frame is None else sample_support_figure(moment_frame)
    )
    if sample_support_plot is not None:
        st.plotly_chart(sample_support_plot, width="stretch")

    consistency_figure = (
        None if moment_frame is None else moment_consistency_figure(moment_frame)
    )
    if consistency_figure is not None:
        st.plotly_chart(consistency_figure, width="stretch")
    elif moment_frame is None or moment_frame.empty:
        st.info("Waiting for moment_predictions.parquet.")

    state_figure = (
        None if state_frame is None else posterior_state_occupancy_figure(state_frame)
    )
    if state_figure is not None:
        st.plotly_chart(state_figure, width="stretch")
    elif state_frame is None or state_frame.empty:
        st.info("Waiting for posterior_state_tokens.parquet.")

    latent_frame = snapshot.get("posterior_latent_samples")
    if latent_frame is not None and not latent_frame.empty:
        with st.expander("Posterior latent samples", expanded=False):
            st.dataframe(latent_frame.head(400), width="stretch", hide_index=True)
    weights_frame = snapshot.get("moment_oracle_weights")
    if weights_frame is not None and not weights_frame.empty:
        with st.expander("Moment oracle candidate weights", expanded=False):
            st.dataframe(
                weights_frame.head(600),
                width="stretch",
                hide_index=True,
            )
    feature_frame = snapshot.get("nmr_structural_features")
    if feature_frame is not None and not feature_frame.empty:
        with st.expander("NMR structural feature adapter rows", expanded=False):
            st.dataframe(
                feature_frame.head(600),
                width="stretch",
                hide_index=True,
            )


def render_candidate_free_panels(st: Any, snapshot: dict[str, Any]) -> None:
    """Render candidate-free posterior diagnostics when available."""

    report = snapshot.get("candidate_free_report") or {}
    predictions = snapshot.get("candidate_free_predictions")
    masked = snapshot.get("candidate_free_masked_predictions")
    states = snapshot.get("candidate_free_state_tokens")
    reference_dir = _initial_checkpoint_figures_dir(snapshot)
    is_candidate_free = bool(report) or (
        predictions is not None and not predictions.empty
    ) or reference_dir is not None
    if not is_candidate_free:
        return

    st.subheader("Candidate-Free NMR Posterior")
    if not report and not _frame_available(predictions) and reference_dir is not None:
        st.info(
            "Current run has not produced candidate-free artifacts yet. "
            "Showing initial-checkpoint reference figures until the first "
            "best-preview or final render is available."
        )
    metric_cols = st.columns(4)
    best_values = (
        (snapshot.get("summary") or {})
        .get("best_checkpoint_metric", {})
        .get("values", {})
    )
    val_masked_values = (report.get("val_masked") or {}) if isinstance(report, dict) else {}
    hn_cprime_value = best_values.get("val_masked_HN_Cprime_ccc_macro")
    if hn_cprime_value is None:
        hn_cprime_value = val_masked_values.get("HN_Cprime_macro")
    hn_value = best_values.get("val_masked_HN_ccc_macro")
    if hn_value is None and isinstance(val_masked_values.get("HN"), dict):
        hn_value = val_masked_values["HN"].get("ccc")
    cprime_value = best_values.get("val_masked_C'_ccc_macro")
    if cprime_value is None and isinstance(val_masked_values.get("C'"), dict):
        cprime_value = val_masked_values["C'"].get("ccc")
    with metric_cols[0]:
        st.metric(
            "Best HN/C' CCC",
            _format_metric_value(hn_cprime_value),
            help="Primary candidate-free checkpoint metric.",
        )
    with metric_cols[1]:
        st.metric(
            "Best HN CCC",
            _format_metric_value(hn_value),
        )
    with metric_cols[2]:
        st.metric(
            "Best C' CCC",
            _format_metric_value(cprime_value),
        )
    with metric_cols[3]:
        target = report.get("target_ccc", 0.95)
        st.metric("Target", _format_metric_value(target))

    if report:
        with st.expander("Candidate-free contract", expanded=False):
            st.json(report, expanded=False)

    figure_cols = st.columns(3)
    render_candidate_free_current_or_reference_image(
        figure_cols[0],
        snapshot,
        "candidate_free_hn_masked_scatter.png",
        "Candidate-free masked HN",
    )
    render_candidate_free_current_or_reference_image(
        figure_cols[1],
        snapshot,
        "candidate_free_cprime_masked_scatter.png",
        "Candidate-free masked C'",
    )
    render_candidate_free_current_or_reference_image(
        figure_cols[2],
        snapshot,
        "candidate_free_hn_cprime_target_progress.png",
        "HN/C' target progress",
    )

    diagnostic_cols = st.columns(3)
    render_candidate_free_current_or_reference_image(
        diagnostic_cols[0],
        snapshot,
        "candidate_free_ccc_decomposition.png",
        "CCC decomposition",
    )
    render_candidate_free_current_or_reference_image(
        diagnostic_cols[1],
        snapshot,
        "candidate_free_sigma_calibration.png",
        "Sigma calibration",
    )
    render_candidate_free_current_or_reference_image(
        diagnostic_cols[2],
        snapshot,
        "candidate_free_outlier_entry_mae.png",
        "Worst outlier entries",
    )
    outlier_cols = st.columns(2)
    render_candidate_free_current_or_reference_image(
        outlier_cols[0],
        snapshot,
        "candidate_free_outlier_ceiling.png",
        "Outlier ceiling",
    )
    render_candidate_free_current_or_reference_image(
        outlier_cols[1],
        snapshot,
        "candidate_free_hn_residue_risk.png",
        "HN residue-risk diagnostic",
    )
    physics_cols = st.columns(4)
    render_candidate_free_current_or_reference_image(
        physics_cols[0],
        snapshot,
        "candidate_free_hn_physics_risk.png",
        "HN physical-risk context diagnostic",
    )
    render_candidate_free_current_or_reference_image(
        physics_cols[1],
        snapshot,
        "candidate_free_hn_condition_gap.png",
        "HN sample-condition gap",
    )
    render_candidate_free_current_or_reference_image(
        physics_cols[2],
        snapshot,
        "candidate_free_hn_correction_screen.png",
        "HN correction screen",
    )
    render_candidate_free_current_or_reference_image(
        physics_cols[3],
        snapshot,
        "candidate_free_hn_specialist_screen.png",
        "HN specialist screen",
    )
    macro_cols = st.columns(2)
    render_candidate_free_current_or_reference_image(
        macro_cols[0],
        snapshot,
        "candidate_free_hn_entry_macro_bottlenecks.png",
        "HN entry macro bottlenecks",
    )
    entry_bottlenecks = snapshot.get("candidate_free_hn_entry_bottlenecks")
    if _frame_available(entry_bottlenecks):
        with macro_cols[1].expander("Worst HN entry macro bottlenecks", expanded=True):
            st.dataframe(
                entry_bottlenecks.head(80),
                width="stretch",
                hide_index=True,
            )
    strategy_cols = st.columns(2)
    render_candidate_free_current_or_reference_image(
        strategy_cols[0],
        snapshot,
        "candidate_free_hn_strategy_audit.png",
        "HN strategy audit: observed correction vs masked generalization",
    )
    strategy_audit = snapshot.get("candidate_free_hn_strategy_audit")
    if _frame_available(strategy_audit):
        with strategy_cols[1].expander(
            "HN strategy audit table",
            expanded=True,
        ):
            st.caption(
                "Compares raw and corrected HN predictions per validation entry. "
                "Rows flagged as strategy_mismatch are the cases where observed-row "
                "correction preference does not match masked-holdout behavior."
            )
            st.dataframe(
                strategy_audit.head(80),
                width="stretch",
                hide_index=True,
            )
    ceiling_cols = st.columns(2)
    render_candidate_free_current_or_reference_image(
        ceiling_cols[0],
        snapshot,
        "candidate_free_hn_ceiling_policy.png",
        "HN ceiling policy: next-action buckets",
    )
    ceiling_policy = snapshot.get("candidate_free_hn_ceiling_policy")
    if _frame_available(ceiling_policy):
        with ceiling_cols[1].expander(
            "HN ceiling policy table",
            expanded=True,
        ):
            st.caption(
                "Classifies hard HN entries into next-action buckets. This is "
                "diagnostic-only and uses masked targets only after prediction "
                "to choose the next experiment, not to calibrate the current run."
            )
            st.dataframe(
                ceiling_policy.head(80),
                width="stretch",
                hide_index=True,
            )
    structure_cols = st.columns(2)
    render_candidate_free_current_or_reference_image(
        structure_cols[0],
        snapshot,
        "candidate_free_hn_structure_proxy_audit.png",
        "HN structure proxy audit",
    )
    structure_audit = snapshot.get("candidate_free_hn_structure_proxy_audit")
    if _frame_available(structure_audit):
        with structure_cols[1].expander(
            "HN structure proxy audit table",
            expanded=True,
        ):
            st.caption(
                "Checks whether masked HN bottlenecks have usable AF/BioEmu "
                "geometry proxies before relying on structure-aware repair."
            )
            st.dataframe(
                structure_audit.head(80),
                width="stretch",
                hide_index=True,
            )

    if masked is not None and not masked.empty:
        audit = (
            masked.groupby(["split", "atom_family"], dropna=False)
            .agg(
                masked_rows=("is_masked_holdout", "sum"),
                mean_sigma=("posterior_sigma", "mean"),
            )
            .reset_index()
        )
        st.write("Evidence coverage / mask audit")
        st.dataframe(audit, width="stretch", hide_index=True)
    else:
        st.info("Waiting for candidate_free_masked_predictions.parquet.")

    render_candidate_free_physics_risk_panels(st, snapshot)
    render_candidate_free_hn95_panels(st, snapshot)
    render_candidate_free_repair_panels(st, snapshot)

    if states is not None and not states.empty:
        with st.expander("Candidate-free latent state diagnostics", expanded=False):
            st.dataframe(states.head(300), width="stretch", hide_index=True)


def render_candidate_free_current_or_reference_image(
    st_container: Any,
    snapshot: dict[str, Any],
    file_name: str,
    caption: str,
) -> None:
    """Render current candidate-free PNG, or initial-checkpoint reference PNG."""

    paths = snapshot["paths"]
    path = _current_or_reference_figure(snapshot, paths.figures_dir / file_name)
    if _is_reference_figure(snapshot, path):
        caption = f"Reference initial checkpoint: {caption}"
    render_image_or_wait(st_container, path, caption, show_path=True)


def render_candidate_free_physics_risk_panels(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render candidate-free HN physical-risk diagnostics."""

    report = snapshot.get("candidate_free_physics_risk_report") or {}
    risk_table = snapshot.get("candidate_free_hn_physics_risk")
    outlier_rows = snapshot.get("candidate_free_hn_outlier_rows")
    if not report and not _frame_available(risk_table):
        return

    st.subheader("HN Physical-Risk / Outlier Screen")
    st.caption(
        "This panel separates HN errors by sequence-context risk tags. It is "
        "designed to avoid masked-row leakage: simple affine/offset screens "
        "are reported, but only enabled if they improve masked CCC."
    )
    if report:
        baseline = report.get("baseline") or {}
        cols = st.columns(4)
        cols[0].metric("HN CCC", _format_metric_value(baseline.get("ccc")))
        cols[1].metric("HN MAE", _format_metric_value(baseline.get("mae")))
        cols[2].metric("HN RMSE", _format_metric_value(baseline.get("rmse")))
        cols[3].metric(
            "Gap to 0.95",
            _format_metric_value(
                (report.get("target_ccc", 0.95) - baseline.get("ccc"))
                if baseline.get("ccc") is not None
                else None
            ),
        )
        correction_screen = report.get("correction_screen") or []
        if correction_screen:
            with st.expander("Rejected simple HN correction screens", expanded=False):
                st.dataframe(correction_screen, width="stretch", hide_index=True)
        with st.expander("HN physics-risk report JSON", expanded=False):
            st.json(report, expanded=False)

    if _frame_available(risk_table):
        st.write("HN risk-tag summary")
        st.dataframe(risk_table.head(80), width="stretch", hide_index=True)
    if _frame_available(outlier_rows):
        with st.expander("Worst HN outlier rows with physical-risk tags", expanded=True):
            st.dataframe(outlier_rows.head(120), width="stretch", hide_index=True)


def render_candidate_free_hn95_panels(st: Any, snapshot: dict[str, Any]) -> None:
    """Render HN 0.95 target diagnostics."""

    report = snapshot.get("hn95_report") or {}
    regime_table = snapshot.get("hn_regime_predictions")
    graph_rows = snapshot.get("hn_evidence_graph_residuals")
    reference_dir = _initial_checkpoint_figures_dir(snapshot)
    has_reference = reference_dir is not None and any(
        (reference_dir / name).exists()
        for name in (
            "hn95_progress.png",
            "hn_regime_ccc.png",
            "hn_state_signature_audit.png",
            "hn_state_tangent_audit.png",
            "hn_tangent_sign_separability_audit.png",
            "hn_tangent_variant_compare.png",
            "hn_graph_before_after.png",
            "hn_tail_outlier_audit.png",
        )
    )
    has_artifact = (
        bool(report)
        or _frame_available(regime_table)
        or _frame_available(graph_rows)
        or has_reference
    )
    if not has_artifact:
        return

    st.subheader("HN 0.95 Target Diagnostics")
    if not report and has_reference:
        st.info(
            "Current HN95 diagnostics are not rendered yet. Showing the "
            "initial-checkpoint reference HN95 PNGs."
        )
    st.caption(
        "HN is tracked as the primary repair target while C' is held as a "
        "guardrail. Regime and graph panels are evidence-only diagnostics for "
        "exchange, ring-current, terminal/disorder, alignment, and charged/sulfur rows."
    )
    if report:
        hn = report.get("HN") or {}
        cprime = report.get("C'") or {}
        cols = st.columns(4)
        cols[0].metric("HN / 0.95", _format_metric_value(hn.get("ccc")))
        cols[1].metric(
            "HN gap",
            _format_metric_value(
                report.get("target_hn_ccc", 0.95) - hn.get("ccc")
                if hn.get("ccc") is not None
                else None
            ),
        )
        cols[2].metric("C' guardrail", _format_metric_value(cprime.get("ccc")))
        cols[3].metric(
            "HN/C' macro",
            _format_metric_value(report.get("HN_Cprime_macro")),
        )
        graph_delta = report.get("hn_graph_delta") or {}
        st.write(
            "HN graph delta: "
            f"mean={_format_metric_value(graph_delta.get('mean'))}, "
            f"mean abs={_format_metric_value(graph_delta.get('mean_abs'))}, "
            f"p95 abs={_format_metric_value(graph_delta.get('p95_abs'))}"
        )

    figure_cols = st.columns(4)
    render_candidate_free_current_or_reference_image(
        figure_cols[0],
        snapshot,
        "hn95_progress.png",
        "HN 0.95 progress",
    )
    render_candidate_free_current_or_reference_image(
        figure_cols[1],
        snapshot,
        "hn_regime_ccc.png",
        "HN regime CCC",
    )
    render_candidate_free_current_or_reference_image(
        figure_cols[2],
        snapshot,
        "hn_graph_before_after.png",
        "HN graph before/after",
    )
    render_candidate_free_current_or_reference_image(
        figure_cols[3],
        snapshot,
        "hn_tail_outlier_audit.png",
        "HN tail outlier audit",
    )
    state_cols = st.columns(5)
    render_candidate_free_current_or_reference_image(
        state_cols[0],
        snapshot,
        "hn_state_signature_audit.png",
        "HN state-coordinate audit",
    )
    render_candidate_free_current_or_reference_image(
        state_cols[1],
        snapshot,
        "hn_state_tangent_audit.png",
        "HN state tangent audit",
    )
    render_candidate_free_current_or_reference_image(
        state_cols[2],
        snapshot,
        "hn_tangent_sign_separability_audit.png",
        "HN tangent sign separability",
    )
    render_candidate_free_current_or_reference_image(
        state_cols[3],
        snapshot,
        "hn_tangent_variant_compare.png",
        "HN tangent variants",
    )
    render_candidate_free_current_or_reference_image(
        state_cols[4],
        snapshot,
        "candidate_free_hn_entry_macro_bottlenecks.png",
        "HN entry bottlenecks",
    )
    if _frame_available(regime_table):
        with st.expander("HN regime metrics", expanded=True):
            st.dataframe(regime_table.head(80), width="stretch", hide_index=True)
    prediction_frame = snapshot.get("candidate_free_predictions")
    preservation = candidate_free_hn_entry_preservation_table(prediction_frame)
    if _frame_available(preservation):
        with st.expander("HN entry preservation: raw vs calibrated", expanded=True):
            st.caption(
                "Shows whether evidence calibration helps each validation entry. "
                "Negative delta means the post-hoc posterior correction hurt that entry."
            )
            st.dataframe(preservation.head(80), width="stretch", hide_index=True)
    strategy_audit = snapshot.get("candidate_free_hn_strategy_audit")
    if _frame_available(strategy_audit):
        with st.expander(
            "HN strategy audit: observed correction vs masked generalization",
            expanded=True,
        ):
            st.caption(
                "This is the selector failure audit for HN repair. If observed_delta_ccc "
                "and masked_delta_ccc disagree, stronger correction can improve training "
                "loss while hurting validation macro CCC."
            )
            st.dataframe(strategy_audit.head(80), width="stretch", hide_index=True)
    if _frame_available(graph_rows):
        with st.expander("HN evidence graph residual rows", expanded=False):
            st.dataframe(graph_rows.head(160), width="stretch", hide_index=True)


def render_candidate_free_repair_panels(st: Any, snapshot: dict[str, Any]) -> None:
    """Render residue-numbering repair and physics-risk diagnostics."""

    repaired = snapshot.get("candidate_free_repaired_targets")
    if not _frame_available(repaired):
        repaired = snapshot.get("candidate_free_predictions")
    if not _frame_available(repaired):
        st.info("Waiting for candidate-free repair diagnostics.")
        return

    frame = repaired.copy()
    if "model_path" in frame.columns:
        frame = frame.loc[frame["model_path"].astype(str) == "candidate_free"].copy()
    if frame.empty or "repair_status" not in frame.columns:
        st.info("This run predates residue-numbering repair diagnostics.")
        return

    st.subheader("Residue Numbering Repair / Physics Outlier Audit")
    report = snapshot.get("residue_numbering_alignment_report") or {}
    if report:
        cols = st.columns(4)
        cols[0].metric("Repair rows", str(report.get("rows", len(frame))))
        cols[1].metric(
            "Mean alignment confidence",
            _format_metric_value(report.get("mean_alignment_confidence")),
        )
        status_counts = report.get("status_counts") or {}
        cols[2].metric("Offset rows", str(status_counts.get("offset", 0)))
        cols[3].metric(
            "Risk rows",
            str(
                int(status_counts.get("low_confidence", 0))
                + int(status_counts.get("out_of_range", 0))
            ),
        )
        worst = report.get("worst_entities") or []
        if worst:
            with st.expander("Worst alignment-risk entries", expanded=False):
                st.dataframe(worst, width="stretch", hide_index=True)

    if {"split", "atom_family", "repair_status"}.issubset(frame.columns):
        aggregations: dict[str, tuple[str, str]] = {"rows": ("target_id", "count")}
        if "alignment_confidence" in frame.columns:
            aggregations["mean_alignment_confidence"] = (
                "alignment_confidence",
                "mean",
            )
        if "posterior_sigma" in frame.columns:
            aggregations["mean_sigma"] = ("posterior_sigma", "mean")
        if "outlier_reliability" in frame.columns:
            aggregations["mean_outlier_reliability"] = (
                "outlier_reliability",
                "mean",
            )
        status_table = (
            frame.groupby(["split", "atom_family", "repair_status"], dropna=False)
            .agg(**aggregations)
            .reset_index()
        )
    else:
        status_table = []
    st.write("Repair status by split/family")
    st.dataframe(status_table, width="stretch", hide_index=True)

    eval_frame = _candidate_free_eval_frame(snapshot)
    if not _frame_available(eval_frame):
        return
    risk_columns = [
        column
        for column in [
            "alignment_confidence",
            "alignment_risk",
            "exchange_risk_proxy",
            "ring_current_proxy",
            "cprime_plane_proxy",
            "disorder_proxy",
            "outlier_reliability",
            "abs_error",
            "z_error",
        ]
        if column in eval_frame.columns
    ]
    if risk_columns:
        worst_columns = [
            column
            for column in [
                "split",
                "entity_uid",
                "target_id",
                "atom_family",
                "original_residue_index",
                "aligned_residue_index",
                "repair_status",
                "target_value",
                "predicted_value",
                *risk_columns,
            ]
            if column in eval_frame.columns
        ]
        worst_rows = eval_frame.sort_values("abs_error", ascending=False).head(50)
        with st.expander("Worst HN/C' outlier rows with repair/physics tags", expanded=True):
            st.dataframe(worst_rows[worst_columns], width="stretch", hide_index=True)


def render_task_scope_notice(
    st: Any,
    snapshot: dict[str, Any],
    *,
    title: str,
    text: str,
    expected: list[str],
) -> None:
    """Render a clear non-waiting notice for task-specific artifacts."""

    st.info(f"{title}\n\n{text}")
    render_candidate_free_metric_strip(st, snapshot)
    with st.expander("Expected artifacts for this tab", expanded=False):
        st.dataframe(
            [
                {
                    "artifact": artifact,
                    "status": "not produced by current task",
                }
                for artifact in expected
            ],
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "For this run, use Overview latest PNGs, Analysis PNGs, and the "
        "NMR Chemical Shifts tab as the canonical candidate-free analysis."
    )


def render_candidate_free_nmr_reuse(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render NMR-style diagnostics from candidate-free prediction tables."""

    frame = _candidate_free_eval_frame(snapshot)
    if frame.empty:
        return
    st.subheader("Candidate-Free NMR Posterior Mean Diagnostics")
    st.caption(
        "Candidate-free does not emit quantile posterior tables yet, but its "
        "mean/sigma rows support scatter, CCC decomposition, residual profile, "
        "and sigma-vs-error diagnostics."
    )
    table = _candidate_free_family_table(frame)
    if not table.empty:
        st.dataframe(table, width="stretch", hide_index=True)
    cols = st.columns(2)
    scatter = _candidate_free_scatter_figure(
        frame,
        title="Candidate-free posterior mean vs experiment",
    )
    if scatter is not None:
        cols[0].plotly_chart(
            scatter,
            width="stretch",
            key="candidate_free_nmr_scatter",
        )
    calibration = _candidate_free_uncertainty_calibration_figure(frame)
    if calibration is not None:
        cols[1].plotly_chart(
            calibration,
            width="stretch",
            key="candidate_free_nmr_calibration",
        )
    render_candidate_free_decomposition_and_topology(
        st,
        frame,
        key_prefix="candidate_free_nmr",
    )
    render_candidate_free_pair_proxy(st, frame, key_prefix="candidate_free_nmr")
    render_candidate_free_optional_evaluation_slots(st, snapshot)


def render_candidate_free_decomposition_and_topology(
    st: Any,
    frame: Any,
    *,
    key_prefix: str,
) -> None:
    """Render CCC decomposition and residue-error topology diagnostics."""

    st.subheader("Candidate-Free CCC Decomposition")
    decomposition = _candidate_free_decomposition_table(frame)
    if not decomposition.empty:
        st.dataframe(decomposition, width="stretch", hide_index=True)
    else:
        st.info("Insufficient rows for CCC decomposition.")

    cols = st.columns(2)
    z_histogram = _candidate_free_z_error_histogram(frame)
    if z_histogram is not None:
        cols[0].plotly_chart(
            z_histogram,
            width="stretch",
            key=f"{key_prefix}_z_error_histogram",
        )
    segment_plot = _candidate_free_segment_figure(frame)
    if segment_plot is not None:
        cols[1].plotly_chart(
            segment_plot,
            width="stretch",
            key=f"{key_prefix}_segment_plot",
        )

    worst_entries = _candidate_free_worst_entry_table(frame)
    if not worst_entries.empty:
        with st.expander("Worst-entry HN/C' dashboard", expanded=False):
            st.dataframe(worst_entries.head(80), width="stretch", hide_index=True)

    segments = _candidate_free_error_segments(frame)
    if not segments.empty:
        with st.expander("Residue error topology segments", expanded=False):
            st.dataframe(segments.head(120), width="stretch", hide_index=True)


def render_candidate_free_pair_proxy(
    st: Any,
    frame: Any,
    *,
    key_prefix: str,
) -> None:
    """Render same-residue or neighbor-context paired NMR proxy panels."""

    st.subheader("Candidate-Free Joint-Style Pair Proxies")
    pair_specs = [
        ("HN_N", "HN", "N", 0, "same-residue HN-N proxy"),
        ("Cprime_CA", "C'", "CA", 0, "same-residue C'-CA proxy"),
        ("Cprime_prev_CA", "C'", "CA", -1, "C' with previous-residue CA proxy"),
        ("Cprime_next_CA", "C'", "CA", 1, "C' with next-residue CA proxy"),
    ]
    cols = st.columns(2)
    rendered = 0
    pair_tables = []
    for pair_index, (name, target_family, source_family, offset, title) in enumerate(
        pair_specs
    ):
        pair_frame = _candidate_free_pair_frame(
            frame,
            target_family=target_family,
            source_family=source_family,
            source_residue_offset=offset,
        )
        if pair_frame.empty:
            pair_tables.append(
                {
                    "pair": name,
                    "status": "insufficient paired rows",
                    "rows": 0,
                }
            )
            continue
        pair_tables.append(
            {
                "pair": name,
                "status": "ok",
                "rows": int(len(pair_frame)),
                "target_residual_source_corr": _safe_corr_series(
                    pair_frame["target_residual"],
                    pair_frame["source_residual"],
                ),
                "target_value_source_value_corr": _safe_corr_series(
                    pair_frame["target_value"],
                    pair_frame["source_value"],
                ),
            }
        )
        figure = _candidate_free_pair_figure(pair_frame, title=title)
        if figure is not None and rendered < 2:
            cols[rendered].plotly_chart(
                figure,
                width="stretch",
                key=f"{key_prefix}_{name}_pair_proxy",
            )
            rendered += 1

    if pair_tables:
        try:
            import pandas as pd

            pair_table = pd.DataFrame(pair_tables)
        except ImportError:
            pair_table = pair_tables
        with st.expander("Pair proxy row summary", expanded=False):
            st.dataframe(pair_table, width="stretch", hide_index=True)


def render_candidate_free_optional_evaluation_slots(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Show optional candidate-free ablation artifacts when available."""

    paths = snapshot["paths"]
    specs = [
        (
            "Leave-family-out predictions",
            paths.arrays_dir / "candidate_free_leave_family_out_predictions.parquet",
        ),
        (
            "Block-mask predictions",
            paths.arrays_dir / "candidate_free_block_mask_predictions.parquet",
        ),
        (
            "Candidate-free ablation report",
            paths.run_dir / "reports" / "metrics" / "candidate_free_ablation_report.json",
        ),
    ]
    rows = []
    for label, path in specs:
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "status": (
                    "available"
                    if path.exists()
                    else "requires ablation evaluation pass"
                ),
            }
        )
    with st.expander("Optional candidate-free evaluation artifacts", expanded=False):
        st.dataframe(rows, width="stretch", hide_index=True)
        for label, path in specs:
            if path.suffix == ".json" and path.exists():
                try:
                    st.json(json.loads(path.read_text()), expanded=False)
                except (OSError, json.JSONDecodeError):
                    st.warning(f"{label} is present but could not be read yet.")
            elif path.suffix == ".parquet" and path.exists():
                frame = _read_optional_parquet(path)
                if _frame_available(frame):
                    st.write(label)
                    st.dataframe(frame.head(300), width="stretch", hide_index=True)


def render_candidate_free_metric_strip(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render compact candidate-free best-checkpoint cards."""

    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    best_values = (summary.get("best_checkpoint_metric") or {}).get("values", {})
    report = snapshot.get("candidate_free_report") or {}
    cols = st.columns(5)
    with cols[0]:
        st.metric("Best epoch", _display(summary.get("best_epoch")))
    with cols[1]:
        st.metric(
            "HN/C' CCC",
            _format_metric_value(best_values.get("val_masked_HN_Cprime_ccc_macro")),
        )
    with cols[2]:
        st.metric(
            "HN CCC",
            _format_metric_value(best_values.get("val_masked_HN_ccc_macro")),
        )
    with cols[3]:
        st.metric(
            "C' CCC",
            _format_metric_value(best_values.get("val_masked_C'_ccc_macro")),
        )
    with cols[4]:
        st.metric("Target", _format_metric_value(report.get("target_ccc", 0.95)))


def _is_candidate_free_snapshot(snapshot: dict[str, Any]) -> bool:
    """Return whether the current snapshot is from a candidate-free run."""

    if isinstance(snapshot.get("candidate_free_report"), dict) and snapshot.get(
        "candidate_free_report"
    ):
        return True
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    if isinstance(summary.get("candidate_free"), dict):
        return True
    return _frame_available(snapshot.get("candidate_free_predictions")) or _frame_available(
        snapshot.get("candidate_free_masked_predictions")
    )


def _is_bioemu_latent_snapshot(snapshot: dict[str, Any]) -> bool:
    """Return whether the current snapshot is from the BioEmu latent atlas task."""

    paths = snapshot.get("paths")
    run_name = str(getattr(paths, "run_dir", ""))
    if "bioemu_latent_atlas" in run_name or "bioemu_latent_nmr" in run_name:
        return True
    if isinstance(snapshot.get("bioemu_live_epoch_progress"), dict) and snapshot.get(
        "bioemu_live_epoch_progress"
    ):
        return True
    if paths is not None and getattr(paths, "bioemu_live_epoch_progress", None) is not None:
        try:
            if paths.bioemu_live_epoch_progress.exists():
                return True
        except OSError:
            pass
    if isinstance(snapshot.get("bioemu_latent_nmr_report"), dict) and snapshot.get(
        "bioemu_latent_nmr_report"
    ):
        return True
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    if summary.get("task") == "posterior_nmr_bioemu_latent_atlas_v1":
        return True
    if isinstance(summary.get("provider"), dict) and summary.get("provider", {}).get("name"):
        return True
    metric_frame = snapshot.get("metric_frame")
    if metric_frame is not None and not getattr(metric_frame, "empty", True):
        try:
            metric_names = set(metric_frame["metric_name"].astype(str).unique().tolist())
        except Exception:
            metric_names = set()
        if {
            "masked_HN_Cprime_family_cprime_balanced_ccc_macro",
            "posterior_bridge_norm",
            "posterior_ess",
        } & metric_names:
            return True
    return _frame_available(snapshot.get("bioemu_latent_nmr_predictions")) or _frame_available(
        snapshot.get("bioemu_latent_nmr_masked_predictions")
    )


def _frame_available(frame: Any) -> bool:
    """Return True when a loaded optional DataFrame has rows."""

    return frame is not None and not getattr(frame, "empty", True)


def candidate_free_hn_entry_preservation_table(frame: Any) -> Any:
    """Summarize per-entry HN raw-vs-calibrated preservation."""

    if not _frame_available(frame):
        return None
    required = {
        "entity_uid",
        "split",
        "atom_family",
        "is_masked_holdout",
        "target_value",
        "predicted_value",
        "raw_predicted_value",
    }
    if not required.issubset(set(frame.columns)):
        return None
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return None
    work = frame.loc[
        frame["split"].astype(str).eq("val")
        & frame["atom_family"].astype(str).eq("HN")
        & frame["is_masked_holdout"].astype(bool)
    ].copy()
    if "model_path" in work.columns:
        work = work.loc[work["model_path"].astype(str).eq("candidate_free")].copy()
    if work.empty:
        return None
    rows: list[dict[str, Any]] = []
    for entity_uid, group in work.groupby("entity_uid", dropna=False):
        if len(group) < 2:
            continue
        target = _to_numeric_series(group["target_value"])
        predicted = _to_numeric_series(group["predicted_value"])
        raw = _to_numeric_series(group["raw_predicted_value"])
        pred_residual = predicted - target
        raw_residual = raw - target
        delta_ccc = None
        raw_ccc = _safe_ccc_series(raw, target)
        calibrated_ccc = _safe_ccc_series(predicted, target)
        if raw_ccc is not None and calibrated_ccc is not None:
            delta_ccc = calibrated_ccc - raw_ccc
        rows.append(
            {
                "entity_uid": entity_uid,
                "rows": int(len(group)),
                "raw_ccc": raw_ccc,
                "calibrated_ccc": calibrated_ccc,
                "delta_ccc": delta_ccc,
                "raw_mae": _safe_mean(raw_residual.abs()),
                "calibrated_mae": _safe_mean(pred_residual.abs()),
                "delta_mae": (
                    _safe_mean(pred_residual.abs()) - _safe_mean(raw_residual.abs())
                    if _safe_mean(pred_residual.abs()) is not None
                    and _safe_mean(raw_residual.abs()) is not None
                    else None
                ),
                "target_std": _safe_std(target),
                "mean_alignment_confidence": _safe_mean(
                    _to_numeric_series(group["alignment_confidence"])
                )
                if "alignment_confidence" in group.columns
                else None,
                "mean_outlier_reliability": _safe_mean(
                    _to_numeric_series(group["outlier_reliability"])
                )
                if "outlier_reliability" in group.columns
                else None,
            }
        )
    if not rows:
        return None
    table = pd.DataFrame(rows)
    table["preservation_flag"] = np.where(
        table["delta_ccc"].fillna(0.0) < -0.01,
        "correction_hurt",
        np.where(table["calibrated_ccc"].fillna(0.0) < 0.85, "needs_repair", "ok"),
    )
    return table.sort_values(["delta_ccc", "calibrated_ccc"], ascending=[True, True])


def _has_energy_landscape_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether BioEmu-style landscape artifacts are present."""

    paths = snapshot["paths"]
    return _frame_available(snapshot.get("projection_coordinates")) or (
        paths.figures_dir / "landscape_teacher_vs_student.png"
    ).exists()


def _has_ccc_geometry_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether CCC support/oracle geometry artifacts are present."""

    benchmark = snapshot.get("benchmark_report")
    has_support = isinstance(benchmark, dict) and bool(benchmark.get("support_ceiling"))
    return (
        has_support
        or _frame_available(snapshot.get("ccc_support_oracle"))
        or _frame_available(snapshot.get("forward_residual_predictions"))
        or _frame_available(snapshot.get("residue_atom_forward_residuals"))
    )


def _has_multi_observable_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether multi-observable adapter artifacts are present."""

    return bool(snapshot.get("observable_adapter_report")) or _frame_available(
        snapshot.get("observable_oracle_weights")
    )


def _has_moment_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether moment-diffusion artifacts are present."""

    return (
        bool(snapshot.get("moment_oracle_report"))
        or bool(snapshot.get("moment_consistency_report"))
        or _frame_available(snapshot.get("moment_predictions"))
        or _frame_available(snapshot.get("posterior_state_tokens"))
        or _frame_available(snapshot.get("posterior_latent_samples"))
    )


def _has_conformation_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether structure uncertainty artifacts are present."""

    paths = snapshot["paths"]
    return (
        paths.annotated_representative_pdb.exists()
        or _frame_available(snapshot.get("ensemble_states"))
        or _frame_available(snapshot.get("conformer_state_assignments"))
        or (paths.figures_dir / "ensemble_local_confidence_structure.png").exists()
    )


def _candidate_free_eval_frame(
    snapshot: dict[str, Any],
    *,
    masked_only: bool = True,
) -> Any:
    """Return candidate-free prediction rows with derived residual columns."""

    key = "candidate_free_masked_predictions" if masked_only else "candidate_free_predictions"
    frame = snapshot.get(key)
    if not _frame_available(frame):
        frame = snapshot.get("candidate_free_predictions")
    if not _frame_available(frame):
        return frame
    working = frame.copy()
    for column in ["target_value", "predicted_value", "posterior_sigma"]:
        if column in working.columns:
            working[column] = _to_numeric_series(working[column])
    if masked_only and "is_masked_holdout" in working.columns:
        masked = working.loc[working["is_masked_holdout"].astype(bool)].copy()
        if not masked.empty:
            working = masked
    required = {"target_value", "predicted_value"}
    if required.issubset(working.columns):
        working = working.dropna(subset=["target_value", "predicted_value"]).copy()
        working["residual"] = working["predicted_value"] - working["target_value"]
        working["abs_error"] = working["residual"].abs()
        working["squared_error"] = working["residual"] ** 2
        if "posterior_sigma" in working.columns:
            sigma = working["posterior_sigma"].where(working["posterior_sigma"] > 0)
            working["z_error"] = working["residual"] / sigma
            working["covered_1sigma"] = working["abs_error"] <= sigma
            working["covered_2sigma"] = working["abs_error"] <= (2.0 * sigma)
    return working


def _candidate_free_family_table(frame: Any) -> Any:
    """Build family-level candidate-free CCC and calibration rows."""

    if not _frame_available(frame):
        return []
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in ["split", "atom_family"]
        if column in frame.columns
    ]
    if not group_cols:
        group_cols = ["atom_family"] if "atom_family" in frame.columns else []
    iterator = frame.groupby(group_cols, dropna=False) if group_cols else [("all", frame)]
    for key, group in iterator:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        target = _to_numeric_series(group["target_value"])
        predicted = _to_numeric_series(group["predicted_value"])
        residual = predicted - target
        sigma = (
            _to_numeric_series(group["posterior_sigma"])
            if "posterior_sigma" in group.columns
            else None
        )
        row.update(
            {
                "rows": int(len(group)),
                "ccc": _safe_ccc_series(predicted, target),
                "pearson_r": _safe_corr_series(predicted, target),
                "mae": _safe_mean(residual.abs()),
                "rmse": _safe_rmse(residual),
                "bias": _safe_mean(residual),
                "target_std": _safe_std(target),
                "predicted_std": _safe_std(predicted),
            }
        )
        if row["target_std"] not in (None, 0.0):
            row["scale_ratio"] = row["predicted_std"] / row["target_std"]
        else:
            row["scale_ratio"] = None
        if sigma is not None:
            row["mean_sigma"] = _safe_mean(sigma)
            row["sigma_abs_error_corr"] = _safe_corr_series(sigma, residual.abs())
            row["coverage_1sigma"] = _safe_mean(residual.abs() <= sigma)
            row["coverage_2sigma"] = _safe_mean(residual.abs() <= 2.0 * sigma)
        rows.append(row)
    try:
        import pandas as pd

        result = pd.DataFrame(rows)
        sort_cols = [column for column in ["split", "atom_family"] if column in result]
        if sort_cols:
            result = result.sort_values(sort_cols, kind="stable")
        return result
    except ImportError:
        return rows


def _candidate_free_coverage_table(frame: Any) -> Any:
    """Return candidate-free evidence/mask coverage by family."""

    if not _frame_available(frame):
        return []
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in ["split", "atom_family"]
        if column in frame.columns
    ]
    iterator = frame.groupby(group_cols, dropna=False) if group_cols else [("all", frame)]
    for key, group in iterator:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        evidence = (
            group["is_evidence"].astype(bool)
            if "is_evidence" in group.columns
            else group.iloc[:, 0].astype(bool) & False
        )
        masked = (
            group["is_masked_holdout"].astype(bool)
            if "is_masked_holdout" in group.columns
            else group.iloc[:, 0].astype(bool) & False
        )
        row.update(
            {
                "rows": int(len(group)),
                "evidence_rows": int(evidence.sum()),
                "masked_holdout_rows": int(masked.sum()),
                "evaluated_rows": int((~evidence | masked).sum()),
                "mean_sigma": (
                    _safe_mean(_to_numeric_series(group["posterior_sigma"]))
                    if "posterior_sigma" in group.columns
                    else None
                ),
            }
        )
        rows.append(row)
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _candidate_free_residue_profile_table(frame: Any) -> Any:
    """Return largest residue-level candidate-free uncertainty/error rows."""

    if not _frame_available(frame):
        return []
    group_cols = [
        column
        for column in ["split", "entity_uid", "atom_family", "residue_index"]
        if column in frame.columns
    ]
    if not group_cols:
        return []
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            rows=("target_value", "size"),
            mean_abs_error=("abs_error", "mean"),
            mean_residual=("residual", "mean"),
            mean_sigma=("posterior_sigma", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["mean_abs_error", "mean_sigma"],
        ascending=[False, False],
        kind="stable",
    )


def _candidate_free_scatter_figure(frame: Any, *, title: str) -> Any | None:
    """Build candidate-free posterior mean scatter."""

    if not _frame_available(frame) or not _plotly_available_local():
        return None
    import plotly.express as px

    plot_frame = frame.copy()
    figure = px.scatter(
        plot_frame,
        x="target_value",
        y="predicted_value",
        color="atom_family" if "atom_family" in plot_frame.columns else None,
        symbol="split" if "split" in plot_frame.columns else None,
        hover_data=[
            column
            for column in ["entity_uid", "target_id", "residue_index", "posterior_sigma"]
            if column in plot_frame.columns
        ],
        title=title,
    )
    lower = float(min(plot_frame["target_value"].min(), plot_frame["predicted_value"].min()))
    upper = float(max(plot_frame["target_value"].max(), plot_frame["predicted_value"].max()))
    figure.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"dash": "dash", "color": "gray"},
    )
    figure.update_layout(
        xaxis_title="Experimental chemical shift (ppm)",
        yaxis_title="Candidate-free posterior mean (ppm)",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_sigma_error_figure(frame: Any) -> Any | None:
    """Build posterior sigma versus absolute error figure."""

    if not _frame_available(frame) or "posterior_sigma" not in frame.columns:
        return None
    if not _plotly_available_local():
        return None
    import plotly.express as px

    plot_frame = frame.dropna(subset=["posterior_sigma", "abs_error"]).copy()
    if plot_frame.empty:
        return None
    figure = px.scatter(
        plot_frame,
        x="posterior_sigma",
        y="abs_error",
        color="atom_family" if "atom_family" in plot_frame.columns else None,
        title="Posterior sigma vs absolute error",
        hover_data=[
            column
            for column in ["entity_uid", "target_id", "residue_index"]
            if column in plot_frame.columns
        ],
    )
    figure.update_layout(
        xaxis_title="Posterior sigma",
        yaxis_title="Absolute error (ppm)",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_uncertainty_calibration_figure(frame: Any) -> Any | None:
    """Build coverage by sigma multiplier for candidate-free rows."""

    if not _frame_available(frame) or "posterior_sigma" not in frame.columns:
        return None
    if not _plotly_available_local():
        return None
    import pandas as pd
    import plotly.express as px

    rows: list[dict[str, Any]] = []
    for split, split_frame in frame.groupby("split", dropna=False):
        for family, family_frame in split_frame.groupby("atom_family", dropna=False):
            sigma = _to_numeric_series(family_frame["posterior_sigma"])
            abs_error = _to_numeric_series(family_frame["abs_error"])
            valid = family_frame.loc[sigma.notna() & abs_error.notna()].copy()
            if valid.empty:
                continue
            for multiplier in [0.5, 1.0, 1.5, 2.0, 3.0]:
                rows.append(
                    {
                        "split": split,
                        "atom_family": family,
                        "sigma_multiplier": multiplier,
                        "coverage": float(
                            (
                                valid["abs_error"]
                                <= multiplier * valid["posterior_sigma"]
                            ).mean()
                        ),
                        "rows": int(len(valid)),
                    }
                )
    if not rows:
        return None
    plot_frame = pd.DataFrame(rows)
    figure = px.line(
        plot_frame,
        x="sigma_multiplier",
        y="coverage",
        color="atom_family",
        line_dash="split",
        markers=True,
        title="Candidate-free sigma coverage curve",
    )
    figure.update_layout(
        xaxis_title="Posterior sigma multiplier",
        yaxis_title="Empirical coverage",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_residual_profile_figure(
    frame: Any,
    *,
    value_column: str = "residual",
    title: str = "Candidate-free residual profile by residue",
) -> Any | None:
    """Build residue-index profile for candidate-free diagnostics."""

    required = {"residue_index", "atom_family", value_column}
    if not _frame_available(frame) or not required.issubset(frame.columns):
        return None
    if not _plotly_available_local():
        return None
    import plotly.express as px

    profile = (
        frame.groupby(["split", "atom_family", "residue_index"], dropna=False)[
            value_column
        ]
        .mean()
        .reset_index()
        .sort_values(["split", "atom_family", "residue_index"], kind="stable")
    )
    if profile.empty:
        return None
    figure = px.line(
        profile,
        x="residue_index",
        y=value_column,
        color="atom_family",
        line_dash="split",
        markers=False,
        title=title,
    )
    figure.update_layout(
        xaxis_title="Residue index",
        yaxis_title=value_column.replace("_", " "),
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_coverage_figure(frame: Any) -> Any | None:
    """Build evidence/masked row coverage bar chart."""

    table = _candidate_free_coverage_table(frame)
    if not _frame_available(table) or not _plotly_available_local():
        return None
    import plotly.express as px

    value_vars = [
        column
        for column in ["evidence_rows", "masked_holdout_rows"]
        if column in table.columns
    ]
    if not value_vars:
        return None
    id_vars = [
        column for column in ["split", "atom_family"] if column in table.columns
    ]
    plot_frame = table.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="row_type",
        value_name="row_count",
    )
    figure = px.bar(
        plot_frame,
        x="atom_family" if "atom_family" in plot_frame.columns else "row_type",
        y="row_count",
        color="row_type",
        facet_col="split" if "split" in plot_frame.columns else None,
        barmode="group",
        title="Candidate-free evidence and masked-holdout coverage",
    )
    figure.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return figure


def _candidate_free_decomposition_table(frame: Any) -> Any:
    """Return CCC decomposition rows by split and atom family."""

    if not _frame_available(frame):
        return []
    rows: list[dict[str, Any]] = []
    group_cols = [
        column for column in ["split", "atom_family"] if column in frame.columns
    ]
    iterator = frame.groupby(group_cols, dropna=False) if group_cols else [("all", frame)]
    for key, group in iterator:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        target = _to_numeric_series(group["target_value"])
        predicted = _to_numeric_series(group["predicted_value"])
        common = target.dropna().index.intersection(predicted.dropna().index)
        if len(common) < 2:
            continue
        target = target.loc[common]
        predicted = predicted.loc[common]
        target_mean = float(target.mean())
        predicted_mean = float(predicted.mean())
        target_std = float(target.std(ddof=0))
        predicted_std = float(predicted.std(ddof=0))
        pearson = _safe_corr_series(predicted, target)
        ccc = _safe_ccc_series(predicted, target)
        if target_std > 0 and predicted_std > 0:
            scale_ratio = predicted_std / target_std
            location_shift = (predicted_mean - target_mean) / (
                (predicted_std * target_std) ** 0.5
            )
            accuracy_cb = 2.0 / (
                scale_ratio + (1.0 / scale_ratio) + location_shift**2
            )
        else:
            scale_ratio = None
            location_shift = None
            accuracy_cb = None
        row.update(
            {
                "rows": int(len(common)),
                "ccc": ccc,
                "pearson_r": pearson,
                "accuracy_cb": accuracy_cb,
                "location_shift_u": location_shift,
                "scale_ratio_v": scale_ratio,
                "bias_ppm": predicted_mean - target_mean,
                "target_std": target_std,
                "predicted_std": predicted_std,
                "ccc_gap_to_095": None if ccc is None else 0.95 - ccc,
            }
        )
        rows.append(row)
    try:
        import pandas as pd

        result = pd.DataFrame(rows)
        sort_cols = [column for column in ["split", "atom_family"] if column in result]
        if sort_cols:
            result = result.sort_values(sort_cols, kind="stable")
        return result
    except ImportError:
        return rows


def _candidate_free_worst_entry_table(frame: Any) -> Any:
    """Return worst candidate-free entry/family rows."""

    required = {"entity_uid", "atom_family", "target_value", "predicted_value"}
    if not _frame_available(frame) or not required.issubset(frame.columns):
        return []
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in ["split", "entity_uid", "atom_family"]
        if column in frame.columns
    ]
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        target = _to_numeric_series(group["target_value"])
        predicted = _to_numeric_series(group["predicted_value"])
        residual = predicted - target
        row.update(
            {
                "rows": int(len(group)),
                "ccc": _safe_ccc_series(predicted, target),
                "mae": _safe_mean(residual.abs()),
                "rmse": _safe_rmse(residual),
                "bias": _safe_mean(residual),
                "mean_sigma": (
                    _safe_mean(group["posterior_sigma"])
                    if "posterior_sigma" in group.columns
                    else None
                ),
            }
        )
        rows.append(row)
    try:
        import pandas as pd

        result = pd.DataFrame(rows)
        if "atom_family" in result.columns:
            result = result.loc[result["atom_family"].isin(["HN", "C'"])].copy()
        return result.sort_values(
            ["ccc", "rmse"],
            ascending=[True, False],
            kind="stable",
        )
    except ImportError:
        return rows


def _candidate_free_error_segments(frame: Any) -> Any:
    """Return contiguous high-error residue segments."""

    required = {"entity_uid", "atom_family", "residue_index", "abs_error"}
    if not _frame_available(frame) or not required.issubset(frame.columns):
        return []
    rows: list[dict[str, Any]] = []
    group_cols = [
        column
        for column in ["split", "entity_uid", "atom_family"]
        if column in frame.columns
    ]
    for key, group in frame.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        working = group.dropna(subset=["residue_index", "abs_error"]).copy()
        if working.empty:
            continue
        threshold = max(
            float(working["abs_error"].quantile(0.90)),
            float(2.0 * working["abs_error"].median()),
        )
        high = working.loc[working["abs_error"] >= threshold].sort_values(
            "residue_index",
            kind="stable",
        )
        if high.empty:
            continue
        current: list[Any] = []
        previous_index: int | None = None
        for _, row in high.iterrows():
            residue_index = int(row["residue_index"])
            if previous_index is None or residue_index <= previous_index + 1:
                current.append(row)
            else:
                _append_candidate_free_segment(rows, base, current, threshold)
                current = [row]
            previous_index = residue_index
        if current:
            _append_candidate_free_segment(rows, base, current, threshold)
    try:
        import pandas as pd

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        return result.sort_values(
            ["mean_abs_error", "length"],
            ascending=[False, False],
            kind="stable",
        )
    except ImportError:
        return rows


def _append_candidate_free_segment(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    segment_rows: list[Any],
    threshold: float,
) -> None:
    """Append one contiguous candidate-free high-error segment row."""

    if not segment_rows:
        return
    residue_indices = [int(row["residue_index"]) for row in segment_rows]
    abs_errors = [float(row["abs_error"]) for row in segment_rows]
    sigmas = [
        float(row["posterior_sigma"])
        for row in segment_rows
        if "posterior_sigma" in row and row["posterior_sigma"] == row["posterior_sigma"]
    ]
    rows.append(
        {
            **base,
            "segment_start": min(residue_indices),
            "segment_end": max(residue_indices),
            "length": len(residue_indices),
            "threshold": threshold,
            "mean_abs_error": sum(abs_errors) / len(abs_errors),
            "max_abs_error": max(abs_errors),
            "mean_sigma": (sum(sigmas) / len(sigmas)) if sigmas else None,
        }
    )


def _candidate_free_z_error_histogram(frame: Any) -> Any | None:
    """Build z-error histogram for candidate-free posterior sigma."""

    if not _frame_available(frame) or "z_error" not in frame.columns:
        return None
    if not _plotly_available_local():
        return None
    import plotly.express as px

    plot_frame = frame.dropna(subset=["z_error"]).copy()
    if plot_frame.empty:
        return None
    plot_frame["z_error_clipped"] = plot_frame["z_error"].clip(-8.0, 8.0)
    figure = px.histogram(
        plot_frame,
        x="z_error_clipped",
        color="atom_family" if "atom_family" in plot_frame.columns else None,
        facet_col="split" if "split" in plot_frame.columns else None,
        nbins=80,
        barmode="overlay",
        opacity=0.68,
        title="Candidate-free z-error histogram",
    )
    figure.update_layout(
        xaxis_title="z-error = residual / posterior sigma",
        yaxis_title="Rows",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_segment_figure(frame: Any) -> Any | None:
    """Build a compact segment-level high-error bar plot."""

    segments = _candidate_free_error_segments(frame)
    if not _frame_available(segments) or not _plotly_available_local():
        return None
    import plotly.express as px

    plot_frame = segments.head(40).copy()
    plot_frame["segment"] = (
        plot_frame["entity_uid"].astype(str)
        + ":"
        + plot_frame["atom_family"].astype(str)
        + ":"
        + plot_frame["segment_start"].astype(str)
        + "-"
        + plot_frame["segment_end"].astype(str)
    )
    figure = px.bar(
        plot_frame,
        x="mean_abs_error",
        y="segment",
        color="atom_family" if "atom_family" in plot_frame.columns else None,
        orientation="h",
        hover_data=["length", "max_abs_error", "mean_sigma"],
        title="Top contiguous high-error residue segments",
    )
    figure.update_layout(
        xaxis_title="Mean absolute error (ppm)",
        yaxis_title="Segment",
        yaxis={"autorange": "reversed"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_free_pair_frame(
    frame: Any,
    *,
    target_family: str,
    source_family: str,
    source_residue_offset: int = 0,
) -> Any:
    """Return paired target/source family rows for joint-style diagnostics."""

    required = {
        "split",
        "entity_uid",
        "atom_family",
        "residue_index",
        "target_value",
        "predicted_value",
        "residual",
    }
    try:
        import pandas as pd
    except ImportError:
        return []
    if not _frame_available(frame) or not required.issubset(frame.columns):
        return pd.DataFrame()
    target = frame.loc[frame["atom_family"].astype(str) == target_family].copy()
    source = frame.loc[frame["atom_family"].astype(str) == source_family].copy()
    if target.empty or source.empty:
        return pd.DataFrame()
    target["join_residue_index"] = _to_numeric_series(target["residue_index"])
    source["join_residue_index"] = (
        _to_numeric_series(source["residue_index"]) - int(source_residue_offset)
    )
    target_cols = [
        "split",
        "entity_uid",
        "join_residue_index",
        "residue_index",
        "target_value",
        "predicted_value",
        "residual",
        "posterior_sigma",
    ]
    source_cols = [
        "split",
        "entity_uid",
        "join_residue_index",
        "residue_index",
        "target_value",
        "predicted_value",
        "residual",
        "posterior_sigma",
    ]
    paired = target[target_cols].merge(
        source[source_cols],
        on=["split", "entity_uid", "join_residue_index"],
        how="inner",
        suffixes=("_target", "_source"),
    )
    if paired.empty:
        return paired
    return paired.rename(
        columns={
            "residue_index_target": "target_residue_index",
            "target_value_target": "target_value",
            "predicted_value_target": "target_predicted_value",
            "residual_target": "target_residual",
            "posterior_sigma_target": "target_sigma",
            "residue_index_source": "source_residue_index",
            "target_value_source": "source_value",
            "predicted_value_source": "source_predicted_value",
            "residual_source": "source_residual",
            "posterior_sigma_source": "source_sigma",
        }
    )


def _candidate_free_pair_figure(pair_frame: Any, *, title: str) -> Any | None:
    """Build paired source/target residual scatter figure."""

    if not _frame_available(pair_frame) or not _plotly_available_local():
        return None
    import plotly.express as px

    figure = px.scatter(
        pair_frame,
        x="source_residual",
        y="target_residual",
        color="split" if "split" in pair_frame.columns else None,
        hover_data=[
            column
            for column in [
                "entity_uid",
                "target_residue_index",
                "source_residue_index",
                "target_value",
                "source_value",
            ]
            if column in pair_frame.columns
        ],
        title=title,
    )
    figure.add_shape(
        type="line",
        x0=float(pair_frame["source_residual"].min()),
        y0=0.0,
        x1=float(pair_frame["source_residual"].max()),
        y1=0.0,
        line={"dash": "dot", "color": "gray"},
    )
    figure.add_shape(
        type="line",
        x0=0.0,
        y0=float(pair_frame["target_residual"].min()),
        x1=0.0,
        y1=float(pair_frame["target_residual"].max()),
        line={"dash": "dot", "color": "gray"},
    )
    figure.update_layout(
        xaxis_title="Source-family residual (ppm)",
        yaxis_title="Target-family residual (ppm)",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _read_optional_parquet(path: Path) -> Any:
    """Read an optional Parquet file for monitor-only diagnostic slots."""

    try:
        import pandas as pd

        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception:
        try:
            import pandas as pd

            return pd.DataFrame()
        except ImportError:
            return []


def _to_numeric_series(series: Any) -> Any:
    """Convert a Series-like object to numeric values."""

    try:
        import pandas as pd

        return pd.to_numeric(series, errors="coerce")
    except ImportError:
        return series


def _safe_ccc_series(predicted: Any, target: Any) -> float | None:
    """Return Lin's concordance correlation coefficient."""

    pred = _to_numeric_series(predicted).dropna()
    targ = _to_numeric_series(target).dropna()
    common = pred.index.intersection(targ.index)
    if len(common) < 2:
        return None
    pred = pred.loc[common]
    targ = targ.loc[common]
    pred_mean = float(pred.mean())
    targ_mean = float(targ.mean())
    pred_var = float(((pred - pred_mean) ** 2).mean())
    targ_var = float(((targ - targ_mean) ** 2).mean())
    cov = float(((pred - pred_mean) * (targ - targ_mean)).mean())
    denom = pred_var + targ_var + (pred_mean - targ_mean) ** 2
    if denom <= 0:
        return None
    return float(2.0 * cov / denom)


def _safe_corr_series(left: Any, right: Any) -> float | None:
    """Return a finite Pearson correlation when possible."""

    left_series = _to_numeric_series(left).dropna()
    right_series = _to_numeric_series(right).dropna()
    common = left_series.index.intersection(right_series.index)
    if len(common) < 2:
        return None
    value = left_series.loc[common].corr(right_series.loc[common])
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_mean(values: Any) -> float | None:
    """Return finite mean for Series-like values."""

    try:
        value = float(_to_numeric_series(values).mean())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_std(values: Any) -> float | None:
    """Return finite population-like standard deviation."""

    try:
        value = float(_to_numeric_series(values).std(ddof=0))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_rmse(residual: Any) -> float | None:
    """Return finite RMSE for residual values."""

    try:
        value = float((_to_numeric_series(residual) ** 2).mean() ** 0.5)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _plotly_available_local() -> bool:
    """Return whether Plotly can be imported from the monitor runtime."""

    try:
        import plotly.express  # noqa: F401
    except ImportError:
        return False
    return True


def render_moment_v3_panels(st: Any, snapshot: dict[str, Any]) -> None:
    """Render measure-aware v3 calibration and joint-posterior diagnostics."""

    calibration_report = snapshot.get("entry_family_calibration_report") or {}
    ccc_report = snapshot.get("ccc_decomposition_report") or {}
    measure_frame = snapshot.get("measure_aware_targets")
    joint_frame = snapshot.get("joint_nmr_posteriors")
    card_cols = st.columns(4)
    val_calibration = (calibration_report.get("splits") or {}).get("val", {})
    with card_cols[0]:
        st.metric(
            "Raw CCC mean",
            _format_metric_value(val_calibration.get("raw_ccc_mean")),
            help="Entry/family raw MomentHead CCC averaged across calibration groups.",
        )
    with card_cols[1]:
        st.metric(
            "Calibrated CCC mean",
            _format_metric_value(val_calibration.get("calibrated_ccc_mean")),
            help="Entry/family affine-calibrated CCC for measure-aware diagnostics.",
        )
    with card_cols[2]:
        st.metric(
            "Mean reliability",
            _format_metric_value(val_calibration.get("mean_reliability")),
            help="Average NMR measure-aware row reliability.",
        )
    with card_cols[3]:
        joint_rows = 0 if joint_frame is None or joint_frame.empty else len(joint_frame)
        st.metric("Joint posterior rows", joint_rows)

    with st.expander("CCC decomposition report", expanded=False):
        if ccc_report:
            st.json(ccc_report, expanded=False)
        else:
            st.info("Waiting for ccc_decomposition_report.json.")
    with st.expander("Entry/family calibration report", expanded=False):
        if calibration_report:
            st.json(calibration_report, expanded=False)
        else:
            st.info("Waiting for entry_family_calibration_report.json.")
    if measure_frame is not None and not measure_frame.empty:
        columns = [
            column
            for column in [
                "split",
                "entity_uid",
                "atom_family",
                "target_id",
                "target_value",
                "moment_predicted_value",
                "calibrated_moment_predicted_value",
                "entry_family_scale",
                "entry_family_offset",
                "entry_family_reliability",
                "calibrated_residual",
            ]
            if column in measure_frame.columns
        ]
        with st.expander("Measure-aware target rows", expanded=False):
            st.dataframe(
                measure_frame[columns].head(800),
                width="stretch",
                hide_index=True,
            )
    if joint_frame is not None and not joint_frame.empty:
        with st.expander("Joint NMR posterior rows", expanded=False):
            st.dataframe(joint_frame.head(800), width="stretch", hide_index=True)


def _format_metric_value(value: Any) -> str:
    """Format optional monitor metric values compactly."""
    if value is None:
        return "waiting"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def render_moment_v2_cards(
    st: Any,
    moment_frame: Any,
    state_frame: Any,
) -> None:
    """Render v2-specific HN/C' and state-diversity cards."""

    card_cols = st.columns(4)
    if moment_frame is not None and not moment_frame.empty:
        frame = moment_frame.copy()
        if "split" in frame.columns and (frame["split"].astype(str) == "val").any():
            frame = frame.loc[frame["split"].astype(str) == "val"].copy()
        hn_frame = (
            frame.loc[frame["atom_family"].astype(str) == "HN"]
            if "atom_family" in frame.columns
            else frame.iloc[0:0]
        )
        cprime_frame = (
            frame.loc[frame["atom_family"].astype(str) == "C'"]
            if "atom_family" in frame.columns
            else frame.iloc[0:0]
        )
        with card_cols[0]:
            st.metric(
                "HN rows",
                int(len(hn_frame)),
                help="Rows used for HN variance-calibration diagnostics.",
            )
        with card_cols[1]:
            sigma = (
                hn_frame["moment_sigma"].mean()
                if not hn_frame.empty and "moment_sigma" in hn_frame.columns
                else None
            )
            st.metric(
                "HN mean sigma",
                "n/a" if sigma is None else f"{float(sigma):.3g}",
                help="MomentHead posterior sigma for HN rows.",
            )
        with card_cols[2]:
            outlier = (
                cprime_frame["outlier_score"].mean()
                if not cprime_frame.empty and "outlier_score" in cprime_frame.columns
                else None
            )
            st.metric(
                "C' outlier score",
                "n/a" if outlier is None else f"{float(outlier):.3g}",
                help="Mean robust outlier score for C' rows.",
            )
    else:
        with card_cols[0]:
            st.metric("HN rows", "waiting")
        with card_cols[1]:
            st.metric("HN mean sigma", "waiting")
        with card_cols[2]:
            st.metric("C' outlier score", "waiting")

    with card_cols[3]:
        if state_frame is not None and not state_frame.empty:
            frame = state_frame.copy()
            if "split" in frame.columns and (frame["split"].astype(str) == "val").any():
                frame = frame.loc[frame["split"].astype(str) == "val"].copy()
            effective = (
                frame["effective_state_count"].mean()
                if "effective_state_count" in frame.columns
                else None
            )
            st.metric(
                "Effective states",
                "n/a" if effective is None else f"{float(effective):.2f}",
                help="Inverse-simpson effective posterior state count.",
            )
        else:
            st.metric("Effective states", "waiting")

    if moment_frame is not None and not moment_frame.empty:
        frame = moment_frame.copy()
        if "atom_family" in frame.columns:
            focus = frame.loc[frame["atom_family"].astype(str).isin(["HN", "C'"])]
            if not focus.empty:
                columns = [
                    column
                    for column in [
                        "split",
                        "atom_family",
                        "target_id",
                        "target_value",
                        "moment_predicted_value",
                        "moment_sigma",
                        "atom_family_sigma_scale",
                        "outlier_score",
                    ]
                    if column in focus.columns
                ]
                with st.expander("HN / C' bottleneck rows", expanded=False):
                    st.dataframe(
                        focus[columns].head(600),
                        width="stretch",
                        hide_index=True,
                    )


def render_conformation_tab(
    st: Any,
    components: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render structure uncertainty panels."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        ENSEMBLE_LOCAL_CONFIDENCE_LABEL,
        "This is ensemble internal local structural consistency on a 0-100 "
        "scale, not an accuracy estimate.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_conformation_uncertainty.png",
        "Panel-level conformation uncertainty analysis",
    )
    round_audit = snapshot.get("bioemu_round_conformer_landscape_audit")
    round_audit_path = paths.figures_dir / "bioemu_round_conformer_landscape_audit.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(round_audit) or round_audit_path.exists()
    ):
        render_image_or_wait(
            st,
            round_audit_path,
            "Round conformer ensemble and energy-landscape readiness",
        )
        if _frame_available(round_audit) and "check_name" in round_audit.columns:
            verdict = round_audit.loc[
                round_audit["check_name"].astype(str).eq(
                    "actual_conformer_learning_verdict"
                )
            ]
            if not verdict.empty and str(verdict.iloc[0].get("status", "")) != "pass":
                st.warning(str(verdict.iloc[0].get("interpretation", "")))
    probe_manifest = snapshot.get("bioemu_detached_conformer_probe_manifest")
    probe_manifest_path = paths.figures_dir / "bioemu_detached_conformer_probe_manifest.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_manifest) or probe_manifest_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_manifest_path,
            "Next-round detached conformer probe queue",
        )
        if _frame_available(probe_manifest):
            st.dataframe(probe_manifest.head(12), width="stretch", hide_index=True)
    probe_inputs = snapshot.get("bioemu_detached_conformer_probe_inputs")
    probe_inputs_path = paths.figures_dir / "bioemu_detached_conformer_probe_inputs.png"
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_inputs) or probe_inputs_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_inputs_path,
            "Prepared detached conformer probe FASTA inputs",
        )
        if _frame_available(probe_inputs):
            display_columns = [
                column
                for column in (
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "sequence_length",
                    "sample_count",
                    "ready",
                    "fasta_path",
                    "output_dir",
                )
                if column in probe_inputs.columns
            ]
            st.dataframe(
                probe_inputs[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    probe_output_audit = snapshot.get("bioemu_detached_conformer_probe_output_audit")
    probe_output_path = (
        paths.figures_dir / "bioemu_detached_conformer_probe_output_audit.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(probe_output_audit) or probe_output_path.exists()
    ):
        render_image_or_wait(
            st,
            probe_output_path,
            "Detached BioEmu conformer probe generated outputs",
        )
        if _frame_available(probe_output_audit):
            display_columns = [
                column
                for column in (
                    "probe_name",
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "requested_samples",
                    "actual_samples",
                    "residue_count",
                    "status",
                    "context_mode",
                    "topology_path",
                    "xtc_path",
                )
                if column in probe_output_audit.columns
            ]
            st.dataframe(
                probe_output_audit[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    ensemble_quality = snapshot.get("bioemu_detached_conformer_ensemble_quality")
    ensemble_quality_path = (
        paths.figures_dir / "bioemu_detached_conformer_ensemble_quality.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(ensemble_quality) or ensemble_quality_path.exists()
    ):
        render_image_or_wait(
            st,
            ensemble_quality_path,
            "Detached BioEmu conformer diversity and PCA landscape",
        )
        if _frame_available(ensemble_quality):
            display_columns = [
                column
                for column in (
                    "probe_name",
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "actual_samples",
                    "residue_count",
                    "quality_flag",
                    "quality_outlier_score",
                    "family_quality_outlier_count",
                    "family_quality_outlier_fraction",
                    "family_repeated_quality_outlier",
                    "mean_pairwise_rmsd",
                    "max_pairwise_rmsd",
                    "rg_mean",
                    "rg_std",
                    "contact_density_mean",
                    "pca_spread",
                )
                if column in ensemble_quality.columns
            ]
            st.dataframe(
                ensemble_quality[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    family_quality_guard = snapshot.get(
        "bioemu_detached_conformer_family_quality_guard"
    )
    family_quality_guard_path = (
        paths.figures_dir / "bioemu_detached_conformer_family_quality_guard.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(family_quality_guard) or family_quality_guard_path.exists()
    ):
        render_image_or_wait(
            st,
            family_quality_guard_path,
            "Family-level detached conformer geometry-spread guard",
        )
        if _frame_available(family_quality_guard):
            display_columns = [
                column
                for column in (
                    "top_atom_family",
                    "ready_entities",
                    "quality_outlier_entities",
                    "quality_outlier_fraction",
                    "max_quality_outlier_score",
                    "outlier_entities",
                    "guard_priority_score",
                    "recommended_guard_action",
                )
                if column in family_quality_guard.columns
            ]
            st.dataframe(
                family_quality_guard[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    support_action = snapshot.get("bioemu_detached_support_expansion_action_plan")
    support_action_path = (
        paths.figures_dir / "bioemu_detached_support_expansion_action_plan.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(support_action) or support_action_path.exists()
    ):
        render_image_or_wait(
            st,
            support_action_path,
            "Detached probe support-expansion action plan",
        )
        if _frame_available(support_action):
            display_columns = [
                column
                for column in (
                    "entity_uid",
                    "top_atom_family",
                    "quality_flag",
                    "support_guard_level",
                    "combined_priority_score",
                    "priority_score",
                    "low_ess_tail_rows",
                    "rg_std",
                    "pca_spread",
                    "recommended_next_action",
                )
                if column in support_action.columns
            ]
            st.dataframe(
                support_action[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    nmr_coupling = snapshot.get("bioemu_detached_conformer_nmr_coupling")
    nmr_coupling_path = (
        paths.figures_dir / "bioemu_detached_conformer_nmr_coupling.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(nmr_coupling) or nmr_coupling_path.exists()
    ):
        render_image_or_wait(
            st,
            nmr_coupling_path,
            "NMR bottleneck and conformer-ensemble coupling",
        )
        if _frame_available(nmr_coupling):
            display_columns = [
                column
                for column in (
                    "rank",
                    "entity_uid",
                    "top_atom_family",
                    "priority_score",
                    "hn_cprime_mae",
                    "all_family_mae",
                    "mean_pairwise_rmsd",
                    "rg_std",
                    "contact_density_mean",
                    "nmr_structure_coupling_score",
                    "coupling_interpretation",
                )
                if column in nmr_coupling.columns
            ]
            st.dataframe(
                nmr_coupling[display_columns].head(12),
                width="stretch",
                hide_index=True,
            )
    residue_coupling = snapshot.get(
        "bioemu_detached_conformer_residue_shift_coupling"
    )
    residue_coupling_path = (
        paths.figures_dir / "bioemu_detached_conformer_residue_shift_coupling.png"
    )
    if _is_bioemu_latent_snapshot(snapshot) and (
        _frame_available(residue_coupling) or residue_coupling_path.exists()
    ):
        render_image_or_wait(
            st,
            residue_coupling_path,
            "Residue-level NMR residual and conformer-mobility coupling",
        )
        if _frame_available(residue_coupling):
            display_columns = [
                column
                for column in (
                    "probe_rank",
                    "entity_uid",
                    "residue_index",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "predicted_value",
                    "guarded_predicted_value",
                    "active_abs_error",
                    "posterior_sigma",
                    "residue_rmsf",
                    "local_contact_variability",
                    "residue_shift_mobility_score",
                    "selected_strategy",
                    "guard_action",
                )
                if column in residue_coupling.columns
            ]
            st.dataframe(
                residue_coupling[display_columns].head(20),
                width="stretch",
                hide_index=True,
            )
    if _is_bioemu_latent_snapshot(snapshot) and not _has_conformation_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="Actual conformer ensemble artifacts are not ready for this round.",
            text=(
                "The current BioEmu latent round does not include decoded "
                "conformer state clusters, medoid structures, or local confidence "
                "profiles. The latent atlas is useful mechanistically, but it is "
                "not enough to claim that the actual solution conformer ensemble "
                "has been learned."
            ),
            expected=[
                "reports/arrays/ensemble_states.parquet",
                "reports/arrays/conformer_state_assignments.parquet",
                "reports/figures/ensemble_local_confidence_structure.png",
            ],
        )
        return
    if _is_candidate_free_snapshot(snapshot) and not _has_conformation_artifacts(
        snapshot
    ):
        render_task_scope_notice(
            st,
            snapshot,
            title="Structure uncertainty artifacts are not produced for this run.",
            text=(
                "The current candidate-free run predicts NMR posterior moments "
                "without representative structures. Conformation uncertainty "
                "requires a later structure/landscape render pass."
                " Candidate-free residue uncertainty proxies are consolidated "
                "in the NMR Chemical Shifts tab."
            ),
            expected=[
                "reports/structures/annotated_representative.pdb",
                "reports/figures/ensemble_local_confidence_structure.png",
                "reports/arrays/ensemble_states.parquet",
            ],
        )
        return
    st.dataframe(confidence_legend_rows(), width="stretch", hide_index=True)

    metrics = _nested_metrics(
        snapshot.get("uncertainty_report"),
        ["summary", "metrics"],
        fallback_key="summary",
    )
    render_metric_dict(st, metrics)

    cols = st.columns(2)
    render_image_or_wait(
        cols[0],
        paths.figures_dir / "ensemble_local_confidence_structure.png",
        ENSEMBLE_LOCAL_CONFIDENCE_LABEL,
    )
    render_image_or_wait(
        cols[1],
        paths.figures_dir / "distance_uncertainty_heatmap.png",
        "Distance uncertainty matrix",
    )

    st.write("Interactive representative structure")
    viewer_html = build_structure_viewer_html(paths.annotated_representative_pdb)
    if viewer_html is None:
        st.info("Waiting for annotated_representative.pdb or py3Dmol.")
    else:
        components.html(viewer_html, height=560)

    st.write("Ensemble State Gallery")
    st.caption(
        "Representative cluster conformers are selected as weighted medoids on "
        "the BioEmu-style energy landscape. This shows multiple structural "
        "states rather than treating one conformer as the whole ensemble."
    )
    state_frame = snapshot.get("ensemble_states")
    assignment_frame = snapshot.get("conformer_state_assignments")
    representative_entity = (
        snapshot.get("uncertainty_report", {}).get("representative_entity_uid")
        if isinstance(snapshot.get("uncertainty_report"), dict)
        else None
    )
    if state_frame is not None and not state_frame.empty and representative_entity:
        state_frame = state_frame.loc[
            state_frame["entity_uid"].astype(str) == str(representative_entity)
        ].copy()
    if (
        assignment_frame is not None
        and not assignment_frame.empty
        and representative_entity
        and "entity_uid" in assignment_frame.columns
    ):
        assignment_frame = assignment_frame.loc[
            assignment_frame["entity_uid"].astype(str) == str(representative_entity)
        ].copy()

    state_static_cols = st.columns(3)
    render_image_or_wait(
        state_static_cols[0],
        paths.figures_dir / "landscape_state_clusters.png",
        "Landscape state clusters",
    )
    render_image_or_wait(
        state_static_cols[1],
        paths.figures_dir / "state_mass_free_energy.png",
        "State mass and free energy",
    )
    render_image_or_wait(
        state_static_cols[2],
        paths.figures_dir / "state_nmr_evidence_heatmap.png",
        "State NMR evidence heatmap",
    )

    if state_frame is None or state_frame.empty:
        st.info("Waiting for ensemble_states.parquet.")
    else:
        state_metric_cols = st.columns(2)
        state_mass_plot = ensemble_state_mass_figure(state_frame)
        if state_mass_plot is not None:
            state_metric_cols[0].plotly_chart(
                state_mass_plot,
                width="stretch",
            )
        state_metric_cols[1].dataframe(
            state_frame.head(12),
            width="stretch",
            hide_index=True,
        )

    state_scatter = (
        None
        if assignment_frame is None
        else ensemble_state_scatter_figure(assignment_frame)
    )
    if state_scatter is not None:
        st.plotly_chart(state_scatter, width="stretch")
    elif assignment_frame is None or assignment_frame.empty:
        st.info("Waiting for conformer_state_assignments.parquet.")

    state_ids = [] if state_frame is None else available_ensemble_state_ids(state_frame)
    if state_ids:
        selected_state = st.selectbox(
            "Representative state conformer",
            options=state_ids,
            index=0,
            help="Each state conformer is a weighted medoid from its landscape cluster.",
        )
        state_viewer = build_structure_viewer_html(
            paths.structures_dir / f"{selected_state}_representative.pdb"
        )
        if state_viewer is None:
            st.info(f"Waiting for {selected_state}_representative.pdb or py3Dmol.")
        else:
            components.html(state_viewer, height=560)

    st.write("RCI Adapter Calibration")
    st.caption(
        "RCI panels compare NMR-shift-derived flexibility against AtypEmu "
        "ensemble structural uncertainty. pRCI is shown as an RCI adapter, "
        "not as an AlphaFold-style confidence score."
    )
    st.dataframe(rci_legend_rows(), width="stretch", hide_index=True)
    rci_calibration = (
        snapshot.get("uncertainty_report", {}).get("rci_calibration")
        if isinstance(snapshot.get("uncertainty_report"), dict)
        else None
    )
    if isinstance(rci_calibration, dict):
        status = str(rci_calibration.get("status", "waiting"))
        if status == "missing_rci_profile":
            st.info("Waiting for RCI adapter profile.")
        elif status == "no_overlap":
            st.warning(
                "RCI adapter profile is available, but no residue overlap was found."
            )
        else:
            render_metric_dict(st, rci_calibration)
    else:
        st.info("Waiting for RCI adapter calibration summary.")

    rci_cols = st.columns(2)
    render_image_or_wait(
        rci_cols[0],
        paths.figures_dir / "rci_vs_ensemble_rmsf.png",
        "RCI adapter vs ensemble RMSF",
    )
    render_image_or_wait(
        rci_cols[1],
        paths.figures_dir / "rci_local_confidence_profile.png",
        "RCI adapter and local confidence profile",
    )
    rci_frame = snapshot.get("rci_vs_ensemble_profile")
    if rci_frame is not None and not rci_frame.empty:
        with st.expander("RCI adapter joined residue profile", expanded=False):
            st.dataframe(rci_frame.head(400), width="stretch", hide_index=True)


def render_bioemu_latent_atlas_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render BioEmu latent atlas artifacts when the new task produces them."""

    paths = snapshot["paths"]
    predictions = snapshot.get("bioemu_latent_nmr_predictions")
    masked_predictions = snapshot.get("bioemu_latent_nmr_masked_predictions")
    sample_weights = snapshot.get("bioemu_latent_sample_weights")
    atlas = snapshot.get("bioemu_latent_atlas_coordinates")
    physics_atlas = snapshot.get("bioemu_physics_atlas_coordinates")
    family_tangent = snapshot.get("bioemu_family_tangent_adapter_audit")
    posterior_moments = snapshot.get("bioemu_posterior_moments")
    bridge_audit = snapshot.get("bioemu_posterior_bridge_audit")
    family_conflict = snapshot.get("bioemu_family_conflict_audit")
    hn_cprime_direction = snapshot.get("bioemu_hn_cprime_direction_conflict_audit")
    ring_physics = snapshot.get("bioemu_ring_physics_audit")
    carbonyl_backbone = snapshot.get("bioemu_carbonyl_backbone_audit")
    rare_regime_atlas = snapshot.get("bioemu_rare_regime_atlas")
    rare_regime_summary = snapshot.get("bioemu_rare_regime_summary")
    rare_regime_acceptance = snapshot.get("bioemu_rare_regime_acceptance")
    report = snapshot.get("bioemu_latent_nmr_report") or {}
    render_panel_header(
        st,
        "BioEmu Latent NMR Atlas",
        "Sequence + observed NMR evidence + BioEmu latent samples are monitored "
        "as posterior weights, atlas coordinates, and HN rare-regime subcharts.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_bioemu_atlas.png",
        "Panel-level BioEmu atlas bottleneck analysis",
    )
    if (
        (predictions is None or predictions.empty)
        and (sample_weights is None or sample_weights.empty)
        and (atlas is None or atlas.empty)
        and (physics_atlas is None or physics_atlas.empty)
        and (family_tangent is None or family_tangent.empty)
        and (posterior_moments is None or posterior_moments.empty)
        and (bridge_audit is None or bridge_audit.empty)
        and (hn_cprime_direction is None or hn_cprime_direction.empty)
        and (ring_physics is None or ring_physics.empty)
        and (carbonyl_backbone is None or carbonyl_backbone.empty)
        and not report
    ):
        st.info(
            "Waiting for BioEmu latent atlas artifacts in reports/arrays and "
            "reports/figures."
        )
        return

    cards = st.columns(4)
    prediction_count = 0 if predictions is None or predictions.empty else len(predictions)
    masked_count = (
        0
        if predictions is None
        or predictions.empty
        or "is_masked_holdout" not in predictions
        else int(predictions["is_masked_holdout"].astype(bool).sum())
    )
    mean_ess = (
        float(predictions["posterior_ess"].mean())
        if predictions is not None
        and not predictions.empty
        and "posterior_ess" in predictions
        else float("nan")
    )
    mean_sigma = (
        float(predictions["posterior_sigma"].mean())
        if predictions is not None
        and not predictions.empty
        and "posterior_sigma" in predictions
        else float("nan")
    )
    cards[0].metric("Prediction rows", _display(prediction_count))
    cards[1].metric("Masked rows", _display(masked_count))
    cards[2].metric("Mean ESS", _display(mean_ess))
    cards[3].metric("Mean sigma", _display(mean_sigma))

    if bridge_audit is not None and not bridge_audit.empty:
        bridge_cols = st.columns(3)
        if "chart_entropy" in bridge_audit:
            bridge_cols[0].metric(
                "Chart entropy",
                _display(float(bridge_audit["chart_entropy"].mean())),
            )
        if "mean_covariance_trace" in bridge_audit:
            bridge_cols[1].metric(
                "Mean covariance",
                _display(float(bridge_audit["mean_covariance_trace"].mean())),
            )
        if "bridge_norm" in bridge_audit:
            bridge_cols[2].metric(
                "Bridge norm",
                _display(float(bridge_audit["bridge_norm"].mean())),
            )

    if predictions is not None and not predictions.empty:
        _bioemu_show_family_metric_block(
            st,
            snapshot,
            key_prefix="bioemu_atlas",
        )

    plot_cols = st.columns(3)
    render_image_or_wait(
        plot_cols[0],
        paths.figures_dir / "bioemu_latent_hn_masked_scatter.png",
        "BioEmu masked HN scatter",
    )
    render_image_or_wait(
        plot_cols[1],
        paths.figures_dir / "bioemu_latent_cprime_masked_scatter.png",
        "BioEmu masked C' scatter",
    )
    render_image_or_wait(
        plot_cols[2],
        paths.figures_dir / "bioemu_latent_hn_cprime_target_progress.png",
        "BioEmu HN/C' progress",
    )

    family_cols = st.columns(4)
    render_image_or_wait(
        family_cols[0],
        paths.figures_dir / "bioemu_latent_all_family_progress.png",
        "All-family masked CCC progress",
    )
    render_image_or_wait(
        family_cols[1],
        paths.figures_dir / "bioemu_latent_n_masked_scatter.png",
        "BioEmu masked N scatter",
    )
    render_image_or_wait(
        family_cols[2],
        paths.figures_dir / "bioemu_latent_ca_masked_scatter.png",
        "BioEmu masked CA scatter",
    )
    render_image_or_wait(
        family_cols[3],
        paths.figures_dir / "bioemu_latent_cb_masked_scatter.png",
        "BioEmu masked CB scatter",
    )

    audit_cols = st.columns(5)
    render_image_or_wait(
        audit_cols[0],
        paths.figures_dir / "hn_subfeature_bundle_audit.png",
        "HN subfeature bundle audit",
    )
    render_image_or_wait(
        audit_cols[1],
        paths.figures_dir / "bioemu_observation_chain_audit.png",
        "Observation-chain ESS audit",
    )
    render_image_or_wait(
        audit_cols[2],
        paths.figures_dir / "bioemu_component_persistence_audit.png",
        "Latent component persistence",
    )
    render_image_or_wait(
        audit_cols[3],
        paths.figures_dir / "bioemu_family_conflict_audit.png",
        "Family direction-conflict audit",
    )
    render_image_or_wait(
        audit_cols[4],
        paths.figures_dir / "bioemu_sample_weight_entropy.png",
        "Sample weight entropy",
    )

    physics_cols = st.columns(3)
    render_image_or_wait(
        physics_cols[0],
        paths.figures_dir / "bioemu_hn_cprime_direction_conflict_audit.png",
        "v36 HN/C' physics direction conflict",
    )
    render_image_or_wait(
        physics_cols[1],
        paths.figures_dir / "bioemu_ring_physics_audit.png",
        "v36 HN ring physics bins",
    )
    render_image_or_wait(
        physics_cols[2],
        paths.figures_dir / "bioemu_carbonyl_backbone_audit.png",
        "v36 C' carbonyl/backbone bins",
    )

    signed_cols = st.columns(3)
    render_image_or_wait(
        signed_cols[0],
        paths.figures_dir / "bioemu_mechanism_signed_direction_audit.png",
        "v101 mechanism-signed rare-regime priority",
    )
    render_image_or_wait(
        signed_cols[1],
        paths.figures_dir / "bioemu_hn_ring_signed_subregime_audit.png",
        "v101 HN ring signed subregimes",
    )
    render_image_or_wait(
        signed_cols[2],
        paths.figures_dir / "bioemu_cprime_carbonyl_subregime_audit.png",
        "v101 C' carbonyl signed subregimes",
    )

    _bioemu_ucbshift_factor_atlas_block(st, snapshot)

    st.subheader("Rare-Regime Physics Atlas")
    rare_cols = st.columns(3)
    render_image_or_wait(
        rare_cols[0],
        paths.figures_dir / "bioemu_rare_regime_conflict_heatmap.png",
        "Regime conflict heatmap",
    )
    render_image_or_wait(
        rare_cols[1],
        paths.figures_dir / "bioemu_rare_regime_router_confidence.png",
        "Chart router confidence",
    )
    render_image_or_wait(
        rare_cols[2],
        paths.figures_dir / "bioemu_rare_regime_tangent_alignment.png",
        "Tangent-vector alignment",
    )
    rare_cols = st.columns(3)
    render_image_or_wait(
        rare_cols[0],
        paths.figures_dir / "bioemu_rare_regime_residue_family_expert_gain.png",
        "Residue-family expert gain",
    )
    render_image_or_wait(
        rare_cols[1],
        paths.figures_dir / "bioemu_rare_regime_ensemble_occupancy_shift.png",
        "Ensemble occupancy shift",
    )
    render_image_or_wait(
        rare_cols[2],
        paths.figures_dir / "bioemu_rare_regime_posterior_calibration.png",
        "Posterior ESS/entropy calibration",
    )
    if rare_regime_acceptance is not None and not rare_regime_acceptance.empty:
        st.write("Rare-regime operational acceptance checks")
        st.dataframe(rare_regime_acceptance, width="stretch", hide_index=True)
    if rare_regime_summary is not None and not rare_regime_summary.empty:
        st.write("Rare-regime chart summary")
        st.dataframe(rare_regime_summary.head(200), width="stretch", hide_index=True)
    if rare_regime_atlas is not None and not rare_regime_atlas.empty:
        with st.expander("Rare-regime atlas rows", expanded=False):
            st.dataframe(rare_regime_atlas.head(500), width="stretch", hide_index=True)

    moment_cols = st.columns(2)
    render_image_or_wait(
        moment_cols[0],
        paths.figures_dir / "bioemu_posterior_moment_calibration.png",
        "Posterior moment calibration",
    )
    render_image_or_wait(
        moment_cols[1],
        paths.figures_dir / "bioemu_posterior_bridge_vector_field.png",
        "Posterior bridge vector field",
    )

    _bioemu_family_calibration_strategy_block(st, snapshot)
    _bioemu_family_affine_calibration_block(st, snapshot)
    _bioemu_family_chart_calibration_block(st, snapshot)
    _bioemu_cprime_chart_calibration_block(st, snapshot)

    if predictions is not None and not predictions.empty:
        ccc_source = (
            masked_predictions
            if masked_predictions is not None and not masked_predictions.empty
            else predictions
        )
        ccc_table = _bioemu_masked_ccc_table(ccc_source)
        if ccc_table:
            st.write("Masked posterior mean CCC by atom family")
            st.dataframe(ccc_table, width="stretch", hide_index=True)
        if masked_predictions is not None and not masked_predictions.empty:
            st.write("BioEmu latent masked NMR predictions")
            st.dataframe(masked_predictions.head(500), width="stretch", hide_index=True)
        st.write("BioEmu latent NMR predictions")
        st.dataframe(predictions.head(500), width="stretch", hide_index=True)

    if atlas is not None and not atlas.empty:
        st.write("Atlas coordinate summary")
        coordinate_summary = (
            atlas.groupby("coordinate_name", dropna=False)
            .agg(
                row_count=("coordinate_value", "size"),
                mean_abs_coordinate=("coordinate_value", lambda value: value.abs().mean()),
                mean_reliability=("reliability", "mean"),
                mean_uncertainty=("uncertainty", "mean"),
                mean_chart_probability=("chart_probability", "mean"),
            )
            .reset_index()
        )
        st.dataframe(coordinate_summary, width="stretch", hide_index=True)
        if {"atom_family", "coordinate_name"}.issubset(atlas.columns):
            st.write("Atlas coordinate summary by atom family")
            family_coordinate_summary = (
                atlas.groupby(["atom_family", "coordinate_name"], dropna=False)
                .agg(
                    row_count=("coordinate_value", "size"),
                    mean_abs_coordinate=(
                        "coordinate_value",
                        lambda value: value.abs().mean(),
                    ),
                    mean_reliability=("reliability", "mean"),
                    mean_uncertainty=("uncertainty", "mean"),
                    mean_chart_probability=("chart_probability", "mean"),
                )
                .reset_index()
                .sort_values(["atom_family", "mean_uncertainty"], ascending=[True, False])
            )
            st.dataframe(
                family_coordinate_summary,
                width="stretch",
                hide_index=True,
            )
        if "uncertainty" in atlas:
            high_uncertainty_columns = [
                column
                for column in [
                    "split",
                    "entity_uid",
                    "target_id",
                    "atom_family",
                    "coordinate_name",
                    "coordinate_value",
                    "reliability",
                    "uncertainty",
                    "chart_probability",
                    "is_masked_holdout",
                ]
                if column in atlas.columns
            ]
            with st.expander("Highest atlas uncertainty rows", expanded=False):
                st.dataframe(
                    atlas.sort_values("uncertainty", ascending=False)[
                        high_uncertainty_columns
                    ].head(120),
                    width="stretch",
                    hide_index=True,
                )

    if physics_atlas is not None and not physics_atlas.empty:
        st.write("v36 physics-atlas coordinate summary")
        coordinate_summary = (
            physics_atlas.groupby("coordinate_name", dropna=False)
            .agg(
                row_count=("coordinate_value", "size"),
                mean_abs_coordinate=("coordinate_value", lambda value: value.abs().mean()),
                mean_reliability=("reliability", "mean"),
                mean_uncertainty=("uncertainty", "mean"),
                sidecar_target_rows=("has_sidecar_target", "sum")
                if "has_sidecar_target" in physics_atlas.columns
                else ("coordinate_value", "size"),
            )
            .reset_index()
            .sort_values("mean_uncertainty", ascending=False, kind="stable")
        )
        st.dataframe(coordinate_summary, width="stretch", hide_index=True)
        if {"atom_family", "coordinate_name"}.issubset(physics_atlas.columns):
            family_coordinate_summary = (
                physics_atlas.groupby(["atom_family", "coordinate_name"], dropna=False)
                .agg(
                    row_count=("coordinate_value", "size"),
                    mean_abs_coordinate=("coordinate_value", lambda value: value.abs().mean()),
                    mean_reliability=("reliability", "mean"),
                    mean_uncertainty=("uncertainty", "mean"),
                )
                .reset_index()
                .sort_values(
                    ["atom_family", "mean_uncertainty"],
                    ascending=[True, False],
                    kind="stable",
                )
            )
            with st.expander("v36 physics atlas by atom family", expanded=False):
                st.dataframe(family_coordinate_summary, width="stretch", hide_index=True)

    if family_tangent is not None and not family_tangent.empty:
        st.write("v36 family tangent adapter audit")
        tangent_columns = [
            column
            for column in [
                "split",
                "entity_uid",
                "target_id",
                "atom_family",
                "family_tangent_delta",
                "family_chart_bias_delta",
                "family_adapter_delta",
                "is_masked_holdout",
            ]
            if column in family_tangent.columns
        ]
        st.dataframe(
            family_tangent[tangent_columns].head(240),
            width="stretch",
            hide_index=True,
        )

    if family_conflict is not None and not family_conflict.empty:
        st.write("Atlas family direction-conflict audit")
        conflict_columns = [
            column
            for column in [
                "split",
                "coordinate_name",
                "coordinate_bin",
                "row_count",
                "target_family_count",
                "support_family_count",
                "hn_cprime_ccc",
                "n_ca_ccc",
                "ccc_gap_hn_cprime_minus_n_ca",
                "hn_cprime_residual_z_mean",
                "n_ca_residual_z_mean",
                "combined_conflict_score",
                "mean_reliability",
                "mean_uncertainty",
            ]
            if column in family_conflict.columns
        ]
        st.dataframe(
            family_conflict.sort_values(
                "combined_conflict_score",
                ascending=False,
                kind="stable",
            )[conflict_columns].head(120),
            width="stretch",
            hide_index=True,
        )

    for title, frame in [
        ("v36 HN/C' physics direction conflict audit", hn_cprime_direction),
        ("v36 HN ring physics audit", ring_physics),
        ("v36 C' carbonyl/backbone audit", carbonyl_backbone),
    ]:
        if frame is not None and not frame.empty:
            with st.expander(title, expanded=False):
                st.dataframe(frame.head(180), width="stretch", hide_index=True)

    if sample_weights is not None and not sample_weights.empty:
        st.write("Sample posterior weight summary")
        weight_summary = (
            sample_weights.groupby("entity_uid", dropna=False)
            .agg(
                sample_count=("posterior_weight", "size"),
                max_weight=("posterior_weight", "max"),
                weight_entropy=(
                    "posterior_weight",
                    lambda value: float(
                        -sum(
                            float(weight) * math.log(max(float(weight), 1e-12))
                            for weight in value
                        )
                    ),
                ),
            )
            .reset_index()
        )
        st.dataframe(weight_summary.head(300), width="stretch", hide_index=True)

    if posterior_moments is not None and not posterior_moments.empty:
        st.write("Posterior moment chart summary")
        chart_figure = _bioemu_chart_probability_figure(posterior_moments)
        if chart_figure is not None:
            st.plotly_chart(
                chart_figure,
                width="stretch",
                key="bioemu_atlas_chart_probability",
            )
        moment_summary = (
            posterior_moments.groupby("chart_index", dropna=False)
            .agg(
                row_count=("chart_probability", "size"),
                mean_chart_probability=("chart_probability", "mean"),
                mean_effective_probability=("effective_chart_probability", "mean"),
                mean_reliability=("chart_reliability", "mean"),
                mean_covariance_trace=("covariance_trace", "mean"),
            )
            .reset_index()
        )
        st.dataframe(moment_summary, width="stretch", hide_index=True)
        if {"atom_family", "chart_index"}.issubset(posterior_moments.columns):
            st.write("Posterior moment chart summary by atom family")
            moment_family_summary = (
                posterior_moments.groupby(["atom_family", "chart_index"], dropna=False)
                .agg(
                    row_count=("chart_probability", "size"),
                    mean_chart_probability=("chart_probability", "mean"),
                    mean_effective_probability=(
                        "effective_chart_probability",
                        "mean",
                    ),
                    mean_reliability=("chart_reliability", "mean"),
                    mean_covariance_trace=("covariance_trace", "mean"),
                )
                .reset_index()
            )
            st.dataframe(moment_family_summary, width="stretch", hide_index=True)

    if bridge_audit is not None and not bridge_audit.empty:
        bridge_figure = _bioemu_bridge_family_figure(bridge_audit)
        if bridge_figure is not None:
            st.plotly_chart(
                bridge_figure,
                width="stretch",
                key="bioemu_atlas_bridge_family",
            )
        st.write("Posterior bridge audit")
        st.dataframe(bridge_audit.head(500), width="stretch", hide_index=True)

    if isinstance(report, dict) and report:
        with st.expander("BioEmu latent NMR report", expanded=False):
            st.json(report)


def _bioemu_masked_ccc_table(predictions: Any) -> list[dict[str, Any]]:
    """Return masked CCC rows for BioEmu latent atlas predictions."""

    return _bioemu_family_metric_frame(predictions).to_dict("records")


def _bioemu_family_metric_frame(
    predictions: Any,
    *,
    masked_only: bool = True,
) -> pd.DataFrame:
    """Compute family-wide BioEmu posterior mean diagnostics."""

    required = {"atom_family", "target_value", "predicted_value"}
    if (
        predictions is None
        or predictions.empty
        or not required.issubset(predictions.columns)
    ):
        return pd.DataFrame(
            columns=[
                "atom_family",
                "masked_rows",
                "ccc",
                "mae",
                "bias",
                "rmse",
                "target_std",
                "pred_std",
                "std_ratio",
                "mean_sigma",
                "mean_abs_error",
                "sigma_error_corr",
                "readout_affine_applied_rows",
                "mean_abs_readout_affine_delta",
                "max_abs_readout_affine_delta",
            ]
        )
    frame = predictions.copy()
    if masked_only and "is_masked_holdout" in frame:
        frame = frame.loc[frame["is_masked_holdout"].astype(bool)].copy()
    frame = frame.dropna(subset=["atom_family", "target_value", "predicted_value"])
    if frame.empty:
        return pd.DataFrame()

    def family_sort_key(family: str) -> tuple[int, str]:
        if family in BIOEMU_FAMILY_ORDER:
            return (BIOEMU_FAMILY_ORDER.index(family), family)
        return (len(BIOEMU_FAMILY_ORDER), family)

    families = sorted(
        frame["atom_family"].astype(str).unique().tolist(),
        key=family_sort_key,
    )
    rows = [
        _bioemu_family_metric_row(
            frame.loc[frame["atom_family"].astype(str).eq(family)],
            family,
        )
        for family in families
    ]
    hn_cprime_row = _bioemu_combined_metric_row(
        rows,
        label="HN/C' combined",
        families=("HN", "C'"),
    )
    if hn_cprime_row is not None:
        rows.append(hn_cprime_row)
    return pd.DataFrame(rows)


def _bioemu_family_metric_row(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    """Compute one BioEmu family diagnostic row."""

    target = frame["target_value"].astype(float)
    predicted = frame["predicted_value"].astype(float)
    residual = predicted - target
    abs_error = residual.abs()
    count = int(len(frame))
    target_std = float(target.std(ddof=0)) if count else float("nan")
    pred_std = float(predicted.std(ddof=0)) if count else float("nan")
    std_ratio = (
        pred_std / target_std
        if math.isfinite(target_std) and abs(target_std) > 1.0e-12
        else float("nan")
    )
    sigma = (
        frame["posterior_sigma"].astype(float)
        if "posterior_sigma" in frame
        else pd.Series(dtype=float)
    )
    sigma_error_corr = float("nan")
    if len(sigma) >= 2 and abs_error.nunique(dropna=True) >= 2:
        corr = sigma.corr(abs_error)
        sigma_error_corr = float(corr) if corr is not None else float("nan")
    readout_delta = (
        frame["readout_affine_delta"].astype(float)
        if "readout_affine_delta" in frame
        else pd.Series(dtype=float)
    )
    readout_delta = readout_delta.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).dropna()
    return {
        "atom_family": label,
        "masked_rows": count,
        "ccc": _bioemu_ccc(predicted, target),
        "mae": float(abs_error.mean()) if count else float("nan"),
        "bias": float(residual.mean()) if count else float("nan"),
        "rmse": float((residual.pow(2).mean()) ** 0.5) if count else float("nan"),
        "target_std": target_std,
        "pred_std": pred_std,
        "std_ratio": std_ratio,
        "mean_sigma": float(sigma.mean()) if len(sigma) else float("nan"),
        "mean_abs_error": float(abs_error.mean()) if count else float("nan"),
        "sigma_error_corr": sigma_error_corr,
        "readout_affine_applied_rows": int(
            (readout_delta.abs() > 1.0e-8).sum()
        )
        if len(readout_delta)
        else 0,
        "mean_abs_readout_affine_delta": float(readout_delta.abs().mean())
        if len(readout_delta)
        else float("nan"),
        "max_abs_readout_affine_delta": float(readout_delta.abs().max())
        if len(readout_delta)
        else float("nan"),
    }


def _bioemu_combined_metric_row(
    rows: list[dict[str, Any]],
    *,
    label: str,
    families: tuple[str, ...],
) -> dict[str, Any] | None:
    """Combine family rows without pooling incompatible chemical-shift scales."""

    selected = [row for row in rows if row.get("atom_family") in families]
    if not selected:
        return None
    total_rows = sum(int(row.get("masked_rows") or 0) for row in selected)
    if total_rows <= 0:
        return None

    def weighted_mean(field: str) -> float:
        values = [
            (
                float(row[field]),
                int(row.get("masked_rows") or 0),
            )
            for row in selected
            if field in row and math.isfinite(float(row[field]))
        ]
        weight_sum = sum(weight for _, weight in values)
        if weight_sum <= 0:
            return float("nan")
        return sum(value * weight for value, weight in values) / weight_sum

    def finite_mean(field: str) -> float:
        values = [
            float(row[field])
            for row in selected
            if field in row and math.isfinite(float(row[field]))
        ]
        return float(sum(values) / len(values)) if values else float("nan")

    return {
        "atom_family": label,
        "masked_rows": total_rows,
        "ccc": finite_mean("ccc"),
        "mae": weighted_mean("mae"),
        "bias": weighted_mean("bias"),
        "rmse": finite_mean("rmse"),
        "target_std": finite_mean("target_std"),
        "pred_std": finite_mean("pred_std"),
        "std_ratio": finite_mean("std_ratio"),
        "mean_sigma": weighted_mean("mean_sigma"),
        "mean_abs_error": weighted_mean("mean_abs_error"),
        "sigma_error_corr": finite_mean("sigma_error_corr"),
        "readout_affine_applied_rows": sum(
            int(row.get("readout_affine_applied_rows") or 0) for row in selected
        ),
        "mean_abs_readout_affine_delta": weighted_mean(
            "mean_abs_readout_affine_delta"
        ),
        "max_abs_readout_affine_delta": max(
            [
                float(row.get("max_abs_readout_affine_delta", float("nan")))
                for row in selected
                if math.isfinite(
                    float(row.get("max_abs_readout_affine_delta", float("nan")))
                )
            ],
            default=float("nan"),
        ),
    }


def _bioemu_ccc(predicted: pd.Series, target: pd.Series) -> float:
    """Compute concordance correlation for pandas series."""

    if len(predicted) < 2 or len(target) < 2:
        return float("nan")
    pred_mean = float(predicted.mean())
    target_mean = float(target.mean())
    pred_var = float(((predicted - pred_mean) ** 2).mean())
    target_var = float(((target - target_mean) ** 2).mean())
    covariance = float(((predicted - pred_mean) * (target - target_mean)).mean())
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    return float("nan") if denominator <= 1.0e-12 else 2.0 * covariance / denominator


def _bioemu_prediction_source(snapshot: dict[str, Any]) -> pd.DataFrame:
    """Return masked BioEmu predictions when available, otherwise all predictions."""

    masked = snapshot.get("bioemu_latent_nmr_masked_predictions")
    if masked is not None and not masked.empty:
        return masked
    predictions = snapshot.get("bioemu_latent_nmr_predictions")
    if predictions is not None and not predictions.empty:
        return predictions
    return pd.DataFrame()


def _bioemu_filtered_prediction_frame(frame: pd.DataFrame, atom_family: str) -> pd.DataFrame:
    """Filter BioEmu prediction rows for a selected family or all families."""

    if frame.empty or "atom_family" not in frame:
        return pd.DataFrame()
    if "is_masked_holdout" in frame:
        frame = frame.loc[frame["is_masked_holdout"].astype(bool)].copy()
    if atom_family != "All":
        frame = frame.loc[frame["atom_family"].astype(str).eq(atom_family)].copy()
    return frame


def _bioemu_worst_entry_table(
    frame: pd.DataFrame,
    *,
    atom_family: str,
    limit: int = 40,
) -> pd.DataFrame:
    """Return worst BioEmu validation entries for the selected family."""

    filtered = _bioemu_filtered_prediction_frame(frame, atom_family)
    if filtered.empty:
        return pd.DataFrame()
    filtered = filtered.copy()
    filtered["residual"] = (
        filtered["predicted_value"].astype(float) - filtered["target_value"].astype(float)
    )
    filtered["abs_error"] = filtered["residual"].abs()
    group_cols = [
        column
        for column in ["split", "entity_uid", "atom_family"]
        if column in filtered.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    table = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            rows=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("residual", lambda value: float((value.pow(2).mean()) ** 0.5)),
            bias=("residual", "mean"),
            max_abs_error=("abs_error", "max"),
            mean_sigma=(
                "posterior_sigma",
                "mean",
            )
            if "posterior_sigma" in filtered
            else ("abs_error", "size"),
        )
        .reset_index()
        .sort_values(["mae", "max_abs_error"], ascending=False)
        .head(limit)
    )
    return table


def _plotly_express() -> Any:
    """Import plotly.express lazily so monitoring tests do not require it."""

    try:
        import plotly.express as px
    except Exception:
        return None
    return px


def _bioemu_family_progress_figure(metrics: pd.DataFrame) -> Any:
    """Build a masked CCC/std-ratio summary plot for BioEmu families."""

    px = _plotly_express()
    if px is None or metrics.empty:
        return None
    return px.bar(
        metrics,
        x="atom_family",
        y="ccc",
        color="std_ratio",
        color_continuous_scale="Viridis",
        range_y=[-1.0, 1.0],
        hover_data=[
            "masked_rows",
            "mae",
            "bias",
            "rmse",
            "target_std",
            "pred_std",
            "mean_sigma",
            "sigma_error_corr",
            "readout_affine_applied_rows",
            "mean_abs_readout_affine_delta",
            "max_abs_readout_affine_delta",
        ],
        title="Masked posterior mean CCC by atom family",
    )


def _bioemu_family_scatter_figure(frame: pd.DataFrame, atom_family: str) -> Any:
    """Build a target-vs-posterior-mean scatter plot."""

    px = _plotly_express()
    filtered = _bioemu_filtered_prediction_frame(frame, atom_family)
    if px is None or filtered.empty:
        return None
    return px.scatter(
        filtered,
        x="target_value",
        y="predicted_value",
        color="atom_family",
        hover_data=[
            column
            for column in ["entity_uid", "target_id", "posterior_sigma", "residue_index"]
            if column in filtered.columns
        ],
        title=f"{atom_family} masked target vs posterior mean",
    )


def _bioemu_residual_histogram_figure(frame: pd.DataFrame, atom_family: str) -> Any:
    """Build a BioEmu residual histogram."""

    px = _plotly_express()
    filtered = _bioemu_filtered_prediction_frame(frame, atom_family)
    if px is None or filtered.empty:
        return None
    filtered = filtered.copy()
    filtered["residual"] = (
        filtered["predicted_value"].astype(float) - filtered["target_value"].astype(float)
    )
    return px.histogram(
        filtered,
        x="residual",
        color="atom_family",
        nbins=60,
        marginal="box",
        title=f"{atom_family} residual distribution",
    )


def _bioemu_sigma_error_figure(frame: pd.DataFrame, atom_family: str) -> Any:
    """Build a posterior sigma versus absolute error calibration plot."""

    px = _plotly_express()
    filtered = _bioemu_filtered_prediction_frame(frame, atom_family)
    if (
        px is None
        or filtered.empty
        or "posterior_sigma" not in filtered
    ):
        return None
    filtered = filtered.copy()
    filtered["abs_error"] = (
        filtered["predicted_value"].astype(float) - filtered["target_value"].astype(float)
    ).abs()
    return px.scatter(
        filtered,
        x="posterior_sigma",
        y="abs_error",
        color="atom_family",
        hover_data=[
            column
            for column in ["entity_uid", "target_id", "residue_index"]
            if column in filtered.columns
        ],
        title=f"{atom_family} posterior sigma vs absolute error",
    )


def _bioemu_chart_probability_figure(posterior_moments: Any) -> Any:
    """Build chart probability heatmap/summary for BioEmu moment rows."""

    px = _plotly_express()
    if (
        px is None
        or posterior_moments is None
        or posterior_moments.empty
        or not {"atom_family", "chart_index", "chart_probability"}.issubset(
            posterior_moments.columns
        )
    ):
        return None
    summary = (
        posterior_moments.groupby(["atom_family", "chart_index"], dropna=False)
        .agg(mean_chart_probability=("chart_probability", "mean"))
        .reset_index()
    )
    return px.density_heatmap(
        summary,
        x="chart_index",
        y="atom_family",
        z="mean_chart_probability",
        histfunc="avg",
        color_continuous_scale="Blues",
        title="Mean chart probability by atom family",
    )


def _bioemu_bridge_family_figure(bridge_audit: Any) -> Any:
    """Build bridge stability figure grouped by atom family when possible."""

    px = _plotly_express()
    if px is None or bridge_audit is None or bridge_audit.empty:
        return None
    if not {"atom_family", "bridge_norm"}.issubset(bridge_audit.columns):
        return None
    return px.box(
        bridge_audit,
        x="atom_family",
        y="bridge_norm",
        color="atom_family",
        points="suspectedoutliers",
        title="Posterior bridge norm by atom family",
    )


def _bioemu_available_families(frame: pd.DataFrame) -> list[str]:
    """Return BioEmu family selector options in the preferred order."""

    if frame.empty or "atom_family" not in frame:
        return ["HN", "C'", "All"]
    present = set(frame["atom_family"].dropna().astype(str).unique().tolist())
    ordered = [family for family in BIOEMU_FAMILY_ORDER if family in present]
    extras = sorted(present.difference(BIOEMU_FAMILY_ORDER))
    return [*ordered, *extras, "All"]


def _bioemu_static_scatter_name(atom_family: str) -> str | None:
    """Return the static BioEmu scatter PNG name for a family."""

    if atom_family == "All":
        return None
    slug = _atom_family_slug(atom_family).lower()
    return f"bioemu_latent_{slug}_masked_scatter.png"


def _bioemu_show_family_metric_block(
    st: Any,
    snapshot: dict[str, Any],
    *,
    key_prefix: str,
) -> pd.DataFrame:
    """Render the shared BioEmu family metric table and progress plot."""

    source = _bioemu_prediction_source(snapshot)
    metrics = _bioemu_family_metric_frame(source)
    if metrics.empty:
        st.info("Waiting for BioEmu prediction parquet rows.")
        return metrics
    st.write("BioEmu masked family diagnostics")
    st.dataframe(metrics, width="stretch", hide_index=True)
    figure = _bioemu_family_progress_figure(metrics)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", key=f"{key_prefix}_family_progress")
    return metrics


def _bioemu_ucbshift_factor_atlas_block(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render UCBShift-aware factor-bin bottleneck diagnostics."""

    paths = snapshot["paths"]
    factor_audit = snapshot.get("bioemu_ucbshift_factor_atlas_audit")
    signed_audit = snapshot.get("bioemu_mechanism_signed_direction_audit")
    figure_paths = [
        (
            paths.figures_dir / "bioemu_ucbshift_factor_atlas_summary.png",
            "UCBShift factor atlas summary",
        ),
        (
            paths.figures_dir / "bioemu_ucbshift_factor_atlas_hn_ring_bins.png",
            "HN ring-current proxy bins",
        ),
        (
            paths.figures_dir / "bioemu_ucbshift_factor_atlas_cprime_carbonyl_bins.png",
            "C' carbonyl/backbone proxy bins",
        ),
        (
            paths.figures_dir / "bioemu_ucbshift_factor_atlas_hn_cprime_conflict.png",
            "HN/C' factor conflict score",
        ),
        (
            paths.figures_dir / "bioemu_mechanism_signed_direction_audit.png",
            "v101 mechanism-signed rare-regime priority",
        ),
        (
            paths.figures_dir / "bioemu_hn_ring_signed_subregime_audit.png",
            "v101 HN ring signed subregimes",
        ),
        (
            paths.figures_dir / "bioemu_cprime_carbonyl_subregime_audit.png",
            "v101 C' carbonyl signed subregimes",
        ),
    ]
    if not any(path.exists() for path, _caption in figure_paths) and (
        (factor_audit is None or factor_audit.empty)
        and (signed_audit is None or signed_audit.empty)
    ):
        return

    st.write("UCBShift factor atlas")
    st.caption(
        "Factor-conditioned diagnostics map UCBShift-style physical terms "
        "(ring current, H-bond geometry, carbonyl/backbone charts) onto the "
        "BioEmu atlas. The v101 mechanism-signed audit then separates bins "
        "where the model correction has the opposite sign from the detached "
        "physics residual, which is the current HN/C' bottleneck."
    )
    cols = st.columns(2)
    for index, (path, caption) in enumerate(figure_paths):
        render_image_or_wait(cols[index % 2], path, caption)

    if factor_audit is not None and not factor_audit.empty:
        audit = factor_audit.copy()
        if "ccc_gap_to_target" in audit:
            audit = audit.sort_values(
                ["ccc_gap_to_target", "row_count"],
                ascending=[False, False],
            )
        display_columns = [
            "factor_family",
            "ucbshift_proxy",
            "coordinate_name",
            "coordinate_bin",
            "atom_family",
            "row_count",
            "ccc",
            "ccc_gap_to_target",
            "mae",
            "bias",
            "std_ratio",
            "mean_reliability",
            "mean_uncertainty",
        ]
        st.write("Worst UCBShift factor-conditioned bins")
        st.dataframe(
            audit[[column for column in display_columns if column in audit.columns]].head(24),
            width="stretch",
            hide_index=True,
        )
    if signed_audit is not None and not signed_audit.empty:
        audit = signed_audit.copy()
        if "priority_score" in audit:
            audit = audit.sort_values(
                ["priority_score", "ccc_gap_to_target", "row_count"],
                ascending=[False, False, False],
            )
        display_columns = [
            "mechanism_family",
            "mechanism_route",
            "coordinate_name",
            "coordinate_bin",
            "atom_family",
            "row_count",
            "ccc",
            "ccc_gap_to_target",
            "signed_residual_direction",
            "signed_model_delta_direction",
            "direction_mismatch",
            "priority_score",
            "recommended_action",
        ]
        st.write("v101 signed mechanism action queue")
        st.dataframe(
            audit[[column for column in display_columns if column in audit.columns]].head(24),
            width="stretch",
            hide_index=True,
        )


def _bioemu_family_calibration_strategy_block(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render selected per-family calibration strategy diagnostics."""

    paths = snapshot["paths"]
    probe = snapshot.get("bioemu_family_calibration_strategy_probe")
    strategy_metrics = snapshot.get("bioemu_family_calibration_strategy_metrics")
    recommended_metrics = snapshot.get("bioemu_recommended_family_metrics")
    calibrated = snapshot.get("bioemu_family_calibration_strategy_predictions")
    residue_audit = snapshot.get("bioemu_family_calibration_strategy_residue_audit")
    residue_type_audit = snapshot.get(
        "bioemu_family_calibration_strategy_residue_type_audit"
    )
    residue_type_guard = snapshot.get(
        "bioemu_family_calibration_strategy_residue_type_guard_probe"
    )
    residue_type_guard_predictions = snapshot.get(
        "bioemu_family_calibration_strategy_residue_type_guard_predictions"
    )
    proline_n_audit = snapshot.get("bioemu_proline_n_rare_regime_audit")
    proline_n_expert_probe = snapshot.get("bioemu_proline_n_expert_probe")
    proline_n_expert_predictions = snapshot.get(
        "bioemu_proline_n_expert_predictions"
    )
    proline_n_mobility_expert_probe = snapshot.get(
        "bioemu_proline_n_mobility_expert_probe"
    )
    proline_n_mobility_expert_split_guard = snapshot.get(
        "bioemu_proline_n_mobility_expert_split_guard"
    )
    proline_n_mobility_expert_predictions = snapshot.get(
        "bioemu_proline_n_mobility_expert_predictions"
    )
    proline_n_mobility_entity_holdout = snapshot.get(
        "bioemu_proline_n_mobility_entity_holdout"
    )
    proline_n_mobility_entity_holdout_predictions = snapshot.get(
        "bioemu_proline_n_mobility_entity_holdout_predictions"
    )
    residue_family_expert_probe = snapshot.get(
        "bioemu_residue_family_expert_probe"
    )
    residue_family_expert_predictions = snapshot.get(
        "bioemu_residue_family_expert_predictions"
    )
    residue_family_policy_probe = snapshot.get(
        "bioemu_residue_family_expert_policy_probe"
    )
    residue_family_policy_predictions = snapshot.get(
        "bioemu_residue_family_expert_policy_predictions"
    )
    residue_family_reliability_probe = snapshot.get(
        "bioemu_residue_family_expert_reliability_gate_probe"
    )
    residue_family_reliability_predictions = snapshot.get(
        "bioemu_residue_family_expert_reliability_gate_predictions"
    )
    residue_family_atlas_probe = snapshot.get(
        "bioemu_residue_family_atlas_local_expert_probe"
    )
    residue_family_atlas_predictions = snapshot.get(
        "bioemu_residue_family_atlas_local_expert_predictions"
    )
    image_path = paths.figures_dir / "bioemu_family_calibration_strategy_probe.png"
    recommended_gap_path = paths.figures_dir / "bioemu_recommended_family_ccc95_gap.png"
    progress_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_all_family_progress.png"
    )
    heatmap_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_entry_family_mae_heatmap.png"
    )
    worst_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_worst_residue_errors.png"
    )
    residue_type_heatmap_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_residue_type_family_mae.png"
    )
    residue_type_bottleneck_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_residue_type_bottlenecks.png"
    )
    residue_type_guard_progress_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_residue_type_guard_progress.png"
    )
    residue_type_guard_bottleneck_path = (
        paths.figures_dir
        / "bioemu_family_calibration_strategy_residue_type_guard_bottlenecks.png"
    )
    proline_n_scatter_path = (
        paths.figures_dir / "bioemu_proline_n_rare_regime_scatter.png"
    )
    proline_n_error_path = (
        paths.figures_dir / "bioemu_proline_n_rare_regime_residue_errors.png"
    )
    proline_n_expert_probe_path = (
        paths.figures_dir / "bioemu_proline_n_expert_probe.png"
    )
    proline_n_expert_error_path = (
        paths.figures_dir / "bioemu_proline_n_expert_residue_errors.png"
    )
    proline_n_mobility_expert_path = (
        paths.figures_dir / "bioemu_proline_n_mobility_expert_probe.png"
    )
    proline_n_mobility_split_guard_path = (
        paths.figures_dir / "bioemu_proline_n_mobility_expert_split_guard.png"
    )
    proline_n_mobility_entity_holdout_path = (
        paths.figures_dir / "bioemu_proline_n_mobility_entity_holdout.png"
    )
    residue_family_expert_gain_path = (
        paths.figures_dir / "bioemu_residue_family_expert_top_gains.png"
    )
    residue_family_expert_bottleneck_path = (
        paths.figures_dir / "bioemu_residue_family_expert_bottlenecks.png"
    )
    residue_family_policy_progress_path = (
        paths.figures_dir / "bioemu_residue_family_expert_policy_progress.png"
    )
    residue_family_policy_actions_path = (
        paths.figures_dir / "bioemu_residue_family_expert_policy_actions.png"
    )
    residue_family_reliability_progress_path = (
        paths.figures_dir
        / "bioemu_residue_family_expert_reliability_gate_progress.png"
    )
    residue_family_reliability_actions_path = (
        paths.figures_dir
        / "bioemu_residue_family_expert_reliability_gate_actions.png"
    )
    residue_family_atlas_progress_path = (
        paths.figures_dir / "bioemu_residue_family_atlas_local_expert_progress.png"
    )
    residue_family_atlas_bottleneck_path = (
        paths.figures_dir
        / "bioemu_residue_family_atlas_local_expert_bottlenecks.png"
    )
    if (
        (probe is None or probe.empty)
        and (strategy_metrics is None or strategy_metrics.empty)
        and (recommended_metrics is None or recommended_metrics.empty)
        and (calibrated is None or calibrated.empty)
        and (residue_audit is None or residue_audit.empty)
        and (residue_type_audit is None or residue_type_audit.empty)
        and (residue_type_guard is None or residue_type_guard.empty)
        and (
            residue_type_guard_predictions is None
            or residue_type_guard_predictions.empty
        )
        and (proline_n_audit is None or proline_n_audit.empty)
        and (proline_n_expert_probe is None or proline_n_expert_probe.empty)
        and (
            proline_n_expert_predictions is None
            or proline_n_expert_predictions.empty
        )
        and (
            proline_n_mobility_expert_probe is None
            or proline_n_mobility_expert_probe.empty
        )
        and (
            proline_n_mobility_expert_split_guard is None
            or proline_n_mobility_expert_split_guard.empty
        )
        and (
            proline_n_mobility_expert_predictions is None
            or proline_n_mobility_expert_predictions.empty
        )
        and (
            proline_n_mobility_entity_holdout is None
            or proline_n_mobility_entity_holdout.empty
        )
        and (
            proline_n_mobility_entity_holdout_predictions is None
            or proline_n_mobility_entity_holdout_predictions.empty
        )
        and (
            residue_family_expert_probe is None
            or residue_family_expert_probe.empty
        )
        and (
            residue_family_expert_predictions is None
            or residue_family_expert_predictions.empty
        )
        and (
            residue_family_policy_probe is None
            or residue_family_policy_probe.empty
        )
        and (
            residue_family_policy_predictions is None
            or residue_family_policy_predictions.empty
        )
        and (
            residue_family_reliability_probe is None
            or residue_family_reliability_probe.empty
        )
        and (
            residue_family_reliability_predictions is None
            or residue_family_reliability_predictions.empty
        )
        and (residue_family_atlas_probe is None or residue_family_atlas_probe.empty)
        and (
            residue_family_atlas_predictions is None
            or residue_family_atlas_predictions.empty
        )
        and not image_path.exists()
        and not recommended_gap_path.exists()
        and not progress_path.exists()
        and not heatmap_path.exists()
        and not worst_path.exists()
        and not residue_type_heatmap_path.exists()
        and not residue_type_bottleneck_path.exists()
        and not residue_type_guard_progress_path.exists()
        and not residue_type_guard_bottleneck_path.exists()
        and not proline_n_scatter_path.exists()
        and not proline_n_error_path.exists()
        and not proline_n_expert_probe_path.exists()
        and not proline_n_expert_error_path.exists()
        and not proline_n_mobility_expert_path.exists()
        and not proline_n_mobility_split_guard_path.exists()
        and not proline_n_mobility_entity_holdout_path.exists()
        and not residue_family_expert_gain_path.exists()
        and not residue_family_expert_bottleneck_path.exists()
        and not residue_family_policy_progress_path.exists()
        and not residue_family_policy_actions_path.exists()
        and not residue_family_reliability_progress_path.exists()
        and not residue_family_reliability_actions_path.exists()
        and not residue_family_atlas_progress_path.exists()
        and not residue_family_atlas_bottleneck_path.exists()
    ):
        return
    render_panel_header(
        st,
        "Family Calibration Strategy Probe",
        "Baseline, chart-bin, and scale-up affine probes are compared per atom "
        "family. The selected strategy must clear a small CCC gain threshold, "
        "otherwise baseline is kept to avoid noisy over-correction.",
    )
    render_image_or_wait(
        st,
        recommended_gap_path,
        "Recommended family readout vs CCC 0.95 target",
    )
    render_image_or_wait(
        st,
        image_path,
        "Selected family calibration strategy",
    )
    render_image_or_wait(
        st,
        progress_path,
        "Selected strategy all-family posterior mean progress",
    )
    render_image_or_wait(
        st,
        heatmap_path,
        "Selected strategy entry-family posterior mean MAE heatmap",
    )
    render_image_or_wait(
        st,
        worst_path,
        "Worst residue-atom posterior mean errors",
    )
    render_image_or_wait(
        st,
        residue_type_heatmap_path,
        "Residue-type x atom-family posterior mean MAE",
    )
    render_image_or_wait(
        st,
        residue_type_bottleneck_path,
        "Residue-type bottlenecks after selected calibration",
    )
    render_image_or_wait(
        st,
        residue_type_guard_progress_path,
        "Residue-type guard posterior mean progress",
    )
    render_image_or_wait(
        st,
        residue_type_guard_bottleneck_path,
        "Residue-type guard bottlenecks",
    )
    render_image_or_wait(
        st,
        proline_n_scatter_path,
        "PRO:N rare-regime posterior mean",
    )
    render_image_or_wait(
        st,
        proline_n_error_path,
        "PRO:N rare-regime residue errors",
    )
    render_image_or_wait(
        st,
        proline_n_expert_probe_path,
        "PRO:N train-fitted expert probe",
    )
    render_image_or_wait(
        st,
        proline_n_expert_error_path,
        "PRO:N train-fitted expert residue errors",
    )
    render_image_or_wait(
        st,
        proline_n_mobility_expert_path,
        "PRO:N mobility/contact expert probe",
    )
    render_image_or_wait(
        st,
        proline_n_mobility_split_guard_path,
        "PRO:N mobility/contact expert split guard",
    )
    render_image_or_wait(
        st,
        proline_n_mobility_entity_holdout_path,
        "PRO:N mobility/contact leave-one-entry-out guard",
    )
    render_image_or_wait(
        st,
        residue_family_expert_gain_path,
        "Train-fitted residue-family expert gains",
    )
    render_image_or_wait(
        st,
        residue_family_expert_bottleneck_path,
        "Residual bottlenecks after residue-family experts",
    )
    render_image_or_wait(
        st,
        residue_family_policy_progress_path,
        "Promote-only residue-family expert policy progress",
    )
    render_image_or_wait(
        st,
        residue_family_policy_actions_path,
        "Promote-only residue-family expert policy actions",
    )
    render_image_or_wait(
        st,
        residue_family_reliability_progress_path,
        "Row-level reliability gate progress",
    )
    render_image_or_wait(
        st,
        residue_family_reliability_actions_path,
        "Row-level reliability gate actions",
    )
    render_image_or_wait(
        st,
        residue_family_atlas_progress_path,
        "Atlas-conditioned residue-family local expert progress",
    )
    render_image_or_wait(
        st,
        residue_family_atlas_bottleneck_path,
        "Atlas-conditioned residue-family local bottlenecks",
    )
    if recommended_metrics is not None and not recommended_metrics.empty:
        metric_columns = [
            column
            for column in [
                "atom_family",
                "promote_recommendation",
                "selected_strategy",
                "selected_coordinate_name",
                "baseline_ccc",
                "recommended_ccc",
                "delta_ccc",
                "recommended_gap_to_0p95",
                "baseline_mae",
                "recommended_mae",
                "baseline_std_ratio",
                "selected_std_ratio",
                "rows",
            ]
            if column in recommended_metrics.columns
        ]
        st.write("Recommended viewer metric summary")
        st.dataframe(
            (
                recommended_metrics[metric_columns]
                if metric_columns
                else recommended_metrics
            ),
            width="stretch",
            hide_index=True,
        )
    if strategy_metrics is not None and not strategy_metrics.empty:
        metric_columns = [
            column
            for column in [
                "atom_family",
                "selected_strategy",
                "selected_coordinate_name",
                "baseline_ccc",
                "selected_ccc",
                "delta_ccc",
                "ccc_gap_to_0p95",
                "baseline_mae",
                "selected_mae",
                "baseline_bias",
                "selected_bias",
                "baseline_std_ratio",
                "selected_std_ratio",
                "rows",
            ]
            if column in strategy_metrics.columns
        ]
        st.write("Selected strategy metric summary")
        st.dataframe(
            strategy_metrics[metric_columns] if metric_columns else strategy_metrics,
            width="stretch",
            hide_index=True,
        )
    if probe is not None and not probe.empty:
        display_columns = [
            column
            for column in [
                "atom_family",
                "selected_strategy",
                "selected_coordinate_name",
                "baseline_ccc",
                "chart_ccc",
                "affine_ccc",
                "selected_ccc",
                "delta_ccc",
                "baseline_mae",
                "selected_mae",
                "baseline_bias",
                "selected_bias",
                "baseline_std_ratio",
                "selected_std_ratio",
                "reason",
            ]
            if column in probe.columns
        ]
        st.dataframe(
            probe[display_columns] if display_columns else probe,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Waiting for bioemu_family_calibration_strategy_probe.parquet.")
    if calibrated is not None and not calibrated.empty:
        with st.expander("Selected strategy validation rows", expanded=False):
            st.dataframe(calibrated.head(800), width="stretch", hide_index=True)
    if residue_audit is not None and not residue_audit.empty:
        with st.expander("Residue-atom posterior mean alignment audit", expanded=False):
            display_columns = [
                column
                for column in [
                    "selected_error_rank",
                    "entity_uid",
                    "target_id",
                    "atom_family",
                    "target_value",
                    "predicted_value",
                    "calibrated_predicted_value",
                    "selected_abs_error",
                    "delta_abs_error",
                    "selected_strategy",
                    "selected_coordinate_name",
                ]
                if column in residue_audit.columns
            ]
            st.dataframe(
                residue_audit[display_columns].head(300)
                if display_columns
                else residue_audit.head(300),
                width="stretch",
                hide_index=True,
            )
    if residue_type_audit is not None and not residue_type_audit.empty:
        with st.expander("Residue-type family bottleneck audit", expanded=False):
            display_columns = [
                column
                for column in [
                    "residue_name",
                    "atom_family",
                    "rows",
                    "entity_count",
                    "baseline_mae",
                    "selected_mae",
                    "delta_mae",
                    "selected_bias",
                    "max_selected_abs_error",
                    "worst_entity_uid",
                    "worst_target_id",
                    "selected_strategy",
                ]
                if column in residue_type_audit.columns
            ]
            st.dataframe(
                residue_type_audit[display_columns].head(120)
                if display_columns
                else residue_type_audit.head(120),
                width="stretch",
                hide_index=True,
            )
    if residue_type_guard is not None and not residue_type_guard.empty:
        with st.expander("Residue-type guard probe", expanded=False):
            display_columns = [
                column
                for column in [
                    "residue_name",
                    "atom_family",
                    "rows",
                    "entity_count",
                    "baseline_mae",
                    "selected_mae",
                    "guarded_mae",
                    "delta_selected_vs_baseline_mae",
                    "delta_guarded_vs_selected_mae",
                    "guard_action",
                    "guard_reason",
                    "worst_entity_uid",
                    "worst_target_id",
                ]
                if column in residue_type_guard.columns
            ]
            st.dataframe(
                residue_type_guard[display_columns].head(160)
                if display_columns
                else residue_type_guard.head(160),
                width="stretch",
                hide_index=True,
            )
    if proline_n_audit is not None and not proline_n_audit.empty:
        with st.expander("PRO:N rare-regime audit", expanded=True):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "target_value",
                    "baseline_predicted_value",
                    "selected_predicted_value",
                    "guarded_predicted_value",
                    "baseline_abs_error",
                    "selected_abs_error",
                    "guarded_abs_error",
                    "delta_selected_vs_baseline_abs_error",
                    "delta_guarded_vs_selected_abs_error",
                    "guard_action",
                    "recommended_next_action",
                ]
                if column in proline_n_audit.columns
            ]
            st.dataframe(
                proline_n_audit[display_columns]
                if display_columns
                else proline_n_audit,
                width="stretch",
                hide_index=True,
            )
    if proline_n_expert_probe is not None and not proline_n_expert_probe.empty:
        with st.expander("PRO:N train-fitted expert probe", expanded=True):
            display_columns = [
                column
                for column in [
                    "expert_name",
                    "fit_rows",
                    "eval_rows",
                    "raw_mean_shift",
                    "shrinkage",
                    "expert_mean_shift",
                    "guarded_mae",
                    "expert_mae",
                    "delta_expert_vs_guarded_mae",
                    "guarded_ccc",
                    "expert_ccc",
                    "support_status",
                    "recommended_next_action",
                ]
                if column in proline_n_expert_probe.columns
            ]
            st.dataframe(
                proline_n_expert_probe[display_columns]
                if display_columns
                else proline_n_expert_probe,
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_expert_predictions is not None
        and not proline_n_expert_predictions.empty
    ):
        with st.expander("PRO:N expert validation rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "target_value",
                    "predicted_value",
                    "guarded_predicted_value",
                    "expert_predicted_value",
                    "guarded_abs_error",
                    "expert_abs_error",
                    "delta_expert_vs_guarded_abs_error",
                    "expert_action",
                    "expert_reason",
                ]
                if column in proline_n_expert_predictions.columns
            ]
            st.dataframe(
                proline_n_expert_predictions[display_columns]
                if display_columns
                else proline_n_expert_predictions,
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_mobility_expert_probe is not None
        and not proline_n_mobility_expert_probe.empty
    ):
        with st.expander("PRO:N mobility/contact expert probe", expanded=True):
            display_columns = [
                column
                for column in [
                    "expert_name",
                    "fit_mode",
                    "rows",
                    "fit_rows",
                    "eval_rows",
                    "active_mae",
                    "mobility_expert_mae",
                    "delta_mobility_expert_vs_active_mae",
                    "promote_rows",
                    "correction_cap",
                    "mean_delta",
                    "max_abs_delta",
                    "recommended_next_action",
                ]
                if column in proline_n_mobility_expert_probe.columns
            ]
            st.dataframe(
                proline_n_mobility_expert_probe[display_columns]
                if display_columns
                else proline_n_mobility_expert_probe,
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_mobility_expert_split_guard is not None
        and not proline_n_mobility_expert_split_guard.empty
    ):
        with st.expander("PRO:N mobility/contact split guard", expanded=True):
            display_columns = [
                column
                for column in [
                    "expert_name",
                    "fit_mode",
                    "fit_rows",
                    "eval_rows",
                    "fit_active_mae",
                    "fit_mobility_expert_mae",
                    "delta_fit_mobility_expert_vs_active_mae",
                    "eval_active_mae",
                    "eval_mobility_expert_mae",
                    "delta_eval_mobility_expert_vs_active_mae",
                    "eval_promote_rows",
                    "split_guard_status",
                    "recommended_next_action",
                ]
                if column in proline_n_mobility_expert_split_guard.columns
            ]
            st.dataframe(
                proline_n_mobility_expert_split_guard[display_columns]
                if display_columns
                else proline_n_mobility_expert_split_guard,
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_mobility_entity_holdout is not None
        and not proline_n_mobility_entity_holdout.empty
    ):
        with st.expander("PRO:N mobility/contact leave-one-entry-out guard", expanded=True):
            display_columns = [
                column
                for column in [
                    "expert_name",
                    "heldout_entity_uid",
                    "train_entity_count",
                    "train_rows",
                    "holdout_rows",
                    "holdout_active_mae",
                    "holdout_entity_expert_mae",
                    "delta_entity_expert_vs_active_mae",
                    "holdout_promote_rows",
                    "correction_cap",
                    "max_abs_delta",
                    "entity_holdout_status",
                ]
                if column in proline_n_mobility_entity_holdout.columns
            ]
            st.dataframe(
                proline_n_mobility_entity_holdout[display_columns]
                if display_columns
                else proline_n_mobility_entity_holdout,
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_mobility_expert_predictions is not None
        and not proline_n_mobility_expert_predictions.empty
    ):
        with st.expander("PRO:N mobility/contact expert rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_index",
                    "target_value",
                    "active_predicted_value",
                    "mobility_expert_predicted_value",
                    "active_abs_error",
                    "mobility_expert_abs_error",
                    "delta_mobility_expert_vs_active_abs_error",
                    "residue_rmsf",
                    "local_contact_variability",
                    "mobility_expert_delta",
                    "mobility_expert_action",
                    "mobility_expert_reason",
                ]
                if column in proline_n_mobility_expert_predictions.columns
            ]
            st.dataframe(
                proline_n_mobility_expert_predictions[display_columns].head(120)
                if display_columns
                else proline_n_mobility_expert_predictions.head(120),
                width="stretch",
                hide_index=True,
            )
    if (
        proline_n_mobility_entity_holdout_predictions is not None
        and not proline_n_mobility_entity_holdout_predictions.empty
    ):
        with st.expander("PRO:N mobility/contact entity-holdout rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_index",
                    "target_value",
                    "active_predicted_value",
                    "entity_holdout_predicted_value",
                    "active_abs_error",
                    "entity_holdout_abs_error",
                    "delta_entity_holdout_vs_active_abs_error",
                    "residue_rmsf",
                    "local_contact_variability",
                    "entity_holdout_delta",
                    "entity_holdout_action",
                    "entity_holdout_reason",
                ]
                if column in proline_n_mobility_entity_holdout_predictions.columns
            ]
            st.dataframe(
                proline_n_mobility_entity_holdout_predictions[
                    display_columns
                ].head(120)
                if display_columns
                else proline_n_mobility_entity_holdout_predictions.head(120),
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_expert_probe is not None
        and not residue_family_expert_probe.empty
    ):
        with st.expander("Residue-family expert candidates", expanded=True):
            display_columns = [
                column
                for column in [
                    "residue_family",
                    "fit_rows",
                    "eval_rows",
                    "fit_entity_count",
                    "eval_entity_count",
                    "expert_mean_shift",
                    "guarded_mae",
                    "expert_mae",
                    "delta_expert_vs_guarded_mae",
                    "guarded_ccc",
                    "expert_ccc",
                    "support_status",
                    "recommended_next_action",
                    "worst_entity_uid",
                    "worst_target_id",
                ]
                if column in residue_family_expert_probe.columns
            ]
            st.dataframe(
                residue_family_expert_probe[display_columns].head(160)
                if display_columns
                else residue_family_expert_probe.head(160),
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_expert_predictions is not None
        and not residue_family_expert_predictions.empty
    ):
        with st.expander("Residue-family expert validation rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "guarded_predicted_value",
                    "expert_predicted_value",
                    "guarded_abs_error",
                    "expert_abs_error",
                    "delta_expert_vs_guarded_abs_error",
                    "expert_action",
                    "expert_reason",
                ]
                if column in residue_family_expert_predictions.columns
            ]
            st.dataframe(
                residue_family_expert_predictions[display_columns].head(300)
                if display_columns
                else residue_family_expert_predictions.head(300),
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_policy_probe is not None
        and not residue_family_policy_probe.empty
    ):
        with st.expander(
            "Promote-only residue-family expert policy",
            expanded=True,
        ):
            display_columns = [
                column
                for column in [
                    "atom_family",
                    "rows",
                    "applied_rows",
                    "abstained_rows",
                    "guarded_ccc",
                    "policy_ccc",
                    "delta_policy_vs_guarded_ccc",
                    "guarded_mae",
                    "policy_mae",
                    "delta_policy_vs_guarded_mae",
                    "mean_abs_policy_shift",
                ]
                if column in residue_family_policy_probe.columns
            ]
            st.dataframe(
                residue_family_policy_probe[display_columns]
                if display_columns
                else residue_family_policy_probe,
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_policy_predictions is not None
        and not residue_family_policy_predictions.empty
    ):
        with st.expander("Residue-family policy validation rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "guarded_predicted_value",
                    "expert_predicted_value",
                    "policy_predicted_value",
                    "guarded_abs_error",
                    "policy_abs_error",
                    "delta_policy_vs_guarded_abs_error",
                    "policy_action",
                    "policy_reason",
                ]
                if column in residue_family_policy_predictions.columns
            ]
            st.dataframe(
                residue_family_policy_predictions[display_columns].head(300)
                if display_columns
                else residue_family_policy_predictions.head(300),
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_reliability_probe is not None
        and not residue_family_reliability_probe.empty
    ):
        with st.expander("Row-level reliability-gated expert policy", expanded=True):
            display_columns = [
                column
                for column in [
                    "atom_family",
                    "rows",
                    "applied_rows",
                    "abstained_rows",
                    "direction_abstained_rows",
                    "guarded_ccc",
                    "reliability_ccc",
                    "delta_reliability_vs_guarded_ccc",
                    "guarded_mae",
                    "reliability_mae",
                    "delta_reliability_vs_guarded_mae",
                    "mean_abs_reliability_shift",
                ]
                if column in residue_family_reliability_probe.columns
            ]
            st.dataframe(
                residue_family_reliability_probe[display_columns]
                if display_columns
                else residue_family_reliability_probe,
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_reliability_predictions is not None
        and not residue_family_reliability_predictions.empty
    ):
        with st.expander("Reliability-gated validation rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "guarded_predicted_value",
                    "expert_predicted_value",
                    "reliability_predicted_value",
                    "gate_center",
                    "gate_margin",
                    "gate_score",
                    "guarded_abs_error",
                    "reliability_abs_error",
                    "delta_reliability_vs_guarded_abs_error",
                    "reliability_action",
                    "reliability_reason",
                ]
                if column in residue_family_reliability_predictions.columns
            ]
            st.dataframe(
                residue_family_reliability_predictions[display_columns].head(300)
                if display_columns
                else residue_family_reliability_predictions.head(300),
                width="stretch",
                hide_index=True,
            )
    if residue_family_atlas_probe is not None and not residue_family_atlas_probe.empty:
        with st.expander("Atlas-conditioned residue-family local experts", expanded=True):
            display_columns = [
                column
                for column in [
                    "residue_family",
                    "fit_rows",
                    "eval_rows",
                    "feature_count",
                    "shrinkage",
                    "mean_abs_atlas_correction",
                    "guarded_mae",
                    "atlas_mae",
                    "delta_atlas_vs_guarded_mae",
                    "guarded_ccc",
                    "atlas_ccc",
                    "recommended_next_action",
                    "worst_entity_uid",
                    "worst_target_id",
                ]
                if column in residue_family_atlas_probe.columns
            ]
            st.dataframe(
                residue_family_atlas_probe[display_columns].head(180)
                if display_columns
                else residue_family_atlas_probe.head(180),
                width="stretch",
                hide_index=True,
            )
    if (
        residue_family_atlas_predictions is not None
        and not residue_family_atlas_predictions.empty
    ):
        with st.expander("Atlas-local expert validation rows", expanded=False):
            display_columns = [
                column
                for column in [
                    "entity_uid",
                    "target_id",
                    "residue_name",
                    "atom_family",
                    "target_value",
                    "guarded_predicted_value",
                    "atlas_predicted_value",
                    "atlas_policy_predicted_value",
                    "atlas_correction",
                    "guarded_abs_error",
                    "atlas_policy_abs_error",
                    "delta_atlas_policy_vs_guarded_abs_error",
                    "atlas_policy_action",
                    "atlas_policy_reason",
                ]
                if column in residue_family_atlas_predictions.columns
            ]
            st.dataframe(
                residue_family_atlas_predictions[display_columns].head(300)
                if display_columns
                else residue_family_atlas_predictions.head(300),
                width="stretch",
                hide_index=True,
            )


def _bioemu_family_affine_calibration_block(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render family affine calibration for posterior-mean shrinkage."""

    paths = snapshot["paths"]
    probe = snapshot.get("bioemu_family_affine_calibration_probe")
    calibrated = snapshot.get("bioemu_family_affine_calibrated_predictions")
    image_path = paths.figures_dir / "bioemu_family_affine_calibration_probe.png"
    if (
        (probe is None or probe.empty)
        and (calibrated is None or calibrated.empty)
        and not image_path.exists()
    ):
        return
    render_panel_header(
        st,
        "Family Affine Calibration Probe",
        "Each atom family fits a scale-up affine map on train split masked rows, "
        "then applies that fixed map to validation. Slopes below 1 are clipped "
        "to avoid adding more posterior-mean collapse while still repairing bias.",
    )
    render_image_or_wait(
        st,
        image_path,
        "All-family train-fit / validation-applied affine calibration probe",
    )
    if probe is not None and not probe.empty:
        display_columns = [
            column
            for column in [
                "atom_family",
                "fit_rows",
                "eval_rows",
                "slope",
                "intercept",
                "baseline_ccc",
                "calibrated_ccc",
                "delta_ccc",
                "baseline_std_ratio",
                "calibrated_std_ratio",
                "baseline_mae",
                "calibrated_mae",
                "baseline_bias",
                "calibrated_bias",
            ]
            if column in probe.columns
        ]
        st.dataframe(
            probe[display_columns] if display_columns else probe,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Waiting for bioemu_family_affine_calibration_probe.parquet.")
    if calibrated is not None and not calibrated.empty:
        with st.expander("Family affine calibrated validation rows", expanded=False):
            st.dataframe(calibrated.head(800), width="stretch", hide_index=True)


def _bioemu_family_chart_calibration_block(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render all-family train-fit/validation-applied chart calibration."""

    paths = snapshot["paths"]
    probe = snapshot.get("bioemu_family_chart_calibration_probe")
    calibrated = snapshot.get("bioemu_family_chart_calibrated_predictions")
    image_path = paths.figures_dir / "bioemu_family_chart_calibration_probe.png"
    if (
        (probe is None or probe.empty)
        and (calibrated is None or calibrated.empty)
        and not image_path.exists()
    ):
        return
    render_panel_header(
        st,
        "Family Chart Calibration Probe",
        "Each atom family is calibrated independently: train split residuals "
        "define atlas-coordinate quartile corrections, then validation rows "
        "receive only the correction from their coordinate bin. This separates "
        "bias repair from posterior-mean shrinkage diagnostics.",
    )
    render_image_or_wait(
        st,
        image_path,
        "All-family train-fit / validation-applied chart calibration probe",
    )
    if probe is not None and not probe.empty:
        table = probe.copy()
        if "calibration_kind" in table:
            chart_rows = table.loc[
                table["calibration_kind"].astype(str).ne("baseline")
            ].copy()
        else:
            chart_rows = table.copy()
        if (
            not chart_rows.empty
            and {"atom_family", "calibrated_ccc"}.issubset(chart_rows.columns)
        ):
            best_rows = (
                chart_rows.sort_values(
                    ["atom_family", "calibrated_ccc", "delta_ccc"],
                    ascending=[True, False, False],
                    kind="stable",
                )
                .groupby("atom_family", dropna=False)
                .head(1)
            )
            st.write("Best chart calibration per atom family")
            best_columns = [
                column
                for column in [
                    "atom_family",
                    "coordinate_name",
                    "baseline_ccc",
                    "calibrated_ccc",
                    "delta_ccc",
                    "baseline_mae",
                    "calibrated_mae",
                    "baseline_bias",
                    "calibrated_bias",
                    "mean_abs_applied_correction",
                ]
                if column in best_rows.columns
            ]
            st.dataframe(
                best_rows[best_columns] if best_columns else best_rows,
                width="stretch",
                hide_index=True,
            )
        with st.expander("All family chart calibration rows", expanded=False):
            st.dataframe(table.head(800), width="stretch", hide_index=True)
    else:
        st.info("Waiting for bioemu_family_chart_calibration_probe.parquet.")
    if calibrated is not None and not calibrated.empty:
        with st.expander("All-family calibrated validation rows", expanded=False):
            st.dataframe(calibrated.head(800), width="stretch", hide_index=True)


def _bioemu_cprime_chart_calibration_block(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render train-fit/validation-applied C' chart calibration diagnostics."""

    paths = snapshot["paths"]
    probe = snapshot.get("bioemu_cprime_chart_calibration_probe")
    calibrated = snapshot.get("bioemu_cprime_chart_calibrated_predictions")
    image_path = paths.figures_dir / "bioemu_cprime_chart_calibration_probe.png"
    if (
        (probe is None or probe.empty)
        and (calibrated is None or calibrated.empty)
        and not image_path.exists()
    ):
        return
    render_panel_header(
        st,
        "C' Chart Calibration Probe",
        "Train split C' residuals define atlas-coordinate quartile corrections, "
        "then those corrections are applied to validation C' rows. This is a "
        "leakage-safe probe for post-hoc posterior-mean calibration, not a "
        "training-time adapter.",
    )
    render_image_or_wait(
        st,
        image_path,
        "C' train-fit / validation-applied chart calibration probe",
    )
    if probe is not None and not probe.empty:
        table = probe.copy()
        if "calibration_kind" in table:
            chart_rows = table.loc[
                table["calibration_kind"].astype(str).ne("baseline")
            ].copy()
        else:
            chart_rows = table.copy()
        if not chart_rows.empty and "calibrated_ccc" in chart_rows:
            chart_rows = chart_rows.sort_values(
                ["calibrated_ccc", "delta_ccc"],
                ascending=False,
                kind="stable",
            )
            best = chart_rows.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Best coordinate", str(best.get("coordinate_name", "")))
            cols[1].metric("Calibrated CCC", _format_metric_value(best.get("calibrated_ccc")))
            cols[2].metric("Delta CCC", _format_metric_value(best.get("delta_ccc")))
            cols[3].metric(
                "Calibrated bias",
                _format_metric_value(best.get("calibrated_bias")),
            )
        display_columns = [
            column
            for column in [
                "coordinate_name",
                "fit_split",
                "eval_split",
                "fit_rows",
                "eval_rows",
                "bin_count",
                "baseline_ccc",
                "calibrated_ccc",
                "delta_ccc",
                "baseline_mae",
                "calibrated_mae",
                "delta_mae",
                "baseline_bias",
                "calibrated_bias",
                "delta_abs_bias",
                "mean_correction",
                "max_abs_correction",
                "mean_abs_applied_correction",
                "calibration_kind",
            ]
            if column in table.columns
        ]
        st.dataframe(
            table[display_columns] if display_columns else table,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Waiting for bioemu_cprime_chart_calibration_probe.parquet.")
    if calibrated is not None and not calibrated.empty:
        with st.expander("C' calibrated validation rows", expanded=False):
            st.dataframe(calibrated.head(800), width="stretch", hide_index=True)


def render_bioemu_nmr_chemical_shift_panels(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render BioEmu chemical-shift diagnostics without redirecting to Atlas."""

    paths = snapshot["paths"]
    predictions = _bioemu_prediction_source(snapshot)
    render_panel_header(
        st,
        "BioEmu Chemical-Shift Family Diagnostics",
        "Masked posterior means are analyzed directly from BioEmu latent NMR "
        "prediction parquet files. HN/C' remain the target guardrail, but N, "
        "CA, and CB are shown with the same CCC, shrinkage, bias, and "
        "calibration diagnostics.",
    )
    if predictions.empty:
        st.info("Waiting for bioemu_latent_nmr_predictions.parquet.")
        return
    _bioemu_show_family_metric_block(st, snapshot, key_prefix="bioemu_nmr")
    family_options = _bioemu_available_families(predictions)
    atom_family = st.selectbox(
        "BioEmu atom family",
        family_options,
        index=0,
        key="bioemu_nmr_atom_family",
    )
    static_name = _bioemu_static_scatter_name(atom_family)
    if static_name is not None:
        render_image_or_wait(
            st,
            paths.figures_dir / static_name,
            f"Static {atom_family} masked scatter",
        )
    figures = [
        ("scatter", _bioemu_family_scatter_figure(predictions, atom_family)),
        ("residual", _bioemu_residual_histogram_figure(predictions, atom_family)),
        ("sigma_error", _bioemu_sigma_error_figure(predictions, atom_family)),
    ]
    for figure_name, figure in figures:
        if figure is not None:
            st.plotly_chart(
                figure,
                width="stretch",
                key=f"bioemu_nmr_{_atom_family_slug(atom_family)}_{figure_name}",
            )

    worst = _bioemu_worst_entry_table(predictions, atom_family=atom_family)
    if not worst.empty:
        st.write(f"Worst entries for {atom_family}")
        st.dataframe(worst, width="stretch", hide_index=True)

    filtered = _bioemu_filtered_prediction_frame(predictions, atom_family)
    if not filtered.empty:
        with st.expander("Masked prediction rows", expanded=False):
            st.dataframe(filtered.head(800), width="stretch", hide_index=True)

    if atom_family in {"C'", "All"}:
        _bioemu_cprime_chart_calibration_block(st, snapshot)
    if atom_family == "All":
        _bioemu_family_calibration_strategy_block(st, snapshot)
        _bioemu_family_affine_calibration_block(st, snapshot)
        _bioemu_family_chart_calibration_block(st, snapshot)


def render_bioemu_moment_posterior_panels(
    st: Any,
    snapshot: dict[str, Any],
) -> None:
    """Render BioEmu moment-diffusion posterior diagnostics."""

    paths = snapshot["paths"]
    posterior_moments = snapshot.get("bioemu_posterior_moments")
    bridge_audit = snapshot.get("bioemu_posterior_bridge_audit")
    predictions = _bioemu_prediction_source(snapshot)
    _bioemu_show_family_metric_block(st, snapshot, key_prefix="bioemu_moment")
    moment_cols = st.columns(2)
    render_image_or_wait(
        moment_cols[0],
        paths.figures_dir / "bioemu_posterior_moment_calibration.png",
        "Posterior moment calibration",
    )
    render_image_or_wait(
        moment_cols[1],
        paths.figures_dir / "bioemu_posterior_bridge_vector_field.png",
        "Posterior bridge vector field",
    )
    if posterior_moments is not None and not posterior_moments.empty:
        chart_figure = _bioemu_chart_probability_figure(posterior_moments)
        if chart_figure is not None:
            st.plotly_chart(
                chart_figure,
                width="stretch",
                key="bioemu_moment_chart_probability",
            )
        summary_cols = [
            "atom_family",
            "chart_index",
            "chart_probability",
            "effective_chart_probability",
            "chart_reliability",
            "covariance_trace",
        ]
        available_cols = [
            column for column in summary_cols if column in posterior_moments.columns
        ]
        if {"atom_family", "chart_index"}.issubset(posterior_moments.columns):
            moment_summary = (
                posterior_moments.groupby(["atom_family", "chart_index"], dropna=False)
                .agg(
                    row_count=("chart_probability", "size"),
                    mean_chart_probability=("chart_probability", "mean"),
                    mean_effective_probability=(
                        "effective_chart_probability",
                        "mean",
                    ),
                    mean_reliability=("chart_reliability", "mean"),
                    mean_covariance_trace=("covariance_trace", "mean"),
                )
                .reset_index()
            )
            st.write("Moment chart summary by atom family")
            st.dataframe(moment_summary, width="stretch", hide_index=True)
        with st.expander("Posterior moment rows", expanded=False):
            st.dataframe(
                posterior_moments[available_cols].head(800)
                if available_cols
                else posterior_moments.head(800),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Waiting for bioemu_posterior_moments.parquet.")

    if bridge_audit is not None and not bridge_audit.empty:
        bridge_figure = _bioemu_bridge_family_figure(bridge_audit)
        if bridge_figure is not None:
            st.plotly_chart(
                bridge_figure,
                width="stretch",
                key="bioemu_moment_bridge_family",
            )
        bridge_cols = st.columns(4)
        for column, (label, field) in zip(
            bridge_cols,
            [
                ("Bridge norm", "bridge_norm"),
                ("Chart entropy", "chart_entropy"),
                ("Mean covariance", "mean_covariance_trace"),
                ("Reliability", "mean_reliability"),
            ],
            strict=False,
        ):
            if field in bridge_audit:
                column.metric(label, _display(float(bridge_audit[field].mean())))
        with st.expander("Posterior bridge audit rows", expanded=False):
            st.dataframe(bridge_audit.head(800), width="stretch", hide_index=True)
    else:
        st.info("Waiting for bioemu_posterior_bridge_audit.parquet.")

    if not predictions.empty:
        calibration_figure = _bioemu_sigma_error_figure(predictions, "All")
        if calibration_figure is not None:
            st.plotly_chart(
                calibration_figure,
                width="stretch",
                key="bioemu_moment_sigma_error_all",
            )

    _bioemu_family_calibration_strategy_block(st, snapshot)
    _bioemu_family_affine_calibration_block(st, snapshot)
    _bioemu_cprime_chart_calibration_block(st, snapshot)
    _bioemu_family_chart_calibration_block(st, snapshot)


def render_nmr_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render NMR-style chemical-shift uncertainty panels."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "NMR Chemical-Shift Uncertainty",
        "Posterior means, intervals, pseudo-HSQC panels, and UCBShift2.0 "
        "comparison are shown when artifacts become available.",
    )
    render_panel_analysis_png(
        st,
        snapshot,
        "bioemu_panel_nmr_chemical_shifts.png",
        "Panel-level NMR chemical-shift analysis",
    )
    if _is_bioemu_latent_snapshot(snapshot) and snapshot[
        "chemical_shift_posteriors"
    ].empty:
        render_bioemu_nmr_chemical_shift_panels(st, snapshot)
        st.info(
            "BioEmu latent atlas runs do not feed decoded structure/PDB features "
            "into the NMR head. This tab therefore reads posterior mean, "
            "uncertainty, and masked-row diagnostics from BioEmu prediction "
            "parquet files directly instead of redirecting to the Atlas panel."
        )
        return
    if _is_candidate_free_snapshot(snapshot) and snapshot[
        "chemical_shift_posteriors"
    ].empty:
        render_candidate_free_panels(st, snapshot)
        render_candidate_free_nmr_reuse(st, snapshot)
        st.info(
            "Full posterior-distribution artifacts are not produced by the "
            "candidate-free path yet. This tab therefore uses the candidate-free "
            "mean/sigma prediction tables as the canonical analysis."
        )
        with st.expander("Full posterior artifacts not produced by this run"):
            st.dataframe(
                [
                    {
                        "artifact": artifact,
                        "status": "not produced by current task",
                    }
                    for artifact in [
                        "reports/arrays/chemical_shift_posteriors.parquet",
                        "reports/arrays/chemical_shift_joint_posteriors.parquet",
                        "reports/figures/pseudo_hsqc_density.png",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        return
    st.dataframe(
        chemical_shift_legend_rows(),
        width="stretch",
        hide_index=True,
    )
    cols = st.columns(2)
    render_image_or_wait(
        cols[0],
        paths.figures_dir / "chemical_shift_violin_HN.png",
        "HN posterior uncertainty",
    )
    render_image_or_wait(
        cols[1],
        paths.figures_dir / "pseudo_hsqc_density.png",
        "Pseudo-HSQC density",
    )

    posterior_frame = snapshot["chemical_shift_posteriors"]
    if posterior_frame.empty:
        st.info("Waiting for chemical_shift_posteriors.parquet.")
        return

    atom_family = st.selectbox(
        "Atom family",
        (
            sorted(posterior_frame["atom_family"].dropna().astype(str).unique())
            if "atom_family" in posterior_frame.columns
            else ["HN"]
        ),
    )
    render_image_or_wait(
        st,
        paths.figures_dir
        / f"chemical_shift_violin_{_atom_family_slug(atom_family)}.png",
        f"{atom_family} posterior uncertainty",
    )
    figures = [
        chemical_shift_uncertainty_figure(posterior_frame, atom_family=atom_family),
        pseudo_hsqc_figure(posterior_frame),
        chemical_shift_scatter_figure(posterior_frame),
    ]
    for figure in figures:
        if figure is not None:
            st.plotly_chart(figure, width="stretch")

    coverage = coverage_table(posterior_frame)
    if not coverage.empty:
        st.write("Posterior interval calibration")
        st.dataframe(coverage, width="stretch", hide_index=True)

    calibration_report = snapshot.get("chemical_shift_calibration_report") or {}
    calibrated_frame = snapshot.get("chemical_shift_predictions_calibrated")
    render_panel_header(
        st,
        "Posterior Mean Calibration",
        "Atom-family affine calibration for posterior mean vs experimental "
        "chemical shifts.",
    )
    render_image_or_wait(
        st,
        paths.figures_dir / "posterior_mean_vs_experiment_calibrated.png",
        "Posterior mean vs experiment after affine calibration",
    )
    if calibration_report:
        st.json(calibration_report, expanded=False)
    else:
        st.info("Waiting for chemical_shift_calibration.json.")
    if calibrated_frame is not None and not calibrated_frame.empty:
        with st.expander("Calibrated prediction rows", expanded=False):
            st.dataframe(
                calibrated_frame.head(500),
                width="stretch",
                hide_index=True,
            )

    render_panel_header(
        st,
        "Multi-Atom Posterior",
        "Residue-level joint posterior diagnostics for paired or grouped "
        "chemical-shift evidence spaces.",
    )
    joint_cols = st.columns(2)
    render_image_or_wait(
        joint_cols[0],
        paths.figures_dir / "pseudo_cacb_density.png",
        "CA-CB posterior density",
    )
    render_image_or_wait(
        joint_cols[1],
        paths.figures_dir / "pseudo_ca_c_density.png",
        "CA-C' posterior density",
    )
    joint_frame = snapshot.get("chemical_shift_joint_posteriors")
    atom_sets = available_joint_atom_sets(joint_frame)
    if atom_sets:
        atom_set = st.selectbox("Joint atom set", atom_sets)
        figure = joint_posterior_parallel_figure(joint_frame, atom_set=atom_set)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        with st.expander("Joint posterior rows", expanded=False):
            st.dataframe(
                joint_frame.loc[joint_frame["atom_set"].astype(str) == atom_set].head(
                    400
                ),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Waiting for chemical_shift_joint_posteriors.parquet.")

    baseline_report = snapshot.get("chemical_shift_baseline_report") or {}
    with st.expander("UCBShift2.0 baseline report", expanded=False):
        st.json(baseline_report)


def render_raw_reports_tab(st: Any, snapshot: dict[str, Any]) -> None:
    """Render raw JSON report expanders for debugging."""

    render_viewer_capability_audit(st, snapshot, expanded=True)

    if _is_bioemu_latent_snapshot(snapshot):
        render_bioemu_run_audit(st, snapshot)

    for title, key in [
        ("summary.json", "summary"),
        ("benchmark_report.json", "benchmark_report"),
        ("observable_adapter_report.json", "observable_adapter_report"),
        (
            "multi_observable_support_ceiling_report.json",
            "multi_observable_support_report",
        ),
        ("best_preview_report.json", "best_preview_report"),
        ("moment_oracle_report.json", "moment_oracle_report"),
        ("moment_consistency_report.json", "moment_consistency_report"),
        ("candidate_free_report.json", "candidate_free_report"),
        ("bioemu_latent_nmr_report.json", "bioemu_latent_nmr_report"),
        ("chemical_shift_baseline_report.json", "chemical_shift_baseline_report"),
        ("uncertainty_report.json", "uncertainty_report"),
        ("UCBShift sidecar summary", "sidecar_summary"),
        ("Teacher materialization summary", "materialization_summary"),
    ]:
        with st.expander(title, expanded=False):
            payload = snapshot.get(key)
            if payload is None:
                st.info("Waiting for artifact.")
            else:
                st.code(
                    json.dumps(payload, indent=2, sort_keys=True),
                    language="json",
                )


def render_bioemu_run_audit(st: Any, snapshot: dict[str, Any]) -> None:
    """Render BioEmu artifact readiness, row counts, and run-level audit data."""

    paths = snapshot["paths"]
    render_panel_header(
        st,
        "BioEmu Run Audit",
        "Canonical BioEmu moment-atlas artifacts and row counts. This is the "
        "fast sanity layer before digging into metric plots.",
    )
    table_specs = [
        ("predictions", paths.bioemu_latent_nmr_predictions, "bioemu_latent_nmr_predictions"),
        (
            "masked_predictions",
            paths.bioemu_latent_nmr_masked_predictions,
            "bioemu_latent_nmr_masked_predictions",
        ),
        ("sample_weights", paths.bioemu_latent_sample_weights, "bioemu_latent_sample_weights"),
        (
            "atlas_coordinates",
            paths.bioemu_latent_atlas_coordinates,
            "bioemu_latent_atlas_coordinates",
        ),
        ("posterior_moments", paths.bioemu_posterior_moments, "bioemu_posterior_moments"),
        ("bridge_audit", paths.bioemu_posterior_bridge_audit, "bioemu_posterior_bridge_audit"),
        (
            "ucbshift_cnnls_teacher_weights",
            paths.bioemu_ucbshift_cnnls_teacher_weights,
            "bioemu_ucbshift_cnnls_teacher_weights",
        ),
        (
            "ucbshift_cnnls_teacher_predictions",
            paths.bioemu_ucbshift_cnnls_teacher_predictions,
            "bioemu_ucbshift_cnnls_teacher_predictions",
        ),
        (
            "ucbshift_cnnls_teacher_audit",
            paths.bioemu_ucbshift_cnnls_teacher_audit,
            "bioemu_ucbshift_cnnls_teacher_audit",
        ),
        (
            "ucbshift_cnnls_teacher_skipped",
            paths.bioemu_ucbshift_cnnls_teacher_skipped,
            "bioemu_ucbshift_cnnls_teacher_skipped",
        ),
        (
            "family_conflict_audit",
            paths.bioemu_family_conflict_audit,
            "bioemu_family_conflict_audit",
        ),
        (
            "physics_atlas_coordinates",
            paths.bioemu_physics_atlas_coordinates,
            "bioemu_physics_atlas_coordinates",
        ),
        (
            "family_tangent_adapter_audit",
            paths.bioemu_family_tangent_adapter_audit,
            "bioemu_family_tangent_adapter_audit",
        ),
        (
            "hn_cprime_direction_conflict_audit",
            paths.bioemu_hn_cprime_direction_conflict_audit,
            "bioemu_hn_cprime_direction_conflict_audit",
        ),
        (
            "ring_physics_audit",
            paths.bioemu_ring_physics_audit,
            "bioemu_ring_physics_audit",
        ),
        (
            "carbonyl_backbone_audit",
            paths.bioemu_carbonyl_backbone_audit,
            "bioemu_carbonyl_backbone_audit",
        ),
        (
            "mechanism_signed_direction_audit",
            paths.bioemu_mechanism_signed_direction_audit,
            "bioemu_mechanism_signed_direction_audit",
        ),
        (
            "family_chart_calibration_probe",
            paths.bioemu_family_chart_calibration_probe,
            "bioemu_family_chart_calibration_probe",
        ),
        (
            "family_chart_calibrated_predictions",
            paths.bioemu_family_chart_calibrated_predictions,
            "bioemu_family_chart_calibrated_predictions",
        ),
        (
            "family_affine_calibration_probe",
            paths.bioemu_family_affine_calibration_probe,
            "bioemu_family_affine_calibration_probe",
        ),
        (
            "family_affine_calibrated_predictions",
            paths.bioemu_family_affine_calibrated_predictions,
            "bioemu_family_affine_calibrated_predictions",
        ),
        (
            "family_calibration_strategy_probe",
            paths.bioemu_family_calibration_strategy_probe,
            "bioemu_family_calibration_strategy_probe",
        ),
        (
            "family_calibration_strategy_metrics",
            paths.bioemu_family_calibration_strategy_metrics,
            "bioemu_family_calibration_strategy_metrics",
        ),
        (
            "recommended_family_metrics",
            paths.bioemu_recommended_family_metrics,
            "bioemu_recommended_family_metrics",
        ),
        (
            "family_calibration_strategy_predictions",
            paths.bioemu_family_calibration_strategy_predictions,
            "bioemu_family_calibration_strategy_predictions",
        ),
        (
            "family_calibration_strategy_residue_audit",
            paths.bioemu_family_calibration_strategy_residue_audit,
            "bioemu_family_calibration_strategy_residue_audit",
        ),
        (
            "family_calibration_strategy_residue_type_audit",
            paths.bioemu_family_calibration_strategy_residue_type_audit,
            "bioemu_family_calibration_strategy_residue_type_audit",
        ),
        (
            "family_calibration_strategy_residue_type_guard_probe",
            paths.bioemu_family_calibration_strategy_residue_type_guard_probe,
            "bioemu_family_calibration_strategy_residue_type_guard_probe",
        ),
        (
            "family_calibration_strategy_residue_type_guard_predictions",
            paths.bioemu_family_calibration_strategy_residue_type_guard_predictions,
            "bioemu_family_calibration_strategy_residue_type_guard_predictions",
        ),
        (
            "proline_n_rare_regime_audit",
            paths.bioemu_proline_n_rare_regime_audit,
            "bioemu_proline_n_rare_regime_audit",
        ),
        (
            "proline_n_expert_probe",
            paths.bioemu_proline_n_expert_probe,
            "bioemu_proline_n_expert_probe",
        ),
        (
            "proline_n_expert_predictions",
            paths.bioemu_proline_n_expert_predictions,
            "bioemu_proline_n_expert_predictions",
        ),
        (
            "proline_n_mobility_expert_probe",
            paths.bioemu_proline_n_mobility_expert_probe,
            "bioemu_proline_n_mobility_expert_probe",
        ),
        (
            "proline_n_mobility_expert_split_guard",
            paths.bioemu_proline_n_mobility_expert_split_guard,
            "bioemu_proline_n_mobility_expert_split_guard",
        ),
        (
            "proline_n_mobility_expert_predictions",
            paths.bioemu_proline_n_mobility_expert_predictions,
            "bioemu_proline_n_mobility_expert_predictions",
        ),
        (
            "proline_n_mobility_entity_holdout",
            paths.bioemu_proline_n_mobility_entity_holdout,
            "bioemu_proline_n_mobility_entity_holdout",
        ),
        (
            "proline_n_mobility_entity_holdout_predictions",
            paths.bioemu_proline_n_mobility_entity_holdout_predictions,
            "bioemu_proline_n_mobility_entity_holdout_predictions",
        ),
        (
            "residue_family_expert_probe",
            paths.bioemu_residue_family_expert_probe,
            "bioemu_residue_family_expert_probe",
        ),
        (
            "residue_family_expert_predictions",
            paths.bioemu_residue_family_expert_predictions,
            "bioemu_residue_family_expert_predictions",
        ),
        (
            "residue_family_expert_policy_probe",
            paths.bioemu_residue_family_expert_policy_probe,
            "bioemu_residue_family_expert_policy_probe",
        ),
        (
            "residue_family_expert_policy_predictions",
            paths.bioemu_residue_family_expert_policy_predictions,
            "bioemu_residue_family_expert_policy_predictions",
        ),
        (
            "residue_family_expert_reliability_gate_probe",
            paths.bioemu_residue_family_expert_reliability_gate_probe,
            "bioemu_residue_family_expert_reliability_gate_probe",
        ),
        (
            "residue_family_expert_reliability_gate_predictions",
            paths.bioemu_residue_family_expert_reliability_gate_predictions,
            "bioemu_residue_family_expert_reliability_gate_predictions",
        ),
        (
            "residue_family_atlas_local_expert_probe",
            paths.bioemu_residue_family_atlas_local_expert_probe,
            "bioemu_residue_family_atlas_local_expert_probe",
        ),
        (
            "residue_family_atlas_local_expert_predictions",
            paths.bioemu_residue_family_atlas_local_expert_predictions,
            "bioemu_residue_family_atlas_local_expert_predictions",
        ),
        (
            "round_conformer_landscape_audit",
            paths.bioemu_round_conformer_landscape_audit,
            "bioemu_round_conformer_landscape_audit",
        ),
        (
            "detached_conformer_probe_manifest",
            paths.bioemu_detached_conformer_probe_manifest,
            "bioemu_detached_conformer_probe_manifest",
        ),
        (
            "detached_conformer_probe_inputs",
            paths.bioemu_detached_conformer_probe_inputs,
            "bioemu_detached_conformer_probe_inputs",
        ),
        (
            "detached_conformer_probe_output_audit",
            paths.bioemu_detached_conformer_probe_output_audit,
            "bioemu_detached_conformer_probe_output_audit",
        ),
        (
            "detached_conformer_ensemble_quality",
            paths.bioemu_detached_conformer_ensemble_quality,
            "bioemu_detached_conformer_ensemble_quality",
        ),
        (
            "detached_conformer_ensemble_pca",
            paths.bioemu_detached_conformer_ensemble_pca,
            "bioemu_detached_conformer_ensemble_pca",
        ),
        (
            "detached_support_expansion_action_plan",
            paths.bioemu_detached_support_expansion_action_plan,
            "bioemu_detached_support_expansion_action_plan",
        ),
        (
            "detached_conformer_nmr_coupling",
            paths.bioemu_detached_conformer_nmr_coupling,
            "bioemu_detached_conformer_nmr_coupling",
        ),
        (
            "detached_conformer_residue_shift_coupling",
            paths.bioemu_detached_conformer_residue_shift_coupling,
            "bioemu_detached_conformer_residue_shift_coupling",
        ),
        (
            "cprime_chart_calibration_probe",
            paths.bioemu_cprime_chart_calibration_probe,
            "bioemu_cprime_chart_calibration_probe",
        ),
        (
            "cprime_chart_calibrated_predictions",
            paths.bioemu_cprime_chart_calibrated_predictions,
            "bioemu_cprime_chart_calibrated_predictions",
        ),
    ]
    rows = []
    for label, path, key in table_specs:
        frame = snapshot.get(key)
        rows.append(
            {
                "artifact": label,
                "exists": path.exists(),
                "rows": 0 if frame is None or frame.empty else int(len(frame)),
                "path": str(path),
            }
        )
    png_paths = (
        sorted(paths.figures_dir.glob("*.png")) if paths.figures_dir.exists() else []
    )
    rows.append(
        {
            "artifact": "figures_png",
            "exists": bool(png_paths),
            "rows": len(png_paths),
            "path": str(paths.figures_dir),
        }
    )
    st.dataframe(rows, width="stretch", hide_index=True)
    metrics = _bioemu_family_metric_frame(_bioemu_prediction_source(snapshot))
    if not metrics.empty:
        st.write("Masked family metric audit")
        st.dataframe(metrics, width="stretch", hide_index=True)
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    if summary:
        compact_summary = {
            key: summary.get(key)
            for key in [
                "best_epoch",
                "best_metric",
                "train_examples",
                "val_examples",
                "config_path",
            ]
            if key in summary
        }
        if compact_summary:
            st.json(compact_summary, expanded=False)
    if png_paths:
        with st.expander("PNG inventory", expanded=False):
            st.dataframe(
                [
                    {
                        "file": path.name,
                        "size_kb": round(path.stat().st_size / 1024.0, 1),
                        "path": str(path),
                    }
                    for path in png_paths
                ],
                width="stretch",
                hide_index=True,
            )


def render_status_box(st: Any, title: str, payload: Any) -> None:
    """Render one compact status expander."""

    if not isinstance(payload, dict):
        st.markdown(
            _status_card_html(title, "waiting", "artifact not available yet", "muted"),
            unsafe_allow_html=True,
        )
        return
    status = payload.get("status") or payload.get("readiness_status") or "available"
    tone = _status_tone(status)
    caption = _status_caption(payload)
    st.markdown(
        _status_card_html(title, str(status), caption, tone),
        unsafe_allow_html=True,
    )
    with st.expander(f"{title} details", expanded=False):
        st.json(payload)


def render_metric_dict(st: Any, payload: Any) -> None:
    """Render a metric dictionary as compact cards when possible."""

    if not isinstance(payload, dict) or not payload:
        st.info("Waiting for metrics.")
        return
    rows = []
    for key, value in payload.items():
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if isinstance(value, (int, float)):
            rows.append({"metric": key, "value": float(value)})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.json(payload, expanded=False)


def render_image_or_wait(
    st_container: Any,
    path: Path,
    caption: str,
    *,
    show_path: bool = False,
) -> None:
    """Render an image artifact or a waiting state."""

    data_uri = _image_data_uri(path)
    if data_uri is None:
        st_container.info(f"Waiting for {path.name}")
        return
    path_html = (
        f"<div class='aemu-image-path'>{escape(str(path))}</div>" if show_path else ""
    )
    st_container.markdown(
        f"""
        <figure class="aemu-image-frame">
          <img src="{data_uri}" alt="{escape(caption)}" loading="eager" />
          <figcaption>{escape(caption)}</figcaption>
          {path_html}
        </figure>
        """,
        unsafe_allow_html=True,
    )


def render_native_image_or_wait(
    st_container: Any,
    path: Path,
    caption: str,
    *,
    show_path: bool = False,
) -> None:
    """Render an image with Streamlit's native image element."""

    if not path.exists() or not path.is_file():
        st_container.info(f"Waiting for {path.name}")
        return
    try:
        payload = path.read_bytes()
    except OSError:
        st_container.info(f"Waiting for {path.name}")
        return
    if not _is_complete_image(path, payload):
        st_container.info(f"Waiting for complete {path.name}")
        return
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = None
    if modified is not None:
        updated = pd.Timestamp.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S")
        caption = f"{caption} | updated={updated}"
    # Passing bytes instead of the filesystem path avoids stale browser/Streamlit
    # caching when a PNG is regenerated in-place with the same filename.
    st_container.image(payload, caption=caption, width="stretch")
    if show_path:
        st_container.code(str(path), language="text")


def _pretty_figure_caption(file_name: str) -> str:
    """Return a readable caption for a PNG artifact name."""

    return file_name.removesuffix(".png").replace("_", " ").title()


def _current_or_reference_figure(snapshot: dict[str, Any], path: Path) -> Path:
    """Return current figure path, falling back to initial-checkpoint figures."""

    if path.exists():
        return path
    reference_dir = _initial_checkpoint_figures_dir(snapshot)
    if reference_dir is None:
        return path
    reference_path = reference_dir / path.name
    return reference_path if reference_path.exists() else path


def _is_reference_figure(snapshot: dict[str, Any], path: Path) -> bool:
    """Return whether a figure came from the initial-checkpoint reference run."""

    reference_dir = _initial_checkpoint_figures_dir(snapshot)
    if reference_dir is None:
        return False
    try:
        return path.resolve().parent == reference_dir.resolve()
    except OSError:
        return False


def _initial_checkpoint_figures_dir(snapshot: dict[str, Any]) -> Path | None:
    """Resolve the reference figures directory from student_training_config.json."""

    paths = snapshot.get("paths")
    if paths is None:
        return None
    config_path = paths.run_dir / "student_training_config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint_path = config.get("initial_checkpoint_path")
    if not checkpoint_path:
        return None
    checkpoint = Path(str(checkpoint_path)).expanduser()
    run_dir = checkpoint.parent if checkpoint.name else checkpoint
    figures_dir = run_dir / "reports" / "figures"
    return figures_dir if figures_dir.exists() else None


def _nested_metrics(
    payload: Any,
    path: list[str],
    fallback_key: str,
) -> dict[str, Any]:
    """Read nested metrics with fallback to a top-level key."""

    if not isinstance(payload, dict):
        return {}
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            current = None
            break
        current = current[key]
    if isinstance(current, dict):
        return current
    fallback = payload.get(fallback_key)
    return fallback if isinstance(fallback, dict) else {}


def _dict_table(
    payload: dict[str, Any], key_name: str, value_name: str
) -> list[dict[str, Any]]:
    """Convert a dictionary to a two-column table."""

    return [
        {key_name: key, value_name: value}
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
    ]


def _learnability_table(payload: Any) -> Any:
    """Return one compact learnability table for monitor display."""
    if not isinstance(payload, dict) or not payload:
        return []
    rows = []
    for variant, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        row = {"variant": variant}
        for key in [
            "cs_family_ccc_macro",
            "cs_rmse_z_macro",
            "teacher_kl_macro",
            "teacher_top10_mass_overlap_macro",
            "cs_HN_ccc_macro",
            "cs_N_ccc_macro",
            "cs_CA_ccc_macro",
            "cs_CB_ccc_macro",
            "cs_C'_ccc_macro",
        ]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return rows


def _display(value: Any) -> str:
    """Return a compact display string for status cards."""

    if value is None:
        return "waiting"
    return str(value)


def latest_metric_value(
    metric_frame: Any,
    metric_name: str,
    *,
    split: str,
    aggregation: str = "macro",
) -> float | None:
    """Return the latest metric value for one live metric curve."""
    try:
        frame = metric_frame.loc[
            (metric_frame["metric_name"].astype(str) == metric_name)
            & (metric_frame["split"].astype(str) == split)
            & (metric_frame["aggregation"].astype(str) == aggregation)
        ]
    except Exception:
        return None
    if frame.empty:
        return None
    value = frame.sort_values("epoch").iloc[-1].get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _atom_family_slug(atom_family: str) -> str:
    """Return the static figure slug for one atom family."""

    return str(atom_family).replace("'", "prime").replace("/", "_")


def inject_theme(st: Any) -> None:
    """Inject lightweight CSS for the monitor UI."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(20, 36, 54, 0.10);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }
        .aemu-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 24px;
            background:
              radial-gradient(circle at 12% 20%, rgba(70,130,180,0.22), transparent 28%),
              radial-gradient(circle at 85% 15%, rgba(245,158,11,0.20), transparent 26%),
              linear-gradient(135deg, #0f172a 0%, #16324f 50%, #28506f 100%);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22);
            margin-bottom: 1rem;
        }
        .aemu-title {
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
        }
        .aemu-subtitle {
            color: rgba(248,250,252,0.78);
            max-width: 860px;
            line-height: 1.45;
        }
        .aemu-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.05rem;
        }
        .aemu-chip {
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.20);
            color: #f8fafc;
            font-size: 0.83rem;
        }
        .aemu-card {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(15,23,42,0.10);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
            min-height: 112px;
        }
        .aemu-card.blue { border-top: 4px solid #2f80ed; }
        .aemu-card.green { border-top: 4px solid #22a06b; }
        .aemu-card.amber { border-top: 4px solid #d97706; }
        .aemu-card.red { border-top: 4px solid #dc2626; }
        .aemu-card.muted { border-top: 4px solid #94a3b8; }
        .aemu-label {
            color: #64748b;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .aemu-value {
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 800;
            margin-top: 0.3rem;
            overflow-wrap: anywhere;
        }
        .aemu-caption {
            color: #64748b;
            font-size: 0.84rem;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .aemu-panel {
            padding: 0.95rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
            border: 1px solid rgba(15,23,42,0.08);
            margin: 0.15rem 0 1rem 0;
        }
        .aemu-panel-title {
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .aemu-panel-text {
            color: #475569;
            line-height: 1.45;
        }
        .aemu-image-frame {
            margin: 0;
            padding: 0.7rem;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.10);
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
            min-height: 260px;
            contain: layout paint;
        }
        .aemu-image-frame img {
            display: block;
            width: 100%;
            height: auto;
            max-height: 680px;
            object-fit: contain;
            border-radius: 12px;
            background: #f8fafc;
        }
        .aemu-image-frame figcaption {
            margin-top: 0.55rem;
            color: #475569;
            font-size: 0.88rem;
            text-align: center;
        }
        .aemu-image-path {
            margin-top: 0.35rem;
            color: #64748b;
            font-size: 0.72rem;
            text-align: center;
            overflow-wrap: anywhere;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(st: Any, snapshot: dict[str, Any], sidebar: dict[str, Any]) -> None:
    """Render the monitor hero section."""

    summary = snapshot["summary"] if isinstance(snapshot["summary"], dict) else {}
    job_status = snapshot.get("job_status") or {}
    current_epoch = _display(snapshot.get("current_epoch"))
    state = _display(job_status.get("state"))
    best_epoch = _display(summary.get("best_epoch"))
    run_dir = escape(str(sidebar.get("run_dir") or ""))
    html = f"""
    <div class="aemu-hero">
      <div class="aemu-title">AtypEmu Training Monitor</div>
      <div class="aemu-subtitle">
        Live, read-only dashboard for teacher-first training metrics,
        BioEmu-style ensemble landscapes, local conformation uncertainty,
        and NMR chemical-shift posterior uncertainty.
      </div>
      <div class="aemu-chip-row">
        <span class="aemu-chip">state: {escape(state)}</span>
        <span class="aemu-chip">epoch: {escape(current_epoch)}</span>
        <span class="aemu-chip">best epoch: {escape(best_epoch)}</span>
        <span class="aemu-chip">run: {run_dir}</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    st_container: Any,
    label: str,
    value: str,
    caption: str | None = None,
    tone: str = "muted",
) -> None:
    """Render one compact styled metric card."""

    st_container.markdown(
        f"""
        <div class="aemu-card {escape(tone)}">
          <div class="aemu-label">{escape(label)}</div>
          <div class="aemu-value">{escape(value)}</div>
          <div class="aemu-caption">{escape(caption or "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_header(st: Any, title: str, text: str) -> None:
    """Render a styled section header."""

    st.markdown(
        f"""
        <div class="aemu-panel">
          <div class="aemu-panel-title">{escape(title)}</div>
          <div class="aemu-panel-text">{escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_artifact_progress(st: Any, snapshot: dict[str, Any]) -> None:
    """Render compact artifact readiness progress."""

    paths = snapshot["paths"]
    if _is_bioemu_latent_snapshot(snapshot):
        checks = [
            ("history", paths.history.exists()),
            ("live epoch progress", paths.bioemu_live_epoch_progress.exists()),
            ("summary", paths.summary.exists()),
            ("BioEmu report", paths.bioemu_latent_nmr_report.exists()),
            ("BioEmu predictions", paths.bioemu_latent_nmr_predictions.exists()),
            (
                "BioEmu masked predictions",
                paths.bioemu_latent_nmr_masked_predictions.exists(),
            ),
            ("sample weights", paths.bioemu_latent_sample_weights.exists()),
            ("atlas coordinates", paths.bioemu_latent_atlas_coordinates.exists()),
            ("posterior moments", paths.bioemu_posterior_moments.exists()),
            ("bridge audit", paths.bioemu_posterior_bridge_audit.exists()),
            (
                "UCBShift2.0/CNNLS teacher weights",
                paths.bioemu_ucbshift_cnnls_teacher_weights.exists(),
            ),
            (
                "UCBShift2.0/CNNLS teacher audit",
                paths.bioemu_ucbshift_cnnls_teacher_audit.exists(),
            ),
            ("family conflict audit", paths.bioemu_family_conflict_audit.exists()),
            (
                "v36 physics atlas coordinates",
                paths.bioemu_physics_atlas_coordinates.exists(),
            ),
            (
                "v36 family tangent adapter audit",
                paths.bioemu_family_tangent_adapter_audit.exists(),
            ),
            (
                "v36 HN/C' direction audit",
                paths.bioemu_hn_cprime_direction_conflict_audit.exists(),
            ),
            (
                "v36 ring physics audit",
                paths.bioemu_ring_physics_audit.exists(),
            ),
            (
                "v36 carbonyl/backbone audit",
                paths.bioemu_carbonyl_backbone_audit.exists(),
            ),
            (
                "UCBShift factor atlas audit",
                paths.bioemu_ucbshift_factor_atlas_audit.exists()
                or paths.bioemu_ucbshift_factor_atlas_audit.with_suffix(".tsv").exists(),
            ),
            (
                "v101 mechanism-signed audit",
                paths.bioemu_mechanism_signed_direction_audit.exists()
                or paths.bioemu_mechanism_signed_direction_audit.with_suffix(".tsv").exists(),
            ),
            (
                "HN scatter",
                (paths.figures_dir / "bioemu_latent_hn_masked_scatter.png").exists(),
            ),
            (
                "C' scatter",
                (
                    paths.figures_dir / "bioemu_latent_cprime_masked_scatter.png"
                ).exists(),
            ),
            (
                "target progress",
                (
                    paths.figures_dir / "bioemu_latent_hn_cprime_target_progress.png"
                ).exists(),
            ),
            (
                "sample entropy",
                (paths.figures_dir / "bioemu_sample_weight_entropy.png").exists(),
            ),
            (
                "component persistence",
                (
                    paths.figures_dir / "bioemu_component_persistence_audit.png"
                ).exists(),
            ),
            (
                "family conflict",
                (
                    paths.figures_dir / "bioemu_family_conflict_audit.png"
                ).exists(),
            ),
            (
                "v36 HN/C' direction conflict",
                (
                    paths.figures_dir
                    / "bioemu_hn_cprime_direction_conflict_audit.png"
                ).exists(),
            ),
            (
                "v36 ring physics",
                (paths.figures_dir / "bioemu_ring_physics_audit.png").exists(),
            ),
            (
                "v36 carbonyl/backbone",
                (
                    paths.figures_dir / "bioemu_carbonyl_backbone_audit.png"
                ).exists(),
            ),
            (
                "UCBShift factor atlas",
                (
                    paths.figures_dir / "bioemu_ucbshift_factor_atlas_summary.png"
                ).exists(),
            ),
            (
                "v101 signed rare-regime",
                (
                    paths.figures_dir / "bioemu_mechanism_signed_direction_audit.png"
                ).exists(),
            ),
            (
                "family chart calibration",
                (
                    paths.figures_dir / "bioemu_family_chart_calibration_probe.png"
                ).exists(),
            ),
            (
                "family affine calibration",
                (
                    paths.figures_dir / "bioemu_family_affine_calibration_probe.png"
                ).exists(),
            ),
            (
                "family calibration strategy",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_probe.png"
                ).exists(),
            ),
            (
                "family calibration strategy progress",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_all_family_progress.png"
                ).exists(),
            ),
            (
                "family calibration strategy residue heatmap",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_entry_family_mae_heatmap.png"
                ).exists(),
            ),
            (
                "family calibration strategy worst residues",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_worst_residue_errors.png"
                ).exists(),
            ),
            (
                "family calibration strategy residue-type heatmap",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_residue_type_family_mae.png"
                ).exists(),
            ),
            (
                "family calibration strategy residue-type bottlenecks",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_residue_type_bottlenecks.png"
                ).exists(),
            ),
            (
                "family calibration strategy residue-type guard progress",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_residue_type_guard_progress.png"
                ).exists(),
            ),
            (
                "family calibration strategy residue-type guard bottlenecks",
                (
                    paths.figures_dir
                    / "bioemu_family_calibration_strategy_residue_type_guard_bottlenecks.png"
                ).exists(),
            ),
            (
                "PRO:N rare-regime scatter",
                (
                    paths.figures_dir / "bioemu_proline_n_rare_regime_scatter.png"
                ).exists(),
            ),
            (
                "PRO:N rare-regime residue errors",
                (
                    paths.figures_dir
                    / "bioemu_proline_n_rare_regime_residue_errors.png"
                ).exists(),
            ),
            (
                "PRO:N train-fitted expert probe",
                (paths.figures_dir / "bioemu_proline_n_expert_probe.png").exists(),
            ),
            (
                "PRO:N train-fitted expert residue errors",
                (
                    paths.figures_dir / "bioemu_proline_n_expert_residue_errors.png"
                ).exists(),
            ),
            (
                "PRO:N mobility/contact expert probe",
                (
                    paths.figures_dir
                    / "bioemu_proline_n_mobility_expert_probe.png"
                ).exists(),
            ),
            (
                "PRO:N mobility/contact split guard",
                (
                    paths.figures_dir
                    / "bioemu_proline_n_mobility_expert_split_guard.png"
                ).exists(),
            ),
            (
                "PRO:N mobility/contact entity holdout",
                (
                    paths.figures_dir
                    / "bioemu_proline_n_mobility_entity_holdout.png"
                ).exists(),
            ),
            (
                "residue-family expert top gains",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_top_gains.png"
                ).exists(),
            ),
            (
                "residue-family expert bottlenecks",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_bottlenecks.png"
                ).exists(),
            ),
            (
                "residue-family expert policy progress",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_policy_progress.png"
                ).exists(),
            ),
            (
                "residue-family expert policy actions",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_policy_actions.png"
                ).exists(),
            ),
            (
                "residue-family expert reliability gate progress",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_reliability_gate_progress.png"
                ).exists(),
            ),
            (
                "residue-family expert reliability gate actions",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_expert_reliability_gate_actions.png"
                ).exists(),
            ),
            (
                "residue-family atlas local expert progress",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_atlas_local_expert_progress.png"
                ).exists(),
            ),
            (
                "residue-family atlas local expert bottlenecks",
                (
                    paths.figures_dir
                    / "bioemu_residue_family_atlas_local_expert_bottlenecks.png"
                ).exists(),
            ),
            (
                "round conformer landscape audit",
                (
                    paths.figures_dir
                    / "bioemu_round_conformer_landscape_audit.png"
                ).exists(),
            ),
            (
                "detached conformer probe manifest",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_probe_manifest.png"
                ).exists(),
            ),
            (
                "detached conformer probe inputs",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_probe_inputs.png"
                ).exists(),
            ),
            (
                "detached conformer probe output audit",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_probe_output_audit.png"
                ).exists(),
            ),
            (
                "detached conformer ensemble quality",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_ensemble_quality.png"
                ).exists(),
            ),
            (
                "detached conformer NMR coupling",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_nmr_coupling.png"
                ).exists(),
            ),
            (
                "detached conformer residue-shift coupling",
                (
                    paths.figures_dir
                    / "bioemu_detached_conformer_residue_shift_coupling.png"
                ).exists(),
            ),
            (
                "C' chart calibration",
                (
                    paths.figures_dir / "bioemu_cprime_chart_calibration_probe.png"
                ).exists(),
            ),
            (
                "moment calibration",
                (
                    paths.figures_dir / "bioemu_posterior_moment_calibration.png"
                ).exists(),
            ),
        ]
        label = "BioEmu latent atlas artifact readiness"
    elif _is_candidate_free_snapshot(snapshot):
        checks = [
            ("history", paths.history.exists()),
            ("summary", paths.summary.exists()),
            ("candidate-free report", paths.candidate_free_report.exists()),
            (
                "candidate-free predictions",
                paths.candidate_free_predictions.exists(),
            ),
            (
                "masked predictions",
                paths.candidate_free_masked_predictions.exists(),
            ),
            (
                "HN scatter",
                (paths.figures_dir / "candidate_free_hn_masked_scatter.png").exists(),
            ),
            (
                "C' scatter",
                (
                    paths.figures_dir / "candidate_free_cprime_masked_scatter.png"
                ).exists(),
            ),
            (
                "target progress",
                (
                    paths.figures_dir / "candidate_free_hn_cprime_target_progress.png"
                ).exists(),
            ),
            ("candidate-free states", paths.candidate_free_state_tokens.exists()),
        ]
        label = "Candidate-free artifact readiness"
    else:
        checks = [
            ("history", paths.history.exists()),
            ("summary", paths.summary.exists()),
            ("benchmark", paths.benchmark_report.exists()),
            (
                "landscape",
                (paths.figures_dir / "landscape_teacher_vs_student.png").exists(),
            ),
            (
                "structure",
                (
                    paths.figures_dir / "ensemble_local_confidence_structure.png"
                ).exists(),
            ),
            ("rci adapter", paths.rci_vs_ensemble_profile.exists()),
            ("ensemble states", paths.ensemble_states.exists()),
            ("ccc geometry", paths.ccc_support_oracle.exists()),
            (
                "forward residual",
                paths.forward_residual_predictions.exists()
                or paths.residue_atom_forward_residuals.exists(),
            ),
            ("nmr posterior", paths.chemical_shift_posteriors.exists()),
            ("secondary shift", paths.secondary_shift_posteriors.exists()),
            ("joint cs posterior", paths.chemical_shift_joint_posteriors.exists()),
        ]
        label = "Artifact readiness"
    ready = sum(1 for _, ok in checks if ok)
    fraction = ready / max(len(checks), 1)
    st.progress(fraction, text=f"{label}: {ready}/{len(checks)} available")
    with st.expander("Artifact checklist", expanded=False):
        st.dataframe(
            [{"artifact": name, "available": ok} for name, ok in checks],
            width="stretch",
            hide_index=True,
        )


def _status_card_html(title: str, status: str, caption: str, tone: str) -> str:
    """Return HTML for a status card."""

    return f"""
    <div class="aemu-card {escape(tone)}">
      <div class="aemu-label">{escape(title)}</div>
      <div class="aemu-value">{escape(status)}</div>
      <div class="aemu-caption">{escape(caption)}</div>
    </div>
    """


def _status_caption(payload: dict[str, Any]) -> str:
    """Return a compact status caption from a JSON payload."""

    for key in [
        "written_examples",
        "written_candidates",
        "baseline_rows",
        "failed_examples",
        "generated_examples",
    ]:
        if key in payload:
            return f"{key}={payload[key]}"
    return "details available"


def _status_tone(status: Any) -> str:
    """Map status text to card tone."""

    value = "" if status is None else str(status).lower()
    if any(token in value for token in ["run", "ok", "ready", "complete", "available"]):
        return "green"
    if any(token in value for token in ["pend", "wait", "partial"]):
        return "amber"
    if any(token in value for token in ["fail", "error", "cancel", "timeout"]):
        return "red"
    return "muted"


def _image_data_uri(path: Path) -> str | None:
    """Return an inline data URI for a complete static image artifact."""

    if not path.exists() or not path.is_file():
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if not _is_complete_image(path, payload):
        return None
    suffix = path.suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix)
    if mime_type is None:
        return None
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_complete_image(path: Path, payload: bytes) -> bool:
    """Return whether a static image looks complete enough to render."""

    if not payload:
        return False
    suffix = path.suffix.lower()
    if suffix == ".png":
        return payload.startswith(b"\x89PNG\r\n\x1a\n") and b"IEND" in payload[-32:]
    if suffix in {".jpg", ".jpeg"}:
        return payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    if suffix == ".svg":
        stripped = payload.strip()
        return stripped.startswith((b"<svg", b"<?xml")) and b"</svg>" in stripped[-256:]
    return True


if __name__ == "__main__":
    main()
