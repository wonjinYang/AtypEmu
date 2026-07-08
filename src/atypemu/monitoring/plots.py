"""Plot helpers for the AtypEmu Streamlit training monitor."""

from __future__ import annotations

import math
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ENSEMBLE_LOCAL_CONFIDENCE_LABEL = "AtypEmu ensemble local confidence"
DEFAULT_METRICS = [
    "loss",
    "ccc_loss",
    "aux_family_ccc_loss",
    "family_ccc_target_loss",
    "ucbshift_cnnls_teacher_mean_loss",
    "cprime_guardrail",
    "bridge_norm_loss",
    "masked_HN_Cprime_family_cprime_balanced_ccc",
    "masked_family_ccc",
    "masked_HN_ccc",
    "masked_C'_ccc",
    "posterior_ess",
    "posterior_bridge_norm",
    "total_loss",
    "weight_kl",
    "chemical_shift_loss",
    "j_coupling_loss",
    "noe_loss",
    "observable_oracle_kl",
    "moment_reconstruction_family_ccc_loss",
    "moment_masked_holdout_family_ccc_loss",
    "moment_oracle_distillation",
    "moment_sample_consistency",
    "hn_variance_calibration",
    "cprime_robust_likelihood",
    "state_entropy_floor",
    "state_repulsion",
    "basin_coverage",
    "teacher_kl_macro",
    "teacher_js_macro",
    "cs_masked_holdout_family_ccc",
    "cs_masked_holdout_whitened_family_ccc",
    "cs_reconstruction_family_ccc",
    "cs_rmse_z_macro",
    "noe_violation_rate_macro",
]


def plotly_available() -> bool:
    """Return whether Plotly can be imported."""

    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        return False
    return True


def py3dmol_available() -> bool:
    """Return whether py3Dmol can be imported."""

    try:
        import py3Dmol  # noqa: F401
    except ImportError:
        return False
    return True


def metric_curve_figure(
    metric_frame: pd.DataFrame,
    metric_names: list[str] | None = None,
) -> Any | None:
    """Build a TensorBoard-style metric curve figure."""

    if metric_frame.empty or not plotly_available():
        return None
    import plotly.express as px

    selected = metric_frame.copy()
    if metric_names:
        selected = selected.loc[selected["metric_name"].isin(metric_names)].copy()
    if selected.empty:
        return None
    selected["series"] = (
        selected["split"].astype(str)
        + " / "
        + selected["metric_name"].astype(str)
        + " / "
        + selected["aggregation"].astype(str)
    )
    figure = px.line(
        selected,
        x="epoch",
        y="value",
        color="series",
        markers=True,
        hover_data=[
            "eligible_examples",
            "eligible_measurements",
            "tier",
            "source",
        ],
    )
    figure.update_layout(
        title="Training Metrics",
        xaxis_title="Epoch",
        yaxis_title="Value",
        legend_title="Metric",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def projection_scatter_figure(
    projection_frame: pd.DataFrame,
    color_by: str = "student_minus_teacher",
) -> Any | None:
    """Build an interactive teacher/student landscape scatter plot."""

    required = {"projection_x", "projection_y"}
    if projection_frame.empty or not required.issubset(projection_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = projection_frame.copy()
    predicted = _numeric_column(frame, "predicted_weight")
    teacher = _numeric_column(frame, "teacher_weight")
    frame["student_minus_teacher"] = predicted - teacher
    if not _is_numeric_with_values(frame, color_by):
        color_by = "student_minus_teacher"
    size_column = (
        "predicted_weight"
        if "predicted_weight" in frame.columns
        else "teacher_weight" if "teacher_weight" in frame.columns else None
    )
    color_scale = {
        "student_minus_teacher": "RdBu",
        "teacher_score": "Viridis",
        "teacher_energy": "Cividis",
        "cs_mae_ppm": "Magma",
        "cs_rmse_z": "Magma",
        "rci_mismatch": "Plasma",
    }.get(color_by, "Viridis")
    figure = px.scatter(
        frame,
        x="projection_x",
        y="projection_y",
        color=color_by,
        size=size_column,
        hover_data=[
            column
            for column in [
                "entity_uid",
                "candidate_id",
                "teacher_weight",
                "predicted_weight",
                "teacher_score",
                "teacher_energy",
                "cs_mae_ppm",
                "cs_rmse_z",
                "rci_mismatch",
            ]
            if column in frame.columns
        ],
        color_continuous_scale=color_scale,
    )
    figure.update_layout(
        title=f"Shared Weighted-PCA Landscape Coordinates ({color_by})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def ensemble_state_scatter_figure(
    assignment_frame: pd.DataFrame,
) -> Any | None:
    """Build an interactive landscape state scatter plot."""

    required = {"projection_x", "projection_y", "state_id"}
    if assignment_frame.empty or not required.issubset(assignment_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = assignment_frame.copy()
    size_column = "predicted_weight" if "predicted_weight" in frame.columns else None
    hover_columns = [
        column
        for column in [
            "entity_uid",
            "candidate_id",
            "state_id",
            "state_rank",
            "predicted_weight",
            "teacher_weight",
            "cs_mae_ppm",
            "cs_rmse_z",
            "rci_mismatch",
            "is_state_representative",
        ]
        if column in frame.columns
    ]
    figure = px.scatter(
        frame,
        x="projection_x",
        y="projection_y",
        color="state_id",
        size=size_column,
        symbol=(
            "is_state_representative"
            if "is_state_representative" in frame.columns
            else None
        ),
        hover_data=hover_columns,
    )
    figure.update_layout(
        title="Energy Landscape State Assignments",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="State",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def ensemble_state_mass_figure(state_frame: pd.DataFrame) -> Any | None:
    """Build an interactive state mass and free-energy figure."""

    required = {"state_id", "state_mass", "relative_free_energy_kbt"}
    if state_frame.empty or not required.issubset(state_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.graph_objects as go

    frame = state_frame.sort_values("state_rank").copy()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["state_id"],
            y=frame["state_mass"],
            name="State mass",
            marker_color="#4c78a8",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["state_id"],
            y=frame["relative_free_energy_kbt"],
            name="Relative free energy",
            yaxis="y2",
            mode="lines+markers",
            line={"color": "#f58518"},
        )
    )
    figure.update_layout(
        title="State Occupancy and Relative Free Energy",
        xaxis_title="State",
        yaxis_title="State mass",
        yaxis2={
            "title": "Relative ΔG / kBT",
            "overlaying": "y",
            "side": "right",
        },
        legend_title="Legend",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def ccc_oracle_vs_model_figure(oracle_frame: pd.DataFrame) -> Any | None:
    """Build an interactive CCC oracle-versus-model scatter."""

    if oracle_frame.empty or not plotly_available():
        return None
    import plotly.express as px

    val = oracle_frame.loc[oracle_frame.get("split", "").astype(str) == "val"].copy()
    if val.empty:
        val = oracle_frame.copy()
    pivot = val.pivot_table(
        index="entity_uid",
        columns="variant",
        values="cs_family_ccc",
        aggfunc="max",
    ).reset_index()
    required = {"model", "projected_simplex_ccc_oracle"}
    if not required.issubset(pivot.columns):
        return None
    pivot["model_gap"] = pivot["projected_simplex_ccc_oracle"] - pivot["model"]
    figure = px.scatter(
        pivot,
        x="model",
        y="projected_simplex_ccc_oracle",
        color="model_gap",
        hover_data=["entity_uid", "model_gap"],
        color_continuous_scale="Magma",
    )
    low = float(min(pivot["model"].min(), pivot["projected_simplex_ccc_oracle"].min()))
    high = float(max(pivot["model"].max(), pivot["projected_simplex_ccc_oracle"].max()))
    figure.add_shape(
        type="line",
        x0=low,
        y0=low,
        x1=high,
        y1=high,
        line={"dash": "dash", "color": "gray"},
    )
    figure.update_layout(
        title="CCC Support Oracle vs Model",
        xaxis_title="Model CS family CCC",
        yaxis_title="Projected-simplex oracle CCC",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def ccc_family_gap_figure(oracle_frame: pd.DataFrame) -> Any | None:
    """Build a bar chart of oracle-model CCC gaps by atom family."""

    if oracle_frame.empty or not plotly_available():
        return None
    import plotly.express as px

    val = oracle_frame.loc[oracle_frame.get("split", "").astype(str) == "val"].copy()
    if val.empty:
        val = oracle_frame.copy()
    model = val.loc[val["variant"].astype(str) == "model"]
    oracle = val.loc[val["variant"].astype(str) == "projected_simplex_ccc_oracle"]
    rows = []
    for family in ["HN", "N", "CA", "CB", "C'"]:
        column = f"cs_{family}_ccc"
        if column not in model.columns or column not in oracle.columns:
            continue
        rows.append(
            {
                "atom_family": family,
                "model_ccc": float(
                    pd.to_numeric(model[column], errors="coerce").mean()
                ),
                "oracle_ccc": float(
                    pd.to_numeric(oracle[column], errors="coerce").mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame["oracle_minus_model"] = frame["oracle_ccc"] - frame["model_ccc"]
    figure = px.bar(
        frame,
        x="atom_family",
        y="oracle_minus_model",
        hover_data=["model_ccc", "oracle_ccc"],
        color="oracle_minus_model",
        color_continuous_scale="Oranges",
    )
    figure.update_layout(
        title="CCC Support Gap by Atom Family",
        xaxis_title="Atom family",
        yaxis_title="Oracle - model CCC",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def available_ensemble_state_ids(state_frame: pd.DataFrame) -> list[str]:
    """Return available representative state ids for one run."""

    if state_frame.empty or "state_id" not in state_frame.columns:
        return []
    if "state_rank" in state_frame.columns:
        frame = state_frame.sort_values("state_rank")
    else:
        frame = state_frame
    return frame["state_id"].dropna().astype(str).unique().tolist()


def chemical_shift_uncertainty_figure(
    posterior_frame: pd.DataFrame,
    atom_family: str = "HN",
    max_rows: int = 160,
) -> Any | None:
    """Build a residue-wise chemical-shift uncertainty panel."""

    if posterior_frame.empty or not plotly_available():
        return None
    import plotly.graph_objects as go

    frame = posterior_frame.loc[
        posterior_frame.get("atom_family", pd.Series(dtype=str)).astype(str)
        == atom_family
    ].copy()
    if frame.empty:
        return None
    frame["residue_index"] = _target_id_series(frame).map(_residue_index)
    frame = frame.dropna(subset=["residue_index", "predicted_value"]).head(max_rows)
    if frame.empty:
        return None

    figure = go.Figure()
    sorted_frame = frame.sort_values("residue_index")
    for index, (_, row) in enumerate(sorted_frame.iterrows()):
        x_value = float(row["residue_index"])
        mean = float(row["predicted_value"])
        q05 = _as_float(row.get("posterior_q05"), mean)
        q25 = _as_float(row.get("posterior_q25"), mean)
        q75 = _as_float(row.get("posterior_q75"), mean)
        q95 = _as_float(row.get("posterior_q95"), mean)
        figure.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[q05, q95],
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.25)", "width": 5},
                name="90% posterior interval (q05-q95)",
                legendgroup="interval90",
                showlegend=index == 0,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[q25, q75],
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.70)", "width": 9},
                name="50% posterior interval (q25-q75)",
                legendgroup="interval50",
                showlegend=index == 0,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frame["residue_index"],
            y=frame["predicted_value"],
            mode="markers",
            name="Posterior mean",
            marker={"color": "#1f77b4", "size": 6},
        )
    )
    observed = frame.dropna(subset=["target_value"])
    if not observed.empty:
        figure.add_trace(
            go.Scatter(
                x=observed["residue_index"],
                y=observed["target_value"],
                mode="markers",
                name="Experimental",
                marker={"color": "#dd8452", "size": 6, "symbol": "x"},
            )
        )
    figure.update_layout(
        title=f"{atom_family} Chemical-Shift Posterior Uncertainty",
        xaxis_title="Residue index",
        yaxis_title=f"{atom_family} chemical shift (ppm)",
        legend_title="Legend",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def pseudo_hsqc_figure(posterior_frame: pd.DataFrame) -> Any | None:
    """Build a lightweight pseudo-HSQC uncertainty scatter."""

    if posterior_frame.empty or not plotly_available():
        return None
    import plotly.graph_objects as go

    paired = paired_hsqc_frame(posterior_frame)
    if paired.empty:
        return None
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=paired["h_mean"],
            y=paired["n_mean"],
            error_x={"array": paired["h_std"], "visible": True},
            error_y={"array": paired["n_std"], "visible": True},
            mode="markers",
            name="Posterior mean",
            marker={"color": "#1f77b4", "size": 7},
            text=paired["residue_index"].astype(str),
        )
    )
    observed = paired.dropna(subset=["exp_h", "exp_n"])
    if not observed.empty:
        figure.add_trace(
            go.Scatter(
                x=observed["exp_h"],
                y=observed["exp_n"],
                mode="markers",
                name="Experimental",
                marker={"color": "#dd8452", "size": 7, "symbol": "x"},
            )
        )
    figure.update_layout(
        title="Pseudo-HSQC 1H-15N Posterior Panel",
        xaxis_title="1H (ppm)",
        yaxis_title="15N (ppm)",
        xaxis={"autorange": "reversed"},
        yaxis={"autorange": "reversed"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def chemical_shift_scatter_figure(posterior_frame: pd.DataFrame) -> Any | None:
    """Build posterior mean versus experimental chemical-shift scatter."""

    if posterior_frame.empty or not plotly_available():
        return None
    import plotly.express as px

    frame = posterior_frame.dropna(subset=["predicted_value", "target_value"]).copy()
    if frame.empty:
        return None
    figure = px.scatter(
        frame,
        x="target_value",
        y="predicted_value",
        color="atom_family" if "atom_family" in frame.columns else None,
        hover_data=[
            column
            for column in ["entity_uid", "target_id", "posterior_std"]
            if column in frame.columns
        ],
    )
    lower = float(min(frame["target_value"].min(), frame["predicted_value"].min()))
    upper = float(max(frame["target_value"].max(), frame["predicted_value"].max()))
    figure.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"dash": "dash", "color": "gray"},
    )
    figure.update_layout(
        title="Posterior Mean vs Experimental Chemical Shift",
        xaxis_title="Experimental (ppm)",
        yaxis_title="Posterior mean (ppm)",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def coverage_table(posterior_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute interval coverage rows for chemical-shift posteriors."""

    if posterior_frame.empty:
        return pd.DataFrame()
    if "target_value" not in posterior_frame.columns:
        return pd.DataFrame()
    frame = posterior_frame.dropna(subset=["target_value"]).copy()
    rows: list[dict[str, Any]] = []
    for posterior_kind, prefix in [("raw", ""), ("calibrated", "calibrated_")]:
        for nominal, low_col, high_col in [
            (0.50, f"{prefix}posterior_q25", f"{prefix}posterior_q75"),
            (0.80, f"{prefix}posterior_q10", f"{prefix}posterior_q90"),
            (0.95, f"{prefix}posterior_q05", f"{prefix}posterior_q95"),
        ]:
            if low_col not in frame.columns or high_col not in frame.columns:
                continue
            eligible = frame.dropna(subset=[low_col, high_col, "target_value"])
            if eligible.empty:
                continue
            covered = (eligible["target_value"] >= eligible[low_col]) & (
                eligible["target_value"] <= eligible[high_col]
            )
            width = eligible[high_col] - eligible[low_col]
            rows.append(
                {
                    "posterior_kind": posterior_kind,
                    "nominal_coverage": nominal,
                    "empirical_coverage": float(covered.mean()),
                    "mean_interval_width": float(width.mean()),
                    "eligible_measurements": int(len(eligible)),
                }
            )
    return pd.DataFrame(rows)


def paired_hsqc_frame(posterior_frame: pd.DataFrame) -> pd.DataFrame:
    """Return paired HN/N rows for pseudo-HSQC rendering."""

    if posterior_frame.empty:
        return pd.DataFrame()
    frame = posterior_frame.copy()
    frame["residue_index"] = _target_id_series(frame).map(_residue_index)
    frame = frame.dropna(subset=["residue_index"])
    keys = ["entity_uid", "residue_index"]
    h_frame = _select_atom_family(frame, "HN", keys, prefix="h")
    n_frame = _select_atom_family(frame, "N", keys, prefix="n")
    if h_frame.empty or n_frame.empty:
        return pd.DataFrame()
    paired = h_frame.merge(n_frame, on=keys, how="inner")
    return paired.sort_values(keys).reset_index(drop=True)


def available_joint_atom_sets(joint_frame: pd.DataFrame) -> list[str]:
    """Return available multi-atom posterior atom sets."""

    if joint_frame.empty or "atom_set" not in joint_frame.columns:
        return []
    return sorted(joint_frame["atom_set"].dropna().astype(str).unique().tolist())


def joint_posterior_parallel_figure(
    joint_frame: pd.DataFrame,
    atom_set: str,
    max_rows: int = 120,
) -> Any | None:
    """Build an interactive multi-atom posterior profile panel."""

    if joint_frame.empty or not plotly_available():
        return None
    import plotly.express as px

    subset = joint_frame.loc[joint_frame["atom_set"].astype(str) == atom_set].copy()
    if subset.empty:
        return None
    rows: list[dict[str, Any]] = []
    for _, row in subset.head(max_rows).iterrows():
        try:
            atom_families = json.loads(row["atom_families_json"])
            means = json.loads(row["posterior_mean_vector_json"])
            targets = json.loads(row["target_vector_json"])
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            continue
        residue_label = (
            f"{row.get('entity_uid', '')}:"
            f"{row.get('chain_id', '')}:"
            f"{row.get('residue_index', '')}"
        )
        for atom_family, mean_value, target_value in zip(
            atom_families,
            means,
            targets,
            strict=False,
        ):
            rows.append(
                {
                    "residue": residue_label,
                    "atom_family": atom_family,
                    "value": float(mean_value),
                    "kind": "posterior_mean",
                    "joint_nll": row.get("joint_nll"),
                    "mahalanobis_distance": row.get("mahalanobis_distance"),
                }
            )
            rows.append(
                {
                    "residue": residue_label,
                    "atom_family": atom_family,
                    "value": float(target_value),
                    "kind": "experimental",
                    "joint_nll": row.get("joint_nll"),
                    "mahalanobis_distance": row.get("mahalanobis_distance"),
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    figure = px.line(
        frame,
        x="atom_family",
        y="value",
        color="residue",
        line_dash="kind",
        hover_data=["joint_nll", "mahalanobis_distance"],
    )
    figure.update_layout(
        title=f"{atom_set} Joint Chemical-Shift Posterior",
        xaxis_title="Atom family",
        yaxis_title="Chemical shift (ppm)",
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def observable_support_table(report: dict[str, Any]) -> pd.DataFrame:
    """Return a monitor-ready multi-observable support table."""

    channels = report.get("channels", {}) if isinstance(report, dict) else {}
    rows = []
    if isinstance(channels, dict):
        for observable, payload in channels.items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "observable": observable,
                    "status": payload.get("status"),
                    "adapter_type": payload.get("adapter_type"),
                    "metric_name": payload.get("metric_name"),
                    "examples": payload.get("examples"),
                    "eligible_measurements": payload.get("eligible_measurements"),
                    "model_score_macro": payload.get("model_score_macro"),
                    "oracle_score_macro": payload.get("oracle_score_macro"),
                    "oracle_minus_model_macro": payload.get("oracle_minus_model_macro"),
                    "observable_conflict_score": payload.get(
                        "observable_conflict_score"
                    ),
                }
            )
    saxs = report.get("saxs") if isinstance(report, dict) else None
    if isinstance(saxs, dict):
        rows.append(
            {
                "observable": "saxs",
                "status": saxs.get("status"),
                "adapter_type": saxs.get("adapter_type"),
                "metric_name": None,
                "examples": 0,
                "eligible_measurements": 0,
                "model_score_macro": None,
                "oracle_score_macro": None,
                "oracle_minus_model_macro": None,
                "observable_conflict_score": None,
            }
        )
    return pd.DataFrame(rows)


def observable_support_figure(report: dict[str, Any]) -> Any | None:
    """Build a model-vs-oracle support ceiling chart by observable."""

    table = observable_support_table(report)
    if table.empty or not plotly_available():
        return None
    import plotly.express as px

    long = table.melt(
        id_vars=["observable", "metric_name"],
        value_vars=[
            column
            for column in ["model_score_macro", "oracle_score_macro"]
            if column in table.columns
        ],
        var_name="series",
        value_name="score",
    )
    long["score"] = pd.to_numeric(long["score"], errors="coerce")
    long = long.dropna(subset=["score"])
    if long.empty:
        return None
    figure = px.bar(
        long,
        x="observable",
        y="score",
        color="series",
        barmode="group",
        hover_data=["metric_name"],
    )
    figure.update_layout(
        title="Multi-Observable Adapter Support",
        xaxis_title="Observable",
        yaxis_title="Higher-is-better score",
        legend_title="Series",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def observable_conflict_figure(summary_frame: pd.DataFrame) -> Any | None:
    """Build an observable conflict score chart from adapter summary rows."""

    if summary_frame.empty or "observable_conflict_score" not in summary_frame.columns:
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = summary_frame.copy()
    frame["observable_conflict_score"] = pd.to_numeric(
        frame["observable_conflict_score"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["observable_conflict_score"])
    if frame.empty:
        return None
    grouped = (
        frame.groupby(["split", "observable"], dropna=False)[
            "observable_conflict_score"
        ]
        .mean()
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x="observable",
        y="observable_conflict_score",
        color="split",
        barmode="group",
    )
    figure.update_layout(
        title="Observable Conflict Score",
        xaxis_title="Observable",
        yaxis_title="Mean pairwise JS divergence",
        legend_title="Split",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def moment_oracle_table(report: dict[str, Any]) -> pd.DataFrame:
    """Return split-level MomentHead/oracle summary rows."""
    splits = report.get("splits", {}) if isinstance(report, dict) else {}
    rows = []
    if isinstance(splits, dict):
        for split, payload in splits.items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "split": split,
                    "examples": payload.get("examples"),
                    "oracle_family_ccc_macro": payload.get("oracle_family_ccc_macro"),
                    "moment_family_ccc_macro": payload.get("moment_family_ccc_macro"),
                    "sample_family_ccc_macro": payload.get("sample_family_ccc_macro"),
                    "support_gap_macro": payload.get("support_gap_macro"),
                }
            )
    return pd.DataFrame(rows)


def moment_target_figure(moment_frame: pd.DataFrame) -> Any | None:
    """Build a MomentHead-vs-experimental chemical-shift scatter figure."""
    required = {"target_value", "moment_predicted_value"}
    if moment_frame.empty or not required.issubset(moment_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = moment_frame.copy()
    figure = px.scatter(
        frame,
        x="target_value",
        y="moment_predicted_value",
        color="atom_family" if "atom_family" in frame.columns else None,
        facet_col="split" if "split" in frame.columns else None,
        hover_data=[
            column
            for column in [
                "entity_uid",
                "target_id",
                "sample_mean_value",
                "oracle_mean_value",
                "moment_sample_abs_error",
                "moment_oracle_abs_error",
            ]
            if column in frame.columns
        ],
    )
    figure.update_layout(
        title="Fast MomentHead vs Experimental Chemical Shift",
        xaxis_title="Experimental chemical shift",
        yaxis_title="Fast moment prediction",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def sample_support_figure(moment_frame: pd.DataFrame) -> Any | None:
    """Build a candidate-sample support diagnostic against experiment."""
    required = {"target_value", "sample_mean_value"}
    if moment_frame.empty or not required.issubset(moment_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = moment_frame.copy()
    figure = px.scatter(
        frame,
        x="target_value",
        y="sample_mean_value",
        color="atom_family" if "atom_family" in frame.columns else None,
        facet_col="split" if "split" in frame.columns else None,
        hover_data=[
            column
            for column in [
                "entity_uid",
                "target_id",
                "moment_predicted_value",
                "oracle_mean_value",
                "moment_sample_abs_error",
            ]
            if column in frame.columns
        ],
    )
    figure.update_layout(
        title="Candidate-Sample Weighted Mean vs Experimental Chemical Shift",
        xaxis_title="Experimental chemical shift",
        yaxis_title="Candidate-sample weighted mean",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def moment_consistency_figure(moment_frame: pd.DataFrame) -> Any | None:
    """Build a MomentHead-vs-candidate-sample diagnostic figure."""
    required = {"sample_mean_value", "moment_predicted_value"}
    if moment_frame.empty or not required.issubset(moment_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = moment_frame.copy()
    figure = px.scatter(
        frame,
        x="sample_mean_value",
        y="moment_predicted_value",
        color="atom_family" if "atom_family" in frame.columns else None,
        facet_col="atom_family" if "atom_family" in frame.columns else None,
        hover_data=[
            column
            for column in [
                "entity_uid",
                "split",
                "target_id",
                "target_value",
                "oracle_mean_value",
                "moment_sample_abs_error",
                "moment_oracle_abs_error",
            ]
            if column in frame.columns
        ],
    )
    figure.update_layout(
        title="MomentHead vs Candidate-Sample Mean Diagnostic",
        xaxis_title="Candidate-sample weighted mean",
        yaxis_title="Fast moment prediction",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def posterior_state_occupancy_figure(state_frame: pd.DataFrame) -> Any | None:
    """Build oracle-vs-model posterior state occupancy bars."""
    required = {"state_id", "state_mass", "oracle_state_mass"}
    if state_frame.empty or not required.issubset(state_frame.columns):
        return None
    if not plotly_available():
        return None
    import plotly.express as px

    frame = state_frame.copy()
    if "entity_uid" in frame.columns and not frame.empty:
        first_entity = str(frame["entity_uid"].iloc[0])
        frame = frame.loc[frame["entity_uid"].astype(str) == first_entity].copy()
    if "state_rank" in frame.columns:
        frame = frame.sort_values("state_rank")
    long = frame.melt(
        id_vars=[
            column
            for column in ["entity_uid", "state_id", "state_rank", "split"]
            if column in frame.columns
        ],
        value_vars=["state_mass", "oracle_state_mass"],
        var_name="series",
        value_name="mass",
    )
    figure = px.bar(
        long,
        x="state_id",
        y="mass",
        color="series",
        barmode="group",
        hover_data=[column for column in ["entity_uid", "split"] if column in long],
    )
    figure.update_layout(
        title="Posterior State Occupancy",
        xaxis_title="State",
        yaxis_title="Mass",
        legend_title="Series",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def build_structure_viewer_html(pdb_path: str | Path) -> str | None:
    """Build an interactive annotated-structure viewer HTML payload."""

    path = Path(pdb_path)
    if not path.exists() or not py3dmol_available():
        return None
    import py3Dmol

    pdb_text = path.read_text()
    view = py3Dmol.view(width=850, height=520)
    view.addModel(pdb_text, "pdb")
    view.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 0,
                    "max": 100,
                }
            }
        }
    )
    view.zoomTo()
    return view._make_html()


def confidence_legend_rows() -> pd.DataFrame:
    """Return display bins for local ensemble confidence."""

    return pd.DataFrame(
        [
            {"bin": ">90", "interpretation": "Very consistent local geometry"},
            {"bin": "70-90", "interpretation": "Mostly consistent"},
            {"bin": "50-70", "interpretation": "Heterogeneous local ensemble"},
            {"bin": "<50", "interpretation": "Highly flexible or multi-state"},
        ]
    )


def rci_legend_rows() -> pd.DataFrame:
    """Return display legend rows for RCI adapter panels."""

    return pd.DataFrame(
        [
            {
                "visual": "Orange line / points",
                "quantity": "RCI adapter flexibility",
                "scale": "0-1",
                "interpretation": "Higher means NMR-shift-derived flexibility is higher.",
            },
            {
                "visual": "Blue line",
                "quantity": ENSEMBLE_LOCAL_CONFIDENCE_LABEL,
                "scale": "0-100",
                "interpretation": "Higher means local geometry is more consistent in the ensemble.",
            },
            {
                "visual": "Blue scatter y-axis",
                "quantity": "Ensemble RMSF / residue variance",
                "scale": "Å / variance proxy",
                "interpretation": "Higher means larger conformational spread across candidates.",
            },
        ]
    )


def chemical_shift_legend_rows() -> pd.DataFrame:
    """Return display legend rows for chemical-shift posterior panels."""

    return pd.DataFrame(
        [
            {
                "visual": "Blue violin / filled density",
                "quantity": "Candidate-support posterior density",
                "interpretation": "Approximate ensemble-weighted distribution for one residue and atom family.",
            },
            {
                "visual": "Blue dot / line center",
                "quantity": "Posterior mean",
                "interpretation": "Ensemble-weighted expected chemical shift used for point metrics.",
            },
            {
                "visual": "Dark blue interval",
                "quantity": "50% posterior interval",
                "interpretation": "Interquartile interval, q25-q75.",
            },
            {
                "visual": "Light blue interval",
                "quantity": "90% posterior interval",
                "interpretation": "Outer interval, q05-q95.",
            },
            {
                "visual": "Orange x / marker",
                "quantity": "Experimental chemical shift",
                "interpretation": "Observed BMRB target value when available.",
            },
        ]
    )


def available_default_metrics(metric_frame: pd.DataFrame) -> list[str]:
    """Return default monitor metrics that are present in the run history."""

    if metric_frame.empty:
        return DEFAULT_METRICS
    present = set(metric_frame["metric_name"].astype(str))
    selected = [name for name in DEFAULT_METRICS if name in present]
    return selected or sorted(present)[:8]


def available_landscape_color_columns(projection_frame: pd.DataFrame) -> list[str]:
    """Return landscape overlay columns that are usable for interactive coloring."""

    required = {"projection_x", "projection_y"}
    if projection_frame.empty or not required.issubset(projection_frame.columns):
        return []
    frame = projection_frame.copy()
    predicted = _numeric_column(frame, "predicted_weight")
    teacher = _numeric_column(frame, "teacher_weight")
    frame["student_minus_teacher"] = predicted - teacher
    candidates = [
        "student_minus_teacher",
        "teacher_score",
        "teacher_energy",
        "cs_mae_ppm",
        "cs_rmse_z",
        "rci_mismatch",
    ]
    return [column for column in candidates if _is_numeric_with_values(frame, column)]


def _select_atom_family(
    frame: pd.DataFrame,
    atom_family: str,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Select one atom family and rename columns for paired plots."""

    if "atom_family" not in frame.columns:
        return pd.DataFrame()
    subset = frame.loc[frame["atom_family"].astype(str) == atom_family].copy()
    if subset.empty:
        return pd.DataFrame()
    columns = keys + ["predicted_value", "posterior_std", "target_value"]
    subset = subset[[column for column in columns if column in subset.columns]].copy()
    rename = {
        "predicted_value": f"{prefix}_mean",
        "posterior_std": f"{prefix}_std",
        "target_value": f"exp_{prefix}",
    }
    return subset.rename(columns=rename)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column or zeros when it is absent."""

    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _is_numeric_with_values(frame: pd.DataFrame, column: str) -> bool:
    """Return whether a column can drive a numeric color scale."""

    if column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().any())


def _target_id_series(frame: pd.DataFrame) -> pd.Series:
    """Return target identifiers as a Series."""

    if "target_id" not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame["target_id"].astype(str)


def _residue_index(value: Any) -> float:
    """Extract a residue index from target identifiers used by training."""

    if value is None:
        return math.nan
    matches = re.findall(r"[-+]?\d+", str(value))
    if not matches:
        return math.nan
    try:
        return float(matches[-1])
    except ValueError:
        return math.nan


def _as_float(value: Any, default: float) -> float:
    """Convert a value to finite float with fallback."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric
