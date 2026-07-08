"""Reporting helpers for AtypEmu training metrics and benchmarks."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atypemu.datasets import IntegratedDataRegistry, MetaDataFrameRegistry
from atypemu.training.config import StudentTrainingConfig


def build_benchmark_report(
    data_root: str | Path,
    integrated_root: str | Path,
    train_epoch: dict[str, Any],
    val_epoch: dict[str, Any],
    checkpoint_metric: dict[str, Any],
) -> dict[str, Any]:
    """Build one source-oriented benchmark dashboard report."""
    data_root_path = Path(data_root)
    _ = Path(integrated_root)

    integrated_registry = IntegratedDataRegistry.from_data_root(data_root_path)
    meta_registry = MetaDataFrameRegistry.from_data_root(data_root_path)

    teacher_examples = integrated_registry.load_teacher_examples()
    observable_supervision = integrated_registry.load_observable_supervision()
    benchmark_entries = integrated_registry.load_benchmark_entries()
    meta_entities = meta_registry.load_entity_index()
    meta_tags = meta_registry.load_table("tags")

    source_scorecards = {
        "bmrb": _build_bmrb_scorecard(train_epoch, val_epoch),
        "ped": _build_ped_scorecard(benchmark_entries, teacher_examples),
        "sasbdb": _build_sasbdb_scorecard(observable_supervision),
        "mfib": _build_mfib_scorecard(meta_entities, meta_tags),
        "fuzdb": _build_fuzdb_scorecard(meta_entities, meta_tags),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_metric": checkpoint_metric,
        "source_scorecards": source_scorecards,
    }


def build_chemical_shift_baseline_report(
    prediction_rows: list[dict[str, Any]],
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Build one chemical-shift baseline comparison report."""
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_name": "UCBSHIFT2",
        "panels": {},
    }
    prediction_frame = pd.DataFrame(prediction_rows)
    if prediction_frame.empty:
        report["panels"]["AtypEmu_holdout_vs_ucbshift2"] = {
            "status": "no_chemical_shift_predictions",
        }
        report["panels"]["UCBSHIFT2_reference_corpus"] = {
            "status": "no_chemical_shift_predictions",
        }
        return report

    if not config.chemical_shift_baseline_path:
        report["panels"]["AtypEmu_holdout_vs_ucbshift2"] = {
            "status": "missing_baseline",
        }
        report["panels"]["UCBSHIFT2_reference_corpus"] = {
            "status": "missing_baseline",
        }
        return report

    baseline_path = Path(config.chemical_shift_baseline_path)
    if not baseline_path.exists():
        report["panels"]["AtypEmu_holdout_vs_ucbshift2"] = {
            "status": "missing_baseline",
            "path": str(baseline_path),
        }
        report["panels"]["UCBSHIFT2_reference_corpus"] = {
            "status": "missing_baseline",
            "path": str(baseline_path),
        }
        return report

    baseline_frame = _normalize_baseline_frame(_load_table_file(baseline_path))
    merged = prediction_frame.merge(
        baseline_frame,
        on=["entity_uid", "target_id"],
        how="inner",
    )
    report["panels"]["AtypEmu_holdout_vs_ucbshift2"] = _baseline_panel(merged)

    reference_frame = _resolve_reference_panel_frame(
        merged=merged,
        baseline_frame=baseline_frame,
        config=config,
    )
    if reference_frame is None:
        report["panels"]["UCBSHIFT2_reference_corpus"] = {
            "status": "missing_reference_corpus_manifest",
        }
    else:
        report["panels"]["UCBSHIFT2_reference_corpus"] = _baseline_panel(
            reference_frame
        )
    return report


def build_chemical_shift_calibration_artifacts(
    train_prediction_rows: list[dict[str, Any]],
    val_prediction_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit atom-family affine CS calibration and write report artifacts."""
    output_dir_path = Path(output_dir)
    arrays_dir = output_dir_path / "reports" / "arrays"
    figures_dir = output_dir_path / "reports" / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.DataFrame(train_prediction_rows)
    val_frame = pd.DataFrame(val_prediction_rows)
    if train_frame.empty or val_frame.empty:
        return {
            "status": "no_prediction_rows",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    calibration = _fit_atom_family_affine_calibration(train_frame)
    calibrated_train = _apply_atom_family_affine_calibration(
        train_frame,
        calibration,
    )
    calibrated_val = _apply_atom_family_affine_calibration(val_frame, calibration)
    calibrated_frame = pd.concat(
        [calibrated_train, calibrated_val],
        ignore_index=True,
        sort=False,
    )
    calibrated_frame.to_parquet(
        arrays_dir / "chemical_shift_predictions_calibrated.parquet",
        index=False,
    )
    _render_calibrated_shift_scatter(
        calibrated_val,
        figures_dir / "posterior_mean_vs_experiment_calibrated.png",
    )
    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": calibration,
        "metrics": {
            "train": _calibrated_shift_metrics(calibrated_train),
            "val": _calibrated_shift_metrics(calibrated_val),
        },
        "artifacts": {
            "predictions": "reports/arrays/chemical_shift_predictions_calibrated.parquet",
            "scatter": "reports/figures/posterior_mean_vs_experiment_calibrated.png",
        },
    }


def _fit_atom_family_affine_calibration(
    frame: pd.DataFrame,
    ridge: float = 1e-3,
) -> dict[str, dict[str, float | int]]:
    """Fit one ridge-stabilized affine calibration per atom family."""
    calibration: dict[str, dict[str, float | int]] = {}
    for atom_family, subset in frame.groupby("atom_family", dropna=True):
        clean = subset.dropna(subset=["predicted_value", "target_value"]).copy()
        if len(clean) < 2:
            calibration[str(atom_family)] = {
                "slope": 1.0,
                "intercept": 0.0,
                "training_measurements": int(len(clean)),
                "status": "insufficient_data",
            }
            continue
        x = clean["predicted_value"].astype(float).to_numpy()
        y = clean["target_value"].astype(float).to_numpy()
        design = np.column_stack([x, np.ones_like(x)])
        penalty = np.diag([ridge, 0.0])
        target = design.T @ y + np.asarray([ridge, 0.0], dtype=np.float64)
        try:
            slope, intercept = np.linalg.solve(design.T @ design + penalty, target)
        except np.linalg.LinAlgError:
            slope, intercept = 1.0, 0.0
        calibration[str(atom_family)] = {
            "slope": _json_float_or_none(slope) or 1.0,
            "intercept": _json_float_or_none(intercept) or 0.0,
            "training_measurements": int(len(clean)),
            "status": "ok",
        }
    return calibration


def _apply_atom_family_affine_calibration(
    frame: pd.DataFrame,
    calibration: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    """Apply atom-family affine calibration to one prediction table."""
    calibrated = frame.copy()
    calibrated["calibrated_predicted_value"] = calibrated["predicted_value"]
    calibrated["calibration_slope"] = 1.0
    calibrated["calibration_intercept"] = 0.0
    for atom_family, params in calibration.items():
        mask = calibrated["atom_family"].astype(str) == str(atom_family)
        slope = float(params.get("slope", 1.0))
        intercept = float(params.get("intercept", 0.0))
        calibrated.loc[mask, "calibrated_predicted_value"] = (
            slope * pd.to_numeric(calibrated.loc[mask, "predicted_value"]) + intercept
        )
        calibrated.loc[mask, "calibration_slope"] = slope
        calibrated.loc[mask, "calibration_intercept"] = intercept
    return calibrated


def _calibrated_shift_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return raw and calibrated shift metrics by atom family."""
    if frame.empty:
        return {"status": "empty"}
    panels: dict[str, Any] = {"overall": _calibrated_shift_metric_panel(frame)}
    for atom_family, subset in frame.groupby("atom_family", dropna=True):
        panels[str(atom_family)] = _calibrated_shift_metric_panel(subset)
    return {"status": "ok", "panels": panels}


def _calibrated_shift_metric_panel(
    frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    """Return one raw/calibrated metric panel."""
    clean = frame.dropna(
        subset=["predicted_value", "calibrated_predicted_value", "target_value"]
    )
    if clean.empty:
        return {"measurements": 0}
    target = clean["target_value"].astype(float)
    raw = clean["predicted_value"].astype(float)
    calibrated = clean["calibrated_predicted_value"].astype(float)
    raw_abs = (raw - target).abs()
    calibrated_abs = (calibrated - target).abs()
    return {
        "measurements": int(len(clean)),
        "raw_mae_ppm": _json_float_or_none(raw_abs.mean()),
        "calibrated_mae_ppm": _json_float_or_none(calibrated_abs.mean()),
        "delta_mae_ppm": _json_float_or_none(calibrated_abs.mean() - raw_abs.mean()),
        "raw_ccc": _safe_ccc(raw.to_numpy(), target.to_numpy()),
        "calibrated_ccc": _safe_ccc(calibrated.to_numpy(), target.to_numpy()),
    }


def _render_calibrated_shift_scatter(frame: pd.DataFrame, path: Path) -> None:
    """Render raw vs calibrated posterior mean scatter for validation rows."""
    if frame.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    clean = frame.dropna(
        subset=["predicted_value", "calibrated_predicted_value", "target_value"]
    )
    if clean.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, column, title in [
        (axes[0], "predicted_value", "Raw posterior mean"),
        (axes[1], "calibrated_predicted_value", "Affine calibrated mean"),
    ]:
        axis.scatter(clean["target_value"], clean[column], s=8, alpha=0.45)
        lower = float(min(clean["target_value"].min(), clean[column].min()))
        upper = float(max(clean["target_value"].max(), clean[column].max()))
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set_xlabel("Experimental chemical shift")
        axis.set_ylabel("Predicted chemical shift")
        axis.set_title(title)
    fig.suptitle("Posterior mean vs experimental chemical shift")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _safe_ccc(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return Lin's concordance correlation coefficient with safe fallback."""
    if left.size < 2:
        return None
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    left_var = float(np.mean(np.square(left - left_mean)))
    right_var = float(np.mean(np.square(right - right_mean)))
    covariance = float(np.mean((left - left_mean) * (right - right_mean)))
    denominator = left_var + right_var + (left_mean - right_mean) ** 2
    if denominator <= 1e-12:
        return None
    return _json_float_or_none((2.0 * covariance) / denominator)


def _json_float_or_none(value: Any) -> float | None:
    """Return a finite JSON float or ``None``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _build_bmrb_scorecard(
    train_epoch: dict[str, Any],
    val_epoch: dict[str, Any],
) -> dict[str, Any]:
    """Build the BMRB scorecard from evaluated metric payloads."""
    return {
        "status": "evaluated",
        "train_metrics": dict(train_epoch.get("metric_summary", {})),
        "val_metrics": dict(val_epoch.get("metric_summary", {})),
        "train_eligible_counts": dict(train_epoch.get("eligible_counts", {})),
        "val_eligible_counts": dict(val_epoch.get("eligible_counts", {})),
    }


def _build_ped_scorecard(
    benchmark_entries: pd.DataFrame,
    teacher_examples: pd.DataFrame,
) -> dict[str, Any]:
    """Build the PED coverage and leakage-safe holdout dashboard."""
    ped = benchmark_entries.loc[benchmark_entries["source_id"] == "ped"].copy()
    train_entities = set(
        teacher_examples.loc[teacher_examples["split"] == "train", "entity_uid"]
        .astype(str)
        .tolist()
    )
    linked_lists = (
        ped["linked_bmrb_entity_uids"]
        .fillna("")
        .astype(str)
        .apply(lambda value: [token for token in value.split("|") if token])
    )
    holdout_safe_mask = linked_lists.apply(
        lambda tokens: not bool(set(tokens) & train_entities)
    )
    return {
        "status": "coverage_only",
        "total_entries": int(len(ped)),
        "linked_bmrb_entries": int((ped["linked_bmrb_count"] > 0).sum()),
        "holdout_safe_entries": int(holdout_safe_mask.sum()),
        "training_policy_counts": ped["training_policy"].value_counts().to_dict(),
        "evaluation_track_counts": ped["evaluation_track"].value_counts().to_dict(),
    }


def _build_sasbdb_scorecard(observable_supervision: pd.DataFrame) -> dict[str, Any]:
    """Build the SASBDB dashboard placeholder and supervision counts."""
    sasbdb = observable_supervision.loc[
        observable_supervision["source_id"] == "sasbdb"
    ].copy()
    counts = (
        sasbdb.groupby("supervision_kind")["measurement_count"].sum().to_dict()
        if not sasbdb.empty
        else {}
    )
    return {
        "status": "awaiting_external_adapter",
        "available_measurement_counts": counts,
        "metrics": {
            "rg_guinier_mae": None,
            "rg_pddf_mae": None,
            "dmax_mae": None,
            "pddf_w1": None,
            "intensity_log_chi2": None,
        },
    }


def _build_mfib_scorecard(
    meta_entities: pd.DataFrame,
    meta_tags: pd.DataFrame,
) -> dict[str, Any]:
    """Build the MFIB slice dashboard."""
    entities = meta_entities.loc[meta_entities["source_id"] == "mfib"].copy()
    tags = meta_tags.loc[meta_tags["source_id"] == "mfib"].copy()
    return {
        "status": "slice_dashboard",
        "total_entries": int(len(entities)),
        "pair_count_bins": _bin_series_counts(
            entities["geometry_pair_count"],
            [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, None, "4+")],
        ),
        "max_interface_area_bins": _bin_series_counts(
            entities["geometry_max_interface_area"],
            [
                (0.0, 500.0, "<=500"),
                (500.0, 1000.0, "500-1000"),
                (1000.0, 2000.0, "1000-2000"),
                (2000.0, None, "2000+"),
            ],
        ),
        "max_contact_count_bins": _bin_series_counts(
            entities["geometry_max_contact_count"],
            [
                (0, 25, "<=25"),
                (26, 100, "26-100"),
                (101, 250, "101-250"),
                (251, None, "251+"),
            ],
        ),
        "evidence_level_counts": tags.loc[tags["namespace"] == "evidence_level", "tag"]
        .value_counts()
        .to_dict(),
    }


def _build_fuzdb_scorecard(
    meta_entities: pd.DataFrame,
    meta_tags: pd.DataFrame,
) -> dict[str, Any]:
    """Build the FuzDB context and mechanism slice dashboard."""
    entities = meta_entities.loc[meta_entities["source_id"] == "fuzdb"].copy()
    tags = meta_tags.loc[meta_tags["source_id"] == "fuzdb"].copy()
    return {
        "status": "slice_dashboard",
        "total_entries": int(len(entities)),
        "topology_class_counts": tags.loc[tags["namespace"] == "topology_class", "tag"]
        .value_counts()
        .to_dict(),
        "mechanism_category_counts": tags.loc[
            tags["namespace"] == "mechanism_category", "tag"
        ]
        .value_counts()
        .to_dict(),
        "detection_method_counts": tags.loc[
            tags["namespace"] == "detection_method", "tag"
        ]
        .value_counts()
        .to_dict(),
        "llps_role_counts": tags.loc[tags["namespace"] == "llps_role", "tag"]
        .value_counts()
        .to_dict(),
    }


def _load_table_file(path: Path) -> pd.DataFrame:
    """Load one tabular file with a suffix-based reader."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix == ".json":
        payload = json.loads(path.read_text())
        return pd.DataFrame(payload)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported baseline file format: {path}")


def _normalize_baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one external chemical-shift baseline table."""
    baseline = frame.copy()
    rename_map: dict[str, str] = {}
    for candidate in ["prediction", "predicted_value", "chemical_shift_prediction"]:
        if candidate in baseline.columns:
            rename_map[candidate] = "baseline_predicted_value"
            break
    if "target_uid" in baseline.columns:
        rename_map["target_uid"] = "target_id"
    if "bmrb_entity_uid" in baseline.columns:
        rename_map["bmrb_entity_uid"] = "entity_uid"
    baseline = baseline.rename(columns=rename_map)
    if "entity_uid" not in baseline.columns and "bmrb_id" in baseline.columns:
        baseline["entity_uid"] = "bmrb:" + baseline["bmrb_id"].astype(str)
    if "baseline_predicted_value" not in baseline.columns:
        raise ValueError("Baseline table must provide a predicted value column.")
    if "target_id" not in baseline.columns:
        raise ValueError("Baseline table must provide a target identifier column.")
    columns = ["entity_uid", "target_id", "baseline_predicted_value"]
    if "corpus_panel" in baseline.columns:
        columns.append("corpus_panel")
    return baseline.loc[:, columns].copy()


def _resolve_reference_panel_frame(
    merged: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    config: StudentTrainingConfig,
) -> pd.DataFrame | None:
    """Resolve the reference-corpus panel frame when possible."""
    if "corpus_panel" in baseline_frame.columns:
        panel = merged.loc[merged["corpus_panel"] == "ucbshift2_reference_corpus"]
        if not panel.empty:
            return panel.copy()
    if not config.chemical_shift_reference_corpus_path:
        return None
    manifest_path = Path(config.chemical_shift_reference_corpus_path)
    if not manifest_path.exists():
        return None
    manifest = _load_table_file(manifest_path)
    if "entity_uid" not in manifest.columns and "bmrb_id" in manifest.columns:
        manifest["entity_uid"] = "bmrb:" + manifest["bmrb_id"].astype(str)
    if "entity_uid" not in manifest.columns:
        return None
    allowed = set(manifest["entity_uid"].astype(str).tolist())
    panel = merged.loc[merged["entity_uid"].astype(str).isin(allowed)].copy()
    return panel if not panel.empty else None


def _baseline_panel(frame: pd.DataFrame) -> dict[str, Any]:
    """Build one baseline comparison panel from merged prediction rows."""
    if frame.empty:
        return {"status": "no_overlap"}

    atypemu_residual = frame["predicted_value"] - frame["target_value"]
    baseline_residual = frame["baseline_predicted_value"] - frame["target_value"]
    example_stats = (
        frame.assign(
            atypemu_abs=atypemu_residual.abs(),
            baseline_abs=baseline_residual.abs(),
        )
        .groupby("entity_uid")[["atypemu_abs", "baseline_abs"]]
        .mean()
    )
    atom_family_delta: dict[str, float] = {}
    if "atom_family" in frame.columns:
        for atom_family, atom_frame in frame.groupby("atom_family"):
            if pd.isna(atom_family):
                continue
            atom_family_delta[str(atom_family)] = float(
                (atom_frame["predicted_value"] - atom_frame["target_value"])
                .abs()
                .mean()
                - (atom_frame["baseline_predicted_value"] - atom_frame["target_value"])
                .abs()
                .mean()
            )
    return {
        "status": "ok",
        "matched_examples": int(frame["entity_uid"].nunique()),
        "matched_measurements": int(len(frame)),
        "atypemu_mae_ppm": float(atypemu_residual.abs().mean()),
        "baseline_mae_ppm": float(baseline_residual.abs().mean()),
        "delta_mae_ppm": float(
            atypemu_residual.abs().mean() - baseline_residual.abs().mean()
        ),
        "atypemu_rmse_ppm": float((atypemu_residual.pow(2).mean()) ** 0.5),
        "baseline_rmse_ppm": float((baseline_residual.pow(2).mean()) ** 0.5),
        "delta_rmse_ppm": float(
            (atypemu_residual.pow(2).mean() ** 0.5)
            - (baseline_residual.pow(2).mean() ** 0.5)
        ),
        "win_rate_by_example": float(
            (example_stats["atypemu_abs"] < example_stats["baseline_abs"]).mean()
        ),
        "atom_family_delta_mae_ppm": atom_family_delta,
    }


def _bin_series_counts(
    series: pd.Series,
    bins: list[tuple[float | int, float | int | None, str]],
) -> dict[str, int]:
    """Count values in inclusive numeric bins."""
    counts: dict[str, int] = {}
    numeric = pd.to_numeric(series, errors="coerce")
    for lower, upper, label in bins:
        if upper is None:
            mask = numeric >= lower
        else:
            mask = (numeric >= lower) & (numeric <= upper)
        counts[label] = int(mask.fillna(False).sum())
    return counts
