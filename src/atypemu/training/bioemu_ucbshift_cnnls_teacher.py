"""Build BioEmu UCBShift2.0/CS-reweighting population teachers.

The BioEmu latent-atlas training path can predict chemical shifts without using
decoded structure features in its forward pass.  This module adds the offline
teacher path that *does* use decoded BioEmu conformers: sample an ensemble,
run UCBShift2.0 for each conformer, fit constrained non-negative population
weights against experimental chemical shifts, and write those weights as a
teacher for later BioEmu posterior/fine-tuning runs.  The canonical artifact
stem keeps the historical ``cnnls`` name for downstream compatibility, but the
default fitting method is now the CS-reweighting paper-style atom offset
correction followed by MaxEnt/BFGS reweighting.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import logsumexp

from atypemu.adapters.cnnls_adapter import load_candidate_shift_file
from atypemu.reweighting import EuclideanSimplexReweighter, MaxEntReweighter
from atypemu.energy import MultiObservablePosteriorEnergy
from atypemu.types import (
    CandidatePool,
    CandidateRecord,
    EnergyBreakdown,
    NMRTargetBundle,
    ObservableMatrix,
    ObservableBundle,
    WeightSolution,
    canonical_atom_name,
)


CANONICAL_ARTIFACT_STEM = "bioemu_ucbshift_cnnls_teacher"
PAPER_CS_REWEIGHTING_ATOMS = ("H", "N", "CA", "CB")
CS_REWEIGHTING_ALIASES = {
    "cs_reweighting",
    "cs-reweighting",
    "csreweighting",
    "cs_rw",
    "csrw",
    "paper_cs_reweighting",
    "paper-cs-reweighting",
    "cs_reweighting_paper",
    "cs-reweighting-paper",
    "csrw_paper",
}
LEGACY_CS_REWEIGHTING_ALIASES = {
    "legacy_cs_reweighting",
    "legacy-cs-reweighting",
    "cs_reweighting_legacy",
    "cs-reweighting-legacy",
    "csrw_legacy",
}
CS_REWEIGHTING_ATOM_WEIGHTS = {
    "H": 4.0,
    "N": 1.0,
    "CA": 1.0,
    "CB": 1.0,
    "C": 1.0,
}
CS_REWEIGHTING_SCALING = {
    "H": 0.31,
    "N": 2.62,
    "CA": 1.22,
    "CB": 1.10,
    "C": 1.00,
}
CS_REWEIGHTING_OUTLIER_Z = 3.0
PAPER_CS_REWEIGHTING_OFFSET_ITERATIONS = 3
PAPER_CS_REWEIGHTING_OFFSET_GRID_POINTS = 9
PAPER_CS_REWEIGHTING_MAX_SELECTED_CONFORMERS = 300
RESIDUE_ALIGNMENT_MAX_ABS_OFFSET = 30
STUDENT_BME_ALIASES = {
    "student_bme",
    "student-bme",
    "student",
    "student_t",
    "student-t",
    "robust_bme",
    "robust-bme",
}
GOODBAD_BME_ALIASES = {
    "goodbad_bme",
    "goodbad-bme",
    "good_bad_bme",
    "good-bad-bme",
    "good_bad",
    "good-bad",
}


@dataclass(slots=True)
class BioEmuCnnlsTeacherSummary:
    """Run-level summary for one BioEmu/UCBShift teacher build."""

    status: str
    message: str
    entity_count: int
    fitted_entity_count: int
    skipped_entity_count: int
    weight_rows: int
    prediction_rows: int
    audit_rows: int
    artifacts: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "entity_count": self.entity_count,
            "fitted_entity_count": self.fitted_entity_count,
            "skipped_entity_count": self.skipped_entity_count,
            "weight_rows": self.weight_rows,
            "prediction_rows": self.prediction_rows,
            "audit_rows": self.audit_rows,
            "artifacts": dict(self.artifacts),
        }


def run_bioemu_ucbshift_cnnls_teacher(
    *,
    data_root: str | Path,
    run_dir: str | Path,
    shift_manifest_path: str | Path,
    output_dir: str | Path | None = None,
    target_manifest_path: str | Path | None = None,
    method: str = "cs_reweighting",
    lambda_reg: float = 0.01,
    default_cs_sigma: float = 1.0,
    max_entities: int | None = None,
    require_all_sidecars: bool = False,
) -> BioEmuCnnlsTeacherSummary:
    """Fit per-entry population weights from BioEmu conformer UCBShift sidecars.

    Args:
        data_root: Repository data root used to resolve target bundle paths.
        run_dir: BioEmu run directory. Used for default artifact placement.
        shift_manifest_path: Table listing BioEmu conformer UCBShift sidecars.
        output_dir: Optional report root. Defaults to ``run_dir/reports``.
        target_manifest_path: Optional entity -> target bundle manifest.
        method: ``cs_reweighting`` for paper-style atom offset correction plus
            MaxEnt/BFGS, ``legacy_cs_reweighting`` for the earlier local
            simplex objective, ``cnnls``/``nnls`` for legacy simplex NLL, or
            ``maxent``.
        lambda_reg: KL regularization strength for ``maxent``.
        default_cs_sigma: Fallback chemical-shift uncertainty in ppm.
        max_entities: Optional cap for smoke runs.
        require_all_sidecars: Fail an entity when any listed sidecar is missing.

    Returns:
        A compact summary with generated artifact paths.
    """

    data_root_path = Path(data_root)
    run_dir_path = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir_path / "reports"
    arrays_dir = report_dir / "arrays"
    figures_dir = report_dir / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    shift_manifest = _read_table(shift_manifest_path)
    if shift_manifest.empty:
        raise ValueError(f"BioEmu UCBShift manifest is empty: {shift_manifest_path}")
    shift_manifest = _normalize_shift_manifest(
        shift_manifest,
        base_dir=Path(shift_manifest_path).parent,
    )
    if max_entities is not None:
        keep = list(dict.fromkeys(shift_manifest["entity_uid"].astype(str)))[:max_entities]
        shift_manifest = shift_manifest.loc[
            shift_manifest["entity_uid"].astype(str).isin(keep)
        ].copy()

    target_manifest = _load_target_manifest(
        data_root=data_root_path,
        target_manifest_path=target_manifest_path,
    )

    weight_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for entity_uid, entity_shift_manifest in shift_manifest.groupby(
        "entity_uid",
        sort=False,
    ):
        entity_uid = str(entity_uid)
        target_path = target_manifest.get(entity_uid)
        if target_path is None:
            skipped.append(
                {
                    "entity_uid": entity_uid,
                    "status": "missing_target_bundle",
                    "message": "No target bundle path was available.",
                }
            )
            continue
        existing = entity_shift_manifest.loc[
            entity_shift_manifest["chemical_shift_path"].map(Path).map(Path.exists)
        ].copy()
        missing_count = int(len(entity_shift_manifest) - len(existing))
        if missing_count and require_all_sidecars:
            skipped.append(
                {
                    "entity_uid": entity_uid,
                    "status": "missing_ucbshift_sidecar",
                    "message": f"{missing_count} conformer sidecars were missing.",
                }
            )
            continue
        if existing.empty:
            skipped.append(
                {
                    "entity_uid": entity_uid,
                    "status": "no_ucbshift_sidecars",
                    "message": "No usable UCBShift2.0 conformer sidecars were found.",
                }
            )
            continue
        try:
            bundle = NMRTargetBundle.from_json(target_path)
            weights, predictions, audit = _fit_entity_teacher(
                entity_uid=entity_uid,
                bundle=bundle,
                manifest=existing,
                method=method,
                lambda_reg=lambda_reg,
                default_cs_sigma=default_cs_sigma,
                missing_sidecar_count=missing_count,
            )
        except Exception as exc:
            skipped.append(
                {
                    "entity_uid": entity_uid,
                    "status": "fit_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        weight_frames.append(weights)
        prediction_frames.append(predictions)
        audit_rows.extend(audit)

    weights_frame = (
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    predictions_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    audit_frame = pd.DataFrame(audit_rows)
    skipped_frame = pd.DataFrame(skipped)

    artifacts: dict[str, str] = {}
    artifacts.update(
        _write_table_pair(
            weights_frame,
            arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_weights.parquet",
        )
    )
    artifacts.update(
        _write_table_pair(
            predictions_frame,
            arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_predictions.parquet",
        )
    )
    artifacts.update(
        _write_table_pair(
            audit_frame,
            arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_audit.parquet",
        )
    )
    artifacts.update(
        _write_table_pair(
            skipped_frame,
            arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_skipped.parquet",
        )
    )
    _render_teacher_weight_plot(
        weights_frame,
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_weights.png",
    )
    _render_teacher_audit_plot(
        audit_frame,
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_audit.png",
    )
    artifacts[f"{CANONICAL_ARTIFACT_STEM}_weights_png"] = str(
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_weights.png"
    )
    artifacts[f"{CANONICAL_ARTIFACT_STEM}_audit_png"] = str(
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_audit.png"
    )

    fitted_entities = int(weights_frame["entity_uid"].nunique()) if not weights_frame.empty else 0
    selected_entities = int(shift_manifest["entity_uid"].nunique())
    status = "ok" if fitted_entities else "failed"
    message = (
        "BioEmu UCBShift2.0/CS-reweighting teacher generated."
        if fitted_entities
        else "No BioEmu UCBShift2.0/CS-reweighting teachers could be fitted."
    )
    summary = BioEmuCnnlsTeacherSummary(
        status=status,
        message=message,
        entity_count=selected_entities,
        fitted_entity_count=fitted_entities,
        skipped_entity_count=int(len(skipped_frame)),
        weight_rows=int(len(weights_frame)),
        prediction_rows=int(len(predictions_frame)),
        audit_rows=int(len(audit_frame)),
        artifacts=artifacts,
    )
    summary_path = arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_summary.json"
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    artifacts["summary_json"] = str(summary_path)
    return summary


def merge_bioemu_ucbshift_cnnls_teacher_shards(
    *,
    shard_output_dirs: list[str | Path] | None = None,
    shard_root: str | Path | None = None,
    output_dir: str | Path,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Merge sharded BioEmu/UCBShift teacher reports.

    Shard jobs intentionally write into separate report roots so they can run
    concurrently without racing on the canonical teacher parquet files.  This
    helper concatenates those shard-local artifacts back into the existing
    canonical filenames consumed by BioEmu latent-atlas training.
    """

    shard_dirs = _resolve_teacher_shard_dirs(
        shard_output_dirs=shard_output_dirs,
        shard_root=shard_root,
    )
    if require_nonempty and not shard_dirs:
        raise FileNotFoundError("No teacher shard output directories were found.")

    output_path = Path(output_dir)
    arrays_dir = output_path / "arrays"
    figures_dir = output_path / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, pd.DataFrame] = {}
    row_counts: dict[str, int] = {}
    artifacts: dict[str, str] = {}
    for suffix in ["weights", "predictions", "audit", "skipped"]:
        frames: list[pd.DataFrame] = []
        for shard_dir in shard_dirs:
            shard_arrays = shard_dir / "arrays"
            shard_path = shard_arrays / f"{CANONICAL_ARTIFACT_STEM}_{suffix}.parquet"
            if not shard_path.exists():
                continue
            frame = pd.read_parquet(shard_path)
            if "teacher_shard_dir" not in frame.columns:
                frame = frame.copy()
                frame["teacher_shard_dir"] = str(shard_dir)
            frames.append(frame)
        merged_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        merged[suffix] = merged_frame
        row_counts[suffix] = int(len(merged_frame))
        artifacts.update(
            _write_table_pair(
                merged_frame,
                arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_{suffix}.parquet",
            )
        )

    _render_teacher_weight_plot(
        merged["weights"],
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_weights.png",
    )
    _render_teacher_audit_plot(
        merged["audit"],
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_audit.png",
    )
    artifacts[f"{CANONICAL_ARTIFACT_STEM}_weights_png"] = str(
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_weights.png"
    )
    artifacts[f"{CANONICAL_ARTIFACT_STEM}_audit_png"] = str(
        figures_dir / f"{CANONICAL_ARTIFACT_STEM}_audit.png"
    )

    fitted_entity_count = (
        int(merged["weights"]["entity_uid"].astype(str).nunique())
        if not merged["weights"].empty and "entity_uid" in merged["weights"].columns
        else 0
    )
    summary = {
        "status": "ok" if fitted_entity_count else "failed",
        "message": (
            "Sharded BioEmu UCBShift2.0 teacher artifacts merged."
            if fitted_entity_count
            else "No fitted teacher shard rows were available to merge."
        ),
        "shard_count": int(len(shard_dirs)),
        "shard_output_dirs": [str(path) for path in shard_dirs],
        "fitted_entity_count": fitted_entity_count,
        "row_counts": row_counts,
        "artifacts": artifacts,
    }
    summary_path = arrays_dir / f"{CANONICAL_ARTIFACT_STEM}_merged_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    artifacts["merged_summary_json"] = str(summary_path)
    if require_nonempty and fitted_entity_count == 0:
        raise ValueError(summary["message"])
    return summary


def _resolve_teacher_shard_dirs(
    *,
    shard_output_dirs: list[str | Path] | None,
    shard_root: str | Path | None,
) -> list[Path]:
    dirs: list[Path] = []
    if shard_output_dirs:
        dirs.extend(Path(path) for path in shard_output_dirs)
    if shard_root is not None:
        dirs.extend(sorted(Path(shard_root).glob("shard_*_of_*")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if (resolved / "arrays").exists():
            unique.append(resolved)
    return unique


def _fit_entity_teacher(
    *,
    entity_uid: str,
    bundle: NMRTargetBundle,
    manifest: pd.DataFrame,
    method: str,
    lambda_reg: float,
    default_cs_sigma: float,
    missing_sidecar_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    pool = _candidate_pool_from_manifest(manifest)
    matrix, alignment_metadata = _build_residue_aligned_shift_matrix(pool, bundle)
    observables = ObservableBundle(chemical_shifts=matrix)
    prior = _prior_weights_from_manifest(manifest)
    resolved_method = str(method).strip().lower()
    cs_fit_metadata: dict[str, Any] = {}
    teacher_mean: np.ndarray
    if resolved_method in {"maxent", "max_entropy"}:
        reweighter = MaxEntReweighter()
        regularizer = "maxent"
        method_label = "maxent"
        energy = MultiObservablePosteriorEnergy(
            reweighter=reweighter,
            beta_cs=1.0,
            beta_j=0.0,
            beta_noe=0.0,
            lambda_reg=float(lambda_reg),
            regularizer=regularizer,
            default_cs_sigma=float(default_cs_sigma),
        )
        breakdown = energy.score(bundle, observables, prior_weights=prior)
        teacher_mean = energy.predictions(breakdown.weights, observables)[
            "chemical_shifts"
        ]
    elif resolved_method in {"cnnls", "nnls", "euclidean"}:
        reweighter = EuclideanSimplexReweighter()
        regularizer = "none"
        method_label = "cnnls"
        energy = MultiObservablePosteriorEnergy(
            reweighter=reweighter,
            beta_cs=1.0,
            beta_j=0.0,
            beta_noe=0.0,
            lambda_reg=float(lambda_reg),
            regularizer=regularizer,
            default_cs_sigma=float(default_cs_sigma),
        )
        breakdown = energy.score(bundle, observables, prior_weights=prior)
        teacher_mean = energy.predictions(breakdown.weights, observables)[
            "chemical_shifts"
        ]
    elif resolved_method in CS_REWEIGHTING_ALIASES:
        method_label = "cs_reweighting"
        breakdown, teacher_mean, cs_fit_metadata = _fit_paper_cs_reweighting_teacher(
            matrix=matrix,
            bundle=bundle,
            lambda_reg=lambda_reg,
        )
    elif resolved_method in LEGACY_CS_REWEIGHTING_ALIASES:
        method_label = "legacy_cs_reweighting"
        breakdown, teacher_mean, cs_fit_metadata = _fit_legacy_cs_reweighting_teacher(
            matrix=matrix,
            bundle=bundle,
            prior=prior,
            lambda_reg=lambda_reg,
            default_cs_sigma=default_cs_sigma,
        )
    elif resolved_method in STUDENT_BME_ALIASES:
        method_label = "student_bme"
        breakdown, teacher_mean, cs_fit_metadata = _fit_legacy_cs_reweighting_teacher(
            matrix=matrix,
            bundle=bundle,
            prior=prior,
            lambda_reg=lambda_reg,
            default_cs_sigma=default_cs_sigma,
            robust_likelihood="student",
        )
    elif resolved_method in GOODBAD_BME_ALIASES:
        method_label = "goodbad_bme"
        breakdown, teacher_mean, cs_fit_metadata = _fit_legacy_cs_reweighting_teacher(
            matrix=matrix,
            bundle=bundle,
            prior=prior,
            lambda_reg=lambda_reg,
            default_cs_sigma=default_cs_sigma,
            robust_likelihood="good_bad",
        )
    else:
        raise ValueError(
            "method must be one of: cs_reweighting, student_bme, goodbad_bme, "
            "legacy_cs_reweighting, cnnls, nnls, euclidean, maxent"
        )
    cs_fit_metadata = {**alignment_metadata, **cs_fit_metadata}

    manifest = manifest.reset_index(drop=True)
    weight_frame = manifest[
        [
            column
            for column in [
                "entity_uid",
                "candidate_id",
                "sample_index",
                "structure_path",
                "chemical_shift_path",
                "chemical_shift_format",
                "bioemu_log_prob",
                "provider",
                "round_id",
            ]
            if column in manifest.columns
        ]
    ].copy()
    weight_frame["teacher_weight"] = breakdown.weights
    weight_frame["prior_weight"] = prior
    weight_frame["teacher_weight_rank"] = (
        weight_frame["teacher_weight"].rank(method="first", ascending=False).astype(int)
    )
    weight_frame["teacher_method"] = method_label
    weight_frame["teacher_energy"] = float(breakdown.energy)
    weight_frame["teacher_ess"] = float(breakdown.ess)
    weight_frame["teacher_entropy"] = float(breakdown.entropy)
    weight_frame["teacher_converged"] = bool(breakdown.converged)
    weight_frame["missing_sidecar_count"] = int(missing_sidecar_count)
    if cs_fit_metadata:
        weight_frame["teacher_fit_rows"] = int(cs_fit_metadata.get("fit_rows", 0))
        weight_frame["teacher_outlier_rows"] = int(
            cs_fit_metadata.get("outlier_rows", 0)
        )
        weight_frame["teacher_atom_offsets_json"] = _json_dumps_clean(
            cs_fit_metadata.get("atom_offsets", {})
        )
        weight_frame["teacher_atom_scaling_json"] = _json_dumps_clean(
            cs_fit_metadata.get("atom_scaling", {})
        )
        weight_frame["teacher_atom_weights_json"] = _json_dumps_clean(
            cs_fit_metadata.get("atom_weights", {})
        )
        if "algorithm" in cs_fit_metadata:
            weight_frame["teacher_algorithm"] = str(cs_fit_metadata["algorithm"])
        if "theta" in cs_fit_metadata:
            weight_frame["teacher_theta"] = float(cs_fit_metadata["theta"])
        if "prior_source" in cs_fit_metadata:
            weight_frame["teacher_prior_source"] = str(cs_fit_metadata["prior_source"])
        if "prior_mode" in cs_fit_metadata:
            weight_frame["teacher_prior_mode"] = str(cs_fit_metadata["prior_mode"])
        if "fit_atoms" in cs_fit_metadata:
            weight_frame["teacher_fit_atoms_json"] = _json_dumps_clean(
                cs_fit_metadata.get("fit_atoms", [])
            )
        if "atom_scores" in cs_fit_metadata:
            weight_frame["teacher_atom_scores_json"] = _json_dumps_clean(
                cs_fit_metadata.get("atom_scores", {})
            )
        if "total_score" in cs_fit_metadata:
            weight_frame["teacher_paper_total_score"] = float(
                cs_fit_metadata["total_score"]
            )
        if "bfgs_success" in cs_fit_metadata:
            weight_frame["teacher_bfgs_success"] = bool(
                cs_fit_metadata["bfgs_success"]
            )
        if "bfgs_message" in cs_fit_metadata:
            weight_frame["teacher_bfgs_message"] = str(
                cs_fit_metadata["bfgs_message"]
            )
        if "residue_offset" in cs_fit_metadata:
            weight_frame["teacher_residue_offset"] = int(
                cs_fit_metadata["residue_offset"]
            )
            weight_frame["teacher_residue_alignment_status"] = str(
                cs_fit_metadata.get("residue_alignment_status", "")
            )
            weight_frame["teacher_residue_alignment_match_fraction"] = float(
                cs_fit_metadata.get("residue_alignment_match_fraction", 0.0)
            )
            weight_frame["teacher_residue_alignment_best_matches"] = int(
                cs_fit_metadata.get("residue_alignment_best_matches", 0)
            )
            weight_frame["teacher_residue_alignment_direct_matches"] = int(
                cs_fit_metadata.get("residue_alignment_direct_matches", 0)
            )
        if "robust_likelihood" in cs_fit_metadata:
            weight_frame["teacher_robust_likelihood"] = str(
                cs_fit_metadata["robust_likelihood"]
            )
        if "robust_inlier_responsibility_mean" in cs_fit_metadata:
            weight_frame["teacher_inlier_responsibility_mean"] = float(
                cs_fit_metadata["robust_inlier_responsibility_mean"]
            )

    prediction_rows: list[dict[str, Any]] = []
    row_offsets = cs_fit_metadata.get("row_offsets", []) if cs_fit_metadata else []
    fit_row_mask = cs_fit_metadata.get("fit_row_mask", []) if cs_fit_metadata else []
    residue_offset = int(cs_fit_metadata.get("residue_offset", 0)) if cs_fit_metadata else 0
    for row_index, target in enumerate(bundle.chemical_shifts):
        prediction = float(teacher_mean[row_index])
        target_value = float(target.value)
        atom_family = _atom_family(target.atom_id)
        row = {
            "entity_uid": entity_uid,
            "target_id": target.target_id(),
            "seq_id": int(target.seq_id),
            "residue_name": str(target.comp_id),
            "atom_id": str(target.atom_id),
            "atom_family": atom_family,
            "target_value": target_value,
            "teacher_mean": prediction,
            "teacher_abs_error": abs(prediction - target_value),
            "teacher_residual": prediction - target_value,
            "valid_conformer_count": int(matrix.mask[row_index].sum()),
            "candidate_count": int(len(pool.records)),
            "teacher_method": method_label,
            "teacher_ess": float(breakdown.ess),
            "teacher_entropy": float(breakdown.entropy),
            "teacher_energy": float(breakdown.energy),
            "teacher_converged": bool(breakdown.converged),
            "teacher_residue_offset": int(residue_offset),
            "sidecar_seq_id": int(target.seq_id) + int(residue_offset),
        }
        if cs_fit_metadata and "residue_alignment_status" in cs_fit_metadata:
            row["teacher_residue_alignment_status"] = str(
                cs_fit_metadata.get("residue_alignment_status", "")
            )
            row["teacher_residue_alignment_match_fraction"] = float(
                cs_fit_metadata.get("residue_alignment_match_fraction", 0.0)
            )
        if cs_fit_metadata and "algorithm" in cs_fit_metadata:
            row["teacher_algorithm"] = str(cs_fit_metadata["algorithm"])
        if "robust_inlier_responsibility_mean" in cs_fit_metadata:
            row["teacher_inlier_responsibility_mean"] = float(
                cs_fit_metadata["robust_inlier_responsibility_mean"]
            )
        if row_offsets:
            row["teacher_shift_offset"] = float(row_offsets[row_index])
        if fit_row_mask:
            row["teacher_fit_row"] = bool(fit_row_mask[row_index])
        prediction_rows.append(row)
    prediction_frame = pd.DataFrame(prediction_rows)
    audit_rows = _entity_audit_rows(
        prediction_frame,
        entity_uid=entity_uid,
        candidate_count=len(pool.records),
        missing_sidecar_count=missing_sidecar_count,
        ess=float(breakdown.ess),
        entropy=float(breakdown.entropy),
        energy=float(breakdown.energy),
        converged=bool(breakdown.converged),
        method=method_label,
    )
    if cs_fit_metadata:
        for row in audit_rows:
            if "algorithm" in cs_fit_metadata:
                row["teacher_algorithm"] = str(cs_fit_metadata["algorithm"])
            if "residue_offset" in cs_fit_metadata:
                row["teacher_residue_offset"] = int(cs_fit_metadata["residue_offset"])
                row["teacher_residue_alignment_status"] = str(
                    cs_fit_metadata.get("residue_alignment_status", "")
                )
                row["teacher_residue_alignment_match_fraction"] = float(
                    cs_fit_metadata.get("residue_alignment_match_fraction", 0.0)
                )
                row["teacher_residue_alignment_best_matches"] = int(
                    cs_fit_metadata.get("residue_alignment_best_matches", 0)
                )
                row["teacher_residue_alignment_direct_matches"] = int(
                    cs_fit_metadata.get("residue_alignment_direct_matches", 0)
                )
            if "total_score" in cs_fit_metadata:
                row["teacher_paper_total_score"] = float(
                    cs_fit_metadata["total_score"]
                )
    return weight_frame, prediction_frame, audit_rows


def _candidate_pool_from_manifest(manifest: pd.DataFrame) -> CandidatePool:
    records: list[CandidateRecord] = []
    for row in manifest.itertuples(index=False):
        records.append(
            CandidateRecord(
                candidate_id=str(getattr(row, "candidate_id")),
                structure_path=str(getattr(row, "structure_path", "")),
                source="bioemu_ucbshift2_teacher",
                chemical_shift_path=str(getattr(row, "chemical_shift_path")),
                chemical_shift_format=str(
                    getattr(row, "chemical_shift_format", "ucbshift")
                    or "ucbshift"
                ),
                metadata={
                    "sample_index": int(getattr(row, "sample_index", 0)),
                    "entity_uid": str(getattr(row, "entity_uid")),
                },
            )
        )
    return CandidatePool(records=records)


def _prior_weights_from_manifest(manifest: pd.DataFrame) -> np.ndarray:
    if "prior_weight" in manifest.columns:
        prior = pd.to_numeric(manifest["prior_weight"], errors="coerce").to_numpy(float)
        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        if prior.sum() > 0.0:
            return prior / prior.sum()
    if "bioemu_log_prob" in manifest.columns:
        logits = pd.to_numeric(manifest["bioemu_log_prob"], errors="coerce").to_numpy(float)
        finite = np.isfinite(logits)
        if finite.any():
            logits = np.where(finite, logits, np.nanmin(logits[finite]))
            logits = logits - float(np.max(logits))
            prior = np.exp(logits)
            if prior.sum() > 0.0:
                return prior / prior.sum()
    return np.full(len(manifest), 1.0 / max(len(manifest), 1), dtype=float)


def _build_residue_aligned_shift_matrix(
    pool: CandidatePool,
    bundle: NMRTargetBundle,
) -> tuple[ObservableMatrix, dict[str, Any]]:
    """Build a chemical-shift matrix after inferring target-to-sidecar offset.

    BMRB ``Seq_ID`` values and UCBShift ``RESNUM`` values are not guaranteed to
    share the same origin.  The CS-reweighting paper loader aligns on residue
    numbers after data preparation; here we infer a single integer offset from
    sidecar residue names and apply it before matching atom rows.
    """

    residue_offset, alignment_metadata = _infer_residue_number_offset(pool, bundle)
    targets = bundle.chemical_shifts
    values = np.zeros((len(targets), len(pool.records)), dtype=float)
    mask = np.zeros_like(values, dtype=bool)
    target_ids = [target.target_id() for target in targets]
    cache: dict[str, dict[tuple[int, str], float]] = {}

    for column, record in enumerate(pool.records):
        if not record.chemical_shift_path:
            continue
        cache_key = f"{record.chemical_shift_path}:{record.chemical_shift_format}"
        if cache_key not in cache:
            cache[cache_key] = load_candidate_shift_file(
                record.chemical_shift_path,
                record.chemical_shift_format,
            )
        mapping = cache[cache_key]
        for row, target in enumerate(targets):
            sidecar_seq_id = int(target.seq_id) + int(residue_offset)
            key = (sidecar_seq_id, canonical_atom_name(target.atom_id))
            if key not in mapping:
                continue
            values[row, column] = mapping[key]
            mask[row, column] = True

    matrix = ObservableMatrix(
        label="chemical_shifts",
        values=values,
        mask=mask,
        target_ids=target_ids,
        candidate_ids=pool.candidate_ids,
        transform="identity",
    )
    alignment_metadata = dict(alignment_metadata)
    alignment_metadata["residue_offset"] = int(residue_offset)
    return matrix, alignment_metadata


def _infer_residue_number_offset(
    pool: CandidatePool,
    bundle: NMRTargetBundle,
    *,
    max_abs_offset: int = RESIDUE_ALIGNMENT_MAX_ABS_OFFSET,
    max_sidecars: int = 8,
) -> tuple[int, dict[str, Any]]:
    target_residues = _target_residue_names(bundle)
    sidecar_residues = _sidecar_residue_names(pool, max_sidecars=max_sidecars)
    metadata: dict[str, Any] = {
        "residue_alignment_target_residue_count": int(len(target_residues)),
        "residue_alignment_sidecar_residue_count": int(len(sidecar_residues)),
        "residue_alignment_source": "ucbshift_resname_offset_scan",
    }
    if not target_residues or not sidecar_residues:
        metadata.update(
            {
                "residue_alignment_status": "unavailable_residue_names",
                "residue_alignment_direct_matches": 0,
                "residue_alignment_best_matches": 0,
                "residue_alignment_best_available": 0,
                "residue_alignment_match_fraction": 0.0,
            }
        )
        return 0, metadata

    def score(offset: int) -> tuple[int, int]:
        matches = 0
        available = 0
        for seq_id, residue_name in target_residues.items():
            sidecar_name = sidecar_residues.get(int(seq_id) + int(offset))
            if sidecar_name is None:
                continue
            available += 1
            if sidecar_name == residue_name:
                matches += 1
        return matches, available

    direct_matches, direct_available = score(0)
    best_offset = 0
    best_matches = direct_matches
    best_available = direct_available
    for offset in range(-int(max_abs_offset), int(max_abs_offset) + 1):
        matches, available = score(offset)
        candidate_key = (
            matches,
            available,
            -abs(offset),
            1 if offset == 0 else 0,
        )
        best_key = (
            best_matches,
            best_available,
            -abs(best_offset),
            1 if best_offset == 0 else 0,
        )
        if candidate_key > best_key:
            best_offset = offset
            best_matches = matches
            best_available = available

    denominator = max(len(target_residues), 1)
    metadata.update(
        {
            "residue_alignment_status": (
                "offset_corrected" if best_offset != 0 else "direct_or_unresolved"
            ),
            "residue_alignment_direct_matches": int(direct_matches),
            "residue_alignment_direct_available": int(direct_available),
            "residue_alignment_best_matches": int(best_matches),
            "residue_alignment_best_available": int(best_available),
            "residue_alignment_match_fraction": float(best_matches / denominator),
        }
    )
    return int(best_offset), metadata


def _target_residue_names(bundle: NMRTargetBundle) -> dict[int, str]:
    residues: dict[int, str] = {}
    for target in bundle.chemical_shifts:
        residue_name = _normalize_residue_name(target.comp_id)
        if not residue_name:
            continue
        residues.setdefault(int(target.seq_id), residue_name)
    return residues


def _sidecar_residue_names(
    pool: CandidatePool,
    *,
    max_sidecars: int,
) -> dict[int, str]:
    votes: dict[int, dict[str, int]] = {}
    inspected = 0
    for record in pool.records:
        if inspected >= max_sidecars:
            break
        if not record.chemical_shift_path:
            continue
        path = Path(record.chemical_shift_path)
        if not path.exists() or path.suffix.lower() not in {".csv", ".tab", ".cs", ""}:
            continue
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "RESNUM" not in reader.fieldnames:
                    continue
                residue_column = _first_existing_column(
                    reader.fieldnames,
                    ["RESNAME", "RES", "RESIDUE", "AA", "COMP_ID"],
                )
                if residue_column is None:
                    continue
                for row in reader:
                    try:
                        seq_id = int(float(str(row.get("RESNUM", "")).strip()))
                    except ValueError:
                        continue
                    residue_name = _normalize_residue_name(row.get(residue_column, ""))
                    if not residue_name:
                        continue
                    residue_votes = votes.setdefault(seq_id, {})
                    residue_votes[residue_name] = residue_votes.get(residue_name, 0) + 1
            inspected += 1
        except OSError:
            continue
    residues: dict[int, str] = {}
    for seq_id, residue_votes in votes.items():
        residues[seq_id] = max(
            residue_votes.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
    return residues


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {str(column).upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in normalized:
            return normalized[candidate.upper()]
    return None


def _normalize_residue_name(value: Any) -> str:
    residue = str(value or "").strip().upper()
    if not residue or residue in {".", "?", "NAN", "NONE"}:
        return ""
    one_to_three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }
    return one_to_three.get(residue, residue[:3])


def _fit_paper_cs_reweighting_teacher(
    *,
    matrix: ObservableMatrix,
    bundle: NMRTargetBundle,
    lambda_reg: float,
) -> tuple[EnergyBreakdown, np.ndarray, dict[str, Any]]:
    """Fit the CS-reweighting paper-style offset + MaxEnt/BFGS teacher."""

    values = np.asarray(matrix.values, dtype=float)
    mask = np.asarray(matrix.mask, dtype=bool) & np.isfinite(values)
    target_values = np.asarray(
        [float(target.value) for target in bundle.chemical_shifts],
        dtype=float,
    )
    atoms = [canonical_atom_name(target.atom_id) for target in bundle.chemical_shifts]
    num_targets, num_candidates = values.shape
    if num_candidates == 0:
        raise ValueError("Paper CS-reweighting teacher requires at least one conformer.")

    fit_row_mask = np.asarray(
        [
            atom in PAPER_CS_REWEIGHTING_ATOMS
            and np.isfinite(target_values[index])
            and bool(mask[index].any())
            for index, atom in enumerate(atoms)
        ],
        dtype=bool,
    )
    fit_rows = np.flatnonzero(fit_row_mask)
    if fit_rows.size == 0:
        raise ValueError(
            "No H/N/CA/CB rows with UCBShift sidecar support were available for "
            "paper CS-reweighting."
        )

    fit_values = _fill_missing_shift_rows(
        values[fit_rows],
        mask[fit_rows],
        target_values[fit_rows],
    )
    fit_targets = target_values[fit_rows]
    fit_atoms = [atoms[index] for index in fit_rows]
    theta = max(float(lambda_reg), 1.0e-10)
    atom_offsets, offset_diagnostics = _paper_optimize_atom_offsets(
        data=fit_values,
        ref=fit_targets,
        atoms=fit_atoms,
    )

    row_offsets = np.zeros(num_targets, dtype=float)
    for row_index, atom in enumerate(atoms):
        row_offsets[row_index] = float(atom_offsets.get(atom, 0.0))
    corrected_fit = fit_values + row_offsets[fit_rows, np.newaxis]
    weights, bfgs_diagnostics = _paper_maxent_bfgs(
        data=corrected_fit,
        ref=fit_targets,
        atoms=fit_atoms,
        theta=theta,
    )
    teacher_mean = _cs_reweighting_predictions(
        values=values,
        mask=mask,
        weights=weights,
        row_offsets=row_offsets,
    )
    atom_scores = _paper_atom_scores(
        data=corrected_fit,
        ref=fit_targets,
        weights=weights,
        atoms=fit_atoms,
    )
    total_score = _paper_total_score(atom_scores)
    entropy = _weight_entropy(weights)
    ess = _effective_sample_size(weights)
    energy = float(bfgs_diagnostics.get("objective", np.nan))
    diagnostics = {
        "chemical_shifts_count": float(fit_rows.size),
        "paper_cs_reweighting_theta": float(theta),
        "paper_cs_reweighting_total_score": float(total_score),
        "paper_cs_reweighting_bfgs_iterations": float(
            bfgs_diagnostics.get("iterations", 0.0)
        ),
        "paper_cs_reweighting_bfgs_success": float(
            1.0 if bfgs_diagnostics.get("success", False) else 0.0
        ),
        **{
            f"paper_cs_reweighting_score_{atom}": float(score)
            for atom, score in atom_scores.items()
        },
        **{
            f"paper_cs_reweighting_offset_{atom}": float(offset)
            for atom, offset in atom_offsets.items()
        },
        **offset_diagnostics,
    }
    breakdown = EnergyBreakdown(
        energy=energy,
        weights=weights,
        channel_scores={"chemical_shifts": energy},
        ess=ess,
        entropy=entropy,
        iterations=int(bfgs_diagnostics.get("iterations", 0)),
        converged=bool(bfgs_diagnostics.get("success", False)),
        diagnostics=diagnostics,
    )
    metadata = {
        "fit_rows": int(fit_rows.size),
        "outlier_rows": 0,
        "algorithm": "paper_cs_reweighting_maxent_bfgs",
        "theta": float(theta),
        "fit_atoms": list(PAPER_CS_REWEIGHTING_ATOMS),
        "atom_offsets": atom_offsets,
        "atom_scaling": bfgs_diagnostics.get("atom_ranges", {}),
        "atom_weights": {atom: 1.0 for atom in PAPER_CS_REWEIGHTING_ATOMS},
        "atom_scores": atom_scores,
        "total_score": float(total_score),
        "row_offsets": row_offsets.tolist(),
        "fit_row_mask": fit_row_mask.tolist(),
        "prior_mode": "uniform",
        "prior_source": "uniform_bioemu_conformer_pool",
        "offset_iterations": int(PAPER_CS_REWEIGHTING_OFFSET_ITERATIONS),
        "offset_grid_points": int(PAPER_CS_REWEIGHTING_OFFSET_GRID_POINTS),
        "bfgs_success": bool(bfgs_diagnostics.get("success", False)),
        "bfgs_message": str(bfgs_diagnostics.get("message", "")),
    }
    return breakdown, teacher_mean, metadata


def _fit_legacy_cs_reweighting_teacher(
    *,
    matrix: ObservableMatrix,
    bundle: NMRTargetBundle,
    prior: np.ndarray,
    lambda_reg: float,
    default_cs_sigma: float,
    robust_likelihood: str = "gaussian",
) -> tuple[EnergyBreakdown, np.ndarray, dict[str, Any]]:
    """Fit CS-reweighting-style simplex weights on UCBShift sidecars.

    This mirrors the practical CS-reweighting recipe: estimate atom-family
    offsets from candidate ensemble means, remove extreme offset outliers from
    the fit system, scale each atom family by its empirical CS noise scale, and
    emphasize proton rows.  The final weights still live on the BioEmu conformer
    simplex so they can act as population teachers for the generator.
    """

    values = np.asarray(matrix.values, dtype=float)
    mask = np.asarray(matrix.mask, dtype=bool) & np.isfinite(values)
    target_values = np.asarray(
        [float(target.value) for target in bundle.chemical_shifts],
        dtype=float,
    )
    atoms = [canonical_atom_name(target.atom_id) for target in bundle.chemical_shifts]
    num_targets, num_candidates = values.shape
    if num_candidates == 0:
        raise ValueError("CS-reweighting teacher requires at least one conformer.")

    prior = np.asarray(prior, dtype=float)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior_sum = float(prior.sum())
    if prior.shape != (num_candidates,) or prior_sum <= 0.0:
        prior = np.full(num_candidates, 1.0 / num_candidates, dtype=float)
    else:
        prior = prior / prior_sum

    row_offsets = np.zeros(num_targets, dtype=float)
    fit_row_mask = np.zeros(num_targets, dtype=bool)
    atom_offsets: dict[str, float] = {}
    atom_fit_rows: dict[str, int] = {}
    atom_outlier_rows: dict[str, int] = {}

    for atom in sorted(set(atoms)):
        row_indices = np.asarray(
            [index for index, row_atom in enumerate(atoms) if row_atom == atom],
            dtype=int,
        )
        if row_indices.size == 0:
            continue
        row_means = _masked_row_means(values[row_indices], mask[row_indices])
        residuals = target_values[row_indices] - row_means
        finite = np.isfinite(residuals)
        if not finite.any():
            offset = 0.0
            kept = np.zeros_like(finite, dtype=bool)
        else:
            offset = float(np.nanmedian(residuals[finite]))
            kept = finite.copy()
            if int(finite.sum()) >= 4:
                spread = float(np.nanstd(residuals[finite]))
                if spread > 1.0e-12:
                    kept &= (
                        np.abs((residuals - offset) / spread)
                        <= CS_REWEIGHTING_OUTLIER_Z
                    )
        row_offsets[row_indices] = offset
        fit_row_mask[row_indices[kept]] = True
        atom_offsets[atom] = offset
        atom_fit_rows[atom] = int(kept.sum())
        atom_outlier_rows[atom] = int(finite.sum() - kept.sum())

    fit_rows = np.flatnonzero(fit_row_mask)
    if fit_rows.size == 0:
        raise ValueError("No usable chemical-shift rows were available for CS-reweighting.")

    design_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    for row_index in fit_rows:
        atom = atoms[row_index]
        scale = float(
            CS_REWEIGHTING_SCALING.get(
                atom,
                default_cs_sigma if default_cs_sigma > 0.0 else 1.0,
            )
        )
        scale = max(scale, 1.0e-6)
        atom_weight = float(CS_REWEIGHTING_ATOM_WEIGHTS.get(atom, 1.0))
        row_weight = float(np.sqrt(max(atom_weight, 1.0e-12)))
        target_value = float(target_values[row_index])
        corrected = values[row_index] + row_offsets[row_index]
        valid = mask[row_index] & np.isfinite(corrected)
        neutral = np.full(num_candidates, target_value, dtype=float)
        row_values = np.where(valid, corrected, neutral)
        design_rows.append((row_values / scale) * row_weight)
        target_rows.append((target_value / scale) * row_weight)

    design = np.vstack(design_rows).astype(float, copy=False)
    target = np.asarray(target_rows, dtype=float)
    normalizer = float(np.sqrt(max(len(target_rows), 1)))
    design /= normalizer
    target /= normalizer
    reg = max(float(lambda_reg), 0.0)
    robust_kind = str(robust_likelihood).strip().lower()

    if robust_kind in {"", "gaussian", "mse", "quadratic"}:

        def objective(weights: np.ndarray) -> float:
            delta = design @ weights - target
            prior_delta = weights - prior
            return float(np.dot(delta, delta) + reg * np.dot(prior_delta, prior_delta))

        def gradient(weights: np.ndarray) -> np.ndarray:
            delta = design @ weights - target
            prior_delta = weights - prior
            return 2.0 * (design.T @ delta + reg * prior_delta)

        solution = EuclideanSimplexReweighter().fit(
            num_candidates=num_candidates,
            objective_fn=objective,
            gradient_fn=gradient,
            prior_weights=prior,
        )
    else:
        solution = _fit_robust_bme_mirror_teacher(
            design=design,
            target=target,
            prior=prior,
            lambda_reg=reg,
            likelihood=robust_kind,
        )
    teacher_mean = _cs_reweighting_predictions(
        values=values,
        mask=mask,
        weights=solution.weights,
        row_offsets=row_offsets,
    )
    weights = np.asarray(solution.weights, dtype=float)
    entropy = _weight_entropy(weights)
    ess = _effective_sample_size(weights)
    diagnostics = {
        "chemical_shifts_count": float(fit_rows.size),
        "cs_reweighting_outlier_rows": float(
            sum(atom_outlier_rows.values())
        ),
        "cs_reweighting_lambda_reg": float(reg),
        "prior_mass": float(prior.sum()),
        "robust_likelihood": (
            0.0
            if robust_kind in {"", "gaussian", "mse", "quadratic"}
            else 1.0
        ),
        **{f"cs_reweighting_fit_rows_{atom}": float(count) for atom, count in atom_fit_rows.items()},
        **{
            f"cs_reweighting_offset_{atom}": float(offset)
            for atom, offset in atom_offsets.items()
        },
        **solution.diagnostics,
    }
    breakdown = EnergyBreakdown(
        energy=float(solution.energy),
        weights=weights,
        channel_scores={"chemical_shifts": float(solution.energy)},
        ess=ess,
        entropy=entropy,
        iterations=int(solution.iterations),
        converged=bool(solution.converged),
        diagnostics=diagnostics,
    )
    metadata = {
        "fit_rows": int(fit_rows.size),
        "outlier_rows": int(sum(atom_outlier_rows.values())),
        "atom_offsets": atom_offsets,
        "atom_scaling": {
            atom: float(CS_REWEIGHTING_SCALING.get(atom, default_cs_sigma))
            for atom in sorted(set(atoms))
        },
        "atom_weights": {
            atom: float(CS_REWEIGHTING_ATOM_WEIGHTS.get(atom, 1.0))
            for atom in sorted(set(atoms))
        },
        "robust_likelihood": robust_kind or "gaussian",
        "row_offsets": row_offsets.tolist(),
        "fit_row_mask": fit_row_mask.tolist(),
        "fit_atoms": sorted(set(atoms)),
        "prior_mode": "configured_support_prior",
        "prior_source": "input_candidate_prior",
    }
    if "inlier_responsibility_mean" in solution.diagnostics:
        metadata["robust_inlier_responsibility_mean"] = float(
            solution.diagnostics["inlier_responsibility_mean"]
        )
    return breakdown, teacher_mean, metadata


def _fill_missing_shift_rows(
    values: np.ndarray,
    mask: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    target_values = np.asarray(target_values, dtype=float)
    row_means = _masked_row_means(values, mask)
    fill_values = np.where(np.isfinite(row_means), row_means, target_values)
    return np.where(mask, values, fill_values[:, np.newaxis])


def _paper_optimize_atom_offsets(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    atoms: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    data = np.asarray(data, dtype=float)
    ref = np.asarray(ref, dtype=float)
    unique_atoms = [atom for atom in PAPER_CS_REWEIGHTING_ATOMS if atom in set(atoms)]
    offsets = {atom: 0.0 for atom in unique_atoms}
    best_scores: dict[str, float] = {}
    evaluation_count = 0

    for _ in range(PAPER_CS_REWEIGHTING_OFFSET_ITERATIONS):
        for atom in unique_atoms:
            search_radius = 0.2 if atom.startswith("H") else 1.0
            increments = np.linspace(
                -search_radius,
                search_radius,
                PAPER_CS_REWEIGHTING_OFFSET_GRID_POINTS,
            )
            best_increment = 0.0
            best_score = -float("inf")
            for increment in increments:
                trial_offsets = dict(offsets)
                trial_offsets[atom] = float(trial_offsets.get(atom, 0.0) + increment)
                corrected = _paper_apply_offsets(data, atoms, trial_offsets)
                score = _paper_offset_objective(
                    corrected=corrected,
                    ref=ref,
                    atoms=atoms,
                )
                evaluation_count += 1
                if score > best_score:
                    best_score = float(score)
                    best_increment = float(increment)
            offsets[atom] = float(offsets.get(atom, 0.0) + best_increment)
            best_scores[atom] = float(best_score)

    diagnostics = {
        "paper_cs_reweighting_offset_evaluations": float(evaluation_count),
        **{
            f"paper_cs_reweighting_offset_objective_{atom}": float(score)
            for atom, score in best_scores.items()
        },
    }
    return offsets, diagnostics


def _paper_apply_offsets(
    data: np.ndarray,
    atoms: list[str],
    offsets: dict[str, float],
) -> np.ndarray:
    corrected = np.asarray(data, dtype=float).copy()
    for row_index, atom in enumerate(atoms):
        corrected[row_index, :] += float(offsets.get(atom, 0.0))
    return corrected


def _paper_offset_objective(
    *,
    corrected: np.ndarray,
    ref: np.ndarray,
    atoms: list[str],
) -> float:
    scaled_data, scaled_ref, _ = _paper_scaling_min_max(
        data=corrected,
        ref=ref,
        atoms=atoms,
    )
    selected = _paper_select_columns_via_nnls(scaled_data, scaled_ref)
    if selected.size == 0:
        return -float("inf")
    selected_scaled = scaled_data[:, selected]
    initial = _paper_nnls_with_reg(selected_scaled, scaled_ref)
    weights = _paper_slsqp_weights(
        data=selected_scaled,
        ref=scaled_ref,
        initial=initial,
    )
    scores = _paper_atom_scores(
        data=corrected[:, selected],
        ref=ref,
        weights=weights,
        atoms=atoms,
    )
    return _paper_total_score(scores)


def _paper_scaling_min_max(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    atoms: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    data = np.asarray(data, dtype=float)
    ref = np.asarray(ref, dtype=float)
    atom_ranges = _paper_atom_ranges(ref, atoms)
    with np.errstate(invalid="ignore"):
        std_deviation = np.sqrt(np.nanmean(np.square(data - ref[:, np.newaxis]), axis=1))
        row_max = np.nanmax(data, axis=1)
        row_min = np.nanmin(data, axis=1)
    row_max = np.maximum(row_max, np.where(np.isfinite(ref), ref, -np.inf))
    row_min = np.minimum(row_min, np.where(np.isfinite(ref), ref, np.inf))
    ranges = row_max - row_min
    safe_ranges = np.where(np.isfinite(ranges) & (np.abs(ranges) > 1.0e-12), ranges, 1.0)
    data_scaled = (data - row_min[:, np.newaxis]) / safe_ranges[:, np.newaxis]
    ref_scaled = (ref - row_min) / safe_ranges
    row_atom_ranges = np.asarray(
        [max(float(atom_ranges.get(atom, 1.0)), 1.0e-12) for atom in atoms],
        dtype=float,
    )
    scale_factors = np.log1p(std_deviation / row_atom_ranges)
    scale_factors = np.nan_to_num(scale_factors, nan=1.0, posinf=1.0, neginf=1.0)
    data_scaled *= scale_factors[:, np.newaxis]
    ref_scaled *= scale_factors
    return data_scaled, ref_scaled, atom_ranges


def _paper_scaling_mean_center(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    atoms: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    data = np.asarray(data, dtype=float)
    ref = np.asarray(ref, dtype=float)
    atom_ranges = _paper_atom_ranges(ref, atoms)
    row_mean = np.mean(data, axis=1)
    data_scaled = data - row_mean[:, np.newaxis]
    ref_scaled = ref - row_mean
    row_atom_ranges = np.asarray(
        [max(float(atom_ranges.get(atom, 1.0)), 1.0e-12) for atom in atoms],
        dtype=float,
    )
    data_scaled /= row_atom_ranges[:, np.newaxis]
    ref_scaled /= row_atom_ranges
    return data_scaled, ref_scaled, atom_ranges


def _paper_atom_ranges(ref: np.ndarray, atoms: list[str]) -> dict[str, float]:
    ref = np.asarray(ref, dtype=float)
    ranges: dict[str, float] = {}
    for atom in sorted(set(atoms)):
        atom_values = ref[
            np.asarray([row_atom == atom for row_atom in atoms], dtype=bool)
        ]
        atom_values = atom_values[np.isfinite(atom_values)]
        if atom_values.size >= 2:
            atom_range = float(np.nanmax(atom_values) - np.nanmin(atom_values))
        else:
            atom_range = 0.0
        if not np.isfinite(atom_range) or atom_range <= 1.0e-12:
            atom_range = float(CS_REWEIGHTING_SCALING.get(atom, 1.0))
        ranges[atom] = atom_range
    return ranges


def _paper_select_columns_via_nnls(
    scaled_data: np.ndarray,
    scaled_ref: np.ndarray,
    *,
    max_selected: int = PAPER_CS_REWEIGHTING_MAX_SELECTED_CONFORMERS,
) -> np.ndarray:
    scaled_data = np.asarray(scaled_data, dtype=float)
    scaled_ref = np.asarray(scaled_ref, dtype=float)
    finite_rows = np.isfinite(scaled_ref) & np.all(np.isfinite(scaled_data), axis=1)
    data = scaled_data[finite_rows]
    ref = scaled_ref[finite_rows]
    num_candidates = scaled_data.shape[1]
    if num_candidates == 0:
        return np.asarray([], dtype=int)
    if data.shape[0] == 0:
        return np.arange(min(num_candidates, max_selected), dtype=int)
    try:
        standard_coef, _ = nnls(data, ref)
    except Exception:
        standard_coef = np.zeros(num_candidates, dtype=float)
    try:
        regularized_coef = _paper_nnls_with_reg(data, ref)
    except Exception:
        regularized_coef = np.zeros(num_candidates, dtype=float)
    combined = np.maximum(standard_coef, regularized_coef)
    selected = np.flatnonzero(combined > 1.0e-12)
    if selected.size == 0:
        residual = np.sum(np.square(data - ref[:, np.newaxis]), axis=0)
        selected = np.argsort(residual)[: min(num_candidates, max_selected)]
    if selected.size > max_selected:
        order = np.argsort(combined[selected])[::-1]
        selected = selected[order[:max_selected]]
    return np.asarray(np.sort(selected), dtype=int)


def _paper_nnls_with_reg(
    data: np.ndarray,
    ref: np.ndarray,
    *,
    regularization: float = 10000.0,
) -> np.ndarray:
    reg_row = regularization * np.ones((1, data.shape[1]), dtype=float)
    augmented_data = np.vstack([reg_row, data])
    augmented_ref = np.insert(np.asarray(ref, dtype=float), 0, regularization)
    coefficients, _ = nnls(augmented_data, augmented_ref)
    return np.asarray(coefficients, dtype=float)


def _paper_slsqp_weights(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    initial: np.ndarray,
    minkowski_order: int = 2,
) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    ref = np.asarray(ref, dtype=float)
    initial = np.asarray(initial, dtype=float)
    num_selected = data.shape[1]
    if num_selected == 1:
        return np.ones(1, dtype=float)
    if initial.shape != (num_selected,) or float(np.sum(initial)) <= 1.0e-12:
        initial = np.full(num_selected, 1.0 / num_selected, dtype=float)
    else:
        initial = np.clip(initial, 0.0, None)
        initial /= max(float(initial.sum()), 1.0e-12)

    def objective(weights: np.ndarray) -> float:
        safe_weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        safe_sum = max(float(np.sum(safe_weights)), 1.0e-12)
        safe_weights = safe_weights / safe_sum
        residual = np.abs(ref - data @ safe_weights)
        return float(np.sum(residual**minkowski_order))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, None)] * num_selected,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 300, "ftol": 1.0e-12, "disp": False},
    )
    weights = np.asarray(result.x if result.x is not None else initial, dtype=float)
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if float(weights.sum()) <= 1.0e-12:
        weights = initial
    return weights / max(float(weights.sum()), 1.0e-12)


def _paper_maxent_bfgs(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    atoms: list[str],
    theta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled_data, scaled_ref, atom_ranges = _paper_scaling_mean_center(
        data=data,
        ref=ref,
        atoms=atoms,
    )
    num_rows, num_candidates = scaled_data.shape
    if num_candidates <= 0:
        raise ValueError("MaxEnt BFGS requires at least one conformer.")
    if num_rows == 0:
        return (
            np.full(num_candidates, 1.0 / num_candidates, dtype=float),
            {
                "objective": float("nan"),
                "iterations": 0,
                "success": False,
                "message": "No scaled rows available.",
                "atom_ranges": atom_ranges,
            },
        )

    theta = max(float(theta), 1.0e-10)

    def objective(lagrange: np.ndarray) -> float:
        logits = -scaled_data.T @ lagrange
        return float(
            logsumexp(logits)
            - math.log(num_candidates)
            + np.dot(lagrange, scaled_ref)
            + theta * np.dot(lagrange, lagrange)
        )

    def gradient(lagrange: np.ndarray) -> np.ndarray:
        logits = -scaled_data.T @ lagrange
        weights = np.exp(logits - logsumexp(logits))
        return -scaled_data @ weights + scaled_ref + 2.0 * theta * lagrange

    initial = np.ones(num_rows, dtype=float)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1.0e-6, "disp": False},
    )
    lagrange = np.asarray(result.x if result.x is not None else initial, dtype=float)
    logits = -scaled_data.T @ lagrange
    weights = np.exp(logits - logsumexp(logits))
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if float(weights.sum()) <= 1.0e-12:
        weights = np.full(num_candidates, 1.0 / num_candidates, dtype=float)
    else:
        weights /= float(weights.sum())
    diagnostics = {
        "objective": float(objective(lagrange)),
        "iterations": int(getattr(result, "nit", 0)),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "atom_ranges": atom_ranges,
    }
    return weights, diagnostics


def _paper_atom_scores(
    *,
    data: np.ndarray,
    ref: np.ndarray,
    weights: np.ndarray,
    atoms: list[str],
) -> dict[str, float]:
    data = np.asarray(data, dtype=float)
    ref = np.asarray(ref, dtype=float)
    weights = np.asarray(weights, dtype=float)
    predictions = data @ weights
    scores: dict[str, float] = {}
    for atom in PAPER_CS_REWEIGHTING_ATOMS:
        atom_mask = np.asarray([row_atom == atom for row_atom in atoms], dtype=bool)
        if not atom_mask.any():
            continue
        atom_ref = ref[atom_mask]
        atom_pred = predictions[atom_mask]
        finite = np.isfinite(atom_ref) & np.isfinite(atom_pred)
        if not finite.any():
            scores[atom] = 0.0
            continue
        residual_sum_squares = float(np.sum(np.square(atom_pred[finite] - atom_ref[finite])))
        total_sum_squares = float(
            np.sum(np.square(atom_ref[finite] - np.mean(atom_ref[finite])))
        )
        if total_sum_squares <= 1.0e-12:
            scores[atom] = 1.0 if residual_sum_squares <= 1.0e-12 else 0.0
        else:
            scores[atom] = float(1.0 - residual_sum_squares / total_sum_squares)
    return scores


def _paper_total_score(atom_scores: dict[str, float]) -> float:
    if not atom_scores:
        return 0.0
    scores = [float(score) for score in atom_scores.values()]
    if all(score > 0.0 for score in scores):
        total = 1.0
        for score in scores:
            total *= score
        return float(total)
    negative_scores = [score for score in scores if score < 0.0]
    if negative_scores:
        return float(sum(negative_scores))
    return float(sum(scores))


def _fit_robust_bme_mirror_teacher(
    *,
    design: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    lambda_reg: float,
    likelihood: str,
    max_iter: int = 1500,
    learning_rate: float = 0.08,
    tolerance: float = 1.0e-8,
    weight_floor: float = 1.0e-12,
) -> WeightSolution:
    """Fit robust BME weights with entropic mirror descent on the simplex."""

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    prior = np.asarray(prior, dtype=float)
    num_candidates = int(prior.size)
    if num_candidates <= 0:
        raise ValueError("Robust BME mirror teacher requires candidates.")
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior = np.clip(prior, weight_floor, None)
    prior = prior / max(float(prior.sum()), weight_floor)
    weights = prior.copy()
    theta = max(float(lambda_reg), 0.0)
    kind = str(likelihood).strip().lower()
    if kind in {"student", "student_t", "student-t"}:
        kind = "student"
    elif kind in {"goodbad", "good_bad", "good-bad"}:
        kind = "good_bad"
    else:
        raise ValueError(f"Unsupported robust BME likelihood: {likelihood}")

    row_count = max(int(target.size), 1)
    previous_objective = float("inf")
    converged = False
    last_diagnostics: dict[str, float] = {}

    for iteration in range(1, max_iter + 1):
        residual = design @ weights - target
        nll, influence, likelihood_diagnostics = _robust_bme_loss_and_influence(
            residual,
            likelihood=kind,
        )
        safe_weights = np.clip(weights, weight_floor, None)
        kl = float(np.sum(safe_weights * (np.log(safe_weights) - np.log(prior))))
        objective = float(nll + theta * kl)
        gradient = (design.T @ influence) / float(row_count)
        if theta > 0.0:
            gradient = gradient + theta * (np.log(safe_weights / prior) + 1.0)
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

        logits = np.log(safe_weights) - learning_rate * gradient
        logits -= float(np.max(logits))
        updated = np.exp(logits)
        updated = np.clip(updated, weight_floor, None)
        updated /= max(float(updated.sum()), weight_floor)

        step_norm = float(np.linalg.norm(updated - weights, ord=1))
        if step_norm < tolerance or abs(previous_objective - objective) < tolerance:
            weights = updated
            previous_objective = objective
            last_diagnostics = likelihood_diagnostics
            converged = True
            break
        weights = updated
        previous_objective = objective
        last_diagnostics = likelihood_diagnostics

    residual = design @ weights - target
    nll, _, likelihood_diagnostics = _robust_bme_loss_and_influence(
        residual,
        likelihood=kind,
    )
    safe_weights = np.clip(weights, weight_floor, None)
    kl = float(np.sum(safe_weights * (np.log(safe_weights) - np.log(prior))))
    energy = float(nll + theta * kl)
    diagnostics = {
        "learning_rate": float(learning_rate),
        "lambda_reg": float(theta),
        "robust_bme_nll": float(nll),
        "robust_bme_kl": float(kl),
        "robust_likelihood_code": 1.0 if kind == "student" else 2.0,
        **last_diagnostics,
        **likelihood_diagnostics,
    }
    return WeightSolution(
        method=f"{kind}_bme_mirror",
        weights=weights,
        energy=energy,
        converged=converged,
        iterations=int(iteration),
        diagnostics=diagnostics,
    )


def _robust_bme_loss_and_influence(
    residual: np.ndarray,
    *,
    likelihood: str,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return robust negative log-likelihood and dL/dr influence."""

    residual = np.asarray(residual, dtype=float)
    kind = str(likelihood).strip().lower()
    if kind == "student":
        nu = 4.0
        nll_rows = 0.5 * (nu + 1.0) * np.log1p(np.square(residual) / nu)
        influence = ((nu + 1.0) * residual) / (nu + np.square(residual))
        return (
            float(np.mean(nll_rows)),
            np.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0),
            {
                "student_nu": float(nu),
                "robust_abs_residual_mean": float(np.mean(np.abs(residual))),
            },
        )
    if kind == "good_bad":
        outlier_prob = 0.08
        bad_scale = 8.0
        good_log = np.log(max(1.0 - outlier_prob, 1.0e-8)) - 0.5 * np.square(residual)
        bad_log = (
            np.log(max(outlier_prob, 1.0e-8))
            - np.log(max(bad_scale, 1.0e-8))
            - 0.5 * np.square(residual / bad_scale)
        )
        max_log = np.maximum(good_log, bad_log)
        log_mix = max_log + np.log(
            np.exp(good_log - max_log) + np.exp(bad_log - max_log)
        )
        gamma = np.exp(good_log - log_mix)
        influence = gamma * residual + (1.0 - gamma) * residual / (bad_scale * bad_scale)
        return (
            float(-np.mean(log_mix)),
            np.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0),
            {
                "good_bad_outlier_prob": float(outlier_prob),
                "good_bad_bad_scale": float(bad_scale),
                "inlier_responsibility_mean": float(np.mean(gamma)),
                "robust_abs_residual_mean": float(np.mean(np.abs(residual))),
            },
        )
    raise ValueError(f"Unsupported robust BME likelihood: {likelihood}")


def _masked_row_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    safe_values = np.where(mask, values, 0.0)
    counts = mask.sum(axis=1).astype(float)
    sums = safe_values.sum(axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=counts > 0.0,
    )


def _cs_reweighting_predictions(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    row_offsets: np.ndarray,
) -> np.ndarray:
    predictions: list[float] = []
    for row_index in range(values.shape[0]):
        corrected = values[row_index] + row_offsets[row_index]
        valid = mask[row_index] & np.isfinite(corrected)
        weighted_mass = float(np.dot(valid.astype(float), weights))
        if weighted_mass <= 1.0e-12:
            predictions.append(float("nan"))
            continue
        predictions.append(float(np.dot(np.where(valid, corrected, 0.0), weights) / weighted_mass))
    return np.asarray(predictions, dtype=float)


def _effective_sample_size(weights: np.ndarray) -> float:
    denom = float(np.sum(np.square(weights)))
    if denom <= 1.0e-12:
        return 0.0
    return float(1.0 / denom)


def _weight_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1.0e-12, None)
    return float(-np.sum(clipped * np.log(clipped)))


def _json_dumps_clean(payload: Any) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    return json.dumps(clean(payload), sort_keys=True)


def _entity_audit_rows(
    prediction_frame: pd.DataFrame,
    *,
    entity_uid: str,
    candidate_count: int,
    missing_sidecar_count: int,
    ess: float,
    entropy: float,
    energy: float,
    converged: bool,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom_family, frame in prediction_frame.groupby("atom_family", dropna=False):
        target = pd.to_numeric(frame["target_value"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(frame["teacher_mean"], errors="coerce").to_numpy(float)
        residual = pred - target
        rows.append(
            {
                "entity_uid": entity_uid,
                "atom_family": str(atom_family),
                "rows": int(len(frame)),
                "candidate_count": int(candidate_count),
                "missing_sidecar_count": int(missing_sidecar_count),
                "teacher_mae": float(np.nanmean(np.abs(residual))),
                "teacher_rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
                "teacher_bias": float(np.nanmean(residual)),
                "teacher_ccc": _ccc(pred, target),
                "teacher_ess": float(ess),
                "teacher_entropy": float(entropy),
                "teacher_energy": float(energy),
                "teacher_converged": bool(converged),
                "teacher_method": method,
            }
        )
    if rows:
        all_target = pd.to_numeric(
            prediction_frame["target_value"],
            errors="coerce",
        ).to_numpy(float)
        all_pred = pd.to_numeric(
            prediction_frame["teacher_mean"],
            errors="coerce",
        ).to_numpy(float)
        residual = all_pred - all_target
        rows.append(
            {
                "entity_uid": entity_uid,
                "atom_family": "All",
                "rows": int(len(prediction_frame)),
                "candidate_count": int(candidate_count),
                "missing_sidecar_count": int(missing_sidecar_count),
                "teacher_mae": float(np.nanmean(np.abs(residual))),
                "teacher_rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
                "teacher_bias": float(np.nanmean(residual)),
                "teacher_ccc": _ccc(all_pred, all_target),
                "teacher_ess": float(ess),
                "teacher_entropy": float(entropy),
                "teacher_energy": float(energy),
                "teacher_converged": bool(converged),
                "teacher_method": method,
            }
        )
    return rows


def _normalize_shift_manifest(frame: pd.DataFrame, *, base_dir: Path) -> pd.DataFrame:
    required = {"entity_uid", "chemical_shift_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "BioEmu UCBShift manifest requires columns "
            f"{sorted(required)}; missing {sorted(missing)}."
        )
    output = frame.copy()
    output["entity_uid"] = output["entity_uid"].astype(str)
    if "candidate_id" not in output.columns:
        output["candidate_id"] = [
            f"{entity_uid}:sample_{index:05d}"
            for index, entity_uid in enumerate(output["entity_uid"].astype(str))
        ]
    if "sample_index" not in output.columns:
        output["sample_index"] = output.groupby("entity_uid").cumcount()
    if "structure_path" not in output.columns:
        output["structure_path"] = ""
    if "chemical_shift_format" not in output.columns:
        output["chemical_shift_format"] = "ucbshift"
    output["chemical_shift_path"] = output["chemical_shift_path"].map(
        lambda value: str(_resolve_path(value, base_dir))
    )
    output["structure_path"] = output["structure_path"].map(
        lambda value: "" if pd.isna(value) or not str(value) else str(_resolve_path(value, base_dir))
    )
    output["sample_index"] = pd.to_numeric(
        output["sample_index"],
        errors="coerce",
    ).fillna(0).astype(int)
    return output.sort_values(
        ["entity_uid", "sample_index", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def _load_target_manifest(
    *,
    data_root: Path,
    target_manifest_path: str | Path | None,
) -> dict[str, Path]:
    if target_manifest_path is not None:
        frame = _read_table(target_manifest_path)
        if {"entity_uid", "target_bundle_path"} - set(frame.columns):
            raise ValueError(
                "target manifest must include entity_uid and target_bundle_path."
            )
        base = Path(target_manifest_path).parent
        return {
            str(row.entity_uid): _resolve_path(row.target_bundle_path, base)
            for row in frame.itertuples(index=False)
        }
    from atypemu.datasets import IntegratedDataRegistry

    registry = IntegratedDataRegistry.from_data_root(data_root)
    teacher_examples = registry.load_teacher_examples()
    mapping: dict[str, Path] = {}
    for row in teacher_examples.itertuples(index=False):
        path_value = getattr(row, "target_bundle_path", None)
        if path_value is None or pd.isna(path_value):
            continue
        mapping[str(getattr(row, "entity_uid"))] = _resolve_path(
            path_value,
            data_root,
        )
    return mapping


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".json":
        payload = json.loads(table_path.read_text())
        return pd.DataFrame(payload if isinstance(payload, list) else payload.get("rows", []))
    return pd.read_csv(table_path)


def _write_table_pair(frame: pd.DataFrame, parquet_path: Path) -> dict[str, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path)
    tsv_path = parquet_path.with_suffix(".tsv")
    frame.to_csv(tsv_path, sep="\t", index=False)
    return {
        parquet_path.stem + "_parquet": str(parquet_path),
        parquet_path.stem + "_tsv": str(tsv_path),
    }


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _atom_family(atom_id: str) -> str:
    atom = canonical_atom_name(atom_id)
    if atom == "H":
        return "HN"
    if atom == "C":
        return "C'"
    return atom


def _ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    mask = np.isfinite(predictions) & np.isfinite(targets)
    if int(mask.sum()) < 2:
        return float("nan")
    x = predictions[mask]
    y = targets[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    denom = vx + vy + (x_mean - y_mean) ** 2
    if denom <= 1.0e-12:
        return float("nan")
    return float(2.0 * cov / denom)


def _render_teacher_weight_plot(frame: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No teacher weights", ha="center", va="center")
            axis.set_axis_off()
    else:
        summary = (
            frame.groupby("entity_uid", dropna=False)
            .agg(
                sample_count=("teacher_weight", "size"),
                max_weight=("teacher_weight", "max"),
                ess=("teacher_ess", "first"),
                entropy=("teacher_entropy", "first"),
            )
            .reset_index()
            .sort_values("ess", kind="stable")
        )
        axes[0].barh(summary["entity_uid"], summary["ess"], color="#0f766e")
        axes[0].set_xlabel("teacher ESS")
        axes[0].set_title("BioEmu conformer population support")
        top = frame.sort_values("teacher_weight", ascending=False, kind="stable").head(24)
        labels = [
            f"{row.entity_uid}\n{row.candidate_id}"
            for row in top.itertuples(index=False)
        ]
        axes[1].barh(labels[::-1], top["teacher_weight"].iloc[::-1], color="#2563eb")
        axes[1].set_xlabel("Teacher population weight")
        axes[1].set_title("Top weighted conformers")
        axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_teacher_audit_plot(frame: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No teacher audit rows", ha="center", va="center")
            axis.set_axis_off()
    else:
        family = (
            frame.loc[frame["atom_family"].astype(str).ne("All")]
            .groupby("atom_family", dropna=False)
            .agg(
                rows=("rows", "sum"),
                mae=("teacher_mae", "mean"),
                ccc=("teacher_ccc", "mean"),
            )
            .reset_index()
            .sort_values("mae", ascending=False, kind="stable")
        )
        x = np.arange(len(family))
        axes[0].bar(x, family["mae"], color="#dc2626")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(family["atom_family"], rotation=25, ha="right")
        axes[0].set_ylabel("Teacher MAE")
        axes[0].set_title("UCBShift2.0 teacher posterior mean error")
        axes[1].bar(x, family["ccc"], color="#7c3aed")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(family["atom_family"], rotation=25, ha="right")
        axes[1].set_ylabel("Teacher CCC")
        axes[1].set_title("Family-wise teacher CCC")
        axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
