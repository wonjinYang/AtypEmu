"""Export source-aware CS-reweighting teachers for BioEmu latent training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atypemu.energy import LikelihoodAggregator
from atypemu.training.data import parse_chemical_shift_target_id
from atypemu.training.materialize import (
    namespaced_integrated_artifact_path,
    resolve_existing_path,
)
from atypemu.types import CandidatePool, EnergyBreakdown, NMRTargetBundle, ObservableBundle


SOURCE_MASS_COLUMNS = {
    "AF3": "source_mass_af3",
    "BioEmu": "source_mass_bioemu",
    "CALVADOS2": "source_mass_calvados2",
}

ADAPTIVE_LAMBDA_COLUMNS = [
    "adaptive_lambda_requested_floor",
    "adaptive_lambda_selected",
    "adaptive_lambda_fit_count",
    "adaptive_lambda_floor_met",
]

DEFAULT_FAMILY_TOLERANCES = {
    "HN": 0.02,
    "C'": 0.05,
    "CA": 0.10,
    "CB": 0.10,
    "N": 0.20,
}

PREDICTION_COLUMNS = [
    "entity_uid",
    "bmrb_id",
    "target_id",
    "seq_id",
    "atom_family",
    "target_value",
    "teacher_mean",
    "teacher_ess",
    "teacher_entropy",
    "source_mass_bioemu",
    "source_mass_af3",
    "source_mass_calvados2",
    "teacher_policy",
    "top_source",
    "top_source_mass",
    "bioemu_only_teacher_mean",
    "bioemu_only_delta",
    "valid_candidate_fraction",
    "af3_candidate_count",
    "bioemu_candidate_count",
    "calvados2_candidate_count",
    *ADAPTIVE_LAMBDA_COLUMNS,
]

GUARD_COLUMNS = [
    "top_atom_family",
    "outlier_entities",
    "recommended_guard_action",
    "quality_outlier_fraction",
    "repeated_quality_outlier",
    "guard_reasons",
    "teacher_ess",
    "top_source",
    "top_source_mass",
    "missing_sidecar_fraction",
]

AUDIT_COLUMNS = [
    "entity_uid",
    "bmrb_id",
    "teacher_policy",
    "candidate_count",
    "teacher_ess",
    "teacher_entropy",
    "top_source",
    "top_source_mass",
    "missing_sidecar_fraction",
    "source_mass_bioemu",
    "source_mass_af3",
    "source_mass_calvados2",
    "af3_candidate_count",
    "bioemu_candidate_count",
    "calvados2_candidate_count",
    *ADAPTIVE_LAMBDA_COLUMNS,
]


@dataclass(slots=True)
class CSReweightingTeacherExportSummary:
    """Summary for one source-aware teacher export."""

    status: str
    prediction_path: str
    guard_path: str
    audit_path: str
    prediction_rows: int
    guard_rows: int
    skipped_examples: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize the summary."""

        return {
            "status": self.status,
            "prediction_path": self.prediction_path,
            "guard_path": self.guard_path,
            "audit_path": self.audit_path,
            "prediction_rows": self.prediction_rows,
            "guard_rows": self.guard_rows,
            "skipped_examples": self.skipped_examples,
        }


def export_cs_reweighting_teacher_for_bioemu(
    *,
    data_root: str | Path,
    integrated_root: str | Path,
    output_dir: str | Path,
    teacher_namespace: str | None = None,
    bioemu_only_namespace: str | None = None,
    selected_splits: list[str] | None = None,
    teacher_policy: str = "multisource_cs_reweighting",
    min_teacher_ess: float = 20.0,
    non_bioemu_source_mass_threshold: float = 0.90,
    family_tolerances: dict[str, float] | None = None,
) -> CSReweightingTeacherExportSummary:
    """Export materialized CS-reweighting teachers as BioEmu trainer inputs."""

    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    repo_root = data_root_path.parent
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    teacher_examples_path = integrated_root_path / "datasets" / "teacher_examples.parquet"
    teacher_examples = pd.read_parquet(teacher_examples_path)
    if selected_splits:
        teacher_examples = teacher_examples.loc[
            teacher_examples["split"].isin(selected_splits)
        ].copy()

    bioemu_only_means = (
        _load_teacher_mean_lookup(
            teacher_examples=teacher_examples,
            data_root=data_root_path,
            repo_root=repo_root,
            namespace=bioemu_only_namespace,
        )
        if bioemu_only_namespace
        else {}
    )
    tolerances = dict(DEFAULT_FAMILY_TOLERANCES)
    if family_tolerances:
        tolerances.update({str(key): float(value) for key, value in family_tolerances.items()})

    prediction_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    skipped_examples = 0

    for row in teacher_examples.to_dict(orient="records"):
        loaded = _load_materialized_teacher(
            row=row,
            data_root=data_root_path,
            repo_root=repo_root,
            namespace=teacher_namespace,
        )
        if loaded is None:
            skipped_examples += 1
            continue
        bundle, pool, observables, breakdown = loaded
        if observables.chemical_shifts is None:
            skipped_examples += 1
            continue

        entity_uid = str(row["entity_uid"])
        bmrb_id = str(row["bmrb_id"])
        weights = _normalized_weights(breakdown.weights)
        source_masses = _source_masses(pool, weights)
        source_counts = _source_counts(pool)
        top_source, top_source_mass = _top_source(source_masses)
        missing_sidecar_fraction = _missing_sidecar_fraction(pool)
        adaptive_lambda_diagnostics = _adaptive_lambda_diagnostics(breakdown)
        entity_predictions = _prediction_rows_for_entity(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            bundle=bundle,
            observables=observables,
            weights=weights,
            source_masses=source_masses,
            source_counts=source_counts,
            breakdown=breakdown,
            teacher_policy=teacher_policy,
            top_source=top_source,
            top_source_mass=top_source_mass,
            bioemu_only_means=bioemu_only_means,
        )
        prediction_rows.extend(entity_predictions)
        audit_rows.append(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "teacher_policy": teacher_policy,
                "candidate_count": len(pool.records),
                "teacher_ess": float(breakdown.ess),
                "teacher_entropy": float(breakdown.entropy),
                "top_source": top_source,
                "top_source_mass": top_source_mass,
                "missing_sidecar_fraction": missing_sidecar_fraction,
                **source_masses,
                **{f"{source.lower()}_candidate_count": count for source, count in source_counts.items()},
                **adaptive_lambda_diagnostics,
            }
        )
        guard_rows.extend(
            _guard_rows_for_entity(
                entity_uid=entity_uid,
                predictions=entity_predictions,
                teacher_ess=float(breakdown.ess),
                min_teacher_ess=min_teacher_ess,
                top_source=top_source,
                top_source_mass=top_source_mass,
                non_bioemu_source_mass_threshold=non_bioemu_source_mass_threshold,
                missing_sidecar_fraction=missing_sidecar_fraction,
                family_tolerances=tolerances,
            )
        )

    prediction_path = output_dir_path / "bioemu_cs_reweighting_teacher_predictions.parquet"
    guard_path = output_dir_path / "bioemu_cs_reweighting_teacher_guard.parquet"
    audit_path = output_dir_path / "bioemu_cs_reweighting_teacher_source_audit.parquet"
    _write_parquet_with_schema(prediction_rows, PREDICTION_COLUMNS, prediction_path)
    _write_parquet_with_schema(guard_rows, GUARD_COLUMNS, guard_path)
    _write_parquet_with_schema(audit_rows, AUDIT_COLUMNS, audit_path)
    summary = CSReweightingTeacherExportSummary(
        status="ok",
        prediction_path=str(prediction_path),
        guard_path=str(guard_path),
        audit_path=str(audit_path),
        prediction_rows=len(prediction_rows),
        guard_rows=len(guard_rows),
        skipped_examples=skipped_examples,
    )
    (output_dir_path / "bioemu_cs_reweighting_teacher_export_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2, sort_keys=True)
    )
    return summary


def _load_teacher_mean_lookup(
    *,
    teacher_examples: pd.DataFrame,
    data_root: Path,
    repo_root: Path,
    namespace: str,
) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in teacher_examples.to_dict(orient="records"):
        loaded = _load_materialized_teacher(
            row=row,
            data_root=data_root,
            repo_root=repo_root,
            namespace=namespace,
        )
        if loaded is None:
            continue
        _, _, observables, breakdown = loaded
        if observables.chemical_shifts is None:
            continue
        weights = _normalized_weights(breakdown.weights)
        matrix = observables.chemical_shifts
        for row_index, target_id in enumerate(matrix.target_ids):
            prediction, _ = LikelihoodAggregator.normalized_prediction(
                matrix.values[row_index],
                matrix.mask[row_index],
                weights,
                matrix.transform,
            )
            if prediction is not None and math.isfinite(float(prediction)):
                lookup[(str(row["entity_uid"]), str(target_id))] = float(prediction)
    return lookup


def _write_parquet_with_schema(
    rows: list[dict[str, Any]],
    columns: list[str],
    path: Path,
) -> None:
    """Write a parquet table while preserving columns for empty exports."""

    pd.DataFrame(rows).reindex(columns=columns).to_parquet(path)


def _load_materialized_teacher(
    *,
    row: dict[str, Any],
    data_root: Path,
    repo_root: Path,
    namespace: str | None,
) -> tuple[NMRTargetBundle, CandidatePool, ObservableBundle, EnergyBreakdown] | None:
    bundle_path = resolve_existing_path(
        row["target_bundle_path"],
        data_root=data_root,
        repo_root=repo_root,
    )
    candidate_pool_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["candidate_pool_path"],
            namespace=namespace,
        )
    )
    observable_bundle_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["observable_bundle_path"],
            namespace=namespace,
        )
    )
    breakdown_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["teacher_breakdown_path"],
            namespace=namespace,
        )
    )
    if (
        bundle_path is None
        or not bundle_path.exists()
        or not candidate_pool_path.exists()
        or not observable_bundle_path.exists()
        or not breakdown_path.exists()
    ):
        return None
    return (
        NMRTargetBundle.from_json(bundle_path),
        CandidatePool.from_jsonl(candidate_pool_path),
        ObservableBundle.from_npz(observable_bundle_path),
        EnergyBreakdown.from_json(breakdown_path),
    )


def _prediction_rows_for_entity(
    *,
    entity_uid: str,
    bmrb_id: str,
    bundle: NMRTargetBundle,
    observables: ObservableBundle,
    weights: np.ndarray,
    source_masses: dict[str, float],
    source_counts: dict[str, int],
    breakdown: EnergyBreakdown,
    teacher_policy: str,
    top_source: str,
    top_source_mass: float,
    bioemu_only_means: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    matrix = observables.chemical_shifts
    if matrix is None:
        return []
    rows: list[dict[str, Any]] = []
    target_by_id = {target.target_id(): target for target in bundle.chemical_shifts}
    for row_index, target_id in enumerate(matrix.target_ids):
        prediction, _ = LikelihoodAggregator.normalized_prediction(
            matrix.values[row_index],
            matrix.mask[row_index],
            weights,
            matrix.transform,
        )
        if prediction is None or not math.isfinite(float(prediction)):
            continue
        parsed = parse_chemical_shift_target_id(str(target_id))
        target = target_by_id.get(str(target_id))
        bioemu_only_mean = bioemu_only_means.get((entity_uid, str(target_id)))
        row = {
            "entity_uid": entity_uid,
            "bmrb_id": bmrb_id,
            "target_id": str(target_id),
            "seq_id": parsed["residue_index"],
            "atom_family": parsed["atom_family"],
            "target_value": float(target.value) if target is not None else float("nan"),
            "teacher_mean": float(prediction),
            "teacher_ess": float(breakdown.ess),
            "teacher_entropy": float(breakdown.entropy),
            "teacher_policy": teacher_policy,
            "top_source": top_source,
            "top_source_mass": float(top_source_mass),
            "bioemu_only_teacher_mean": (
                float(bioemu_only_mean) if bioemu_only_mean is not None else float("nan")
            ),
            "bioemu_only_delta": (
                float(prediction - bioemu_only_mean)
                if bioemu_only_mean is not None
                else float("nan")
            ),
            "valid_candidate_fraction": float(matrix.mask[row_index].mean()),
        }
        row.update(source_masses)
        row.update({f"{source.lower()}_candidate_count": count for source, count in source_counts.items()})
        row.update(_adaptive_lambda_diagnostics(breakdown))
        rows.append(row)
    return rows


def _guard_rows_for_entity(
    *,
    entity_uid: str,
    predictions: list[dict[str, Any]],
    teacher_ess: float,
    min_teacher_ess: float,
    top_source: str,
    top_source_mass: float,
    non_bioemu_source_mass_threshold: float,
    missing_sidecar_fraction: float,
    family_tolerances: dict[str, float],
) -> list[dict[str, Any]]:
    reasons_by_family: dict[str, set[str]] = {}
    if teacher_ess < min_teacher_ess:
        for row in predictions:
            family = str(row.get("atom_family"))
            reasons_by_family.setdefault(family, set()).add("low_teacher_ess")
    if top_source != "BioEmu" and top_source_mass > non_bioemu_source_mass_threshold:
        for row in predictions:
            family = str(row.get("atom_family"))
            reasons_by_family.setdefault(family, set()).add("non_bioemu_source_dominance")
    if missing_sidecar_fraction > 0.0:
        for row in predictions:
            family = str(row.get("atom_family"))
            reasons_by_family.setdefault(family, set()).add("missing_sidecar_fraction")
    for row in predictions:
        family = str(row.get("atom_family"))
        delta = _safe_float(row.get("bioemu_only_delta"))
        tolerance = family_tolerances.get(family)
        if tolerance is not None and math.isfinite(delta) and abs(delta) > tolerance:
            reasons_by_family.setdefault(family, set()).add("bioemu_support_gap")
    return [
        {
            "top_atom_family": family,
            "outlier_entities": entity_uid,
            "recommended_guard_action": "exclude_mean_loss",
            "quality_outlier_fraction": 1.0,
            "repeated_quality_outlier": True,
            "guard_reasons": ",".join(sorted(reasons)),
            "teacher_ess": teacher_ess,
            "top_source": top_source,
            "top_source_mass": top_source_mass,
            "missing_sidecar_fraction": missing_sidecar_fraction,
        }
        for family, reasons in sorted(reasons_by_family.items())
        if family and family != "None" and reasons
    ]


def _source_masses(pool: CandidatePool, weights: np.ndarray) -> dict[str, float]:
    masses = {column: 0.0 for column in SOURCE_MASS_COLUMNS.values()}
    for record, weight in zip(pool.records, weights, strict=False):
        column = SOURCE_MASS_COLUMNS.get(str(record.source))
        if column:
            masses[column] += float(weight)
    return masses


def _source_counts(pool: CandidatePool) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_MASS_COLUMNS}
    for record in pool.records:
        source = str(record.source)
        if source in counts:
            counts[source] += 1
    return counts


def _top_source(source_masses: dict[str, float]) -> tuple[str, float]:
    source_by_column = {value: key for key, value in SOURCE_MASS_COLUMNS.items()}
    if not source_masses:
        return "", 0.0
    column, mass = max(source_masses.items(), key=lambda item: item[1])
    return source_by_column.get(column, column), float(mass)


def _missing_sidecar_fraction(pool: CandidatePool) -> float:
    if not pool.records:
        return 0.0
    missing = sum(1 for record in pool.records if not record.chemical_shift_path)
    return float(missing / len(pool.records))


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(safe.sum())
    if total <= 0.0:
        return np.full(len(safe), 1.0 / max(len(safe), 1), dtype=float)
    return safe / total


def _adaptive_lambda_diagnostics(breakdown: EnergyBreakdown) -> dict[str, float]:
    diagnostics = breakdown.diagnostics or {}
    return {
        "adaptive_lambda_requested_floor": _safe_float(
            diagnostics.get("adaptive_lambda_requested_floor")
        ),
        "adaptive_lambda_selected": _safe_float(
            diagnostics.get("adaptive_lambda_selected")
        ),
        "adaptive_lambda_fit_count": _safe_float(
            diagnostics.get("adaptive_lambda_fit_count")
        ),
        "adaptive_lambda_floor_met": _safe_float(
            diagnostics.get("adaptive_lambda_floor_met")
        ),
    }


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")
