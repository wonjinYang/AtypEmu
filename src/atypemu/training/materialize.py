"""Offline teacher materialization for staged AtypEmu training."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atypemu.adapters.cnnls_adapter import build_prediction_map, normalize_candidate_key
from atypemu.cli.common import save_weights_csv
from atypemu.datasets import IntegratedDataRegistry
from atypemu.energy import MultiObservablePosteriorEnergy
from atypemu.integrated import export_training_dataframes
from atypemu.observables import ChemicalShiftMatrixBuilder, JCouplingHead, NOEHead
from atypemu.reweighting import EuclideanSimplexReweighter, MaxEntReweighter
from atypemu.structures import (
    build_candidate_pool,
    infer_candidate_metadata,
    load_stage_manifest,
)
from atypemu.training.config import TeacherMaterializationConfig
from atypemu.types import (
    CandidatePool,
    CandidateRecord,
    NMRTargetBundle,
    ObservableBundle,
    WeightSolution,
)


@dataclass(slots=True)
class MaterializedTeacherResult:
    """One per-example teacher materialization outcome."""

    entity_uid: str
    bmrb_id: str
    status: str
    message: str
    candidate_count: int = 0
    chemical_shift_valid_fraction: float = 0.0
    j_coupling_valid_fraction: float = 0.0
    noe_valid_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-compatible dictionary."""
        return {
            "entity_uid": self.entity_uid,
            "bmrb_id": self.bmrb_id,
            "status": self.status,
            "message": self.message,
            "candidate_count": self.candidate_count,
            "chemical_shift_valid_fraction": self.chemical_shift_valid_fraction,
            "j_coupling_valid_fraction": self.j_coupling_valid_fraction,
            "noe_valid_fraction": self.noe_valid_fraction,
        }


def _canonical_bmrb_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    text = text.strip().lower()
    if text.isdigit():
        return f"bmr{text}"
    return text


def _canonical_entity_uid(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"bmrb:bmr{text}"
    if ":" not in text and text.lower().startswith("bmr"):
        return f"bmrb:{text.lower()}"
    return text.lower()


def materialize_teacher_examples(
    data_root: str | Path,
    integrated_root: str | Path,
    config: TeacherMaterializationConfig,
) -> dict[str, Any]:
    """Materialize offline teacher artifacts for integrated BMRB examples.

    Args:
        data_root: Repository ``data`` root.
        integrated_root: Integrated workspace root such as ``data/integrated``.
        config: Teacher materialization configuration.

    Returns:
        Summary dictionary describing the materialized teacher outputs.
    """
    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    repo_root = data_root_path.parent
    registry = IntegratedDataRegistry.from_data_root(data_root_path)
    teacher_examples = registry.load_teacher_examples()
    selected = teacher_examples.loc[
        teacher_examples["split"].isin(config.selected_splits)
    ].copy()
    if config.excluded_bmrb_ids:
        excluded = {
            _canonical_bmrb_id(bmrb_id) for bmrb_id in config.excluded_bmrb_ids
        }
        selected = selected.loc[
            ~selected["bmrb_id"].map(_canonical_bmrb_id).isin(excluded)
        ].copy()
    if config.selected_entity_uids:
        selected_entity_uids = {
            _canonical_entity_uid(entity_uid)
            for entity_uid in config.selected_entity_uids
        }
        selected = selected.loc[
            selected["entity_uid"].map(_canonical_entity_uid).isin(selected_entity_uids)
        ].copy()
    if config.selected_bmrb_ids:
        selected_bmrb_ids = {
            _canonical_bmrb_id(bmrb_id) for bmrb_id in config.selected_bmrb_ids
        }
        selected = selected.loc[
            selected["bmrb_id"].map(_canonical_bmrb_id).isin(selected_bmrb_ids)
        ].copy()
    if config.max_examples is not None:
        selected = selected.head(config.max_examples)

    selected_rows = selected.to_dict(orient="records")
    total_rows = len(selected_rows)
    workers = min(max(1, int(config.materialization_worker_count)), max(total_rows, 1))
    print(
        "[materialize] "
        f"{config.artifact_namespace or 'default'} "
        f"requested={total_rows} workers={workers} "
        f"parse_candidate_structures={config.parse_candidate_structures}",
        flush=True,
    )
    results = _materialize_rows(
        rows=selected_rows,
        data_root=data_root_path,
        integrated_root=integrated_root_path,
        repo_root=repo_root,
        config=config,
        workers=workers,
    )

    summary_path = (
        integrated_root_path
        / "teachers"
        / (
            config.artifact_namespace
            if config.artifact_namespace
            else "materialization_summary.json"
        )
    )
    if config.artifact_namespace:
        summary_path = summary_path / "materialization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "selected_splits": list(config.selected_splits),
        "selected_sources": list(config.selected_sources),
        "excluded_bmrb_ids": list(config.excluded_bmrb_ids),
        "selected_entity_uids": list(config.selected_entity_uids),
        "selected_bmrb_ids": list(config.selected_bmrb_ids),
        "artifact_namespace": config.artifact_namespace,
        "prior_policy": config.prior_policy,
        "requested_examples": len(selected),
        "written_examples": sum(result.status == "written" for result in results),
        "skipped_examples": sum(
            result.status.startswith("skipped") for result in results
        ),
        "failed_examples": sum(result.status == "failed" for result in results),
        "results": [result.as_dict() for result in results],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    if config.refresh_training_bundle:
        export_training_dataframes(
            data_root=data_root_path,
            integrated_root=integrated_root_path,
            refresh=True,
        )

    return {
        "summary_path": str(summary_path),
        "requested_examples": len(selected),
        "written_examples": sum(result.status == "written" for result in results),
        "skipped_examples": sum(
            result.status.startswith("skipped") for result in results
        ),
        "failed_examples": sum(result.status == "failed" for result in results),
    }


def _materialize_rows(
    rows: list[dict[str, Any]],
    data_root: Path,
    integrated_root: Path,
    repo_root: Path,
    config: TeacherMaterializationConfig,
    workers: int,
) -> list[MaterializedTeacherResult]:
    """Materialize rows sequentially or with entry-level multiprocessing."""

    if workers <= 1 or len(rows) <= 1:
        return [
            _materialize_one_row_for_pool(
                row_index=index,
                total_rows=len(rows),
                row=row,
                data_root=data_root,
                integrated_root=integrated_root,
                repo_root=repo_root,
                config=config,
            )
            for index, row in enumerate(rows, start=1)
        ]

    ordered_results: list[MaterializedTeacherResult | None] = [None] * len(rows)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _materialize_one_row_for_pool,
                row_index=index,
                total_rows=len(rows),
                row=row,
                data_root=data_root,
                integrated_root=integrated_root,
                repo_root=repo_root,
                config=config,
            ): index
            for index, row in enumerate(rows, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            ordered_results[index - 1] = result
    return [result for result in ordered_results if result is not None]


def _materialize_one_row_for_pool(
    row_index: int,
    total_rows: int,
    row: dict[str, Any],
    data_root: Path,
    integrated_root: Path,
    repo_root: Path,
    config: TeacherMaterializationConfig,
) -> MaterializedTeacherResult:
    """Materialize one row and emit process-safe progress lines."""

    namespace = config.artifact_namespace or "default"
    bmrb_id = row.get("bmrb_id")
    print(f"[materialize] {namespace} {row_index}/{total_rows} {bmrb_id}", flush=True)
    try:
        result = materialize_teacher_row(
            row=row,
            data_root=data_root,
            integrated_root=integrated_root,
            repo_root=repo_root,
            config=config,
        )
    except Exception as exc:
        if config.fail_fast:
            raise
        result = MaterializedTeacherResult(
            entity_uid=str(row["entity_uid"]),
            bmrb_id=str(row["bmrb_id"]),
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )
    print(
        f"[materialize] {namespace} {row_index}/{total_rows} {bmrb_id} "
        f"status={result.status} candidates={result.candidate_count}",
        flush=True,
    )
    return result


def materialize_teacher_row(
    row: dict[str, Any],
    data_root: Path,
    integrated_root: Path,
    repo_root: Path,
    config: TeacherMaterializationConfig,
) -> MaterializedTeacherResult:
    """Materialize one teacher example from one integrated dataframe row."""
    entity_uid = str(row["entity_uid"])
    bmrb_id = str(row["bmrb_id"])

    bundle_path = resolve_existing_path(
        row["target_bundle_path"],
        data_root=data_root,
        repo_root=repo_root,
    )
    if bundle_path is None or not bundle_path.exists():
        return MaterializedTeacherResult(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            status="skipped_missing_target",
            message=f"Missing target bundle: {row['target_bundle_path']}",
        )

    candidate_pool_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["candidate_pool_path"],
            namespace=config.artifact_namespace,
        )
    )
    observable_bundle_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["observable_bundle_path"],
            namespace=config.artifact_namespace,
        )
    )
    solution_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["teacher_solution_path"],
            namespace=config.artifact_namespace,
        )
    )
    breakdown_path = data_root / str(
        namespaced_integrated_artifact_path(
            row["teacher_breakdown_path"],
            namespace=config.artifact_namespace,
        )
    )
    summary_path = solution_path.with_name("materialization.json")
    weights_csv_path = solution_path.with_name("weights.csv")

    if (
        not config.refresh_outputs
        and candidate_pool_path.exists()
        and observable_bundle_path.exists()
        and solution_path.exists()
        and breakdown_path.exists()
    ):
        return MaterializedTeacherResult(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            status="skipped_existing",
            message="Teacher artifacts already exist.",
        )

    bundle = NMRTargetBundle.from_json(bundle_path)
    candidate_roots = json.loads(str(row["candidate_structure_roots"]))
    candidate_roots = {
        str(source): str(path)
        for source, path in candidate_roots.items()
        if str(source) in set(config.selected_sources)
    }
    pool = build_multisource_candidate_pool(
        candidate_roots=candidate_roots,
        bmrb_id=bmrb_id,
        data_root=data_root,
        repo_root=repo_root,
        chemical_shift_dir_templates=config.chemical_shift_dir_templates,
        chemical_shift_formats=config.chemical_shift_formats,
        max_candidates_per_example=config.max_candidates_per_example,
        parse_structures=config.parse_candidate_structures,
    )
    if not pool.records:
        return MaterializedTeacherResult(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            status="skipped_missing_pool",
            message="No candidate structures were discovered for this accession.",
        )

    if config.require_chemical_shifts and bundle.chemical_shifts:
        if not any(record.chemical_shift_path for record in pool.records):
            return MaterializedTeacherResult(
                entity_uid=entity_uid,
                bmrb_id=bmrb_id,
                status="skipped_missing_shifts",
                message="No candidate chemical-shift sidecars were available.",
                candidate_count=len(pool.records),
            )

    observables = build_observable_bundle(pool=pool, bundle=bundle)
    coverage = observable_valid_fraction(observables)
    if config.require_chemical_shifts and bundle.chemical_shifts:
        if (
            observables.chemical_shifts is None
            or not observables.chemical_shifts.mask.any()
        ):
            return MaterializedTeacherResult(
                entity_uid=entity_uid,
                bmrb_id=bmrb_id,
                status="skipped_no_cs_coverage",
                message="Chemical-shift targets exist but no valid predictions were found.",
                candidate_count=len(pool.records),
            )

    if not observables.iter_channels():
        return MaterializedTeacherResult(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            status="skipped_no_observables",
            message="No observable channels could be built for this accession.",
            candidate_count=len(pool.records),
        )

    candidate_pool_path.parent.mkdir(parents=True, exist_ok=True)
    observable_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)

    pool.to_jsonl(candidate_pool_path)
    observables.to_npz(observable_bundle_path)

    prior_weights = build_candidate_prior_weights(pool, policy=config.prior_policy)
    breakdown = fit_energy_breakdown_with_adaptive_lambda(
        config=config,
        bundle=bundle,
        observables=observables,
        prior_weights=prior_weights,
    )
    solution = WeightSolution(
        method=config.reweighting_method,
        weights=breakdown.weights,
        energy=breakdown.energy,
        converged=breakdown.converged,
        iterations=breakdown.iterations,
        diagnostics=dict(breakdown.diagnostics),
    )
    solution.to_json(solution_path)
    breakdown.to_json(breakdown_path)
    save_weights_csv(weights_csv_path, observables.candidate_ids(), breakdown.weights)
    summary_path.write_text(
        json.dumps(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "target_bundle_path": str(bundle_path),
                "candidate_count": len(pool.records),
                "candidate_source_counts": candidate_source_counts(pool),
                "prior_policy": config.prior_policy,
                "lambda_reg": float(config.lambda_reg),
                "adaptive_lambda_ess_floor": config.adaptive_lambda_ess_floor,
                "adaptive_lambda_selected": breakdown.diagnostics.get(
                    "adaptive_lambda_selected", float(config.lambda_reg)
                ),
                "chemical_shift_valid_fraction": coverage.get("chemical_shifts", 0.0),
                "j_coupling_valid_fraction": coverage.get("j_couplings", 0.0),
                "noe_valid_fraction": coverage.get("noe_restraints", 0.0),
            },
            indent=2,
            sort_keys=True,
        )
    )

    return MaterializedTeacherResult(
        entity_uid=entity_uid,
        bmrb_id=bmrb_id,
        status="written",
        message="Teacher artifacts materialized.",
        candidate_count=len(pool.records),
        chemical_shift_valid_fraction=coverage.get("chemical_shifts", 0.0),
        j_coupling_valid_fraction=coverage.get("j_couplings", 0.0),
        noe_valid_fraction=coverage.get("noe_restraints", 0.0),
    )


def build_multisource_candidate_pool(
    candidate_roots: dict[str, str],
    bmrb_id: str,
    data_root: Path,
    repo_root: Path,
    chemical_shift_dir_templates: dict[str, str],
    chemical_shift_formats: dict[str, str],
    max_candidates_per_example: int | None = None,
    parse_structures: bool = True,
) -> CandidatePool:
    """Build one combined candidate pool across AF3, CALVADOS2, and BioEmu."""
    records: list[CandidateRecord] = []
    for source, root_value in sorted(candidate_roots.items()):
        structure_dir = resolve_existing_path(
            root_value,
            data_root=data_root,
            repo_root=repo_root,
        )
        if structure_dir is None or not structure_dir.exists():
            continue
        shift_dir = resolve_shift_dir(
            template=chemical_shift_dir_templates.get(source),
            bmrb_id=bmrb_id,
            source=source,
            data_root=data_root,
            repo_root=repo_root,
        )
        if parse_structures:
            source_pool = build_candidate_pool(
                structure_dir=structure_dir,
                source=source,
                chemical_shift_dir=shift_dir,
                chemical_shift_format=chemical_shift_formats.get(source),
            )
        else:
            source_pool = build_lightweight_candidate_pool(
                structure_dir=structure_dir,
                source=source,
                chemical_shift_dir=shift_dir,
                chemical_shift_format=chemical_shift_formats.get(source),
            )
        for record in source_pool.records:
            raw_candidate_id = record.candidate_id
            record.candidate_id = f"{source}:{raw_candidate_id}"
            record.metadata = dict(record.metadata)
            record.metadata["raw_candidate_id"] = raw_candidate_id
            records.append(record)
    if max_candidates_per_example is not None:
        records = _cap_candidate_records_by_example(
            records=records,
            max_candidates=max_candidates_per_example,
        )
    return CandidatePool(records=records)


def build_lightweight_candidate_pool(
    structure_dir: str | Path,
    source: str,
    chemical_shift_dir: str | Path | None = None,
    chemical_shift_format: str | None = None,
) -> CandidatePool:
    """Build a candidate pool without parsing PDB atom records.

    CS-only teacher materialization only needs stable candidate IDs plus
    precomputed shift sidecars. Skipping Biopython structure parsing avoids
    spending minutes per accession on metadata that is not used by CS
    reweighting.
    """

    structure_root = Path(structure_dir)
    prediction_map = (
        build_prediction_map(chemical_shift_dir)
        if chemical_shift_dir is not None
        else {}
    )
    stage_manifest = load_stage_manifest(structure_root)
    records: list[CandidateRecord] = []
    for path in _iter_pdb_paths(structure_root):
        candidate_id = normalize_candidate_key(path)
        relative_path = path.relative_to(structure_root).as_posix()
        metadata = infer_candidate_metadata(path)
        metadata.update(stage_manifest.get(relative_path, {}))
        metadata["structure_parse_skipped"] = True
        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                structure_path=str(path),
                source=source,
                chemical_shift_path=prediction_map.get(candidate_id),
                chemical_shift_format=chemical_shift_format,
                is_protonated=None,
                residue_keys=[],
                metadata=metadata,
            )
        )
    return CandidatePool(records=records)


def _iter_pdb_paths(structure_root: Path) -> list[Path]:
    """Return PDB paths while following staged symlink directories."""

    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(structure_root, followlinks=True):
        for filename in filenames:
            if filename.endswith(".pdb"):
                paths.append(Path(dirpath) / filename)
    return sorted(paths)


def build_candidate_prior_weights(
    pool: CandidatePool,
    *,
    policy: str = "source_balanced",
) -> np.ndarray | None:
    """Return prior weights for one candidate pool.

    ``source_balanced`` gives each source equal total prior mass and then spreads
    that mass uniformly inside the source. This prevents AF3/CALVADOS/BioEmu
    count imbalance from becoming an accidental teacher prior.
    """

    candidate_count = len(pool.records)
    if candidate_count <= 0:
        return None
    normalized_policy = str(policy).strip().lower()
    if normalized_policy in {"", "uniform", "none"}:
        return np.full(candidate_count, 1.0 / candidate_count, dtype=float)
    if normalized_policy != "source_balanced":
        raise ValueError(f"Unsupported teacher prior policy: {policy}")
    by_source: dict[str, list[int]] = {}
    for index, record in enumerate(pool.records):
        by_source.setdefault(str(record.source), []).append(index)
    if not by_source:
        return np.full(candidate_count, 1.0 / candidate_count, dtype=float)
    weights = np.zeros(candidate_count, dtype=float)
    source_mass = 1.0 / float(len(by_source))
    for indices in by_source.values():
        if not indices:
            continue
        weights[indices] = source_mass / float(len(indices))
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(candidate_count, 1.0 / candidate_count, dtype=float)
    return weights / total


def candidate_source_counts(pool: CandidatePool) -> dict[str, int]:
    """Return candidate counts keyed by source."""

    counts: dict[str, int] = {}
    for record in pool.records:
        source = str(record.source)
        counts[source] = counts.get(source, 0) + 1
    return counts


def namespaced_integrated_artifact_path(
    path_value: str | Path,
    *,
    namespace: str | None,
) -> Path:
    """Insert an optional namespace below integrated artifact roots."""

    raw_path = Path(str(path_value))
    if not namespace or raw_path.is_absolute():
        return raw_path
    parts = raw_path.parts
    if len(parts) >= 3 and parts[0] == "integrated" and parts[1] in {
        "pools",
        "observables",
        "teachers",
    }:
        return Path(parts[0]) / parts[1] / str(namespace) / Path(*parts[2:])
    return raw_path


def _cap_candidate_records_by_example(
    records: list[CandidateRecord],
    max_candidates: int,
) -> list[CandidateRecord]:
    """Apply a source-balanced total candidate cap to one accession pool."""

    by_source: dict[str, list[CandidateRecord]] = {}
    for record in records:
        source = str(record.source)
        by_source.setdefault(source, []).append(record)
    for source_records in by_source.values():
        source_records.sort(key=lambda record: record.candidate_id)
    capped: list[CandidateRecord] = []
    source_order = sorted(by_source)
    while len(capped) < max_candidates:
        progressed = False
        for source in source_order:
            if len(capped) >= max_candidates:
                break
            if not by_source[source]:
                continue
            capped.append(by_source[source].pop(0))
            progressed = True
        if not progressed:
            break
    return capped


def build_observable_bundle(
    pool: CandidatePool,
    bundle: NMRTargetBundle,
) -> ObservableBundle:
    """Build one observable bundle for one materialized teacher example."""
    return ObservableBundle(
        chemical_shifts=(
            ChemicalShiftMatrixBuilder().build(pool, bundle)
            if bundle.chemical_shifts
            else None
        ),
        j_couplings=JCouplingHead().build(pool, bundle) if bundle.j_couplings else None,
        noe_restraints=NOEHead().build(pool, bundle) if bundle.noe_restraints else None,
    )


def observable_valid_fraction(observables: ObservableBundle) -> dict[str, float]:
    """Return per-channel matrix validity fractions."""
    fractions: dict[str, float] = {}
    for name, matrix in observables.iter_channels():
        total = float(matrix.mask.size)
        fractions[name] = 0.0 if total <= 0 else float(matrix.mask.sum() / total)
    return fractions


def build_energy_model(
    config: TeacherMaterializationConfig,
    *,
    lambda_reg: float | None = None,
) -> MultiObservablePosteriorEnergy:
    """Build one configured offline teacher energy model."""
    if config.reweighting_method == "maxent":
        reweighter = MaxEntReweighter()
        regularizer = "maxent"
    elif config.reweighting_method == "euclidean":
        reweighter = EuclideanSimplexReweighter()
        regularizer = "euclidean"
    else:
        raise ValueError(f"Unsupported reweighting method: {config.reweighting_method}")
    return MultiObservablePosteriorEnergy(
        reweighter=reweighter,
        beta_cs=config.beta_cs,
        beta_j=config.beta_j,
        beta_noe=config.beta_noe,
        lambda_reg=config.lambda_reg if lambda_reg is None else float(lambda_reg),
        regularizer=regularizer,
    )


def fit_energy_breakdown_with_adaptive_lambda(
    *,
    config: TeacherMaterializationConfig,
    bundle: NMRTargetBundle,
    observables: ObservableBundle,
    prior_weights: np.ndarray | None,
):
    """Fit teacher weights, optionally raising MaxEnt KL pressure to protect ESS."""

    schedule = adaptive_lambda_schedule(config)
    best_breakdown = None
    requested_floor = config.adaptive_lambda_ess_floor
    for step_index, lambda_value in enumerate(schedule, start=1):
        energy_model = build_energy_model(config, lambda_reg=lambda_value)
        breakdown = energy_model.score(bundle, observables, prior_weights=prior_weights)
        diagnostics = dict(breakdown.diagnostics)
        diagnostics.update(
            {
                "adaptive_lambda_enabled": float(len(schedule) > 1),
                "adaptive_lambda_requested_floor": float(requested_floor or 0.0),
                "adaptive_lambda_selected": float(lambda_value),
                "adaptive_lambda_fit_count": float(step_index),
                "adaptive_lambda_floor_met": float(
                    requested_floor is not None and breakdown.ess >= requested_floor
                ),
            }
        )
        breakdown.diagnostics = diagnostics
        best_breakdown = breakdown
        if requested_floor is None or breakdown.ess >= requested_floor:
            break
    if best_breakdown is None:
        raise RuntimeError("Adaptive teacher fitting did not produce a breakdown.")
    return best_breakdown


def adaptive_lambda_schedule(config: TeacherMaterializationConfig) -> list[float]:
    """Return the per-entry lambda search schedule for ESS-floor teacher fitting."""

    base_lambda = max(float(config.lambda_reg), 0.0)
    floor = config.adaptive_lambda_ess_floor
    if floor is None or str(config.reweighting_method).lower() != "maxent":
        return [base_lambda]
    max_steps = max(1, int(config.adaptive_lambda_max_steps))
    growth = max(float(config.adaptive_lambda_growth), 1.0)
    lambda_max = (
        max(base_lambda, float(config.adaptive_lambda_max))
        if config.adaptive_lambda_max is not None
        else base_lambda * (growth ** max(0, max_steps - 1))
    )
    values: list[float] = []
    current = base_lambda
    for _ in range(max_steps):
        values.append(min(current, lambda_max))
        if values[-1] >= lambda_max:
            break
        current = current * growth if current > 0.0 else growth
    return values or [base_lambda]


def resolve_existing_path(
    path_value: str | Path | None,
    data_root: Path,
    repo_root: Path,
) -> Path | None:
    """Resolve one repo-relative or data-relative path robustly."""
    if path_value is None:
        return None
    raw_path = Path(str(path_value))
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        repo_root / raw_path,
        data_root / raw_path,
        repo_root / "data" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if str(raw_path).startswith("data/"):
        return repo_root / raw_path
    return data_root / raw_path


def resolve_shift_dir(
    template: str | None,
    bmrb_id: str,
    source: str,
    data_root: Path,
    repo_root: Path,
) -> Path | None:
    """Resolve one source-specific chemical-shift directory template."""
    if not template:
        return None
    rendered = template.format(bmrb_id=bmrb_id, source=source, native_id=bmrb_id)
    return resolve_existing_path(rendered, data_root=data_root, repo_root=repo_root)
