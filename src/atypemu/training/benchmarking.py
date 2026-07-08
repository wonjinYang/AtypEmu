"""Benchmark rendering and uncertainty visualization for AtypEmu runs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser
from scipy.stats import gaussian_kde, norm

from atypemu.datasets import IntegratedDataRegistry, MetaDataFrameRegistry
from atypemu.training.config import BenchmarkRenderConfig
from atypemu.training.materialize import resolve_existing_path
from atypemu.training.metrics import (
    ATOM_FAMILY_MAP,
    EvaluatedExample,
    collect_chemical_shift_prediction_rows,
)
from atypemu.training.rci import compute_prci_profile_from_posterior
from atypemu.types import CandidatePool, ObservableBundle

CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES = ["HN", "N", "CA", "CB", "C'"]
JOINT_CHEMICAL_SHIFT_ATOM_SETS: dict[str, tuple[str, ...]] = {
    "HN_N": ("HN", "N"),
    "CA_CB": ("CA", "CB"),
    "CA_C": ("CA", "C'"),
    "HN_N_CA": ("HN", "N", "CA"),
    "CA_CB_C": ("CA", "CB", "C'"),
}


def render_benchmark_artifacts(
    data_root: str | Path,
    integrated_root: str | Path,
    output_dir: str | Path,
    epoch_payload: dict[str, Any],
    render_config: BenchmarkRenderConfig | None = None,
) -> dict[str, Any]:
    """Render benchmark arrays, figures, and uncertainty summaries."""
    config = render_config or BenchmarkRenderConfig()
    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    output_dir_path = Path(output_dir)

    reports_root = output_dir_path / "reports"
    metrics_dir = reports_root / "metrics"
    figures_dir = reports_root / "figures"
    structures_dir = reports_root / "structures"
    arrays_dir = reports_root / "arrays"
    for directory in [metrics_dir, figures_dir, structures_dir, arrays_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    evaluated_examples: list[EvaluatedExample] = list(
        epoch_payload.get("evaluated_examples", [])
    )
    if not evaluated_examples:
        empty_report = {
            "status": "no_evaluated_examples",
            "generated_examples": 0,
        }
        _save_json(metrics_dir / "uncertainty_report.json", empty_report)
        _write_dataframe(
            arrays_dir / "chemical_shift_posteriors.parquet",
            pd.DataFrame(),
        )
        _write_dataframe(
            arrays_dir / "chemical_shift_joint_posteriors.parquet",
            pd.DataFrame(),
        )
        _write_dataframe(arrays_dir / "ensemble_states.parquet", pd.DataFrame())
        _write_dataframe(
            arrays_dir / "conformer_state_assignments.parquet",
            pd.DataFrame(),
        )
        return {
            "benchmark_overlay": {
                "status": "no_evaluated_examples",
                "ensemble_fidelity": {},
                "uncertainty_calibration": {},
                "ensemble_states": {},
                "representative_entity_uid": None,
            },
            "uncertainty_report": empty_report,
        }

    registry = IntegratedDataRegistry.from_data_root(data_root_path)
    teacher_examples = registry.load_teacher_examples()
    teacher_by_entity = {
        str(row["entity_uid"]): row
        for row in teacher_examples.to_dict(orient="records")
    }

    projection_frames: list[pd.DataFrame] = []
    posterior_frames: list[pd.DataFrame] = []
    joint_posterior_frames: list[pd.DataFrame] = []
    rci_profile_frames: list[pd.DataFrame] = []
    ensemble_state_frames: list[pd.DataFrame] = []
    state_assignment_frames: list[pd.DataFrame] = []
    landscape_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    representative_bundle: dict[str, Any] | None = None
    rci_source_frame = _load_rci_profile_frame(
        config=config,
        data_root=data_root_path,
        repo_root=data_root_path.parent,
    )

    prediction_rows = collect_chemical_shift_prediction_rows(
        evaluated_examples,
        split=str(epoch_payload.get("split", "val")),
    )
    if prediction_rows:
        prediction_frame = pd.DataFrame(prediction_rows)
        posterior_frames.append(prediction_frame)
        if rci_source_frame is None and config.emit_rci_adapter:
            rci_source_frame = compute_prci_profile_from_posterior(prediction_frame)

    for payload in evaluated_examples:
        row = teacher_by_entity.get(payload.entity_uid)
        if row is None:
            continue
        analysis = _analyze_example(
            payload=payload,
            row=row,
            data_root=data_root_path,
            integrated_root=integrated_root_path,
            config=config,
            rci_source_frame=rci_source_frame,
        )
        if analysis is None:
            continue
        if analysis.get("projection_frame") is not None:
            projection_frames.append(analysis["projection_frame"])
        if analysis.get("rci_profile_frame") is not None:
            rci_profile_frames.append(analysis["rci_profile_frame"])
        if analysis.get("joint_posterior_frame") is not None:
            joint_posterior_frames.append(analysis["joint_posterior_frame"])
        if analysis.get("ensemble_states_frame") is not None:
            ensemble_state_frames.append(analysis["ensemble_states_frame"])
        if analysis.get("state_assignments_frame") is not None:
            state_assignment_frames.append(analysis["state_assignments_frame"])
        if analysis.get("landscape_metrics"):
            landscape_rows.append(analysis["landscape_metrics"])
        if analysis.get("uncertainty_metrics"):
            uncertainty_rows.append(analysis["uncertainty_metrics"])
        if representative_bundle is None and analysis.get("representative_ready"):
            representative_bundle = analysis

    projection_frame = (
        pd.concat(projection_frames, ignore_index=True)
        if projection_frames
        else pd.DataFrame()
    )
    posterior_frame = (
        pd.concat(posterior_frames, ignore_index=True)
        if posterior_frames
        else pd.DataFrame()
    )
    posterior_frame = _calibrate_chemical_shift_posteriors(posterior_frame)
    joint_posterior_frame = (
        pd.concat(joint_posterior_frames, ignore_index=True)
        if joint_posterior_frames
        else pd.DataFrame()
    )
    rci_profile_frame = (
        pd.concat(rci_profile_frames, ignore_index=True)
        if rci_profile_frames
        else pd.DataFrame()
    )
    ensemble_states_frame = (
        pd.concat(ensemble_state_frames, ignore_index=True)
        if ensemble_state_frames
        else pd.DataFrame()
    )
    state_assignments_frame = (
        pd.concat(state_assignment_frames, ignore_index=True)
        if state_assignment_frames
        else pd.DataFrame()
    )
    _write_dataframe(arrays_dir / "projection_coordinates.parquet", projection_frame)
    _write_dataframe(arrays_dir / "chemical_shift_posteriors.parquet", posterior_frame)
    _write_dataframe(
        arrays_dir / "chemical_shift_joint_posteriors.parquet",
        joint_posterior_frame,
    )
    _write_dataframe(
        arrays_dir / "rci_vs_ensemble_profile.parquet",
        rci_profile_frame,
    )
    _write_dataframe(arrays_dir / "ensemble_states.parquet", ensemble_states_frame)
    _write_dataframe(
        arrays_dir / "conformer_state_assignments.parquet",
        state_assignments_frame,
    )

    ensemble_fidelity = _aggregate_metric_rows(
        landscape_rows,
        [
            "landscape_js",
            "free_energy_mae",
            "free_energy_rmse",
            "mass_coverage_95",
            "basin_occupancy_js",
        ],
    )
    uncertainty_summary = _aggregate_metric_rows(
        uncertainty_rows,
        [
            "ensemble_local_confidence_mean",
            "ensemble_local_confidence_vs_variance_corr",
            "low_confidence_residue_fraction",
            "distance_uncertainty_mean",
            "interdomain_uncertainty_score",
        ],
    )
    rci_calibration = _rci_calibration_summary(
        rci_source_frame=rci_source_frame,
        rci_profile_frame=rci_profile_frame,
    )
    chemical_shift_posterior_calibration = (
        _chemical_shift_posterior_calibration_summary(posterior_frame)
    )
    joint_posterior_summary = _joint_posterior_summary(joint_posterior_frame)
    ensemble_state_summary = _ensemble_state_summary(ensemble_states_frame)

    if representative_bundle is not None:
        if config.emit_landscape and representative_bundle.get("landscape_bundle"):
            _render_landscape_figures(
                figures_dir=figures_dir,
                arrays_dir=arrays_dir,
                bundle=representative_bundle["landscape_bundle"],
            )
            _render_landscape_overlay_figures(
                figures_dir=figures_dir,
                bundle=representative_bundle["landscape_bundle"],
            )
        if config.emit_structure_uncertainty and representative_bundle.get(
            "structure_uncertainty_bundle"
        ):
            _render_structure_uncertainty(
                figures_dir=figures_dir,
                structures_dir=structures_dir,
                arrays_dir=arrays_dir,
                bundle=representative_bundle["structure_uncertainty_bundle"],
            )
        if config.emit_nmr_posteriors and not posterior_frame.empty:
            _render_chemical_shift_figures(
                figures_dir=figures_dir,
                posterior_frame=posterior_frame,
                joint_posterior_frame=joint_posterior_frame,
                representative_entity_uid=representative_bundle["entity_uid"],
            )
        if (
            config.emit_rci_adapter
            and representative_bundle.get("rci_profile_frame") is not None
        ):
            _render_rci_adapter_figures(
                figures_dir=figures_dir,
                rci_profile=representative_bundle["rci_profile_frame"],
            )
        if (
            config.emit_ensemble_states
            and representative_bundle.get("ensemble_state_bundle") is not None
        ):
            _render_ensemble_state_gallery(
                figures_dir=figures_dir,
                structures_dir=structures_dir,
                bundle=representative_bundle["ensemble_state_bundle"],
            )

    external_slice_calibration = _build_external_slice_calibration(
        data_root=data_root_path,
        uncertainty_rows=uncertainty_rows,
    )
    uncertainty_report = {
        "generated_examples": int(len(uncertainty_rows)),
        "representative_entity_uid": (
            None
            if representative_bundle is None
            else representative_bundle["entity_uid"]
        ),
        "summary": uncertainty_summary,
        "rci_calibration": rci_calibration,
        "chemical_shift_posterior_calibration": chemical_shift_posterior_calibration,
        "chemical_shift_joint_posterior": joint_posterior_summary,
        "ensemble_states": ensemble_state_summary,
        "external_slice_calibration": external_slice_calibration,
    }
    _save_json(metrics_dir / "uncertainty_report.json", uncertainty_report)

    return {
        "benchmark_overlay": {
            "status": "ok",
            "ensemble_fidelity": ensemble_fidelity,
            "uncertainty_calibration": uncertainty_summary,
            "rci_calibration": rci_calibration,
            "chemical_shift_posterior_calibration": (
                chemical_shift_posterior_calibration
            ),
            "chemical_shift_joint_posterior": joint_posterior_summary,
            "ensemble_states": ensemble_state_summary,
            "representative_entity_uid": (
                None
                if representative_bundle is None
                else representative_bundle["entity_uid"]
            ),
        },
        "uncertainty_report": uncertainty_report,
    }


def _analyze_example(
    payload: EvaluatedExample,
    row: dict[str, Any],
    data_root: Path,
    integrated_root: Path,
    config: BenchmarkRenderConfig,
    rci_source_frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Analyze one evaluated example for rendering and uncertainty metrics."""
    _ = integrated_root
    pool_path = resolve_existing_path(
        row.get("candidate_pool_path"),
        data_root=data_root,
        repo_root=data_root.parent,
    )
    if pool_path is None or not pool_path.exists():
        return None
    if payload.predicted_weights is None or payload.teacher_weights is None:
        return None

    pool = CandidatePool.from_jsonl(pool_path)
    if len(pool.records) != len(payload.predicted_weights):
        return None
    observable_bundle = _load_observable_bundle(
        row=row,
        data_root=data_root,
        repo_root=data_root.parent,
    )

    coordinate_bundle = _load_candidate_coordinates(
        pool=pool,
        max_pair_feature_residues=config.max_pair_feature_residues,
        max_pair_distance_features=config.max_pair_distance_features,
    )
    if coordinate_bundle is None:
        return None

    candidate_evidence = _candidate_nmr_evidence_frame(
        payload=payload,
        observables=observable_bundle,
        candidate_ids=pool.candidate_ids,
    )
    joint_posterior_frame = _joint_chemical_shift_posterior_frame(
        payload=payload,
        observables=observable_bundle,
        candidate_ids=pool.candidate_ids,
        weights=np.asarray(payload.predicted_weights, dtype=np.float64),
    )
    rci_for_entity = _rci_profile_for_entity(
        rci_source_frame=rci_source_frame,
        entity_uid=payload.entity_uid,
    )
    projection_bundle = _compute_landscape_bundle(
        entity_uid=payload.entity_uid,
        candidate_ids=pool.candidate_ids,
        feature_matrix=coordinate_bundle["feature_matrix"],
        teacher_weights=np.asarray(payload.teacher_weights, dtype=np.float64),
        predicted_weights=np.asarray(payload.predicted_weights, dtype=np.float64),
        config=config,
        candidate_evidence=candidate_evidence,
    )
    uncertainty_bundle = _compute_structure_uncertainty_bundle(
        entity_uid=payload.entity_uid,
        coordinate_bundle=coordinate_bundle,
        predicted_weights=np.asarray(payload.predicted_weights, dtype=np.float64),
    )
    rci_profile = _rci_vs_ensemble_profile(
        rci_profile=rci_for_entity,
        uncertainty_bundle=uncertainty_bundle,
    )
    rci_mismatch = _candidate_rci_mismatch(
        rci_profile=rci_profile,
        uncertainty_bundle=uncertainty_bundle,
    )
    if rci_mismatch is not None:
        projection_bundle["projection_frame"]["rci_mismatch"] = rci_mismatch
    ensemble_state_bundle = _compute_ensemble_state_bundle(
        entity_uid=payload.entity_uid,
        coordinate_bundle=coordinate_bundle,
        projection_bundle=projection_bundle,
        uncertainty_bundle=uncertainty_bundle,
        teacher_weights=np.asarray(payload.teacher_weights, dtype=np.float64),
        predicted_weights=np.asarray(payload.predicted_weights, dtype=np.float64),
        config=config,
    )
    if ensemble_state_bundle is not None:
        assignments = ensemble_state_bundle["state_assignments_frame"][
            ["entity_uid", "candidate_id", "state_id", "is_state_representative"]
        ]
        projection_bundle["projection_frame"] = projection_bundle[
            "projection_frame"
        ].merge(
            assignments,
            on=["entity_uid", "candidate_id"],
            how="left",
        )
    return {
        "entity_uid": payload.entity_uid,
        "projection_frame": projection_bundle["projection_frame"],
        "joint_posterior_frame": joint_posterior_frame,
        "rci_profile_frame": rci_profile,
        "ensemble_states_frame": (
            None
            if ensemble_state_bundle is None
            else ensemble_state_bundle["ensemble_states_frame"]
        ),
        "state_assignments_frame": (
            None
            if ensemble_state_bundle is None
            else ensemble_state_bundle["state_assignments_frame"]
        ),
        "landscape_metrics": projection_bundle["metrics"],
        "uncertainty_metrics": uncertainty_bundle["metrics"],
        "representative_ready": True,
        "landscape_bundle": projection_bundle,
        "structure_uncertainty_bundle": uncertainty_bundle,
        "ensemble_state_bundle": ensemble_state_bundle,
    }


def _load_candidate_coordinates(
    pool: CandidatePool,
    max_pair_feature_residues: int,
    max_pair_distance_features: int,
) -> dict[str, Any] | None:
    """Load shared C-alpha coordinates and descriptor features for one pool."""
    parser = PDBParser(QUIET=True)
    coord_maps: list[dict[str, np.ndarray]] = []
    ordered_keys_by_record: list[list[str]] = []
    valid_records = []

    for record in pool.records:
        structure_path = Path(record.structure_path)
        if not structure_path.exists():
            return None
        structure = parser.get_structure(structure_path.stem, str(structure_path))
        coord_map: dict[str, np.ndarray] = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" not in residue:
                        continue
                    residue_key = (
                        f"{chain.id}:{residue.id[1]}:{residue.resname.strip().upper()}"
                    )
                    coord_map[residue_key] = residue["CA"].coord.astype(np.float64)
            break
        ordered_keys = [key for key in record.residue_keys if key in coord_map]
        if not ordered_keys:
            return None
        coord_maps.append(coord_map)
        ordered_keys_by_record.append(ordered_keys)
        valid_records.append(record)

    common_keys = list(ordered_keys_by_record[0])
    for ordered_keys in ordered_keys_by_record[1:]:
        ordered_set = set(ordered_keys)
        common_keys = [key for key in common_keys if key in ordered_set]
    if len(common_keys) < 2:
        return None

    coord_stack = np.stack(
        [
            np.stack([coord_map[key] for key in common_keys], axis=0)
            for coord_map in coord_maps
        ],
        axis=0,
    )
    selected_indices = _subsample_indices(
        total=len(common_keys),
        max_count=max_pair_feature_residues,
    )
    selected_coords = coord_stack[:, selected_indices, :]
    feature_matrix = _compute_feature_matrix(
        coords=coord_stack,
        selected_coords=selected_coords,
        max_pair_distance_features=max_pair_distance_features,
    )
    return {
        "records": valid_records,
        "common_keys": common_keys,
        "coords": coord_stack,
        "feature_matrix": feature_matrix,
    }


def _load_observable_bundle(
    row: dict[str, Any],
    data_root: Path,
    repo_root: Path,
) -> ObservableBundle | None:
    """Load one materialized observable bundle when available."""

    observables_path = resolve_existing_path(
        row.get("observable_bundle_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    if observables_path is None or not observables_path.exists():
        return None
    try:
        return ObservableBundle.from_npz(observables_path)
    except Exception:
        return None


def _candidate_nmr_evidence_frame(
    payload: EvaluatedExample,
    observables: ObservableBundle | None,
    candidate_ids: list[str],
) -> pd.DataFrame:
    """Return candidate-level chemical-shift evidence overlay metrics."""

    channel = payload.channels.get("chemical_shifts")
    if observables is None or observables.chemical_shifts is None or channel is None:
        return pd.DataFrame()

    matrix = observables.chemical_shifts
    target_lookup = {
        target_id: (float(target), float(sigma))
        for target_id, target, sigma in zip(
            channel.target_ids,
            channel.targets,
            channel.sigmas,
            strict=True,
        )
    }
    row_indices: list[int] = []
    targets: list[float] = []
    sigmas: list[float] = []
    for row_index, target_id in enumerate(matrix.target_ids):
        if target_id not in target_lookup:
            continue
        target_value, sigma = target_lookup[target_id]
        row_indices.append(row_index)
        targets.append(target_value)
        sigmas.append(max(sigma, 1e-6))
    if not row_indices:
        return pd.DataFrame()

    values = matrix.values[np.asarray(row_indices, dtype=int), :]
    masks = matrix.mask[np.asarray(row_indices, dtype=int), :].astype(bool)
    target_array = np.asarray(targets, dtype=np.float64)
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    matrix_candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(matrix.candidate_ids)
    }

    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        column_index = matrix_candidate_index.get(candidate_id)
        if column_index is None:
            rows.append(
                {
                    "entity_uid": payload.entity_uid,
                    "candidate_id": candidate_id,
                    "cs_mae_ppm": math.nan,
                    "cs_rmse_z": math.nan,
                }
            )
            continue
        valid = masks[:, column_index]
        if not np.any(valid):
            rows.append(
                {
                    "entity_uid": payload.entity_uid,
                    "candidate_id": candidate_id,
                    "cs_mae_ppm": math.nan,
                    "cs_rmse_z": math.nan,
                }
            )
            continue
        residuals = values[valid, column_index] - target_array[valid]
        z_errors = residuals / sigma_array[valid]
        rows.append(
            {
                "entity_uid": payload.entity_uid,
                "candidate_id": candidate_id,
                "cs_mae_ppm": float(np.mean(np.abs(residuals))),
                "cs_rmse_z": float(np.sqrt(np.mean(np.square(z_errors)))),
            }
        )
    return pd.DataFrame(rows)


def _joint_chemical_shift_posterior_frame(
    payload: EvaluatedExample,
    observables: ObservableBundle | None,
    candidate_ids: list[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Return residue-level multi-atom chemical-shift posterior rows."""

    channel = payload.channels.get("chemical_shifts")
    if observables is None or observables.chemical_shifts is None or channel is None:
        return pd.DataFrame()
    matrix = observables.chemical_shifts
    if len(candidate_ids) != len(weights):
        return pd.DataFrame()

    target_lookup = {
        target_id: (float(target), float(sigma))
        for target_id, target, sigma in zip(
            channel.target_ids,
            channel.targets,
            channel.sigmas,
            strict=True,
        )
    }
    candidate_weight_lookup = {
        candidate_id: float(weight)
        for candidate_id, weight in zip(candidate_ids, weights, strict=True)
    }
    matrix_weights = np.asarray(
        [
            candidate_weight_lookup.get(candidate_id, math.nan)
            for candidate_id in matrix.candidate_ids
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix_weights).any():
        return pd.DataFrame()

    residue_targets: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row_index, target_id in enumerate(matrix.target_ids):
        if target_id not in target_lookup:
            continue
        metadata = _chemical_shift_target_metadata(target_id)
        if metadata is None:
            continue
        chain_id, residue_index, residue_name, atom_family = metadata
        target_value, target_sigma = target_lookup[target_id]
        residue_key = (chain_id, residue_index, residue_name)
        residue_targets.setdefault(residue_key, {})[atom_family] = {
            "row_index": row_index,
            "target_id": target_id,
            "target_value": target_value,
            "target_sigma": target_sigma,
        }

    rows: list[dict[str, Any]] = []
    for (chain_id, residue_index, residue_name), family_map in sorted(
        residue_targets.items()
    ):
        for atom_set, atom_families in JOINT_CHEMICAL_SHIFT_ATOM_SETS.items():
            if not set(atom_families).issubset(family_map):
                continue
            row = _joint_posterior_row(
                payload=payload,
                matrix=matrix,
                matrix_weights=matrix_weights,
                chain_id=chain_id,
                residue_index=residue_index,
                residue_name=residue_name,
                atom_set=atom_set,
                atom_families=atom_families,
                family_map=family_map,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _joint_posterior_row(
    payload: EvaluatedExample,
    matrix: Any,
    matrix_weights: np.ndarray,
    chain_id: str,
    residue_index: int,
    residue_name: str,
    atom_set: str,
    atom_families: tuple[str, ...],
    family_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Compute one Gaussian-approximated joint posterior row."""

    row_indices = [int(family_map[family]["row_index"]) for family in atom_families]
    target_ids = [str(family_map[family]["target_id"]) for family in atom_families]
    target_vector = np.asarray(
        [float(family_map[family]["target_value"]) for family in atom_families],
        dtype=np.float64,
    )
    target_sigma_vector = np.asarray(
        [
            max(float(family_map[family]["target_sigma"]), 1e-6)
            for family in atom_families
        ],
        dtype=np.float64,
    )
    masks = matrix.mask[np.asarray(row_indices, dtype=int), :].astype(bool)
    valid = np.all(masks, axis=0) & np.isfinite(matrix_weights) & (matrix_weights > 0.0)
    if int(np.sum(valid)) < 2:
        return None

    sample_values = matrix.values[np.asarray(row_indices, dtype=int), :][:, valid].T
    sample_weights = matrix_weights[valid].astype(np.float64, copy=False)
    sample_weights = sample_weights / np.clip(sample_weights.sum(), 1e-12, None)
    mean_vector = np.average(sample_values, axis=0, weights=sample_weights)
    centered = sample_values - mean_vector[None, :]
    covariance = (centered * sample_weights[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T)
    atom_count = len(atom_families)
    jitter = max(1e-6, 1e-6 * float(np.trace(covariance)) / max(atom_count, 1))
    covariance_jittered = covariance + np.eye(atom_count, dtype=np.float64) * jitter
    inv_cov = np.linalg.pinv(covariance_jittered)
    residual = mean_vector - target_vector
    mahalanobis_squared = float(residual.T @ inv_cov @ residual)
    sign, logdet = np.linalg.slogdet(covariance_jittered)
    if sign <= 0 or not math.isfinite(float(logdet)):
        eigenvalues = np.linalg.eigvalsh(covariance_jittered)
        logdet = float(np.sum(np.log(np.clip(eigenvalues, 1e-12, None))))
    joint_nll = 0.5 * (
        atom_count * math.log(2.0 * math.pi) + float(logdet) + mahalanobis_squared
    )
    quantiles = _joint_marginal_quantiles(
        sample_values=sample_values,
        sample_weights=sample_weights,
    )
    return {
        "entity_uid": payload.entity_uid,
        "bmrb_id": payload.bmrb_id,
        "chain_id": chain_id,
        "residue_index": int(residue_index),
        "residue_name": residue_name,
        "atom_set": atom_set,
        "atom_families_json": _json_dumps(list(atom_families)),
        "target_ids_json": _json_dumps(target_ids),
        "posterior_mean_vector_json": _json_dumps(mean_vector.tolist()),
        "target_vector_json": _json_dumps(target_vector.tolist()),
        "target_sigma_vector_json": _json_dumps(target_sigma_vector.tolist()),
        "marginal_std_vector_json": _json_dumps(
            np.sqrt(np.clip(np.diag(covariance), 0.0, None)).tolist()
        ),
        "covariance_json": _json_dumps(covariance_jittered.tolist()),
        "candidate_count": int(np.sum(valid)),
        "atom_count": int(atom_count),
        "mahalanobis_distance": float(math.sqrt(max(mahalanobis_squared, 0.0))),
        "joint_nll": float(joint_nll),
        "joint_l2_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "rectangular_coverage_50": _rectangular_coverage(
            target_vector,
            quantiles["q25"],
            quantiles["q75"],
        ),
        "rectangular_coverage_80": _rectangular_coverage(
            target_vector,
            quantiles["q10"],
            quantiles["q90"],
        ),
        "rectangular_coverage_95": _rectangular_coverage(
            target_vector,
            quantiles["q05"],
            quantiles["q95"],
        ),
    }


def _joint_marginal_quantiles(
    sample_values: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return marginal quantiles for each coordinate of a joint posterior."""

    quantiles: dict[str, list[float]] = {
        "q05": [],
        "q10": [],
        "q25": [],
        "q75": [],
        "q90": [],
        "q95": [],
    }
    for column_index in range(sample_values.shape[1]):
        values = sample_values[:, column_index]
        for key, quantile in [
            ("q05", 0.05),
            ("q10", 0.10),
            ("q25", 0.25),
            ("q75", 0.75),
            ("q90", 0.90),
            ("q95", 0.95),
        ]:
            quantiles[key].append(
                _weighted_quantile_1d(values, sample_weights, quantile)
            )
    return {
        key: np.asarray(values, dtype=np.float64) for key, values in quantiles.items()
    }


def _chemical_shift_target_metadata(
    target_id: str,
) -> tuple[str, int, str, str] | None:
    """Parse chain, residue, residue name, and atom family from a CS target id."""

    if not str(target_id).startswith("cs:"):
        return None
    parts = str(target_id).split(":")
    if len(parts) < 5:
        return None
    atom_family = ATOM_FAMILY_MAP.get(parts[-1])
    if atom_family is None:
        return None
    try:
        residue_index = int(parts[2])
    except ValueError:
        return None
    return parts[1], residue_index, parts[3], atom_family


def _rectangular_coverage(
    target_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    """Return whether all target coordinates fall inside marginal intervals."""

    return bool(np.all((target_vector >= lower) & (target_vector <= upper)))


def _compute_feature_matrix(
    coords: np.ndarray,
    selected_coords: np.ndarray,
    max_pair_distance_features: int,
) -> np.ndarray:
    """Compute structure descriptors for weighted PCA and medoid selection."""
    centers = coords.mean(axis=1, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(np.square(coords - centers), axis=2), axis=1))
    end_to_end = np.linalg.norm(coords[:, 0, :] - coords[:, -1, :], axis=1)

    residue_count = selected_coords.shape[1]
    pair_indices = np.triu_indices(residue_count, k=1)
    pair_distances = np.linalg.norm(
        selected_coords[:, pair_indices[0], :] - selected_coords[:, pair_indices[1], :],
        axis=2,
    )
    if pair_distances.shape[1] > max_pair_distance_features:
        variances = np.var(pair_distances, axis=0)
        selected = np.argsort(variances)[-max_pair_distance_features:]
        pair_distances = pair_distances[:, selected]

    return np.concatenate(
        [
            rg[:, None],
            end_to_end[:, None],
            pair_distances,
        ],
        axis=1,
    ).astype(np.float64)


def _compute_landscape_bundle(
    entity_uid: str,
    candidate_ids: list[str],
    feature_matrix: np.ndarray,
    teacher_weights: np.ndarray,
    predicted_weights: np.ndarray,
    config: BenchmarkRenderConfig,
    candidate_evidence: pd.DataFrame,
) -> dict[str, Any]:
    """Compute shared projection coordinates and free-energy surfaces."""
    combined_weights = 0.5 * (teacher_weights + predicted_weights)
    combined_weights = combined_weights / np.clip(combined_weights.sum(), 1e-12, None)
    centered = feature_matrix - np.average(
        feature_matrix, axis=0, weights=combined_weights
    )
    weighted = centered * np.sqrt(combined_weights[:, None])
    _, _, vh = np.linalg.svd(weighted, full_matrices=False)
    components = vh[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    coordinates = centered @ components[:, :2]

    grid_x, grid_y, teacher_density = _density_grid(
        coordinates=coordinates,
        weights=teacher_weights,
        grid_size=config.grid_size,
        method=config.landscape_density_method,
    )
    _, _, predicted_density = _density_grid(
        coordinates=coordinates,
        weights=predicted_weights,
        grid_size=config.grid_size,
        method=config.landscape_density_method,
    )
    teacher_free_energy = _free_energy_from_density(teacher_density)
    predicted_free_energy = _free_energy_from_density(predicted_density)
    delta_free_energy = predicted_free_energy - teacher_free_energy
    finite_mask = np.isfinite(teacher_free_energy) & np.isfinite(predicted_free_energy)

    projection_frame = pd.DataFrame(
        {
            "entity_uid": entity_uid,
            "candidate_id": list(candidate_ids),
            "projection_x": coordinates[:, 0],
            "projection_y": coordinates[:, 1],
            "teacher_weight": teacher_weights,
            "predicted_weight": predicted_weights,
        }
    )
    projection_frame["teacher_score"] = projection_frame["teacher_weight"]
    teacher_energy = -np.log(np.clip(teacher_weights, 1e-12, None))
    projection_frame["teacher_energy"] = teacher_energy - float(
        np.nanmin(teacher_energy)
    )
    if not candidate_evidence.empty:
        projection_frame = projection_frame.merge(
            candidate_evidence,
            on=["entity_uid", "candidate_id"],
            how="left",
        )
    else:
        projection_frame["cs_mae_ppm"] = math.nan
        projection_frame["cs_rmse_z"] = math.nan
    return {
        "projection_frame": projection_frame,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "teacher_density": teacher_density,
        "predicted_density": predicted_density,
        "teacher_free_energy": teacher_free_energy,
        "predicted_free_energy": predicted_free_energy,
        "delta_free_energy": delta_free_energy,
        "metrics": {
            "entity_uid": entity_uid,
            "landscape_js": _discrete_js(teacher_density, predicted_density),
            "free_energy_mae": (
                float(np.mean(np.abs(delta_free_energy[finite_mask])))
                if np.any(finite_mask)
                else math.nan
            ),
            "free_energy_rmse": (
                float(np.sqrt(np.mean(np.square(delta_free_energy[finite_mask]))))
                if np.any(finite_mask)
                else math.nan
            ),
            "mass_coverage_95": _mass_coverage_95(
                teacher_density=teacher_density,
                predicted_density=predicted_density,
            ),
            "basin_occupancy_js": math.nan,
        },
    }


def _compute_structure_uncertainty_bundle(
    entity_uid: str,
    coordinate_bundle: dict[str, Any],
    predicted_weights: np.ndarray,
) -> dict[str, Any]:
    """Compute representative-structure uncertainty summaries."""
    weights = predicted_weights / np.clip(predicted_weights.sum(), 1e-12, None)
    feature_matrix = coordinate_bundle["feature_matrix"]
    distances = np.linalg.norm(
        feature_matrix[:, None, :] - feature_matrix[None, :, :],
        axis=2,
    )
    medoid_index = int(np.argmin(distances @ weights))

    reference_coords = coordinate_bundle["coords"][medoid_index]
    aligned_coords = np.stack(
        [
            _align_coords_to_reference(coords, reference_coords)
            for coords in coordinate_bundle["coords"]
        ],
        axis=0,
    )
    residue_count = reference_coords.shape[0]
    pairwise_distances = np.linalg.norm(
        coordinate_bundle["coords"][:, :, None, :]
        - coordinate_bundle["coords"][:, None, :, :],
        axis=3,
    )
    distance_mean = np.tensordot(weights, pairwise_distances, axes=(0, 0))
    distance_second_moment = np.tensordot(
        weights,
        np.square(pairwise_distances),
        axes=(0, 0),
    )
    distance_std = np.sqrt(
        np.clip(distance_second_moment - np.square(distance_mean), 0.0, None)
    )

    local_confidence = _local_confidence_profile(
        aligned_coords=aligned_coords,
        reference_coords=reference_coords,
        weights=weights,
    )
    mean_coords = np.tensordot(weights, aligned_coords, axes=(0, 0))
    residue_variance = np.sqrt(
        np.tensordot(
            weights,
            np.sum(np.square(aligned_coords - mean_coords[None, :, :]), axis=2),
            axes=(0, 0),
        )
    )
    confidence_corr = _safe_corr(local_confidence, residue_variance)
    upper_triangle = distance_std[np.triu_indices(residue_count, k=1)]
    uncertainty_profile = pd.DataFrame(
        {
            "entity_uid": entity_uid,
            "residue_index": np.arange(1, residue_count + 1),
            "residue_id": coordinate_bundle["common_keys"],
            "ensemble_local_confidence": local_confidence,
            "residue_variance": residue_variance,
        }
    )
    return {
        "medoid_index": medoid_index,
        "record": coordinate_bundle["records"][medoid_index],
        "common_keys": coordinate_bundle["common_keys"],
        "reference_coords": reference_coords,
        "aligned_coords": aligned_coords,
        "local_confidence": local_confidence,
        "residue_variance": residue_variance,
        "distance_uncertainty_matrix": distance_std,
        "uncertainty_profile": uncertainty_profile,
        "metrics": {
            "entity_uid": entity_uid,
            "ensemble_local_confidence_mean": float(np.mean(local_confidence)),
            "ensemble_local_confidence_vs_variance_corr": confidence_corr,
            "low_confidence_residue_fraction": float(np.mean(local_confidence < 70.0)),
            "distance_uncertainty_mean": (
                float(np.mean(upper_triangle)) if upper_triangle.size else math.nan
            ),
            "interdomain_uncertainty_score": (
                float(np.quantile(upper_triangle, 0.90))
                if upper_triangle.size
                else math.nan
            ),
        },
    }


def _compute_ensemble_state_bundle(
    entity_uid: str,
    coordinate_bundle: dict[str, Any],
    projection_bundle: dict[str, Any],
    uncertainty_bundle: dict[str, Any],
    teacher_weights: np.ndarray,
    predicted_weights: np.ndarray,
    config: BenchmarkRenderConfig,
) -> dict[str, Any] | None:
    """Cluster one ensemble into landscape states and state medoids."""

    projection_frame = projection_bundle.get("projection_frame")
    if not isinstance(projection_frame, pd.DataFrame) or projection_frame.empty:
        return None
    required = {"candidate_id", "projection_x", "projection_y"}
    if not required.issubset(projection_frame.columns):
        return None

    frame = projection_frame.copy().reset_index(drop=True)
    coordinates = frame[["projection_x", "projection_y"]].to_numpy(dtype=np.float64)
    finite = np.all(np.isfinite(coordinates), axis=1)
    if not bool(np.all(finite)):
        frame = frame.loc[finite].reset_index(drop=True)
        coordinates = coordinates[finite]
    if frame.empty:
        return None

    candidate_count = len(frame)
    weights = _normalize_weight_vector(predicted_weights[:candidate_count])
    teacher = _normalize_weight_vector(teacher_weights[:candidate_count])
    max_states = max(1, min(int(config.max_ensemble_states), candidate_count))
    seed_indices = _select_landscape_state_seeds(
        coordinates=coordinates,
        weights=weights,
        max_states=max_states,
    )
    if not seed_indices:
        return None
    labels = _assign_to_seeds(coordinates=coordinates, seed_indices=seed_indices)
    labels, seed_indices = _drop_small_state_seeds(
        coordinates=coordinates,
        weights=weights,
        seed_indices=seed_indices,
        labels=labels,
        min_state_mass=max(float(config.min_ensemble_state_mass), 0.0),
    )

    feature_matrix = coordinate_bundle["feature_matrix"][:candidate_count]
    coords = coordinate_bundle["coords"][:candidate_count]
    state_rows: list[dict[str, Any]] = []
    assignment = frame.copy()
    assignment["predicted_weight"] = weights
    assignment["teacher_weight"] = teacher
    assignment["state_id"] = ""
    assignment["state_rank"] = -1
    assignment["is_state_representative"] = False
    state_details: list[dict[str, Any]] = []

    raw_states = []
    for raw_label in sorted(set(labels.tolist())):
        mask = labels == raw_label
        if not bool(np.any(mask)):
            continue
        state_mass = float(np.sum(weights[mask]))
        raw_states.append((raw_label, state_mass))
    raw_states.sort(key=lambda item: item[1], reverse=True)
    max_mass = max([state_mass for _, state_mass in raw_states] or [1e-12])

    for state_rank, (raw_label, state_mass) in enumerate(raw_states, start=1):
        mask = labels == raw_label
        state_id = f"state_{state_rank:02d}"
        member_indices = np.flatnonzero(mask)
        member_weights = _normalize_weight_vector(weights[member_indices])
        representative_index = _weighted_medoid_index(
            feature_matrix=feature_matrix,
            member_indices=member_indices,
            weights=member_weights,
        )
        state_local = _state_local_uncertainty(
            coords=coords,
            member_indices=member_indices,
            representative_index=representative_index,
            weights=member_weights,
        )
        centroid = np.average(coordinates[mask], axis=0, weights=weights[mask])
        assignment.loc[mask, "state_id"] = state_id
        assignment.loc[mask, "state_rank"] = state_rank
        assignment.loc[
            assignment.index == representative_index,
            "is_state_representative",
        ] = True
        state_rows.append(
            {
                "entity_uid": entity_uid,
                "state_id": state_id,
                "state_rank": int(state_rank),
                "representative_candidate_id": str(
                    frame.loc[representative_index, "candidate_id"]
                ),
                "representative_index": int(representative_index),
                "state_mass": state_mass,
                "teacher_mass": float(np.sum(teacher[mask])),
                "relative_free_energy_kbt": float(
                    -math.log(max(state_mass, 1e-12) / max(max_mass, 1e-12))
                ),
                "candidate_count": int(np.sum(mask)),
                "projection_x_centroid": float(centroid[0]),
                "projection_y_centroid": float(centroid[1]),
                "state_local_confidence_mean": float(
                    np.mean(state_local["local_confidence"])
                ),
                "state_rmsf_mean": float(np.mean(state_local["residue_variance"])),
                "cs_mae_ppm_weighted_mean": _weighted_column_mean(
                    frame.loc[mask],
                    "cs_mae_ppm",
                    weights[mask],
                ),
                "cs_rmse_z_weighted_mean": _weighted_column_mean(
                    frame.loc[mask],
                    "cs_rmse_z",
                    weights[mask],
                ),
                "rci_mismatch_weighted_mean": _weighted_column_mean(
                    frame.loc[mask],
                    "rci_mismatch",
                    weights[mask],
                ),
            }
        )
        state_details.append(
            {
                "state_id": state_id,
                "state_rank": int(state_rank),
                "representative_index": int(representative_index),
                "record": coordinate_bundle["records"][representative_index],
                "local_confidence": state_local["local_confidence"],
                "residue_variance": state_local["residue_variance"],
            }
        )

    assignment = assignment.sort_values(
        ["state_rank", "predicted_weight"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    state_frame = pd.DataFrame(state_rows)
    if not state_frame.empty:
        state_frame["state_entropy"] = _state_entropy(state_frame["state_mass"])

    return {
        "entity_uid": entity_uid,
        "ensemble_states_frame": state_frame,
        "state_assignments_frame": assignment,
        "state_details": state_details,
        "common_keys": uncertainty_bundle["common_keys"],
    }


def _select_landscape_state_seeds(
    coordinates: np.ndarray,
    weights: np.ndarray,
    max_states: int,
) -> list[int]:
    """Select deterministic weighted farthest-point seeds in 2D landscape space."""

    if coordinates.size == 0 or max_states <= 0:
        return []
    seeds = [int(np.argmax(weights))]
    while len(seeds) < max_states:
        distances = np.min(
            np.linalg.norm(
                coordinates[:, None, :] - coordinates[seeds][None, :, :], axis=2
            ),
            axis=1,
        )
        scores = weights * distances
        scores[seeds] = -math.inf
        next_index = int(np.argmax(scores))
        if not math.isfinite(float(scores[next_index])) or scores[next_index] <= 0.0:
            break
        seeds.append(next_index)
    return seeds


def _assign_to_seeds(coordinates: np.ndarray, seed_indices: list[int]) -> np.ndarray:
    """Assign each candidate to the nearest state seed."""

    seed_coords = coordinates[np.asarray(seed_indices, dtype=int)]
    distances = np.linalg.norm(
        coordinates[:, None, :] - seed_coords[None, :, :], axis=2
    )
    return np.argmin(distances, axis=1).astype(int)


def _drop_small_state_seeds(
    coordinates: np.ndarray,
    weights: np.ndarray,
    seed_indices: list[int],
    labels: np.ndarray,
    min_state_mass: float,
) -> tuple[np.ndarray, list[int]]:
    """Drop tiny landscape states and reassign them to retained seeds."""

    if len(seed_indices) <= 1 or min_state_mass <= 0.0:
        return labels, seed_indices
    masses = {
        label: float(np.sum(weights[labels == label]))
        for label in sorted(set(labels.tolist()))
    }
    retained_labels = [
        label for label, mass in masses.items() if mass >= min_state_mass
    ]
    if not retained_labels:
        retained_labels = [max(masses, key=masses.get)]
    retained = [seed_indices[label] for label in retained_labels]
    new_labels = _assign_to_seeds(coordinates=coordinates, seed_indices=retained)
    return new_labels, retained


def _weighted_medoid_index(
    feature_matrix: np.ndarray,
    member_indices: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Return the global candidate index closest to one weighted state."""

    if len(member_indices) == 1:
        return int(member_indices[0])
    state_features = feature_matrix[member_indices]
    distances = np.linalg.norm(
        state_features[:, None, :] - state_features[None, :, :],
        axis=2,
    )
    local_index = int(np.argmin(distances @ weights))
    return int(member_indices[local_index])


def _state_local_uncertainty(
    coords: np.ndarray,
    member_indices: np.ndarray,
    representative_index: int,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute within-state local confidence and RMSF-like residue variance."""

    reference = coords[representative_index]
    aligned = np.stack(
        [
            _align_coords_to_reference(coords[index], reference)
            for index in member_indices
        ],
        axis=0,
    )
    local_confidence = _local_confidence_profile(
        aligned_coords=aligned,
        reference_coords=reference,
        weights=weights,
    )
    mean_coords = np.tensordot(weights, aligned, axes=(0, 0))
    residue_variance = np.sqrt(
        np.tensordot(
            weights,
            np.sum(np.square(aligned - mean_coords[None, :, :]), axis=2),
            axes=(0, 0),
        )
    )
    return {
        "local_confidence": local_confidence,
        "residue_variance": residue_variance,
    }


def _normalize_weight_vector(values: np.ndarray) -> np.ndarray:
    """Return a finite probability vector."""

    weights = np.asarray(values, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights, dtype=np.float64) / max(len(weights), 1)
    return weights / total


def _weighted_column_mean(
    frame: pd.DataFrame,
    column: str,
    weights: np.ndarray,
) -> float | None:
    """Return a weighted mean for one optional state evidence column."""

    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not bool(np.any(valid)):
        return None
    valid_weights = weights[valid] / np.clip(float(np.sum(weights[valid])), 1e-12, None)
    return float(np.sum(values[valid] * valid_weights))


def _state_entropy(state_mass: pd.Series) -> float:
    """Return entropy of one entity's state-mass distribution."""

    mass = pd.to_numeric(state_mass, errors="coerce").dropna().to_numpy(dtype=float)
    mass = mass[mass > 0.0]
    if mass.size == 0:
        return math.nan
    mass = mass / np.sum(mass)
    return float(-np.sum(mass * np.log(np.clip(mass, 1e-12, None))))


def _load_rci_profile_frame(
    config: BenchmarkRenderConfig,
    data_root: Path,
    repo_root: Path,
) -> pd.DataFrame | None:
    """Load an optional external RCI adapter profile."""

    if not config.emit_rci_adapter or not config.rci_profile_path:
        return None
    path = _resolve_adapter_path(
        config.rci_profile_path,
        data_root=data_root,
        repo_root=repo_root,
    )
    if path is None or not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            separator = "\t" if path.suffix in {".tsv", ".tab"} else ","
            frame = pd.read_csv(path, sep=separator)
    except Exception:
        return None

    required = {"entity_uid", "residue_index", "rci_flexibility"}
    if not required.issubset(frame.columns):
        return None
    normalized = frame.copy()
    normalized["entity_uid"] = normalized["entity_uid"].astype(str)
    normalized["residue_index"] = pd.to_numeric(
        normalized["residue_index"],
        errors="coerce",
    )
    normalized["rci_flexibility"] = pd.to_numeric(
        normalized["rci_flexibility"],
        errors="coerce",
    ).clip(lower=0.0, upper=1.0)
    normalized = normalized.dropna(subset=["residue_index", "rci_flexibility"])
    normalized["residue_index"] = normalized["residue_index"].astype(int)
    keep_columns = [
        column
        for column in [
            "entity_uid",
            "chain_id",
            "residue_index",
            "residue_id",
            "rci_value",
            "rci_flexibility",
            "rci_source",
        ]
        if column in normalized.columns
    ]
    return normalized[keep_columns].drop_duplicates(
        subset=["entity_uid", "residue_index"],
        keep="first",
    )


def _resolve_adapter_path(
    value: str | Path | None,
    data_root: Path,
    repo_root: Path,
) -> Path | None:
    """Resolve an optional adapter path from common run/config contexts."""

    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return repo_root / path
    data_relative = data_root / path
    if data_relative.exists():
        return data_relative
    return repo_root / path


def _rci_profile_for_entity(
    rci_source_frame: pd.DataFrame | None,
    entity_uid: str,
) -> pd.DataFrame | None:
    """Return one entity-specific RCI profile if available."""

    if rci_source_frame is None or rci_source_frame.empty:
        return None
    selected = rci_source_frame.loc[
        rci_source_frame["entity_uid"].astype(str) == str(entity_uid)
    ].copy()
    return None if selected.empty else selected


def _rci_vs_ensemble_profile(
    rci_profile: pd.DataFrame | None,
    uncertainty_bundle: dict[str, Any],
) -> pd.DataFrame | None:
    """Join one RCI adapter profile with ensemble uncertainty residues."""

    if rci_profile is None or rci_profile.empty:
        return None
    ensemble = uncertainty_bundle["uncertainty_profile"].copy()
    joined = ensemble.merge(
        rci_profile,
        on=["entity_uid", "residue_index"],
        how="inner",
        suffixes=("", "_rci"),
    )
    if joined.empty:
        return pd.DataFrame()
    joined["rci_flexibility"] = pd.to_numeric(
        joined["rci_flexibility"],
        errors="coerce",
    )
    joined = joined.dropna(subset=["rci_flexibility"])
    return joined.reset_index(drop=True)


def _candidate_rci_mismatch(
    rci_profile: pd.DataFrame | None,
    uncertainty_bundle: dict[str, Any],
) -> np.ndarray | None:
    """Return candidate-level mismatch between displacement and RCI flexibility."""

    if rci_profile is None or rci_profile.empty:
        return None
    residue_count = len(uncertainty_bundle["common_keys"])
    flexibility = np.full(residue_count, np.nan, dtype=np.float64)
    for _, row in rci_profile.iterrows():
        residue_index = int(row["residue_index"]) - 1
        if 0 <= residue_index < residue_count:
            flexibility[residue_index] = float(row["rci_flexibility"])
    valid = np.isfinite(flexibility)
    if not np.any(valid):
        return None

    aligned_coords = uncertainty_bundle["aligned_coords"]
    reference_coords = uncertainty_bundle["reference_coords"]
    displacement = np.linalg.norm(aligned_coords - reference_coords[None, :, :], axis=2)
    mismatch: list[float] = []
    for candidate_profile in displacement:
        selected = candidate_profile[valid]
        max_value = float(np.max(selected)) if selected.size else 0.0
        normalized = selected / max(max_value, 1e-8)
        mismatch.append(float(np.mean(np.abs(normalized - flexibility[valid]))))
    return np.asarray(mismatch, dtype=np.float64)


def _rci_calibration_summary(
    rci_source_frame: pd.DataFrame | None,
    rci_profile_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize optional RCI adapter calibration against ensemble profiles."""

    if rci_source_frame is None:
        return {
            "status": "missing_rci_profile",
            "eligible_residues": 0,
            "ensemble_rmsf_rci_corr": None,
            "local_confidence_rci_corr": None,
        }
    if rci_profile_frame.empty:
        return {
            "status": "no_overlap",
            "eligible_residues": 0,
            "ensemble_rmsf_rci_corr": None,
            "local_confidence_rci_corr": None,
        }
    frame = rci_profile_frame.dropna(
        subset=["rci_flexibility", "residue_variance", "ensemble_local_confidence"]
    )
    if frame.empty:
        return {
            "status": "no_overlap",
            "eligible_residues": 0,
            "ensemble_rmsf_rci_corr": None,
            "local_confidence_rci_corr": None,
        }
    rci = frame["rci_flexibility"].to_numpy(dtype=np.float64)
    residue_variance = frame["residue_variance"].to_numpy(dtype=np.float64)
    local_confidence = frame["ensemble_local_confidence"].to_numpy(dtype=np.float64)
    return {
        "status": "ok",
        "eligible_residues": int(len(frame)),
        "ensemble_rmsf_rci_corr": _json_float_or_none(
            _safe_corr(residue_variance, rci)
        ),
        "local_confidence_rci_corr": _json_float_or_none(
            _safe_corr(local_confidence, rci)
        ),
    }


def _calibrate_chemical_shift_posteriors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add atom-family calibrated posterior intervals to a posterior frame."""

    if frame.empty or "atom_family" not in frame.columns:
        return frame
    calibrated = frame.copy()
    for atom_family, family_frame in calibrated.groupby("atom_family", dropna=True):
        scale = _family_interval_scale(family_frame)
        mask = calibrated["atom_family"].astype(str) == str(atom_family)
        calibrated.loc[mask, "posterior_calibration_scale"] = scale
        for column in [
            "posterior_std",
            "posterior_q05",
            "posterior_q10",
            "posterior_q25",
            "posterior_q50",
            "posterior_q75",
            "posterior_q90",
            "posterior_q95",
        ]:
            if column not in calibrated.columns:
                continue
            calibrated_column = f"calibrated_{column}"
            if column == "posterior_std":
                calibrated.loc[mask, calibrated_column] = (
                    pd.to_numeric(calibrated.loc[mask, column], errors="coerce") * scale
                )
                continue
            center = pd.to_numeric(
                calibrated.loc[mask, "predicted_value"],
                errors="coerce",
            )
            value = pd.to_numeric(calibrated.loc[mask, column], errors="coerce")
            calibrated.loc[mask, calibrated_column] = center + scale * (value - center)
    return calibrated


def _family_interval_scale(family_frame: pd.DataFrame) -> float:
    """Return a conservative 95-percent coverage scale for one atom family."""

    required = {"predicted_value", "target_value", "posterior_q05", "posterior_q95"}
    if family_frame.empty or not required.issubset(family_frame.columns):
        return 1.0
    frame = family_frame.dropna(subset=list(required)).copy()
    if frame.empty:
        return 1.0
    center = pd.to_numeric(frame["predicted_value"], errors="coerce")
    target = pd.to_numeric(frame["target_value"], errors="coerce")
    lower_delta = center - pd.to_numeric(frame["posterior_q05"], errors="coerce")
    upper_delta = pd.to_numeric(frame["posterior_q95"], errors="coerce") - center
    residual = target - center
    denominator = np.where(
        residual.to_numpy(dtype=np.float64) < 0.0,
        lower_delta.to_numpy(dtype=np.float64),
        upper_delta.to_numpy(dtype=np.float64),
    )
    ratio = np.abs(residual.to_numpy(dtype=np.float64)) / np.clip(
        denominator,
        1e-6,
        None,
    )
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return 1.0
    return float(np.clip(np.quantile(ratio, 0.95), 1.0, 25.0))


def _chemical_shift_posterior_calibration_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize raw and calibrated CS posterior coverage by atom family."""

    if frame.empty or "atom_family" not in frame.columns:
        return {"status": "missing_posterior", "atom_families": {}}
    summaries: dict[str, Any] = {}
    for atom_family in CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES:
        subset = frame.loc[frame["atom_family"].astype(str) == atom_family].copy()
        if subset.empty:
            continue
        summaries[atom_family] = {
            "eligible_measurements": int(len(subset)),
            "calibration_scale": _json_float_or_none(
                subset["posterior_calibration_scale"].dropna().iloc[0]
                if "posterior_calibration_scale" in subset.columns
                and not subset["posterior_calibration_scale"].dropna().empty
                else 1.0
            ),
            "raw_coverage_50": _coverage_fraction(
                subset,
                "posterior_q25",
                "posterior_q75",
            ),
            "raw_coverage_80": _coverage_fraction(
                subset,
                "posterior_q10",
                "posterior_q90",
            ),
            "raw_coverage_95": _coverage_fraction(
                subset,
                "posterior_q05",
                "posterior_q95",
            ),
            "calibrated_coverage_50": _coverage_fraction(
                subset,
                "calibrated_posterior_q25",
                "calibrated_posterior_q75",
            ),
            "calibrated_coverage_80": _coverage_fraction(
                subset,
                "calibrated_posterior_q10",
                "calibrated_posterior_q90",
            ),
            "calibrated_coverage_95": _coverage_fraction(
                subset,
                "calibrated_posterior_q05",
                "calibrated_posterior_q95",
            ),
            "mean_posterior_std": _series_mean_or_none(subset, "posterior_std"),
            "mean_calibrated_posterior_std": _series_mean_or_none(
                subset,
                "calibrated_posterior_std",
            ),
        }
    return {"status": "ok", "atom_families": summaries}


def _joint_posterior_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize final-render multi-atom chemical-shift posterior metrics."""

    if frame.empty:
        return {"status": "missing_joint_posteriors", "atom_sets": {}}
    atom_sets: dict[str, Any] = {}
    for atom_set, subset in frame.groupby("atom_set"):
        atom_sets[str(atom_set)] = {
            "eligible_residues": int(len(subset)),
            "joint_nll_macro": _series_mean_or_none(subset, "joint_nll"),
            "mahalanobis_macro": _series_mean_or_none(
                subset,
                "mahalanobis_distance",
            ),
            "joint_l2_rmse_macro": _series_mean_or_none(subset, "joint_l2_rmse"),
            "joint_coverage_50_macro": _series_mean_or_none(
                subset,
                "rectangular_coverage_50",
            ),
            "joint_coverage_80_macro": _series_mean_or_none(
                subset,
                "rectangular_coverage_80",
            ),
            "joint_coverage_95_macro": _series_mean_or_none(
                subset,
                "rectangular_coverage_95",
            ),
        }
    return {
        "status": "ok",
        "eligible_residues": int(len(frame)),
        "cs_joint_nll_macro": _series_mean_or_none(frame, "joint_nll"),
        "cs_joint_mahalanobis_macro": _series_mean_or_none(
            frame,
            "mahalanobis_distance",
        ),
        "cs_joint_l2_rmse_macro": _series_mean_or_none(frame, "joint_l2_rmse"),
        "cs_joint_coverage_50_macro": _series_mean_or_none(
            frame,
            "rectangular_coverage_50",
        ),
        "cs_joint_coverage_80_macro": _series_mean_or_none(
            frame,
            "rectangular_coverage_80",
        ),
        "cs_joint_coverage_95_macro": _series_mean_or_none(
            frame,
            "rectangular_coverage_95",
        ),
        "atom_sets": atom_sets,
    }


def _ensemble_state_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize landscape state-gallery artifacts."""

    if frame.empty:
        return {"status": "missing_ensemble_states", "entities": 0}
    entity_counts = frame.groupby("entity_uid", dropna=True)["state_id"].nunique()
    top_mass = (
        frame.sort_values(["entity_uid", "state_rank"], kind="stable")
        .groupby("entity_uid", dropna=True)["state_mass"]
        .first()
    )
    entropy = frame.groupby("entity_uid", dropna=True)["state_entropy"].first()
    return {
        "status": "ok",
        "entities": int(entity_counts.size),
        "state_count_macro": _json_float_or_none(float(entity_counts.mean())),
        "top_state_mass_macro": _json_float_or_none(float(top_mass.mean())),
        "state_entropy_macro": _json_float_or_none(float(entropy.mean())),
        "max_state_count": int(entity_counts.max()) if not entity_counts.empty else 0,
    }


def _coverage_fraction(
    frame: pd.DataFrame, lower_column: str, upper_column: str
) -> float | None:
    """Return empirical coverage for one interval column pair."""

    required = {lower_column, upper_column, "target_value"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    eligible = frame.dropna(subset=[lower_column, upper_column, "target_value"])
    if eligible.empty:
        return None
    covered = (eligible["target_value"] >= eligible[lower_column]) & (
        eligible["target_value"] <= eligible[upper_column]
    )
    return float(covered.mean())


def _series_mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    """Return a JSON-safe mean for one dataframe column."""

    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if series.empty else float(series.mean())


def _render_landscape_figures(
    figures_dir: Path,
    arrays_dir: Path,
    bundle: dict[str, Any],
) -> None:
    """Render canonical teacher-student landscape figures."""
    plt = _load_pyplot()
    _save_npz(
        arrays_dir / "free_energy_grid.npz",
        {
            "grid_x": bundle["grid_x"],
            "grid_y": bundle["grid_y"],
            "teacher_density": bundle["teacher_density"],
            "predicted_density": bundle["predicted_density"],
            "teacher_free_energy": bundle["teacher_free_energy"],
            "predicted_free_energy": bundle["predicted_free_energy"],
            "delta_free_energy": bundle["delta_free_energy"],
        },
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, free_energy, title in [
        (axes[0], bundle["teacher_free_energy"], "Teacher"),
        (axes[1], bundle["predicted_free_energy"], "Student"),
    ]:
        contour = axis.contourf(
            bundle["grid_x"],
            bundle["grid_y"],
            free_energy,
            levels=16,
            cmap="viridis",
        )
        axis.set_title(f"{title} Free Energy")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    figure.colorbar(contour, ax=axes, shrink=0.9, label="ΔG / kBT")
    figure.savefig(figures_dir / "landscape_teacher_vs_student.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    heatmap = axis.imshow(
        bundle["delta_free_energy"],
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
    )
    axis.set_title("Student - Teacher ΔG")
    axis.set_xlabel("Grid X")
    axis.set_ylabel("Grid Y")
    figure.colorbar(heatmap, ax=axis, shrink=0.9, label="ΔΔG / kBT")
    figure.savefig(figures_dir / "landscape_delta_free_energy.png", dpi=200)
    plt.close(figure)


def _render_landscape_overlay_figures(
    figures_dir: Path,
    bundle: dict[str, Any],
) -> None:
    """Render candidate-level NMR evidence overlays on landscape coordinates."""

    projection_frame = bundle.get("projection_frame")
    if not isinstance(projection_frame, pd.DataFrame) or projection_frame.empty:
        return
    required = {"projection_x", "projection_y"}
    if not required.issubset(projection_frame.columns):
        return

    overlay_specs = [
        (
            "teacher_score",
            "landscape_teacher_score.png",
            "Landscape colored by teacher score",
            "viridis",
        ),
        (
            "cs_rmse_z",
            "landscape_cs_error.png",
            "Landscape colored by chemical-shift z-error",
            "magma",
        ),
        (
            "rci_mismatch",
            "landscape_rci_mismatch.png",
            "Landscape colored by RCI mismatch",
            "plasma",
        ),
    ]
    for column, filename, title, cmap in overlay_specs:
        if column not in projection_frame.columns:
            continue
        values = pd.to_numeric(projection_frame[column], errors="coerce")
        if values.dropna().empty:
            continue
        _render_landscape_scatter_overlay(
            figures_dir=figures_dir,
            frame=projection_frame,
            color_values=values,
            output_name=filename,
            title=title,
            color_label=column,
            cmap=cmap,
        )


def _render_landscape_scatter_overlay(
    figures_dir: Path,
    frame: pd.DataFrame,
    color_values: pd.Series,
    output_name: str,
    title: str,
    color_label: str,
    cmap: str,
) -> None:
    """Render one static landscape scatter overlay."""

    plt = _load_pyplot()
    plot_frame = frame.copy()
    plot_frame["_color_value"] = color_values
    plot_frame = plot_frame.dropna(
        subset=["projection_x", "projection_y", "_color_value"]
    )
    if plot_frame.empty:
        return
    if "predicted_weight" in plot_frame.columns:
        sizes = pd.to_numeric(plot_frame["predicted_weight"], errors="coerce").fillna(
            0.0
        )
    else:
        sizes = pd.Series([0.0] * len(plot_frame), index=plot_frame.index)
    max_size = float(sizes.max()) if not sizes.empty else 0.0
    marker_sizes = 35.0 + 220.0 * sizes / max(max_size, 1e-8)

    figure, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    scatter = axis.scatter(
        plot_frame["projection_x"],
        plot_frame["projection_y"],
        c=plot_frame["_color_value"],
        s=marker_sizes,
        cmap=cmap,
        alpha=0.86,
        edgecolors="none",
    )
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    figure.colorbar(scatter, ax=axis, shrink=0.88, label=color_label)
    figure.savefig(figures_dir / output_name, dpi=200)
    plt.close(figure)


def _render_structure_uncertainty(
    figures_dir: Path,
    structures_dir: Path,
    arrays_dir: Path,
    bundle: dict[str, Any],
) -> None:
    """Render structure-level uncertainty artifacts."""
    plt = _load_pyplot()
    _write_dataframe(
        structures_dir.parent / "arrays" / "uncertainty_profile.tsv",
        bundle["uncertainty_profile"],
        sep="\t",
    )
    np.savez_compressed(
        arrays_dir / "distance_uncertainty_matrix.npz",
        distance_uncertainty=bundle["distance_uncertainty_matrix"],
        residue_ids=np.asarray(bundle["common_keys"], dtype=object),
    )
    annotated_path = structures_dir / "annotated_representative.pdb"
    _write_annotated_structure(
        source_path=Path(bundle["record"].structure_path),
        residue_ids=bundle["common_keys"],
        local_confidence=bundle["local_confidence"],
        output_path=annotated_path,
    )
    _write_viewer_scripts(structures_dir)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 4.5),
        constrained_layout=True,
        height_ratios=[2, 0.6],
    )
    residue_index = np.arange(1, len(bundle["common_keys"]) + 1)
    axes[0].plot(
        residue_index,
        bundle["local_confidence"],
        color="#1f77b4",
        linewidth=1.5,
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("AtypEmu ensemble local confidence")
    color_strip = np.tile(bundle["local_confidence"], (1, 1))
    axes[1].imshow(
        color_strip,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
        origin="lower",
    )
    axes[1].set_yticks([])
    axes[1].set_xlabel("Residue index")
    figure.savefig(figures_dir / "ensemble_local_confidence_structure.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    heatmap = axis.imshow(
        bundle["distance_uncertainty_matrix"],
        origin="lower",
        cmap="magma",
        aspect="auto",
    )
    axis.set_title("Distance uncertainty matrix")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Residue index")
    figure.colorbar(heatmap, ax=axis, shrink=0.85, label="Cα distance std (Å)")
    figure.savefig(figures_dir / "distance_uncertainty_heatmap.png", dpi=200)
    plt.close(figure)


def _render_ensemble_state_gallery(
    figures_dir: Path,
    structures_dir: Path,
    bundle: dict[str, Any],
) -> None:
    """Render representative state structures and state-gallery figures."""

    states = bundle.get("ensemble_states_frame")
    assignments = bundle.get("state_assignments_frame")
    if not isinstance(states, pd.DataFrame) or states.empty:
        return
    if not isinstance(assignments, pd.DataFrame) or assignments.empty:
        return

    for state_detail in bundle.get("state_details", []):
        state_id = str(state_detail["state_id"])
        _write_annotated_structure(
            source_path=Path(state_detail["record"].structure_path),
            residue_ids=bundle["common_keys"],
            local_confidence=state_detail["local_confidence"],
            output_path=structures_dir / f"{state_id}_representative.pdb",
        )
    _write_state_viewer_scripts(structures_dir, states["state_id"].astype(str).tolist())
    _render_state_cluster_figure(figures_dir, states, assignments)
    _render_state_mass_figure(figures_dir, states)
    _render_state_nmr_evidence_figure(figures_dir, states)


def _render_state_cluster_figure(
    figures_dir: Path,
    states: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    """Render landscape states with representative medoids marked."""

    required = {"projection_x", "projection_y", "state_id", "predicted_weight"}
    if not required.issubset(assignments.columns):
        return
    plt = _load_pyplot()
    frame = assignments.dropna(subset=["projection_x", "projection_y"]).copy()
    if frame.empty:
        return
    state_ids = states.sort_values("state_rank")["state_id"].astype(str).tolist()
    color_map = {
        state_id: plt.get_cmap("tab10")(index % 10)
        for index, state_id in enumerate(state_ids)
    }
    weights = pd.to_numeric(frame["predicted_weight"], errors="coerce").fillna(0.0)
    marker_sizes = 35.0 + 260.0 * weights / max(float(weights.max()), 1e-8)

    figure, axis = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    for state_id in state_ids:
        subset = frame.loc[frame["state_id"].astype(str) == state_id]
        if subset.empty:
            continue
        subset_sizes = marker_sizes.loc[subset.index]
        axis.scatter(
            subset["projection_x"],
            subset["projection_y"],
            s=subset_sizes,
            color=color_map[state_id],
            alpha=0.74,
            edgecolors="none",
            label=state_id,
        )
    representatives = frame.loc[frame["is_state_representative"].astype(bool)]
    if not representatives.empty:
        axis.scatter(
            representatives["projection_x"],
            representatives["projection_y"],
            s=220,
            marker="*",
            color="black",
            edgecolors="white",
            linewidths=0.8,
            label="state medoid",
        )
    axis.set_title("Energy landscape representative states")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(loc="best", frameon=True, framealpha=0.94, title="State")
    figure.savefig(figures_dir / "landscape_state_clusters.png", dpi=200)
    plt.close(figure)


def _render_state_mass_figure(figures_dir: Path, states: pd.DataFrame) -> None:
    """Render state mass and relative free energy bars."""

    required = {"state_id", "state_mass", "relative_free_energy_kbt"}
    if states.empty or not required.issubset(states.columns):
        return
    plt = _load_pyplot()
    frame = states.sort_values("state_rank").copy()
    figure, left_axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    right_axis = left_axis.twinx()
    x = np.arange(len(frame))
    left_axis.bar(
        x,
        frame["state_mass"],
        color="#4c78a8",
        alpha=0.86,
        label="Student posterior state mass",
    )
    right_axis.plot(
        x,
        frame["relative_free_energy_kbt"],
        color="#f58518",
        marker="o",
        linewidth=1.6,
        label="Relative free energy",
    )
    left_axis.set_xticks(x)
    left_axis.set_xticklabels(frame["state_id"].astype(str), rotation=25, ha="right")
    left_axis.set_ylabel("State mass")
    right_axis.set_ylabel("Relative ΔG / kBT")
    left_axis.set_title("State occupancy and relative free energy")
    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
        frameon=True,
        framealpha=0.94,
        title="Legend",
    )
    figure.savefig(figures_dir / "state_mass_free_energy.png", dpi=200)
    plt.close(figure)


def _render_state_nmr_evidence_figure(figures_dir: Path, states: pd.DataFrame) -> None:
    """Render state-level NMR evidence summary heatmap."""

    metrics = [
        "state_mass",
        "cs_mae_ppm_weighted_mean",
        "cs_rmse_z_weighted_mean",
        "rci_mismatch_weighted_mean",
        "state_local_confidence_mean",
    ]
    available = [column for column in metrics if column in states.columns]
    if not available:
        return
    frame = states.sort_values("state_rank").copy()
    matrix = frame[available].apply(pd.to_numeric, errors="coerce")
    if matrix.dropna(how="all").empty:
        return
    normalized = matrix.copy()
    for column in available:
        values = normalized[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        normalized[column] = (values - low) / max(high - low, 1e-8)

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    heatmap = axis.imshow(normalized.to_numpy(dtype=np.float64), cmap="viridis")
    axis.set_yticks(np.arange(len(frame)))
    axis.set_yticklabels(frame["state_id"].astype(str))
    axis.set_xticks(np.arange(len(available)))
    axis.set_xticklabels(available, rotation=35, ha="right")
    axis.set_title("State-level NMR evidence diagnostics")
    figure.colorbar(heatmap, ax=axis, shrink=0.86, label="Column-normalized value")
    figure.savefig(figures_dir / "state_nmr_evidence_heatmap.png", dpi=200)
    plt.close(figure)


def _render_rci_adapter_figures(
    figures_dir: Path,
    rci_profile: pd.DataFrame,
) -> None:
    """Render RCI adapter calibration figures against ensemble diagnostics."""

    if rci_profile.empty:
        return
    required = {"residue_index", "rci_flexibility"}
    if not required.issubset(rci_profile.columns):
        return

    plt = _load_pyplot()
    frame = rci_profile.copy()
    for column in [
        "residue_index",
        "rci_flexibility",
        "residue_variance",
        "ensemble_local_confidence",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    scatter_frame = frame.dropna(subset=["rci_flexibility", "residue_variance"])
    if not scatter_frame.empty:
        figure, axis = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
        source_label = _rci_source_label(scatter_frame)
        axis.scatter(
            scatter_frame["rci_flexibility"],
            scatter_frame["residue_variance"],
            color="#d97706",
            alpha=0.78,
            edgecolors="none",
            label=f"{source_label} flexibility vs ensemble RMSF",
        )
        axis.set_xlabel("RCI adapter flexibility")
        axis.set_ylabel("Ensemble RMSF / residue variance (Å)")
        axis.set_title("RCI adapter vs ensemble RMSF")
        axis.legend(
            loc="best",
            frameon=True,
            framealpha=0.96,
            title="Legend",
        )
        axis.grid(alpha=0.25, linewidth=0.6)
        figure.savefig(figures_dir / "rci_vs_ensemble_rmsf.png", dpi=200)
        plt.close(figure)

    profile_frame = frame.dropna(
        subset=["residue_index", "rci_flexibility", "ensemble_local_confidence"]
    ).sort_values("residue_index")
    if not profile_frame.empty:
        figure, left_axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
        right_axis = left_axis.twinx()
        confidence_line = left_axis.plot(
            profile_frame["residue_index"],
            profile_frame["ensemble_local_confidence"],
            color="#1f77b4",
            linewidth=1.6,
            label="AtypEmu ensemble local confidence (0-100)",
        )
        rci_line = right_axis.plot(
            profile_frame["residue_index"],
            profile_frame["rci_flexibility"],
            color="#d97706",
            linewidth=1.5,
            alpha=0.86,
            label=f"{_rci_source_label(profile_frame)} flexibility (0-1)",
        )
        left_axis.set_ylim(0, 100)
        right_axis.set_ylim(0, 1)
        left_axis.set_xlabel("Residue index")
        left_axis.set_ylabel("AtypEmu ensemble local confidence")
        right_axis.set_ylabel("RCI adapter flexibility")
        left_axis.set_title("Local confidence profile with RCI adapter")
        left_axis.grid(alpha=0.25, linewidth=0.6)
        legend_handles = confidence_line + rci_line
        left_axis.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            frameon=True,
            framealpha=0.96,
            title="Legend",
        )
        figure.savefig(figures_dir / "rci_local_confidence_profile.png", dpi=200)
        plt.close(figure)


def _rci_source_label(frame: pd.DataFrame) -> str:
    """Return a compact display label for an RCI adapter source."""

    if "rci_source" not in frame.columns:
        return "RCI adapter"
    sources = [
        str(value)
        for value in frame["rci_source"].dropna().astype(str).unique().tolist()
        if value
    ]
    if not sources:
        return "RCI adapter"
    source = sources[0]
    if source == "pRCI_random_coil_proximity":
        return "pRCI adapter"
    return source


def _render_chemical_shift_figures(
    figures_dir: Path,
    posterior_frame: pd.DataFrame,
    joint_posterior_frame: pd.DataFrame,
    representative_entity_uid: str,
) -> None:
    """Render posterior chemical-shift uncertainty figures."""
    representative = posterior_frame.loc[
        posterior_frame["entity_uid"] == representative_entity_uid
    ].copy()
    representative["residue_index"] = representative["target_id"].map(
        _residue_index_from_target_id
    )

    for atom_family in CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES:
        _render_atom_family_posterior_figure(
            figures_dir=figures_dir,
            representative=representative,
            atom_family=atom_family,
        )

    _render_pair_density_figure(
        figures_dir=figures_dir,
        representative=representative,
        atom_x="HN",
        atom_y="N",
        output_name="pseudo_hsqc_density.png",
        title="Pseudo-HSQC density",
        x_label="1H (ppm)",
        y_label="15N (ppm)",
        reverse_axes=True,
    )
    _render_pair_density_figure(
        figures_dir=figures_dir,
        representative=representative,
        atom_x="CA",
        atom_y="CB",
        output_name="pseudo_cacb_density.png",
        title="CA-CB chemical-shift posterior density",
        x_label="CA (ppm)",
        y_label="CB (ppm)",
        reverse_axes=False,
    )
    _render_pair_density_figure(
        figures_dir=figures_dir,
        representative=representative,
        atom_x="CA",
        atom_y="C'",
        output_name="pseudo_ca_c_density.png",
        title="CA-C' chemical-shift posterior density",
        x_label="CA (ppm)",
        y_label="C' (ppm)",
        reverse_axes=False,
    )
    _render_joint_parallel_figure(
        figures_dir=figures_dir,
        joint_posterior_frame=joint_posterior_frame,
        representative_entity_uid=representative_entity_uid,
        atom_set="HN_N_CA",
        output_name="chemical_shift_joint_HN_N_CA.png",
    )
    _render_joint_parallel_figure(
        figures_dir=figures_dir,
        joint_posterior_frame=joint_posterior_frame,
        representative_entity_uid=representative_entity_uid,
        atom_set="CA_CB_C",
        output_name="chemical_shift_joint_CA_CB_C.png",
    )


def _render_atom_family_posterior_figure(
    figures_dir: Path,
    representative: pd.DataFrame,
    atom_family: str,
) -> None:
    """Render one atom-family posterior uncertainty strip."""

    plt = _load_pyplot()
    frame = representative.loc[representative["atom_family"] == atom_family].dropna(
        subset=["residue_index", "predicted_value"]
    )
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for index, (_, row) in enumerate(frame.sort_values("residue_index").iterrows()):
        std = _row_finite_float(row, "calibrated_posterior_std")
        if std is None:
            std = _row_finite_float(row, "posterior_std")
        if std is None:
            std = 0.1
        y_axis = np.linspace(
            row["predicted_value"] - 3.0 * std,
            row["predicted_value"] + 3.0 * std,
            200,
        )
        density = norm.pdf(y_axis, loc=row["predicted_value"], scale=max(std, 1e-3))
        density = 0.35 * density / np.max(density)
        axis.fill_betweenx(
            y_axis,
            row["residue_index"] - density,
            row["residue_index"] + density,
            color="#4c72b0",
            alpha=0.35,
            label="Posterior density" if index == 0 else None,
        )
        if pd.notna(row.get("target_value")):
            axis.scatter(
                row["residue_index"],
                row["target_value"],
                color="#dd8452",
                s=10,
                zorder=3,
                label="Experimental chemical shift" if index == 0 else None,
            )
    axis.scatter(
        frame["residue_index"],
        frame["predicted_value"],
        color="#1f77b4",
        s=12,
        zorder=4,
        label="Posterior mean",
    )
    axis.set_xlabel("Residue index")
    axis.set_ylabel(f"{atom_family} chemical shift (ppm)")
    axis.set_title(f"{atom_family} posterior uncertainty")
    axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.96,
        title="Legend",
    )
    axis.grid(alpha=0.22, linewidth=0.6)
    figure.savefig(
        figures_dir / f"chemical_shift_violin_{_atom_family_slug(atom_family)}.png",
        dpi=200,
    )
    plt.close(figure)


def _render_pair_density_figure(
    figures_dir: Path,
    representative: pd.DataFrame,
    atom_x: str,
    atom_y: str,
    output_name: str,
    title: str,
    x_label: str,
    y_label: str,
    reverse_axes: bool,
) -> None:
    """Render a 2D chemical-shift posterior density panel."""

    paired = _paired_atom_family_frame(representative, atom_x, atom_y)
    if paired.empty:
        return
    plt = _load_pyplot()
    paired["x_std"] = pd.to_numeric(paired["x_std"], errors="coerce").fillna(0.1)
    paired["y_std"] = pd.to_numeric(paired["y_std"], errors="coerce").fillna(0.1)
    x_padding = max(0.5 * float(paired["x_std"].median()), 0.5)
    y_padding = max(0.5 * float(paired["y_std"].median()), 0.5)
    x_grid = np.linspace(
        paired["x_mean"].min() - x_padding,
        paired["x_mean"].max() + x_padding,
        160,
    )
    y_grid = np.linspace(
        paired["y_mean"].min() - y_padding,
        paired["y_mean"].max() + y_padding,
        160,
    )
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(grid_x, dtype=np.float64)
    for _, row in paired.iterrows():
        x_sigma = max(float(row["x_std"]), 1e-3)
        y_sigma = max(float(row["y_std"]), 1e-3)
        density += np.exp(
            -0.5
            * (
                np.square((grid_x - row["x_mean"]) / x_sigma)
                + np.square((grid_y - row["y_mean"]) / y_sigma)
            )
        ) / (2.0 * np.pi * x_sigma * y_sigma)

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.contour(grid_x, grid_y, density, levels=8, cmap="Blues")
    axis.scatter(
        paired["x_mean"],
        paired["y_mean"],
        color="#1f77b4",
        s=10,
        alpha=0.65,
        label="Posterior mean",
    )
    observed = paired.dropna(subset=["exp_x", "exp_y"])
    if not observed.empty:
        for _, row in observed.iterrows():
            axis.plot(
                [row["x_mean"], row["exp_x"]],
                [row["y_mean"], row["exp_y"]],
                color="#64748b",
                alpha=0.18,
                linewidth=0.8,
            )
        axis.scatter(
            observed["exp_x"],
            observed["exp_y"],
            color="#dd8452",
            s=12,
            label="Experimental",
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    if reverse_axes:
        axis.invert_xaxis()
        axis.invert_yaxis()
    axis.legend(loc="upper right")
    figure.savefig(figures_dir / output_name, dpi=200)
    plt.close(figure)


def _render_joint_parallel_figure(
    figures_dir: Path,
    joint_posterior_frame: pd.DataFrame,
    representative_entity_uid: str,
    atom_set: str,
    output_name: str,
) -> None:
    """Render a compact parallel-coordinate joint posterior diagnostic."""

    if joint_posterior_frame.empty:
        return
    subset = joint_posterior_frame.loc[
        (joint_posterior_frame["entity_uid"] == representative_entity_uid)
        & (joint_posterior_frame["atom_set"] == atom_set)
    ].copy()
    if subset.empty:
        return
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for _, row in subset.head(80).iterrows():
        atom_families = json.loads(row["atom_families_json"])
        mean_vector = np.asarray(
            json.loads(row["posterior_mean_vector_json"]), dtype=float
        )
        target_vector = np.asarray(json.loads(row["target_vector_json"]), dtype=float)
        x_values = np.arange(len(atom_families), dtype=float)
        axis.plot(x_values, mean_vector, color="#1f77b4", alpha=0.30, linewidth=1.0)
        axis.scatter(x_values, target_vector, color="#dd8452", alpha=0.45, s=10)
    axis.set_xticks(np.arange(len(json.loads(subset.iloc[0]["atom_families_json"]))))
    axis.set_xticklabels(json.loads(subset.iloc[0]["atom_families_json"]))
    axis.set_ylabel("Chemical shift (ppm)")
    axis.set_title(f"{atom_set} joint posterior profile")
    figure.savefig(figures_dir / output_name, dpi=200)
    plt.close(figure)


def _paired_atom_family_frame(
    representative: pd.DataFrame,
    atom_x: str,
    atom_y: str,
) -> pd.DataFrame:
    """Return residue-wise paired atom-family posterior rows."""

    keys = ["entity_uid", "residue_index"]
    x_frame = _select_pair_atom_family(representative, atom_x, keys, prefix="x")
    y_frame = _select_pair_atom_family(representative, atom_y, keys, prefix="y")
    if x_frame.empty or y_frame.empty:
        return pd.DataFrame()
    return x_frame.merge(y_frame, on=keys, how="inner").sort_values(keys)


def _select_pair_atom_family(
    frame: pd.DataFrame,
    atom_family: str,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Select and rename one atom family for static pair rendering."""

    subset = frame.loc[frame["atom_family"].astype(str) == atom_family].copy()
    if subset.empty:
        return pd.DataFrame()
    std_column = (
        "calibrated_posterior_std"
        if "calibrated_posterior_std" in subset.columns
        else "posterior_std"
    )
    selected = subset[keys + ["predicted_value", std_column, "target_value"]].copy()
    return selected.rename(
        columns={
            "predicted_value": f"{prefix}_mean",
            std_column: f"{prefix}_std",
            "target_value": f"exp_{prefix}",
        }
    )


def _row_finite_float(row: pd.Series, column: str) -> float | None:
    """Return a finite float from one dataframe row."""

    if column not in row:
        return None
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _build_external_slice_calibration(
    data_root: Path,
    uncertainty_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build MFIB and FuzDB uncertainty-linked slice dashboards."""
    if not uncertainty_rows:
        return {}

    low_confidence_entities = {
        row["entity_uid"]
        for row in uncertainty_rows
        if float(row.get("ensemble_local_confidence_mean", math.inf)) < 70.0
    }
    if not low_confidence_entities:
        return {}

    meta_registry = MetaDataFrameRegistry.from_data_root(data_root)
    links = meta_registry.load_table("links")
    tags = meta_registry.load_table("tags")

    linked_by_source: dict[str, set[str]] = {"mfib": set(), "fuzdb": set()}
    for entity_uid in low_confidence_entities:
        left_matches = links.loc[links["left_entity_uid"] == entity_uid].copy()
        right_matches = links.loc[links["right_entity_uid"] == entity_uid].copy()
        for source_id, target_column in [
            ("mfib", "right_entity_uid"),
            ("fuzdb", "right_entity_uid"),
        ]:
            linked_by_source[source_id].update(
                left_matches.loc[
                    left_matches["right_source_id"] == source_id, target_column
                ]
                .astype(str)
                .tolist()
            )
        for source_id, target_column in [
            ("mfib", "left_entity_uid"),
            ("fuzdb", "left_entity_uid"),
        ]:
            linked_by_source[source_id].update(
                right_matches.loc[
                    right_matches["left_source_id"] == source_id, target_column
                ]
                .astype(str)
                .tolist()
            )

    report: dict[str, Any] = {}
    for source_id, entity_uids in linked_by_source.items():
        if not entity_uids:
            report[source_id] = {"linked_entry_count": 0}
            continue
        source_tags = tags.loc[tags["entry_uid"].isin(entity_uids)].copy()
        namespaces = (
            ["evidence_level"]
            if source_id == "mfib"
            else ["topology_class", "mechanism_category"]
        )
        report[source_id] = {
            "linked_entry_count": int(len(entity_uids)),
            "tag_counts": {
                namespace: source_tags.loc[source_tags["namespace"] == namespace, "tag"]
                .value_counts()
                .to_dict()
                for namespace in namespaces
            },
        }
    return report


def _density_grid(
    coordinates: np.ndarray,
    weights: np.ndarray,
    grid_size: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one normalized 2D density grid."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x_pad = 0.1 * max(float(np.ptp(x)), 1.0)
    y_pad = 0.1 * max(float(np.ptp(y)), 1.0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(float(x.min() - x_pad), float(x.max() + x_pad), grid_size),
        np.linspace(float(y.min() - y_pad), float(y.max() + y_pad), grid_size),
    )
    normalized = weights / np.clip(weights.sum(), 1e-12, None)

    density: np.ndarray
    if method == "weighted_kde" and coordinates.shape[0] >= 3:
        try:
            kde = gaussian_kde(coordinates.T, weights=normalized)
            density = kde(np.vstack([grid_x.reshape(-1), grid_y.reshape(-1)])).reshape(
                grid_x.shape
            )
        except Exception:
            density = _histogram_density(x, y, normalized, grid_x, grid_y)
    else:
        density = _histogram_density(x, y, normalized, grid_x, grid_y)
    density = density / np.clip(density.sum(), 1e-12, None)
    return grid_x, grid_y, density


def _histogram_density(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """Return one weighted histogram density on a precomputed grid."""
    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=[grid_x.shape[1], grid_y.shape[0]],
        range=[
            [float(grid_x.min()), float(grid_x.max())],
            [float(grid_y.min()), float(grid_y.max())],
        ],
        weights=weights,
    )
    return hist.T + 1e-12


def _free_energy_from_density(density: np.ndarray) -> np.ndarray:
    """Convert one density grid into a relative free-energy grid."""
    free_energy = -np.log(np.clip(density, 1e-12, None))
    return free_energy - np.nanmin(free_energy)


def _mass_coverage_95(
    teacher_density: np.ndarray,
    predicted_density: np.ndarray,
) -> float:
    """Return student mass inside the teacher 95-percent support."""
    teacher_flat = teacher_density.reshape(-1)
    predicted_flat = predicted_density.reshape(-1)
    order = np.argsort(teacher_flat)[::-1]
    cumulative = np.cumsum(teacher_flat[order])
    keep = order[cumulative <= 0.95]
    if keep.size == 0:
        keep = order[:1]
    return float(predicted_flat[keep].sum())


def _discrete_js(p: np.ndarray, q: np.ndarray) -> float:
    """Return one discrete Jensen-Shannon divergence."""
    p_norm = p.reshape(-1) / np.clip(p.sum(), 1e-12, None)
    q_norm = q.reshape(-1) / np.clip(q.sum(), 1e-12, None)
    mixture = 0.5 * (p_norm + q_norm)
    return float(
        0.5
        * np.sum(
            p_norm
            * (
                np.log(np.clip(p_norm, 1e-12, None))
                - np.log(np.clip(mixture, 1e-12, None))
            )
        )
        + 0.5
        * np.sum(
            q_norm
            * (
                np.log(np.clip(q_norm, 1e-12, None))
                - np.log(np.clip(mixture, 1e-12, None))
            )
        )
    )


def _subsample_indices(total: int, max_count: int) -> np.ndarray:
    """Return evenly spaced residue indices."""
    if total <= max_count:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, max_count, dtype=int)


def _align_coords_to_reference(
    coords: np.ndarray,
    reference_coords: np.ndarray,
) -> np.ndarray:
    """Align one coordinate set to the reference with a Kabsch transform."""
    center = coords.mean(axis=0)
    reference_center = reference_coords.mean(axis=0)
    mobile = coords - center
    target = reference_coords - reference_center
    covariance = mobile.T @ target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    return mobile @ rotation + reference_center


def _local_confidence_profile(
    aligned_coords: np.ndarray,
    reference_coords: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return one lDDT-like local-consistency profile on a 0-100 scale."""
    residue_count = reference_coords.shape[0]
    reference_distances = np.linalg.norm(
        reference_coords[:, None, :] - reference_coords[None, :, :],
        axis=2,
    )
    thresholds = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    confidence = np.zeros(residue_count, dtype=np.float64)
    for residue_index in range(residue_count):
        neighbors = np.where(
            (reference_distances[residue_index] < 15.0)
            & (np.arange(residue_count) != residue_index)
        )[0]
        if neighbors.size == 0:
            confidence[residue_index] = 100.0
            continue
        sample_scores = []
        reference_local = reference_distances[residue_index, neighbors]
        for coords in aligned_coords:
            local_distances = np.linalg.norm(
                coords[residue_index] - coords[neighbors],
                axis=1,
            )
            delta = np.abs(local_distances - reference_local)[:, None]
            sample_scores.append(float(np.mean(delta < thresholds)))
        confidence[residue_index] = 100.0 * float(np.dot(weights, sample_scores))
    return confidence


def _write_annotated_structure(
    source_path: Path,
    residue_ids: list[str],
    local_confidence: np.ndarray,
    output_path: Path,
) -> None:
    """Write one representative PDB with confidence in the B-factor field."""
    parser = PDBParser(QUIET=True)
    structure = copy.deepcopy(parser.get_structure(source_path.stem, str(source_path)))
    confidence_by_residue = {
        residue_id: float(value)
        for residue_id, value in zip(residue_ids, local_confidence, strict=True)
    }
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = (
                    f"{chain.id}:{residue.id[1]}:{residue.resname.strip().upper()}"
                )
                bfactor = confidence_by_residue.get(residue_key)
                if bfactor is None:
                    continue
                for atom in residue:
                    atom.set_bfactor(float(bfactor))
        break
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def _write_viewer_scripts(structures_dir: Path) -> None:
    """Write optional ChimeraX and PyMOL helper scripts."""
    (structures_dir / "view_chimerax.cxc").write_text(
        "\n".join(
            [
                "open annotated_representative.pdb",
                "color byattribute bfactor palette alphafold",
                "labelopt info residue",
                "",
            ]
        )
    )
    (structures_dir / "view_pymol.pml").write_text(
        "\n".join(
            [
                "load annotated_representative.pdb",
                "spectrum b, blue_cyan_yellow_red, minimum=0, maximum=100",
                "show cartoon",
                "",
            ]
        )
    )


def _write_state_viewer_scripts(structures_dir: Path, state_ids: list[str]) -> None:
    """Write optional viewer scripts for representative ensemble states."""

    state_files = [f"{state_id}_representative.pdb" for state_id in state_ids]
    (structures_dir / "view_ensemble_states_chimerax.cxc").write_text(
        "\n".join(
            [
                *(f"open {filename}" for filename in state_files),
                "color byattribute bfactor palette alphafold",
                "tile",
                "",
            ]
        )
    )
    (structures_dir / "view_ensemble_states_pymol.pml").write_text(
        "\n".join(
            [
                *(
                    f"load {filename}, {Path(filename).stem}"
                    for filename in state_files
                ),
                "spectrum b, blue_cyan_yellow_red, minimum=0, maximum=100",
                "show cartoon",
                "",
            ]
        )
    )


def _paired_hsqc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one paired HN/N frame for pseudo-HSQC rendering."""
    hn = frame.loc[frame["atom_family"] == "HN"].copy()
    nn = frame.loc[frame["atom_family"] == "N"].copy()
    if hn.empty or nn.empty:
        return pd.DataFrame()
    key_columns = ["entity_uid", "residue_index"]
    paired = hn.merge(
        nn,
        on=key_columns,
        suffixes=("_h", "_n"),
        how="inner",
    )
    if paired.empty:
        return paired
    return pd.DataFrame(
        {
            "entity_uid": paired["entity_uid"],
            "residue_index": paired["residue_index"],
            "h_mean": paired["predicted_value_h"],
            "n_mean": paired["predicted_value_n"],
            "h_std": paired["posterior_std_h"].fillna(0.05),
            "n_std": paired["posterior_std_n"].fillna(0.5),
            "exp_h": paired["target_value_h"],
            "exp_n": paired["target_value_n"],
        }
    )


def _residue_index_from_target_id(target_id: str) -> int | None:
    """Parse one residue index from a target identifier."""
    parts = str(target_id).split(":")
    if len(parts) < 4:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _atom_family_slug(atom_family: str) -> str:
    """Return a filesystem-safe atom-family slug."""

    return str(atom_family).replace("'", "prime").replace("/", "_")


def _aggregate_metric_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, float | None]:
    """Aggregate example-level metric rows into macro summaries."""
    if not rows:
        return {name: None for name in metric_names}
    frame = pd.DataFrame(rows)
    payload: dict[str, float | None] = {}
    for metric_name in metric_names:
        if metric_name not in frame:
            payload[metric_name] = None
            continue
        series = pd.to_numeric(frame[metric_name], errors="coerce").dropna()
        payload[metric_name] = None if series.empty else float(series.mean())
    return payload


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation when finite and well-defined."""
    if x.size < 2 or y.size < 2:
        return math.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _weighted_quantile_1d(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return a weighted empirical quantile for one vector."""

    if values.size == 0:
        return math.nan
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / np.clip(cumulative[-1], 1e-12, None)
    return float(np.interp(quantile, cumulative, sorted_values))


def _json_float_or_none(value: float) -> float | None:
    """Return a JSON-safe float or ``None`` for non-finite values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_dumps(value: Any) -> str:
    """Return compact deterministic JSON for dataframe object columns."""

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _load_pyplot():
    """Import matplotlib pyplot with the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_json(path: Path, payload: Any) -> None:
    """Write one JSON payload with deterministic formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write one compressed NPZ archive."""
    np.savez_compressed(path, **payload)


def _write_dataframe(path: Path, frame: pd.DataFrame, sep: str | None = None) -> None:
    """Write one dataframe to parquet or TSV depending on suffix."""
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if path.suffix == ".tsv":
        frame.to_csv(path, sep=sep or "\t", index=False)
        return
    raise ValueError(f"Unsupported dataframe output format: {path}")
