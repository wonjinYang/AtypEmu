"""Moment-consistent BioEmu-style posterior artifacts for AtypEmu."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atypemu.training.config import StudentTrainingConfig
from atypemu.training.data import CHEMICAL_SHIFT_FAMILIES, PreparedTeacherExample
from atypemu.training.secondary_shift import random_coil_reference_for_target_id


def build_moment_diffusion_artifacts(
    *,
    model: Any,
    train_examples: list[PreparedTeacherExample],
    val_examples: list[PreparedTeacherExample],
    device: Any,
    config: StudentTrainingConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write moment-oracle and state-token diagnostics for one best checkpoint."""
    output_dir_path = Path(output_dir)
    arrays_dir = output_dir_path / "reports" / "arrays"
    metrics_dir = output_dir_path / "reports" / "metrics"
    figures_dir = output_dir_path / "reports" / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not config.enable_moment_head:
        _write_empty_artifacts(arrays_dir)
        report = {"status": "disabled"}
        _save_json(metrics_dir / "moment_oracle_report.json", report)
        _save_json(metrics_dir / "moment_consistency_report.json", report)
        return report

    prediction_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    masked_prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    was_training = bool(getattr(model, "training", False))
    model.eval()
    for split_name, examples in [("train", train_examples), ("val", val_examples)]:
        for example_index, example in enumerate(examples):
            payload = _example_payload(
                model=model,
                example=example,
                split_name=split_name,
                example_index=example_index,
                device=device,
                config=config,
            )
            prediction_rows.extend(payload["prediction_rows"])
            weight_rows.extend(payload["weight_rows"])
            state_rows.extend(payload["state_rows"])
            latent_rows.extend(payload["latent_rows"])
            feature_rows.extend(payload["feature_rows"])
            masked_prediction_rows.extend(payload["masked_prediction_rows"])
            if payload["summary_row"]:
                summary_rows.append(payload["summary_row"])
    if was_training:
        model.train()

    prediction_frame = pd.DataFrame(prediction_rows)
    weight_frame = pd.DataFrame(weight_rows)
    state_frame = pd.DataFrame(state_rows)
    latent_frame = pd.DataFrame(latent_rows)
    feature_frame = pd.DataFrame(feature_rows)
    masked_prediction_frame = pd.DataFrame(masked_prediction_rows)
    summary_frame = pd.DataFrame(summary_rows)
    prediction_frame.to_parquet(arrays_dir / "moment_predictions.parquet", index=False)
    prediction_frame.to_parquet(
        arrays_dir / "moment_oracle_predictions.parquet", index=False
    )
    weight_frame.to_parquet(arrays_dir / "moment_oracle_weights.parquet", index=False)
    state_frame.to_parquet(
        arrays_dir / "moment_oracle_state_occupancy.parquet", index=False
    )
    state_frame.to_parquet(arrays_dir / "posterior_state_tokens.parquet", index=False)
    latent_frame.to_parquet(
        arrays_dir / "posterior_latent_samples.parquet", index=False
    )
    feature_frame.to_parquet(
        arrays_dir / "nmr_structural_features.parquet", index=False
    )
    masked_prediction_frame.to_parquet(
        arrays_dir / "moment_masked_holdout_predictions.parquet", index=False
    )
    measure_frame, calibration_report = _build_measure_aware_targets(
        prediction_frame
    )
    joint_frame, joint_report = _build_joint_nmr_posteriors(prediction_frame)
    ccc_decomposition_report = _build_ccc_decomposition_report(measure_frame)
    measure_frame.to_parquet(
        arrays_dir / "measure_aware_targets.parquet", index=False
    )
    joint_frame.to_parquet(arrays_dir / "joint_nmr_posteriors.parquet", index=False)

    oracle_report = _summarize_oracle(summary_frame, prediction_frame)
    consistency_report = _summarize_consistency(prediction_frame)
    _save_json(metrics_dir / "moment_oracle_report.json", oracle_report)
    _save_json(metrics_dir / "moment_consistency_report.json", consistency_report)
    _save_json(
        metrics_dir / "entry_family_calibration_report.json",
        calibration_report,
    )
    _save_json(metrics_dir / "ccc_decomposition_report.json", ccc_decomposition_report)
    _save_json(output_dir_path / "moment_oracle_report.json", oracle_report)
    _render_measure_aware_figures(
        measure_frame=measure_frame,
        joint_frame=joint_frame,
        masked_prediction_frame=masked_prediction_frame,
        figures_dir=figures_dir,
    )
    _render_moment_figures(
        prediction_frame=prediction_frame,
        state_frame=state_frame,
        figures_dir=figures_dir,
    )
    return {
        "status": "ok" if summary_rows else "no_chemical_shift_examples",
        "oracle": oracle_report,
        "consistency": consistency_report,
    }


def _write_empty_artifacts(arrays_dir: Path) -> None:
    """Write empty parquet placeholders for disabled moment diagnostics."""
    for name in [
        "moment_predictions.parquet",
        "moment_oracle_predictions.parquet",
        "moment_oracle_weights.parquet",
        "moment_oracle_state_occupancy.parquet",
        "posterior_state_tokens.parquet",
        "posterior_latent_samples.parquet",
        "nmr_structural_features.parquet",
        "moment_masked_holdout_predictions.parquet",
        "measure_aware_targets.parquet",
        "joint_nmr_posteriors.parquet",
    ]:
        pd.DataFrame().to_parquet(arrays_dir / name, index=False)


def _example_payload(
    *,
    model: Any,
    example: PreparedTeacherExample,
    split_name: str,
    example_index: int,
    device: Any,
    config: StudentTrainingConfig,
) -> dict[str, Any]:
    """Return moment/oracle rows for one example."""
    from atypemu.training.trainer import (
        chemical_shift_evidence_mask,
        model_outputs_for_example,
        to_tensors,
    )

    channel = example.channels.get("chemical_shifts")
    if channel is None:
        return _empty_payload()
    tensors = to_tensors(example, device)
    with _torch_no_grad():
        outputs = model_outputs_for_example(
            model=model,
            tensors=tensors,
            evidence_mask=None,
            config=config,
        )
    weights = outputs["weights"].detach().cpu().numpy().astype(np.float64)
    moment_mu = outputs["moment_mu"].detach().cpu().numpy().astype(np.float64)
    moment_sigma = _optional_output_vector(outputs, "moment_sigma", len(moment_mu))
    sigma_scale = _optional_output_vector(outputs, "moment_sigma_scale", len(moment_mu))
    adapter_ids = _optional_output_vector(
        outputs,
        "moment_family_adapter_id",
        len(moment_mu),
    )
    outlier_scores = _optional_output_vector(
        outputs,
        "moment_outlier_score",
        len(moment_mu),
    )
    values = channel.values.astype(np.float64, copy=False)
    mask = channel.mask.astype(bool, copy=False)
    targets = channel.target_values.astype(np.float64, copy=False)
    oracle_weights = _projected_simplex_oracle(
        values=values, mask=mask, targets=targets
    )
    oracle_mean = _weighted_expectation(
        values=values, mask=mask, weights=oracle_weights
    )
    sample_mean = _weighted_expectation(values=values, mask=mask, weights=weights)
    usable = (
        np.any(mask, axis=1)
        & np.isfinite(targets)
        & np.isfinite(oracle_mean)
        & np.isfinite(sample_mean)
    )
    if moment_mu.shape[0] != targets.shape[0]:
        moment_mu = sample_mean.copy()
    if moment_sigma.shape[0] != targets.shape[0]:
        moment_sigma = np.ones(targets.shape[0], dtype=np.float64)
    if sigma_scale.shape[0] != targets.shape[0]:
        sigma_scale = np.ones(targets.shape[0], dtype=np.float64)
    if adapter_ids.shape[0] != targets.shape[0]:
        adapter_ids = np.zeros(targets.shape[0], dtype=np.float64)
    if outlier_scores.shape[0] != targets.shape[0]:
        outlier_scores = np.zeros(targets.shape[0], dtype=np.float64)
    masked_prediction_rows = []
    tensor_channel = tensors["channels"].get("chemical_shifts")
    evidence_mask_tensor = chemical_shift_evidence_mask(
        tensor_channel,
        config=config,
        training=False,
        device=device,
        example_index=example_index,
    )
    if evidence_mask_tensor is not None:
        with _torch_no_grad():
            masked_outputs = model_outputs_for_example(
                model=model,
                tensors=tensors,
                evidence_mask=evidence_mask_tensor,
                config=config,
            )
        masked_mu = masked_outputs["moment_mu"].detach().cpu().numpy().astype(
            np.float64
        )
        masked_sigma = _optional_output_vector(
            masked_outputs, "moment_sigma", len(masked_mu)
        )
        if masked_mu.shape[0] != targets.shape[0]:
            masked_mu = sample_mean.copy()
        if masked_sigma.shape[0] != targets.shape[0]:
            masked_sigma = np.ones(targets.shape[0], dtype=np.float64)
        holdout_mask = ~evidence_mask_tensor.detach().cpu().numpy().astype(bool)
        for row_index, target_id in enumerate(channel.target_ids):
            if not usable[row_index] or not holdout_mask[row_index]:
                continue
            masked_prediction_rows.append(
                {
                    "entity_uid": example.entity_uid,
                    "bmrb_id": example.metadata.get("bmrb_id", ""),
                    "split": split_name,
                    "target_id": target_id,
                    "atom_family": _atom_family_from_target_id(target_id),
                    "target_value": float(targets[row_index]),
                    "moment_predicted_value": float(masked_mu[row_index]),
                    "moment_sigma": float(masked_sigma[row_index]),
                    "sample_mean_value": float(sample_mean[row_index]),
                    "oracle_mean_value": float(oracle_mean[row_index]),
                    "is_masked_holdout": True,
                }
            )
    prediction_rows = []
    for row_index, target_id in enumerate(channel.target_ids):
        if not usable[row_index]:
            continue
        prediction_rows.append(
            {
                "entity_uid": example.entity_uid,
                "bmrb_id": example.metadata.get("bmrb_id", ""),
                "split": split_name,
                "target_id": target_id,
                "atom_family": _atom_family_from_target_id(target_id),
                "target_value": float(targets[row_index]),
                "moment_predicted_value": float(moment_mu[row_index]),
                "moment_sigma": float(moment_sigma[row_index]),
                "atom_family_sigma_scale": float(sigma_scale[row_index]),
                "family_adapter_id": int(round(float(adapter_ids[row_index]))),
                "outlier_score": float(outlier_scores[row_index]),
                "sample_mean_value": float(sample_mean[row_index]),
                "oracle_mean_value": float(oracle_mean[row_index]),
                "moment_sample_abs_error": float(
                    abs(moment_mu[row_index] - sample_mean[row_index])
                ),
                "moment_oracle_abs_error": float(
                    abs(moment_mu[row_index] - oracle_mean[row_index])
                ),
            }
        )
    weight_rows = [
        {
            "entity_uid": example.entity_uid,
            "bmrb_id": example.metadata.get("bmrb_id", ""),
            "split": split_name,
            "candidate_index": int(index),
            "candidate_id": (
                example.candidate_ids[index]
                if index < len(example.candidate_ids)
                else str(index)
            ),
            "model_weight": float(weights[index]),
            "oracle_weight": float(oracle_weights[index]),
        }
        for index in range(len(oracle_weights))
    ]
    state_rows = _state_rows(
        outputs=outputs,
        oracle_weights=oracle_weights,
        example=example,
        split_name=split_name,
    )
    latent_rows = _latent_rows(outputs=outputs, example=example, split_name=split_name)
    feature_rows = _nmr_structural_feature_rows(
        example=example,
        channel=channel,
        split_name=split_name,
    )
    summary_row = {
        "entity_uid": example.entity_uid,
        "bmrb_id": example.metadata.get("bmrb_id", ""),
        "split": split_name,
        "oracle_family_ccc": _family_macro_ccc(
            oracle_mean, targets, channel.target_ids, usable
        ),
        "moment_family_ccc": _family_macro_ccc(
            moment_mu, targets, channel.target_ids, usable
        ),
        "sample_family_ccc": _family_macro_ccc(
            sample_mean, targets, channel.target_ids, usable
        ),
        "moment_sample_mae": _safe_mean_abs(moment_mu, sample_mean, usable),
        "moment_oracle_mae": _safe_mean_abs(moment_mu, oracle_mean, usable),
        "support_ceiling_gap": (
            _family_macro_ccc(oracle_mean, targets, channel.target_ids, usable)
            - _family_macro_ccc(moment_mu, targets, channel.target_ids, usable)
        ),
    }
    return {
        "prediction_rows": prediction_rows,
        "weight_rows": weight_rows,
        "state_rows": state_rows,
        "latent_rows": latent_rows,
        "feature_rows": feature_rows,
        "masked_prediction_rows": masked_prediction_rows,
        "summary_row": summary_row,
    }


def _empty_payload() -> dict[str, Any]:
    """Return an empty example artifact payload."""
    return {
        "prediction_rows": [],
        "weight_rows": [],
        "state_rows": [],
        "latent_rows": [],
        "feature_rows": [],
        "masked_prediction_rows": [],
        "summary_row": {},
    }


def _optional_output_vector(
    outputs: dict[str, Any],
    key: str,
    length: int,
) -> np.ndarray:
    """Return one optional model output vector as a finite numpy array."""
    value = outputs.get(key)
    if value is None or getattr(value, "numel", lambda: 0)() == 0:
        default = 1.0 if "sigma" in key or "scale" in key else 0.0
        return np.full(length, default, dtype=np.float64)
    array = value.detach().cpu().numpy().astype(np.float64)
    if array.shape[0] != length:
        default = 1.0 if "sigma" in key or "scale" in key else 0.0
        return np.full(length, default, dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _nmr_structural_feature_rows(
    *,
    example: PreparedTeacherExample,
    channel: Any,
    split_name: str,
) -> list[dict[str, Any]]:
    """Return stable target-level NMR structural proxy feature rows."""
    sequence_length = max(len(example.sequence_tokens), 1)
    rows = []
    for target_id in channel.target_ids:
        parsed = _parse_target_id(target_id)
        residue_index = int(parsed["residue_index"])
        residue_name = str(parsed["residue_name"])
        atom_family = _atom_family_from_target_id(target_id)
        normalized_position = float(residue_index) / float(sequence_length)
        rows.append(
            {
                "entity_uid": example.entity_uid,
                "bmrb_id": example.metadata.get("bmrb_id", ""),
                "split": split_name,
                "target_id": target_id,
                "chain_id": parsed["chain_id"],
                "residue_index": residue_index,
                "residue_name": residue_name,
                "atom_family": atom_family,
                "normalized_position": normalized_position,
                "is_n_terminal": bool(normalized_position <= 0.05),
                "is_c_terminal": bool(normalized_position >= 0.95),
                "is_gly": residue_name == "GLY",
                "is_pro": residue_name == "PRO",
                "is_hn_sensitive": atom_family == "HN",
                "is_cprime_sensitive": atom_family == "C'",
                "local_ca_density": 0.0,
                "contact_degree": 0.0,
                "compactness": 0.0,
                "rci_flexibility": None,
                "feature_available_mask": "target_metadata",
            }
        )
    return rows


def _parse_target_id(target_id: str) -> dict[str, Any]:
    """Parse a chemical-shift target id into stable feature fields."""
    parts = str(target_id).split(":")
    if len(parts) != 5 or parts[0] != "cs":
        return {
            "chain_id": "_",
            "residue_index": 0,
            "residue_name": "",
        }
    try:
        residue_index = int(parts[2])
    except ValueError:
        residue_index = 0
    return {
        "chain_id": parts[1],
        "residue_index": max(residue_index, 0),
        "residue_name": parts[3].upper(),
    }


def _projected_simplex_oracle(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    steps: int = 240,
) -> np.ndarray:
    """Return a robust simplex-constrained least-squares oracle."""
    candidate_count = int(values.shape[1]) if values.ndim == 2 else 0
    if candidate_count == 0:
        return np.zeros(0, dtype=np.float64)
    usable = np.any(mask, axis=1) & np.isfinite(targets)
    if np.count_nonzero(usable) < 2:
        return np.ones(candidate_count, dtype=np.float64) / candidate_count
    matrix = np.where(mask[usable], values[usable], 0.0)
    row_counts = np.clip(mask[usable].sum(axis=1, keepdims=True), 1.0, None)
    row_means = matrix.sum(axis=1, keepdims=True) / row_counts
    matrix = np.where(mask[usable], values[usable], row_means)
    target = targets[usable]
    weights = np.ones(candidate_count, dtype=np.float64) / candidate_count
    spectral = np.linalg.norm(matrix, ord=2)
    step_size = 1.0 / max((spectral * spectral) / max(matrix.shape[0], 1), 1e-6)
    for _ in range(max(int(steps), 1)):
        residual = matrix @ weights - target
        gradient = matrix.T @ residual / max(matrix.shape[0], 1)
        weights = _project_to_simplex(weights - step_size * gradient)
    return weights


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""
    if values.size == 0:
        return values
    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, values.size + 1)
    positive = sorted_values - cssv / indices > 0
    if not np.any(positive):
        return np.ones_like(values) / values.size
    rho = indices[positive][-1]
    theta = cssv[positive][-1] / float(rho)
    projected = np.maximum(values - theta, 0.0)
    return projected / np.clip(projected.sum(), 1e-12, None)


def _weighted_expectation(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return per-target weighted expectation with missing-value handling."""
    denominator = (mask.astype(np.float64) * weights[None, :]).sum(axis=1)
    numerator = (mask.astype(np.float64) * values * weights[None, :]).sum(axis=1)
    output = numerator / np.clip(denominator, 1e-12, None)
    output[denominator <= 1e-12] = np.nan
    return output


def _state_rows(
    *,
    outputs: dict[str, Any],
    oracle_weights: np.ndarray,
    example: PreparedTeacherExample,
    split_name: str,
) -> list[dict[str, Any]]:
    """Return state occupancy rows from posterior state token outputs."""
    state_weights = outputs.get("posterior_state_weights")
    assignments = outputs.get("posterior_state_assignments")
    if state_weights is None or assignments is None or state_weights.numel() == 0:
        return []
    state_np = state_weights.detach().cpu().numpy().astype(np.float64)
    assignment_np = assignments.detach().cpu().numpy().astype(np.float64)
    oracle_state = assignment_np.T @ oracle_weights
    if oracle_state.sum() > 0:
        oracle_state = oracle_state / oracle_state.sum()
    max_mass = float(np.max(state_np)) if state_np.size else 1.0
    entropy = _normalized_entropy(state_np)
    effective_state_count = float(
        1.0 / np.clip(np.sum(np.square(state_np)), 1e-12, None)
    )
    repulsion_score = _state_repulsion_score(outputs.get("posterior_state_tokens"))
    rows = []
    for index, mass in enumerate(state_np):
        rows.append(
            {
                "entity_uid": example.entity_uid,
                "bmrb_id": example.metadata.get("bmrb_id", ""),
                "split": split_name,
                "state_id": f"posterior_state_{index + 1:02d}",
                "state_rank": int(index + 1),
                "state_mass": float(mass),
                "oracle_state_mass": float(oracle_state[index]),
                "relative_free_energy_kbt": float(
                    -math.log(max(float(mass), 1e-12) / max(max_mass, 1e-12))
                ),
                "state_mass_gap": float(mass - oracle_state[index]),
                "state_entropy": entropy,
                "effective_state_count": effective_state_count,
                "repulsion_score": repulsion_score,
            }
        )
    return rows


def _normalized_entropy(values: np.ndarray) -> float:
    """Return normalized entropy for one nonnegative vector."""
    if values.size <= 1:
        return 0.0
    probabilities = values.astype(np.float64, copy=False)
    probabilities = probabilities / np.clip(probabilities.sum(), 1e-12, None)
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(float(values.size))


def _state_repulsion_score(tokens: Any) -> float | None:
    """Return mean positive off-diagonal cosine score for state tokens."""
    if tokens is None or getattr(tokens, "numel", lambda: 0)() == 0:
        return None
    array = tokens.detach().cpu().numpy().astype(np.float64)
    if array.ndim != 2 or array.shape[0] <= 1:
        return None
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    return float(np.mean(np.square(np.maximum(similarity[mask], 0.0))))


def _latent_rows(
    *,
    outputs: dict[str, Any],
    example: PreparedTeacherExample,
    split_name: str,
) -> list[dict[str, Any]]:
    """Return compact latent sample rows for monitor diagnostics."""
    samples = outputs.get("posterior_latent_samples")
    if samples is None or samples.numel() == 0:
        return []
    sample_np = samples.detach().cpu().numpy().astype(np.float64)
    rows = []
    for state_index, vector in enumerate(sample_np):
        row = {
            "entity_uid": example.entity_uid,
            "bmrb_id": example.metadata.get("bmrb_id", ""),
            "split": split_name,
            "state_id": f"posterior_state_{state_index + 1:02d}",
            "sample_index": int(state_index),
            "latent_norm": float(np.linalg.norm(vector)),
        }
        for dim_index, value in enumerate(vector[:8]):
            row[f"latent_dim_{dim_index + 1}"] = float(value)
        rows.append(row)
    return rows


def _family_macro_ccc(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_ids: list[str],
    usable: np.ndarray,
) -> float:
    """Return atom-family macro CCC for one prediction vector."""
    values = []
    for family in CHEMICAL_SHIFT_FAMILIES:
        family_mask = np.asarray(
            [
                _atom_family_from_target_id(target_id) == family
                for target_id in target_ids
            ],
            dtype=bool,
        )
        combined = usable & family_mask & np.isfinite(predictions)
        if np.count_nonzero(combined) < 2:
            continue
        ccc = _safe_ccc(predictions[combined], targets[combined])
        if math.isfinite(ccc):
            values.append(ccc)
    return float(np.mean(values)) if values else float("nan")


def _safe_ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's CCC with safe finite handling."""
    if predictions.size < 2:
        return float("nan")
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    covariance = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    if denominator <= 1e-12:
        return float("nan")
    return float((2.0 * covariance) / denominator)


def _safe_mean_abs(left: np.ndarray, right: np.ndarray, usable: np.ndarray) -> float:
    """Return finite mean absolute difference for two vectors."""
    combined = usable & np.isfinite(left) & np.isfinite(right)
    if not np.any(combined):
        return float("nan")
    return float(np.mean(np.abs(left[combined] - right[combined])))


def _summarize_oracle(
    frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Return split-level oracle and support-ceiling summary."""
    if frame.empty:
        return {"status": "empty"}
    rows: dict[str, Any] = {"status": "ok", "splits": {}}
    for split, split_frame in frame.groupby("split", dropna=False):
        split_key = str(split)
        rows["splits"][split_key] = {
            "examples": int(len(split_frame)),
            "oracle_family_ccc_macro": _json_float(
                split_frame["oracle_family_ccc"].mean()
            ),
            "moment_family_ccc_macro": _json_float(
                split_frame["moment_family_ccc"].mean()
            ),
            "sample_family_ccc_macro": _json_float(
                split_frame["sample_family_ccc"].mean()
            ),
            "support_gap_macro": _json_float(split_frame["support_ceiling_gap"].mean()),
        }
        rows["splits"][split_key]["family_metrics"] = _family_report(
            prediction_frame.loc[prediction_frame["split"].astype(str) == split_key]
            if not prediction_frame.empty and "split" in prediction_frame.columns
            else pd.DataFrame()
        )
    val = rows["splits"].get("val", {})
    oracle = val.get("oracle_family_ccc_macro")
    rows["ccc_095_status"] = (
        "support_below_target"
        if oracle is not None and float(oracle) < 0.95
        else "support_can_reach_target"
    )
    return rows


def _family_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Return atom-family split metrics for moment, sample, and oracle means."""
    if frame.empty:
        return {}
    report = {}
    for family, family_frame in frame.groupby("atom_family", dropna=False):
        if len(family_frame) < 2:
            continue
        targets = family_frame["target_value"].to_numpy(dtype=np.float64)
        report[str(family)] = {
            "rows": int(len(family_frame)),
            "moment_ccc": _json_float(
                _safe_ccc(
                    family_frame["moment_predicted_value"].to_numpy(dtype=np.float64),
                    targets,
                )
            ),
            "sample_ccc": _json_float(
                _safe_ccc(
                    family_frame["sample_mean_value"].to_numpy(dtype=np.float64),
                    targets,
                )
            ),
            "simplex_ls_oracle_ccc": _json_float(
                _safe_ccc(
                    family_frame["oracle_mean_value"].to_numpy(dtype=np.float64),
                    targets,
                )
            ),
            "moment_minus_sample_mae": _json_float(
                family_frame["moment_sample_abs_error"].mean()
            ),
            "moment_minus_oracle_mae": _json_float(
                family_frame["moment_oracle_abs_error"].mean()
            ),
            "mean_sigma": _json_float(
                family_frame.get("moment_sigma", pd.Series(dtype=float)).mean()
            ),
            "mean_outlier_score": _json_float(
                family_frame.get("outlier_score", pd.Series(dtype=float)).mean()
            ),
        }
    return report


def _summarize_consistency(frame: pd.DataFrame) -> dict[str, Any]:
    """Return moment/sample consistency metrics."""
    if frame.empty:
        return {"status": "empty"}
    rows: dict[str, Any] = {"status": "ok", "splits": {}}
    for split, split_frame in frame.groupby("split", dropna=False):
        rows["splits"][str(split)] = {
            "rows": int(len(split_frame)),
            "moment_sample_mae": _json_float(
                split_frame["moment_sample_abs_error"].mean()
            ),
            "moment_oracle_mae": _json_float(
                split_frame["moment_oracle_abs_error"].mean()
            ),
        }
    return rows


def _build_measure_aware_targets(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return calibrated chemical-shift rows and entry/family report."""
    if frame.empty:
        return pd.DataFrame(), {"status": "empty"}
    rows = frame.copy()
    parsed = rows["target_id"].map(_parse_target_id)
    rows["chain_id"] = [item["chain_id"] for item in parsed]
    rows["residue_index"] = [int(item["residue_index"]) for item in parsed]
    rows["residue_name"] = [item["residue_name"] for item in parsed]
    rows["random_coil_reference"] = [
        random_coil_reference_for_target_id(str(target_id))
        for target_id in rows["target_id"]
    ]
    rows["secondary_target_value"] = (
        rows["target_value"] - rows["random_coil_reference"]
    )
    rows["secondary_moment_predicted_value"] = (
        rows["moment_predicted_value"] - rows["random_coil_reference"]
    )
    rows["entry_family_offset"] = 0.0
    rows["entry_family_scale"] = 1.0
    rows["entry_family_reliability"] = 1.0
    rows["calibrated_moment_predicted_value"] = rows["moment_predicted_value"]
    report_rows: list[dict[str, Any]] = []
    group_columns = ["split", "entity_uid", "atom_family"]
    for group_key, group in rows.groupby(group_columns, dropna=False):
        split_name, entity_uid, family = group_key
        indices = group.index
        predictions = group["moment_predicted_value"].to_numpy(dtype=np.float64)
        targets = group["target_value"].to_numpy(dtype=np.float64)
        finite = np.isfinite(predictions) & np.isfinite(targets)
        if np.count_nonzero(finite) >= 3:
            slope, offset = _fit_affine(predictions[finite], targets[finite])
        else:
            slope, offset = 1.0, 0.0
        calibrated = slope * predictions + offset
        residual = calibrated - targets
        reliability = _entry_family_reliability(
            family=str(family),
            residue_index=group["residue_index"].to_numpy(dtype=np.float64),
            residue_name=group["residue_name"].astype(str).to_numpy(),
            residual=residual,
        )
        rows.loc[indices, "entry_family_offset"] = float(offset)
        rows.loc[indices, "entry_family_scale"] = float(slope)
        rows.loc[indices, "entry_family_reliability"] = reliability
        rows.loc[indices, "calibrated_moment_predicted_value"] = calibrated
        report_rows.append(
            {
                "split": str(split_name),
                "entity_uid": str(entity_uid),
                "atom_family": str(family),
                "rows": int(len(group)),
                "offset": _json_float(offset),
                "scale": _json_float(slope),
                "mean_reliability": _json_float(np.nanmean(reliability)),
                "raw_ccc": _json_float(_safe_ccc(predictions[finite], targets[finite])),
                "calibrated_ccc": _json_float(
                    _safe_ccc(calibrated[finite], targets[finite])
                ),
            }
        )
    rows["calibrated_residual"] = (
        rows["calibrated_moment_predicted_value"] - rows["target_value"]
    )
    report_frame = pd.DataFrame(report_rows)
    report: dict[str, Any] = {
        "status": "ok",
        "rows": int(len(rows)),
        "groups": report_rows[:200],
        "splits": {},
    }
    if not report_frame.empty:
        for split_name, split_frame in report_frame.groupby("split", dropna=False):
            report["splits"][str(split_name)] = {
                "groups": int(len(split_frame)),
                "raw_ccc_mean": _json_float(split_frame["raw_ccc"].mean()),
                "calibrated_ccc_mean": _json_float(
                    split_frame["calibrated_ccc"].mean()
                ),
                "mean_reliability": _json_float(
                    split_frame["mean_reliability"].mean()
                ),
            }
    return rows, report


def _build_joint_nmr_posteriors(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return residue-level diagonal Gaussian joint posterior rows."""
    if frame.empty:
        return pd.DataFrame(), {"status": "empty"}
    atom_sets = {
        "HN_N": ("HN", "N"),
        "CA_CB": ("CA", "CB"),
        "CA_C": ("CA", "C'"),
        "HN_N_CA": ("HN", "N", "CA"),
        "CA_CB_C": ("CA", "CB", "C'"),
    }
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    parsed = work["target_id"].map(_parse_target_id)
    work["chain_id"] = [item["chain_id"] for item in parsed]
    work["residue_index"] = [int(item["residue_index"]) for item in parsed]
    work["residue_name"] = [item["residue_name"] for item in parsed]
    group_columns = ["split", "entity_uid", "chain_id", "residue_index", "residue_name"]
    for group_key, group in work.groupby(group_columns, dropna=False):
        split_name, entity_uid, chain_id, residue_index, residue_name = group_key
        by_family = {
            str(row.atom_family): row
            for row in group.itertuples(index=False)
            if pd.notna(row.atom_family)
        }
        for atom_set_name, atom_set in atom_sets.items():
            if not all(atom in by_family for atom in atom_set):
                continue
            means = np.asarray(
                [float(by_family[atom].moment_predicted_value) for atom in atom_set],
                dtype=np.float64,
            )
            targets = np.asarray(
                [float(by_family[atom].target_value) for atom in atom_set],
                dtype=np.float64,
            )
            sigmas = np.asarray(
                [
                    max(float(getattr(by_family[atom], "moment_sigma", 1.0)), 1e-3)
                    for atom in atom_set
                ],
                dtype=np.float64,
            )
            finite = np.isfinite(means) & np.isfinite(targets) & np.isfinite(sigmas)
            if not np.all(finite):
                continue
            residual = (means - targets) / sigmas
            covariance = np.diag(np.square(sigmas))
            rows.append(
                {
                    "entity_uid": str(entity_uid),
                    "split": str(split_name),
                    "chain_id": str(chain_id),
                    "residue_index": int(residue_index),
                    "residue_name": str(residue_name),
                    "atom_set": atom_set_name,
                    "atom_families_json": json.dumps(list(atom_set)),
                    "posterior_mean_json": json.dumps(means.tolist()),
                    "target_vector_json": json.dumps(targets.tolist()),
                    "marginal_std_json": json.dumps(sigmas.tolist()),
                    "covariance_json": json.dumps(covariance.tolist()),
                    "mahalanobis_distance": _json_float(
                        math.sqrt(float(np.sum(np.square(residual))))
                    ),
                    "joint_nll": _json_float(
                        0.5 * float(np.sum(np.square(residual)))
                        + float(np.sum(np.log(sigmas)))
                    ),
                    "coverage_50": bool(np.all(np.abs(residual) <= 0.674)),
                    "coverage_80": bool(np.all(np.abs(residual) <= 1.282)),
                    "coverage_95": bool(np.all(np.abs(residual) <= 1.960)),
                }
            )
    joint_frame = pd.DataFrame(rows)
    report = {
        "status": "ok" if rows else "no_complete_atom_sets",
        "rows": int(len(joint_frame)),
        "atom_sets": {},
    }
    if not joint_frame.empty:
        for atom_set, group in joint_frame.groupby("atom_set", dropna=False):
            report["atom_sets"][str(atom_set)] = {
                "rows": int(len(group)),
                "mean_mahalanobis": _json_float(group["mahalanobis_distance"].mean()),
                "mean_joint_nll": _json_float(group["joint_nll"].mean()),
                "coverage_95": _json_float(group["coverage_95"].mean()),
            }
    return joint_frame, report


def _build_ccc_decomposition_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Return split/family CCC decomposition for raw and calibrated predictions."""
    if frame.empty:
        return {"status": "empty"}
    prediction_columns = {
        "raw": "moment_predicted_value",
        "calibrated": "calibrated_moment_predicted_value",
    }
    report: dict[str, Any] = {"status": "ok", "splits": {}}
    for split_name, split_frame in frame.groupby("split", dropna=False):
        split_report = {"families": {}}
        for family, family_frame in split_frame.groupby("atom_family", dropna=False):
            target = family_frame["target_value"].to_numpy(dtype=np.float64)
            family_report: dict[str, Any] = {"rows": int(len(family_frame))}
            for label, column in prediction_columns.items():
                if column not in family_frame.columns:
                    continue
                pred = family_frame[column].to_numpy(dtype=np.float64)
                finite = np.isfinite(pred) & np.isfinite(target)
                if np.count_nonzero(finite) < 2:
                    continue
                family_report[label] = _ccc_decomposition(pred[finite], target[finite])
            split_report["families"][str(family)] = family_report
        report["splits"][str(split_name)] = split_report
    return report


def _fit_affine(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Fit ``target ~= scale * prediction + offset`` robustly enough for reports."""
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    denominator = float(np.sum(np.square(predictions - pred_mean)))
    if denominator <= 1e-8:
        return 1.0, target_mean - pred_mean
    scale = float(np.sum((predictions - pred_mean) * (targets - target_mean)))
    scale /= denominator
    return scale, target_mean - scale * pred_mean


def _entry_family_reliability(
    *,
    family: str,
    residue_index: np.ndarray,
    residue_name: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Return report-time reliability weights from simple NMR context proxies."""
    reliability = np.ones_like(residual, dtype=np.float64)
    if family == "HN":
        reliability = np.where(residue_index <= 2, reliability * 0.75, reliability)
        reliability = np.where(residue_name == "PRO", reliability * 0.60, reliability)
    if family == "C'":
        scale = float(np.nanstd(residual))
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        reliability *= 1.0 / (1.0 + np.abs(residual / scale) / 4.0)
    return np.clip(reliability, 0.05, 1.0)


def _ccc_decomposition(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Return CCC plus its correlation, scale, and bias ingredients."""
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_std = float(np.std(predictions))
    target_std = float(np.std(targets))
    if pred_std <= 1e-12 or target_std <= 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "ccc": _json_float(_safe_ccc(predictions, targets)),
        "corr": _json_float(corr),
        "scale_ratio": _json_float(pred_std / target_std if target_std > 0 else None),
        "bias_ppm": _json_float(pred_mean - target_mean),
        "pred_std": _json_float(pred_std),
        "target_std": _json_float(target_std),
    }


def _render_measure_aware_figures(
    *,
    measure_frame: pd.DataFrame,
    joint_frame: pd.DataFrame,
    masked_prediction_frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Render v3 measure-aware static diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not measure_frame.empty:
        frame = measure_frame.loc[measure_frame["split"].astype(str) == "val"]
        if frame.empty:
            frame = measure_frame
        _render_ccc_decomposition_by_family(frame, figures_dir, plt)
        _render_entry_family_calibration_heatmap(frame, figures_dir, plt)
        _render_hn_context_audit(frame, figures_dir, plt)
        _render_cprime_residual_trend_audit(frame, figures_dir, plt)
        _render_pair_evidence_audit(
            frame=frame,
            target_family="HN",
            source_family="N",
            output_path=figures_dir / "hn_n_pair_evidence_audit.png",
            plt=plt,
        )
        _render_pair_evidence_audit(
            frame=frame,
            target_family="C'",
            source_family="CA",
            output_path=figures_dir / "cprime_ca_pair_evidence_audit.png",
            plt=plt,
        )
    if not masked_prediction_frame.empty:
        frame = masked_prediction_frame.loc[
            masked_prediction_frame["split"].astype(str) == "val"
        ]
        if frame.empty:
            frame = masked_prediction_frame
        _render_masked_holdout_scatter(
            frame=frame,
            family="HN",
            output_path=figures_dir / "hn_masked_holdout_scatter.png",
            plt=plt,
        )
        _render_masked_holdout_scatter(
            frame=frame,
            family="C'",
            output_path=figures_dir / "cprime_masked_holdout_scatter.png",
            plt=plt,
        )
    if not joint_frame.empty:
        frame = joint_frame.loc[joint_frame["split"].astype(str) == "val"]
        if frame.empty:
            frame = joint_frame
        _render_joint_density(
            frame=frame,
            atom_set="HN_N",
            output_path=figures_dir / "joint_hn_n_density.png",
            plt=plt,
        )
        _render_joint_density(
            frame=frame,
            atom_set="CA_C",
            output_path=figures_dir / "joint_ca_c_density.png",
            plt=plt,
        )


def _render_ccc_decomposition_by_family(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render raw versus calibrated CCC by atom family."""
    rows = []
    for family, group in frame.groupby("atom_family", dropna=False):
        if len(group) < 2:
            continue
        target = group["target_value"].to_numpy(dtype=np.float64)
        for label, column in [
            ("raw", "moment_predicted_value"),
            ("calibrated", "calibrated_moment_predicted_value"),
        ]:
            pred = group[column].to_numpy(dtype=np.float64)
            finite = np.isfinite(pred) & np.isfinite(target)
            if np.count_nonzero(finite) < 2:
                continue
            rows.append(
                {
                    "family": str(family),
                    "metric": label,
                    "ccc": _safe_ccc(pred[finite], target[finite]),
                }
            )
    if not rows:
        return
    plot_frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x_positions = np.arange(len(CHEMICAL_SHIFT_FAMILIES))
    width = 0.35
    for offset, label in [(-width / 2, "raw"), (width / 2, "calibrated")]:
        values = []
        for family in CHEMICAL_SHIFT_FAMILIES:
            subset = plot_frame.loc[
                (plot_frame["family"] == family) & (plot_frame["metric"] == label)
            ]
            values.append(float(subset["ccc"].iloc[0]) if not subset.empty else np.nan)
        axis.bar(x_positions + offset, values, width=width, label=label)
    axis.axhline(0.816506, color="#756bb1", linestyle="--", label="fgate baseline")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(CHEMICAL_SHIFT_FAMILIES)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("CCC")
    axis.set_title("CCC decomposition by atom family")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures_dir / "ccc_decomposition_by_family.png", dpi=200)
    plt.close(figure)


def _render_masked_holdout_scatter(
    *,
    frame: pd.DataFrame,
    family: str,
    output_path: Path,
    plt: Any,
) -> None:
    """Render prediction-vs-target scatter for deterministic masked holdout rows."""
    if frame.empty or "atom_family" not in frame.columns:
        return
    subset = frame.loc[frame["atom_family"].astype(str) == family].copy()
    if len(subset) < 2:
        return
    target = subset["target_value"].to_numpy(dtype=np.float64)
    pred = subset["moment_predicted_value"].to_numpy(dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(pred)
    if np.count_nonzero(finite) < 2:
        return
    target = target[finite]
    pred = pred[finite]
    ccc = _safe_ccc(pred, target)
    mae = float(np.mean(np.abs(pred - target)))
    low = float(min(np.min(target), np.min(pred)))
    high = float(max(np.max(target), np.max(pred)))
    padding = 0.05 * max(high - low, 1e-6)
    low -= padding
    high += padding
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(target, pred, s=10, alpha=0.38, color="#2b8cbe", label="masked rows")
    axis.plot([low, high], [low, high], color="#252525", linewidth=1.0, label="ideal")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel(f"Experimental {family} chemical shift (ppm)")
    axis.set_ylabel(f"Predicted masked {family} posterior mean (ppm)")
    axis.set_title(f"Masked-holdout {family}: CCC={ccc:.3f}, MAE={mae:.3f} ppm")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _render_entry_family_calibration_heatmap(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render entry/family calibration scale heatmap for the most populated entries."""
    required = {"entity_uid", "atom_family", "entry_family_scale"}
    if frame.empty or not required.issubset(frame.columns):
        return
    grouped = (
        frame.groupby(["entity_uid", "atom_family"], dropna=False)[
            "entry_family_scale"
        ]
        .mean()
        .reset_index()
    )
    top_entities = (
        frame["entity_uid"].value_counts().head(20).index.astype(str).tolist()
    )
    matrix = np.full((len(top_entities), len(CHEMICAL_SHIFT_FAMILIES)), np.nan)
    for row_index, entity_uid in enumerate(top_entities):
        for col_index, family in enumerate(CHEMICAL_SHIFT_FAMILIES):
            subset = grouped.loc[
                (grouped["entity_uid"].astype(str) == entity_uid)
                & (grouped["atom_family"].astype(str) == family)
            ]
            if not subset.empty:
                matrix[row_index, col_index] = float(
                    subset["entry_family_scale"].iloc[0]
                )
    if matrix.size == 0:
        return
    figure, axis = plt.subplots(figsize=(7.4, max(4.0, 0.28 * len(top_entities))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.5)
    axis.set_xticks(np.arange(len(CHEMICAL_SHIFT_FAMILIES)))
    axis.set_xticklabels(CHEMICAL_SHIFT_FAMILIES)
    axis.set_yticks(np.arange(len(top_entities)))
    axis.set_yticklabels(top_entities)
    axis.set_title("Entry/family calibration scale")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("affine scale")
    figure.tight_layout()
    figure.savefig(figures_dir / "entry_family_calibration_heatmap.png", dpi=200)
    plt.close(figure)


def _render_hn_context_audit(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render HN reliability and residual context diagnostics."""
    hn_frame = frame.loc[frame["atom_family"].astype(str) == "HN"].copy()
    required = {"residue_index", "calibrated_residual", "entry_family_reliability"}
    if hn_frame.empty or not required.issubset(hn_frame.columns):
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    scatter = axis.scatter(
        hn_frame["residue_index"],
        hn_frame["calibrated_residual"],
        c=hn_frame["entry_family_reliability"],
        cmap="viridis",
        s=16,
        alpha=0.75,
        label="HN rows",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, label="zero residual")
    axis.set_title("HN context audit")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Calibrated residual")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("measure-aware reliability")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures_dir / "hn_context_audit.png", dpi=200)
    plt.close(figure)


def _render_cprime_residual_trend_audit(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render C' residual trend after entry/family calibration."""
    cprime_frame = frame.loc[frame["atom_family"].astype(str) == "C'"].copy()
    required = {"target_value", "calibrated_residual", "entry_family_reliability"}
    if cprime_frame.empty or not required.issubset(cprime_frame.columns):
        return
    x_values = cprime_frame["target_value"].to_numpy(dtype=np.float64)
    y_values = cprime_frame["calibrated_residual"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    scatter = axis.scatter(
        x_values[finite],
        y_values[finite],
        c=cprime_frame["entry_family_reliability"].to_numpy(dtype=np.float64)[finite],
        cmap="magma",
        s=18,
        alpha=0.75,
        label="C' rows",
    )
    if np.count_nonzero(finite) >= 2:
        slope, intercept = np.polyfit(x_values[finite], y_values[finite], 1)
        grid = np.linspace(float(np.nanmin(x_values[finite])), float(np.nanmax(x_values[finite])), 100)
        axis.plot(grid, slope * grid + intercept, color="#d95f02", label="trend")
    axis.axhline(0.0, color="black", linewidth=1.0, label="zero residual")
    axis.set_title("C' residual trend audit")
    axis.set_xlabel("Experimental C' shift")
    axis.set_ylabel("Calibrated residual")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("measure-aware reliability")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures_dir / "cprime_residual_trend_audit.png", dpi=200)
    plt.close(figure)


def _render_joint_density(
    *,
    frame: pd.DataFrame,
    atom_set: str,
    output_path: Path,
    plt: Any,
) -> None:
    """Render one compact joint posterior residual diagnostic."""
    subset = frame.loc[frame["atom_set"].astype(str) == atom_set]
    if subset.empty:
        return
    figure, axis = plt.subplots(figsize=(6.0, 4.8))
    scatter = axis.scatter(
        subset["mahalanobis_distance"],
        subset["joint_nll"],
        c=subset["coverage_95"].astype(float),
        cmap="viridis",
        s=18,
        alpha=0.75,
        label=atom_set,
    )
    axis.set_title(f"Joint NMR posterior: {atom_set}")
    axis.set_xlabel("Mahalanobis distance")
    axis.set_ylabel("Diagonal Gaussian joint NLL")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("95% rectangular coverage")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _render_pair_evidence_audit(
    *,
    frame: pd.DataFrame,
    target_family: str,
    source_family: str,
    output_path: Path,
    plt: Any,
) -> None:
    """Render same-residue evidence pairing used by masked imputation."""
    required = {
        "entity_uid",
        "chain_id",
        "residue_index",
        "residue_name",
        "atom_family",
        "target_value",
        "moment_predicted_value",
    }
    if frame.empty or not required.issubset(frame.columns):
        return
    keys = ["entity_uid", "chain_id", "residue_index", "residue_name"]
    target = frame.loc[frame["atom_family"].astype(str) == target_family]
    source = frame.loc[frame["atom_family"].astype(str) == source_family]
    if target.empty or source.empty:
        return
    paired = target.merge(
        source[keys + ["target_value"]],
        on=keys,
        how="inner",
        suffixes=("_target", "_source"),
    )
    if paired.empty:
        return
    x_values = paired["target_value_source"].to_numpy(dtype=np.float64)
    y_values = paired["target_value_target"].to_numpy(dtype=np.float64)
    pred_values = paired["moment_predicted_value"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(pred_values)
    if np.count_nonzero(finite) < 3:
        return
    slope, intercept = _fit_affine(x_values[finite], y_values[finite])
    grid = np.linspace(float(np.nanmin(x_values[finite])), float(np.nanmax(x_values[finite])), 100)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(x_values[finite], y_values[finite], s=12, alpha=0.55)
    axes[0].plot(grid, slope * grid + intercept, color="#d95f02", label="evidence fit")
    axes[0].set_title(f"{target_family} evidence from {source_family}")
    axes[0].set_xlabel(f"Observed {source_family}")
    axes[0].set_ylabel(f"Observed {target_family}")
    axes[0].legend(loc="best")
    imputed = slope * x_values[finite] + intercept
    axes[1].scatter(imputed, pred_values[finite], s=12, alpha=0.55)
    low = float(np.nanmin([np.nanmin(imputed), np.nanmin(pred_values[finite])]))
    high = float(np.nanmax([np.nanmax(imputed), np.nanmax(pred_values[finite])]))
    axes[1].plot([low, high], [low, high], color="black", label="identity")
    axes[1].set_title("MomentHead vs paired-evidence imputation")
    axes[1].set_xlabel(f"Imputed {target_family}")
    axes[1].set_ylabel(f"Predicted {target_family}")
    axes[1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _render_moment_figures(
    *,
    prediction_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Render static moment consistency and posterior state figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not prediction_frame.empty:
        frame = prediction_frame.loc[prediction_frame["split"].astype(str) == "val"]
        if frame.empty:
            frame = prediction_frame
        _render_family_scatter_grid(
            frame=frame,
            x_column="target_value",
            y_column="moment_predicted_value",
            x_label="Experimental chemical shift",
            y_label="Fast MomentHead posterior mean",
            title="MomentHead vs experimental chemical shift",
            output_path=figures_dir / "moment_vs_target.png",
            plt=plt,
        )
        _render_family_scatter_grid(
            frame=frame,
            x_column="target_value",
            y_column="sample_mean_value",
            x_label="Experimental chemical shift",
            y_label="Candidate-sample weighted mean",
            title="Candidate-sample support vs experimental shift",
            output_path=figures_dir / "sample_vs_target_support.png",
            plt=plt,
        )
        _render_family_scatter_grid(
            frame=frame,
            x_column="sample_mean_value",
            y_column="moment_predicted_value",
            x_label="Candidate-sample weighted mean",
            y_label="Fast MomentHead posterior mean",
            title="MomentHead vs candidate-sample diagnostic",
            output_path=figures_dir / "moment_sample_consistency.png",
            plt=plt,
        )
        _render_hn_variance_calibration(frame, figures_dir, plt)
        _render_cprime_outlier_audit(frame, figures_dir, plt)
    if not state_frame.empty:
        frame = state_frame.loc[state_frame["split"].astype(str) == "val"]
        if frame.empty:
            frame = state_frame
        first_entity = str(frame["entity_uid"].iloc[0])
        frame = frame.loc[frame["entity_uid"].astype(str) == first_entity].copy()
        frame = frame.sort_values("state_rank")
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axis.bar(
            frame["state_id"],
            frame["oracle_state_mass"],
            alpha=0.6,
            label="Oracle state mass",
        )
        axis.plot(
            frame["state_id"],
            frame["state_mass"],
            marker="o",
            color="#d95f02",
            label="Moment posterior state mass",
        )
        axis.set_ylabel("State occupancy")
        axis.set_title("Prior vs posterior landscape state occupancy")
        axis.tick_params(axis="x", rotation=30)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(figures_dir / "prior_vs_posterior_landscape.png", dpi=200)
        plt.close(figure)
        _render_state_diversity_diagnostics(frame, figures_dir, plt)


def _render_family_scatter_grid(
    *,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    plt: Any,
) -> None:
    """Render one atom-family faceted chemical-shift scatter diagnostic."""
    required = {x_column, y_column, "atom_family"}
    if frame.empty or not required.issubset(frame.columns):
        return
    observed_families = set(frame["atom_family"].astype(str))
    families = [
        family for family in CHEMICAL_SHIFT_FAMILIES if family in observed_families
    ]
    if not families:
        families = sorted(
            str(value) for value in frame["atom_family"].dropna().unique()
        )
    if not families:
        return
    column_count = min(3, len(families))
    row_count = int(math.ceil(len(families) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.2 * column_count, 3.8 * row_count),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_visible(False)
    for index, family in enumerate(families):
        axis = axes.flat[index]
        axis.set_visible(True)
        subset = frame.loc[frame["atom_family"].astype(str) == family]
        x_values = subset[x_column].to_numpy(dtype=np.float64)
        y_values = subset[y_column].to_numpy(dtype=np.float64)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite]
        y_values = y_values[finite]
        if x_values.size == 0:
            continue
        axis.scatter(x_values, y_values, s=7, alpha=0.45, label="chemical-shift rows")
        low = float(np.nanmin([np.nanmin(x_values), np.nanmin(y_values)]))
        high = float(np.nanmax([np.nanmax(x_values), np.nanmax(y_values)]))
        if math.isfinite(low) and math.isfinite(high):
            axis.plot(
                [low, high],
                [low, high],
                color="black",
                linewidth=1.0,
                label="identity",
            )
        axis.set_title(str(family))
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(loc="best", fontsize="small")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _render_hn_variance_calibration(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render HN moment variance calibration diagnostics."""
    if "atom_family" not in frame.columns or "HN" not in set(frame["atom_family"]):
        return
    hn_frame = frame.loc[frame["atom_family"].astype(str) == "HN"].copy()
    required = {"target_value", "moment_predicted_value", "moment_sigma"}
    if len(hn_frame) < 2 or not required.issubset(hn_frame.columns):
        return
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    axis.errorbar(
        hn_frame["target_value"],
        hn_frame["moment_predicted_value"],
        yerr=hn_frame["moment_sigma"].clip(lower=0.0),
        fmt="o",
        markersize=3.0,
        alpha=0.55,
        ecolor="#6baed6",
        color="#08519c",
        label="HN target rows",
    )
    low = float(
        np.nanmin(
            [hn_frame["target_value"].min(), hn_frame["moment_predicted_value"].min()]
        )
    )
    high = float(
        np.nanmax(
            [hn_frame["target_value"].max(), hn_frame["moment_predicted_value"].max()]
        )
    )
    axis.plot([low, high], [low, high], color="black", linewidth=1.0, label="identity")
    target_std = float(np.nanstd(hn_frame["target_value"]))
    pred_std = float(np.nanstd(hn_frame["moment_predicted_value"]))
    sigma_mean = float(np.nanmean(hn_frame["moment_sigma"]))
    axis.set_title(
        "HN variance calibration "
        f"(target std={target_std:.3g}, pred std={pred_std:.3g}, sigma={sigma_mean:.3g})"
    )
    axis.set_xlabel("Experimental HN shift")
    axis.set_ylabel("MomentHead HN shift")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures_dir / "hn_variance_calibration.png", dpi=200)
    plt.close(figure)


def _render_cprime_outlier_audit(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render C' robust residual and outlier-score diagnostics."""
    if "atom_family" not in frame.columns or "C'" not in set(frame["atom_family"]):
        return
    cprime_frame = frame.loc[frame["atom_family"].astype(str) == "C'"].copy()
    required = {"target_value", "moment_predicted_value", "outlier_score"}
    if cprime_frame.empty or not required.issubset(cprime_frame.columns):
        return
    residual = cprime_frame["moment_predicted_value"] - cprime_frame["target_value"]
    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    scatter = axis.scatter(
        cprime_frame["target_value"],
        residual,
        c=cprime_frame["outlier_score"],
        cmap="magma",
        s=18,
        alpha=0.75,
        label="C' target rows",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, label="zero residual")
    axis.set_title("C' outlier audit")
    axis.set_xlabel("Experimental C' shift")
    axis.set_ylabel("MomentHead residual")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("C' outlier score")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures_dir / "cprime_outlier_audit.png", dpi=200)
    plt.close(figure)


def _render_state_diversity_diagnostics(
    frame: pd.DataFrame,
    figures_dir: Path,
    plt: Any,
) -> None:
    """Render posterior state diversity diagnostics for one representative entity."""
    if frame.empty:
        return
    frame = frame.sort_values("state_rank") if "state_rank" in frame.columns else frame
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(frame["state_id"], frame["state_mass"], color="#3182bd", alpha=0.85)
    axes[0].set_title("Posterior state mass")
    axes[0].set_ylabel("Mass")
    axes[0].tick_params(axis="x", rotation=30)
    if {"state_entropy", "effective_state_count", "repulsion_score"}.issubset(
        frame.columns
    ):
        diagnostics = pd.DataFrame(
            {
                "metric": [
                    "state_entropy",
                    "effective_state_count",
                    "repulsion_score",
                ],
                "value": [
                    float(frame["state_entropy"].iloc[0]),
                    float(frame["effective_state_count"].iloc[0]),
                    (
                        float(frame["repulsion_score"].iloc[0])
                        if pd.notna(frame["repulsion_score"].iloc[0])
                        else 0.0
                    ),
                ],
            }
        )
        axes[1].barh(diagnostics["metric"], diagnostics["value"], color="#fd8d3c")
        axes[1].set_title("Diversity diagnostics")
        axes[1].set_xlabel("Value")
    else:
        axes[1].text(0.5, 0.5, "No diversity columns", ha="center", va="center")
        axes[1].set_axis_off()
    figure.tight_layout()
    figure.savefig(figures_dir / "state_diversity_diagnostics.png", dpi=200)
    plt.close(figure)


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return the canonical atom-family label for a CS target id."""
    if not str(target_id).startswith("cs:"):
        return None
    atom_name = str(target_id).rsplit(":", 1)[-1]
    return {"H": "HN", "HN": "HN", "N": "N", "CA": "CA", "CB": "CB", "C": "C'"}.get(
        atom_name
    )


def _json_float(value: Any) -> float | None:
    """Return a JSON-safe finite float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class _torch_no_grad:
    """Tiny context wrapper to avoid importing torch at module import time."""

    def __enter__(self) -> None:
        import torch

        self._context = torch.no_grad()
        self._context.__enter__()

    def __exit__(self, *args: object) -> None:
        self._context.__exit__(*args)
