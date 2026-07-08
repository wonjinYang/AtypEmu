"""CCC support geometry and oracle artifacts for AtypEmu training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from atypemu.training.config import StudentTrainingConfig
from atypemu.training.data import CHEMICAL_SHIFT_FAMILIES, PreparedTeacherExample

if TYPE_CHECKING:
    import torch


ATOM_FAMILY_INDEX = {
    family: index for index, family in enumerate(CHEMICAL_SHIFT_FAMILIES)
}


def build_ccc_geometry_artifacts(
    model: torch.nn.Module,
    train_examples: list[PreparedTeacherExample],
    val_examples: list[PreparedTeacherExample],
    device: torch.device,
    config: StudentTrainingConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write CCC support-oracle tables and figures for one final checkpoint."""
    import torch

    output_dir_path = Path(output_dir)
    arrays_dir = output_dir_path / "reports" / "arrays"
    figures_dir = output_dir_path / "reports" / "figures"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not config.enable_ccc_support_oracle:
        pd.DataFrame().to_parquet(arrays_dir / "ccc_support_oracle.parquet")
        pd.DataFrame().to_parquet(arrays_dir / "ccc_oracle_weights.parquet")
        return {"status": "disabled"}

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for split_name, examples in [("train", train_examples), ("val", val_examples)]:
            for example in examples:
                model_weights, residual_mu = _model_prediction(
                    model=model,
                    example=example,
                    device=device,
                    enable_residual=(
                        config.enable_forward_residual_head
                        or config.enable_residue_atom_residual_head
                    ),
                    config=config,
                )
                example_rows, example_weight_rows = _oracle_rows_for_example(
                    example=example,
                    split_name=split_name,
                    model_weights=model_weights,
                    residual_mu=residual_mu,
                    config=config,
                )
                rows.extend(example_rows)
                weight_rows.extend(example_weight_rows)
    if was_training:
        model.train()

    oracle_frame = pd.DataFrame(rows)
    weights_frame = pd.DataFrame(weight_rows)
    oracle_frame.to_parquet(arrays_dir / "ccc_support_oracle.parquet", index=False)
    weights_frame.to_parquet(arrays_dir / "ccc_oracle_weights.parquet", index=False)
    _render_ccc_geometry_figures(oracle_frame=oracle_frame, figures_dir=figures_dir)
    return _ccc_geometry_summary(oracle_frame)


def _model_prediction(
    model: torch.nn.Module,
    example: PreparedTeacherExample,
    device: torch.device,
    enable_residual: bool,
    config: StudentTrainingConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return model posterior weights and optional forward residuals."""
    from atypemu.training.trainer import model_outputs_for_example, to_tensors

    tensors = to_tensors(example, device)
    outputs = model_outputs_for_example(
        model=model,
        tensors=tensors,
        evidence_mask=None,
        config=config,
    )
    residual = (
        outputs["forward_residual_mu"].detach().cpu().numpy()
        if enable_residual and "forward_residual_mu" in outputs
        else None
    )
    return outputs["weights"].detach().cpu().numpy(), residual


def _oracle_rows_for_example(
    example: PreparedTeacherExample,
    split_name: str,
    model_weights: np.ndarray,
    residual_mu: np.ndarray | None,
    config: StudentTrainingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return support-oracle rows and candidate-weight rows for one example."""
    channel = example.channels.get("chemical_shifts")
    if channel is None:
        return [], []
    values = _values_with_residual(channel.values, channel.target_ids, residual_mu)
    mask = channel.mask.astype(bool, copy=False)
    targets = channel.target_values.astype(np.float64, copy=False)
    families = np.asarray(
        [_atom_family_from_target_id(target_id) for target_id in channel.target_ids],
        dtype=object,
    )
    candidate_count = int(values.shape[1])
    uniform = np.ones(candidate_count, dtype=np.float64) / max(candidate_count, 1)
    teacher = _normalize(example.teacher_weights)
    model = _normalize(model_weights)
    single_weights = _best_single_candidate_weights(values, mask, targets, families)
    softmax_weights = _softmax_candidate_ccc_weights(
        values=values,
        mask=mask,
        targets=targets,
        families=families,
        tau=config.ccc_oracle_temperature,
    )
    family_mixture = _family_mixture_weights(
        values=values,
        mask=mask,
        targets=targets,
        families=families,
        tau=config.ccc_oracle_temperature,
    )
    projected = _projected_simplex_ccc_oracle(
        values=values,
        mask=mask,
        targets=targets,
        families=families,
        initial_weights=softmax_weights,
        steps=max(int(config.ccc_oracle_steps), 1),
    )
    variants = {
        "uniform": uniform,
        "teacher": teacher,
        "model": model,
        "best_single_candidate": single_weights,
        "softmax_candidate_ccc": softmax_weights,
        "projected_simplex_ccc_oracle": projected,
        "family_mixture_oracle": family_mixture,
    }

    rows: list[dict[str, Any]] = []
    for variant_name, weights in variants.items():
        row = _score_row(
            example=example,
            split_name=split_name,
            variant=variant_name,
            weights=weights,
            values=values,
            mask=mask,
            targets=targets,
            families=families,
        )
        rows.append(row)
        if variant_name in {"model", "projected_simplex_ccc_oracle"}:
            rows.append(
                _score_row(
                    example=example,
                    split_name=split_name,
                    variant=f"{variant_name}_affine_calibrated",
                    weights=weights,
                    values=values,
                    mask=mask,
                    targets=targets,
                    families=families,
                    affine_calibrated=True,
                )
            )

    weight_rows = []
    for variant_name in ["model", "projected_simplex_ccc_oracle"]:
        for candidate_index, weight in enumerate(variants[variant_name]):
            candidate_id = (
                example.candidate_ids[candidate_index]
                if candidate_index < len(example.candidate_ids)
                else str(candidate_index)
            )
            weight_rows.append(
                {
                    "entity_uid": example.entity_uid,
                    "bmrb_id": example.metadata.get("bmrb_id", ""),
                    "split": split_name,
                    "variant": variant_name,
                    "candidate_index": int(candidate_index),
                    "candidate_id": candidate_id,
                    "weight": float(weight),
                }
            )
    return rows, weight_rows


def _score_row(
    example: PreparedTeacherExample,
    split_name: str,
    variant: str,
    weights: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    affine_calibrated: bool = False,
) -> dict[str, Any]:
    """Score one support-weight variant with family macro CCC."""
    predictions, usable = _weighted_predictions(weights, values, mask)
    if affine_calibrated:
        predictions = _affine_calibrate_by_family(
            predictions, targets, families, usable
        )
    family_scores: dict[str, float] = {}
    for family in CHEMICAL_SHIFT_FAMILIES:
        family_mask = usable & (families == family)
        family_scores[family] = (
            _safe_ccc(predictions[family_mask], targets[family_mask])
            if np.count_nonzero(family_mask) >= 2
            else math.nan
        )
    finite_scores = [score for score in family_scores.values() if math.isfinite(score)]
    row = {
        "entity_uid": example.entity_uid,
        "bmrb_id": example.metadata.get("bmrb_id", ""),
        "split": split_name,
        "variant": variant,
        "cs_family_ccc": (float(np.mean(finite_scores)) if finite_scores else math.nan),
        "support_size": int(weights.size),
        "effective_support_size": _effective_support_size(weights),
    }
    for family, score in family_scores.items():
        row[f"cs_{family}_ccc"] = score
    return row


def _projected_simplex_ccc_oracle(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    initial_weights: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Approximate max_w family CCC(Xw, y) with softmax logits."""
    import torch

    device = torch.device("cpu")
    if values.shape[1] <= 1:
        return _normalize(initial_weights)
    value_t = torch.tensor(values, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask.astype(np.float32), dtype=torch.float32, device=device)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    logits = torch.nn.Parameter(
        torch.log(torch.tensor(_normalize(initial_weights), dtype=torch.float32) + 1e-8)
    )
    optimizer = torch.optim.Adam([logits], lr=0.1)
    best_weights = _normalize(initial_weights)
    best_score = _family_macro_ccc_np(best_weights, values, mask, targets, families)
    uniform = np.ones_like(best_weights) / max(best_weights.size, 1)
    uniform_score = _family_macro_ccc_np(uniform, values, mask, targets, families)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        denominator = (mask_t * weights.unsqueeze(0)).sum(dim=1).clamp_min(1e-8)
        predictions = (mask_t * value_t * weights.unsqueeze(0)).sum(dim=1) / denominator
        losses = []
        for family in CHEMICAL_SHIFT_FAMILIES:
            family_mask_np = (families == family) & (mask.sum(axis=1) > 0)
            if np.count_nonzero(family_mask_np) < 2:
                continue
            family_mask = torch.tensor(family_mask_np, dtype=torch.bool, device=device)
            losses.append(
                _torch_ccc_loss(predictions[family_mask], target_t[family_mask])
            )
        if not losses:
            break
        loss = torch.mean(torch.stack(losses))
        if not loss.requires_grad:
            break
        loss.backward()
        optimizer.step()
        candidate_weights = torch.softmax(logits.detach(), dim=0).cpu().numpy()
        score = _family_macro_ccc_np(candidate_weights, values, mask, targets, families)
        if math.isfinite(score) and (
            not math.isfinite(best_score) or score > best_score
        ):
            best_score = score
            best_weights = candidate_weights
    if math.isfinite(uniform_score) and (
        not math.isfinite(best_score) or uniform_score > best_score
    ):
        return uniform
    return _normalize(best_weights)


def _torch_ccc_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return differentiable 1 - CCC for oracle optimization."""
    import torch

    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)
    pred_var = torch.mean(torch.square(predictions - pred_mean))
    target_var = torch.mean(torch.square(targets - target_mean))
    cov = torch.mean((predictions - pred_mean) * (targets - target_mean))
    ccc = (
        2.0
        * cov
        / (pred_var + target_var + torch.square(pred_mean - target_mean)).clamp_min(
            1e-8
        )
    )
    return 1.0 - ccc


def _best_single_candidate_weights(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
) -> np.ndarray:
    """Return one-hot weights for the best single candidate by family CCC."""
    scores = _candidate_family_macro_scores(values, mask, targets, families)
    best_index = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
    weights = np.zeros(values.shape[1], dtype=np.float64)
    weights[best_index] = 1.0
    return weights


def _softmax_candidate_ccc_weights(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return softmax weights over candidate family-CCC proxy scores."""
    scores = _candidate_family_macro_scores(values, mask, targets, families)
    return _softmax_scores(scores, tau)


def _family_mixture_weights(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Average per-family softmax candidate weights."""
    weights = []
    for family in CHEMICAL_SHIFT_FAMILIES:
        scores = np.full(values.shape[1], np.nan, dtype=np.float64)
        family_mask = families == family
        if np.count_nonzero(family_mask) < 2:
            continue
        for candidate_index in range(values.shape[1]):
            valid = mask[:, candidate_index] & family_mask
            if np.count_nonzero(valid) >= 2:
                scores[candidate_index] = _safe_ccc(
                    values[valid, candidate_index],
                    targets[valid],
                )
        weights.append(_softmax_scores(scores, tau))
    if not weights:
        return np.ones(values.shape[1], dtype=np.float64) / max(values.shape[1], 1)
    return _normalize(np.mean(np.stack(weights, axis=0), axis=0))


def _candidate_family_macro_scores(
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
) -> np.ndarray:
    """Return candidate-level family macro CCC scores."""
    scores = np.full(values.shape[1], np.nan, dtype=np.float64)
    for candidate_index in range(values.shape[1]):
        family_scores = []
        for family in CHEMICAL_SHIFT_FAMILIES:
            valid = mask[:, candidate_index] & (families == family)
            if np.count_nonzero(valid) >= 2:
                score = _safe_ccc(values[valid, candidate_index], targets[valid])
                if math.isfinite(score):
                    family_scores.append(score)
        if family_scores:
            scores[candidate_index] = float(np.mean(family_scores))
    return scores


def _weighted_predictions(
    weights: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted chemical-shift predictions and usable target mask."""
    weights = _normalize(weights)
    validity = mask.astype(np.float64, copy=False)
    denominator = (validity * weights[None, :]).sum(axis=1)
    usable = denominator > 1e-8
    predictions = (validity * values * weights[None, :]).sum(axis=1) / np.clip(
        denominator,
        1e-8,
        None,
    )
    return predictions, usable


def _values_with_residual(
    values: np.ndarray,
    target_ids: list[str],
    residual_mu: np.ndarray | None,
) -> np.ndarray:
    """Apply optional candidate by atom-family residual corrections."""
    adjusted = values.astype(np.float64, copy=True)
    if residual_mu is None:
        return adjusted
    residual = np.asarray(residual_mu, dtype=np.float64)
    if residual.shape == adjusted.shape:
        return adjusted + residual
    if residual.ndim != 2 or residual.shape[0] != adjusted.shape[1]:
        return adjusted
    for family, family_index in ATOM_FAMILY_INDEX.items():
        row_mask = np.asarray(
            [
                _atom_family_from_target_id(target_id) == family
                for target_id in target_ids
            ],
            dtype=bool,
        )
        if np.any(row_mask) and family_index < residual.shape[1]:
            adjusted[row_mask] += residual[:, family_index][None, :]
    return adjusted


def _affine_calibrate_by_family(
    predictions: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    usable: np.ndarray,
) -> np.ndarray:
    """Return family-wise affine calibrated predictions."""
    calibrated = predictions.copy()
    for family in CHEMICAL_SHIFT_FAMILIES:
        mask = usable & (families == family)
        if np.count_nonzero(mask) < 2:
            continue
        x = predictions[mask]
        y = targets[mask]
        var = float(np.var(x))
        if var <= 1e-12:
            continue
        slope = float(np.cov(x, y, bias=True)[0, 1] / var)
        intercept = float(np.mean(y) - slope * np.mean(x))
        calibrated[mask] = slope * x + intercept
    return calibrated


def _family_macro_ccc_np(
    weights: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
) -> float:
    """Return family macro CCC for one support-weight vector."""
    predictions, usable = _weighted_predictions(weights, values, mask)
    scores = []
    for family in CHEMICAL_SHIFT_FAMILIES:
        family_mask = usable & (families == family)
        if np.count_nonzero(family_mask) >= 2:
            score = _safe_ccc(predictions[family_mask], targets[family_mask])
            if math.isfinite(score):
                scores.append(score)
    return float(np.mean(scores)) if scores else math.nan


def _softmax_scores(scores: np.ndarray, tau: float) -> np.ndarray:
    """Convert optional scores to a finite simplex vector."""
    numeric = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(numeric).any():
        return np.ones(numeric.size, dtype=np.float64) / max(numeric.size, 1)
    fill = float(np.nanmin(numeric[np.isfinite(numeric)]) - 1.0)
    numeric = np.where(np.isfinite(numeric), numeric, fill)
    scaled = numeric / max(float(tau), 1e-6)
    scaled = scaled - float(np.max(scaled))
    exp = np.exp(np.clip(scaled, -60.0, 60.0))
    return _normalize(exp)


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a finite nonnegative vector to the simplex."""
    weights = np.asarray(values, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / max(weights.size, 1)
    return weights / total


def _effective_support_size(weights: np.ndarray) -> float:
    """Return effective support size."""
    clipped = np.clip(_normalize(weights), 1e-12, None)
    return float(1.0 / np.sum(np.square(clipped)))


def _safe_ccc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient."""
    if predictions.size < 2 or targets.size < 2:
        return math.nan
    pred_mean = float(np.mean(predictions))
    target_mean = float(np.mean(targets))
    pred_var = float(np.mean(np.square(predictions - pred_mean)))
    target_var = float(np.mean(np.square(targets - target_mean)))
    cov = float(np.mean((predictions - pred_mean) * (targets - target_mean)))
    denom = pred_var + target_var + (pred_mean - target_mean) ** 2
    return math.nan if denom <= 1e-12 else float(2.0 * cov / denom)


def _atom_family_from_target_id(target_id: str) -> str | None:
    """Return one canonical chemical-shift atom-family label."""
    if not str(target_id).startswith("cs:"):
        return None
    return {"H": "HN", "HN": "HN", "N": "N", "CA": "CA", "CB": "CB", "C": "C'"}.get(
        str(target_id).rsplit(":", 1)[-1]
    )


def _render_ccc_geometry_figures(oracle_frame: pd.DataFrame, figures_dir: Path) -> None:
    """Render compact CCC oracle/model gap figures."""
    if oracle_frame.empty:
        return
    import matplotlib.pyplot as plt

    val = oracle_frame.loc[oracle_frame["split"].astype(str) == "val"].copy()
    if val.empty:
        val = oracle_frame.copy()
    pivot = val.pivot_table(
        index="entity_uid",
        columns="variant",
        values="cs_family_ccc",
        aggfunc="max",
    )
    if {"model", "projected_simplex_ccc_oracle"}.issubset(pivot.columns):
        figure, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
        axis.scatter(
            pivot["model"],
            pivot["projected_simplex_ccc_oracle"],
            color="#4c78a8",
            alpha=0.78,
            edgecolors="none",
        )
        low = float(
            np.nanmin(pivot[["model", "projected_simplex_ccc_oracle"]].to_numpy())
        )
        high = float(
            np.nanmax(pivot[["model", "projected_simplex_ccc_oracle"]].to_numpy())
        )
        axis.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Model CS family CCC")
        axis.set_ylabel("Projected-simplex oracle CCC")
        axis.set_title("Oracle vs model CCC")
        figure.savefig(figures_dir / "oracle_vs_model_ccc.png", dpi=200)
        plt.close(figure)

    family_columns = [f"cs_{family}_ccc" for family in CHEMICAL_SHIFT_FAMILIES]
    model = val.loc[val["variant"] == "model"]
    oracle = val.loc[val["variant"] == "projected_simplex_ccc_oracle"]
    if not model.empty and not oracle.empty:
        model_means = model[family_columns].mean(numeric_only=True)
        oracle_means = oracle[family_columns].mean(numeric_only=True)
        gap = oracle_means - model_means
        figure, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        axis.bar(
            [column.replace("cs_", "").replace("_ccc", "") for column in gap.index],
            gap.to_numpy(dtype=float),
            color="#f58518",
            alpha=0.86,
        )
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_ylabel("Oracle - model CCC")
        axis.set_title("CCC support gap by atom family")
        figure.savefig(figures_dir / "ccc_support_gap_by_family.png", dpi=200)
        plt.close(figure)


def _ccc_geometry_summary(oracle_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize CCC geometry rows for reports."""
    if oracle_frame.empty:
        return {"status": "missing_oracle_rows"}
    val = oracle_frame.loc[oracle_frame["split"].astype(str) == "val"].copy()
    if val.empty:
        val = oracle_frame.copy()
    pivot = val.pivot_table(
        index="entity_uid",
        columns="variant",
        values="cs_family_ccc",
        aggfunc="max",
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "examples": int(pivot.shape[0]),
    }
    for variant in [
        "uniform",
        "teacher",
        "model",
        "projected_simplex_ccc_oracle",
        "family_mixture_oracle",
        "projected_simplex_ccc_oracle_affine_calibrated",
    ]:
        if variant in pivot.columns:
            summary[f"{variant}_ccc_macro"] = _json_float(pivot[variant].mean())
    if {"model", "projected_simplex_ccc_oracle"}.issubset(pivot.columns):
        gap = pivot["projected_simplex_ccc_oracle"] - pivot["model"]
        summary["model_gap_macro"] = _json_float(gap.mean())
        summary["support_ceiling_macro"] = _json_float(
            pivot["projected_simplex_ccc_oracle"].mean()
        )
    return summary


def _json_float(value: Any) -> float | None:
    """Return finite JSON float or null."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
