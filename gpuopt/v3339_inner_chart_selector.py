#!/usr/bin/env python3
"""Inner-train/dev-only selector for one frozen Phase-D physics-chart menu."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARTIFACT_KIND = "v3339_phase_d_inner_chart_selector_v1"
INPUT_KIND = "v3339_phase_d_inner_selector_inputs_v1"
MENU_SHA256 = "3d120deccf219f517938467bd0e18fb4f2fffe1d9444b09250c6f908e4cac42f"
CONTROL_ATOM27_SOURCE_SHA256 = "0b3e6dc0974e0da9387cfc7cf5113c46c47b88fd85d2373be812e1020f397a41"
FAMILIES = ("C'", "CA", "CB", "HN", "N")
MIN_ACTIVE_ROWS = 10
MIN_ACTIVE_ENTITIES = 3
IMPROVEMENT_EPS = 1.0e-10
TAIL_ATOL = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: Path, expected_sha256: str, name: str) -> Any:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"source SHA256 drift: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [str(value.dtype), list(value.shape)], separators=(",", ":")
    ).encode())
    digest.update(b"\n")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def concordance(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 3 or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        return None
    magnitude = max(float(np.max(np.abs(target))), float(np.max(np.abs(prediction))), 1.0)
    target = target / magnitude
    prediction = prediction / magnitude
    if float(np.var(target)) <= 1.0e-12 / (magnitude * magnitude):
        return None
    target_mean = float(target.mean())
    prediction_mean = float(prediction.mean())
    covariance = float(np.mean((target - target_mean) * (prediction - prediction_mean)))
    denominator = float(
        np.var(target) + np.var(prediction) + (target_mean - prediction_mean) ** 2
    )
    result = 2.0 * covariance / denominator if denominator > 1.0e-12 / (
        magnitude * magnitude
    ) else None
    if result is not None and not math.isfinite(result):
        raise ValueError("CCC became nonfinite")
    return result


def higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def metrics(
    *,
    entity: np.ndarray,
    family: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    target_weight: np.ndarray,
) -> dict[str, Any]:
    if not (
        len(entity) == len(family) == len(target) == len(prediction) == len(scale)
        == len(target_weight)
        and len(target) > 0
        and np.isfinite(target).all()
        and np.isfinite(prediction).all()
        and np.isfinite(scale).all()
        and np.all(scale > 0)
        and np.isfinite(target_weight).all()
        and np.all(target_weight >= 0)
    ):
        raise ValueError("metric arrays are invalid")
    family_ccc: dict[str, float] = {}
    family_cells: dict[str, int] = {}
    for selected_family in FAMILIES:
        cells = []
        for uid in np.unique(entity[family == selected_family]):
            selected = (entity == uid) & (family == selected_family)
            value = concordance(target[selected], prediction[selected])
            if value is not None:
                cells.append(value)
        if not cells:
            raise ValueError(f"family lacks entity CCC cells: {selected_family}")
        family_ccc[selected_family] = float(np.mean(cells))
        family_cells[selected_family] = len(cells)
    standardized = (target - prediction) / scale
    nll = np.log(scale) + 2.5 * np.log1p(standardized * standardized / 4.0)
    entity_data_loss = []
    for uid in np.unique(entity):
        selected = entity == uid
        denominator = float(np.sum(target_weight[selected]))
        if denominator <= 0:
            raise ValueError("entity has no positive production target weight")
        entity_data_loss.append(
            float(np.sum(nll[selected] * target_weight[selected]) / denominator)
        )
    hn = family == "HN"
    raw_hn = np.abs(target[hn] - prediction[hn])
    standardized_hn = np.abs(standardized[hn])
    if not len(raw_hn):
        raise ValueError("metrics lack HN rows")
    return {
        "record_count": len(target),
        "family_entity_cell_count": family_cells,
        "family_entity_macro_ccc": family_ccc,
        "five_family_macro_ccc": float(np.mean(list(family_ccc.values()))),
        "student_t_df4_nll_without_shared_constant": float(np.mean(nll)),
        "production_entity_balanced_student_t_data_loss": float(
            np.mean(entity_data_loss)
        ),
        "hn_tail": {
            "raw_abs_p95": higher_quantile(raw_hn, 0.95),
            "raw_abs_max": float(np.max(raw_hn)),
            "standardized_abs_p95": higher_quantile(standardized_hn, 0.95),
            "standardized_abs_max": float(np.max(standardized_hn)),
        },
    }


def metric_delta(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    return {
        "five_family_macro_ccc": (
            candidate["five_family_macro_ccc"] - control["five_family_macro_ccc"]
        ),
        "hn_entity_macro_ccc": (
            candidate["family_entity_macro_ccc"]["HN"]
            - control["family_entity_macro_ccc"]["HN"]
        ),
        "student_t_df4_nll_without_shared_constant": (
            candidate["student_t_df4_nll_without_shared_constant"]
            - control["student_t_df4_nll_without_shared_constant"]
        ),
        "production_entity_balanced_student_t_data_loss": (
            candidate["production_entity_balanced_student_t_data_loss"]
            - control["production_entity_balanced_student_t_data_loss"]
        ),
        "hn_tail": {
            key: candidate["hn_tail"][key] - control["hn_tail"][key]
            for key in control["hn_tail"]
        },
    }


def read_menu(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(path) != MENU_SHA256:
        raise ValueError("pre-unblinding menu SHA256 drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", ())
    schema = payload.get("candidate_schema", {})
    if (
        payload.get("decision") != "SAFE_MENU"
        or len(candidates) != 12
        or schema.get("centers_nm") != [0.2, 0.3, 0.4, 0.55, 0.75, 1.0]
        or float(schema.get("rbf_width_nm", -1)) != 0.12
        or float(schema.get("magnitude_ppm", -1)) != 0.005
        or sorted(int(value) for value in schema.get("sign_set", ())) != [-1, 1]
        or len({row.get("id") for row in candidates}) != 12
    ):
        raise ValueError("pre-unblinding menu contract drift")
    return payload, list(candidates)


def safe_path(root: Path, value: str) -> Path:
    path = Path(value)
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"input escaped sealed root: {path}")
    cursor = path
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError(f"sealed input has a symbolic path component: {cursor}")
        cursor = cursor.parent
    return path


def read_inputs(path: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("input manifest SHA256 drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(payload.get("sealed_root", ""))
    if payload.get("artifact_kind") != INPUT_KIND or not root.is_dir():
        raise ValueError("sealed input manifest identity drift")
    for forbidden in payload.get("forbidden_paths", ()):
        if Path(forbidden).exists():
            raise ValueError(f"outer-result path is visible in selector container: {forbidden}")
    if len(payload.get("folds", ())) != 3:
        raise ValueError("selector requires exactly three folds")
    for row in payload["folds"]:
        for key in (
            "train_manifest", "dev_manifest", "q_checkpoint",
            "stage_a_checkpoint", "control_checkpoint",
        ):
            item = safe_path(root, row[key])
            if sha256_file(item) != row[f"{key}_sha256"]:
                raise ValueError(f"sealed input SHA256 drift: fold={row.get('fold')} key={key}")
    return root, payload


def sulfur_features_reference(
    *,
    atom27_positions: torch.Tensor,
    atom27_mask: torch.Tensor,
    atom27_role_features: torch.Tensor,
    residue_index: torch.Tensor,
    equivalent_structure_slots: torch.Tensor,
) -> torch.Tensor:
    """Recompute the six sulfur RBFs without using the model's chart helper."""
    batch, support_count, residue_count = atom27_positions.shape[:3]
    target_count = residue_index.shape[1]
    if (
        atom27_positions.shape != (batch, support_count, residue_count, 27, 3)
        or atom27_mask.shape != (batch, support_count, residue_count, 27)
        or atom27_role_features.shape != (batch, support_count, residue_count, 27, 7)
        or equivalent_structure_slots.shape != (batch, target_count, 3)
    ):
        raise ValueError("independent sulfur-chart tensor shape drift")
    declared = equivalent_structure_slots >= 0
    safe = equivalent_structure_slots.clamp_min(0)
    batch_index = torch.arange(batch, device=atom27_positions.device)[:, None, None]
    support_index = torch.arange(
        support_count, device=atom27_positions.device
    )[None, :, None]
    target_residue = residue_index[:, None].expand(-1, support_count, -1)
    local_positions = atom27_positions[batch_index, support_index, target_residue]
    local_mask = atom27_mask[batch_index, support_index, target_residue]
    gather_index = safe[:, None, :, :, None].expand(-1, support_count, -1, -1, 3)
    equivalent_positions = torch.gather(local_positions, 3, gather_index)
    equivalent_present = torch.gather(
        local_mask, 3, safe[:, None].expand(-1, support_count, -1, -1)
    ) & declared[:, None]
    denominator = equivalent_present.sum(dim=-1, keepdim=True).clamp_min(1).to(
        equivalent_positions.dtype
    )
    site = (
        equivalent_positions
        * equivalent_present[..., None].to(equivalent_positions.dtype)
    ).sum(dim=3) / denominator
    site_valid = equivalent_present.any(dim=-1)
    sulfur = atom27_mask & atom27_role_features[..., 4]
    residue_grid = torch.arange(
        residue_count, device=atom27_positions.device
    )[None, None, None, :, None]
    centers = atom27_positions.new_tensor((0.20, 0.30, 0.40, 0.55, 0.75, 1.00))
    chunks: list[torch.Tensor] = []
    for start in range(0, target_count, 64):
        stop = min(start + 64, target_count)
        distance = (
            atom27_positions[:, :, None]
            - site[:, :, start:stop, None, None]
        ).square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
        inter_residue = residue_grid != residue_index[
            :, None, start:stop, None, None
        ]
        selected = (
            sulfur[:, :, None]
            & inter_residue
            & site_valid[:, :, start:stop, None, None]
        )
        radial = torch.exp(-0.5 * (((distance[..., None] - centers) / 0.12).square()))
        count = selected.sum(dim=(-2, -1)).clamp_min(1).to(radial.dtype)
        z = (radial * selected[..., None].to(radial.dtype)).sum(
            dim=(-3, -2)
        ) / count[..., None].sqrt()
        chunks.append(1.0 - torch.exp(-z))
    result = torch.cat(chunks, dim=2)
    if (
        result.shape != (batch, support_count, target_count, 6)
        or torch.any(result < 0)
        or torch.any(result >= 1)
        or not torch.isfinite(result).all()
    ):
        raise RuntimeError("independent sulfur-chart feature contract failed")
    return result


class SplitArrays:
    def __init__(self, candidate_count: int) -> None:
        self.entity: list[np.ndarray] = []
        self.family: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.control_prediction: list[np.ndarray] = []
        self.control_scale: list[np.ndarray] = []
        self.target_weight: list[np.ndarray] = []
        self.candidate_prediction: list[list[np.ndarray]] = [
            [] for _ in range(candidate_count)
        ]
        self.candidate_scale: list[list[np.ndarray]] = [
            [] for _ in range(candidate_count)
        ]
        self.active: list[list[np.ndarray]] = [[] for _ in range(candidate_count)]

    def concatenate(self) -> dict[str, Any]:
        return {
            "entity": np.concatenate(self.entity),
            "family": np.concatenate(self.family),
            "target": np.concatenate(self.target),
            "control_prediction": np.concatenate(self.control_prediction),
            "control_scale": np.concatenate(self.control_scale),
            "target_weight": np.concatenate(self.target_weight),
            "candidate_prediction": [np.concatenate(value) for value in self.candidate_prediction],
            "candidate_scale": [np.concatenate(value) for value in self.candidate_scale],
            "active": [np.concatenate(value) for value in self.active],
        }


def candidate_tensors(
    *,
    trainer: Any,
    batch: dict[str, torch.Tensor],
    standardized_mean: torch.Tensor,
    standardized_log_scale: torch.Tensor,
    positions: torch.Tensor,
    q: torch.Tensor,
    candidates: list[dict[str, Any]],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    features = sulfur_features_reference(
        atom27_positions=positions,
        atom27_mask=batch["atom27_mask"],
        atom27_role_features=batch["atom27_role_features"],
        residue_index=batch["residue_index"],
        equivalent_structure_slots=batch["equivalent_structure_slots"],
    )
    model_features = trainer.D2.phe_hn_inter_sulfur_features(
        atom27_positions=positions,
        atom27_mask=batch["atom27_mask"],
        atom27_role_features=batch["atom27_role_features"],
        residue_index=batch["residue_index"],
        equivalent_structure_slots=batch["equivalent_structure_slots"],
    )
    if not torch.equal(features, model_features):
        raise ValueError("independent sulfur chart differs from model helper")
    target_token = torch.gather(
        batch["stage_a_tokens"], 1, batch["residue_index"]
    )
    gate = (
        (target_token == 5)
        & (batch["atom_index"] == 55)
    )
    raw_support_mean = (
        batch["stage_a_mean"][:, None]
        + standardized_mean * batch["stage_a_scale"][:, None]
    )
    support_log_scale = (
        standardized_log_scale
        + batch["stage_a_scale"].clamp_min(1.0e-6).log()[:, None]
    )
    centers = [0.20, 0.30, 0.40, 0.55, 0.75, 1.00]
    center_index = [centers.index(float(row["center_nm"])) for row in candidates]
    signs = raw_support_mean.new_tensor([int(row["sign"]) for row in candidates])
    selected_features = features[..., center_index].permute(3, 0, 1, 2)
    correction = (
        signs[:, None, None, None]
        * 0.005
        * selected_features
        * gate[None, :, None].to(selected_features.dtype)
    )
    if float(correction.detach().abs().amax().cpu()) > 0.0050000001:
        raise RuntimeError("candidate exceeded its frozen 0.005 ppm chart bound")
    candidate_support = raw_support_mean[None] + correction
    within = torch.einsum(
        "bk,bkt->bt", q, torch.exp(2.0 * support_log_scale)
    )
    control_mean = torch.einsum("bk,bkt->bt", q, raw_support_mean)
    control_between = (
        torch.einsum("bk,bkt->bt", q, raw_support_mean.square())
        - control_mean.square()
    ).clamp_min(0.0)
    control_scale = (within + 0.5 * control_between).clamp_min(1.0e-6).sqrt()
    means, scales, weighted = [], [], []
    for index in range(len(candidates)):
        mean = torch.einsum("bk,bkt->bt", q, candidate_support[index])
        between = (
            torch.einsum("bk,bkt->bt", q, candidate_support[index].square())
            - mean.square()
        ).clamp_min(0.0)
        scale = (within + 0.5 * between).clamp_min(1.0e-6).sqrt()
        means.append(torch.where(gate, mean, control_mean))
        scales.append(torch.where(gate, scale, control_scale))
        weighted.append(torch.einsum("bk,bkt->bt", q, selected_features[index]))
    candidate_mean = torch.stack(means)
    candidate_scale = torch.stack(scales)
    weighted_feature = torch.stack(weighted)
    active = gate[None] & (weighted_feature > 1.0e-8)
    return (
        raw_support_mean, support_log_scale, candidate_support,
        candidate_mean, candidate_scale, active,
    )


def validate_checkpoint(
    *,
    trainer: Any,
    checkpoint_path: Path,
    fold_row: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    binding = checkpoint.get("binding", {})
    if (
        set(checkpoint) != {
            "artifact_kind", "binding", "binding_sha256", "model_state_dict",
            "q_initialization_classification", "selected_inner_dev_loss",
            "selection_used_outer_held",
        }
        or set(binding) != {
            "artifact_kind", "atom27_source_sha256", "comparison_contract_sha256",
            "dev_manifest_sha256", "outer_fold", "outer_held_manifest_sha256",
            "q_initialization", "resource_receipt_sha256", "route", "runtime",
            "schedule", "stage_a_checkpoint_sha256", "train_manifest_sha256",
        }
        or checkpoint.get("binding_sha256") != canonical_sha256(binding)
        or checkpoint.get("artifact_kind") != "v3339_phase_d_atom27_checkpoint_v1"
        or checkpoint.get("selection_used_outer_held") is not False
        or checkpoint.get("q_initialization_classification")
        != "initialization_transfer_not_lineage_continuation"
        or binding.get("route") != "legacy5"
        or int(binding.get("outer_fold", -1)) != int(fold_row["fold"])
        or binding.get("atom27_source_sha256") != CONTROL_ATOM27_SOURCE_SHA256
        or binding.get("train_manifest_sha256")
        != fold_row["original_train_manifest_sha256"]
        or binding.get("dev_manifest_sha256")
        != fold_row["original_dev_manifest_sha256"]
        or binding.get("schedule")
        != {"epochs": 40, "seed": 3339, "learning_rate": 0.0003}
        or not isinstance(checkpoint.get("model_state_dict"), dict)
    ):
        raise ValueError("inner-selected control checkpoint contract drift")
    return checkpoint


def collect_split(
    *,
    trainer: Any,
    model: torch.nn.Module,
    stage_a: torch.nn.Module,
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    device: torch.device,
    declared_candidate_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = SplitArrays(len(candidates))
    q_digest = hashlib.sha256()
    posterior_ess: list[float] = []
    posterior_entropy: list[float] = []
    state_digests = {
        name: hashlib.sha256() for name in (
            "q", "actuated_atom27_positions", "standardized_mean",
            "standardized_log_scale", "raw_support_mean",
            "control_ensemble_mean", "control_ensemble_scale",
        )
    }
    capture: dict[str, torch.Tensor] = {}

    def observer_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        capture["standardized_mean"] = output[0]
        capture["standardized_log_scale"] = output[1]

    handle = model.observer.register_forward_hook(observer_hook, with_kwargs=True)
    try:
        for row in rows:
            raw, metadata = trainer.load_batch(row)
            routed, routed_metadata = trainer.route_batch(
                raw, metadata, "legacy5", evaluation=False
            )
            batch = trainer.V6.attach_frozen_stage_a(
                trainer.batch_to_device(routed, device), stage_a
            )
            capture.clear()
            with torch.no_grad():
                output = model(batch)
            if set(capture) != {"standardized_mean", "standardized_log_scale"}:
                raise RuntimeError("observer diagnostic hook did not fire exactly once")
            q = output["q"]
            (
                raw_support, support_log_scale, candidate_support,
                candidate_mean, candidate_scale, active,
            ) = (
                candidate_tensors(
                    trainer=trainer,
                    batch=batch,
                    standardized_mean=capture["standardized_mean"],
                    standardized_log_scale=capture["standardized_log_scale"],
                    positions=output["actuated_atom27_positions"],
                    q=q,
                    candidates=candidates,
                )
            )
            control_mean = torch.einsum("bk,bkt->bt", q, raw_support)
            within = torch.einsum(
                "bk,bkt->bt", q, torch.exp(2.0 * support_log_scale)
            )
            between = (
                torch.einsum("bk,bkt->bt", q, raw_support.square())
                - control_mean.square()
            ).clamp_min(0.0)
            control_scale = (within + 0.5 * between).clamp_min(1.0e-6).sqrt()
            if declared_candidate_id is None:
                expected_support = raw_support
                expected_mean = control_mean
                expected_scale = control_scale
            else:
                indices = [
                    index for index, candidate in enumerate(candidates)
                    if candidate["id"] == declared_candidate_id
                ]
                if len(indices) != 1:
                    raise ValueError("source declares a candidate outside the frozen menu")
                index = indices[0]
                expected_support = candidate_support[index]
                expected_mean = candidate_mean[index]
                expected_scale = candidate_scale[index]
            if not (
                torch.equal(output["support_mean"], expected_support)
                and torch.equal(output["ensemble_mean"], expected_mean)
                and torch.equal(output["ensemble_scale"], expected_scale)
            ):
                raise ValueError("model output is not the declared frozen chart realization")

            selected = torch.nonzero(
                batch["scoring_mask"][0], as_tuple=False
            ).flatten()
            selected_cpu = selected.detach().cpu().numpy()
            target = batch["target_values"][0, selected].detach().cpu().numpy().astype(np.float64)
            control_prediction = control_mean[0, selected].detach().cpu().numpy().astype(np.float64)
            control_scale_np = control_scale[0, selected].detach().cpu().numpy().astype(np.float64)
            families = np.asarray(routed_metadata["legacy_families"], dtype=object)[selected_cpu]
            target_ids = np.asarray(routed_metadata["target_ids"], dtype=object)[selected_cpu]
            metadata_gate = np.asarray([
                family == "HN" and ":PHE:HN" in str(target_id)
                for family, target_id in zip(families, target_ids, strict=True)
            ])
            tensor_gate = (
                (
                    torch.gather(batch["stage_a_tokens"], 1, batch["residue_index"])
                    == trainer.D2.PHYSICS_CHART_PHE_TOKEN
                )
                & (batch["atom_index"] == trainer.D2.PHYSICS_CHART_HN_ATOM_INDEX)
            )[0, selected].detach().cpu().numpy()
            if not np.array_equal(metadata_gate, tensor_gate):
                raise ValueError("PHE/HN metadata gate differs from frozen token/atom gate")
            arrays.entity.append(np.full(len(selected_cpu), row["entity_uid"], dtype=object))
            arrays.family.append(families)
            arrays.target.append(target)
            arrays.control_prediction.append(control_prediction)
            arrays.control_scale.append(control_scale_np)
            arrays.target_weight.append(
                batch["target_weights"][0, selected]
                .detach().cpu().numpy().astype(np.float64)
            )
            for index in range(len(candidates)):
                arrays.candidate_prediction[index].append(
                    candidate_mean[index, 0, selected].detach().cpu().numpy().astype(np.float64)
                )
                arrays.candidate_scale[index].append(
                    candidate_scale[index, 0, selected].detach().cpu().numpy().astype(np.float64)
                )
                arrays.active[index].append(
                    active[index, 0, selected].detach().cpu().numpy()
                )
            q_digest.update(row["entity_uid"].encode())
            q_digest.update(b"\n")
            q_digest.update(tensor_sha256(q).encode())
            state_tensors = {
                "q": q,
                "actuated_atom27_positions": output["actuated_atom27_positions"],
                "standardized_mean": capture["standardized_mean"],
                "standardized_log_scale": capture["standardized_log_scale"],
                "raw_support_mean": raw_support,
                "control_ensemble_mean": control_mean,
                "control_ensemble_scale": control_scale,
            }
            for name, tensor in state_tensors.items():
                state_digests[name].update(row["entity_uid"].encode())
                state_digests[name].update(b"\n")
                state_digests[name].update(tensor_sha256(tensor).encode())
            q64 = q.detach().double()
            posterior_ess.extend(
                (1.0 / q64.square().sum(dim=1)).cpu().tolist()
            )
            posterior_entropy.extend(
                (-(q64 * q64.clamp_min(1.0e-15).log()).sum(dim=1)).cpu().tolist()
            )
    finally:
        handle.remove()
    merged = arrays.concatenate()
    posterior = {
        "q_sha256": q_digest.hexdigest(),
        "entity_count": len(rows),
        "ess_min": min(posterior_ess),
        "ess_mean": float(np.mean(posterior_ess)),
        "entropy_min": min(posterior_entropy),
        "entropy_mean": float(np.mean(posterior_entropy)),
        "candidate_q_ess_entropy_exactly_invariant_by_post_q_construction": True,
        "control_state_sha256": {
            name: digest.hexdigest() for name, digest in state_digests.items()
        },
    }
    return merged, posterior


def score_arrays(
    arrays: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control = metrics(
        entity=arrays["entity"],
        family=arrays["family"],
        target=arrays["target"],
        prediction=arrays["control_prediction"],
        scale=arrays["control_scale"],
        target_weight=arrays["target_weight"],
    )
    scored = []
    non_hn = arrays["family"] != "HN"
    for index, candidate in enumerate(candidates):
        prediction = arrays["candidate_prediction"][index]
        scale = arrays["candidate_scale"][index]
        if not (
            np.array_equal(prediction[non_hn], arrays["control_prediction"][non_hn])
            and np.array_equal(scale[non_hn], arrays["control_scale"][non_hn])
        ):
            raise ValueError("candidate changed a protected non-HN prediction or scale")
        candidate_metrics = metrics(
            entity=arrays["entity"],
            family=arrays["family"],
            target=arrays["target"],
            prediction=prediction,
            scale=scale,
            target_weight=arrays["target_weight"],
        )
        active = arrays["active"][index]
        active_entities = len(np.unique(arrays["entity"][active]))
        scored.append({
            "id": candidate["id"],
            "metrics": candidate_metrics,
            "delta": metric_delta(candidate_metrics, control),
            "active_row_count": int(np.sum(active)),
            "active_entity_count": active_entities,
            "support_gate_pass": (
                int(np.sum(active)) >= MIN_ACTIVE_ROWS
                and active_entities >= MIN_ACTIVE_ENTITIES
            ),
            "protected_family_prediction_and_scale_exact": True,
        })
    return control, scored


def candidate_passes(fold_results: list[dict[str, Any]], candidate_id: str) -> bool:
    for fold_result in fold_results:
        for split in ("inner_train", "inner_dev"):
            row = next(
                value for value in fold_result[split]["candidates"]
                if value["id"] == candidate_id
            )
            delta = row["delta"]
            if (
                not row["support_gate_pass"]
                or delta["hn_entity_macro_ccc"] <= IMPROVEMENT_EPS
                or delta["five_family_macro_ccc"] <= IMPROVEMENT_EPS
                or delta["student_t_df4_nll_without_shared_constant"] >= -IMPROVEMENT_EPS
                or delta["production_entity_balanced_student_t_data_loss"]
                >= -IMPROVEMENT_EPS
                or any(value > TAIL_ATOL for value in delta["hn_tail"].values())
            ):
                return False
    return True


def ranking_score(fold_results: list[dict[str, Any]], candidate_id: str) -> tuple[float, ...]:
    dev = [
        next(
            value for value in fold_result["inner_dev"]["candidates"]
            if value["id"] == candidate_id
        )
        for fold_result in fold_results
    ]
    return (
        min(value["delta"]["hn_entity_macro_ccc"] for value in dev),
        float(np.mean([value["delta"]["five_family_macro_ccc"] for value in dev])),
        -float(np.mean([
            value["delta"]["production_entity_balanced_student_t_data_loss"]
            for value in dev
        ])),
    )


def declared_candidate(trainer: Any, candidates: list[dict[str, Any]]) -> str | None:
    declared = getattr(trainer.D2, "PHYSICS_CHART_CANDIDATE", None)
    if declared is None:
        return None
    candidate_id = declared.get("id") if isinstance(declared, dict) else None
    matches = [row for row in candidates if row["id"] == candidate_id]
    if (
        len(matches) != 1
        or declared.get("menu_sha256") != MENU_SHA256
        or float(declared.get("center_nm", -1)) != float(matches[0]["center_nm"])
        or int(declared.get("sign", 0)) != int(matches[0]["sign"])
        or float(declared.get("magnitude_ppm", -1)) != 0.005
    ):
        raise ValueError("source physics-chart declaration differs from frozen menu")
    return str(candidate_id)


def validate_baseline_receipt(
    path: Path,
    expected_sha256: str,
    *,
    input_manifest_sha256: str,
    fold_results: list[dict[str, Any]],
) -> str:
    if sha256_file(path) != expected_sha256:
        raise ValueError("baseline selector receipt file SHA256 drift")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    receipt_sha = baseline.pop("receipt_sha256", None)
    if (
        receipt_sha != canonical_sha256(baseline)
        or baseline.get("artifact_kind") != ARTIFACT_KIND
        or baseline.get("menu_sha256") != MENU_SHA256
        or baseline.get("input_manifest_sha256") != input_manifest_sha256
        or baseline.get("source_candidate_id") is not None
        or baseline.get("evaluated_model_id") != "legacy5_control"
        or baseline.get("outer_held_or_external_values_read") is not False
        or len(baseline.get("folds", ())) != 3
    ):
        raise ValueError("baseline selector receipt contract drift")
    baseline_folds = {int(row["fold"]): row for row in baseline["folds"]}
    for current in fold_results:
        prior = baseline_folds.get(int(current["fold"]))
        if prior is None:
            raise ValueError("baseline selector receipt lacks a fold")
        for split in ("inner_train", "inner_dev"):
            current_posterior = current[split]["posterior"]
            prior_posterior = prior[split]["posterior"]
            if (
                current[split]["control"] != prior[split]["control"]
                or current_posterior["q_sha256"] != prior_posterior["q_sha256"]
                or current_posterior["control_state_sha256"]
                != prior_posterior["control_state_sha256"]
                or any(
                    current_posterior[key] != prior_posterior[key]
                    for key in ("ess_min", "ess_mean", "entropy_min", "entropy_mean")
                )
            ):
                raise ValueError("candidate control state differs from frozen baseline")
    return str(receipt_sha)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.source_bundle_sha256) != 64 or len(args.container_image_sha256) != 64:
        raise ValueError("execution binding SHA256 drift")
    _, candidates = read_menu(args.menu)
    sealed_root, inputs = read_inputs(args.input_manifest, args.input_manifest_sha256)
    trainer = load_source(args.trainer, args.trainer_sha256, "v3339_inner_selector_trainer")
    if (
        trainer.D2.PHYSICS_CHART_MENU_SHA256 != MENU_SHA256
        or tuple(trainer.D2.PHYSICS_CHART_RBF_CENTERS_NM)
        != (0.2, 0.3, 0.4, 0.55, 0.75, 1.0)
        or trainer.D2.PHYSICS_CHART_RBF_WIDTH_NM != 0.12
        or trainer.D2.PHYSICS_CHART_PHE_TOKEN != 5
        or trainer.D2.PHYSICS_CHART_HN_ATOM_INDEX != 55
    ):
        raise ValueError("trainer physics-chart interface drift")
    source_candidate_id = declared_candidate(trainer, candidates)
    device = torch.device(args.device)
    if device.type == "cuda" and (
        torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
    ):
        raise ValueError("selector container must expose exactly CUDA ordinal 0")
    runtime = trainer.V6.configure_deterministic_runtime(device)
    fold_results = []
    for fold_row in sorted(inputs["folds"], key=lambda row: int(row["fold"])):
        fold = int(fold_row["fold"])
        train_manifest = safe_path(sealed_root, fold_row["train_manifest"])
        dev_manifest = safe_path(sealed_root, fold_row["dev_manifest"])
        train_rows = trainer.read_manifest(train_manifest, "inner_train", fold, 8)
        dev_rows = trainer.read_manifest(dev_manifest, "inner_dev", fold, 8)
        if {row["entity_uid"] for row in train_rows} & {row["entity_uid"] for row in dev_rows}:
            raise ValueError("inner train/dev entity overlap")
        checkpoint_path = safe_path(sealed_root, fold_row["control_checkpoint"])
        checkpoint = validate_checkpoint(
            trainer=trainer, checkpoint_path=checkpoint_path, fold_row=fold_row
        )
        trainer.V6.seed_runtime(3339, device)
        stage_a, _ = trainer.V6.load_frozen_stage_a(
            args.stage_a_source,
            trainer.STAGE_A_SOURCE_SHA256,
            safe_path(sealed_root, fold_row["stage_a_checkpoint"]),
            trainer.EXPECTED_STAGE_A_SHA256[fold],
            outer_fold=fold,
            device=device,
        )
        model, _, initialization = trainer.initialize_model(
            safe_path(sealed_root, fold_row["q_checkpoint"]),
            args.q_transfer_receipt,
            args.fixed_support_source,
            route="legacy5",
            fold=fold,
            device=device,
        )
        if initialization["classification"] != "initialization_transfer_not_lineage_continuation":
            raise ValueError("Q initialization classification drift")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        split_payload: dict[str, Any] = {"fold": fold}
        for split, rows in (("inner_train", train_rows), ("inner_dev", dev_rows)):
            arrays, posterior = collect_split(
                trainer=trainer,
                model=model,
                stage_a=stage_a,
                rows=rows,
                candidates=candidates,
                device=device,
                declared_candidate_id=source_candidate_id,
            )
            control, scored = score_arrays(arrays, candidates)
            split_payload[split] = {
                "control": control,
                "candidates": scored,
                "posterior": posterior,
            }
        fold_results.append(split_payload)
        del model, stage_a, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_receipt_sha = None
    if source_candidate_id is not None:
        if args.baseline_receipt is None or not args.baseline_receipt_sha256:
            raise ValueError("active chart candidate requires a frozen baseline receipt")
        baseline_receipt_sha = validate_baseline_receipt(
            args.baseline_receipt,
            args.baseline_receipt_sha256,
            input_manifest_sha256=args.input_manifest_sha256,
            fold_results=fold_results,
        )

    passing = sorted(
        row["id"] for row in candidates
        if candidate_passes(fold_results, row["id"])
    )
    selected_id = max(passing, key=lambda value: ranking_score(fold_results, value)) \
        if passing else None
    selector_decision = "SELECT" if selected_id is not None else "NO_CANDIDATE"
    if source_candidate_id is None:
        model_id = "legacy5_control"
        model_scores = [
            row["inner_dev"]["control"]["five_family_macro_ccc"]
            for row in fold_results
        ]
        hard_guards_pass = True
    else:
        model_id = source_candidate_id
        model_scores = [
            next(
                value["metrics"]["five_family_macro_ccc"]
                for value in row["inner_dev"]["candidates"]
                if value["id"] == source_candidate_id
            )
            for row in fold_results
        ]
        hard_guards_pass = source_candidate_id == selected_id
    selected_score = ranking_score(fold_results, selected_id) if selected_id else None
    result = {
        "artifact_kind": ARTIFACT_KIND,
        "menu_sha256": MENU_SHA256,
        "input_manifest_sha256": args.input_manifest_sha256,
        "source_bundle_sha256": args.source_bundle_sha256,
        "container_image_sha256": args.container_image_sha256,
        "trainer_sha256": args.trainer_sha256,
        "source_candidate_id": source_candidate_id,
        "evaluation_scope": "fixed_checkpoint_post_q_inference_chart_only",
        "retraining_or_training_promotion_permitted": False,
        "baseline_receipt_sha256": baseline_receipt_sha,
        "fixed_checkpoint_control_state_matches_baseline": (
            True if source_candidate_id is not None else None
        ),
        "evaluated_model_id": model_id,
        "runtime": runtime,
        "selection_contract": {
            "splits": ["inner_train", "inner_dev"],
            "outer_result_path_visible": False,
            "fold_count": 3,
            "same_candidate_and_sign_required_all_folds_and_splits": True,
            "minimum_active_rows_per_fold_split": MIN_ACTIVE_ROWS,
            "minimum_active_entities_per_fold_split": MIN_ACTIVE_ENTITIES,
            "improvement_epsilon": IMPROVEMENT_EPS,
            "tail_atol": TAIL_ATOL,
            "ranking": [
                "max_min_inner_dev_fold_HN_entity_macro_CCC_delta",
                "max_mean_inner_dev_five_family_macro_CCC_delta",
                "max_mean_inner_dev_production_entity_balanced_Student_t_improvement",
                "lexicographically_smallest_candidate_id_exact_tie_break",
            ],
        },
        "folds": fold_results,
        "passing_candidate_ids": passing,
        "passing_candidate_count": len(passing),
        "selector_decision": selector_decision,
        "selected_candidate_id": selected_id,
        "selected_ranking_score": selected_score,
        "evaluated_model_inner_dev_min_fold_five_family_macro_ccc": min(model_scores),
        "evaluated_model_inner_dev_mean_five_family_macro_ccc": float(np.mean(model_scores)),
        "hard_guards_pass": hard_guards_pass,
        "broad_promotion_permitted": False,
        "outer_held_or_external_values_read": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    pending = Path(pending_name)
    try:
        pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.chmod(0o600)
        os.replace(pending, args.output)
    finally:
        if pending.exists():
            pending.unlink()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trainer", type=Path, required=True)
    result.add_argument("--trainer-sha256", required=True)
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--input-manifest-sha256", required=True)
    result.add_argument("--menu", type=Path, required=True)
    result.add_argument("--stage-a-source", type=Path, required=True)
    result.add_argument("--q-transfer-receipt", type=Path, required=True)
    result.add_argument("--fixed-support-source", type=Path, required=True)
    result.add_argument("--source-bundle-sha256", required=True)
    result.add_argument("--container-image-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--baseline-receipt", type=Path)
    result.add_argument("--baseline-receipt-sha256", default="")
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(
        "METRIC inner_dev_minfold_five_family_ccc="
        f"{result['evaluated_model_inner_dev_min_fold_five_family_macro_ccc']:.12f}"
    )
    print(
        "METRIC inner_dev_mean_five_family_ccc="
        f"{result['evaluated_model_inner_dev_mean_five_family_macro_ccc']:.12f}"
    )
    print(f"METRIC selector_passing_candidate_count={result['passing_candidate_count']}")
    print(f"METRIC hard_guards_pass={int(result['hard_guards_pass'])}")
    print(
        f"selector_decision={result['selector_decision']} "
        f"selected_candidate_id={result['selected_candidate_id']} "
        f"evaluated_model_id={result['evaluated_model_id']}"
    )
    print(f"selector_receipt_sha256={result['receipt_sha256']}")


if __name__ == "__main__":
    main()
