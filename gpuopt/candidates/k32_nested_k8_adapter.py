"""Target-unread adapter for matched K32 and nested-K8 source-gate arms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gpuopt.source_gate_eligibility import validate_eligibility_receipt

SUPPORTS = (1, 32, 63, 94, 126, 157, 188, 221, 251, 281, 312, 344, 376, 407, 438, 469, 501, 533, 565, 595, 626, 656, 687, 719, 751, 781, 811, 843, 876, 906, 937, 968)
SUPPORT_IDS = tuple(f"BioEmu_{value}" for value in SUPPORTS)
NESTED = (0, 4, 8, 12, 16, 20, 24, 28)
HASHES = {
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json": "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json": "01f8d7c6ca18cfdded4a79ea7b23679b72ba28d6b15b772beb15cd0c370943d8",
    "gpuopt/check_k32_dynamic_distance_cache.py": "59b2a3b82c8d7937a34f661ead3a88d1acc13b6ce25505abf6cf762c0e02f21e",
    ".auto/runs/k32_dynamic_distance_cache_independent_check_v1r1.json": "54f16dd0d518aec9f3618df5c1f3240da91a1886d340161d76dfc4084f83eab2",
}
ARRAY_KEYS = {
    "entity_uid", "bmrb_id", "target_ids", "support_ids", "support_indices",
    "seq_ids", "comp_ids", "atom_ids", "element_order",
    "nearest_interresidue_distances_angstrom", "distance_self_jacobian",
    "distance_neighbor_jacobian", "distance_neighbor_seq_ids",
    "target_atom_available", "distance_available",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode() + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Surface:
    target_ids: np.ndarray
    support_ids: tuple[str, ...]
    arrays: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Arm:
    surface: Surface
    observer_state: np.ndarray
    assimilation_state: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ModelInputs:
    surface: Surface
    observer_state: np.ndarray
    torsion: np.ndarray
    geometry: np.ndarray
    support_anchor: np.ndarray


class DynamicDistanceObserver(nn.Module):
    """Small shared observer for target-free dynamic distance channels only."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, distance: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((distance / 10.0, available.to(distance.dtype)), dim=-1)
        return self.network(features).squeeze(-1)


def load(root: Path, entity_uid: str) -> Surface:
    """Load frozen identities/geometry only, never chemical-shift targets."""
    root = root.resolve()
    for relative, expected in HASHES.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"frozen cache binding mismatch: {relative}")
    check = json.loads(
        (root / ".auto/runs/k32_dynamic_distance_cache_independent_check_v1r1.json").read_text()
    )
    required = {
        "passed": True,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "fold_scope_verified_entities": 135,
        "target_support_row_count": 4073120,
    }
    if any(check.get(key) != value for key, value in required.items()):
        raise ValueError("cache qualification mismatch")
    receipt = json.loads(
        (root / ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json").read_text()
    )
    rows = [row for row in receipt["outputs"] if row["entity_uid"] == entity_uid]
    if len(rows) != 1:
        raise ValueError(f"cache entity is not unique: {entity_uid}")
    row = rows[0]
    path = (root / row["output_relative_path"]).resolve()
    if path.parent != root / "data/k32_dynamic_distance_cache_v1":
        raise ValueError("cache path mismatch")
    if sha256(path) != row["output_sha256"]:
        raise ValueError("cache output hash mismatch")
    names = (
        "nearest_interresidue_distances_angstrom",
        "distance_self_jacobian",
        "distance_neighbor_jacobian",
        "distance_neighbor_seq_ids",
        "target_atom_available",
        "distance_available",
    )
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != ARRAY_KEYS:
            raise ValueError("cache array inventory mismatch")
        if values["entity_uid"].astype(str).tolist() != [entity_uid]:
            raise ValueError("cache entity mismatch")
        if values["bmrb_id"].astype(str).tolist() != [row["bmrb_id"]]:
            raise ValueError("cache BMRB identity mismatch")
        if (
            tuple(values["support_ids"].astype(str)) != SUPPORT_IDS
            or values["support_indices"].dtype != np.int16
            or tuple(values["support_indices"].tolist()) != SUPPORTS
            or values["element_order"].astype(str).tolist() != ["H", "C", "N", "O", "S"]
        ):
            raise ValueError("cache support mismatch")
        target_ids = values["target_ids"].astype(str)
        if ids_sha256(target_ids) != row["target_ids_sha256"]:
            raise ValueError("cache target identity mismatch")
        arrays = tuple(np.array(values[name], copy=True) for name in names)
    if any(array.shape[1] != 32 for array in arrays):
        raise ValueError("cache support axis mismatch")
    if [array.dtype for array in arrays] != [
        np.float32, np.float32, np.float32, np.int32, np.bool_, np.bool_
    ]:
        raise ValueError("cache array dtype mismatch")
    if row["target_count"] != len(target_ids) or any(
        list(array.shape) != row["array_shapes"][name]
        for name, array in zip(names, arrays, strict=True)
    ):
        raise ValueError("cache array shape mismatch")
    distance, self_j, neighbor_j, neighbor_seq, atom_available, available = arrays
    missing = ~available
    if (
        np.any(available & ~atom_available[:, :, None])
        or not np.all(distance[available] > 0)
        or not np.all(distance[missing] == 10)
        or not np.all(self_j[missing] == 0)
        or not np.all(neighbor_j[missing] == 0)
        or not np.all(neighbor_seq[missing] == -1)
        or int(atom_available.sum()) != row["target_available_count"]
        or int((~atom_available).sum()) != row["target_missing_count"]
    ):
        raise ValueError("cache availability invariant mismatch")
    return Surface(target_ids, SUPPORT_IDS, arrays)


def nested8(surface: Surface) -> Surface:
    arrays = tuple(np.take(value, NESTED, axis=1) for value in surface.arrays)
    return Surface(
        surface.target_ids.copy(),
        tuple(surface.support_ids[index] for index in NESTED),
        arrays,
    )


def actuate(
    surface: Surface,
    self_delta: np.ndarray,
    neighbor_delta: np.ndarray,
) -> np.ndarray:
    """Apply cached self/neighbor chi Jacobians to coordinate distances."""
    distance, self_j, neighbor_j, _, _, available = surface.arrays
    expected = (*distance.shape[:2], 4)
    if self_delta.shape != expected or neighbor_delta.shape != expected:
        raise ValueError("torsion delta shape mismatch")
    change = np.einsum("tkec,tkc->tke", self_j, self_delta)
    change += np.einsum("tkec,tkc->tke", neighbor_j, neighbor_delta)
    return distance + np.where(available, change, 0.0)


def differentiable_actuate(
    surface: Surface,
    self_delta: torch.Tensor,
    neighbor_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch actuation preserving CS-loss gradients to coordinate deltas."""
    distance, self_j, neighbor_j, _, _, available = surface.arrays
    expected = (*distance.shape[:2], 4)
    if tuple(self_delta.shape) != expected or tuple(neighbor_delta.shape) != expected:
        raise ValueError("torsion delta shape mismatch")
    device = self_delta.device
    base = torch.as_tensor(distance, device=device)
    mask = torch.as_tensor(available, device=device)
    self_tensor = torch.as_tensor(self_j, device=device)
    neighbor_tensor = torch.as_tensor(neighbor_j, device=device)
    change = torch.einsum("tkec,tkc->tke", self_tensor, self_delta)
    change = change + torch.einsum(
        "tkec,tkc->tke", neighbor_tensor, neighbor_delta
    )
    return base + torch.where(mask, change, torch.zeros_like(change)), mask


def require_eligible(values: dict[str, Any]) -> None:
    receipt = values.get("source_eligibility_receipt")
    if not isinstance(receipt, dict) or "frame" not in values:
        raise ValueError("eligibility must precede normalization and fitting")
    validate_eligibility_receipt(values["frame"], receipt)


def fresh_state(rows: int, supports: int) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((rows, supports, 4), np.float32),
        np.zeros((1, supports), np.float32),
        np.zeros((1, 3), np.float32),
    )


def matched_arms(surface: Surface, observer_state: np.ndarray) -> tuple[Arm, Arm]:
    if len(observer_state) != len(surface.target_ids):
        raise ValueError("observer state and target order differ")
    small = nested8(surface)
    return (
        Arm(surface, observer_state, fresh_state(len(surface.target_ids), 32)),
        Arm(small, observer_state, fresh_state(len(surface.target_ids), 8)),
    )


def bind_model_inputs(
    surface: Surface,
    *,
    support_ids: tuple[str, ...],
    observer_state: np.ndarray,
    torsion: np.ndarray,
    geometry: np.ndarray,
    dynamic_distance: np.ndarray,
    support_anchor: np.ndarray,
) -> ModelInputs:
    """Reject an arm unless every support-dependent model input is complete."""
    rows, supports = len(surface.target_ids), len(surface.support_ids)
    if support_ids != surface.support_ids:
        raise ValueError("base-model support roster differs from coordinate cache")
    if len(observer_state) != rows:
        raise ValueError("observer state and target order differ")
    expected_prefix = (rows, supports)
    if (
        torsion.shape[:2] != expected_prefix
        or geometry.shape[:2] != expected_prefix
        or dynamic_distance.shape != surface.arrays[0].shape
        or support_anchor.shape != expected_prefix
    ):
        raise ValueError("base-model support surface is incomplete")
    if not np.array_equal(dynamic_distance, surface.arrays[0]):
        raise ValueError("base geometry is not aligned to emitted-coordinate distances")
    if not all(
        np.isfinite(value).all()
        for value in (observer_state, torsion, geometry, support_anchor)
    ):
        raise ValueError("base-model support surface is nonfinite")
    return ModelInputs(surface, observer_state, torsion, geometry, support_anchor)


def nested_model_inputs(values: ModelInputs) -> ModelInputs:
    small = nested8(values.surface)
    return ModelInputs(
        small,
        values.observer_state,
        np.take(values.torsion, NESTED, axis=1),
        np.take(values.geometry, NESTED, axis=1),
        np.take(values.support_anchor, NESTED, axis=1),
    )
