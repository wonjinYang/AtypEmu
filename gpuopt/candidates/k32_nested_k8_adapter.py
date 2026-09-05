"""Target-unread adapter for matched K32 and nested-K8 source-gate arms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
SEQUENCE_HASHES = {
    "A": "165f34ced53aabb4b908a43716ac8f4b76e4b3990507994b4fea0aa7d4fabd58",
    "B": "97346891f5fa7e7daa41a81a73cf5c90f9bbb577f293f6d816e76ea6576553ee",
}
SEQUENCE_SUMMARY_HASHES = {
    "A": "3f64bba6c197b7318b23f1587c4f218de027e89372aec754a8d21cb876642cef",
    "B": "2419aee1fc84353749565c201275694fcc8f864d972456f276017433a27e7f38",
}
SEQUENCE_RECEIPT_HASH = "2869e7b28d08580de1046c453a6acafe04f77a45a0b1eecd6aba86ec2bd6149f"
RESIDUES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
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
    row_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


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


@dataclass(frozen=True)
class SourceTargets:
    normalized: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    weight: np.ndarray


class DynamicDistanceObserver(nn.Module):
    """Small shared observer for target-free dynamic distance channels only."""

    def __init__(self, atom_levels: int) -> None:
        super().__init__()
        self.atom = nn.Embedding(atom_levels, 16)
        self.residue = nn.Embedding(len(RESIDUES) + 1, 8)
        self.network = nn.Sequential(
            nn.Linear(35, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(
        self,
        distance: torch.Tensor,
        available: torch.Tensor,
        atom_index: torch.Tensor,
        residue_index: torch.Tensor,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        supports = distance.shape[1]
        row = torch.cat(
            (
                self.atom(atom_index),
                self.residue(residue_index),
                relative_position[:, None],
            ),
            dim=1,
        )[:, None, :].expand(-1, supports, -1)
        features = torch.cat(
            (distance / 10.0, available.to(distance.dtype), row), dim=-1
        )
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
        row_context = (
            np.array(values["seq_ids"], copy=True),
            values["comp_ids"].astype(str),
            values["atom_ids"].astype(str),
        )
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
    return Surface(target_ids, SUPPORT_IDS, arrays, row_context)


def nested8(surface: Surface) -> Surface:
    arrays = tuple(np.take(value, NESTED, axis=1) for value in surface.arrays)
    return Surface(
        surface.target_ids.copy(),
        tuple(surface.support_ids[index] for index in NESTED),
        arrays,
        surface.row_context,
    )


def context_indices(
    surface: Surface, atom_inventory: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if surface.row_context is None:
        raise ValueError("surface has no receipted row context")
    seq_ids, comp_ids, atom_ids = surface.row_context
    atom_lookup = {name: index + 1 for index, name in enumerate(atom_inventory)}
    residue_lookup = {name: index + 1 for index, name in enumerate(RESIDUES)}
    atom = np.asarray([atom_lookup.get(value, 0) for value in atom_ids], np.int64)
    residue = np.asarray(
        [residue_lookup.get(value, 0) for value in comp_ids], np.int64
    )
    position = (seq_ids - 1) / max(int(seq_ids.max()) - 1, 1)
    return atom, residue, position.astype(np.float32)


def sequence_anchor(root: Path, fold: str, surface: Surface) -> np.ndarray:
    """Load a fixed-final sequence-only anchor without target values."""
    if fold not in SEQUENCE_HASHES:
        raise ValueError("unknown source fold")
    base = root / "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0"
    prediction = base / fold / "predictions.parquet"
    summary = base / fold / "summary.json"
    if (
        sha256(prediction) != SEQUENCE_HASHES[fold]
        or sha256(summary) != SEQUENCE_SUMMARY_HASHES[fold]
        or sha256(base / "receipt.json") != SEQUENCE_RECEIPT_HASH
    ):
        raise ValueError("sequence-anchor binding mismatch")
    frame = pd.read_parquet(prediction, columns=("target_id", "prediction"))
    if frame["target_id"].duplicated().any():
        raise ValueError("duplicate sequence-anchor target identity")
    aligned = frame.set_index("target_id")["prediction"].reindex(surface.target_ids)
    values = aligned.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("sequence anchor does not cover the K32 target inventory")
    return np.repeat(values[:, None], len(surface.support_ids), axis=1)


def source_targets(
    values: dict[str, Any], surface: Surface, anchor: np.ndarray
) -> SourceTargets:
    """Normalize source targets only after exact eligibility/identity replay."""
    require_eligible(values)
    frame = values["frame"]
    if frame["target_id"].astype(str).tolist() != surface.target_ids.tolist():
        raise ValueError("source target order differs from the frozen K32 surface")
    if surface.row_context is not None:
        seq_ids, comp_ids, atom_ids = surface.row_context
        if (
            frame["seq_id"].to_numpy(dtype=np.int32).tolist() != seq_ids.tolist()
            or frame["comp_id"].astype(str).tolist() != comp_ids.tolist()
            or frame["atom_id"].astype(str).tolist() != atom_ids.tolist()
        ):
            raise ValueError("source row context differs from the frozen K32 surface")
    if anchor.shape != (len(frame), len(surface.support_ids)):
        raise ValueError("source anchor shape mismatch")
    target = frame["target_value"].to_numpy(dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError("source target is nonfinite after eligibility")
    cell = frame.groupby(["comp_id", "atom_id"])["target_value"].std()
    atom = frame.groupby("atom_id")["target_value"].std()
    element = frame.assign(
        element=frame["atom_id"].astype(str).str[0]
    ).groupby("element")["target_value"].std()
    global_scale = max(float(frame["target_value"].std()), 0.1)
    scales = []
    for comp_id, atom_id in frame[["comp_id", "atom_id"]].itertuples(index=False):
        candidates = (
            cell.get((comp_id, atom_id)),
            atom.get(atom_id),
            element.get(str(atom_id)[0]),
        )
        width = next(
            (
                float(value)
                for value in candidates
                if pd.notna(value) and float(value) >= 0.1
            ),
            global_scale,
        )
        scales.append(width)
    counts = frame.groupby("atom_id")["target_id"].transform("count").to_numpy(float)
    weight = 1.0 / np.maximum(counts, 1.0)
    weight /= weight.mean()
    center = anchor.mean(axis=1, dtype=np.float64).astype(np.float32)
    scale = np.asarray(scales, dtype=np.float32)
    return SourceTargets(
        normalized=(target - center) / scale,
        center=center,
        scale=scale,
        weight=weight.astype(np.float32),
    )


def fit_source_observer(
    model: DynamicDistanceObserver,
    surface: Surface,
    targets: SourceTargets,
    context: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int,
) -> tuple[float, float]:
    """Fit one shared K32 observer; held-fold targets are not an input."""
    distance = torch.from_numpy(surface.arrays[0])
    available = torch.from_numpy(surface.arrays[5])
    atom, residue, position = (torch.from_numpy(value) for value in context)
    target = torch.from_numpy(targets.normalized)
    weight = torch.from_numpy(targets.weight)

    def objective() -> torch.Tensor:
        prediction = model(distance, available, atom, residue, position).mean(dim=1)
        row_loss = torch.nn.functional.smooth_l1_loss(
            prediction, target, reduction="none"
        )
        return torch.mean(weight * row_loss)

    initial = float(objective().detach())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, weight_decay=1.0e-4, foreach=False
    )
    for _ in range(epochs):
        loss = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = float(objective().detach())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return initial, final


def actuate(
    surface: Surface,
    self_delta: np.ndarray,
    neighbor_delta: np.ndarray,
) -> np.ndarray:
    """Apply cached self/neighbor chi Jacobians to coordinate distances."""
    distance, self_j, neighbor_j, _, _, available = surface.arrays
    self_shape = (*distance.shape[:2], 4)
    neighbor_shape = (*distance.shape, 4)
    if self_delta.shape != self_shape or neighbor_delta.shape != neighbor_shape:
        raise ValueError("torsion delta shape mismatch")
    change = np.einsum("tkec,tkc->tke", self_j, self_delta)
    change += np.einsum("tkec,tkec->tke", neighbor_j, neighbor_delta)
    return distance + np.where(available, change, 0.0)


def differentiable_actuate(
    surface: Surface,
    self_delta: torch.Tensor,
    neighbor_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch actuation preserving CS-loss gradients to coordinate deltas."""
    distance, self_j, neighbor_j, _, _, available = surface.arrays
    self_shape = (*distance.shape[:2], 4)
    neighbor_shape = (*distance.shape, 4)
    if (
        tuple(self_delta.shape) != self_shape
        or tuple(neighbor_delta.shape) != neighbor_shape
    ):
        raise ValueError("torsion delta shape mismatch")
    device = self_delta.device
    base = torch.as_tensor(distance, device=device)
    mask = torch.as_tensor(available, device=device)
    self_tensor = torch.as_tensor(self_j, device=device)
    neighbor_tensor = torch.as_tensor(neighbor_j, device=device)
    change = torch.einsum("tkec,tkc->tke", self_tensor, self_delta)
    change = change + torch.einsum(
        "tkec,tkec->tke", neighbor_tensor, neighbor_delta
    )
    return base + torch.where(mask, change, torch.zeros_like(change)), mask


def residue_inventory(surface: Surface) -> tuple[int, ...]:
    """Return every self or nearest-neighbor residue touched by the cache."""
    if surface.row_context is None:
        raise ValueError("surface has no receipted residue identities")
    seq_ids = surface.row_context[0]
    neighbor_seq_ids = surface.arrays[3]
    return tuple(
        sorted(
            set(int(value) for value in seq_ids)
            | set(int(value) for value in neighbor_seq_ids.ravel() if value >= 0)
        )
    )


def gather_residue_deltas(
    surface: Surface,
    residue_ids: tuple[int, ...],
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map one residue-shared delta to self and element-specific neighbors."""
    expected = (len(residue_ids), len(surface.support_ids), 4)
    if tuple(delta.shape) != expected or len(set(residue_ids)) != len(residue_ids):
        raise ValueError("residue delta inventory mismatch")
    if surface.row_context is None:
        raise ValueError("surface has no receipted residue identities")
    lookup = {seq_id: index for index, seq_id in enumerate(residue_ids)}
    self_ids = surface.row_context[0]
    neighbor_ids = surface.arrays[3]
    if any(int(value) not in lookup for value in self_ids):
        raise ValueError("self residue is absent from delta inventory")
    missing = sorted(
        {
            int(value)
            for value in neighbor_ids.ravel()
            if value >= 0 and int(value) not in lookup
        }
    )
    if missing:
        raise ValueError(f"neighbor residues absent from delta inventory: {missing}")
    self_index = torch.as_tensor(
        [lookup[int(value)] for value in self_ids], device=delta.device
    )
    sentinel = len(residue_ids)
    neighbor_index = torch.as_tensor(
        [
            lookup[int(value)] if value >= 0 else sentinel
            for value in neighbor_ids.ravel()
        ],
        device=delta.device,
    ).reshape(neighbor_ids.shape)
    support_index = torch.arange(
        len(surface.support_ids), device=delta.device
    )[None, :, None]
    padded = torch.cat((delta, torch.zeros_like(delta[:1])), dim=0)
    return delta[self_index], padded[neighbor_index, support_index]


def require_eligible(values: dict[str, Any]) -> None:
    receipt = values.get("source_eligibility_receipt")
    if not isinstance(receipt, dict) or "frame" not in values:
        raise ValueError("eligibility must precede normalization and fitting")
    validate_eligibility_receipt(values["frame"], receipt)


def fresh_state(residues: int, supports: int) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((residues, supports, 4), np.float32),
        np.zeros((1, supports), np.float32),
        np.zeros((1, 3), np.float32),
    )


def matched_arms(surface: Surface, observer_state: np.ndarray) -> tuple[Arm, Arm]:
    if len(observer_state) != len(surface.target_ids):
        raise ValueError("observer state and target order differ")
    small = nested8(surface)
    residue_count = len(residue_inventory(surface))
    return (
        Arm(surface, observer_state, fresh_state(residue_count, 32)),
        Arm(small, observer_state, fresh_state(residue_count, 8)),
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
