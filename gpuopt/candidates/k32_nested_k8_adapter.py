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

from gpuopt.source_gate_eligibility import (
    build_eligibility_receipt,
    source_eligible_row_mask,
    validate_eligibility_receipt,
)

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
class SourceNormalization:
    cell_scale: tuple[tuple[str, str, float], ...]
    atom_scale: tuple[tuple[str, float], ...]
    element_scale: tuple[tuple[str, float], ...]
    global_scale: float


@dataclass(frozen=True)
class SourceTargets:
    normalized: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    normalization: SourceNormalization


@dataclass(frozen=True)
class AssimilationResult:
    initial_data_loss: float
    final_data_loss: float
    q: np.ndarray
    residue_delta: np.ndarray
    reference_offset: np.ndarray
    coordinate_gradient_norm: float
    prediction: np.ndarray


@dataclass(frozen=True)
class CrossfitHalf:
    fold: str
    held_half: int
    train_entities: tuple[str, ...]
    evaluation_entities: tuple[str, ...]
    train_clusters: tuple[str, ...]
    evaluation_clusters: tuple[str, ...]


@dataclass(frozen=True)
class ConsumedAuthorization:
    source_commitment_sha256: str
    authorization_sha256: str


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


def concatenate_surfaces(surfaces: tuple[Surface, ...]) -> Surface:
    if not surfaces:
        raise ValueError("surface concatenation is empty")
    roster = surfaces[0].support_ids
    nested_roster = tuple(SUPPORT_IDS[index] for index in NESTED)
    if roster not in {SUPPORT_IDS, nested_roster} or any(
        surface.support_ids != roster for surface in surfaces
    ):
        raise ValueError("surface concatenation requires one exact support roster")
    target_ids = np.concatenate([surface.target_ids for surface in surfaces])
    if len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("surface concatenation has duplicate target identities")
    if any(surface.row_context is None for surface in surfaces):
        raise ValueError("surface concatenation lacks row context")
    arrays = tuple(
        np.concatenate([surface.arrays[index] for surface in surfaces], axis=0)
        for index in range(len(surfaces[0].arrays))
    )
    context = tuple(
        np.concatenate([surface.row_context[index] for surface in surfaces])
        for index in range(3)
    )
    return Surface(target_ids, roster, arrays, context)


def subset_surface(surface: Surface, target_ids: tuple[str, ...]) -> Surface:
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("surface subset requests duplicate target identities")
    lookup = {str(value): index for index, value in enumerate(surface.target_ids)}
    if len(lookup) != len(surface.target_ids):
        raise ValueError("source surface contains duplicate target identities")
    try:
        row = np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("surface subset target is absent") from error
    context = (
        None
        if surface.row_context is None
        else tuple(value[row].copy() for value in surface.row_context)
    )
    return Surface(
        surface.target_ids[row].copy(),
        surface.support_ids,
        tuple(value[row].copy() for value in surface.arrays),
        context,
    )


def eligible_source_subset(
    frame: pd.DataFrame,
    surface: Surface,
    *,
    frozen_atom_ids: tuple[str, ...],
    fold: str,
    held_half: int,
    role: str,
) -> tuple[dict[str, Any], Surface]:
    """Apply target-only eligibility before normalization, fitting or assimilation."""
    row_mask = np.asarray(
        source_eligible_row_mask(
            frame["atom_id"].astype(str).tolist(),
            frame["target_value"].astype(float).tolist(),
            frozen_atom_ids,
        ),
        dtype=bool,
    )
    selected = frame.loc[row_mask].reset_index(drop=True)
    receipt = build_eligibility_receipt(
        frame.reset_index(drop=True),
        selected,
        frozen_atom_ids=frozen_atom_ids,
        fold=fold,
        held_half=held_half,
        role=role,
    )
    selected_surface = subset_surface(
        surface, tuple(selected["target_id"].astype(str))
    )
    return {"frame": selected, "source_eligibility_receipt": receipt}, selected_surface


def load_source_entity_targets(
    root: Path,
    surface: Surface,
    *,
    entity_uid: str,
    fold: str,
    expected_sha256: str,
    access: ConsumedAuthorization,
) -> pd.DataFrame:
    """Open one source target only after an external authorization was consumed."""
    if not isinstance(access, ConsumedAuthorization) or not all(
        len(value) == 64
        for value in (access.source_commitment_sha256, access.authorization_sha256)
    ):
        raise PermissionError("source target access requires a consumed authorization")
    if fold not in {"A", "B"} or not expected_sha256:
        raise ValueError("invalid source target binding")
    fields = entity_uid.split(":")
    if len(fields) != 4 or fields[0] != "bmrb" or fields[2] != "entity":
        raise ValueError("invalid source entity identity")
    path = root / "data/all_atom_observer_v1/targets" / f"bmr{fields[1]}.parquet"
    if sha256(path) != expected_sha256:
        raise ValueError("source target file binding mismatch")
    frame = pd.read_parquet(
        path,
        columns=(
            "entity_uid",
            "target_id",
            "seq_id",
            "comp_id",
            "atom_id",
            "target_value",
            "split",
            "observer_fold",
        ),
    )
    frame = frame[frame["entity_uid"].astype(str).eq(entity_uid)].copy()
    if frame["target_id"].duplicated().any():
        raise ValueError("duplicate source target identity")
    if set(frame["split"].astype(str)) != {"train"} or set(
        frame["observer_fold"].astype(str)
    ) != {fold}:
        raise ValueError("source target split or fold mismatch")
    frame = frame.set_index("target_id").reindex(surface.target_ids).reset_index()
    if frame.isna().any().any():
        raise ValueError("source target does not exactly cover its coordinate surface")
    return frame


def source_entity_slices(surface: Surface) -> dict[str, np.ndarray]:
    """Recover entity groups from the canonical entity-qualified target identity."""
    groups: dict[str, list[int]] = {}
    for row, target_id in enumerate(surface.target_ids.astype(str)):
        parts = target_id.split(":target:", maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("target identity is not entity-qualified")
        groups.setdefault(parts[0], []).append(row)
    return {
        entity_uid: np.asarray(rows, dtype=np.int64)
        for entity_uid, rows in sorted(groups.items())
    }


def subset_source_targets(targets: SourceTargets, rows: np.ndarray) -> SourceTargets:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("source target subset is empty or non-vector")
    if np.any(rows < 0) or np.any(rows >= len(targets.normalized)):
        raise ValueError("source target subset index is out of range")
    return SourceTargets(
        normalized=targets.normalized[rows].copy(),
        center=targets.center[rows].copy(),
        scale=targets.scale[rows].copy(),
        weight=targets.weight[rows].copy(),
        normalization=targets.normalization,
    )


def source_crossfit_plan(root: Path) -> tuple[CrossfitHalf, ...]:
    """Build fixed sequence-cluster-disjoint source halves without targets."""
    source_path = root / next(iter(HASHES))
    source = json.loads(source_path.read_text())
    parent_path = root / source["parent_commitment_relative_path"]
    if sha256(parent_path) != source["parent_commitment_sha256"]:
        raise ValueError("parent source commitment mismatch")
    parent = json.loads(parent_path.read_text())
    expected = {
        str(entity["entity_uid"]): entity
        for entity in parent["entities"]
        if entity.get("split") == "train" and entity.get("observer_fold") in {"A", "B"}
    }
    source_entities = {str(entity["entity_uid"]): entity for entity in source["entities"]}
    if source_entities.keys() != expected.keys():
        raise ValueError("K32 cache and source cohort differ")
    output = []
    for fold in ("A", "B"):
        entities = sorted(
            (entity for entity in expected.values() if entity["observer_fold"] == fold),
            key=lambda entity: str(entity["entity_uid"]),
        )
        clusters: dict[str, list[str]] = {}
        for entity in entities:
            clusters.setdefault(str(entity["sequence_cluster_id"]), []).append(
                str(entity["entity_uid"])
            )
        loads = [0, 0]
        assignment: dict[str, int] = {}
        for cluster, members in sorted(
            clusters.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            half = min(range(2), key=lambda index: (loads[index], index))
            assignment[cluster] = half
            loads[half] += len(members)
        if not all(loads):
            raise ValueError("source fold cannot form two nonempty cluster halves")
        for held_half in (0, 1):
            evaluation_clusters = tuple(
                sorted(cluster for cluster, half in assignment.items() if half == held_half)
            )
            train_clusters = tuple(
                sorted(cluster for cluster, half in assignment.items() if half != held_half)
            )
            evaluation = tuple(
                sorted(
                    str(entity["entity_uid"])
                    for entity in entities
                    if str(entity["sequence_cluster_id"]) in evaluation_clusters
                )
            )
            train = tuple(
                sorted(
                    str(entity["entity_uid"])
                    for entity in entities
                    if str(entity["sequence_cluster_id"]) in train_clusters
                )
            )
            output.append(
                CrossfitHalf(
                    fold,
                    held_half,
                    train,
                    evaluation,
                    train_clusters,
                    evaluation_clusters,
                )
            )
    return tuple(output)


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
    values: dict[str, Any],
    surface: Surface,
    anchor: np.ndarray,
    *,
    normalization: SourceNormalization | None = None,
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
    if normalization is None:
        cell_series = frame.groupby(["comp_id", "atom_id"])["target_value"].std()
        atom_series = frame.groupby("atom_id")["target_value"].std()
        element_series = frame.assign(
            element=frame["atom_id"].astype(str).str[0]
        ).groupby("element")["target_value"].std()

        def finite_rows(series: pd.Series) -> tuple[tuple[Any, ...], ...]:
            return tuple(
                (*((key,) if not isinstance(key, tuple) else key), float(value))
                for key, value in series.items()
                if pd.notna(value) and float(value) >= 0.1
            )

        normalization = SourceNormalization(
            cell_scale=finite_rows(cell_series),
            atom_scale=finite_rows(atom_series),
            element_scale=finite_rows(element_series),
            global_scale=max(float(frame["target_value"].std()), 0.1),
        )
    cell = {(comp, atom): scale for comp, atom, scale in normalization.cell_scale}
    atom = dict(normalization.atom_scale)
    element = dict(normalization.element_scale)
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
            normalization.global_scale,
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
        normalization=normalization,
    )


def fit_source_observer(
    model: DynamicDistanceObserver,
    surface: Surface,
    targets: SourceTargets,
    context: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int,
    batch_size: int = 4096,
    seed: int = 20260905,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Fit one shared K32 observer; held-fold targets are not an input."""
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("observer fit requires positive epochs and batch size")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model.to(device)
    distance = torch.as_tensor(surface.arrays[0], device=device)
    available = torch.as_tensor(surface.arrays[5], device=device)
    atom, residue, position = (
        torch.as_tensor(value, device=device) for value in context
    )
    target = torch.as_tensor(targets.normalized, device=device)
    weight = torch.as_tensor(targets.weight, device=device)

    def objective(row: torch.Tensor) -> torch.Tensor:
        prediction = model(
            distance[row], available[row], atom[row], residue[row], position[row]
        ).mean(dim=1)
        row_loss = torch.nn.functional.smooth_l1_loss(
            prediction, target[row], reduction="none"
        )
        return torch.mean(weight[row] * row_loss)

    all_rows = torch.arange(len(target), device=device)
    initial = float(objective(all_rows).detach())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, weight_decay=1.0e-4, foreach=False
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(target), generator=generator, device=device)
        model.train()
        for start in range(0, len(target), batch_size):
            loss = objective(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    final = float(objective(all_rows).detach())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return initial, final


def optimize_assimilation(
    model: DynamicDistanceObserver,
    surface: Surface,
    targets: SourceTargets,
    anchor: np.ndarray,
    context: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    mode: str,
    steps: int,
    device: torch.device | None = None,
) -> AssimilationResult:
    """Freshly optimize one entity with one q shared across every label."""
    modes = {"full", "no_coordinate", "uniform_q", "anchor_only"}
    if mode not in modes:
        raise ValueError(f"unknown assimilation mode: {mode}")
    device = device or next(model.parameters()).device
    if next(model.parameters()).device != device:
        raise ValueError("observer and assimilation device differ")
    coordinate_active = mode in {"full", "uniform_q"}
    q_active = mode in {"full", "no_coordinate"}
    residue_ids = residue_inventory(surface)
    delta_raw = nn.Parameter(
        torch.zeros((len(residue_ids), len(surface.support_ids), 4), device=device)
    )
    q_logits = nn.Parameter(torch.zeros(len(surface.support_ids), device=device))
    reference_raw = nn.Parameter(torch.zeros(3, device=device))
    parameters: list[nn.Parameter] = [reference_raw]
    if coordinate_active:
        parameters.append(delta_raw)
    if q_active:
        parameters.append(q_logits)
    optimizer = torch.optim.Adam(parameters, lr=0.08, foreach=False)
    base_distance = torch.as_tensor(surface.arrays[0], device=device)
    available = torch.as_tensor(surface.arrays[5], device=device)
    atom, residue, position = (
        torch.as_tensor(value, device=device) for value in context
    )
    target = torch.as_tensor(targets.normalized, device=device)
    weight = torch.as_tensor(targets.weight, device=device)
    scale = torch.as_tensor(targets.scale, device=device)
    anchor_deviation = torch.as_tensor(
        ((anchor - targets.center[:, None]) / targets.scale[:, None]).astype(
            np.float32
        ),
        device=device,
    )
    if surface.row_context is None:
        raise ValueError("surface has no receipted atom identities")
    element_lookup = {"H": 0, "C": 1, "N": 2}
    try:
        element = torch.as_tensor(
            [element_lookup[value[0]] for value in surface.row_context[2]],
            device=device,
        )
    except KeyError as error:
        raise ValueError("unsupported reference element") from error
    bounds = torch.tensor((0.1, 0.5, 1.0), device=device)
    with torch.no_grad():
        base = model(base_distance, available, atom, residue, position)
        base_mean = base.mean(dim=1, keepdim=True)

    def objective() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if coordinate_active:
            delta = 0.05 * torch.tanh(delta_raw)
            self_delta, neighbor_delta = gather_residue_deltas(
                surface, residue_ids, delta
            )
            distance, mask = differentiable_actuate(
                surface, self_delta, neighbor_delta
            )
            active = model(distance, mask, atom, residue, position)
        else:
            delta = torch.zeros_like(delta_raw)
            active = base
        support_prediction = 0.1 * base_mean + (active - base_mean)
        support_prediction = support_prediction + anchor_deviation
        q = (
            torch.softmax(q_logits, dim=0)
            if q_active
            else torch.full_like(q_logits, 1.0 / len(q_logits))
        )
        reference = bounds * torch.tanh(reference_raw)
        aggregate = torch.sum(q[None, :] * support_prediction, dim=1)
        aggregate = aggregate + reference[element] / scale
        data_loss = torch.mean(weight * torch.square(aggregate - target))
        regularizer = 3.0e-3 * torch.mean(torch.square(torch.tanh(reference_raw)))
        if coordinate_active:
            regularizer = regularizer + 3.0e-2 * torch.mean(
                torch.square(torch.tanh(delta_raw))
            )
        if q_active:
            regularizer = regularizer + 3.0e-3 * torch.sum(
                q * torch.log((q * len(q)).clamp_min(1.0e-12))
            )
        return data_loss + regularizer, data_loss, q, delta, aggregate

    initial = float(objective()[1].detach())
    for _ in range(steps):
        total, _, _, _, _ = objective()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    _, final_data, q, delta, aggregate = objective()
    gradient_norm = 0.0
    if coordinate_active:
        gradient = torch.autograd.grad(final_data, delta_raw, retain_graph=False)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite coordinate data gradient")
        gradient_norm = float(gradient.norm().detach())
    reference = bounds * torch.tanh(reference_raw.detach())
    return AssimilationResult(
        initial_data_loss=initial,
        final_data_loss=float(final_data.detach()),
        q=q.detach().cpu().numpy(),
        residue_delta=delta.detach().cpu().numpy(),
        reference_offset=reference.cpu().numpy(),
        coordinate_gradient_norm=gradient_norm,
        prediction=(
            targets.center + targets.scale * aggregate.detach().cpu().numpy()
        ).astype(np.float32),
    )


def matched_assimilation(
    model: DynamicDistanceObserver,
    surface: Surface,
    targets: SourceTargets,
    anchor: np.ndarray,
    context: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    mode: str,
    steps: int,
    device: torch.device | None = None,
) -> tuple[AssimilationResult, AssimilationResult]:
    """Run independent K32 and exact nested-K8 optimizers from fresh zeros."""
    if not np.all(anchor == anchor[:, :1]):
        raise ValueError("matched support-capacity gate requires invariant anchors")
    k32 = optimize_assimilation(
        model,
        surface,
        targets,
        anchor,
        context,
        mode=mode,
        steps=steps,
        device=device,
    )
    k8_surface = nested8(surface)
    k8_anchor = np.take(anchor, NESTED, axis=1)
    k8 = optimize_assimilation(
        model,
        k8_surface,
        targets,
        k8_anchor,
        context,
        mode=mode,
        steps=steps,
        device=device,
    )
    return k32, k8


def concordance(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (
        np.mean(target_centered**2)
        + np.mean(prediction_centered**2)
        + float(target.mean() - prediction.mean()) ** 2
    )
    if len(target) < 2 or denominator <= 1.0e-15:
        raise ValueError("undefined CCC input")
    return float(2.0 * np.mean(target_centered * prediction_centered) / denominator)


def macro_atom_id_ccc(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    eligible_atom_ids: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    """Score raw-ppm CCC per frozen Atom_ID, then take its unweighted macro."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("prediction surface is incomplete or nonfinite")
    atom = frame["atom_id"].astype(str).to_numpy()
    target = frame["target_value"].to_numpy(dtype=np.float64)
    per_label = {}
    for atom_id in eligible_atom_ids:
        rows = atom == atom_id
        if rows.sum() < 2 or float(np.var(target[rows])) <= 1.0e-15:
            raise ValueError(f"frozen source label has undefined target CCC: {atom_id}")
        per_label[atom_id] = concordance(target[rows], prediction[rows])
    if set(per_label) != set(eligible_atom_ids) or not per_label:
        raise ValueError("source scorer omitted a frozen Atom_ID")
    return float(np.mean(list(per_label.values()))), per_label


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
