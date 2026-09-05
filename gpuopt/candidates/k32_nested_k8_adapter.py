"""Target-unread adapter for matched K32 and nested-K8 source-gate arms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTS = (1, 32, 63, 94, 126, 157, 188, 221, 251, 281, 312, 344, 376, 407, 438, 469, 501, 533, 565, 595, 626, 656, 687, 719, 751, 781, 811, 843, 876, 906, 937, 968)
SUPPORT_IDS = tuple(f"BioEmu_{value}" for value in SUPPORTS)
NESTED = (0, 4, 8, 12, 16, 20, 24, 28)
HASHES = {
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json": "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json": "01f8d7c6ca18cfdded4a79ea7b23679b72ba28d6b15b772beb15cd0c370943d8",
    "gpuopt/check_k32_dynamic_distance_cache.py": "59b2a3b82c8d7937a34f661ead3a88d1acc13b6ce25505abf6cf762c0e02f21e",
    ".auto/runs/k32_dynamic_distance_cache_independent_check_v1r1.json": "54f16dd0d518aec9f3618df5c1f3240da91a1886d340161d76dfc4084f83eab2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Surface:
    target_ids: np.ndarray
    support_ids: tuple[str, ...]
    arrays: tuple[np.ndarray, ...]


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
    path = root / row["output_relative_path"]
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
        if values["entity_uid"].astype(str).tolist() != [entity_uid]:
            raise ValueError("cache entity mismatch")
        if tuple(values["support_ids"].astype(str)) != SUPPORT_IDS:
            raise ValueError("cache support mismatch")
        target_ids = values["target_ids"].astype(str)
        arrays = tuple(np.array(values[name], copy=True) for name in names)
    if any(array.shape[1] != 32 for array in arrays):
        raise ValueError("cache support axis mismatch")
    return Surface(target_ids, SUPPORT_IDS, arrays)


def nested8(surface: Surface) -> Surface:
    arrays = tuple(np.take(value, NESTED, axis=1) for value in surface.arrays)
    return Surface(
        surface.target_ids.copy(),
        tuple(surface.support_ids[index] for index in NESTED),
        arrays,
    )


def fresh_state(rows: int, supports: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((rows, supports, 4), np.float32), np.zeros((1, supports), np.float32)
