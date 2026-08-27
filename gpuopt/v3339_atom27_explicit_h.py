#!/usr/bin/env python3
"""Target-blind explicit-H atom27 extension of frozen v3339 Stage B/C.

The first fourteen slots preserve every populated OpenFold atom14 heavy slot.
Literal hydrogens occupy only unused slots 5..13 and the thirteen appended
slots 14..26, so the N/CA/C/O/CB anchor convention is unchanged.  Hydrogen
identity is exact except for the declared BMRB alias HN -> H.  Missing
protonation/tautomer sites remain masked; there is no heavy-parent fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


ARTIFACT_KIND = "v3339_explicit_h_atom27_fixed_support_v2"
COORDINATE_SCOPE = (
    "literal_explicit_h_atom27_from_hash_pinned_cached_af_pdbs_"
    "mixed_openmm_provenance_no_uniform_protonation_claim"
)
ATOM_COUNT = 27
ROLE_COUNT = 7
ROLE_NAMES = ("H", "C", "N", "O", "S", "donor", "acceptor")
PHYSICS_CHART_MENU_SHA256 = "3d120deccf219f517938467bd0e18fb4f2fffe1d9444b09250c6f908e4cac42f"
PHYSICS_CHART_CANDIDATE: dict[str, Any] | None = None
PHYSICS_CHART_RBF_CENTERS_NM = (0.20, 0.30, 0.40, 0.55, 0.75, 1.00)
PHYSICS_CHART_RBF_WIDTH_NM = 0.12
PHYSICS_CHART_PHE_TOKEN = 5
PHYSICS_CHART_HN_ATOM_INDEX = 55
PINNED_V6_SUPPORT_SHA256 = "5f40de7904768595508b19fd3f077d9fb55bd9879a129e26725f19473a579a0f"
PINNED_V6_MODEL_SHA256 = "d8348a88740171939d0fabad3430b1f8adf2813c86238c7d12611cf9c79d2d56"
PARENT_V6_MODEL_SHA256 = "5025ac301e5e4444382a09ba4175b775967a775b9445f2f6dd23cbb1dbce3116"
PARENT_D1_ATOM27_SOURCE_SHA256 = "f6d1959d5bb1f77b9abf7a55169546bf20afce4469885ec52f0fa40d34297fc6"
STAGE_C_V6_FAILURE_RECEIPT_SHA256 = "6a441bd7bc79e05917a91a508e36106d31baba620e64e0d89dc8ca470f684bcc"
PINNED_SUPPORT_MANIFEST_SHA256 = "5ab35bd59851c7a3171b4c898a756abdcd6a45e851c7092cb77f5d9db0aeb737"
PINNED_SUPPORT_ROSTER_SHA256 = "a54e5daf8f49cd11d84a3412ba3a9b361db8e18a487a75cecc00904deb29cb17"
PINNED_COORDINATE_RECEIPTS_SHA256 = "a2a3f8161a992abf080d829fdda6bb0886c5164faf0ef5608199e7f028343db9"
PINNED_SCORING_SURFACE_SHA256 = "f9dc2b7881715dba380ce128179557679e7d89acaabb81d194e7cb909f0f4969"
PINNED_TARGET_IDENTITY_SHA256 = "d58b98fbf805ce4e8c2244fa605db2fb7576d98342117bd4a10faec6d2d51e6f"
EXPECTED_BASE_SUPPORT_COUNT = 1_196
EXPECTED_ROTAMER_SUPPORT_COUNT = 6_132
EXPECTED_TOTAL_SUPPORT_COUNT = 7_328
EXPECTED_SUPPORT_KEY_COUNT = 229
EXPECTED_SUPPORTED_ENTITY_COUNT = 235
EXPECTED_SELECTED_RESIDUE_COUNT = 119_668
EXPECTED_SELECTED_HEAVY_RECORD_COUNT = 908_968
EXPECTED_SELECTED_H_RECORD_COUNT = 881_540
EXPECTED_LITERAL_H_NAME_COUNT = 45
EXPECTED_OBSERVED_LITERAL_COMP_H_PAIR_COUNT = 192
EXPECTED_ATOM27_COMP_H_PAIR_COUNT = 194
EXPECTED_MAX_SIMULTANEOUS_ATOMS = 26
EXPECTED_MAX_UNION_ATOMS = 27
EXPECTED_SCORING_ROW_COUNT = 54_632
EXPECTED_SCORING_H_ROW_COUNT = 20_409
EXPECTED_PRIMARY_H_ROW_COUNT = 18_746
EXPECTED_DIRECT_LITERAL_SCORING_H_ROW_COUNT = 20_407
EXPECTED_PROTONATION_GAP_SCORING_H_ROW_COUNT = 2
ATOM27_HARD_GATE_SPECS = (
    ("minimum_nonbonded_radius_ratio", "minimum", 0.35),
    ("minimum_same_residue_distinct_atom_distance_nm", "minimum", 0.07),
    ("maximum_declared_bond_range_violation_nm", "maximum", 0.005),
    ("maximum_declared_bond_length_change_nm", "maximum", 5.0e-4),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sibling(filename: str, expected_sha256: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"pinned v6 dependency SHA256 drift: {filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned v6 dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V6_SUPPORT = _load_sibling(
    "v3339_atom14_support.py", PINNED_V6_SUPPORT_SHA256, "v3339_pinned_atom14_support"
)
V6_MODEL = _load_sibling(
    "v3339_all_atom_e2e.py", PINNED_V6_MODEL_SHA256, "v3339_pinned_all_atom_e2e"
)
AA3_TO_AA1 = V6_SUPPORT.AA3_TO_AA1


# Exact literal H names observed in the frozen 1,196-PDB selected AF cohort.
# Each tuple groups atoms by their covalently bonded heavy parent.
HYDROGEN_GROUPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "ALA": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB1", "HB2", "HB3"))),
    "ARG": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("CD", ("HD2", "HD3")), ("NE", ("HE",)), ("NH1", ("HH11", "HH12")), ("NH2", ("HH21", "HH22"))),
    "ASN": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("ND2", ("HD21", "HD22"))),
    "ASP": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("OD2", ("HD2",))),
    "CYS": (("N", ("H",)), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("SG", ("HG",))),
    "GLN": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("NE2", ("HE21", "HE22"))),
    "GLU": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("OE2", ("HE2",))),
    "GLY": (("N", ("H", "H2", "H3")), ("CA", ("HA2", "HA3"))),
    "HIS": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("ND1", ("HD1",)), ("CD2", ("HD2",)), ("CE1", ("HE1",)), ("NE2", ("HE2",))),
    "ILE": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB",)), ("CG1", ("HG12", "HG13")), ("CG2", ("HG21", "HG22", "HG23")), ("CD1", ("HD11", "HD12", "HD13"))),
    "LEU": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG",)), ("CD1", ("HD11", "HD12", "HD13")), ("CD2", ("HD21", "HD22", "HD23"))),
    "LYS": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("CD", ("HD2", "HD3")), ("CE", ("HE2", "HE3")), ("NZ", ("HZ1", "HZ2", "HZ3"))),
    "MET": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("CE", ("HE1", "HE2", "HE3"))),
    "PHE": (("N", ("H",)), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CD1", ("HD1",)), ("CD2", ("HD2",)), ("CE1", ("HE1",)), ("CE2", ("HE2",)), ("CZ", ("HZ",))),
    "PRO": (("N", ("H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CG", ("HG2", "HG3")), ("CD", ("HD2", "HD3"))),
    "SER": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("OG", ("HG",))),
    "THR": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB",)), ("OG1", ("HG1",)), ("CG2", ("HG21", "HG22", "HG23"))),
    "TRP": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CD1", ("HD1",)), ("NE1", ("HE1",)), ("CE3", ("HE3",)), ("CZ2", ("HZ2",)), ("CZ3", ("HZ3",)), ("CH2", ("HH2",))),
    "TYR": (("N", ("H", "H2", "H3")), ("CA", ("HA",)), ("CB", ("HB2", "HB3")), ("CD1", ("HD1",)), ("CD2", ("HD2",)), ("CE1", ("HE1",)), ("CE2", ("HE2",)), ("OH", ("HH",))),
    "VAL": (("N", ("H",)), ("CA", ("HA",)), ("CB", ("HB",)), ("CG1", ("HG11", "HG12", "HG13")), ("CG2", ("HG21", "HG22", "HG23"))),
}

METHYL_GROUP_ATOMS = {
    ("ALA", "HB"): ("HB1", "HB2", "HB3"),
    ("ILE", "HD1"): ("HD11", "HD12", "HD13"),
    ("ILE", "HG2"): ("HG21", "HG22", "HG23"),
    ("LEU", "HD1"): ("HD11", "HD12", "HD13"),
    ("LEU", "HD2"): ("HD21", "HD22", "HD23"),
    ("MET", "HE"): ("HE1", "HE2", "HE3"),
    ("THR", "HG2"): ("HG21", "HG22", "HG23"),
    ("VAL", "HG1"): ("HG11", "HG12", "HG13"),
    ("VAL", "HG2"): ("HG21", "HG22", "HG23"),
}


@lru_cache(maxsize=1)
def atom27_tables() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[int, ...]]]:
    heavy, _ = V6_SUPPORT.atom14_tables()
    names: dict[str, tuple[str, ...]] = {}
    parents: dict[str, tuple[int, ...]] = {}
    for comp in AA3_TO_AA1:
        row = list(heavy[comp]) + [""] * (ATOM_COUNT - 14)
        # Slot 4 remains the historical CB anchor (blank for GLY).
        free = [slot for slot in range(5, ATOM_COUNT) if not row[slot]]
        parent_by_h = {
            atom: parent
            for parent, atoms in HYDROGEN_GROUPS[comp]
            for atom in atoms
        }
        for atom, slot in zip(parent_by_h, free[: len(parent_by_h)], strict=True):
            row[slot] = atom
        if len(parent_by_h) > len(free):
            raise RuntimeError(f"atom27 capacity exceeded for {comp}")
        oxt_slot = free[len(parent_by_h)]
        row[oxt_slot] = "OXT"
        slot_by_name = {atom: slot for slot, atom in enumerate(row) if atom}
        parent_slots = [-1] * ATOM_COUNT
        for atom, parent in parent_by_h.items():
            parent_slots[slot_by_name[atom]] = slot_by_name[parent]
        parent_slots[oxt_slot] = slot_by_name["C"]
        names[comp] = tuple(row)
        parents[comp] = tuple(parent_slots)
    if (
        sum(len(atoms) for groups in HYDROGEN_GROUPS.values() for _, atoms in groups)
        != EXPECTED_ATOM27_COMP_H_PAIR_COUNT
        or max(sum(bool(atom) for atom in row) for row in names.values())
        != EXPECTED_MAX_UNION_ATOMS
        or any(
            any(atom and row[slot] != atom for slot, atom in enumerate(heavy[comp]))
            for comp, row in names.items()
        )
    ):
        raise RuntimeError("atom27 literal roster contract drift")
    return names, parents


def resolve_structure_slot(comp: str, atom: str, nucleus: str) -> int:
    """Resolve an exact literal atom; the sole H alias is HN -> H."""

    names, _ = atom27_tables()
    if comp not in names:
        return -1
    label = {"C'": "C", "CO": "C"}.get(atom, atom)
    if nucleus == "H":
        label = "H" if atom == "HN" else atom
    return names[comp].index(label) if label in names[comp] else -1


def resolve_parent_slot(comp: str, atom: str, nucleus: str) -> int:
    slot = resolve_structure_slot(comp, atom, nucleus)
    if slot < 0:
        return -1
    _, parents = atom27_tables()
    if nucleus == "H":
        return parents[comp][slot]
    return slot


def equivalent_structure_slots(
    comp: str, atom: str, nucleus: str, methyl_equivalence_group: str = ""
) -> tuple[int, int, int]:
    exact = resolve_structure_slot(comp, atom, nucleus)
    if exact < 0:
        return (-1, -1, -1)
    group = methyl_equivalence_group.strip()
    if not group:
        return (exact, -1, -1)
    members = METHYL_GROUP_ATOMS.get((comp, group))
    if members is None or atom not in members:
        raise ValueError(f"invalid methyl-equivalence annotation: {comp}|{atom}|{group}")
    slots = tuple(resolve_structure_slot(comp, member, "H") for member in members)
    if len(slots) != 3 or any(slot < 0 for slot in slots):
        raise RuntimeError("methyl-equivalence slots are incomplete")
    return slots


def _pdb_element(line: str) -> str:
    if len(line) < 78:
        raise ValueError("PDB atom record lacks the element column")
    element = line[76:78].strip().upper()
    if element not in {"H", "C", "N", "O", "S"}:
        raise ValueError(f"unsupported or absent PDB element: {element!r}")
    return element


def parse_pdb_atom27(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse the unique exact-sequence chain with literal element-column H."""

    names, _ = atom27_tables()
    chains: dict[str, dict[tuple[str, str, str], dict[str, tuple[str, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    order: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54 or line[16] not in {" ", "A"}:
                continue
            atom = line[12:16].strip().upper()
            comp = line[17:20].strip().upper()
            if comp not in AA3_TO_AA1:
                continue
            element = _pdb_element(line)
            if (element == "H") != atom.startswith("H"):
                raise ValueError(f"PDB atom/element disagreement: {path}:{comp}:{atom}:{element}")
            if atom not in names[comp]:
                if element == "H":
                    raise ValueError(f"unregistered literal H name: {path}:{comp}:{atom}")
                continue
            chain = line[21].strip() or "_"
            residue_key = (line[22:26].strip(), line[26].strip(), comp)
            if residue_key not in chains[chain]:
                order[chain].append(residue_key)
            try:
                xyz = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                ) / 10.0
            except ValueError as error:
                raise ValueError(f"invalid PDB coordinate: {path}:{comp}:{atom}") from error
            chains[chain][residue_key][atom] = (element, xyz)
    matches = [
        chain for chain, keys in order.items()
        if "".join(AA3_TO_AA1[key[2]] for key in keys) == sequence
    ]
    if len(matches) != 1:
        raise ValueError(f"PDB must contain exactly one exact sequence chain: {path}")
    chain = matches[0]
    positions = np.zeros((len(sequence), ATOM_COUNT, 3), dtype=np.float32)
    mask = np.zeros((len(sequence), ATOM_COUNT), dtype=np.bool_)
    comps: list[str] = []
    heavy_names, _ = V6_SUPPORT.atom14_tables()
    for residue, key in enumerate(order[chain]):
        comp = key[2]
        comps.append(comp)
        slot_by_name = {atom: slot for slot, atom in enumerate(names[comp]) if atom}
        for atom, (_, xyz) in chains[chain][key].items():
            slot = slot_by_name[atom]
            positions[residue, slot] = xyz
            mask[residue, slot] = True
        expected_heavy = np.asarray([bool(atom) for atom in heavy_names[comp]], dtype=np.bool_)
        if not np.array_equal(mask[residue, :14] & expected_heavy, expected_heavy):
            raise ValueError(f"PDB is missing an expected heavy atom14 coordinate: {path}")
    result_comps = tuple(comps)
    validate_atom27_state(positions, mask, result_comps)
    return positions, mask, result_comps


def validate_atom27_state(
    positions: np.ndarray, mask: np.ndarray, comps: tuple[str, ...]
) -> dict[str, float | int]:
    names, parents = atom27_tables()
    if positions.shape != (len(comps), ATOM_COUNT, 3) or mask.shape != (len(comps), ATOM_COUNT):
        raise ValueError("atom27 state shape mismatch")
    if not np.isfinite(positions[mask]).all():
        raise ValueError("atom27 state contains nonfinite coordinates")
    distances: list[float] = []
    for residue, comp in enumerate(comps):
        for slot, parent in enumerate(parents[comp]):
            if parent < 0 or not mask[residue, slot] or not names[comp][slot].startswith("H"):
                continue
            if not mask[residue, parent]:
                raise ValueError("literal H lacks its declared heavy parent")
            distance = float(np.linalg.norm(positions[residue, slot] - positions[residue, parent]))
            if not 0.08 <= distance <= 0.145:
                raise ValueError(f"X-H bond outside 0.08..0.145 nm: {comp}|{names[comp][slot]}={distance}")
            distances.append(distance)
    if not distances:
        raise ValueError("atom27 state has no literal hydrogen")
    return {
        "literal_h_count": len(distances),
        "minimum_xh_bond_nm": min(distances),
        "maximum_xh_bond_nm": max(distances),
    }


def atom27_bond_adjacency(comps: tuple[str, ...]) -> np.ndarray:
    """Return full same-residue heavy-heavy and H-parent covalent topology."""

    names, parents = atom27_tables()
    heavy_names, _ = V6_SUPPORT.atom14_tables()
    heavy_adjacency = V6_SUPPORT.atom14_bond_adjacency(comps)
    result = np.zeros((len(comps), ATOM_COUNT, ATOM_COUNT), dtype=np.bool_)
    result[:, :14, :14] = heavy_adjacency
    for residue, comp in enumerate(comps):
        if any(
            atom and names[comp][slot] != atom
            for slot, atom in enumerate(heavy_names[comp])
        ):
            raise RuntimeError("atom27 heavy-slot lineage drift")
        for slot, parent in enumerate(parents[comp]):
            if parent >= 0:
                result[residue, slot, parent] = True
                result[residue, parent, slot] = True
    if np.any(np.diagonal(result, axis1=-2, axis2=-1)) or not np.array_equal(
        result, result.transpose(0, 2, 1)
    ):
        raise RuntimeError("atom27 adjacency must be symmetric without self edges")
    return result


def atom27_role_features(
    comps: tuple[str, ...], mask: np.ndarray
) -> np.ndarray:
    """State-specific H/C/N/O/S, donor, acceptor roles.

    Donor and acceptor flags reflect the literal cached state.  This avoids
    presenting a missing titratable H as if a uniform protonation assignment
    had been made.
    """

    names, parents = atom27_tables()
    if mask.shape != (len(comps), ATOM_COUNT):
        raise ValueError("atom27 role mask shape mismatch")
    result = np.zeros((len(comps), ATOM_COUNT, ROLE_COUNT), dtype=np.bool_)
    acceptor_n = {("HIS", "ND1"), ("HIS", "NE2")}
    for residue, comp in enumerate(comps):
        child_by_parent: dict[int, list[int]] = defaultdict(list)
        for slot, parent in enumerate(parents[comp]):
            if parent >= 0 and names[comp][slot].startswith("H"):
                child_by_parent[parent].append(slot)
        for slot, atom in enumerate(names[comp]):
            if not atom:
                continue
            element = "H" if atom.startswith("H") else atom[0]
            if element in ROLE_NAMES[:5]:
                result[residue, slot, ROLE_NAMES.index(element)] = True
            attached_h = any(mask[residue, child] for child in child_by_parent.get(slot, ()))
            if element == "H":
                result[residue, slot, 5] = mask[residue, slot]
            elif element in {"N", "O", "S"} and attached_h:
                result[residue, slot, 5] = True
            if element in {"O", "S"} and not attached_h:
                result[residue, slot, 6] = True
            if (comp, atom) in acceptor_n and not attached_h:
                result[residue, slot, 6] = True
    return result


def atom27_element_index(comps: tuple[str, ...]) -> np.ndarray:
    names, _ = atom27_tables()
    index = np.full((len(comps), ATOM_COUNT), -1, dtype=np.int64)
    for residue, comp in enumerate(comps):
        for slot, atom in enumerate(names[comp]):
            if atom:
                element = "H" if atom.startswith("H") else atom[0]
                index[residue, slot] = ROLE_NAMES.index(element)
    return index


def atom27_torsions(
    positions: np.ndarray, mask: np.ndarray, comps: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    heavy_names, _ = V6_SUPPORT.atom14_tables()
    heavy_mask = np.asarray(
        [[bool(atom) for atom in heavy_names[comp]] for comp in comps], dtype=np.bool_
    )
    return V6_SUPPORT.atom14_torsions(positions[:, :14], heavy_mask, comps)


@lru_cache(maxsize=256)
def atom27_torsion_topology(comps: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Build each moving set from the full covalent graph after cutting its axis."""

    axes, _ = V6_SUPPORT.torsion_topology(comps)
    adjacency = atom27_bond_adjacency(comps)
    length = len(comps)
    graph: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for residue in range(length):
        for left, right in np.argwhere(np.triu(adjacency[residue], k=1)):
            a, b = (residue, int(left)), (residue, int(right))
            graph[a].add(b)
            graph[b].add(a)
        if residue + 1 < length:
            # Atom14 slots 2 and 0 are C and N for every canonical residue.
            a, b = (residue, 2), (residue + 1, 0)
            graph[a].add(b)
            graph[b].add(a)
    moving = np.zeros((length, 6, length, ATOM_COUNT), dtype=np.bool_)
    for residue in range(length):
        for torsion in range(6):
            if np.any(axes[residue, torsion] < 0):
                continue
            r0, a0, r1, a1 = (int(value) for value in axes[residue, torsion])
            left, seed = (r0, a0), (r1, a1)
            visited = {left}
            queue: deque[tuple[int, int]] = deque([seed])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                moving[residue, torsion, node[0], node[1]] = True
                queue.extend(graph[node] - visited)
    return axes, moving


@lru_cache(maxsize=256)
def _heavy_torsion_topology(comps: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    return V6_SUPPORT.torsion_topology(comps)


def rotate_atom27_delta(
    positions: np.ndarray,
    mask: np.ndarray,
    axes: np.ndarray,
    moving: np.ndarray,
    comps: tuple[str, ...],
    *,
    residue: int,
    torsion: int,
    delta_radians: float,
) -> np.ndarray:
    topology = axes[residue, torsion]
    if np.any(topology < 0):
        raise ValueError("requested torsion has no atom27 topology")
    r0, a0, r1, a1 = (int(value) for value in topology)
    if not mask[r0, a0] or not mask[r1, a1]:
        raise ValueError("torsion axis endpoint is absent")
    first, second = positions[r0, a0], positions[r1, a1]
    axis = second - first
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-8:
        raise ValueError("torsion axis is degenerate")
    unit = axis / norm
    vector = positions - first
    cosine, sine = math.cos(delta_radians), math.sin(delta_radians)
    rotated = (
        positions
        + vector * (cosine - 1.0)
        + np.cross(np.broadcast_to(unit, vector.shape), vector) * sine
        + np.sum(vector * unit, axis=-1, keepdims=True) * unit * (1.0 - cosine)
    )
    selected = moving[residue, torsion] & mask
    output = positions.copy()
    output[selected] = rotated[selected]
    # Preserve the frozen v6 heavy path bit-for-bit.  Computing the same
    # rotation over a 27-slot broadcast can differ by one float32 ULP.
    heavy_names, _ = V6_SUPPORT.atom14_tables()
    heavy_mask = np.asarray(
        [[bool(atom) for atom in heavy_names[comp]] for comp in comps], dtype=np.bool_
    )
    heavy_positions = positions[:, :14].copy()
    heavy_positions[~heavy_mask] = 0.0
    old_axes, old_moving = _heavy_torsion_topology(comps)
    heavy_rotated = V6_SUPPORT.rotate_atom14_delta(
        heavy_positions, heavy_mask, old_axes, old_moving,
        residue=residue, torsion=torsion, delta_radians=delta_radians,
    )
    heavy_before_correction = output[:, :14].copy()
    output[:, :14][heavy_mask] = heavy_rotated[heavy_mask]
    atom_names, parents = atom27_tables()
    for res, comp in enumerate(comps):
        for slot, parent in enumerate(parents[comp]):
            if (
                parent >= 0 and atom_names[comp][slot].startswith("H")
                and mask[res, slot] and mask[res, parent]
            ):
                output[res, slot] += (
                    output[res, parent] - heavy_before_correction[res, parent]
                )
    return output.astype(np.float32, copy=False)


def atom27_coordinate_sha256(positions: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(positions.astype("<f4")).tobytes())
    digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
    return digest.hexdigest()


def heavy_atom14_coordinate_sha256(
    positions: np.ndarray, comps: tuple[str, ...]
) -> str:
    heavy_names, _ = V6_SUPPORT.atom14_tables()
    heavy_mask = np.asarray(
        [[bool(atom) for atom in heavy_names[comp]] for comp in comps], dtype=np.bool_
    )
    heavy_positions = positions[:, :14].copy()
    heavy_positions[~heavy_mask] = 0.0
    return V6_SUPPORT.coordinate_sha256(heavy_positions, heavy_mask)


class DifferentiableAtom27TorsionDecoder(nn.Module):
    """Apply phi/psi/chi rotations to every heavy and descendant-H slot."""

    def forward(
        self,
        *,
        base_atom27_positions: torch.Tensor,
        base_atom27_mask: torch.Tensor,
        torsion_delta: torch.Tensor,
        torsion_axis_atom_indices: torch.Tensor,
        torsion_move_mask: torch.Tensor,
        active_torsions: torch.Tensor | None = None,
        active_indices: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_atom27_positions.ndim != 5 or base_atom27_positions.shape[-2:] != (ATOM_COUNT, 3):
            raise ValueError("base atom27 positions must be [B,K,L,27,3]")
        batch, support_count, residue_count = base_atom27_positions.shape[:3]
        if base_atom27_mask.shape != (batch, support_count, residue_count, ATOM_COUNT):
            raise ValueError("base atom27 mask shape mismatch")
        if torsion_delta.shape != (batch, support_count, residue_count, 6):
            raise ValueError("torsion delta shape mismatch")
        if torsion_axis_atom_indices.shape != (batch, residue_count, 6, 4):
            raise ValueError("atom27 torsion-axis topology shape mismatch")
        if torsion_move_mask.shape != (batch, residue_count, 6, residue_count, ATOM_COUNT):
            raise ValueError("atom27 torsion moving-mask shape mismatch")
        positions = base_atom27_positions
        batch_index = torch.arange(batch, device=positions.device)[:, None]
        support_index = torch.arange(support_count, device=positions.device)[None, :]
        if active_torsions is None:
            active_torsions = torch.all(torsion_axis_atom_indices >= 0, dim=-1)
        elif active_torsions.shape != (batch, residue_count, 6):
            raise ValueError("active torsion predicate shape mismatch")
        if active_indices is None:
            active_indices = torch.nonzero(
                torch.any(active_torsions, dim=0), as_tuple=False
            ).cpu().tolist()
        for residue, torsion in active_indices:
                topology = torsion_axis_atom_indices[:, residue, torsion]
                active = active_torsions[:, residue, torsion]
                safe = topology.clamp_min(0)
                first = positions[batch_index, support_index, safe[:, 0][:, None], safe[:, 1][:, None]]
                second = positions[batch_index, support_index, safe[:, 2][:, None], safe[:, 3][:, None]]
                first_valid = base_atom27_mask[batch_index, support_index, safe[:, 0][:, None], safe[:, 1][:, None]]
                second_valid = base_atom27_mask[batch_index, support_index, safe[:, 2][:, None], safe[:, 3][:, None]]
                axis = second - first
                axis_valid = first_valid & second_valid & (torch.linalg.vector_norm(axis, dim=-1) > 1.0e-8)
                unit = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
                vector = positions - first[:, :, None, None]
                angle = torsion_delta[:, :, residue, torsion]
                cosine = torch.cos(angle)[:, :, None, None, None]
                sine = torch.sin(angle)[:, :, None, None, None]
                unit_full = unit[:, :, None, None]
                rotated = positions + (
                    vector * (cosine - 1.0)
                    + torch.linalg.cross(unit_full.expand_as(vector), vector, dim=-1) * sine
                    + unit_full * torch.sum(unit_full * vector, dim=-1, keepdim=True) * (1.0 - cosine)
                )
                move = (torsion_move_mask[:, residue, torsion] & active[:, None, None])[:, None]
                move = move & base_atom27_mask & axis_valid[:, :, None, None]
                positions = torch.where(move[..., None], rotated, positions)
        if not torch.isfinite(positions[base_atom27_mask]).all():
            raise ValueError("actuated atom27 coordinates became nonfinite")
        return positions, base_atom27_mask


def phe_hn_inter_sulfur_features(
    *,
    atom27_positions: torch.Tensor,
    atom27_mask: torch.Tensor,
    atom27_role_features: torch.Tensor,
    residue_index: torch.Tensor,
    equivalent_structure_slots: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return the frozen six-center inter-residue sulfur chart in [0, 1)."""

    if atom27_positions.ndim != 5 or atom27_positions.shape[-2:] != (ATOM_COUNT, 3):
        raise ValueError("atom27_positions must be [B,K,L,27,3]")
    batch, support_count, residue_count = atom27_positions.shape[:3]
    target_count = residue_index.shape[1]
    if (
        atom27_mask.shape != (batch, support_count, residue_count, ATOM_COUNT)
        or atom27_role_features.shape
        != (batch, support_count, residue_count, ATOM_COUNT, ROLE_COUNT)
        or equivalent_structure_slots.shape != (batch, target_count, 3)
        or chunk_size < 1
    ):
        raise ValueError("sulfur-chart tensor shape drift")
    declared = equivalent_structure_slots >= 0
    safe = equivalent_structure_slots.clamp_min(0)
    batch_index = torch.arange(batch, device=atom27_positions.device)[:, None, None]
    support_index = torch.arange(
        support_count, device=atom27_positions.device
    )[None, :, None]
    residue = residue_index[:, None].expand(-1, support_count, -1)
    local_positions = atom27_positions[batch_index, support_index, residue]
    local_mask = atom27_mask[batch_index, support_index, residue]
    gather_index = safe[:, None, :, :, None].expand(-1, support_count, -1, -1, 3)
    equivalent_positions = torch.gather(local_positions, 3, gather_index)
    equivalent_present = torch.gather(
        local_mask,
        3,
        safe[:, None].expand(-1, support_count, -1, -1),
    ) & declared[:, None]
    denominator = equivalent_present.sum(dim=-1, keepdim=True).clamp_min(1).to(
        equivalent_positions.dtype
    )
    site = (
        equivalent_positions
        * equivalent_present[..., None].to(equivalent_positions.dtype)
    ).sum(dim=3) / denominator
    site_valid = equivalent_present.any(dim=-1)
    sulfur = atom27_mask & atom27_role_features[..., ROLE_NAMES.index("S")]
    residue_grid = torch.arange(
        residue_count, device=atom27_positions.device
    )[None, None, None, :, None]
    centers = atom27_positions.new_tensor(PHYSICS_CHART_RBF_CENTERS_NM)
    chunks: list[torch.Tensor] = []
    for start in range(0, target_count, chunk_size):
        stop = min(start + chunk_size, target_count)
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
        radial = torch.exp(-0.5 * (
            (distance[..., None] - centers) / PHYSICS_CHART_RBF_WIDTH_NM
        ).square())
        count = selected.sum(dim=(-2, -1)).clamp_min(1).to(radial.dtype)
        z = (
            radial * selected[..., None].to(radial.dtype)
        ).sum(dim=(-3, -2)) / count[..., None].sqrt()
        chunks.append(1.0 - torch.exp(-z))
    result = torch.cat(chunks, dim=2)
    if result.shape != (batch, support_count, target_count, len(centers)):
        raise RuntimeError("sulfur-chart output shape drift")
    if torch.any(result < 0) or torch.any(result >= 1) or not torch.isfinite(result).all():
        raise RuntimeError("sulfur-chart feature left [0,1)")
    return result


class ExplicitHAtom27Observer(nn.Module):
    """Invariant atom27 observer with exact H-parent direction and methyl pooling."""

    def __init__(
        self,
        *,
        atom_vocab_size: int,
        sequence_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        atom_dim: int = 32,
    ) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(atom_vocab_size, atom_dim)
        self.sequence_dim = int(sequence_dim)
        self.condition_dim = int(condition_dim)
        self.register_buffer(
            "rbf_centers_nm", torch.tensor((0.20, 0.30, 0.40, 0.55, 0.75, 1.00))
        )
        # anchors(10), torsions(12), site(1), role-RBF(84), nearest(14),
        # directional roles(14), parent roles(7), atom/residue/condition.
        fixed_dim = 10 + 12 + 1 + 84 + 14 + 14 + 7
        self.network = nn.Sequential(
            nn.Linear(fixed_dim + atom_dim + sequence_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        *,
        atom27_positions: torch.Tensor,
        atom27_mask: torch.Tensor,
        atom27_role_features: torch.Tensor,
        torsion_angles: torch.Tensor,
        torsion_mask: torch.Tensor,
        sequence_features: torch.Tensor,
        condition_features: torch.Tensor,
        residue_index: torch.Tensor,
        atom_index: torch.Tensor,
        structure_slot: torch.Tensor,
        bonded_parent_slot: torch.Tensor,
        equivalent_structure_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if atom27_positions.ndim != 5 or atom27_positions.shape[-2:] != (ATOM_COUNT, 3):
            raise ValueError("atom27_positions must have shape [B,K,L,27,3]")
        batch, support_count, residue_count = atom27_positions.shape[:3]
        if atom27_mask.shape != (batch, support_count, residue_count, ATOM_COUNT):
            raise ValueError("atom27_mask shape mismatch")
        if atom27_role_features.shape != (
            batch, support_count, residue_count, ATOM_COUNT, ROLE_COUNT
        ):
            raise ValueError("atom27_role_features must be [B,K,L,27,7]")
        if torsion_angles.shape != (batch, support_count, residue_count, 6) or torsion_mask.shape != torsion_angles.shape:
            raise ValueError("torsion tensors shape mismatch")
        if sequence_features.shape[:2] != (batch, residue_count):
            raise ValueError("sequence_features shape mismatch")
        target_count = residue_index.shape[1]
        if (
            atom_index.shape != (batch, target_count)
            or structure_slot.shape != (batch, target_count)
            or bonded_parent_slot.shape != (batch, target_count)
            or equivalent_structure_slots.shape != (batch, target_count, 3)
        ):
            raise ValueError("target atom27 topology shape mismatch")
        if torch.any((structure_slot < 0) | (structure_slot >= ATOM_COUNT)):
            raise ValueError("every target needs a declared exact atom27 union slot")
        if torch.any((bonded_parent_slot < 0) | (bonded_parent_slot >= ATOM_COUNT)):
            raise ValueError("every target needs an exact atom27 parent/anchor slot")
        declared_equivalent = equivalent_structure_slots >= 0
        if not declared_equivalent.any(dim=-1).all() or torch.any(equivalent_structure_slots >= ATOM_COUNT):
            raise ValueError("equivalent atom27 slots must contain 1..3 valid members")

        batch_index = torch.arange(batch, device=atom27_positions.device)[:, None, None]
        support_index = torch.arange(support_count, device=atom27_positions.device)[None, :, None]
        residue = residue_index[:, None].expand(-1, support_count, -1)
        local_positions = atom27_positions[batch_index, support_index, residue]
        local_mask = atom27_mask[batch_index, support_index, residue]
        local_roles = atom27_role_features[batch_index, support_index, residue]

        safe_equivalent = equivalent_structure_slots.clamp_min(0)
        equivalent_index = safe_equivalent[:, None, :, :, None].expand(
            -1, support_count, -1, -1, 3
        )
        equivalent_positions = torch.gather(local_positions, 3, equivalent_index)
        equivalent_present = torch.gather(
            local_mask,
            3,
            safe_equivalent[:, None].expand(-1, support_count, -1, -1),
        ) & declared_equivalent[:, None]
        denominator = equivalent_present.sum(dim=-1, keepdim=True).clamp_min(1).to(
            equivalent_positions.dtype
        )
        selected = (
            equivalent_positions * equivalent_present[..., None].to(equivalent_positions.dtype)
        ).sum(dim=3) / denominator
        selected_valid = equivalent_present.any(dim=-1)
        exact_valid = torch.gather(
            local_mask,
            3,
            structure_slot[:, None, :, None].expand(-1, support_count, -1, 1),
        ).squeeze(3)

        parent_index = bonded_parent_slot[:, None, :, None, None].expand(
            -1, support_count, -1, 1, 3
        )
        parent_position = torch.gather(local_positions, 3, parent_index).squeeze(3)
        parent_valid = torch.gather(
            local_mask,
            3,
            bonded_parent_slot[:, None, :, None].expand(-1, support_count, -1, 1),
        ).squeeze(3)
        reference = selected - parent_position
        reference_valid = (
            selected_valid & parent_valid
            & (reference.square().sum(dim=-1) > 1.0e-12)
        )
        # Heavy targets use self as their declared parent; retain the v6
        # N/CA/C/O/CB fallback only when an H-parent vector is unavailable.
        for anchor_slot in (1, 0, 2, 3, 4):
            candidate = selected - local_positions[..., anchor_slot, :]
            candidate_valid = (
                selected_valid & local_mask[..., anchor_slot]
                & (candidate.square().sum(dim=-1) > 1.0e-12)
                & ~reference_valid
            )
            reference = torch.where(candidate_valid[..., None], candidate, reference)
            reference_valid |= candidate_valid
        reference_unit = reference / reference.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12).sqrt()

        anchors = local_positions[..., :5, :]
        anchor_mask = local_mask[..., :5]
        anchor_distance = (selected[..., None, :] - anchors).square().sum(
            dim=-1
        ).clamp_min(1.0e-12).sqrt()
        anchor_valid = selected_valid[..., None] & anchor_mask
        anchor_distance = anchor_distance.masked_fill(~anchor_valid, 0.0)
        parent_roles = torch.gather(
            local_roles,
            3,
            bonded_parent_slot[:, None, :, None, None].expand(
                -1, support_count, -1, 1, ROLE_COUNT
            ),
        ).squeeze(3)

        environment_chunks: list[torch.Tensor] = []
        nearest_chunks: list[torch.Tensor] = []
        direction_chunks: list[torch.Tensor] = []
        residue_grid = torch.arange(
            residue_count, device=atom27_positions.device
        )[None, None, None, :, None]
        slot_grid = torch.arange(
            ATOM_COUNT, device=atom27_positions.device
        )[None, None, None, None, :]
        def environment_chunk(
            positions: torch.Tensor,
            site: torch.Tensor,
            reference_direction: torch.Tensor,
            site_valid: torch.Tensor,
            target_residue: torch.Tensor,
            target_equivalent: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            chunk_size = site.shape[2]
            displacement = positions[:, :, None] - site[..., None, None, :]
            all_distance = displacement.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
            displacement_unit = displacement / all_distance[..., None].clamp_min(1.0e-6)
            direction = torch.sum(
                displacement_unit * reference_direction[..., None, None, :], dim=-1
            )
            same = residue_grid == target_residue
            self_atom = same & torch.any(
                (target_equivalent >= 0) & (slot_grid[..., None] == target_equivalent),
                dim=-1,
            )
            valid = (
                atom27_mask[:, :, None].expand(-1, -1, chunk_size, -1, -1)
                & ~self_atom
                & site_valid[..., None, None]
            )
            chunk_role = atom27_role_features[:, :, None].expand(
                -1, -1, chunk_size, -1, -1, -1
            )
            radial_basis = torch.exp(-0.5 * (
                (all_distance[..., None] - self.rbf_centers_nm.to(dtype=all_distance.dtype))
                / 0.12
            ).square())
            directional_decay = torch.exp(-all_distance / 0.15)
            large = torch.full_like(all_distance, 2.0)
            feature_parts: list[torch.Tensor] = []
            nearest_parts: list[torch.Tensor] = []
            direction_parts: list[torch.Tensor] = []
            for same_flag in (True, False):
                scope = same if same_flag else ~same
                for role_index in range(ROLE_COUNT):
                    selected_role = valid & scope & chunk_role[..., role_index]
                    rbf = radial_basis * selected_role[..., None]
                    count = selected_role.sum(dim=(-2, -1)).clamp_min(1).to(rbf.dtype)
                    feature_parts.append(rbf.sum(dim=(-3, -2)) / count[..., None].sqrt())
                    nearest = torch.where(selected_role, all_distance, large).amin(dim=(-2, -1))
                    nearest_parts.append(torch.where(
                        selected_role.any(dim=(-2, -1)), nearest, torch.zeros_like(nearest)
                    ))
                    weight = directional_decay * selected_role
                    direction_parts.append(
                        (direction * weight).sum(dim=(-2, -1))
                        / weight.sum(dim=(-2, -1)).clamp_min(1.0e-8)
                    )
            return (
                torch.cat(feature_parts, dim=-1),
                torch.stack(nearest_parts, dim=-1),
                torch.stack(direction_parts, dim=-1),
            )

        for start in range(0, target_count, 64):
            stop = min(start + 64, target_count)
            chunk_inputs = (
                atom27_positions,
                selected[:, :, start:stop],
                reference_unit[:, :, start:stop],
                selected_valid[:, :, start:stop],
                residue_index[:, None, start:stop, None, None],
                equivalent_structure_slots[:, None, start:stop, None, None, :],
            )
            if atom27_positions.requires_grad and torch.is_grad_enabled():
                chunk = checkpoint(environment_chunk, *chunk_inputs, use_reentrant=False)
            else:
                chunk = environment_chunk(*chunk_inputs)
            environment_chunks.append(chunk[0])
            nearest_chunks.append(chunk[1])
            direction_chunks.append(chunk[2])
        environment = torch.cat(environment_chunks, dim=2)
        role_nearest = torch.cat(nearest_chunks, dim=2)
        role_direction = torch.cat(direction_chunks, dim=2)

        local_torsion = torsion_angles[batch_index, support_index, residue]
        local_torsion_mask = torsion_mask[batch_index, support_index, residue]
        periodic = torch.cat(
            (
                torch.sin(local_torsion) * local_torsion_mask,
                torch.cos(local_torsion) * local_torsion_mask,
            ),
            dim=-1,
        )
        sequence = V6_MODEL.gather_residue(sequence_features, residue_index)[:, None].expand(
            -1, support_count, -1, -1
        )
        condition = condition_features[:, None, None].expand(
            -1, support_count, target_count, -1
        )
        atom = self.atom_embedding(atom_index)[:, None].expand(
            -1, support_count, -1, -1
        )
        features = torch.cat(
            (
                anchor_distance,
                anchor_valid.to(anchor_distance.dtype),
                periodic,
                selected_valid.to(anchor_distance.dtype).unsqueeze(-1),
                environment,
                role_nearest,
                role_direction,
                parent_roles.to(anchor_distance.dtype),
                atom,
                sequence,
                condition,
            ),
            dim=-1,
        )
        output = self.network(features)
        return output[..., 0], output[..., 1].clamp(-7.0, 5.0), exact_valid


def actuated_atom27_physics_receipt(
    positions: torch.Tensor,
    mask: torch.Tensor,
    bond_adjacency: torch.Tensor,
    element_index: torch.Tensor,
    reference_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable all-atom bond and element-pair clash receipt."""

    if positions.ndim != 5 or positions.shape[-2:] != (ATOM_COUNT, 3):
        raise ValueError("actuated atom27 positions must be [B,K,L,27,3]")
    if mask.shape != positions.shape[:-1] or reference_positions.shape != positions.shape:
        raise ValueError("actuated atom27 mask/reference shape mismatch")
    batch, support_count, residue_count = positions.shape[:3]
    if bond_adjacency.shape != (batch, residue_count, ATOM_COUNT, ATOM_COUNT):
        raise ValueError("atom27 bond adjacency must be [B,L,27,27]")
    if element_index.shape != (batch, residue_count, ATOM_COUNT):
        raise ValueError("atom27 element index must be [B,L,27]")
    if bond_adjacency.dtype != torch.bool or torch.any(
        torch.diagonal(bond_adjacency, dim1=-2, dim2=-1)
    ) or not torch.equal(bond_adjacency, bond_adjacency.transpose(-1, -2)):
        raise ValueError("atom27 adjacency must be symmetric boolean without self edges")
    if torch.any((element_index < -1) | (element_index > 4)):
        raise ValueError("atom27 element index is invalid")

    local_delta = positions[..., :, None, :] - positions[..., None, :, :]
    local_distance = local_delta.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
    reference_delta = reference_positions[..., :, None, :] - reference_positions[..., None, :, :]
    reference_distance = reference_delta.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
    upper = torch.triu(
        torch.ones(ATOM_COUNT, ATOM_COUNT, dtype=torch.bool, device=positions.device),
        diagonal=1,
    )
    declared_bond = (
        mask[..., :, None] & mask[..., None, :]
        & bond_adjacency[:, None] & upper
    )
    bond_violation = positions.new_zeros(())
    bond_change = positions.new_zeros(())
    if bool(torch.any(declared_bond)):
        selected = local_distance[declared_bond]
        reference_selected = reference_distance[declared_bond]
        element_left = element_index[:, None, :, :, None].expand(
            -1, support_count, -1, -1, ATOM_COUNT
        )[declared_bond]
        element_right = element_index[:, None, :, None, :].expand(
            -1, support_count, -1, ATOM_COUNT, -1
        )[declared_bond]
        hydrogen_bond = (element_left == 0) | (element_right == 0)
        lower = torch.where(hydrogen_bond, selected.new_tensor(0.08), selected.new_tensor(0.09))
        high = torch.where(hydrogen_bond, selected.new_tensor(0.145), selected.new_tensor(0.22))
        bond_violation = (torch.relu(lower - selected) + torch.relu(selected - high)).amax()
        bond_change = (selected - reference_selected).abs().amax()
    peptide_present = mask[..., :-1, 2] & mask[..., 1:, 0]
    peptide_distance = (
        positions[..., :-1, 2, :] - positions[..., 1:, 0, :]
    ).square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
    if bool(torch.any(peptide_present)):
        peptide_violation = torch.relu(0.10 - peptide_distance) + torch.relu(peptide_distance - 0.18)
        bond_violation = torch.maximum(bond_violation, peptide_violation[peptide_present].amax())
        reference_peptide = (
            reference_positions[..., :-1, 2, :] - reference_positions[..., 1:, 0, :]
        ).square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
        bond_change = torch.maximum(
            bond_change,
            (peptide_distance[peptide_present] - reference_peptide[peptide_present]).abs().amax(),
        )

    atom_count = residue_count * ATOM_COUNT
    flat = positions.reshape(batch, support_count, atom_count, 3)
    valid = mask.reshape(batch, support_count, atom_count)
    flat_element = element_index.reshape(batch, atom_count)
    radii = positions.new_tensor((0.12, 0.17, 0.155, 0.152, 0.18))
    safe_element = flat_element.clamp_min(0)
    flat_radius = radii[safe_element]
    all_index = torch.arange(atom_count, device=positions.device)
    residue_ids = torch.arange(residue_count, device=positions.device).repeat_interleave(ATOM_COUNT)
    one_or_two = bond_adjacency | (
        torch.matmul(bond_adjacency.to(torch.float32), bond_adjacency.to(torch.float32)) > 0
    )
    one_or_two |= torch.eye(ATOM_COUNT, dtype=torch.bool, device=positions.device)[None, None]
    same_residue = torch.eye(
        residue_count, dtype=torch.bool, device=positions.device
    )
    excluded = (
        one_or_two[:, :, :, None, :]
        & same_residue[None, :, None, :, None]
    ).reshape(batch, atom_count, atom_count)
    boundaries = torch.arange(
        max(residue_count - 1, 0), device=positions.device
    )
    slots = torch.arange(ATOM_COUNT, device=positions.device)
    left_base = boundaries * ATOM_COUNT
    right_base = (boundaries + 1) * ATOM_COUNT
    left = left_base + 2
    right = right_base
    left_neighbor = left_base[:, None] + slots[None]
    right_neighbor = right_base[:, None] + slots[None]
    excluded[:, left, right] = True
    excluded[:, right, left] = True
    left_is_neighbor = bond_adjacency[:, :-1, 2]
    right_is_neighbor = bond_adjacency[:, 1:, 0]
    excluded[:, left_neighbor, right[:, None]] |= left_is_neighbor
    excluded[:, right[:, None], left_neighbor] |= left_is_neighbor
    excluded[:, left[:, None], right_neighbor] |= right_is_neighbor
    excluded[:, right_neighbor, left[:, None]] |= right_is_neighbor

    penalty_sum = positions.new_zeros(())
    pair_count = positions.new_zeros(())
    minimum = positions.new_tensor(float("inf"))
    minimum_ratio = positions.new_tensor(float("inf"))
    same_distinct_minimum = positions.new_tensor(float("inf"))
    for start in range(0, atom_count, 128):
        stop = min(start + 128, atom_count)
        left_index = all_index[start:stop]
        delta = flat[:, :, start:stop, None, :] - flat[:, :, None, :, :]
        distance = delta.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
        pair_upper = (all_index[None, :] > left_index[:, None])[None, None]
        valid_pair = valid[:, :, start:stop, None] & valid[:, :, None, :] & pair_upper
        same = (
            residue_ids[start:stop, None] == residue_ids[None, :]
        )[None, None]
        same_distinct = valid_pair & same
        if bool(torch.any(same_distinct)):
            same_distinct_minimum = torch.minimum(
                same_distinct_minimum, distance[same_distinct].amin()
            )
        pair = valid_pair & ~excluded[:, None, start:stop, :]
        if not bool(torch.any(pair)):
            continue
        threshold = 0.5 * (
            flat_radius[:, start:stop, None] + flat_radius[:, None, :]
        )[:, None]
        selected_distance = distance[pair]
        selected_threshold = threshold.expand(-1, support_count, -1, -1)[pair]
        ratio = selected_distance / selected_threshold.clamp_min(1.0e-6)
        violation = torch.relu(1.0 - ratio)
        penalty_sum = penalty_sum + violation.square().sum()
        pair_count = pair_count + pair.sum().to(pair_count.dtype)
        minimum = torch.minimum(minimum, selected_distance.amin())
        minimum_ratio = torch.minimum(minimum_ratio, ratio.amin())
    if not bool(torch.isfinite(minimum)):
        minimum = positions.new_zeros(())
        minimum_ratio = positions.new_ones(())
    if not bool(torch.isfinite(same_distinct_minimum)):
        same_distinct_minimum = positions.new_zeros(())
    return {
        "clash_penalty": penalty_sum / pair_count.clamp_min(1.0),
        "minimum_nonbonded_distance_nm": minimum,
        "minimum_nonbonded_radius_ratio": minimum_ratio,
        "minimum_same_residue_distinct_atom_distance_nm": same_distinct_minimum,
        "maximum_declared_bond_range_violation_nm": bond_violation,
        "maximum_declared_bond_length_change_nm": bond_change,
        "declared_same_residue_bond_count": declared_bond.sum(),
    }


def deterministic_atom27_feasibility_projection(
    *,
    torsion_delta: torch.Tensor,
    base_positions: torch.Tensor,
    base_mask: torch.Tensor,
    decode_candidate: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    physics_candidate: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Probe detached atom27 candidates, then replay only the selected one with grad."""

    if not torch.isfinite(torsion_delta).all():
        raise ValueError("atom27 actuator hard bond/clash safety gate torsion proposal is nonfinite")

    def candidate_at(
        scaled_delta: torch.Tensor, identity: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if identity:
            positions, mask = base_positions, base_mask
        else:
            positions, mask = decode_candidate(scaled_delta)
        if positions.shape != base_positions.shape or mask.shape != base_mask.shape:
            raise ValueError("atom27 decoder changed tensor shape")
        if not torch.equal(mask, base_mask):
            raise ValueError("atom27 decoder changed coordinate identity/mask")
        return positions, mask, physics_candidate(positions, mask)

    ladder = V6_MODEL.FEASIBILITY_SCALE_LADDER
    candidate_codes = [-1] * len(ladder)
    chosen_index: int | None = None
    unprojected_physics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for index, scale in enumerate(ladder):
            probe_delta = torsion_delta.detach() * scale
            _, _, probe_physics = candidate_at(probe_delta, scale == 0.0)
            code = V6_MODEL.physics_gate_failure_code(
                probe_physics, ATOM27_HARD_GATE_SPECS
            )
            candidate_codes[index] = code
            if index == 0:
                unprojected_physics = {
                    key: value.detach().clone() for key, value in probe_physics.items()
                }
            if code == 0:
                chosen_index = index
                break

    if chosen_index is None:
        values = {
            key: float(value.detach().cpu())
            for key, value in probe_physics.items()
            if value.numel() == 1
        }
        raise ValueError(
            "atom27 actuator hard bond/clash safety gate identity candidate failed: "
            f"code={candidate_codes[-1]} metrics={json.dumps(values, sort_keys=True)}"
        )

    chosen_scale = ladder[chosen_index]
    selected_delta = torsion_delta * chosen_scale
    selected_positions, selected_mask, selected_physics = candidate_at(
        selected_delta, chosen_scale == 0.0
    )
    if V6_MODEL.physics_gate_failure_code(
        selected_physics, ATOM27_HARD_GATE_SPECS
    ) != 0:
        raise RuntimeError("atom27 deterministic feasibility replay drift")
    return {
        "positions": selected_positions,
        "mask": selected_mask,
        "torsion_delta": selected_delta,
        "physics": selected_physics,
        "unprojected_physics": unprojected_physics,
        "feasibility_scale": torsion_delta.new_tensor(chosen_scale),
        "feasibility_ladder_index": torch.tensor(
            chosen_index, dtype=torch.int64, device=torsion_delta.device
        ),
        "feasibility_candidate_count": torch.tensor(
            chosen_index + 1, dtype=torch.int64, device=torsion_delta.device
        ),
        "feasibility_identity_selected": torch.tensor(
            chosen_scale == 0.0, dtype=torch.bool, device=torsion_delta.device
        ),
        "feasibility_full_step_feasible": torch.tensor(
            candidate_codes[0] == 0, dtype=torch.bool, device=torsion_delta.device
        ),
        "feasibility_candidate_gate_failure_codes": torch.tensor(
            candidate_codes, dtype=torch.int64, device=torsion_delta.device
        ),
    }


class GlobalAllAtomSharedQ(nn.Module):
    """v6 shared-q objective with literal atom27 support coordinates."""

    def __init__(
        self,
        *,
        fixed_support_source: Path,
        atom_vocab_size: int,
        sequence_dim: int,
        condition_dim: int,
        global_latent_dim: int,
        residue_latent_dim: int,
        hidden_dim: int = 128,
        q_rank: int = 16,
        actuator_enabled: bool = False,
        prior_kl_weight: float = 1.0e-3,
        identity_tether_weight: float = 1.0,
        backbone_smoothness_weight: float = 0.1,
        physics_clash_weight: float = 0.1,
    ) -> None:
        super().__init__()
        pinned = V6_MODEL.load_pinned_fixed_support(fixed_support_source)
        self.evidence = V6_MODEL.AllAtomEvidenceEncoder(
            atom_vocab_size=atom_vocab_size,
            sequence_dim=sequence_dim,
            condition_dim=condition_dim,
            evidence_dim=hidden_dim,
        )
        self.observer = ExplicitHAtom27Observer(
            atom_vocab_size=atom_vocab_size,
            sequence_dim=sequence_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
        )
        self.support = V6_MODEL.SupportSummaryEncoder(
            global_latent_dim=global_latent_dim, output_dim=64
        )
        self.measure = pinned.FixedSupportMeasureGenerator(64, hidden_dim, rank=q_rank)
        self.actuator_enabled = bool(actuator_enabled)
        self.prior_kl_weight = float(prior_kl_weight)
        self.identity_tether_weight = float(identity_tether_weight)
        self.backbone_smoothness_weight = float(backbone_smoothness_weight)
        self.physics_clash_weight = float(physics_clash_weight)
        if min(
            self.prior_kl_weight,
            self.identity_tether_weight,
            self.backbone_smoothness_weight,
            self.physics_clash_weight,
        ) < 0:
            raise ValueError("objective weights must be nonnegative")
        self.actuator = V6_MODEL.CompleteStateTorsionActuator(
            residue_latent_dim=residue_latent_dim,
            global_latent_dim=global_latent_dim,
            hidden_dim=hidden_dim,
        ) if actuator_enabled else None
        self.atom27_decoder = DifferentiableAtom27TorsionDecoder()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        atom27_decoder: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        required = {
            "atom27_positions", "atom27_mask", "atom27_role_features",
            "atom27_bond_adjacency", "atom27_element_index",
            "torsion_angles", "torsion_mask", "sequence_features",
            "residue_latent", "condition_features", "support_global_latent",
            "residue_index", "atom_index", "nucleus_index", "stratum_index",
            "structure_slot", "bonded_parent_slot", "equivalent_structure_slots",
            "direct_literal_support_mask", "target_values", "target_centers",
            "target_scales", "conditioning_mask", "scoring_mask", "target_weights",
            "prior_logits", "stage_a_mean", "stage_a_scale",
        }
        missing = required - batch.keys()
        if missing:
            raise ValueError(f"explicit-H batch is missing keys: {sorted(missing)}")
        conditioning, scoring = batch["conditioning_mask"], batch["scoring_mask"]
        if conditioning.dtype != torch.bool or scoring.dtype != torch.bool:
            raise ValueError("conditioning/scoring masks must be boolean")
        if torch.any(conditioning & scoring) or not scoring.any(dim=1).all():
            raise ValueError("conditioning/scoring roles are invalid")
        if batch["stage_a_mean"].shape != batch["target_values"].shape or batch[
            "stage_a_scale"
        ].shape != batch["target_values"].shape:
            raise ValueError("frozen Stage-A prediction shape mismatch")
        if torch.any(batch["stage_a_scale"] <= 0) or not torch.isfinite(
            batch["stage_a_scale"]
        ).all():
            raise ValueError("frozen Stage-A scale must be finite and positive")

        positions = batch["atom27_positions"]
        torsion_angles = batch["torsion_angles"]
        actuator_terms = {
            "identity_tether": positions.new_zeros(()),
            "backbone_smoothness": positions.new_zeros(()),
            "proposal_identity_tether": positions.new_zeros(()),
            "proposal_backbone_smoothness": positions.new_zeros(()),
            "clash_penalty": positions.new_zeros(()),
            "minimum_nonbonded_distance_nm": positions.new_tensor(float("nan")),
            "minimum_nonbonded_radius_ratio": positions.new_tensor(float("nan")),
            "minimum_same_residue_distinct_atom_distance_nm": positions.new_tensor(float("nan")),
            "maximum_declared_bond_range_violation_nm": positions.new_zeros(()),
            "maximum_declared_bond_length_change_nm": positions.new_zeros(()),
            "declared_same_residue_bond_count": positions.new_zeros(()),
            "proposal_torsion_delta": torch.zeros_like(torsion_angles),
            "torsion_delta": torch.zeros_like(torsion_angles),
            "feasibility_scale": positions.new_ones(()),
            "feasibility_ladder_index": torch.zeros(
                (), dtype=torch.int64, device=positions.device
            ),
            "feasibility_candidate_count": torch.zeros(
                (), dtype=torch.int64, device=positions.device
            ),
            "feasibility_identity_selected": torch.zeros(
                (), dtype=torch.bool, device=positions.device
            ),
            "feasibility_full_step_feasible": torch.ones(
                (), dtype=torch.bool, device=positions.device
            ),
            "feasibility_candidate_gate_failure_codes": torch.full(
                (len(V6_MODEL.FEASIBILITY_SCALE_LADDER),), -1,
                dtype=torch.int64, device=positions.device,
            ),
        }
        effective_torsion_mask = batch["torsion_mask"]
        if self.actuator_enabled:
            for key in ("torsion_axis_atom_indices", "torsion_move_mask"):
                if key not in batch:
                    raise ValueError(f"joint explicit-H batch is missing {key}")
            valid_axis = torch.all(batch["torsion_axis_atom_indices"] >= 0, dim=-1)
            valid_moving = torch.any(batch["torsion_move_mask"], dim=(-2, -1))
            effective_torsion_mask = (
                batch["torsion_mask"] & valid_axis[:, None] & valid_moving[:, None]
            )
            if self.actuator is None:
                raise RuntimeError("explicit-H actuator was not constructed")
            actuator_terms = self.actuator(
                residue_latent=batch["residue_latent"],
                support_global_latent=batch["support_global_latent"],
                raw_torsion_angles=torsion_angles,
                torsion_mask=effective_torsion_mask,
            )
            proposal_delta = actuator_terms["torsion_delta"]
            decoder = self.atom27_decoder if atom27_decoder is None else atom27_decoder
            active_indices = torch.nonzero(
                torch.any(valid_axis, dim=0), as_tuple=False
            ).cpu().tolist()

            def decode_candidate(
                candidate_delta: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                kwargs = {
                    "base_atom27_positions": batch["atom27_positions"],
                    "base_atom27_mask": batch["atom27_mask"],
                    "torsion_delta": candidate_delta,
                    "torsion_axis_atom_indices": batch["torsion_axis_atom_indices"],
                    "torsion_move_mask": batch["torsion_move_mask"],
                }
                if atom27_decoder is None:
                    return decoder(
                        **kwargs,
                        active_torsions=valid_axis,
                        active_indices=active_indices,
                    )
                return decoder(**kwargs)

            def physics_candidate(
                candidate_positions: torch.Tensor, candidate_mask: torch.Tensor
            ) -> dict[str, torch.Tensor]:
                return actuated_atom27_physics_receipt(
                    candidate_positions,
                    candidate_mask,
                    batch["atom27_bond_adjacency"],
                    batch["atom27_element_index"],
                    batch["atom27_positions"],
                )

            projection = deterministic_atom27_feasibility_projection(
                torsion_delta=proposal_delta,
                base_positions=batch["atom27_positions"],
                base_mask=batch["atom27_mask"],
                decode_candidate=decode_candidate,
                physics_candidate=physics_candidate,
            )
            positions, decoded_mask = projection["positions"], projection["mask"]
            if positions.shape != batch["atom27_positions"].shape or not torch.equal(
                decoded_mask, batch["atom27_mask"]
            ):
                raise ValueError("atom27 decoder changed coordinate identity/mask")
            torsion_angles = V6_MODEL.torsion_angles_from_delta(
                batch["torsion_angles"], projection["torsion_delta"],
                effective_torsion_mask,
                exact_identity=bool(projection["feasibility_identity_selected"]),
            )
            actuator_terms["proposal_torsion_delta"] = proposal_delta
            actuator_terms["torsion_delta"] = projection["torsion_delta"]
            actuator_terms["proposal_identity_tether"] = actuator_terms["identity_tether"]
            actuator_terms["proposal_backbone_smoothness"] = actuator_terms[
                "backbone_smoothness"
            ]
            scale_square = projection["feasibility_scale"].square()
            actuator_terms["identity_tether"] = (
                actuator_terms["proposal_identity_tether"] * scale_square
            )
            actuator_terms["backbone_smoothness"] = (
                actuator_terms["proposal_backbone_smoothness"] * scale_square
            )
            actuator_terms.update(projection["physics"])
            for key in (
                "feasibility_scale", "feasibility_ladder_index",
                "feasibility_candidate_count", "feasibility_identity_selected",
                "feasibility_full_step_feasible",
                "feasibility_candidate_gate_failure_codes",
            ):
                actuator_terms[key] = projection[key]
            for key, value in projection["unprojected_physics"].items():
                actuator_terms[f"unprojected_{key}"] = value

        evidence = self.evidence(
            sequence_features=batch["sequence_features"],
            condition_features=batch["condition_features"],
            residue_index=batch["residue_index"],
            atom_index=batch["atom_index"],
            nucleus_index=batch["nucleus_index"],
            stratum_index=batch["stratum_index"],
            target_values=batch["target_values"],
            target_centers=batch["stage_a_mean"],
            target_scales=batch["stage_a_scale"],
            conditioning_mask=conditioning,
        )
        # Q initialization is transferred from v6.  Keep its support feature
        # semantics exact: the first fourteen slots are the frozen heavy-atom
        # layout, while literal H inserted into historically empty slots must
        # not change the v6 side-chain occupancy statistic.
        heavy_atom14_mask = (
            batch["atom27_mask"][..., :14]
            & (batch["atom27_element_index"][:, None, :, :14] != 0)
        )
        support = self.support(
            positions[..., :14, :],
            heavy_atom14_mask,
            torsion_angles,
            effective_torsion_mask,
            batch["support_global_latent"],
        )
        measure_output = self.measure(
            support,
            evidence_features=evidence,
            prior_log_probs=batch["prior_logits"],
        )
        q = measure_output["weights"]
        if q.ndim != 2 or not torch.allclose(
            q.sum(dim=1), torch.ones_like(q[:, 0]), atol=1.0e-6
        ):
            raise RuntimeError("shared q is not a per-entity simplex")
        standardized_mean, standardized_log_scale, direct_mask = self.observer(
            atom27_positions=positions,
            atom27_mask=batch["atom27_mask"],
            atom27_role_features=batch["atom27_role_features"],
            torsion_angles=torsion_angles,
            torsion_mask=effective_torsion_mask,
            sequence_features=batch["sequence_features"],
            condition_features=batch["condition_features"],
            residue_index=batch["residue_index"],
            atom_index=batch["atom_index"],
            structure_slot=batch["structure_slot"],
            bonded_parent_slot=batch["bonded_parent_slot"],
            equivalent_structure_slots=batch["equivalent_structure_slots"],
        )
        declared_direct = batch["direct_literal_support_mask"]
        if declared_direct.dtype != torch.bool or declared_direct.shape != direct_mask.shape:
            raise ValueError("direct-literal support mask shape/dtype mismatch")
        if not torch.equal(declared_direct, direct_mask):
            raise ValueError("direct-literal support receipt differs from exact atom27 masks")
        support_mean = batch["stage_a_mean"][:, None] + standardized_mean * batch[
            "stage_a_scale"
        ][:, None]
        if PHYSICS_CHART_CANDIDATE is not None:
            chart = PHYSICS_CHART_CANDIDATE
            if (
                chart.get("menu_sha256") != PHYSICS_CHART_MENU_SHA256
                or float(chart.get("center_nm", -1.0)) not in PHYSICS_CHART_RBF_CENTERS_NM
                or int(chart.get("sign", 0)) not in (-1, 1)
                or float(chart.get("magnitude_ppm", -1.0)) != 0.005
            ):
                raise ValueError("declared physics-chart candidate is outside the frozen menu")
            features = phe_hn_inter_sulfur_features(
                atom27_positions=positions,
                atom27_mask=batch["atom27_mask"],
                atom27_role_features=batch["atom27_role_features"],
                residue_index=batch["residue_index"],
                equivalent_structure_slots=batch["equivalent_structure_slots"],
            )
            center_index = PHYSICS_CHART_RBF_CENTERS_NM.index(
                float(chart["center_nm"])
            )
            target_token = torch.gather(
                batch["stage_a_tokens"], 1, batch["residue_index"]
            )
            target_gate = (
                (target_token == PHYSICS_CHART_PHE_TOKEN)
                & (batch["atom_index"] == PHYSICS_CHART_HN_ATOM_INDEX)
            )
            correction = (
                int(chart["sign"])
                * float(chart["magnitude_ppm"])
                * features[..., center_index]
                * target_gate[:, None].to(features.dtype)
            )
            if correction.detach().abs().amax() > 0.0050000001:
                raise RuntimeError("physics-chart correction exceeded 0.005 ppm")
            support_mean = support_mean + correction
        support_log_scale = standardized_log_scale + batch["stage_a_scale"].clamp_min(
            1.0e-6
        ).log()[:, None]
        mean = torch.einsum("bk,bkt->bt", q, support_mean)
        within = torch.einsum("bk,bkt->bt", q, torch.exp(2.0 * support_log_scale))
        between = (
            torch.einsum("bk,bkt->bt", q, support_mean.square()) - mean.square()
        ).clamp_min(0.0)
        scale = (within + 0.5 * between).clamp_min(1.0e-6).sqrt()
        safe_target = batch["target_values"].masked_fill(~scoring, 0.0)
        safe_mean = mean.masked_fill(~scoring, 0.0)
        residual = (safe_target - safe_mean) / scale
        nll = torch.log(scale) + 2.5 * torch.log1p(residual.square() / 4.0)
        expected_weight = V6_MODEL.balanced_target_weights(
            batch["stratum_index"], batch["atom_index"], scoring
        ).to(device=mean.device, dtype=mean.dtype)
        supplied_weight = batch["target_weights"].to(device=mean.device, dtype=mean.dtype)
        if not torch.allclose(
            supplied_weight, expected_weight, atol=1.0e-7, rtol=1.0e-7
        ):
            raise ValueError("target weights violate the frozen v6 balancing contract")
        target_weight = supplied_weight.masked_fill(~scoring, 0.0)
        target_weight = target_weight / target_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        data_loss = (nll * target_weight).sum(dim=1).mean()
        prior_log = torch.log_softmax(batch["prior_logits"], dim=1)
        prior_kl = (q * (q.clamp_min(1.0e-12).log() - prior_log)).sum(dim=1).mean()
        loss = (
            data_loss
            + self.prior_kl_weight * prior_kl
            + self.identity_tether_weight * actuator_terms["identity_tether"]
            + self.backbone_smoothness_weight * actuator_terms["backbone_smoothness"]
            + self.physics_clash_weight * actuator_terms["clash_penalty"]
        )
        return {
            "loss": loss,
            "data_loss": data_loss,
            "prior_kl": prior_kl,
            "q": q,
            "support_mean": support_mean,
            "ensemble_mean": mean,
            "ensemble_scale": scale,
            "direct_literal_support_mask": direct_mask,
            "direct_literal_target_mask": direct_mask.any(dim=1),
            "protonation_gap_target_mask": (batch["nucleus_index"] == 0) & ~direct_mask.any(dim=1),
            "torsion_identity_tether": actuator_terms["identity_tether"],
            "backbone_smoothness": actuator_terms["backbone_smoothness"],
            "proposal_torsion_identity_tether": actuator_terms["proposal_identity_tether"],
            "proposal_backbone_smoothness": actuator_terms["proposal_backbone_smoothness"],
            "actuated_clash_penalty": actuator_terms["clash_penalty"],
            "actuated_minimum_nonbonded_distance_nm": actuator_terms["minimum_nonbonded_distance_nm"],
            "actuated_minimum_nonbonded_radius_ratio": actuator_terms["minimum_nonbonded_radius_ratio"],
            "actuated_minimum_same_residue_distance_nm": actuator_terms[
                "minimum_same_residue_distinct_atom_distance_nm"
            ],
            "actuated_maximum_declared_bond_violation_nm": actuator_terms["maximum_declared_bond_range_violation_nm"],
            "actuated_maximum_declared_bond_change_nm": actuator_terms["maximum_declared_bond_length_change_nm"],
            "actuated_feasibility_scale": actuator_terms["feasibility_scale"],
            "actuated_feasibility_ladder_index": actuator_terms["feasibility_ladder_index"],
            "actuated_feasibility_candidate_count": actuator_terms["feasibility_candidate_count"],
            "actuated_feasibility_identity_selected": actuator_terms["feasibility_identity_selected"],
            "actuated_feasibility_full_step_feasible": actuator_terms["feasibility_full_step_feasible"],
            "actuated_feasibility_candidate_gate_failure_codes": actuator_terms[
                "feasibility_candidate_gate_failure_codes"
            ],
            "unprojected_minimum_nonbonded_distance_nm": actuator_terms.get(
                "unprojected_minimum_nonbonded_distance_nm",
                actuator_terms["minimum_nonbonded_distance_nm"],
            ),
            "unprojected_minimum_nonbonded_radius_ratio": actuator_terms.get(
                "unprojected_minimum_nonbonded_radius_ratio",
                actuator_terms["minimum_nonbonded_radius_ratio"],
            ),
            "unprojected_minimum_same_residue_distance_nm": actuator_terms.get(
                "unprojected_minimum_same_residue_distinct_atom_distance_nm",
                actuator_terms["minimum_same_residue_distinct_atom_distance_nm"],
            ),
            "unprojected_maximum_declared_bond_violation_nm": actuator_terms.get(
                "unprojected_maximum_declared_bond_range_violation_nm",
                actuator_terms["maximum_declared_bond_range_violation_nm"],
            ),
            "unprojected_maximum_declared_bond_change_nm": actuator_terms.get(
                "unprojected_maximum_declared_bond_length_change_nm",
                actuator_terms["maximum_declared_bond_length_change_nm"],
            ),
            "effective_torsion_mask": effective_torsion_mask,
            "actuated_atom27_positions": positions,
            "actuated_torsion_angles": torsion_angles,
            "actuated_torsion_delta": actuator_terms["torsion_delta"],
            "proposed_torsion_delta": actuator_terms["proposal_torsion_delta"],
        }


def target_topology_arrays(
    target_rows: Iterable[dict[str, str]],
    comps: tuple[str, ...],
    support_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map target identity metadata to literal slots without reading values."""

    rows = list(target_rows)
    if support_mask.ndim != 3 or support_mask.shape[1:] != (len(comps), ATOM_COUNT):
        raise ValueError("support mask must be [K,L,27]")
    structure: list[int] = []
    parent: list[int] = []
    equivalent: list[tuple[int, int, int]] = []
    for row in rows:
        residue = int(row["seq_id"]) - 1
        if not 0 <= residue < len(comps) or row["comp_id"] != comps[residue]:
            raise ValueError("target residue identity differs from structural sequence")
        slot = resolve_structure_slot(row["comp_id"], row["atom_name"], row["nucleus"])
        parent_slot = resolve_parent_slot(row["comp_id"], row["atom_name"], row["nucleus"])
        if slot < 0 or parent_slot < 0:
            raise ValueError(
                f"target lacks an exact atom27 union slot: {row['comp_id']}|{row['atom_name']}"
            )
        structure.append(slot)
        parent.append(parent_slot)
        equivalent.append(
            equivalent_structure_slots(
                row["comp_id"], row["atom_name"], row["nucleus"],
                row.get("methyl_equivalence_group", ""),
            )
        )
    residues = np.asarray([int(row["seq_id"]) - 1 for row in rows], dtype=np.int64)
    structure_array = np.asarray(structure, dtype=np.int64)
    direct = support_mask[:, residues, structure_array]
    return {
        "structure_slot": structure_array,
        "bonded_parent_slot": np.asarray(parent, dtype=np.int64),
        "equivalent_structure_slots": np.asarray(equivalent, dtype=np.int64),
        "direct_literal_support_mask": direct,
        "direct_literal_target_mask": direct.any(axis=0),
        "all_supports_literal_target_mask": direct.all(axis=0),
    }


def phase_d2_contract() -> dict[str, Any]:
    return {
        "artifact_kind": "v3339_phase_d2_atom27_contract_v2",
        "parent_stage": "v3339_phase_d1_atom27_candidate_v1_frozen",
        "parent_d1_atom27_source_sha256": PARENT_D1_ATOM27_SOURCE_SHA256,
        "parent_support_source_sha256": PINNED_V6_SUPPORT_SHA256,
        "parent_v6_model_source_sha256": PARENT_V6_MODEL_SHA256,
        "shared_projection_model_source_sha256": PINNED_V6_MODEL_SHA256,
        "stage_c_v6_failure_receipt_sha256": STAGE_C_V6_FAILURE_RECEIPT_SHA256,
        "coordinate_scope": COORDINATE_SCOPE,
        "atom_slots": ATOM_COUNT,
        "populated_heavy_atom14_slots_preserved": True,
        "literal_h_aliases": {"HN": "H"},
        "heavy_parent_fallback": False,
        "missing_literal_sites_are_masked_not_dropped": True,
        "uniform_protonation_or_tautomer_claim": False,
        "cached_h_provenance": {
            "OpenMM 7.7, 2026-03-14": 864,
            "OpenMM 7.7, 2026-03-15": 304,
            "OpenMM 8.2, 2026-03-15": 28,
            "force_field_ph_assignment_pinned": False,
        },
        "support_contract": {
            "support_manifest_sha256": PINNED_SUPPORT_MANIFEST_SHA256,
            "coordinate_receipts_sha256": PINNED_COORDINATE_RECEIPTS_SHA256,
            "supported_entity_uid_roster_sha256": PINNED_SUPPORT_ROSTER_SHA256,
            "support_key_count": EXPECTED_SUPPORT_KEY_COUNT,
            "supported_entity_count": EXPECTED_SUPPORTED_ENTITY_COUNT,
            "base_support_count": EXPECTED_BASE_SUPPORT_COUNT,
            "rotamer_support_count": EXPECTED_ROTAMER_SUPPORT_COUNT,
            "total_support_count": EXPECTED_TOTAL_SUPPORT_COUNT,
            "train_k": 8,
            "evaluation_k": 32,
            "hierarchical_base_prior_preserved": True,
        },
        "evaluation_contract": {
            "outer_folds": [0, 1, 2],
            "epochs": 40,
            "scoring_row_count": EXPECTED_SCORING_ROW_COUNT,
            "scoring_surface_sha256": PINNED_SCORING_SURFACE_SHA256,
            "canonical_target_identity_sha256": PINNED_TARGET_IDENTITY_SHA256,
            "primary_atom_count": 29,
            "primary_h_atom_count": 18,
            "primary_h_scoring_rows": EXPECTED_PRIMARY_H_ROW_COUNT,
            "all_scored_h_atom_count": 38,
            "all_scored_h_rows": EXPECTED_SCORING_H_ROW_COUNT,
            "direct_literal_scored_h_rows": EXPECTED_DIRECT_LITERAL_SCORING_H_ROW_COUNT,
            "protonation_gap_scored_h_rows": EXPECTED_PROTONATION_GAP_SCORING_H_ROW_COUNT,
            "protonation_gap_scoring_sites": ["GLU|HE2", "HIS|HD1"],
            "primary_goal": "overall29_and_H_C_N_primary_macro_CCC_each_at_least_0.98",
            "gap_rows_neither_dropped_nor_heavy_parent_substituted": True,
        },
        "methyl_observer_contract": {
            "permutation_invariant_coordinate_pooling": True,
            "fixed_scoring_methyl_rows": 3_518,
            "methylene_H2_H3_remain_distinct": True,
        },
        "v6_q_initialization_transfer": {
            "classification": "initialization_transfer_not_lineage_continuation",
            "transferable_state_prefixes": ["evidence.", "support.", "measure."],
            "fresh_atom27_components": ["observer", "actuator"],
            "old_atom14_observer_is_not_reused": True,
            "old_atom14_performance_is_not_attributed_to_atom27": True,
            "atom27_observer_and_shared_q_require_new_long_training": True,
            "support_summary_uses_heavy_atom14_slots_only": True,
            "literal_h_in_historically_empty_atom14_slots_is_excluded": True,
            "support_feature_semantics_match_v6": True,
        },
        "feasibility_projection": {
            "scope": "one_global_scale_per_entity_support_tensor",
            "fixed_scale_ladder": list(V6_MODEL.FEASIBILITY_SCALE_LADDER),
            "selection": "first_feasible_scale_in_declared_order",
            "candidate_gate_scan_has_no_gradient_graph": True,
            "selected_candidate_is_recomputed_with_gradient": True,
            "identity_is_final_explicit_candidate": True,
            "identity_returns_input_coordinates_exactly": True,
            "identity_gate_failure_is_fatal": True,
            "silent_support_drop_or_heavy_parent_fallback": False,
            "proposal_and_selected_delta_are_both_receipted": True,
            "unprojected_gate_metrics_are_receipted": True,
            "candidate_failure_bit_order": [
                {"bit": index, "metric": key, "direction": direction, "limit": limit}
                for index, (key, direction, limit) in enumerate(ATOM27_HARD_GATE_SPECS)
            ],
            "motivating_v6_failure": {
                "completed_optimizer_steps": 12,
                "maximum_absolute_torsion_delta_radians": 0.011193563230335712,
                "maximum_absolute_coordinate_component_change_nm": 1.0021336078643799,
                "sole_failed_invariant": "minimum_nonbonded_distance_nm",
                "unprojected_minimum_nonbonded_distance_nm": 0.05636809021234512,
                "v6_gate_minimum_nonbonded_distance_nm": 0.06,
            },
        },
        "target_values_accessed": False,
        "sealed_targets_accessed": False,
    }


def phase_d1_contract() -> dict[str, Any]:
    """Compatibility entrypoint; v2 callers receive the explicit v2 contract."""

    return phase_d2_contract()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _raw_pdb_inventory(path: Path) -> dict[str, Any]:
    atom_count = heavy_count = hydrogen_count = residue_count = 0
    h_names: set[str] = set()
    comp_h_pairs: set[tuple[str, str]] = set()
    per_residue: Counter[tuple[str, str, str, str]] = Counter()
    provenance = "unreported"
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "CREATED WITH OPENMM" in line:
                provenance = line.strip().removeprefix("REMARK   1 ")
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54 or line[16] not in {" ", "A"}:
                continue
            comp = line[17:20].strip().upper()
            if comp not in AA3_TO_AA1:
                continue
            element = _pdb_element(line)
            atom = line[12:16].strip().upper()
            atom_count += 1
            residue_key = (
                line[21].strip() or "_", line[22:26].strip(), line[26].strip(), comp
            )
            per_residue[residue_key] += 1
            if element == "H":
                hydrogen_count += 1
                h_names.add(atom)
                comp_h_pairs.add((comp, atom))
            else:
                heavy_count += 1
    residue_count = len(per_residue)
    return {
        "atom_count": atom_count,
        "heavy_count": heavy_count,
        "hydrogen_count": hydrogen_count,
        "residue_count": residue_count,
        "h_names": h_names,
        "comp_h_pairs": comp_h_pairs,
        "maximum_simultaneous_atoms": max(per_residue.values(), default=0),
        "provenance": provenance,
    }


def audit_selected_supports(
    coordinate_receipts_path: Path,
    support_registry_path: Path,
    support_root: Path,
) -> dict[str, Any]:
    """Replay all selected states target-blindly and bind heavy lineage."""

    if sha256_file(coordinate_receipts_path) != PINNED_COORDINATE_RECEIPTS_SHA256:
        raise ValueError("coordinate receipt SHA256 drift")
    receipts = _read_tsv(coordinate_receipts_path)
    registry = _read_tsv(support_registry_path)
    if len(registry) != EXPECTED_SUPPORT_KEY_COUNT or len(receipts) != EXPECTED_TOTAL_SUPPORT_COUNT:
        raise ValueError("frozen support inventory count drift")
    sequence_by_key = {row["support_key"]: row["sequence"] for row in registry}
    if len(sequence_by_key) != len(registry):
        raise ValueError("support registry keys are not unique")
    receipt_by_id = {row["support_id"]: row for row in receipts}
    if len(receipt_by_id) != len(receipts):
        raise ValueError("support receipt IDs are not globally unique")
    base_rows = [row for row in receipts if row["support_provenance"] == "af_cached"]
    child_rows = [
        row for row in receipts
        if row["support_provenance"] == "target_blind_canonical_rotamer_augmentation"
    ]
    if len(base_rows) != EXPECTED_BASE_SUPPORT_COUNT or len(child_rows) != EXPECTED_ROTAMER_SUPPORT_COUNT:
        raise ValueError("base/rotamer support count drift")

    raw_totals = Counter()
    h_names: set[str] = set()
    comp_h_pairs: set[tuple[str, str]] = set()
    provenance_counts: Counter[str] = Counter()
    maximum_simultaneous = 0
    pdbs_with_h = 0
    parsed_base: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], str]] = {}
    minimum_xh = float("inf")
    maximum_xh = 0.0
    for row in base_rows:
        path = Path(row["pdb_path"])
        if sha256_file(path) != row["pdb_sha256"]:
            raise ValueError(f"selected PDB SHA256 drift: {path}")
        raw = _raw_pdb_inventory(path)
        raw_totals.update({
            "atom_count": raw["atom_count"],
            "heavy_count": raw["heavy_count"],
            "hydrogen_count": raw["hydrogen_count"],
            "residue_count": raw["residue_count"],
        })
        h_names.update(raw["h_names"])
        comp_h_pairs.update(raw["comp_h_pairs"])
        provenance_counts[raw["provenance"]] += 1
        maximum_simultaneous = max(maximum_simultaneous, raw["maximum_simultaneous_atoms"])
        pdbs_with_h += int(raw["hydrogen_count"] > 0)
        sequence = sequence_by_key[row["support_key"]]
        positions, mask, comps = parse_pdb_atom27(path, sequence)
        state_receipt = validate_atom27_state(positions, mask, comps)
        minimum_xh = min(minimum_xh, float(state_receipt["minimum_xh_bond_nm"]))
        maximum_xh = max(maximum_xh, float(state_receipt["maximum_xh_bond_nm"]))
        if heavy_atom14_coordinate_sha256(positions, comps) != row["coordinate_sha256"]:
            raise ValueError("base atom27 replay changed frozen heavy atom14 coordinates")
        parsed_base[row["support_id"]] = (positions, mask, comps, sequence)

    replay_count = 0
    maximum_heavy_abs_error = 0.0
    maximum_xh_bond_change = 0.0
    maximum_prior_error = 0.0
    base_angles_by_id: dict[str, np.ndarray] = {}
    for registry_row in registry:
        bundle_path = support_root / registry_row["support_bundle"]
        if sha256_file(bundle_path) != registry_row["support_bundle_sha256"]:
            raise ValueError("frozen atom14 support bundle SHA256 drift")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            support_ids = [str(item) for item in bundle["support_ids"]]
            base_ids = [str(item) for item in bundle["base_support_ids"]]
            old_positions = np.asarray(bundle["atom14_positions"], dtype=np.float32)
            old_mask = np.asarray(bundle["atom14_mask"], dtype=np.bool_)
            prior = np.exp(np.asarray(bundle["prior_logits"], dtype=np.float64))
            if (
                len(support_ids) != 32
                or len(base_ids) != 32
                or len(set(base_ids)) != int(registry_row["base_support_count"])
            ):
                raise ValueError("frozen K32/base lineage drift")
            base_counts = Counter(base_ids)
            expected_prior = np.asarray(
                [1.0 / (len(base_counts) * base_counts[item]) for item in base_ids]
            )
            maximum_prior_error = max(
                maximum_prior_error, float(np.max(np.abs(prior - expected_prior)))
            )
            for state_index, support_id in enumerate(support_ids):
                row = receipt_by_id[support_id]
                base_id = base_ids[state_index]
                base_positions, base_mask, comps, _ = parsed_base[base_id]
                if row["support_provenance"] == "af_cached":
                    positions = base_positions
                else:
                    axes, moving = atom27_torsion_topology(comps)
                    residue = int(row["rotamer_residue_index"])
                    torsion = int(row["rotamer_torsion"])
                    rounded_target = float(row["rotamer_target_radians"])
                    target = min(
                        V6_SUPPORT.CANONICAL_CHI_BASINS,
                        key=lambda value: abs(value - rounded_target),
                    )
                    if abs(target - rounded_target) > 1.0e-8:
                        raise ValueError("rotamer receipt is not a canonical chi basin")
                    if base_id not in base_angles_by_id:
                        base_angles_by_id[base_id] = atom27_torsions(
                            base_positions, base_mask, comps
                        )[0]
                    base_angles = base_angles_by_id[base_id]
                    delta = math.atan2(
                        math.sin(target - float(base_angles[residue, torsion])),
                        math.cos(target - float(base_angles[residue, torsion])),
                    )
                    positions = rotate_atom27_delta(
                        base_positions, base_mask, axes, moving, comps,
                        residue=residue, torsion=torsion, delta_radians=delta,
                    )
                    replay_count += 1
                    atom_names, parents = atom27_tables()
                    for res, comp in enumerate(comps):
                        for slot, parent in enumerate(parents[comp]):
                            if (
                                parent < 0 or not atom_names[comp][slot].startswith("H")
                                or not base_mask[res, slot]
                            ):
                                continue
                            before = np.linalg.norm(
                                base_positions[res, slot] - base_positions[res, parent]
                            )
                            after = np.linalg.norm(positions[res, slot] - positions[res, parent])
                            maximum_xh_bond_change = max(
                                maximum_xh_bond_change, float(abs(after - before))
                            )
                heavy_names, _ = V6_SUPPORT.atom14_tables()
                expected_heavy_mask = np.asarray(
                    [[bool(atom) for atom in heavy_names[comp]] for comp in comps],
                    dtype=np.bool_,
                )
                if not np.array_equal(old_mask[state_index], expected_heavy_mask):
                    raise ValueError("frozen heavy atom14 mask differs from residue roster")
                error = np.max(
                    np.abs(
                        positions[:, :14][expected_heavy_mask]
                        - old_positions[state_index][expected_heavy_mask]
                    ),
                    initial=0.0,
                )
                maximum_heavy_abs_error = max(maximum_heavy_abs_error, float(error))
                if heavy_atom14_coordinate_sha256(positions, comps) != row["coordinate_sha256"]:
                    raise ValueError(
                        "atom27 replay changed a frozen heavy coordinate receipt: "
                        f"{support_id} max_abs_error_nm={float(error):.9g}"
                    )

    exact_provenance = {
        "CREATED WITH OPENMM 7.7, 2026-03-14": 864,
        "CREATED WITH OPENMM 7.7, 2026-03-15": 304,
        "CREATED WITH OPENMM 8.2, 2026-03-15": 28,
    }
    if (
        raw_totals["residue_count"] != EXPECTED_SELECTED_RESIDUE_COUNT
        or raw_totals["heavy_count"] != EXPECTED_SELECTED_HEAVY_RECORD_COUNT
        or raw_totals["hydrogen_count"] != EXPECTED_SELECTED_H_RECORD_COUNT
        or len(h_names) != EXPECTED_LITERAL_H_NAME_COUNT
        or len(comp_h_pairs) != EXPECTED_OBSERVED_LITERAL_COMP_H_PAIR_COUNT
        or maximum_simultaneous != EXPECTED_MAX_SIMULTANEOUS_ATOMS
        or dict(provenance_counts) != exact_provenance
        or replay_count != EXPECTED_ROTAMER_SUPPORT_COUNT
        or maximum_heavy_abs_error != 0.0
        or maximum_xh_bond_change > 2.0e-6
        or maximum_prior_error > 2.0e-7
    ):
        raise ValueError(
            "live explicit-H support inventory/replay contract drift: "
            + json.dumps({
                "residues": raw_totals["residue_count"],
                "heavy": raw_totals["heavy_count"],
                "hydrogen": raw_totals["hydrogen_count"],
                "h_names": len(h_names),
                "comp_h_pairs": len(comp_h_pairs),
                "max_simultaneous": maximum_simultaneous,
                "provenance": dict(provenance_counts),
                "replay_count": replay_count,
                "max_heavy_error": maximum_heavy_abs_error,
                "max_xh_change": maximum_xh_bond_change,
                "max_prior_error": maximum_prior_error,
            }, sort_keys=True)
        )
    return {
        "artifact_kind": "v3339_phase_d1_target_blind_inventory_v1",
        "coordinate_scope": COORDINATE_SCOPE,
        "selected_pdb_count": len(base_rows),
        "pdbs_with_literal_h": pdbs_with_h,
        "selected_residue_count": raw_totals["residue_count"],
        "selected_atom_record_count": raw_totals["atom_count"],
        "selected_heavy_record_count": raw_totals["heavy_count"],
        "selected_literal_h_record_count": raw_totals["hydrogen_count"],
        "literal_h_fraction": raw_totals["hydrogen_count"] / raw_totals["atom_count"],
        "literal_h_names": sorted(h_names),
        "literal_comp_h_pair_count": len(comp_h_pairs),
        "maximum_simultaneous_pdb_atoms_per_residue": maximum_simultaneous,
        "maximum_atom27_union_slots": EXPECTED_MAX_UNION_ATOMS,
        "minimum_xh_bond_nm": minimum_xh,
        "maximum_xh_bond_nm": maximum_xh,
        "base_support_count": len(base_rows),
        "rotamer_support_count_replayed": replay_count,
        "total_support_count": len(receipts),
        "maximum_heavy_replay_abs_error_nm": maximum_heavy_abs_error,
        "maximum_xh_bond_change_nm": maximum_xh_bond_change,
        "maximum_hierarchical_prior_abs_error": maximum_prior_error,
        "cached_h_provenance": dict(sorted(provenance_counts.items())),
        "uniform_protonation_or_tautomer_claim": False,
        "target_values_accessed": False,
        "sealed_targets_accessed": False,
    }


def _write_new_private_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError("receipt output must be a new file under an existing directory")
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    staged = Path(temporary)
    try:
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    contract = subparsers.add_parser("write_contract")
    contract.add_argument("--output", required=True, type=Path)
    inventory = subparsers.add_parser("audit_inventory")
    inventory.add_argument("--coordinate-receipts", required=True, type=Path)
    inventory.add_argument("--support-registry", required=True, type=Path)
    inventory.add_argument("--support-root", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.operation == "write_contract":
        payload = phase_d2_contract()
    else:
        payload = audit_selected_supports(
            args.coordinate_receipts, args.support_registry, args.support_root
        )
    _write_new_private_json(args.output, payload)
    print(json.dumps({
        "artifact_kind": payload["artifact_kind"],
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
