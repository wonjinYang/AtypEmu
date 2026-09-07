# ruff: noqa: TRY004
"""Materialize one frozen target-unread ff15ipq all-support entity archive.

Production paths are derived from this script's staged location.  The only
public production selector is ``--entity-uid``; one fresh spawned worker emits
each support-condition state.  ``--self-test`` does not import OpenMM or read
cohort inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import stat
import sys
import tarfile
import types
from itertools import pairwise, product
from pathlib import Path
from typing import Any


def _load_bound_common() -> types.ModuleType:
    script = Path(__file__).absolute()
    stage = script.parent.parent
    if "--self-test" not in sys.argv and (
        script.is_symlink()
        or script.resolve(strict=True) != script
        or os.environ.get("ATYPEMU_STAGE_ROOT") != str(stage)
    ):
        raise PermissionError("producer is outside its launcher-bound stage")
    if script.is_symlink() or script.resolve(strict=True) != script:
        raise PermissionError("producer source path is indirect")
    common_path = script.parent / "ff15ipq_all_support_common.py"
    descriptor = os.open(common_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        raw = os.read(descriptor, info.st_size + 1)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or len(raw) != info.st_size
        or hashlib.sha256(raw).hexdigest()
        != "fa45271637ba6f5868ce78b7d32e6f4dbad06340536d53d1d5ca2bd694438935"
    ):
        raise PermissionError("pre-import common implementation hash drifted")
    module_name = "bound_ff15ipq_all_support_common"
    module = types.ModuleType(module_name)
    module.__file__ = str(common_path)
    sys.modules[module_name] = module
    exec(compile(raw, str(common_path), "exec"), module.__dict__)  # noqa: S102
    return module


common = _load_bound_common()

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
COMMITMENT_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1.json"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1"
)
PLAN_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_plan_v1.json"
)
PROJECTION_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1.json.gz"
)
PROJECTION_SHA256 = "e9b4861216d99e8136df872958a50f568337fc8a6b4991b25df2b906dcee5046"
OPENMM_VERSION = "8.6.0.dev-c6173db"
FORCEFIELD_PATH = "amber14/protein.ff15ipq.xml"
RUNTIME_SIF_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
STATE_FIELDS = {
    "bmrb_id",
    "condition_branch_id",
    "condition_state",
    "entity_uid",
    "parent_raw_pdb_sha256",
    "proposal_pH",
    "support_index",
}
CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stage_paths() -> tuple[Path, Path, Path]:
    scripts = Path(__file__).resolve().parent
    stage = scripts.parent
    return scripts, stage / "inputs", stage / "outputs"


def _read_regular(path: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"invalid regular input: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1_048_576, maximum_bytes + 1)):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"input too large: {path.name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"input changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    return common.parse_json(raw, label)


def _safe_relative(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("unsafe relative path")
    return path.parts


def _read_beneath(root: Path, relative: str, maximum_bytes: int = 8_000_000) -> bytes:
    parts = _safe_relative(relative)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor
        )
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise ValueError("invalid projected PDB")
            raw = b""
            while chunk := os.read(file_descriptor, min(1_048_576, maximum_bytes + 1)):
                raw += chunk
                if len(raw) > maximum_bytes:
                    raise ValueError("projected PDB exceeds size ceiling")
            after = os.fstat(file_descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("projected PDB changed while reading")
            return raw
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)


def _require_source(scripts: Path, inputs: Path) -> None:
    commitment_raw = _read_regular(scripts / COMMITMENT_NAME, 1_000_000)
    expected_commitment = os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
    if (
        expected_commitment is None
        or common.SHA256_RE.fullmatch(expected_commitment) is None
        or sha256(commitment_raw) != expected_commitment
        or os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256") != RUNTIME_SIF_SHA256
        or os.environ.get("ATYPEMU_STAGE_ROOT") != str(scripts.parent)
    ):
        raise PermissionError(
            "external source/runtime/stage binding is absent or wrong"
        )
    commitment = _parse_object(commitment_raw, "source commitment")
    if set(commitment) != {"candidate_id", "contract", "files", "git_commit", "status"}:
        raise ValueError("source commitment schema drifted")
    if not (
        commitment["candidate_id"] == CANDIDATE_ID
        and commitment["contract"] == COMMITMENT_CONTRACT
        and commitment["status"] == "FROZEN_COMMITTED_BOUNDED_SMOKE_ONLY"
        and common.GIT_COMMIT_RE.fullmatch(commitment["git_commit"])
        and os.environ.get("ATYPEMU_GIT_COMMIT") == commitment["git_commit"]
    ):
        raise PermissionError("all-support source is not frozen")
    expected = {
        "gpuopt/candidates/materialize_ff15ipq_all_support_entity.py": Path(
            __file__
        ).name,
        "gpuopt/candidates/check_ff15ipq_all_support_entity.py": "check_ff15ipq_all_support_entity.py",
        "gpuopt/candidates/ff15ipq_all_support_common.py": "ff15ipq_all_support_common.py",
        "gpuopt/candidates/check_ff15ipq_all_support_qualification_plan.py": "check_ff15ipq_all_support_qualification_plan.py",
        "gpuopt/candidates/launch_ff15ipq_all_support_qualification.py": "launch_ff15ipq_all_support_qualification.py",
        f"gpuopt/preunblind/{PLAN_NAME}": PLAN_NAME,
        f"gpuopt/preunblind/{PROJECTION_NAME}": PROJECTION_NAME,
        "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_all_support_implementation_cold_review_receipt_v1.json": "atypemu_nested_support_count_v1_ff15ipq_all_support_implementation_cold_review_receipt_v1.json",
        "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_all_support_plan_validation_receipt_v1.json": "atypemu_nested_support_count_v1_ff15ipq_all_support_plan_validation_receipt_v1.json",
    }
    files = commitment["files"]
    if not isinstance(files, dict) or set(files) != set(expected):
        raise ValueError("source commitment inventory drifted")
    for repository_path, staged_name in expected.items():
        root = inputs if staged_name == PROJECTION_NAME else scripts
        raw = _read_regular(root / staged_name, 8_000_000)
        wanted = files[repository_path]
        if not isinstance(wanted, str) or sha256(raw) != wanted:
            raise ValueError(f"source commitment hash mismatch: {repository_path}")
    plan = _parse_object(_read_regular(scripts / PLAN_NAME, 1_000_000), "plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("state") == "HOLD_EXTERNAL_SOURCE_COMMITMENT_PENDING_UNRUN"
        and plan.get("execution_contract", {}).get("source_commitment")
        == "EXTERNAL_O_EXCL_PENDING"
    ):
        raise PermissionError("frozen plan has not released this execution lane")


def _require_smoke_release(scripts: Path, entity_uid: str) -> None:
    release_root = scripts.parent / "release"
    release_raw = _read_regular(release_root / "execution_release.json", 1_000_000)
    release_hash = sha256(release_raw)
    if os.environ.get("ATYPEMU_EXECUTION_RELEASE_SHA256") != release_hash:
        raise PermissionError("execution release environment binding drifted")
    release = _parse_object(release_raw, "execution release")
    if release_raw != canonical_bytes(release) + b"\n" or set(release) != {
        "bioemu_root",
        "candidate_id",
        "closed_capabilities",
        "contract",
        "entities",
        "git_commit",
        "mode",
        "review_receipt_sha256",
        "runtime_sif_sha256",
        "source_commitment_sha256",
        "state",
    }:
        raise PermissionError("execution release schema or framing drifted")
    if not (
        release["candidate_id"] == CANDIDATE_ID
        and release["closed_capabilities"] == CLOSED_CAPABILITIES
        and release["contract"]
        == "atypemu_nested_support_count_v1_ff15ipq_all_support_execution_release_v1"
        and release["entities"] == [entity_uid]
        and release["git_commit"] == os.environ.get("ATYPEMU_GIT_COMMIT")
        and release["mode"] == "BOUNDED_SMOKE"
        and release["runtime_sif_sha256"]
        == os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256")
        and release["source_commitment_sha256"]
        == os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
        and release["state"] == "EXTERNALLY_RELEASED_ONCE"
        and os.environ.get("ATYPEMU_RELEASED_ENTITY_UID") == entity_uid
    ):
        raise PermissionError("execution release identity or bounded scope drifted")
    review_raw = _read_regular(release_root / "source_cold_review.json", 1_000_000)
    if sha256(review_raw) != release["review_receipt_sha256"]:
        raise PermissionError("source cold-review receipt hash drifted")
    consumed_raw = _read_regular(release_root / "execution_consumed.json", 1_000_000)
    consumed = _parse_object(consumed_raw, "execution consumption")
    if consumed_raw != canonical_bytes(consumed) + b"\n" or not (
        consumed.get("candidate_id") == CANDIDATE_ID
        and consumed.get("contract")
        == "atypemu_ff15ipq_all_support_execution_consumption_v1"
        and consumed.get("execution_release_sha256") == release_hash
        and consumed.get("mode") == "BOUNDED_SMOKE"
        and consumed.get("source_commitment_sha256")
        == os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
        and consumed.get("state") == "CONSUMED_BEFORE_TARGET_UNREAD_PDB_ACCESS"
    ):
        raise PermissionError("execution consumption binding drifted")


def _load_projection(inputs: Path) -> dict[str, Any]:
    raw = _read_regular(inputs / PROJECTION_NAME, 8_000_000)
    if sha256(raw) != PROJECTION_SHA256:
        raise ValueError("runtime input projection hash drifted")
    try:
        decoded = gzip.decompress(raw)
    except (EOFError, OSError) as error:
        raise ValueError("runtime input projection gzip is invalid") from error
    projection = _parse_object(decoded, "runtime input projection")
    if not (
        projection.get("candidate_id") == CANDIDATE_ID
        and projection.get("contract")
        == "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1"
        and projection.get("source_path_template")
        == "data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb"
        and projection.get("runtime_allowed_fields")
        == sorted(STATE_FIELDS | {"missing_support_indices"})
    ):
        raise ValueError("runtime input projection contract drifted")
    return projection


def _entity_states(projection: dict[str, Any], entity_uid: str) -> list[dict[str, Any]]:
    entries = projection.get("entries")
    if not isinstance(entries, list):
        raise ValueError("projection entity inventory is invalid")
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("entity_uid") == entity_uid
    ]
    if len(matches) != 1:
        raise ValueError("entity_uid does not select exactly one projection entity")
    entity = matches[0]
    missing = entity.get("missing_support_indices")
    supports = entity.get("supports")
    branches = entity.get("condition_branches")
    if (
        not isinstance(missing, list)
        or any(type(index) is not int for index in missing)
        or missing != sorted(set(missing))
        or not isinstance(supports, list)
        or not isinstance(branches, list)
    ):
        raise ValueError("projection entity state or missing inventory is invalid")
    output: list[dict[str, Any]] = []
    for support in supports:
        if (
            not isinstance(support, dict)
            or set(support) != {"parent_raw_pdb_sha256", "support_index"}
            or type(support["support_index"]) is not int
            or not 1 <= support["support_index"] <= 1000
            or not isinstance(support["parent_raw_pdb_sha256"], str)
            or common.SHA256_RE.fullmatch(support["parent_raw_pdb_sha256"]) is None
        ):
            raise ValueError("projected support index is invalid")
        for branch in branches:
            if (
                not isinstance(branch, dict)
                or set(branch) != {"condition_branch_id", "proposal_pH"}
                or not isinstance(branch["condition_branch_id"], str)
                or not isinstance(branch["proposal_pH"], str)
            ):
                raise ValueError("projected condition branch is invalid")
            output.append(
                {
                    "bmrb_id": entity["bmrb_id"],
                    "condition_branch_id": branch["condition_branch_id"],
                    "condition_state": entity["condition_state"],
                    "entity_uid": entity_uid,
                    "parent_raw_pdb_sha256": support["parent_raw_pdb_sha256"],
                    "proposal_pH": branch["proposal_pH"],
                    "support_index": support["support_index"],
                }
            )
    output.sort(key=lambda row: (row["support_index"], row["condition_branch_id"]))
    expected_supports = sorted(set(range(1, 1001)) - set(missing))
    actual_supports = sorted({row["support_index"] for row in output})
    expected_branches = (
        [
            {
                "condition_branch_id": "observed-0",
                "proposal_pH": branches[0]["proposal_pH"],
            }
        ]
        if entity["condition_state"] == "observed" and len(branches) == 1
        else [
            {"condition_branch_id": f"regime-{index}", "proposal_pH": ph}
            for index, ph in enumerate(("2.2", "5.45", "7.5", "9.25", "12.0"))
        ]
    )
    branch_count = len(expected_branches)
    if (
        branches != expected_branches
        or actual_supports != expected_supports
        or len(output) != len(expected_supports) * branch_count
        or len({(row["support_index"], row["condition_branch_id"]) for row in output})
        != len(output)
    ):
        raise ValueError("projected entity states are incomplete or duplicate")
    return output


def seed_for(state: dict[str, Any]) -> int:
    material = "\0".join(
        (
            CANDIDATE_ID,
            state["entity_uid"],
            str(state["support_index"]),
            state["condition_branch_id"],
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _minimum_distance_cell_list(
    coordinates: list[tuple[float, float, float]], upper_bound: float
) -> float:
    if len(coordinates) < 2:
        return math.inf
    if not math.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError("nearest-neighbor upper bound must be positive and finite")
    width = upper_bound
    cells: dict[tuple[int, int, int], list[int]] = {}
    minimum_sq = upper_bound * upper_bound
    for index, xyz in enumerate(coordinates):
        cell = tuple(math.floor(value / width) for value in xyz)
        # A pair no farther apart than ``width`` can differ by at most two
        # floor cells when one coordinate lies exactly on a boundary.
        for delta in product((-2, -1, 0, 1, 2), repeat=3):
            neighbor = tuple(cell[axis] + delta[axis] for axis in range(3))
            for other in cells.get(neighbor, ()):  # Every unordered pair exactly once.
                prior = coordinates[other]
                distance_sq = sum((xyz[axis] - prior[axis]) ** 2 for axis in range(3))
                minimum_sq = min(minimum_sq, distance_sq)
        cells.setdefault(cell, []).append(index)
    return math.sqrt(minimum_sq)


def _physicality_fast(
    modeller: Any, forcefield: Any, platform: Any, unit: Any, openmm: Any
) -> dict[str, Any]:
    atoms = list(modeller.topology.atoms())
    positions = list(modeller.positions)
    if len(atoms) != len(positions) or not atoms:
        raise ValueError("atom-position inventory invalid")
    coordinates = [
        tuple(float(value) for value in position.value_in_unit(unit.angstrom))
        for position in positions
    ]
    if not all(math.isfinite(value) for xyz in coordinates for value in xyz):
        raise ValueError("non-finite coordinate")
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in atoms}
    atom_index = {atom: index for index, atom in enumerate(atoms)}
    bonded_distances: list[float] = []
    for left, right in modeller.topology.bonds():
        adjacency[left].append(right)
        adjacency[right].append(left)
        left_element = left.element.symbol.upper() if left.element else ""
        right_element = right.element.symbol.upper() if right.element else ""
        distance = common._distance(
            positions[atom_index[left]], positions[atom_index[right]], unit
        )
        bonded_distances.append(distance)
        if left_element != "H" and right_element != "H" and not 1.0 <= distance <= 2.3:
            raise ValueError("heavy-heavy bond distance outside 1.0..2.3 A")
    if not bonded_distances:
        raise ValueError("topology has no bonded nearest-neighbor upper bound")
    minimum_distance = _minimum_distance_cell_list(coordinates, min(bonded_distances))
    if minimum_distance < 0.5:
        raise ValueError("distinct atoms closer than 0.5 A")
    limits = {"H": 1, "C": 4, "N": 4, "O": 2, "S": 6}
    hydrogen_count = 0
    for atom in atoms:
        element = atom.element.symbol.upper() if atom.element else ""
        if element not in limits or len(adjacency[atom]) > limits[element]:
            raise ValueError("invalid elemental valence")
        if element != "H":
            continue
        hydrogen_count += 1
        parents = [
            other
            for other in adjacency[atom]
            if other.element and other.element.atomic_number != 1
        ]
        if len(parents) != 1 or len(adjacency[atom]) != 1:
            raise ValueError("hydrogen lacks exactly one heavy parent")
        parent = parents[0]
        distance = common._distance(
            positions[atom_index[atom]], positions[atom_index[parent]], unit
        )
        if not common.hydrogen_distance_ok(parent.element.symbol.upper(), distance):
            raise ValueError("hydrogen-heavy distance outside physicality bound")
        others = [
            other
            for other in adjacency[parent]
            if other.element and other.element.atomic_number != 1
        ]
        if not others:
            raise ValueError("hydrogen parent lacks a heavy angle reference")
        for other in others:
            a = positions[atom_index[atom]] - positions[atom_index[parent]]
            b = positions[atom_index[other]] - positions[atom_index[parent]]
            av, bv = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom)
            cosine = sum(float(x * y) for x, y in zip(av, bv, strict=True)) / math.sqrt(
                sum(float(x * x) for x in av) * sum(float(y * y) for y in bv)
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if not 55.0 <= angle <= 180.0:
                raise ValueError("hydrogen bond angle outside physicality bound")
    if hydrogen_count == 0:
        raise ValueError("OpenMM produced no hydrogens")
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        for left, right in pairwise(residues):
            left_c = next((atom for atom in left.atoms() if atom.name == "C"), None)
            right_n = next((atom for atom in right.atoms() if atom.name == "N"), None)
            if left_c is None or right_n is None or right_n not in adjacency[left_c]:
                raise ValueError("backbone topology continuity failed")
            distance = common._distance(
                positions[atom_index[left_c]], positions[atom_index[right_n]], unit
            )
            if not 1.0 <= distance <= 1.8:
                raise ValueError("backbone coordinate continuity failed")
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPositions(modeller.positions)
        energy = float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context
        del integrator
    if not math.isfinite(energy):
        raise ValueError("potential energy is non-finite")
    return {
        "atom_count": len(atoms),
        "hydrogen_count": hydrogen_count,
        "minimum_distinct_atom_distance_A": minimum_distance,
        "potential_energy_kj_per_mol": energy,
    }


def _state_path(state: dict[str, Any]) -> str:
    bmrb = state["bmrb_id"]
    index = state["support_index"]
    if not isinstance(bmrb, str) or re.fullmatch(r"bmr[0-9]+", bmrb) is None:
        raise ValueError("invalid BMRB ID")
    return f"data/BioEmu/{bmrb}/{bmrb}_BioEmu_{index}.pdb"


def _worker(
    task: tuple[str, int, dict[str, Any], str],
) -> tuple[int, bytes, bytes]:
    entity_uid, ordinal, state, work_text = task
    if type(ordinal) is not int or not 0 <= ordinal < 5000:
        raise ValueError("worker state ordinal is invalid")
    scripts, inputs, _ = stage_paths()
    work = Path(work_text)
    if work.is_symlink() or not work.is_dir():
        raise ValueError("worker directory is absent or unsafe")
    _require_source(scripts, inputs)
    _require_smoke_release(scripts, entity_uid)
    if set(state) != STATE_FIELDS:
        raise ValueError("worker state schema drifted")
    if state["entity_uid"] != entity_uid:
        raise ValueError("worker entity identity drifted")
    seed = seed_for(state)
    random.seed(seed)
    raw_pdb = _read_beneath(inputs, _state_path(state))
    if sha256(raw_pdb) != state["parent_raw_pdb_sha256"]:
        raise ValueError("projected raw PDB hash drifted")
    source_records, _, _ = common._pdb_records(raw_pdb)
    source_heavy = common._heavy_records(source_records)
    try:
        import numpy  # type: ignore[import-not-found]
        import openmm  # type: ignore[import-not-found]
        from openmm import Platform, app, unit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("frozen OpenMM runtime is required") from error
    numpy.random.seed(seed % (2**32))
    common._require_runtime(openmm, app)
    pdb = app.PDBFile(io.StringIO(raw_pdb.decode("ascii")))
    if len(list(pdb.topology.atoms())) != len(source_records):
        raise ValueError("OpenMM PDB atom inventory differs from raw parser")
    if not common._heavy_records_numerically_exact(
        common._openmm_heavy_records(pdb.topology, pdb.positions, unit), source_heavy
    ):
        raise ValueError("OpenMM PDB parser changed parent heavy records")
    source_keys = {
        tuple(
            row[key]
            for key in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for row in source_heavy
    }
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.delete(
        [
            atom
            for atom in modeller.topology.atoms()
            if common._atom_key(atom) not in source_keys
        ]
    )
    if not common._heavy_records_numerically_exact(
        common._openmm_heavy_records(modeller.topology, modeller.positions, unit),
        source_heavy,
    ):
        raise ValueError("input hydrogen removal changed parent heavy records")
    forcefield = app.ForceField(FORCEFIELD_PATH)
    platform = Platform.getPlatformByName("Reference")
    returned_variants = modeller.addHydrogens(
        forcefield,
        pH=float(state["proposal_pH"]),
        variants=None,
        platform=platform,
    )
    post_heavy = common._openmm_heavy_records(
        modeller.topology, modeller.positions, unit
    )
    if not common._heavy_records_numerically_exact(post_heavy, source_heavy):
        raise ValueError("protonation changed parent heavy records")
    physicality = _physicality_fast(modeller, forcefield, platform, unit, openmm)
    protonation_signature = common._signature(modeller.topology)
    text = io.StringIO(newline="\n")
    app.PDBFile.writeModel(modeller.topology, modeller.positions, text, keepIds=True)
    app.PDBFile.writeFooter(modeller.topology, text)
    pdb_raw = text.getvalue().encode("ascii")
    emitted_records, _, _ = common._pdb_records(pdb_raw)
    if not common._heavy_records_numerically_exact(
        common._heavy_records(emitted_records), source_heavy
    ):
        raise ValueError("emitted PDB changed parent heavy records")
    compressed_pdb = gzip.compress(pdb_raw, compresslevel=9, mtime=0)
    metadata = {
        "condition_branch_id": state["condition_branch_id"],
        "entity_uid": state["entity_uid"],
        "parent_raw_pdb_sha256": state["parent_raw_pdb_sha256"],
        "parent_heavy_coordinate_sha256": common._heavy_hashes(source_records)[1],
        "parent_heavy_topology_sha256": common._heavy_hashes(source_records)[0],
        "proposal_pH": state["proposal_pH"],
        "protonated_gzip_sha256": sha256(compressed_pdb),
        "protonated_raw_pdb_sha256": sha256(pdb_raw),
        "protonation_signature_sha256": sha256(canonical_bytes(protonation_signature)),
        "returned_variants_sha256": sha256(canonical_bytes(list(returned_variants))),
        "support_index": state["support_index"],
        "atom_count": physicality["atom_count"],
        "hydrogen_count": physicality["hydrogen_count"],
        "potential_energy_kj_per_mol": physicality["potential_energy_kj_per_mol"],
    }
    return ordinal, compressed_pdb, canonical_bytes(metadata) + b"\n"


def _exclusive(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def _archive_bytes(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    for name, raw in sorted(members):
        output.write(_tar_info(name, len(raw)).tobuf(format=tarfile.USTAR_FORMAT))
        output.write(raw)
        output.write(bytes((-len(raw)) % 512))
    output.write(bytes(1024))
    return output.getvalue()


def _hash_file(path: Path, maximum_bytes: int = 2_000_000_000) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError("archive file is invalid or too large")
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("archive changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _stream_entity_archive(
    work: Path,
    states: list[dict[str, Any]],
    metadata_raw: bytes,
    archive_path: Path,
) -> None:
    descriptor = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as archive:
            archive.write(
                _tar_info("metadata.jsonl", len(metadata_raw)).tobuf(
                    format=tarfile.USTAR_FORMAT
                )
            )
            archive.write(metadata_raw)
            archive.write(bytes((-len(metadata_raw)) % 512))
            for ordinal, state in enumerate(states):
                name = (
                    f"states/{state['support_index']:04d}/"
                    f"{state['condition_branch_id']}.pdb.gz"
                )
                pdb_raw = _read_regular(work / f"state-{ordinal:05d}.pdb", 8_000_000)
                archive.write(
                    _tar_info(name, len(pdb_raw)).tobuf(format=tarfile.USTAR_FORMAT)
                )
                archive.write(pdb_raw)
                archive.write(bytes((-len(pdb_raw)) % 512))
            archive.write(bytes(1024))
            archive.flush()
            os.fsync(archive.fileno())
    finally:
        os.close(descriptor)
    os.chmod(archive_path, 0o400)


def _run_entity(entity_uid: str) -> None:
    scripts, inputs, outputs = stage_paths()
    _require_source(scripts, inputs)
    _require_smoke_release(scripts, entity_uid)
    projection = _load_projection(inputs)
    states = _entity_states(projection, entity_uid)
    if states[0]["condition_state"] != "observed":
        raise PermissionError("bounded smoke requires one observed-condition entity")
    token = sha256(entity_uid.encode("utf-8"))
    output_archive = outputs / f"{token}.tar"
    if output_archive.exists():
        raise FileExistsError("entity output already exists")
    work = outputs / f".{token}.work.{os.getpid()}"
    os.mkdir(work, 0o700)
    metadata_lines: list[bytes] = []
    try:
        tasks = (
            (entity_uid, ordinal, state, str(work))
            for ordinal, state in enumerate(states)
        )
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=1, maxtasksperchild=1) as pool:
            for expected, result in enumerate(pool.imap(_worker, tasks, chunksize=1)):
                ordinal, compressed_pdb, metadata_raw = result
                if ordinal != expected:
                    raise ValueError("spawned worker result order drifted")
                _parse_object(metadata_raw, "state metadata")
                _exclusive(work / f"state-{ordinal:05d}.pdb", compressed_pdb, 0o400)
                metadata_lines.append(metadata_raw)
        metadata_raw = b"".join(metadata_lines)
        temporary_archive = work / "entity-output.tar"
        _stream_entity_archive(work, states, metadata_raw, temporary_archive)
        archive_sha256 = _hash_file(temporary_archive)
        os.link(temporary_archive, output_archive, follow_symlinks=False)
        os.chmod(output_archive, 0o444)
        if _hash_file(output_archive) != archive_sha256:
            raise ValueError("published entity archive hash changed")
    finally:
        if work.exists():
            shutil.rmtree(work)


def _brute_minimum(coordinates: list[tuple[float, float, float]]) -> float:
    values = [
        math.dist(left, right)
        for index, left in enumerate(coordinates)
        for right in coordinates[index + 1 :]
    ]
    return min(values, default=math.inf)


def self_test() -> int:
    random.seed(17)
    cases = [
        [],
        [(0.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (0.499999, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
    ]
    cases.extend(
        [tuple(random.uniform(-3.0, 3.0) for _ in range(3)) for _ in range(size)]
        for size in range(2, 40)
    )
    for coordinates in cases:
        wanted = _brute_minimum(list(coordinates))
        upper = wanted if math.isfinite(wanted) else 1.0
        got = _minimum_distance_cell_list(list(coordinates), upper)
        if not (
            (math.isinf(got) and math.isinf(wanted))
            or math.isclose(got, wanted, rel_tol=1e-15, abs_tol=1e-15)
        ):
            raise AssertionError("cell-list minimum differs from brute force")
    first = _archive_bytes([("z", b"2"), ("a", b"1")])
    second = _archive_bytes([("a", b"1"), ("z", b"2")])
    if first != second:
        raise AssertionError("archive serialization is not deterministic")
    if not first.endswith(bytes(1024)) or first.endswith(bytes(1536)):
        raise AssertionError("archive terminator drifted")
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
        if [member.name for member in archive] != ["a", "z"]:
            raise AssertionError("archive order drifted")
    try:
        _safe_relative("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal accepted")
    return len(cases) + 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-uid")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(
            f"STATUS PASS_FF15IPQ_ALL_SUPPORT_GENERATOR_SELF_TEST checks={self_test()}"
        )
        return 0
    if args.entity_uid is None:
        parser.error("--entity-uid is required")
    _run_entity(args.entity_uid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
