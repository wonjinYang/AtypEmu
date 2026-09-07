#!/usr/bin/env python3
"""Freeze, stage, and run the diagnostic-only ff15ipq hydrogen-angle replay."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_v1"
PARENT_CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
)
ENTITY_UID = "bmrb:10109:entity:1"
BMRB_ID = "bmr10109"
MISSING = [272, 795]
SUPPORTS = [index for index in range(1, 1001) if index not in MISSING]
PH = 6.0
BRANCH = "observed-0"
LOW_ANGLE = 55.0
OPENMM_VERSION = "8.6.0.dev-c6173db"
FORCEFIELD = "amber14/protein.ff15ipq.xml"
RUNTIME_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
PLAN_SHA256 = "61c3536239321dd14b0dd8ce092a838643d59ae3ea3ce3fd72926589768561ab"
PROJECTION_SHA256 = "e9b4861216d99e8136df872958a50f568337fc8a6b4991b25df2b906dcee5046"
DECISION_SHA256 = "ab47932dcbce7a2b61a30ed2f17907f3dbdc8dafbf71428463de43393d77db75"
INVENTORY_SHA256 = "eedcd06f38e4234a29eb6b2bc981a87928406b507f9ccb102bd282f4957b881f"
DECISION_REVIEW_SHA256 = (
    "73fade7cf8fe87e551804faf2a09b8ed9a90f61ffa32bf33cba40741900b922e"
)
RECOVERY_PDB_ROOT = Path(
    ".auto/atypemu_nested_support_count_v1_ff15ipq_all_support_recovery_output_v1/"
    "sealed_pdb/bmr10109"
)
PARENT_RUNTIME = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_all_support_stage_v1/"
    "runtime.sif"
)
SCRIPT = Path("gpuopt/candidates/run_ff15ipq_hydrogen_angle_diagnostic_v1.py")
PLAN = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_plan_v1.json"
)
PROJECTION = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1.json.gz"
)
DECISION = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_all_support_bounded_smoke_"
    "terminal_decision_v1.json"
)
INVENTORY = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_all_support_bounded_smoke_"
    "terminal_inventory_v1.json"
)
DECISION_REVIEW = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_all_support_bounded_smoke_"
    "terminal_decision_review_receipt_v1.json"
)
SOURCE_COMMITMENT = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_source_commitment_v1.json"
)
STAGE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_stage_v1"
)
SOURCE_REVIEW = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_source_review_v1.json"
)
RELEASE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_execution_release_v1.json"
)
CONSUMED = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_execution_release_v1.consumed.json"
)
FAILURE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_execution_failure_v1.json"
)
OUTPUT = Path(
    ".auto/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_output_v1"
)
SOURCE_FILES = (SCRIPT, PLAN, PROJECTION, DECISION, INVENTORY, DECISION_REVIEW)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CLOSED = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
HOST_ENV = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "SINGULARITY_CACHEDIR": "/tmp/singularity-cache",
    "SINGULARITY_TMPDIR": "/tmp",
    "TMPDIR": "/tmp",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    def duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise ValueError(f"noncanonical object in {label}")
    return value


def read_file(path: Path, maximum: int = 100_000_000) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"invalid bounded file: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1_048_576, maximum + 1)):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ValueError(f"file exceeds ceiling: {path}")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def exclusive(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def publish_directory(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2 is required for no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        env=HOST_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        env=HOST_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise PermissionError("tracked Git worktree is dirty")
    return result


def committed_bytes(root: Path, commit: str, relative: Path) -> bytes:
    raw = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=root,
        env=HOST_ENV,
        check=True,
        capture_output=True,
    ).stdout
    if raw != read_file(root / relative, 20_000_000):
        raise PermissionError(f"worktree differs from committed bytes: {relative}")
    return raw


def validate_fixed_inputs(raw_by_path: dict[Path, bytes]) -> None:
    expected = {
        PLAN: PLAN_SHA256,
        PROJECTION: PROJECTION_SHA256,
        DECISION: DECISION_SHA256,
        INVENTORY: INVENTORY_SHA256,
        DECISION_REVIEW: DECISION_REVIEW_SHA256,
    }
    if any(sha256(raw_by_path[path]) != digest for path, digest in expected.items()):
        raise PermissionError("fixed diagnostic input hash drifted")
    plan = parse_object(raw_by_path[PLAN], "diagnostic plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("scope") == "TARGET_UNREAD_DIAGNOSTIC_ONLY"
        and plan.get("entity_uid") == ENTITY_UID
        and plan.get("input_contract", {}).get("present_support_count") == 998
        and plan.get("input_contract", {}).get("missing_support_indices") == MISSING
        and plan.get("condition", {}).get("proposal_pH") == "6.0"
        and plan.get("decision_contract", {}).get("full_135_entity_route") == "CLOSED"
        and plan.get("decision_contract", {}).get(
            "promotion_or_feasibility_claim_allowed"
        )
        is False
    ):
        raise PermissionError("diagnostic plan identity drifted")
    decision = parse_object(raw_by_path[DECISION], "terminal decision")
    review = parse_object(raw_by_path[DECISION_REVIEW], "terminal decision review")
    if not (
        decision.get("decision") == "NO_GO_BOUNDED_SMOKE"
        and decision.get("full_135_entity_route") == "CLOSED"
        and decision.get("ordinary_rerun_or_recovery_reuse_allowed") is None
        and decision.get("interpretation", {}).get(
            "ordinary_rerun_or_recovery_reuse_allowed"
        )
        is False
        and review.get("artifact_verdict") == "ARTIFACT_GO"
        and review.get("decision_sha256") == DECISION_SHA256
    ):
        raise PermissionError("parent terminal closure drifted")


def freeze_source(root: Path) -> None:
    commit = git_head(root)
    raw_by_path = {path: committed_bytes(root, commit, path) for path in SOURCE_FILES}
    validate_fixed_inputs(raw_by_path)
    commitment = {
        "candidate_id": CANDIDATE_ID,
        "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_source_commitment_v1",
        "files": {str(path): sha256(raw_by_path[path]) for path in SOURCE_FILES},
        "git_commit": commit,
        "runtime_sif_sha256": RUNTIME_SHA256,
        "state": "FROZEN_DIAGNOSTIC_ONLY_UNRUN",
    }
    exclusive(root / SOURCE_COMMITMENT, canonical(commitment) + b"\n")


def require_source(root: Path, expected_hash: str) -> tuple[dict[str, Any], bytes]:
    if SHA256_RE.fullmatch(expected_hash) is None:
        raise PermissionError("source commitment hash is invalid")
    raw = read_file(root / SOURCE_COMMITMENT, 1_000_000)
    if sha256(raw) != expected_hash:
        raise PermissionError("source commitment hash drifted")
    source = parse_object(raw, "source commitment")
    if set(source) != {
        "candidate_id",
        "contract",
        "files",
        "git_commit",
        "runtime_sif_sha256",
        "state",
    } or not (
        source["candidate_id"] == CANDIDATE_ID
        and source["contract"]
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_source_commitment_v1"
        and source["git_commit"] == git_head(root)
        and source["runtime_sif_sha256"] == RUNTIME_SHA256
        and source["state"] == "FROZEN_DIAGNOSTIC_ONLY_UNRUN"
        and isinstance(source["files"], dict)
        and set(source["files"]) == {str(path) for path in SOURCE_FILES}
    ):
        raise PermissionError("source commitment identity drifted")
    raw_by_path = {
        path: committed_bytes(root, source["git_commit"], path) for path in SOURCE_FILES
    }
    if any(
        sha256(raw_by_path[path]) != source["files"][str(path)] for path in SOURCE_FILES
    ):
        raise PermissionError("source file hash drifted")
    validate_fixed_inputs(raw_by_path)
    return source, raw


def stage_source(root: Path, expected_hash: str) -> None:
    source, source_raw = require_source(root, expected_hash)
    destination = root / STAGE
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    runtime_raw = read_file(root / PARENT_RUNTIME)
    if sha256(runtime_raw) != RUNTIME_SHA256:
        raise PermissionError("OpenMM runtime hash drifted")
    with tempfile.TemporaryDirectory(
        prefix=".ff15ipq-angle-stage-", dir=root / ".auto"
    ) as text:
        temporary = Path(text)
        scripts = temporary / "scripts"
        inputs = temporary / "inputs"
        scripts.mkdir(mode=0o700)
        inputs.mkdir(mode=0o700)
        for relative in SOURCE_FILES:
            raw = committed_bytes(root, source["git_commit"], relative)
            directory = inputs if relative == PROJECTION else scripts
            exclusive(directory / relative.name, raw)
        exclusive(scripts / SOURCE_COMMITMENT.name, source_raw)
        exclusive(temporary / "runtime.sif", runtime_raw)
        os.chmod(scripts, 0o555)
        os.chmod(inputs, 0o555)
        os.chmod(temporary, 0o555)
        publish_directory(temporary, destination)


def seed_for(index: int) -> int:
    material = "\0".join((PARENT_CANDIDATE_ID, ENTITY_UID, str(index), BRANCH)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def angle_degrees(
    hydrogen: tuple[float, float, float],
    parent: tuple[float, float, float],
    other: tuple[float, float, float],
) -> float:
    left = tuple(hydrogen[i] - parent[i] for i in range(3))
    right = tuple(other[i] - parent[i] for i in range(3))
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("degenerate hydrogen angle")
    cosine = sum(left[i] * right[i] for i in range(3)) / denominator
    result = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    if not math.isfinite(result):
        raise ValueError("nonfinite hydrogen angle")
    return result


def atom_identity(atom: Any) -> dict[str, Any]:
    residue = atom.residue
    chain = residue.chain
    return {
        "atom_index": atom.index,
        "atom_name": atom.name,
        "chain_id": chain.id,
        "chain_index": chain.index,
        "residue_id": residue.id,
        "residue_index": residue.index,
        "residue_name": residue.name,
    }


def diagnose_support(task: tuple[int, str, str]) -> dict[str, Any]:
    index, pdb_root_text, expected_hash = task
    try:
        import numpy  # type: ignore[import-not-found]
        import openmm  # type: ignore[import-not-found]
        from openmm import Platform, app, unit  # type: ignore[import-not-found]

        if getattr(openmm, "__version__", None) != OPENMM_VERSION:
            raise RuntimeError("OpenMM version drifted")
        seed = seed_for(index)
        random.seed(seed)
        numpy.random.seed(seed % (2**32))
        path = Path(pdb_root_text) / f"{BMRB_ID}_BioEmu_{index}.pdb"
        raw = read_file(path, 5_000_000)
        if sha256(raw) != expected_hash:
            raise ValueError("projected parent PDB hash drifted")
        pdb = app.PDBFile(io.StringIO(raw.decode("ascii")))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        atoms = list(modeller.topology.atoms())
        if any(atom.element is None for atom in atoms):
            raise ValueError("input atom lacks element")

        def heavy_snapshot() -> list[tuple[Any, ...]]:
            positions = list(modeller.positions)
            snapshot: list[tuple[Any, ...]] = []
            for atom in modeller.topology.atoms():
                if atom.element.atomic_number == 1:
                    continue
                position = positions[atom.index].value_in_unit(unit.angstrom)
                snapshot.append(
                    (
                        atom.residue.chain.index,
                        atom.residue.chain.id,
                        atom.residue.index,
                        atom.residue.id,
                        getattr(atom.residue, "insertionCode", ""),
                        atom.residue.name,
                        atom.index,
                        atom.name,
                        atom.element.symbol,
                        *(float(value) for value in position),
                    )
                )
            return snapshot

        parent_heavy = heavy_snapshot()
        modeller.delete([atom for atom in atoms if atom.element.atomic_number == 1])
        if any(atom.element.atomic_number == 1 for atom in modeller.topology.atoms()):
            raise ValueError("input hydrogen deletion was incomplete")
        deleted_heavy = heavy_snapshot()
        if [row[:6] + row[7:] for row in deleted_heavy] != [
            row[:6] + row[7:] for row in parent_heavy
        ]:
            raise ValueError("input hydrogen deletion changed parent heavy records")
        forcefield = app.ForceField(FORCEFIELD)
        platform = Platform.getPlatformByName("Reference")
        modeller.addHydrogens(forcefield, pH=PH, variants=None, platform=platform)
        protonated_heavy = heavy_snapshot()
        if [row[:6] + row[7:] for row in protonated_heavy] != [
            row[:6] + row[7:] for row in parent_heavy
        ]:
            raise ValueError("protonation changed parent heavy records")
        atoms = list(modeller.topology.atoms())
        positions = list(modeller.positions)
        if not atoms or len(atoms) != len(positions):
            raise ValueError("output atom-position inventory is invalid")
        coordinates = [
            tuple(float(value) for value in position.value_in_unit(unit.angstrom))
            for position in positions
        ]
        adjacency: dict[Any, list[Any]] = {atom: [] for atom in atoms}
        for left, right in modeller.topology.bonds():
            adjacency[left].append(right)
            adjacency[right].append(left)
        angles: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        for hydrogen in atoms:
            if hydrogen.element is None or hydrogen.element.atomic_number != 1:
                continue
            parents = [
                atom
                for atom in adjacency[hydrogen]
                if atom.element is not None and atom.element.atomic_number != 1
            ]
            if len(parents) != 1 or len(adjacency[hydrogen]) != 1:
                raise ValueError("hydrogen lacks exactly one heavy parent")
            parent = parents[0]
            others = [
                atom
                for atom in adjacency[parent]
                if atom.element is not None and atom.element.atomic_number != 1
            ]
            if not others:
                raise ValueError("hydrogen parent lacks heavy angle reference")
            for other in others:
                angle = angle_degrees(
                    coordinates[hydrogen.index],
                    coordinates[parent.index],
                    coordinates[other.index],
                )
                row = {
                    "angle_degrees": angle,
                    "hydrogen": atom_identity(hydrogen),
                    "other": atom_identity(other),
                    "parent": atom_identity(parent),
                }
                angles.append(row)
                if angle < LOW_ANGLE:
                    violations.append(row)
        if not angles:
            raise ValueError("no hydrogen-heavy-heavy angles produced")
        minimum = min(angles, key=lambda row: row["angle_degrees"])
        return {
            "minimum_angle": minimum,
            "status": "OK",
            "support_index": index,
            "violation_count": len(violations),
            "violations": violations,
        }
    except Exception as error:  # Every support must produce a sealed diagnostic row.
        return {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "status": "ERROR",
            "support_index": index,
        }


def load_projection(path: Path) -> dict[int, str]:
    raw = read_file(path, 20_000_000)
    if sha256(raw) != PROJECTION_SHA256:
        raise PermissionError("diagnostic projection hash drifted")
    projection = parse_object(gzip.decompress(raw), "diagnostic projection")
    entries = projection.get("entries")
    if not isinstance(entries, list):
        raise ValueError("projection entries are invalid")
    matches = [row for row in entries if row.get("entity_uid") == ENTITY_UID]
    if len(matches) != 1:
        raise ValueError("diagnostic entity is absent or duplicate")
    entity = matches[0]
    if not (
        entity.get("bmrb_id") == BMRB_ID
        and entity.get("missing_support_indices") == MISSING
        and entity.get("condition_state") == "observed"
        and entity.get("condition_branches")
        == [{"condition_branch_id": BRANCH, "proposal_pH": "6.0"}]
    ):
        raise ValueError("diagnostic entity condition or missingness drifted")
    result: dict[int, str] = {}
    for row in entity.get("supports", []):
        if not (
            isinstance(row, dict)
            and set(row) == {"parent_raw_pdb_sha256", "support_index"}
            and type(row["support_index"]) is int
            and SHA256_RE.fullmatch(row["parent_raw_pdb_sha256"])
        ):
            raise ValueError("projected support row is invalid")
        if row["support_index"] in result:
            raise ValueError("duplicate projected support")
        result[row["support_index"]] = row["parent_raw_pdb_sha256"]
    if sorted(result) != SUPPORTS:
        raise ValueError("projected support inventory drifted")
    return result


def require_worker_release(stage: Path) -> None:
    expected_source = os.environ.get("ATYPEMU_DIAGNOSTIC_SOURCE_SHA256")
    expected_release = os.environ.get("ATYPEMU_DIAGNOSTIC_RELEASE_SHA256")
    if not expected_source or not expected_release:
        raise PermissionError("worker source/release binding is absent")
    source_raw = read_file(stage / "scripts" / SOURCE_COMMITMENT.name, 1_000_000)
    release_raw = read_file(stage / "release/execution_release.json", 1_000_000)
    consumed_raw = read_file(stage / "release/execution_consumed.json", 1_000_000)
    if sha256(source_raw) != expected_source or sha256(release_raw) != expected_release:
        raise PermissionError("worker source/release hash drifted")
    source = parse_object(source_raw, "worker source commitment")
    release = parse_object(release_raw, "worker execution release")
    consumed = parse_object(consumed_raw, "worker execution consumption")
    script_raw = read_file(Path(__file__).absolute(), 2_000_000)
    release_fields = {
        "candidate_id",
        "closed_capabilities",
        "contract",
        "entity_uid",
        "git_commit",
        "mode",
        "runtime_sif_sha256",
        "source_commitment_sha256",
        "source_review_sha256",
        "state",
    }
    consumed_fields = {
        "candidate_id",
        "contract",
        "execution_release_sha256",
        "source_commitment_sha256",
        "state",
    }
    if not (
        set(release) == release_fields
        and set(consumed) == consumed_fields
        and source.get("files", {}).get(str(SCRIPT)) == sha256(script_raw)
        and release.get("candidate_id") == CANDIDATE_ID
        and release.get("closed_capabilities") == CLOSED
        and release.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_execution_release_v1"
        and release.get("entity_uid") == ENTITY_UID
        and release.get("mode") == "DIAGNOSTIC_ONLY_998_SUPPORTS"
        and release.get("source_commitment_sha256") == expected_source
        and release.get("runtime_sif_sha256") == RUNTIME_SHA256
        and os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256") == RUNTIME_SHA256
        and release.get("state") == "EXTERNALLY_RELEASED_ONCE_DIAGNOSTIC_ONLY"
        and consumed.get("candidate_id") == CANDIDATE_ID
        and consumed.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_consumption_v1"
        and consumed.get("execution_release_sha256") == expected_release
        and consumed.get("source_commitment_sha256") == expected_source
        and consumed.get("state") == "CONSUMED_BEFORE_DIAGNOSTIC_PDB_ACCESS"
    ):
        raise PermissionError("worker release/consumption identity drifted")


def worker() -> None:
    script = Path(__file__).absolute()
    stage = script.parent.parent
    if (
        script.is_symlink()
        or script.resolve(strict=True) != script
        or os.environ.get("ATYPEMU_STAGE_ROOT") != str(stage)
    ):
        raise PermissionError("worker script is outside the released stage")
    require_worker_release(stage)
    projection = load_projection(stage / "inputs" / PROJECTION.name)
    pdb_root = stage / "inputs/data/BioEmu/bmr10109"
    entries = list(pdb_root.iterdir())
    actual = {path.name for path in entries}
    expected_names = {f"{BMRB_ID}_BioEmu_{index}.pdb" for index in SUPPORTS}
    if (
        actual != expected_names
        or len(entries) != 998
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise ValueError("diagnostic PDB directory inventory drifted")
    tasks = [(index, str(pdb_root), projection[index]) for index in SUPPORTS]
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=8, maxtasksperchild=1) as pool:
        rows = list(pool.imap(diagnose_support, tasks, chunksize=1))
    if [row.get("support_index") for row in rows] != SUPPORTS:
        raise ValueError("diagnostic result identity drifted")
    failures = [row for row in rows if row.get("status") == "ERROR"]
    violations = [
        row for row in rows if row.get("status") == "OK" and row["violation_count"] > 0
    ]
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    minimum_row = (
        min(ok_rows, key=lambda row: row["minimum_angle"]["angle_degrees"])
        if ok_rows
        else None
    )
    minimum = (
        {
            "support_index": minimum_row["support_index"],
            **minimum_row["minimum_angle"],
        }
        if minimum_row
        else None
    )
    result_class = (
        "DIAGNOSTIC_EXECUTION_FAILED"
        if failures
        else (
            "HYDROGEN_ANGLE_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
            if violations
            else "NO_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
        )
    )
    output = stage / "output"
    result_raw = b"".join(canonical(row) + b"\n" for row in rows)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "error_support_count": len(failures),
        "full_135_entity_route": "CLOSED",
        "global_minimum_angle": minimum,
        "promotion_or_feasibility_claim_allowed": False,
        "result_class": result_class,
        "state": "SEALED_TARGET_UNREAD_DIAGNOSTIC_ONLY",
        "support_count": len(rows),
        "support_results_sha256": sha256(result_raw),
        "violation_support_count": len(violations),
    }
    exclusive(output / "support_results.jsonl", result_raw)
    exclusive(output / "summary.json", canonical(summary) + b"\n")


def validate_release(
    root: Path, source: dict[str, Any], source_hash: str, release_hash: str
) -> tuple[bytes, bytes]:
    if SHA256_RE.fullmatch(release_hash) is None:
        raise PermissionError("external release hash is invalid")
    review_raw = read_file(root / SOURCE_REVIEW, 1_000_000)
    release_raw = read_file(root / RELEASE, 1_000_000)
    if sha256(release_raw) != release_hash:
        raise PermissionError("external release hash drifted")
    review = parse_object(review_raw, "source review")
    release = parse_object(release_raw, "execution release")
    reviews = review.get("reviews")
    review_fields = {
        "candidate_id",
        "closed_capabilities",
        "contract",
        "reviews",
        "source_commitment_sha256",
        "source_git_commit",
        "state",
    }
    release_fields = {
        "candidate_id",
        "closed_capabilities",
        "contract",
        "entity_uid",
        "git_commit",
        "mode",
        "runtime_sif_sha256",
        "source_commitment_sha256",
        "source_review_sha256",
        "state",
    }
    if not (
        set(review) == review_fields
        and review.get("candidate_id") == CANDIDATE_ID
        and review.get("closed_capabilities") == CLOSED
        and review.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_source_review_v1"
        and review.get("source_commitment_sha256") == source_hash
        and review.get("source_git_commit") == source["git_commit"]
        and review.get("state") == "GO_DIAGNOSTIC_ONLY"
        and isinstance(reviews, list)
        and len(reviews) == 3
        and {row.get("scope") for row in reviews}
        == {"operational", "scientific_leakage", "security_provenance"}
        and len({row.get("review_session") for row in reviews}) == 3
        and all(
            isinstance(row, dict)
            and set(row) == {"review_session", "scope", "verdict"}
            and isinstance(row["review_session"], str)
            and bool(row["review_session"])
            and row["verdict"] == "GO"
            for row in reviews
        )
        and set(release) == release_fields
        and release.get("candidate_id") == CANDIDATE_ID
        and release.get("closed_capabilities") == CLOSED
        and release.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_execution_release_v1"
        and release.get("entity_uid") == ENTITY_UID
        and release.get("git_commit") == source["git_commit"]
        and release.get("mode") == "DIAGNOSTIC_ONLY_998_SUPPORTS"
        and release.get("source_commitment_sha256") == source_hash
        and release.get("source_review_sha256") == sha256(review_raw)
        and release.get("runtime_sif_sha256") == RUNTIME_SHA256
        and release.get("state") == "EXTERNALLY_RELEASED_ONCE_DIAGNOSTIC_ONLY"
    ):
        raise PermissionError("source review or execution release drifted")
    return review_raw, release_raw


def validate_result_bytes(summary_raw: bytes, results_raw: bytes) -> dict[str, Any]:
    summary = parse_object(summary_raw, "diagnostic summary")
    lines = results_raw.splitlines(keepends=True)
    if len(lines) != 998 or not results_raw.endswith(b"\n"):
        raise ValueError("diagnostic result row count or framing drifted")
    rows = [
        parse_object(line, f"support result {index}")
        for index, line in enumerate(lines)
    ]
    if [row.get("support_index") for row in rows] != SUPPORTS:
        raise ValueError("diagnostic result support identity drifted")
    errors = [row for row in rows if row.get("status") == "ERROR"]
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    if len(errors) + len(ok_rows) != 998:
        raise ValueError("diagnostic result status drifted")
    violations = [row for row in ok_rows if row.get("violation_count", 0) > 0]
    expected_class = (
        "DIAGNOSTIC_EXECUTION_FAILED"
        if errors
        else (
            "HYDROGEN_ANGLE_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
            if violations
            else "NO_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
        )
    )
    if not (
        set(summary)
        == {
            "candidate_id",
            "error_support_count",
            "full_135_entity_route",
            "global_minimum_angle",
            "promotion_or_feasibility_claim_allowed",
            "result_class",
            "state",
            "support_count",
            "support_results_sha256",
            "violation_support_count",
        }
        and summary.get("candidate_id") == CANDIDATE_ID
        and summary.get("support_count") == 998
        and summary.get("support_results_sha256") == sha256(results_raw)
        and summary.get("error_support_count") == len(errors)
        and summary.get("violation_support_count") == len(violations)
        and summary.get("result_class") == expected_class
        and summary.get("state") == "SEALED_TARGET_UNREAD_DIAGNOSTIC_ONLY"
        and summary.get("full_135_entity_route") == "CLOSED"
        and summary.get("promotion_or_feasibility_claim_allowed") is False
    ):
        raise ValueError("diagnostic summary drifted")
    return summary


def execute_diagnostic(root: Path, source_hash: str, release_hash: str) -> None:
    source, _ = require_source(root, source_hash)
    review_raw, release_raw = validate_release(root, source, source_hash, release_hash)
    stage = root / STAGE
    if stage.is_symlink() or not stage.is_dir():
        raise PermissionError("diagnostic stage is absent or indirect")
    if any(
        (root / path).exists() or (root / path).is_symlink()
        for path in (CONSUMED, FAILURE, OUTPUT)
    ):
        raise FileExistsError("diagnostic terminal path already exists")
    runtime_raw = read_file(stage / "runtime.sif")
    if sha256(runtime_raw) != RUNTIME_SHA256:
        raise PermissionError("staged runtime drifted")
    consumed = {
        "candidate_id": CANDIDATE_ID,
        "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_consumption_v1",
        "execution_release_sha256": release_hash,
        "source_commitment_sha256": source_hash,
        "state": "CONSUMED_BEFORE_DIAGNOSTIC_PDB_ACCESS",
    }
    consumed_raw = canonical(consumed) + b"\n"
    exclusive(root / CONSUMED, consumed_raw)
    output = root / OUTPUT
    with tempfile.TemporaryDirectory(
        prefix=".ff15ipq-angle-output-", dir=root / ".auto"
    ) as text:
        temporary = Path(text)
        (temporary / "release").mkdir(mode=0o700)
        (temporary / "results").mkdir(mode=0o700)
        (temporary / "scripts").mkdir(mode=0o700)
        (temporary / "inputs").mkdir(mode=0o700)
        exclusive(
            temporary / "scripts" / SCRIPT.name,
            committed_bytes(root, source["git_commit"], SCRIPT),
        )
        exclusive(
            temporary / "scripts" / SOURCE_COMMITMENT.name,
            read_file(root / SOURCE_COMMITMENT, 1_000_000),
        )
        exclusive(
            temporary / "inputs" / PROJECTION.name,
            committed_bytes(root, source["git_commit"], PROJECTION),
        )
        exclusive(temporary / "runtime.sif", runtime_raw)
        exclusive(temporary / "release/execution_release.json", release_raw)
        exclusive(temporary / "release/execution_consumed.json", consumed_raw)
        exclusive(temporary / "release/source_review.json", review_raw)
        publish_directory(temporary, output)
    command = [
        "/usr/bin/singularity",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{output / 'scripts' / SCRIPT.name}:/work/scripts/{SCRIPT.name}:ro",
        "--bind",
        f"{output / 'scripts' / SOURCE_COMMITMENT.name}:/work/scripts/{SOURCE_COMMITMENT.name}:ro",
        "--bind",
        f"{output / 'inputs' / PROJECTION.name}:/work/inputs/{PROJECTION.name}:ro",
        "--bind",
        f"{root / RECOVERY_PDB_ROOT}:/work/inputs/data/BioEmu/bmr10109:ro",
        "--bind",
        f"{output / 'release'}:/work/release:ro",
        "--bind",
        f"{output / 'results'}:/work/output:rw",
        "--env",
        (
            f"ATYPEMU_STAGE_ROOT=/work,ATYPEMU_DIAGNOSTIC_SOURCE_SHA256={source_hash},"
            f"ATYPEMU_DIAGNOSTIC_RELEASE_SHA256={release_hash},"
            f"ATYPEMU_RUNTIME_SIF_SHA256={RUNTIME_SHA256}"
        ),
        str(output / "runtime.sif"),
        "python",
        f"/work/scripts/{SCRIPT.name}",
        "--worker",
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=HOST_ENV,
                capture_output=True,
                timeout=21_600,
            )
        except subprocess.TimeoutExpired as error:
            exclusive(output / "stdout.bin", error.stdout or b"")
            exclusive(output / "stderr.bin", error.stderr or b"")
            raise
        exclusive(output / "stdout.bin", completed.stdout)
        exclusive(output / "stderr.bin", completed.stderr)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        summary_raw = read_file(output / "results/summary.json", 2_000_000)
        results_raw = read_file(output / "results/support_results.jsonl", 100_000_000)
        summary = validate_result_bytes(summary_raw, results_raw)
        terminal = {
            "candidate_id": CANDIDATE_ID,
            "execution_release_sha256": release_hash,
            "result_class": summary["result_class"],
            "source_commitment_sha256": source_hash,
            "state": "PASS_DIAGNOSTIC_EXECUTION_ONLY",
            "stderr_sha256": sha256(completed.stderr),
            "stdout_sha256": sha256(completed.stdout),
            "summary_sha256": sha256(summary_raw),
            "support_results_sha256": sha256(results_raw),
        }
        exclusive(output / "terminal_receipt.json", canonical(terminal) + b"\n")
    except BaseException as error:
        failure = {
            "candidate_id": CANDIDATE_ID,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "execution_release_sha256": release_hash,
            "source_commitment_sha256": source_hash,
            "state": "FAILED_AFTER_DIAGNOSTIC_CONSUMPTION",
        }
        exclusive(root / FAILURE, canonical(failure) + b"\n")
        raise


def self_test() -> int:
    for invalid in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b"[]\n"):
        try:
            parse_object(invalid, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON accepted")
    angle = angle_degrees(
        (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, math.sqrt(3) / 2, 0.0)
    )
    if (
        abs(angle - 60.0) > 1e-12
        or len(SUPPORTS) != 998
        or SUPPORTS != sorted(set(SUPPORTS))
    ):
        raise AssertionError("synthetic angle or support partition failed")
    if seed_for(1) == seed_for(2):
        raise AssertionError("support seed collision")
    return 6


def repository_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            env=HOST_ENV,
        ).stdout.strip()
    ).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-source", action="store_true")
    parser.add_argument("--stage-source", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-commitment-sha256")
    parser.add_argument("--execution-release-sha256")
    args = parser.parse_args()
    modes = [
        args.freeze_source,
        args.stage_source,
        args.run,
        args.worker,
        args.self_test,
    ]
    if sum(modes) != 1:
        parser.error("select exactly one mode")
    if args.worker:
        worker()
        return 0
    if args.self_test:
        print(f"STATUS PASS_FF15IPQ_ANGLE_DIAGNOSTIC_SELF_TEST checks={self_test()}")
        return 0
    root = repository_root()
    if args.freeze_source:
        freeze_source(root)
    elif args.stage_source:
        if not args.source_commitment_sha256:
            parser.error("--source-commitment-sha256 is required")
        stage_source(root, args.source_commitment_sha256)
    else:
        if not args.source_commitment_sha256 or not args.execution_release_sha256:
            parser.error("both external hashes are required")
        execute_diagnostic(
            root, args.source_commitment_sha256, args.execution_release_sha256
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
