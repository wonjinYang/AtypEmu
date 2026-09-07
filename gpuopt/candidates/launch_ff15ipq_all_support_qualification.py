#!/usr/bin/env python3
# ruff: noqa: TRY004
"""Freeze, stage, and run the target-unread ff15ipq all-support qualification."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
SOURCE_CONTRACT = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1"
)
RELEASE_CONTRACT = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_execution_release_v1"
)
RUNTIME_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
SCRIPT_RELATIVE = Path("gpuopt/candidates/launch_ff15ipq_all_support_qualification.py")
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_qualification_plan_v1.json"
)
PROJECTION_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_input_projection_v1.json.gz"
)
PLAN_VALIDATION_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_plan_validation_receipt_v1.json"
)
SOURCE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_source_commitment_v1.json"
)
RELEASE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_release_v1.json"
)
CONSUMED_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_release_v1.consumed.json"
)
SOURCE_REVIEW_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_source_commitment_cold_review_receipt_v1.json"
)
STAGE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_all_support_stage_v1"
)
OUTPUT_RELATIVE = Path(
    ".auto/atypemu_nested_support_count_v1_ff15ipq_all_support_output_v1"
)
FAILURE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_failure_receipt_v1.json"
)
FILES = {
    Path("gpuopt/candidates/materialize_ff15ipq_all_support_entity.py"),
    Path("gpuopt/candidates/check_ff15ipq_all_support_entity.py"),
    Path("gpuopt/candidates/ff15ipq_all_support_common.py"),
    Path("gpuopt/candidates/check_ff15ipq_all_support_qualification_plan.py"),
    SCRIPT_RELATIVE,
    PLAN_RELATIVE,
    PLAN_VALIDATION_RELATIVE,
    PROJECTION_RELATIVE,
    Path(
        "gpuopt/preunblind/atypemu_nested_support_count_v1_"
        "ff15ipq_all_support_implementation_cold_review_receipt_v1.json"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT = "/usr/bin/git"
SINGULARITY = "/usr/bin/singularity"
HOST_ENV = {
    "GIT_NO_REPLACE_OBJECTS": "1",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
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


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def canonical_root() -> Path:
    script = Path(__file__).absolute()
    root = script.parents[2]
    if (
        script != root / SCRIPT_RELATIVE
        or script.is_symlink()
        or script.resolve(strict=True) != script
    ):
        raise ValueError("launcher is outside its canonical repository path")
    return root


def checked_relative(relative: Path) -> Path:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe repository-relative path: {relative}")
    return relative


def read_regular(path: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"invalid bounded regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1_048_576):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"file exceeds bound: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def repo_read(root: Path, relative: Path, maximum_bytes: int = 20_000_000) -> bytes:
    relative = checked_relative(relative)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | nofollow,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | nofollow, dir_fd=directory
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise ValueError(f"invalid bounded repository file: {relative}")
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(descriptor, 1_048_576):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(f"repository file exceeds bound: {relative}")
                chunks.append(chunk)
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
                raise ValueError(f"repository file changed while reading: {relative}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def hash_file(path: Path, maximum_bytes: int = 1_000_000_000) -> str:
    return digest(read_regular(path, maximum_bytes))


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


def publish_directory_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-clobber publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(f"publication destination exists: {destination}")
        raise OSError(number, os.strerror(number), str(destination))


def clean_head(root: Path) -> str:
    subprocess.run(
        [GIT, "diff", "--quiet", "HEAD", "--"],
        cwd=root,
        check=True,
        env=HOST_ENV,
    )
    subprocess.run(
        [GIT, "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=root,
        check=True,
        env=HOST_ENV,
    )
    head = subprocess.run(
        [GIT, "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
        env=HOST_ENV,
    ).stdout.strip()
    if GIT_RE.fullmatch(head) is None:
        raise ValueError("current Git HEAD is invalid")
    for relative in FILES:
        current = repo_read(root, relative)
        committed = subprocess.run(
            [GIT, "show", f"{head}:{relative.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
            env=HOST_ENV,
        ).stdout
        if current != committed:
            raise ValueError(f"source differs from Git object: {relative}")
    return head


def validate_plan(root: Path) -> dict[str, Any]:
    plan = parse_object(repo_read(root, PLAN_RELATIVE), "qualification plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("state") == "HOLD_EXTERNAL_SOURCE_COMMITMENT_PENDING_UNRUN"
        and plan.get("execution_contract", {}).get("source_commitment")
        == "EXTERNAL_O_EXCL_PENDING"
        and plan.get("authorization")
        == {
            "consumption_allowed": False,
            "existing_authorizations_reusable": False,
            "request_allowed": False,
        }
        and plan.get("closed_capabilities") == CLOSED_CAPABILITIES
    ):
        raise PermissionError("qualification plan has not released source freezing")
    receipt = parse_object(
        repo_read(root, PLAN_VALIDATION_RELATIVE), "plan validation receipt"
    )
    checked_files = {
        PLAN_RELATIVE,
        PROJECTION_RELATIVE,
        Path("gpuopt/candidates/check_ff15ipq_all_support_qualification_plan.py"),
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_"
            "ff15ipq_all_support_implementation_cold_review_receipt_v1.json"
        ),
    }
    if (
        set(receipt)
        != {
            "candidate_id",
            "checked_file_sha256",
            "closed_capabilities",
            "contract",
            "plan_validator_output",
            "state",
        }
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["contract"]
        != "atypemu_nested_support_count_v1_"
        "ff15ipq_all_support_plan_validation_receipt_v1"
        or receipt["closed_capabilities"] != CLOSED_CAPABILITIES
        or receipt["plan_validator_output"]
        != "STATUS PASS_HOLD_ALL_SUPPORT_PLAN_ONLY checks=135022"
        or receipt["state"] != "PASS_COMMITTED_INPUT_PLAN_VALIDATION"
        or not isinstance(receipt["checked_file_sha256"], dict)
        or set(receipt["checked_file_sha256"])
        != {path.as_posix() for path in checked_files}
        or any(
            receipt["checked_file_sha256"][path.as_posix()]
            != digest(repo_read(root, path))
            for path in checked_files
        )
    ):
        raise ValueError("deep plan-validation receipt drifted")
    return plan


def freeze_source(root: Path) -> None:
    head = clean_head(root)
    validate_plan(root)
    files = {
        relative.as_posix(): digest(repo_read(root, relative)) for relative in FILES
    }
    commitment = {
        "candidate_id": CANDIDATE_ID,
        "contract": SOURCE_CONTRACT,
        "files": files,
        "git_commit": head,
        "status": "FROZEN_COMMITTED_BOUNDED_SMOKE_ONLY",
    }
    output = root / SOURCE_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        output.parent.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
    ):
        raise ValueError("source-commitment directory is indirect")
    exclusive(output, canonical(commitment) + b"\n")
    print(f"STATUS FROZEN source_commitment_sha256={hash_file(output)}")


def require_source(root: Path, expected_hash: str) -> tuple[dict[str, Any], bytes]:
    if SHA256_RE.fullmatch(expected_hash) is None:
        raise ValueError("source commitment hash argument is invalid")
    raw = repo_read(root, SOURCE_RELATIVE, 1_000_000)
    if digest(raw) != expected_hash:
        raise ValueError("source commitment external hash differs")
    value = parse_object(raw, "source commitment")
    if set(value) != {"candidate_id", "contract", "files", "git_commit", "status"}:
        raise ValueError("source commitment schema drifted")
    if not (
        value["candidate_id"] == CANDIDATE_ID
        and value["contract"] == SOURCE_CONTRACT
        and value["status"] == "FROZEN_COMMITTED_BOUNDED_SMOKE_ONLY"
        and isinstance(value["git_commit"], str)
        and GIT_RE.fullmatch(value["git_commit"])
        and isinstance(value["files"], dict)
        and set(value["files"]) == {path.as_posix() for path in FILES}
    ):
        raise ValueError("source commitment identity drifted")
    for relative in FILES:
        current = repo_read(root, relative)
        expected = value["files"][relative.as_posix()]
        if not isinstance(expected, str) or digest(current) != expected:
            raise ValueError(f"source commitment file mismatch: {relative}")
        committed = subprocess.run(
            [GIT, "show", f"{value['git_commit']}:{relative.as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
            env=HOST_ENV,
        ).stdout
        if current != committed:
            raise ValueError(f"source differs from frozen Git object: {relative}")
    return value, raw


def stage(root: Path, expected_source_hash: str, runtime_sif: Path) -> None:
    commitment, commitment_raw = require_source(root, expected_source_hash)
    validate_plan(root)
    if clean_head(root) != commitment["git_commit"]:
        raise ValueError("current HEAD differs from frozen source commit")
    if not runtime_sif.is_absolute() or runtime_sif.is_symlink():
        raise ValueError("runtime SIF must be an absolute direct file")
    runtime_raw = read_regular(runtime_sif, 100_000_000)
    if digest(runtime_raw) != RUNTIME_SHA256:
        raise ValueError("runtime SIF hash drifted")
    final = root / STAGE_RELATIVE
    if final.exists():
        raise FileExistsError("stage already exists; no clobber")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.parent.is_symlink() or final.parent.resolve(strict=True) != final.parent:
        raise ValueError("stage parent directory is indirect")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp.", dir=final.parent))
    scripts, inputs = temporary / "scripts", temporary / "inputs"
    (inputs / "data" / "BioEmu").mkdir(parents=True, mode=0o700)
    scripts.mkdir(parents=True, mode=0o700)
    for relative in FILES - {PROJECTION_RELATIVE}:
        exclusive(scripts / relative.name, repo_read(root, relative))
    exclusive(scripts / SOURCE_RELATIVE.name, commitment_raw)
    exclusive(inputs / PROJECTION_RELATIVE.name, repo_read(root, PROJECTION_RELATIVE))
    exclusive(temporary / "runtime.sif", runtime_raw)
    for relative in FILES - {PROJECTION_RELATIVE}:
        if (
            digest(read_regular(scripts / relative.name, 20_000_000))
            != commitment["files"][relative.as_posix()]
        ):
            raise ValueError(f"staged source hash drifted: {relative}")
    if not (
        digest(read_regular(inputs / PROJECTION_RELATIVE.name, 20_000_000))
        == commitment["files"][PROJECTION_RELATIVE.as_posix()]
        and digest(read_regular(temporary / "runtime.sif", 100_000_000))
        == RUNTIME_SHA256
    ):
        raise ValueError("staged projection or runtime hash drifted")
    for directory in (inputs / "data" / "BioEmu", inputs / "data", inputs, scripts):
        os.chmod(directory, 0o555)
    os.chmod(temporary, 0o555)
    publish_directory_noreplace(temporary, final)
    print(f"STATUS STAGED stage={final}")


def require_absolute_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute direct directory")
    return path.resolve(strict=True)


def require_release(
    root: Path,
    commitment: dict[str, Any],
    source_hash: str,
    bioemu_root: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    raw = repo_read(root, RELEASE_RELATIVE, 1_000_000)
    release = parse_object(raw, "execution release")
    fields = {
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
    }
    if set(release) != fields:
        raise ValueError("execution release schema drifted")
    entities = release["entities"]
    if not isinstance(entities, list) or any(
        not isinstance(uid, str) for uid in entities
    ):
        raise ValueError("execution release entity inventory is invalid")
    projection = parse_object(
        __import__("gzip").decompress(repo_read(root, PROJECTION_RELATIVE)),
        "runtime projection",
    )
    roster = sorted(row["entity_uid"] for row in projection["entries"])
    mode = release["mode"]
    review_raw = repo_read(root, SOURCE_REVIEW_RELATIVE, 1_000_000)
    review = parse_object(review_raw, "source-commitment cold review")
    if set(review) != {
        "candidate_id",
        "closed_capabilities",
        "contract",
        "git_commit",
        "reviews",
        "source_commitment_sha256",
        "state",
    } or not (
        digest(review_raw) == release["review_receipt_sha256"]
        and review["candidate_id"] == CANDIDATE_ID
        and review["closed_capabilities"] == CLOSED_CAPABILITIES
        and review["contract"]
        == "atypemu_nested_support_count_v1_ff15ipq_all_support_source_cold_review_receipt_v1"
        and review["git_commit"] == commitment["git_commit"]
        and review["source_commitment_sha256"] == source_hash
        and review["state"] == "GO_BOUNDED_SMOKE_SOURCE_REVIEW"
        and isinstance(review["reviews"], list)
        and len(review["reviews"]) == 3
        and {row.get("scope") for row in review["reviews"] if isinstance(row, dict)}
        == {"operational", "scientific_leakage", "security_provenance"}
        and len(
            {
                row.get("review_session")
                for row in review["reviews"]
                if isinstance(row, dict)
            }
        )
        == 3
        and all(
            set(row) == {"review_session", "scope", "verdict"}
            and isinstance(row["review_session"], str)
            and row["review_session"]
            and row["verdict"] == "GO"
            for row in review["reviews"]
        )
    ):
        raise PermissionError("source-commitment cold review binding drifted")
    if not (
        release["candidate_id"] == CANDIDATE_ID
        and release["closed_capabilities"] == CLOSED_CAPABILITIES
        and release["contract"] == RELEASE_CONTRACT
        and release["git_commit"] == commitment["git_commit"]
        and release["source_commitment_sha256"] == source_hash
        and release["runtime_sif_sha256"] == RUNTIME_SHA256
        and release["bioemu_root"] == str(bioemu_root)
        and release["state"] == "EXTERNALLY_RELEASED_ONCE"
        and isinstance(release["review_receipt_sha256"], str)
        and SHA256_RE.fullmatch(release["review_receipt_sha256"])
        and entities == sorted(set(entities))
        and set(entities) <= set(roster)
        and mode == "BOUNDED_SMOKE"
        and len(entities) == 1
        and next(
            row for row in projection["entries"] if row["entity_uid"] == entities[0]
        )["condition_state"]
        == "observed"
    ):
        raise PermissionError("execution release binding or scope drifted")
    return release, raw, review_raw


def singularity_command(
    root: Path,
    runtime_sif: Path,
    bioemu_root: Path,
    commitment: dict[str, Any],
    source_hash: str,
    script: str,
    arguments: list[str],
) -> list[str]:
    stage_dir = root / STAGE_RELATIVE
    output_dir = root / OUTPUT_RELATIVE
    release_dir = output_dir / "release"
    return [
        SINGULARITY,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{stage_dir}:/work:ro",
        "--bind",
        f"{output_dir / 'entity_archives'}:/work/outputs:rw",
        "--bind",
        f"{output_dir / 'checker_receipts'}:/work/checks:rw",
        "--bind",
        f"{bioemu_root}:/work/inputs/data/BioEmu:ro",
        "--bind",
        f"{release_dir}:/work/release:ro",
        "--env",
        f"ATYPEMU_STAGE_ROOT=/work,ATYPEMU_SOURCE_COMMITMENT_SHA256={source_hash},"
        f"ATYPEMU_RUNTIME_SIF_SHA256={RUNTIME_SHA256},"
        f"ATYPEMU_GIT_COMMIT={commitment['git_commit']},"
        f"ATYPEMU_EXECUTION_RELEASE_SHA256={os.environ['ATYPEMU_EXECUTION_RELEASE_SHA256']},"
        f"ATYPEMU_RELEASED_ENTITY_UID={arguments[-1]}",
        str(runtime_sif),
        "python",
        f"/work/scripts/{script}",
        *arguments,
    ]


def run_released(
    root: Path,
    expected_source_hash: str,
    expected_release_hash: str,
    bioemu_root: Path,
) -> None:
    if SHA256_RE.fullmatch(expected_release_hash) is None:
        raise PermissionError("external execution-release hash is absent or invalid")
    commitment, _ = require_source(root, expected_source_hash)
    validate_plan(root)
    stage_dir = root / STAGE_RELATIVE
    if not stage_dir.is_dir() or stage_dir.is_symlink():
        raise ValueError("canonical stage is absent or indirect")
    runtime_sif = stage_dir / "runtime.sif"
    if hash_file(runtime_sif, 100_000_000) != RUNTIME_SHA256:
        raise ValueError("runtime SIF hash drifted")
    bioemu_root = require_absolute_directory(bioemu_root, "BioEmu root")
    release, release_raw, review_raw = require_release(
        root, commitment, expected_source_hash, bioemu_root
    )
    if digest(release_raw) != expected_release_hash:
        raise PermissionError("external execution-release hash binding drifted")
    os.environ["ATYPEMU_EXECUTION_RELEASE_SHA256"] = expected_release_hash
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError("bounded-smoke output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        output.parent.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
    ):
        raise ValueError("output parent directory is indirect")
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    (temporary_output / "entity_archives").mkdir(mode=0o700)
    (temporary_output / "checker_receipts").mkdir(mode=0o700)
    release_dir = temporary_output / "release"
    release_dir.mkdir(mode=0o700)
    exclusive(release_dir / "execution_release.json", release_raw)
    exclusive(release_dir / "source_cold_review.json", review_raw)
    consumed = {
        "candidate_id": CANDIDATE_ID,
        "contract": "atypemu_ff15ipq_all_support_execution_consumption_v1",
        "execution_release_sha256": digest(release_raw),
        "mode": release["mode"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "LOCAL_BOUNDED_SMOKE"),
        "source_commitment_sha256": expected_source_hash,
        "state": "CONSUMED_BEFORE_TARGET_UNREAD_PDB_ACCESS",
    }
    consumed_raw = canonical(consumed) + b"\n"
    consumed_recorded = False
    try:
        exclusive(root / CONSUMED_RELATIVE, consumed_raw)
        consumed_recorded = True
        exclusive(release_dir / "execution_consumed.json", consumed_raw)
        publish_directory_noreplace(temporary_output, output)
    except BaseException as error:
        if consumed_recorded:
            failure = {
                "candidate_id": CANDIDATE_ID,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "execution_release_sha256": digest(release_raw),
                "failure_phase": "CONSUMPTION_OR_OUTPUT_PUBLICATION",
                "source_commitment_sha256": expected_source_hash,
                "state": "FAILED_AFTER_ONE_SHOT_RELEASE_CONSUMPTION",
            }
            exclusive(root / FAILURE_RELATIVE, canonical(failure) + b"\n")
        raise

    def execute_entity(uid: str, checker: bool) -> None:
        script = (
            "check_ff15ipq_all_support_entity.py"
            if checker
            else "materialize_ff15ipq_all_support_entity.py"
        )
        process = subprocess.Popen(
            singularity_command(
                root,
                runtime_sif,
                bioemu_root,
                commitment,
                expected_source_hash,
                script,
                ["--entity-uid", uid],
            ),
            cwd="/tmp",
            env=HOST_ENV,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=21_600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError(f"bounded {script} exceeded six hours") from None
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)

    try:
        for checker in (False, True):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(execute_entity, uid, checker)
                    for uid in release["entities"]
                ]
                for future in futures:
                    future.result()
        receipt = {
            "candidate_id": CANDIDATE_ID,
            "closed_capabilities": CLOSED_CAPABILITIES,
            "entity_count": len(release["entities"]),
            "execution_release_sha256": digest(release_raw),
            "mode": release["mode"],
            "source_commitment_sha256": expected_source_hash,
            "state": "PASS_RELEASED_TARGET_UNREAD_BOUNDED_SMOKE_EXECUTION",
        }
        exclusive(output / "execution_receipt.json", canonical(receipt) + b"\n")
    except BaseException as error:
        failure = {
            "candidate_id": CANDIDATE_ID,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "execution_release_sha256": digest(release_raw),
            "failure_phase": "GENERATION_OR_INDEPENDENT_REPLAY",
            "source_commitment_sha256": expected_source_hash,
            "state": "FAILED_AFTER_ONE_SHOT_RELEASE_CONSUMPTION",
        }
        try:
            exclusive(root / FAILURE_RELATIVE, canonical(failure) + b"\n")
        finally:
            raise


def self_test() -> int:
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"):
        try:
            parse_object(raw, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed JSON accepted")
    for relative in (Path("../x"), Path("/x"), Path(".")):
        try:
            checked_relative(relative)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, existing, published = root / "source", root / "existing", root / "new"
        source.mkdir()
        existing.mkdir()
        try:
            publish_directory_noreplace(source, existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("directory publication clobbered an existing path")
        if not source.is_dir() or not existing.is_dir():
            raise AssertionError("failed no-clobber publication changed directories")
        publish_directory_noreplace(source, published)
        if source.exists() or not published.is_dir():
            raise AssertionError("atomic no-clobber publication failed")
    return 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-source", action="store_true")
    actions.add_argument("--stage", action="store_true")
    actions.add_argument("--run-released", action="store_true")
    actions.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-commitment-sha256")
    parser.add_argument("--execution-release-sha256")
    parser.add_argument("--runtime-sif", type=Path)
    parser.add_argument("--bioemu-root", type=Path)
    args = parser.parse_args()
    root = canonical_root()
    if args.self_test:
        print(
            f"STATUS PASS_FF15IPQ_ALL_SUPPORT_LAUNCHER_SELF_TEST checks={self_test()}"
        )
        return 0
    if args.freeze_source:
        freeze_source(root)
        return 0
    if args.source_commitment_sha256 is None:
        parser.error("--source-commitment-sha256 is required")
    if args.stage:
        if args.runtime_sif is None:
            parser.error("--runtime-sif is required for --stage")
        runtime_sif = args.runtime_sif.resolve(strict=True)
        stage(root, args.source_commitment_sha256, runtime_sif)
        return 0
    if args.bioemu_root is None:
        parser.error("--bioemu-root is required for --run-released")
    if args.execution_release_sha256 is None:
        parser.error("--execution-release-sha256 is required for --run-released")
    run_released(
        root,
        args.source_commitment_sha256,
        args.execution_release_sha256,
        args.bioemu_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
