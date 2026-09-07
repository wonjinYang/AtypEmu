#!/usr/bin/env python3
"""Freeze, stage, and run the diagnostic-only ff15ipq hydrogen-angle replay."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
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
import signal
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_recovery_v1"
)
PARENT_CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
)
ENTITY_UID = "bmrb:10109:entity:1"
BMRB_ID = "bmr10109"
MISSING = [272, 795]
SUPPORTS = [index for index in range(1, 1001) if index not in MISSING]
WORKERS = 12
_WORKER_SEAL = object()
PH = 6.0
BRANCH = "observed-0"
LOW_ANGLE = 55.0
OPENMM_VERSION = "8.6.0.dev-c6173db"
OPENMM_GIT_REVISION = "c6173db6e8edd705eb59172bd21e9ce69c572405"
FORCEFIELD = "amber14/protein.ff15ipq.xml"
RUNTIME_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
PLAN_SHA256 = "280f9263ed4699101a90f9f1df8b893f6bf1bcffa280f6709ed07760f1593a49"
PREDECESSOR_FAILURE_SHA256 = (
    "9415a183981e0f8098163faebcb8318e0f5d3a34d0ce256e175d1af097af68c3"
)
PROJECTION_SHA256 = "e9b4861216d99e8136df872958a50f568337fc8a6b4991b25df2b906dcee5046"
DECISION_SHA256 = "ab47932dcbce7a2b61a30ed2f17907f3dbdc8dafbf71428463de43393d77db75"
INVENTORY_SHA256 = "eedcd06f38e4234a29eb6b2bc981a87928406b507f9ccb102bd282f4957b881f"
DECISION_REVIEW_SHA256 = (
    "73fade7cf8fe87e551804faf2a09b8ed9a90f61ffa32bf33cba40741900b922e"
)
RECOVERY_PDB_ROOT = Path("data/BioEmu/bmr10109")
CONSOLIDATION_ARCHIVE = Path(
    "archive/filesystem-consolidation-20260907/archives/home-worktrees.tar.zst"
)
CONSOLIDATION_ARCHIVE_SHA256 = (
    "0fd6c7163bb725d6ca5f3c85c41260397cf5abe1b853aa914ef9bd0f011c263f"
)
ARCHIVE_PREFIX = "AtypEmu-hold-autoresearch"
RUNTIME_MEMBER = (
    f"{ARCHIVE_PREFIX}/.auto/staging/"
    "atypemu_nested_support_count_v1_ff15ipq_all_support_stage_v1/runtime.sif"
)
PREDECESSOR_MEMBERS = {
    "consumed_sha256": (
        f"{ARCHIVE_PREFIX}/.auto/staging/"
        "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
        "diagnostic_execution_release_v1.consumed.json"
    ),
    "summary_sha256": (
        f"{ARCHIVE_PREFIX}/.auto/"
        "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
        "diagnostic_output_v1/results/summary.json"
    ),
    "support_results_sha256": (
        f"{ARCHIVE_PREFIX}/.auto/"
        "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
        "diagnostic_output_v1/results/support_results.jsonl"
    ),
    "terminal_receipt_sha256": (
        f"{ARCHIVE_PREFIX}/.auto/"
        "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
        "diagnostic_output_v1/terminal_receipt.json"
    ),
}
SCRIPT = Path("gpuopt/candidates/run_ff15ipq_hydrogen_angle_diagnostic_v1.py")
PLAN = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_"
    "recovery_plan_v1.json"
)
PREDECESSOR_FAILURE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_diagnostic_"
    "runtime_version_failure_v1.json"
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
SUPERSEDED_SOURCE_COMMITMENT = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v1.json"
)
SUPERSEDED_SOURCE_COMMITMENT_SHA256 = (
    "668580433512de2b5150c52752fc71cd0d8071638bc6765b0deee13aa57d147c"
)
SUPERSEDED_REVIEW_NOGO = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_nogo_v1.json"
)
SUPERSEDED_REVIEW_NOGO_SHA256 = (
    "e49a10d3b08035c65774c719b3beb7ec0b45e1e6e002b296980633f8be11a589"
)
SUPERSEDED_SOURCE_COMMITMENT_V2 = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v2.json"
)
SUPERSEDED_SOURCE_COMMITMENT_V2_SHA256 = (
    "36c46e3af429739b2f918869b488647e7522c4770247904ca9a3a9dd236cf616"
)
SUPERSEDED_REVIEW_NOGO_V2 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_nogo_v2.json"
)
SUPERSEDED_REVIEW_NOGO_V2_SHA256 = (
    "49f7f5f5023bcf5773542de51e41499d7629404ab0867c14e75d190b81ef2352"
)
SUPERSEDED_SOURCE_COMMITMENT_V3 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v3.json"
)
SUPERSEDED_SOURCE_COMMITMENT_V3_SHA256 = (
    "524cf0ed953716a1bbf9e144e83d20275ccc4205f5f5c0b4d00f093d656901fa"
)
SUPERSEDED_REVIEW_NOGO_V3 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_nogo_v3.json"
)
SUPERSEDED_REVIEW_NOGO_V3_SHA256 = (
    "c34aa0af4d3daf43a7e87c1753b68a0a47b8be38004822268d48e7b89ee1dc62"
)
SUPERSEDED_SOURCE_COMMITMENT_V4 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v4.json"
)
SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256 = (
    "b0fdced7d355e6f41f6aae8db2a2d3d503e339f17d115df5dcbbe13f81024b98"
)
SUPERSEDED_REVIEW_NOGO_V4 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_nogo_v4.json"
)
SUPERSEDED_REVIEW_NOGO_V4_SHA256 = (
    "329765795af3e50f7de5e22cc80b83711d1697937ff91530416af665b3703c49"
)
SUPERSEDED_SOURCE_COMMITMENT_V5 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v5.json"
)
SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256 = (
    "00c3b27cef79ae341ac729c5b0c7008bc762819541af5307bd676c378026c776"
)
SUPERSEDED_REVIEW_GO_V5 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_go_v5.json"
)
SUPERSEDED_REVIEW_GO_V5_SHA256 = (
    "3d1d80bb542e36a3cf8f8e7a9b1f2821c9727cdb67b452eb377cbd964de891ab"
)
SUPERSEDED_RELEASE_FAILURE_V5 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_release_validation_failure_v1.json"
)
SUPERSEDED_RELEASE_FAILURE_V5_SHA256 = (
    "e60a9d6a972de5c4b8d4429dbef27082701a3400f4d7ad99741a33b583c22faf"
)
SUPERSEDED_SOURCE_COMMITMENT_V6 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v6.json"
)
SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256 = (
    "c1c4da9f8fa9c18932ad1f0b14ad84475166202a2eadd84f99af1e58ed1882ef"
)
SUPERSEDED_REVIEW_GO_V6 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_final_cold_review_go_v6.json"
)
SUPERSEDED_REVIEW_GO_V6_SHA256 = (
    "ef8cd2e7b44463db576e613efaea064596b06a7afc05e3e8ae6455fb3980c3b0"
)
SUPERSEDED_DECISION_HOLD_V6 = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_downstream_decision_hold_v6.json"
)
SUPERSEDED_DECISION_HOLD_V6_SHA256 = (
    "00315d4107625b3587150a19c7be4db63e8fa7647c690a06090253bc2911845f"
)
SOURCE_COMMITMENT = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_commitment_v7.json"
)
STAGE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_stage_v7"
)
SOURCE_REVIEW = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_source_review_v7.json"
)
RELEASE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_execution_release_v7.json"
)
CONSUMED = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_execution_release_v7.consumed.json"
)
FAILURE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_execution_failure_v7.json"
)
OUTPUT = Path(
    ".auto/atypemu_nested_support_count_v1_ff15ipq_hydrogen_angle_"
    "diagnostic_recovery_output_v7"
)
SOURCE_FILES = (
    SCRIPT,
    PLAN,
    PREDECESSOR_FAILURE,
    PROJECTION,
    DECISION,
    INVENTORY,
    DECISION_REVIEW,
    SUPERSEDED_REVIEW_NOGO,
    SUPERSEDED_REVIEW_NOGO_V2,
    SUPERSEDED_SOURCE_COMMITMENT_V3,
    SUPERSEDED_REVIEW_NOGO_V3,
    SUPERSEDED_SOURCE_COMMITMENT_V4,
    SUPERSEDED_REVIEW_NOGO_V4,
    SUPERSEDED_SOURCE_COMMITMENT_V5,
    SUPERSEDED_REVIEW_GO_V5,
    SUPERSEDED_RELEASE_FAILURE_V5,
    SUPERSEDED_SOURCE_COMMITMENT_V6,
    SUPERSEDED_REVIEW_GO_V6,
    SUPERSEDED_DECISION_HOLD_V6,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
CLOSED = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
RESULT_ACTIONS = {
    "NO_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY": (
        "HOLD_NO_MECHANISM_CLAIM_FREEZE_INDEPENDENT_INTEGRATION_CONTEXT_"
        "DIAGNOSTIC_ONLY"
    ),
    "HYDROGEN_ANGLE_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY": (
        "HOLD_NO_MECHANISM_CLAIM_FREEZE_INDEPENDENT_VIOLATION_REPLAY_ONLY"
    ),
    "DIAGNOSTIC_EXECUTION_FAILED": (
        "HOLD_NO_INFERENCE_SEPARATE_REVIEWED_RECOVERY_ONLY"
    ),
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
_VERIFIED_ARCHIVE_IDENTITY: tuple[int, int, int, int, int] | None = None
PUBLICATION_MARKER = ".atypemu-publication-v1.json"


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


def parse_object(
    raw: bytes, label: str, *, require_canonical: bool = True
) -> dict[str, Any]:
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
    if not isinstance(value, dict) or (
        require_canonical and raw != canonical(value) + b"\n"
    ):
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


def sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file required: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
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
            raise ValueError(f"file changed while hashed: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def require_direct_directory(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(f"indirect directory path rejected: {relative}")
    if root.is_symlink() or root.resolve(strict=True) != root.absolute():
        raise PermissionError("repository root is indirect")
    current = root
    for part in relative.parts:
        current /= part
        metadata = current.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or current.resolve(strict=True) != current.absolute()
        ):
            raise PermissionError(f"directory ancestor is indirect: {current}")
    return current


def require_consolidation_archive(root: Path) -> Path:
    global _VERIFIED_ARCHIVE_IDENTITY
    require_direct_directory(root, CONSOLIDATION_ARCHIVE.parent)
    archive = root / CONSOLIDATION_ARCHIVE
    metadata = archive.stat(follow_symlinks=False)
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if _VERIFIED_ARCHIVE_IDENTITY != identity:
        if sha256_file(archive) != CONSOLIDATION_ARCHIVE_SHA256:
            raise PermissionError("canonical consolidation archive hash drifted")
        _VERIFIED_ARCHIVE_IDENTITY = identity
    return archive


def archive_members_bytes(
    root: Path, ceilings: dict[str, int]
) -> dict[str, bytes]:
    import zstandard  # type: ignore[import-not-found]

    archive = require_consolidation_archive(root)
    found: dict[str, bytes] = {}
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as stream:
                for member in stream:
                    if member.name not in ceilings:
                        continue
                    if member.name in found or not member.isfile():
                        raise ValueError(f"invalid archived member: {member.name}")
                    maximum = ceilings[member.name]
                    if member.size > maximum:
                        raise ValueError(
                            f"archive member exceeds ceiling: {member.name}"
                        )
                    handle = stream.extractfile(member)
                    if handle is None:
                        raise ValueError(
                            f"archive member cannot be read: {member.name}"
                        )
                    raw = handle.read(maximum + 1)
                    if len(raw) != member.size:
                        raise ValueError(
                            f"archive member framing drifted: {member.name}"
                        )
                    found[member.name] = raw
    if set(found) != set(ceilings):
        raise ValueError("required canonical archive member is absent")
    return found


def extract_runtime(root: Path, destination: Path) -> None:
    import zstandard  # type: ignore[import-not-found]

    archive = require_consolidation_archive(root)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            found = False
            with archive.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as stream:
                        for member in stream:
                            if member.name != RUNTIME_MEMBER:
                                continue
                            if found or not member.isfile() or member.size != 78_290_944:
                                raise ValueError("archived runtime member is invalid")
                            source = stream.extractfile(member)
                            if source is None:
                                raise ValueError("archived runtime cannot be read")
                            while chunk := source.read(1_048_576):
                                handle.write(chunk)
                            found = True
            if not found:
                raise ValueError("archived runtime member is absent")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    os.chmod(destination, 0o444)
    if (
        destination.stat().st_size != 78_290_944
        or sha256_file(destination) != RUNTIME_SHA256
    ):
        destination.unlink()
        raise PermissionError("archived OpenMM runtime identity drifted")


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


def publication_inventory(directory: Path) -> tuple[list[str], dict[str, str]]:
    directories: list[str] = []
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if relative == PUBLICATION_MARKER or path.is_symlink():
            if relative == PUBLICATION_MARKER and path.is_file() and not path.is_symlink():
                continue
            raise PermissionError(f"indirect publication member rejected: {relative}")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(relative)
        elif stat.S_ISREG(metadata.st_mode):
            files[relative] = sha256_file(path)
        else:
            raise PermissionError(f"special publication member rejected: {relative}")
    return directories, files


def copy_file_exclusive(source: Path, destination: Path) -> None:
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    destination_descriptor: int | None = None
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"regular publication file required: {source}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            stat.S_IMODE(before.st_mode),
        )
        while chunk := os.read(source_descriptor, 1_048_576):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("publication write made no progress")
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
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
            raise PermissionError(f"publication source changed while copied: {source}")
        os.fchmod(destination_descriptor, stat.S_IMODE(before.st_mode))
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def copy_tree_exclusive(source: Path, destination: Path) -> None:
    source_metadata = source.stat(follow_symlinks=False)
    if source.is_symlink() or not stat.S_ISDIR(source_metadata.st_mode):
        raise PermissionError("publication source directory is indirect or invalid")
    source_mode = stat.S_IMODE(source_metadata.st_mode)
    if source_mode != 0o555:
        raise PermissionError("publication source root must be read-only")
    marker_raw = read_file(source / PUBLICATION_MARKER, 2_000_000)
    if not marker_raw.endswith(b"\n") or marker_raw == b"\n":
        raise PermissionError("publication marker framing drifted")
    destination.mkdir(mode=0o700)
    marker_descriptor = os.open(
        destination / PUBLICATION_MARKER,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o000,
    )
    directories: list[tuple[Path, int]] = []
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if relative.as_posix() == PUBLICATION_MARKER:
                continue
            target = destination / relative
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink():
                raise PermissionError(f"publication symlink rejected: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                target.mkdir(mode=0o700)
                directories.append((target, stat.S_IMODE(metadata.st_mode)))
            elif stat.S_ISREG(metadata.st_mode):
                copy_file_exclusive(path, target)
            else:
                raise PermissionError(f"publication special file rejected: {relative}")
        for path, mode in reversed(directories):
            os.chmod(path, mode)
        os.chmod(destination, source_mode)
        view = memoryview(marker_raw[:-1])
        while view:
            written = os.write(marker_descriptor, view)
            if written <= 0:
                raise OSError("publication marker write made no progress")
            view = view[written:]
        os.fsync(marker_descriptor)
        os.fchmod(marker_descriptor, 0o444)
        if os.write(marker_descriptor, b"\n") != 1:
            raise OSError("publication marker completion write failed")
        os.fsync(marker_descriptor)
    finally:
        os.close(marker_descriptor)


def require_published_directory(directory: Path) -> None:
    metadata = directory.stat(follow_symlinks=False)
    if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("published directory is indirect or invalid")
    if stat.S_IMODE(metadata.st_mode) != 0o555:
        raise PermissionError("published directory root mode drifted")
    marker_raw = read_file(directory / PUBLICATION_MARKER, 2_000_000)
    marker = parse_object(marker_raw, "directory publication marker")
    directories, files = publication_inventory(directory)
    if any(
        stat.S_IMODE((directory / relative).stat(follow_symlinks=False).st_mode)
        != 0o555
        for relative in directories
    ) or any(
        stat.S_IMODE((directory / relative).stat(follow_symlinks=False).st_mode)
        != 0o444
        for relative in [*files, PUBLICATION_MARKER]
    ):
        raise PermissionError("published directory member mode drifted")
    if marker != {
        "contract": "atypemu_no_clobber_directory_publication_v1",
        "destination": directory.name,
        "directories": directories,
        "files": files,
        "root_mode": 0o555,
    }:
        raise PermissionError("directory publication inventory drifted")


def publish_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    directories, files = publication_inventory(source)
    source_mode = stat.S_IMODE(source.stat(follow_symlinks=False).st_mode)
    if source_mode != 0o555:
        raise PermissionError("publication source root must be read-only")
    marker = {
        "contract": "atypemu_no_clobber_directory_publication_v1",
        "destination": destination.name,
        "directories": directories,
        "files": files,
        "root_mode": source_mode,
    }
    try:
        os.chmod(source, 0o700)
        exclusive(source / PUBLICATION_MARKER, canonical(marker) + b"\n")
    finally:
        os.chmod(source, source_mode)
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100, os.fsencode(source), -100, os.fsencode(destination), 1
        ) == 0:
            require_published_directory(destination)
            return
        error = ctypes.get_errno()
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.EXDEV,
        }
        if error not in unsupported:
            raise OSError(error, os.strerror(error), str(destination))
    copy_tree_exclusive(source, destination)
    require_published_directory(destination)


def git_head(root: Path) -> str:
    replacements = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/replace"],
        cwd=root,
        env=HOST_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    alternates = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-path", "objects/info/alternates"],
            cwd=root,
            env=HOST_ENV,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if replacements or alternates.exists():
        raise PermissionError("Git replacement or alternate-object route is open")
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
        PREDECESSOR_FAILURE: PREDECESSOR_FAILURE_SHA256,
        PROJECTION: PROJECTION_SHA256,
        DECISION: DECISION_SHA256,
        INVENTORY: INVENTORY_SHA256,
        DECISION_REVIEW: DECISION_REVIEW_SHA256,
        SUPERSEDED_REVIEW_NOGO: SUPERSEDED_REVIEW_NOGO_SHA256,
        SUPERSEDED_REVIEW_NOGO_V2: SUPERSEDED_REVIEW_NOGO_V2_SHA256,
        SUPERSEDED_SOURCE_COMMITMENT_V3: SUPERSEDED_SOURCE_COMMITMENT_V3_SHA256,
        SUPERSEDED_REVIEW_NOGO_V3: SUPERSEDED_REVIEW_NOGO_V3_SHA256,
        SUPERSEDED_SOURCE_COMMITMENT_V4: SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256,
        SUPERSEDED_REVIEW_NOGO_V4: SUPERSEDED_REVIEW_NOGO_V4_SHA256,
        SUPERSEDED_SOURCE_COMMITMENT_V5: SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256,
        SUPERSEDED_REVIEW_GO_V5: SUPERSEDED_REVIEW_GO_V5_SHA256,
        SUPERSEDED_RELEASE_FAILURE_V5: SUPERSEDED_RELEASE_FAILURE_V5_SHA256,
        SUPERSEDED_SOURCE_COMMITMENT_V6: SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
        SUPERSEDED_REVIEW_GO_V6: SUPERSEDED_REVIEW_GO_V6_SHA256,
        SUPERSEDED_DECISION_HOLD_V6: SUPERSEDED_DECISION_HOLD_V6_SHA256,
    }
    if any(sha256(raw_by_path[path]) != digest for path, digest in expected.items()):
        raise PermissionError("fixed diagnostic input hash drifted")
    plan = parse_object(raw_by_path[PLAN], "diagnostic plan", require_canonical=False)
    predecessor_failure = parse_object(
        raw_by_path[PREDECESSOR_FAILURE],
        "predecessor runtime-version failure",
        require_canonical=False,
    )
    superseded_review = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_NOGO], "superseded final cold-review NO_GO"
    )
    superseded_review_v2 = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_NOGO_V2],
        "superseded v2 final cold-review NO_GO",
    )
    superseded_review_v3 = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_NOGO_V3],
        "superseded v3 final cold-review NO_GO",
    )
    superseded_review_v4 = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_NOGO_V4],
        "superseded v4 final cold-review NO_GO",
    )
    superseded_source_v5 = parse_object(
        raw_by_path[SUPERSEDED_SOURCE_COMMITMENT_V5],
        "superseded v5 source commitment",
    )
    superseded_review_v5 = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_GO_V5],
        "superseded v5 final cold-review GO",
    )
    superseded_release_failure_v5 = parse_object(
        raw_by_path[SUPERSEDED_RELEASE_FAILURE_V5],
        "superseded v5 release-validation failure",
    )
    superseded_source_v6 = parse_object(
        raw_by_path[SUPERSEDED_SOURCE_COMMITMENT_V6],
        "superseded v6 source commitment",
    )
    superseded_review_v6 = parse_object(
        raw_by_path[SUPERSEDED_REVIEW_GO_V6],
        "superseded v6 final cold-review GO",
    )
    superseded_decision_hold_v6 = parse_object(
        raw_by_path[SUPERSEDED_DECISION_HOLD_V6],
        "superseded v6 downstream-decision HOLD",
    )
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("scope") == "TARGET_UNREAD_DIAGNOSTIC_ONLY"
        and plan.get("entity_uid") == ENTITY_UID
        and plan.get("input_contract", {}).get("present_support_count") == 998
        and plan.get("input_contract", {}).get("missing_support_indices") == MISSING
        and plan.get("input_contract", {}).get("parent_pdb_root")
        == RECOVERY_PDB_ROOT.as_posix()
        and plan.get("condition", {}).get("proposal_pH") == "6.0"
        and plan.get("execution_contract", {}).get(
            "canonical_consolidation_archive_sha256"
        )
        == CONSOLIDATION_ARCHIVE_SHA256
        and plan.get("execution_contract", {}).get("runtime_sif_sha256")
        == RUNTIME_SHA256
        and plan.get("execution_contract", {}).get("worker_processes") == WORKERS
        and plan.get("execution_contract", {}).get("source_commitment")
        == "TRACKED_V7_EXACT_BLOB_PLUS_COMMIT_TRAILER_BOUND_TO_V6_DOWNSTREAM_DECISION_HOLD"
        and plan.get("execution_contract", {}).get("drvfs_publication")
        == "ATOMIC_DESTINATION_MKDIR_NO_CLOBBER_ROOT_MODE_BOUND_MARKER_LAST_COMPLETE_HASH_MANIFEST"
        and plan.get("execution_contract", {}).get("failure_sealing")
        == "EVERY_POST_CONSUMPTION_BASEEXCEPTION_SEALED_BEFORE_BEST_EFFORT_CLEANUP"
        and plan.get("review_contract")
        == {
            "any_post_review_byte_change_invalidates_review": True,
            "exactly_one_final_full_cold_review": True,
            "review_bundle": "GIT_COMMIT_PLUS_TRACKED_FROZEN_SOURCE_COMMITMENT",
            "review_source_ref": (
                "refs/heads/autoresearch/all-label-corrected-k8-20260905"
            ),
            "scoped_development_reviewers_allowed": 0,
        }
        and plan.get("superseded_source_bundle")
        == {
            "downstream_decision_hold_sha256": SUPERSEDED_DECISION_HOLD_V6_SHA256,
            "final_cold_review_go_sha256": SUPERSEDED_REVIEW_GO_V6_SHA256,
            "git_commit": "62f7588b2a05e843293844dace33f155128b3379",
            "ordinary_release_allowed": False,
            "source_commitment_sha256": SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
            "state": "V6_REVIEW_GO_RELEASE_HELD_FOR_UNFROZEN_DOWNSTREAM_DECISION_MAP",
        }
        and plan.get("decision_contract", {}).get("full_135_entity_route") == "CLOSED"
        and plan.get("decision_contract", {}).get(
            "promotion_or_feasibility_claim_allowed"
        )
        is False
        and plan.get("decision_contract", {}).get("downstream_action_map")
        == RESULT_ACTIONS
        and plan.get("failed_predecessor", {}).get("ordinary_rerun_allowed") is False
        and plan.get("failed_predecessor", {}).get("failure_receipt_sha256")
        == PREDECESSOR_FAILURE_SHA256
        and predecessor_failure.get("state")
        == "FAILED_CLOSED_PRE_PDB_RUNTIME_VERSION_ATTRIBUTE_MISMATCH"
        and predecessor_failure.get("parent_pdb_bytes_read") is False
        and predecessor_failure.get("recovery_policy", {}).get("ordinary_rerun_allowed")
        is False
        and predecessor_failure.get("recovery_policy", {}).get("required_candidate_id")
        == CANDIDATE_ID
        and predecessor_failure.get("observed_runtime", {}).get("openmm_full_version")
        == OPENMM_VERSION
        and predecessor_failure.get("observed_runtime", {}).get("git_revision")
        == OPENMM_GIT_REVISION
        and superseded_review.get("source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_SHA256
        and superseded_review.get("state") == "NO_GO"
        and superseded_review.get("git_commit")
        == "235470fbdb12e2561906940ead3798a1e0c2ec87"
        and superseded_review_v2.get("source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V2_SHA256
        and superseded_review_v2.get("state") == "NO_GO"
        and superseded_review_v2.get("scope") == "final_full_cold"
        and superseded_review_v2.get("git_commit")
        == "65a4e92fe79fa60667d0e0bff6f452bda364a427"
        and superseded_review_v3.get("source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V3_SHA256
        and superseded_review_v3.get("state") == "NO_GO"
        and superseded_review_v3.get("scope") == "final_full_cold"
        and superseded_review_v3.get("git_commit")
        == "b3b979b22ef38ead133e69ebfc4338db45276e33"
        and superseded_review_v4.get("source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256
        and superseded_review_v4.get("state") == "NO_GO"
        and superseded_review_v4.get("scope") == "final_full_cold"
        and superseded_review_v4.get("git_commit")
        == "e08613c08bdc0982a8968f29551e72e57fdeaaf0"
        and superseded_source_v5.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_commitment_v5"
        and superseded_source_v5.get("superseded_final_review_nogo_sha256")
        == SUPERSEDED_REVIEW_NOGO_V4_SHA256
        and superseded_source_v5.get("superseded_source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256
        and superseded_source_v5.get("state")
        == "FROZEN_RECOVERY_DIAGNOSTIC_ONLY_UNRUN"
        and superseded_review_v5
        == {
            "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_final_cold_review_go_v5",
            "elapsed_seconds": 720.0,
            "git_commit": "262c7e62aa3d68cbb0a0adece5b89395c2024168",
            "review_session": "ff15ipq-v5-final",
            "scope": "final_full_cold",
            "source_commitment_sha256": SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256,
            "state": "GO",
            "verified": [
                "v4_root_mode_blocker_closed",
                "prior_four_v3_blockers_closed",
                "full_frozen_scope_contract_causally_closed",
            ],
        }
        and superseded_release_failure_v5.get("state")
        == "FAIL_CLOSED_PRECONSUMPTION_INVALID_EXTERNAL_RECEIPT"
        and superseded_release_failure_v5.get("git_commit")
        == "262c7e62aa3d68cbb0a0adece5b89395c2024168"
        and superseded_release_failure_v5.get("source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256
        and superseded_release_failure_v5.get("review_sha256")
        == "4e4bcfc4dc81a4fe4254c191052726788e0be82553f13fe288af4f1d7e7bf450"
        and superseded_release_failure_v5.get("release_sha256")
        == "8b2c9f37fea211271af35dc7870a846d5a91957afa64f60e3b907b5bfdd37c8c"
        and superseded_release_failure_v5.get("consumed_marker_absent") is True
        and superseded_release_failure_v5.get("diagnostic_executed") is False
        and superseded_release_failure_v5.get("output_absent") is True
        and superseded_release_failure_v5.get("pdb_or_protected_surface_read")
        is False
        and superseded_source_v6.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_commitment_v6"
        and superseded_source_v6.get("superseded_final_review_go_sha256")
        == SUPERSEDED_REVIEW_GO_V5_SHA256
        and superseded_source_v6.get(
            "superseded_release_validation_failure_sha256"
        )
        == SUPERSEDED_RELEASE_FAILURE_V5_SHA256
        and superseded_source_v6.get("superseded_source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256
        and superseded_source_v6.get("state")
        == "FROZEN_RECOVERY_DIAGNOSTIC_ONLY_UNRUN"
        and superseded_review_v6
        == {
            "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_final_cold_review_go_v6",
            "elapsed_seconds": 300.0,
            "git_commit": "62f7588b2a05e843293844dace33f155128b3379",
            "review_session": "ff15ipq-v6-final",
            "scope": "final_full_cold",
            "source_commitment_sha256": SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
            "state": "GO",
            "verified": [
                "v5_invalid_external_receipts_causally_isolated",
                "all_prior_blockers_closed",
                "v6_full_frozen_scope_valid",
            ],
        }
        and superseded_decision_hold_v6.get("state")
        == "HOLD_BEFORE_V6_RELEASE_DOWNSTREAM_ACTION_MAP_UNFROZEN"
        and superseded_decision_hold_v6.get("controller_v6_git_commit")
        == "62f7588b2a05e843293844dace33f155128b3379"
        and superseded_decision_hold_v6.get(
            "controller_v6_source_commitment_sha256"
        )
        == SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256
        and superseded_decision_hold_v6.get("controller_v6_release_absent") is True
        and superseded_decision_hold_v6.get("v5_release_reusable") is False
        and superseded_decision_hold_v6.get("v5_diagnostic_executed") is False
    ):
        raise PermissionError("diagnostic plan identity drifted")
    decision = parse_object(
        raw_by_path[DECISION], "terminal decision", require_canonical=False
    )
    review = parse_object(
        raw_by_path[DECISION_REVIEW],
        "terminal decision review",
        require_canonical=False,
    )
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


def validate_static_scope(root: Path) -> None:
    if not all(
        not path.is_absolute()
        and ".." not in path.parts
        and path.suffix not in {".tar", ".tgz"}
        for path in SOURCE_FILES
    ):
        raise PermissionError("source path or archive surface is unsafe")
    for relative in SOURCE_FILES:
        require_direct_directory(root, relative.parent)
        current = root / relative
        if (
            current.is_symlink()
            or not current.is_file()
            or current.resolve(strict=True) != current.absolute()
        ):
            raise PermissionError(f"source path is indirect: {relative}")
    tree = ast.parse(read_file(root / SCRIPT, 2_000_000).decode("utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    if imported & {
        "pandas",
        "pyarrow",
        "requests",
        "sklearn",
        "socket",
        "torch",
        "urllib",
    }:
        raise PermissionError(
            "forbidden target, score, model, or network import is open"
        )
    audit_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_static_scope"
    )
    audit_nodes = {id(node) for node in ast.walk(audit_function)}
    strings = {
        node.value
        for node in ast.walk(tree)
        if id(node) not in audit_nodes
        and isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }
    if strings & {
        "/targets/",
        "target.parquet",
        "scores.parquet",
        "outer_metrics.json",
    }:
        raise PermissionError("forbidden target or score route is open")
    if set(CLOSED.values()) != {False}:
        raise PermissionError("a protected capability is open")


def require_source_trailer(root: Path, commit: str, source_hash: str) -> None:
    raw = subprocess.run(
        ["git", "cat-file", "commit", commit],
        cwd=root,
        env=HOST_ENV,
        check=True,
        capture_output=True,
    ).stdout
    if len(raw) > 1_000_000 or b"\n\n" not in raw:
        raise PermissionError("Git source commitment message is invalid")
    message = raw.split(b"\n\n", 1)[1].decode("utf-8", errors="strict")
    expected = (
        "AtypEmu-Source-Commitment-SHA256: " + source_hash
    )
    expected_path = "AtypEmu-Source-Commitment-Path: " + SOURCE_COMMITMENT.as_posix()
    if message.splitlines().count(expected) != 1 or message.splitlines().count(
        expected_path
    ) != 1:
        raise PermissionError("Git source commitment trailer drifted")


def freeze_source(root: Path) -> None:
    raw = read_file(root / SOURCE_COMMITMENT, 1_000_000)
    require_source(root, sha256(raw))


def require_source(
    root: Path, expected_hash: str
) -> tuple[dict[str, Any], bytes, str]:
    if SHA256_RE.fullmatch(expected_hash) is None:
        raise PermissionError("source commitment hash is invalid")
    raw = read_file(root / SOURCE_COMMITMENT, 1_000_000)
    if sha256(raw) != expected_hash:
        raise PermissionError("source commitment hash drifted")
    source = parse_object(raw, "source commitment")
    validate_static_scope(root)
    require_consolidation_archive(root)
    commit = git_head(root)
    if committed_bytes(root, commit, SOURCE_COMMITMENT) != raw:
        raise PermissionError("source commitment is not the committed blob")
    require_source_trailer(root, commit, expected_hash)
    if set(source) != {
        "candidate_id",
        "consolidation_archive_sha256",
        "contract",
        "files",
        "git_binding",
        "runtime_sif_sha256",
        "state",
        "superseded_downstream_decision_hold_sha256",
        "superseded_final_review_go_sha256",
        "superseded_source_commitment_sha256",
    } or not (
        source["candidate_id"] == CANDIDATE_ID
        and source["consolidation_archive_sha256"]
        == CONSOLIDATION_ARCHIVE_SHA256
        and source["contract"]
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_commitment_v7"
        and source["git_binding"]
        == "CURRENT_CLEAN_HEAD_EXACT_BLOB_AND_COMMIT_TRAILER"
        and source["runtime_sif_sha256"] == RUNTIME_SHA256
        and source["state"] == "FROZEN_RECOVERY_DIAGNOSTIC_ONLY_UNRUN"
        and source["superseded_downstream_decision_hold_sha256"]
        == SUPERSEDED_DECISION_HOLD_V6_SHA256
        and source["superseded_final_review_go_sha256"]
        == SUPERSEDED_REVIEW_GO_V6_SHA256
        and source["superseded_source_commitment_sha256"]
        == SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256
        and isinstance(source["files"], dict)
        and set(source["files"]) == {str(path) for path in SOURCE_FILES}
    ):
        raise PermissionError("source commitment identity drifted")
    raw_by_path = {
        path: committed_bytes(root, commit, path) for path in SOURCE_FILES
    }
    if any(
        sha256(raw_by_path[path]) != source["files"][str(path)] for path in SOURCE_FILES
    ):
        raise PermissionError("source file hash drifted")
    validate_fixed_inputs(raw_by_path)
    return source, raw, commit


def stage_source(root: Path, expected_hash: str) -> None:
    _, source_raw, commit = require_source(root, expected_hash)
    destination = root / STAGE
    require_direct_directory(root, STAGE.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ff15ipq-angle-recovery-stage-", dir=(root / STAGE).parent
    ) as text:
        temporary = Path(text)
        scripts = temporary / "scripts"
        inputs = temporary / "inputs"
        scripts.mkdir(mode=0o700)
        inputs.mkdir(mode=0o700)
        for relative in SOURCE_FILES:
            raw = committed_bytes(root, commit, relative)
            directory = inputs if relative == PROJECTION else scripts
            exclusive(directory / relative.name, raw)
        exclusive(scripts / SOURCE_COMMITMENT.name, source_raw)
        extract_runtime(root, temporary / "runtime.sif")
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

        if (
            getattr(openmm.version, "full_version", None) != OPENMM_VERSION
            or getattr(openmm.version, "git_revision", None) != OPENMM_GIT_REVISION
        ):
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


def validate_final_review(
    review: dict[str, Any], source_hash: str, git_commit: str
) -> None:
    reviews = review.get("reviews")
    if not (
        set(review)
        == {
            "candidate_id",
            "closed_capabilities",
            "contract",
            "reviews",
            "source_commitment_sha256",
            "source_git_commit",
            "state",
        }
        and review.get("candidate_id") == CANDIDATE_ID
        and review.get("closed_capabilities") == CLOSED
        and review.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_review_v7"
        and review.get("source_commitment_sha256") == source_hash
        and GIT_COMMIT_RE.fullmatch(git_commit) is not None
        and review.get("source_git_commit") == git_commit
        and review.get("state") == "GO_DIAGNOSTIC_ONLY"
        and isinstance(reviews, list)
        and len(reviews) == 1
        and all(
            isinstance(row, dict)
            and set(row)
            == {"elapsed_seconds", "review_session", "scope", "verdict"}
            and type(row["elapsed_seconds"]) in {int, float}
            and math.isfinite(row["elapsed_seconds"])
            and row["elapsed_seconds"] > 0
            and isinstance(row["review_session"], str)
            and bool(row["review_session"])
            and row["scope"] == "final_full_cold"
            and row["verdict"] == "GO"
            for row in reviews
        )
    ):
        raise PermissionError("final cold-review receipt drifted")


def require_worker_release(stage: Path) -> None:
    expected_source = os.environ.get("ATYPEMU_DIAGNOSTIC_SOURCE_SHA256")
    expected_release = os.environ.get("ATYPEMU_DIAGNOSTIC_RELEASE_SHA256")
    if not expected_source or not expected_release:
        raise PermissionError("worker source/release binding is absent")
    source_raw = read_file(stage / "scripts" / SOURCE_COMMITMENT.name, 1_000_000)
    review_raw = read_file(stage / "release/source_review.json", 1_000_000)
    release_raw = read_file(stage / "release/execution_release.json", 1_000_000)
    consumed_raw = read_file(stage / "release/execution_consumed.json", 1_000_000)
    if sha256(source_raw) != expected_source or sha256(release_raw) != expected_release:
        raise PermissionError("worker source/release hash drifted")
    source = parse_object(source_raw, "worker source commitment")
    review = parse_object(review_raw, "worker source review")
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
    source_fields = {
        "candidate_id",
        "consolidation_archive_sha256",
        "contract",
        "files",
        "git_binding",
        "runtime_sif_sha256",
        "state",
        "superseded_downstream_decision_hold_sha256",
        "superseded_final_review_go_sha256",
        "superseded_source_commitment_sha256",
    }
    validate_final_review(review, expected_source, release.get("git_commit", ""))
    if not (
        set(source) == source_fields
        and source.get("candidate_id") == CANDIDATE_ID
        and source.get("consolidation_archive_sha256")
        == CONSOLIDATION_ARCHIVE_SHA256
        and source.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_commitment_v7"
        and source.get("git_binding")
        == "CURRENT_CLEAN_HEAD_EXACT_BLOB_AND_COMMIT_TRAILER"
        and source.get("runtime_sif_sha256") == RUNTIME_SHA256
        and source.get("state") == "FROZEN_RECOVERY_DIAGNOSTIC_ONLY_UNRUN"
        and source.get("superseded_downstream_decision_hold_sha256")
        == SUPERSEDED_DECISION_HOLD_V6_SHA256
        and source.get("superseded_final_review_go_sha256")
        == SUPERSEDED_REVIEW_GO_V6_SHA256
        and source.get("superseded_source_commitment_sha256")
        == SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256
        and isinstance(source.get("files"), dict)
        and source["files"]
        == {
            str(SCRIPT): sha256(script_raw),
            str(PLAN): PLAN_SHA256,
            str(PREDECESSOR_FAILURE): PREDECESSOR_FAILURE_SHA256,
            str(PROJECTION): PROJECTION_SHA256,
            str(DECISION): DECISION_SHA256,
            str(INVENTORY): INVENTORY_SHA256,
            str(DECISION_REVIEW): DECISION_REVIEW_SHA256,
            str(SUPERSEDED_REVIEW_NOGO): SUPERSEDED_REVIEW_NOGO_SHA256,
            str(SUPERSEDED_REVIEW_NOGO_V2): SUPERSEDED_REVIEW_NOGO_V2_SHA256,
            str(SUPERSEDED_SOURCE_COMMITMENT_V3): SUPERSEDED_SOURCE_COMMITMENT_V3_SHA256,
            str(SUPERSEDED_REVIEW_NOGO_V3): SUPERSEDED_REVIEW_NOGO_V3_SHA256,
            str(SUPERSEDED_SOURCE_COMMITMENT_V4): SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256,
            str(SUPERSEDED_REVIEW_NOGO_V4): SUPERSEDED_REVIEW_NOGO_V4_SHA256,
            str(SUPERSEDED_SOURCE_COMMITMENT_V5): SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256,
            str(SUPERSEDED_REVIEW_GO_V5): SUPERSEDED_REVIEW_GO_V5_SHA256,
            str(SUPERSEDED_RELEASE_FAILURE_V5): SUPERSEDED_RELEASE_FAILURE_V5_SHA256,
            str(SUPERSEDED_SOURCE_COMMITMENT_V6): SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
            str(SUPERSEDED_REVIEW_GO_V6): SUPERSEDED_REVIEW_GO_V6_SHA256,
            str(SUPERSEDED_DECISION_HOLD_V6): SUPERSEDED_DECISION_HOLD_V6_SHA256,
        }
        and set(release) == release_fields
        and set(consumed) == consumed_fields
        and release.get("candidate_id") == CANDIDATE_ID
        and release.get("closed_capabilities") == CLOSED
        and release.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_execution_release_v7"
        and release.get("entity_uid") == ENTITY_UID
        and GIT_COMMIT_RE.fullmatch(release.get("git_commit", "")) is not None
        and release.get("mode") == "DIAGNOSTIC_ONLY_998_SUPPORTS"
        and release.get("source_commitment_sha256") == expected_source
        and release.get("source_review_sha256") == sha256(review_raw)
        and release.get("runtime_sif_sha256") == RUNTIME_SHA256
        and os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256") == RUNTIME_SHA256
        and release.get("state") == "EXTERNALLY_RELEASED_ONCE_RECOVERY_DIAGNOSTIC_ONLY"
        and consumed.get("candidate_id") == CANDIDATE_ID
        and consumed.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_consumption_v7"
        and consumed.get("execution_release_sha256") == expected_release
        and consumed.get("source_commitment_sha256") == expected_source
        and consumed.get("state") == "CONSUMED_BEFORE_RECOVERY_DIAGNOSTIC_PDB_ACCESS"
    ):
        raise PermissionError("worker release/consumption identity drifted")


def worker(access: object) -> None:
    if access is not _WORKER_SEAL:
        raise PermissionError("worker requires a process-local released capability")
    script = Path(__file__).absolute()
    stage = script.parent.parent
    if (
        script.is_symlink()
        or script.resolve(strict=True) != script
        or os.environ.get("ATYPEMU_STAGE_ROOT") != str(stage)
    ):
        raise PermissionError("worker script is outside the released stage")
    require_worker_release(stage)
    output = stage / "output"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise PermissionError(
            "worker result directory is absent, indirect, or nonempty"
        )
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
    with context.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = list(pool.imap(diagnose_support, tasks, chunksize=1))
    if any(type(row.get("support_index")) is not int for row in rows) or [
        row["support_index"] for row in rows
    ] != SUPPORTS:
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
    result_raw = b"".join(canonical(row) + b"\n" for row in rows)
    summary = {
        "candidate_id": CANDIDATE_ID,
        "downstream_action": RESULT_ACTIONS[result_class],
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
    root: Path, git_commit: str, source_hash: str, release_hash: str
) -> tuple[bytes, bytes]:
    if SHA256_RE.fullmatch(release_hash) is None:
        raise PermissionError("external release hash is invalid")
    review_raw = read_file(root / SOURCE_REVIEW, 1_000_000)
    release_raw = read_file(root / RELEASE, 1_000_000)
    if sha256(release_raw) != release_hash:
        raise PermissionError("external release hash drifted")
    review = parse_object(review_raw, "source review")
    release = parse_object(release_raw, "execution release")
    validate_final_review(review, source_hash, git_commit)
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
        set(release) == release_fields
        and release.get("candidate_id") == CANDIDATE_ID
        and release.get("closed_capabilities") == CLOSED
        and release.get("contract")
        == "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_execution_release_v7"
        and release.get("entity_uid") == ENTITY_UID
        and release.get("git_commit") == git_commit
        and release.get("mode") == "DIAGNOSTIC_ONLY_998_SUPPORTS"
        and release.get("source_commitment_sha256") == source_hash
        and release.get("source_review_sha256") == sha256(review_raw)
        and release.get("runtime_sif_sha256") == RUNTIME_SHA256
        and release.get("state") == "EXTERNALLY_RELEASED_ONCE_RECOVERY_DIAGNOSTIC_ONLY"
    ):
        raise PermissionError("source review or execution release drifted")
    return review_raw, release_raw


def validate_angle_record(value: Any, label: str) -> None:
    identity_fields = {
        "atom_index",
        "atom_name",
        "chain_id",
        "chain_index",
        "residue_id",
        "residue_index",
        "residue_name",
    }
    if not isinstance(value, dict) or set(value) != {
        "angle_degrees",
        "hydrogen",
        "other",
        "parent",
    }:
        raise ValueError(f"{label} schema drifted")
    angle = value["angle_degrees"]
    if (
        type(angle) not in {int, float}
        or not math.isfinite(angle)
        or not 0.0 <= angle <= 180.0
    ):
        raise ValueError(f"{label} angle is invalid")
    for role in ("hydrogen", "other", "parent"):
        identity = value[role]
        if not isinstance(identity, dict) or set(identity) != identity_fields:
            raise ValueError(f"{label} {role} identity drifted")
        if not (
            type(identity["atom_index"]) is int
            and identity["atom_index"] >= 0
            and type(identity["chain_index"]) is int
            and identity["chain_index"] >= 0
            and type(identity["residue_index"]) is int
            and identity["residue_index"] >= 0
            and all(
                isinstance(identity[field], str)
                for field in ("atom_name", "chain_id", "residue_id", "residue_name")
            )
            and bool(identity["atom_name"])
            and bool(identity["residue_name"])
        ):
            raise ValueError(f"{label} {role} identity is invalid")


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
    for row in errors:
        if not (
            set(row)
            == {"error_message", "error_type", "status", "support_index"}
            and isinstance(row["error_message"], str)
            and bool(row["error_message"])
            and isinstance(row["error_type"], str)
            and bool(row["error_type"])
        ):
            raise ValueError("diagnostic error row drifted")
    for row in ok_rows:
        if not (
            set(row)
            == {
                "minimum_angle",
                "status",
                "support_index",
                "violation_count",
                "violations",
            }
            and type(row["violation_count"]) is int
            and row["violation_count"] >= 0
            and isinstance(row["violations"], list)
            and row["violation_count"] == len(row["violations"])
        ):
            raise ValueError("diagnostic OK row drifted")
        validate_angle_record(row["minimum_angle"], "support minimum")
        for violation in row["violations"]:
            validate_angle_record(violation, "support violation")
            if violation["angle_degrees"] >= LOW_ANGLE:
                raise ValueError("nonviolating angle was classified as a violation")
        if len({canonical(value) for value in row["violations"]}) != len(
            row["violations"]
        ):
            raise ValueError("duplicate support violation")
        minimum_angle = row["minimum_angle"]["angle_degrees"]
        if row["violations"]:
            if row["minimum_angle"] not in row["violations"] or minimum_angle != min(
                value["angle_degrees"] for value in row["violations"]
            ):
                raise ValueError("support minimum/violation relation drifted")
        elif minimum_angle < LOW_ANGLE:
            raise ValueError("support minimum violation was omitted")
    violations = [row for row in ok_rows if row["violation_count"] > 0]
    minimum_row = (
        min(ok_rows, key=lambda row: row["minimum_angle"]["angle_degrees"])
        if ok_rows
        else None
    )
    expected_minimum = (
        {"support_index": minimum_row["support_index"], **minimum_row["minimum_angle"]}
        if minimum_row
        else None
    )
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
            "downstream_action",
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
        and summary.get("downstream_action") == RESULT_ACTIONS[expected_class]
        and type(summary.get("support_count")) is int
        and type(summary.get("error_support_count")) is int
        and type(summary.get("violation_support_count")) is int
        and summary.get("support_count") == 998
        and summary.get("support_results_sha256") == sha256(results_raw)
        and summary.get("error_support_count") == len(errors)
        and summary.get("violation_support_count") == len(violations)
        and summary.get("global_minimum_angle") == expected_minimum
        and summary.get("result_class") == expected_class
        and summary.get("state") == "SEALED_TARGET_UNREAD_DIAGNOSTIC_ONLY"
        and summary.get("full_135_entity_route") == "CLOSED"
        and summary.get("promotion_or_feasibility_claim_allowed") is False
    ):
        raise ValueError("diagnostic summary drifted")
    return summary


def diagnostic_command(
    root: Path, output: Path, source_hash: str, release_hash: str
) -> list[str]:
    return [
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
        "--container-controller",
    ]


def execute_diagnostic(root: Path, source_hash: str, release_hash: str) -> None:
    _, source_raw, git_commit = require_source(root, source_hash)
    review_raw, release_raw = validate_release(
        root, git_commit, source_hash, release_hash
    )
    stage = require_direct_directory(root, STAGE)
    require_published_directory(stage)
    require_direct_directory(root, OUTPUT.parent)
    require_direct_directory(root, RECOVERY_PDB_ROOT)
    if any(
        (root / path).exists() or (root / path).is_symlink()
        for path in (CONSUMED, FAILURE, OUTPUT)
    ):
        raise FileExistsError("diagnostic terminal path already exists")
    runtime_raw = read_file(stage / "runtime.sif")
    if sha256(runtime_raw) != RUNTIME_SHA256:
        raise PermissionError("staged runtime drifted")
    output = root / OUTPUT

    def seal_failure(error: BaseException) -> None:
        failure = {
            "candidate_id": CANDIDATE_ID,
            "downstream_action": RESULT_ACTIONS["DIAGNOSTIC_EXECUTION_FAILED"],
            "error_message": str(error),
            "error_type": type(error).__name__,
            "execution_release_sha256": release_hash,
            "result_class": "DIAGNOSTIC_EXECUTION_FAILED",
            "source_commitment_sha256": source_hash,
            "state": "FAILED_AFTER_RECOVERY_DIAGNOSTIC_CONSUMPTION",
        }
        exclusive(root / FAILURE, canonical(failure) + b"\n")

    consumed = {
        "candidate_id": CANDIDATE_ID,
        "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_consumption_v7",
        "execution_release_sha256": release_hash,
        "source_commitment_sha256": source_hash,
        "state": "CONSUMED_BEFORE_RECOVERY_DIAGNOSTIC_PDB_ACCESS",
    }
    consumed_raw = canonical(consumed) + b"\n"
    process: subprocess.Popen[bytes] | None = None
    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=".ff15ipq-angle-output-", dir=root / ".auto")
        )
        (temporary / "release").mkdir(mode=0o700)
        (temporary / "results").mkdir(mode=0o700)
        (temporary / "scripts").mkdir(mode=0o700)
        (temporary / "inputs").mkdir(mode=0o700)
        exclusive(
            temporary / "scripts" / SCRIPT.name,
            committed_bytes(root, git_commit, SCRIPT),
        )
        exclusive(
            temporary / "scripts" / SOURCE_COMMITMENT.name,
            source_raw,
        )
        exclusive(
            temporary / "inputs" / PROJECTION.name,
            committed_bytes(root, git_commit, PROJECTION),
        )
        exclusive(temporary / "runtime.sif", runtime_raw)
        exclusive(temporary / "release/execution_release.json", release_raw)
        exclusive(temporary / "release/execution_consumed.json", consumed_raw)
        exclusive(temporary / "release/source_review.json", review_raw)
        command = diagnostic_command(root, temporary, source_hash, release_hash)

        exclusive(root / CONSUMED, consumed_raw)
        process = subprocess.Popen(
            command,
            cwd=root,
            env=HOST_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=21_600)
        exclusive(temporary / "stdout.bin", stdout)
        exclusive(temporary / "stderr.bin", stderr)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )
        summary_raw = read_file(temporary / "results/summary.json", 2_000_000)
        results_raw = read_file(
            temporary / "results/support_results.jsonl", 100_000_000
        )
        summary = validate_result_bytes(summary_raw, results_raw)
        terminal = {
            "candidate_id": CANDIDATE_ID,
            "downstream_action": RESULT_ACTIONS[summary["result_class"]],
            "execution_release_sha256": release_hash,
            "result_class": summary["result_class"],
            "source_commitment_sha256": source_hash,
            "state": "PASS_RECOVERY_DIAGNOSTIC_EXECUTION_ONLY",
            "stderr_sha256": sha256(stderr),
            "stdout_sha256": sha256(stdout),
            "summary_sha256": sha256(summary_raw),
            "support_results_sha256": sha256(results_raw),
        }
        exclusive(
            temporary / "terminal_receipt.json", canonical(terminal) + b"\n"
        )
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()), reverse=True
        ):
            os.chmod(directory, 0o555)
        os.chmod(temporary, 0o555)
        publish_directory(temporary, output)
        require_published_directory(output)
    except BaseException as error:
        if (root / CONSUMED).exists() or (root / CONSUMED).is_symlink():
            seal_failure(error)
        raise
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                pass
        if temporary is not None and temporary.exists():
            try:
                os.chmod(temporary, 0o700)
                for directory in (
                    path for path in temporary.rglob("*") if path.is_dir()
                ):
                    os.chmod(directory, 0o700)
                shutil.rmtree(temporary)
            except BaseException:
                pass


def self_test(root: Path) -> int:
    checks = 0
    for invalid in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b"[]\n"):
        try:
            parse_object(invalid, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON accepted")
        checks += 1
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
    checks += 3
    try:
        worker(None)
    except PermissionError:
        pass
    else:
        raise AssertionError("worker capability failed open")
    checks += 1

    git_head(root)
    validate_static_scope(root)
    fixed = {path: read_file(root / path, 20_000_000) for path in SOURCE_FILES}
    validate_fixed_inputs(fixed)
    if (
        sha256(read_file(root / SUPERSEDED_SOURCE_COMMITMENT_V6, 1_000_000))
        != SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256
    ):
        raise AssertionError("superseded source commitment drifted")
    current_source_raw = read_file(root / SOURCE_COMMITMENT, 1_000_000)
    _, checked_source_raw, checked_commit = require_source(
        root, sha256(current_source_raw)
    )
    if checked_source_raw != current_source_raw or checked_commit != git_head(root):
        raise AssertionError("current source commitment binding drifted")
    if len(load_projection(root / PROJECTION)) != 998:
        raise AssertionError("projection/count replay drifted")
    checks += 6

    predecessor = parse_object(
        read_file(root / PREDECESSOR_FAILURE),
        "predecessor runtime-version failure",
        require_canonical=False,
    )
    predecessor_raw = archive_members_bytes(
        root, {member: 100_000_000 for member in PREDECESSOR_MEMBERS.values()}
    )
    if any(
        sha256(predecessor_raw[member]) != predecessor[key]
        for key, member in PREDECESSOR_MEMBERS.items()
    ):
        raise AssertionError("predecessor terminal evidence drifted")
    predecessor_rows = [
        parse_object(line, f"predecessor result {index}")
        for index, line in enumerate(
            predecessor_raw[
                PREDECESSOR_MEMBERS["support_results_sha256"]
            ].splitlines(keepends=True)
        )
    ]
    if not (
        len(predecessor_rows) == 998
        and [row.get("support_index") for row in predecessor_rows] == SUPPORTS
        and all(
            row.get("status") == "ERROR"
            and row.get("error_type") == "RuntimeError"
            and row.get("error_message") == "OpenMM version drifted"
            for row in predecessor_rows
        )
    ):
        raise AssertionError("predecessor error inventory drifted")
    checks += 2

    synthetic_rows = [
        {
            "error_message": "synthetic",
            "error_type": "RuntimeError",
            "status": "ERROR",
            "support_index": index,
        }
        for index in SUPPORTS
    ]
    synthetic_results = b"".join(canonical(row) + b"\n" for row in synthetic_rows)
    synthetic_summary = {
        "candidate_id": CANDIDATE_ID,
        "downstream_action": RESULT_ACTIONS["DIAGNOSTIC_EXECUTION_FAILED"],
        "error_support_count": 998,
        "full_135_entity_route": "CLOSED",
        "global_minimum_angle": None,
        "promotion_or_feasibility_claim_allowed": False,
        "result_class": "DIAGNOSTIC_EXECUTION_FAILED",
        "state": "SEALED_TARGET_UNREAD_DIAGNOSTIC_ONLY",
        "support_count": 998,
        "support_results_sha256": sha256(synthetic_results),
        "violation_support_count": 0,
    }
    validate_result_bytes(canonical(synthetic_summary) + b"\n", synthetic_results)
    try:
        validate_result_bytes(
            canonical(synthetic_summary) + b"\n",
            b"".join(canonical(row) + b"\n" for row in synthetic_rows[:-1]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete result surface accepted")
    checks += 2

    identity = {
        "atom_index": 1,
        "atom_name": "H",
        "chain_id": "A",
        "chain_index": 0,
        "residue_id": "1",
        "residue_index": 0,
        "residue_name": "ALA",
    }
    angle_record = {
        "angle_degrees": 50.0,
        "hydrogen": identity,
        "other": {**identity, "atom_index": 2, "atom_name": "CA"},
        "parent": {**identity, "atom_index": 0, "atom_name": "N"},
    }
    ok_row = {
        "minimum_angle": angle_record,
        "status": "OK",
        "support_index": SUPPORTS[0],
        "violation_count": 1,
        "violations": [angle_record],
    }
    mixed_rows = [ok_row, *synthetic_rows[1:]]
    mixed_results = b"".join(canonical(row) + b"\n" for row in mixed_rows)
    mixed_summary = {
        **synthetic_summary,
        "error_support_count": 997,
        "global_minimum_angle": {"support_index": SUPPORTS[0], **angle_record},
        "support_results_sha256": sha256(mixed_results),
        "violation_support_count": 1,
    }
    validate_result_bytes(canonical(mixed_summary) + b"\n", mixed_results)
    tampered_summary = {
        **mixed_summary,
        "global_minimum_angle": {
            **mixed_summary["global_minimum_angle"],
            "angle_degrees": 51.0,
        },
    }
    try:
        validate_result_bytes(canonical(tampered_summary) + b"\n", mixed_results)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered global minimum was accepted")
    safe_angle_record = {**angle_record, "angle_degrees": 100.0}
    safe_rows = [
        {
            "minimum_angle": safe_angle_record,
            "status": "OK",
            "support_index": index,
            "violation_count": 0,
            "violations": [],
        }
        for index in SUPPORTS
    ]
    safe_results = b"".join(canonical(row) + b"\n" for row in safe_rows)
    safe_summary = {
        **synthetic_summary,
        "downstream_action": RESULT_ACTIONS[
            "NO_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
        ],
        "error_support_count": 0,
        "global_minimum_angle": {
            "support_index": SUPPORTS[0],
            **safe_angle_record,
        },
        "result_class": "NO_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY",
        "support_results_sha256": sha256(safe_results),
    }
    validate_result_bytes(canonical(safe_summary) + b"\n", safe_results)
    violation_rows = [ok_row, *safe_rows[1:]]
    violation_results = b"".join(
        canonical(row) + b"\n" for row in violation_rows
    )
    violation_summary = {
        **safe_summary,
        "downstream_action": RESULT_ACTIONS[
            "HYDROGEN_ANGLE_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY"
        ],
        "global_minimum_angle": {"support_index": SUPPORTS[0], **angle_record},
        "result_class": "HYDROGEN_ANGLE_FAILURE_REPRODUCED_DIAGNOSTIC_ONLY",
        "support_results_sha256": sha256(violation_results),
        "violation_support_count": 1,
    }
    validate_result_bytes(
        canonical(violation_summary) + b"\n", violation_results
    )
    try:
        validate_result_bytes(
            canonical({**violation_summary, "downstream_action": "PROMOTE"}) + b"\n",
            violation_results,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen downstream action was accepted")
    checks += 5

    synthetic_commit = "0" * 40
    synthetic_review = {
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED,
        "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_review_v7",
        "reviews": [
            {
                "elapsed_seconds": 1.0,
                "review_session": "synthetic-final-review",
                "scope": "final_full_cold",
                "verdict": "GO",
            }
        ],
        "source_commitment_sha256": "0" * 64,
        "source_git_commit": synthetic_commit,
        "state": "GO_DIAGNOSTIC_ONLY",
    }
    validate_final_review(synthetic_review, "0" * 64, synthetic_commit)
    duplicate_review = {
        **synthetic_review,
        "reviews": synthetic_review["reviews"] * 2,
    }
    try:
        validate_final_review(duplicate_review, "0" * 64, synthetic_commit)
    except PermissionError:
        pass
    else:
        raise AssertionError("multiple final reviewers were accepted")
    checks += 2

    with tempfile.TemporaryDirectory(
        prefix="ff15ipq-angle-self-test-", dir=root / ".auto"
    ) as text:
        test_root = Path(text)
        marker = test_root / "consumed.json"
        exclusive(marker, b"{}\n")
        try:
            exclusive(marker, b"{}\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("one-shot receipt overwrite accepted")
        source = test_root / "source"
        destination = test_root / "destination"
        source.mkdir()
        exclusive(source / "payload", b"fixed")
        os.chmod(source, 0o555)
        publish_directory(source, destination)
        if stat.S_IMODE(destination.stat().st_mode) != 0o555:
            raise AssertionError("atomic publication root mode drifted")
        os.chmod(destination, 0o700)
        try:
            require_published_directory(destination)
        except PermissionError:
            pass
        else:
            raise AssertionError("publication root-mode tamper was accepted")
        os.chmod(destination, 0o555)
        replacement = test_root / "replacement"
        replacement.mkdir()
        try:
            publish_directory(replacement, destination)
        except FileExistsError:
            pass
        else:
            raise AssertionError("no-clobber directory publication failed open")
        fallback_source = test_root / "fallback-source"
        fallback_destination = test_root / "fallback-destination"
        fallback_source.mkdir()
        exclusive(fallback_source / "payload", b"fallback")
        directories, files = publication_inventory(fallback_source)
        exclusive(
            fallback_source / PUBLICATION_MARKER,
            canonical(
                {
                    "contract": "atypemu_no_clobber_directory_publication_v1",
                    "destination": fallback_destination.name,
                    "directories": directories,
                    "files": files,
                    "root_mode": 0o555,
                }
            )
            + b"\n",
        )
        os.chmod(fallback_source, 0o555)
        copy_tree_exclusive(fallback_source, fallback_destination)
        require_published_directory(fallback_destination)
        if stat.S_IMODE(fallback_destination.stat().st_mode) != 0o555:
            raise AssertionError("fallback publication root mode drifted")
        partial_source = test_root / "partial-source"
        partial_destination = test_root / "partial-destination"
        partial_source.mkdir()
        exclusive(partial_source / PUBLICATION_MARKER, b"{}\n")
        exclusive(partial_source / "payload", b"partial")
        (partial_source / "z-indirect").symlink_to("payload")
        os.chmod(partial_source, 0o555)
        try:
            copy_tree_exclusive(partial_source, partial_destination)
        except PermissionError:
            pass
        else:
            raise AssertionError("partial fallback publication unexpectedly passed")
        try:
            require_published_directory(partial_destination)
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        else:
            raise AssertionError("partial fallback publication was accepted")
        payload = destination / "payload"
        os.chmod(destination, 0o700)
        os.chmod(payload, 0o600)
        payload.write_bytes(b"changed")
        os.chmod(payload, 0o444)
        os.chmod(destination, 0o555)
        try:
            require_published_directory(destination)
        except PermissionError:
            pass
        else:
            raise AssertionError("tampered publication inventory was accepted")
        direct = test_root / "direct"
        indirect = test_root / "indirect"
        direct.mkdir()
        indirect.symlink_to(direct, target_is_directory=True)
        try:
            require_direct_directory(test_root, Path("indirect"))
        except PermissionError:
            pass
        else:
            raise AssertionError("indirect directory ancestor was accepted")

        worker_stage = test_root / "worker-stage"
        (worker_stage / "scripts").mkdir(parents=True)
        (worker_stage / "release").mkdir()
        script_raw = read_file(root / SCRIPT, 2_000_000)
        source_object = {
            "candidate_id": CANDIDATE_ID,
            "consolidation_archive_sha256": CONSOLIDATION_ARCHIVE_SHA256,
            "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_source_commitment_v7",
            "files": {
                str(SCRIPT): sha256(script_raw),
                str(PLAN): PLAN_SHA256,
                str(PREDECESSOR_FAILURE): PREDECESSOR_FAILURE_SHA256,
                str(PROJECTION): PROJECTION_SHA256,
                str(DECISION): DECISION_SHA256,
                str(INVENTORY): INVENTORY_SHA256,
                str(DECISION_REVIEW): DECISION_REVIEW_SHA256,
                str(SUPERSEDED_REVIEW_NOGO): SUPERSEDED_REVIEW_NOGO_SHA256,
                str(SUPERSEDED_REVIEW_NOGO_V2): SUPERSEDED_REVIEW_NOGO_V2_SHA256,
                str(SUPERSEDED_SOURCE_COMMITMENT_V3): SUPERSEDED_SOURCE_COMMITMENT_V3_SHA256,
                str(SUPERSEDED_REVIEW_NOGO_V3): SUPERSEDED_REVIEW_NOGO_V3_SHA256,
                str(SUPERSEDED_SOURCE_COMMITMENT_V4): SUPERSEDED_SOURCE_COMMITMENT_V4_SHA256,
                str(SUPERSEDED_REVIEW_NOGO_V4): SUPERSEDED_REVIEW_NOGO_V4_SHA256,
                str(SUPERSEDED_SOURCE_COMMITMENT_V5): SUPERSEDED_SOURCE_COMMITMENT_V5_SHA256,
                str(SUPERSEDED_REVIEW_GO_V5): SUPERSEDED_REVIEW_GO_V5_SHA256,
                str(SUPERSEDED_RELEASE_FAILURE_V5): SUPERSEDED_RELEASE_FAILURE_V5_SHA256,
                str(SUPERSEDED_SOURCE_COMMITMENT_V6): SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
                str(SUPERSEDED_REVIEW_GO_V6): SUPERSEDED_REVIEW_GO_V6_SHA256,
                str(SUPERSEDED_DECISION_HOLD_V6): SUPERSEDED_DECISION_HOLD_V6_SHA256,
            },
            "git_binding": "CURRENT_CLEAN_HEAD_EXACT_BLOB_AND_COMMIT_TRAILER",
            "runtime_sif_sha256": RUNTIME_SHA256,
            "state": "FROZEN_RECOVERY_DIAGNOSTIC_ONLY_UNRUN",
            "superseded_downstream_decision_hold_sha256": SUPERSEDED_DECISION_HOLD_V6_SHA256,
            "superseded_final_review_go_sha256": SUPERSEDED_REVIEW_GO_V6_SHA256,
            "superseded_source_commitment_sha256": SUPERSEDED_SOURCE_COMMITMENT_V6_SHA256,
        }
        source_raw = canonical(source_object) + b"\n"
        source_hash = sha256(source_raw)
        worker_review = {
            **synthetic_review,
            "source_commitment_sha256": source_hash,
        }
        worker_review_raw = canonical(worker_review) + b"\n"
        release_object = {
            "candidate_id": CANDIDATE_ID,
            "closed_capabilities": CLOSED,
            "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_execution_release_v7",
            "entity_uid": ENTITY_UID,
            "git_commit": synthetic_commit,
            "mode": "DIAGNOSTIC_ONLY_998_SUPPORTS",
            "runtime_sif_sha256": RUNTIME_SHA256,
            "source_commitment_sha256": source_hash,
            "source_review_sha256": sha256(worker_review_raw),
            "state": "EXTERNALLY_RELEASED_ONCE_RECOVERY_DIAGNOSTIC_ONLY",
        }
        release_raw = canonical(release_object) + b"\n"
        release_hash = sha256(release_raw)
        consumed_object = {
            "candidate_id": CANDIDATE_ID,
            "contract": "atypemu_ff15ipq_hydrogen_angle_diagnostic_recovery_consumption_v7",
            "execution_release_sha256": release_hash,
            "source_commitment_sha256": source_hash,
            "state": "CONSUMED_BEFORE_RECOVERY_DIAGNOSTIC_PDB_ACCESS",
        }
        exclusive(worker_stage / "scripts" / SCRIPT.name, script_raw)
        exclusive(
            worker_stage / "scripts" / SOURCE_COMMITMENT.name, source_raw
        )
        exclusive(worker_stage / "release/source_review.json", worker_review_raw)
        exclusive(worker_stage / "release/execution_release.json", release_raw)
        exclusive(
            worker_stage / "release/execution_consumed.json",
            canonical(consumed_object) + b"\n",
        )
        environment = {
            "ATYPEMU_DIAGNOSTIC_SOURCE_SHA256": source_hash,
            "ATYPEMU_DIAGNOSTIC_RELEASE_SHA256": release_hash,
            "ATYPEMU_RUNTIME_SIF_SHA256": RUNTIME_SHA256,
        }
        previous = {key: os.environ.get(key) for key in environment}
        try:
            os.environ.update(environment)
            require_worker_release(worker_stage)
            review_path = worker_stage / "release/source_review.json"
            os.chmod(review_path, 0o600)
            review_path.write_bytes(worker_review_raw + b"\n")
            try:
                require_worker_release(worker_stage)
            except (PermissionError, ValueError):
                pass
            else:
                raise AssertionError("tampered worker review was accepted")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    checks += 8
    return checks


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
    parser.add_argument("--container-controller", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-commitment-sha256")
    parser.add_argument("--execution-release-sha256")
    args = parser.parse_args()
    modes = [
        args.freeze_source,
        args.stage_source,
        args.run,
        args.container_controller,
        args.self_test,
    ]
    if sum(modes) != 1:
        parser.error("select exactly one mode")
    if args.container_controller:
        worker(_WORKER_SEAL)
        return 0
    root = repository_root()
    if args.self_test:
        print(
            f"STATUS PASS_FF15IPQ_ANGLE_DIAGNOSTIC_SELF_TEST checks={self_test(root)}"
        )
        return 0
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
