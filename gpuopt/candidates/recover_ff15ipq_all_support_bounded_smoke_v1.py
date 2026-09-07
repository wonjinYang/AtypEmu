#!/usr/bin/env python3
"""One-shot recovery for the consumed ff15ipq bounded-smoke mount failure."""

from __future__ import annotations

import argparse
import ctypes
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
RECOVERY_ID = f"{CANDIDATE_ID}_bounded_smoke_recovery_v1"
PARENT_SOURCE_SHA256 = (
    "351514037add35de3167fc69aa387e439e2cfeeb068b960a4f9cf61afdf0d540"
)
PARENT_RELEASE_SHA256 = (
    "331045ca2b360165a8d76c6dc451643f9a3ba1ce903687443946ed5fbb9a3702"
)
PARENT_CONSUMED_SHA256 = (
    "5d2b2cf454b4b8f8fbd6dfdbf85c4cf3c16dc5a75bcb55e747b30d00c551dfb0"
)
PARENT_FAILURE_SHA256 = (
    "1696e06548cd09fb0f5222e68b5b8a5281245701799ee1c262f017e05ef3d2bc"
)
RUNTIME_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
ENTITY_UID = "bmrb:10109:entity:1"
STAGE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_ff15ipq_all_support_stage_v1"
)
PARENT_SOURCE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_source_commitment_v1.json"
)
PARENT_RELEASE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_release_v1.json"
)
PARENT_CONSUMED = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_release_v1.consumed.json"
)
PARENT_FAILURE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_execution_failure_receipt_v1.json"
)
PARENT_REVIEW = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_source_commitment_cold_review_receipt_v1.json"
)
RECOVERY_REVIEW = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_bounded_smoke_recovery_review_v1.json"
)
RECOVERY_RELEASE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_bounded_smoke_recovery_release_v1.json"
)
RECOVERY_CONSUMED = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_bounded_smoke_recovery_release_v1.consumed.json"
)
RECOVERY_FAILURE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_bounded_smoke_recovery_failure_v1.json"
)
OUTPUT = Path(
    ".auto/atypemu_nested_support_count_v1_ff15ipq_all_support_recovery_output_v1"
)
SCRIPT = Path("gpuopt/candidates/recover_ff15ipq_all_support_bounded_smoke_v1.py")
PROJECTION = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1.json.gz"
)
SOURCE_REVIEW_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_"
    "source_commitment_cold_review_receipt_v1.json"
)
SOURCE_COMMITMENT_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1.json"
)
RUNTIME = STAGE / "runtime.sif"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CLOSED_CAPABILITIES = {
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
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise ValueError(f"noncanonical object in {label}")
    return value


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_regular(path: Path, maximum: int = 100_000_000) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"invalid regular file: {path}")
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


def exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        if os.write(descriptor, raw) != len(raw):
            raise OSError("short exclusive write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for no-clobber publication")
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
        if error == 17:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error))


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=HOST_ENV,
    )
    head = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("invalid Git HEAD")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=root, check=True, env=HOST_ENV
    )
    return head


def validate_parent(root: Path) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    expected = {
        PARENT_SOURCE: PARENT_SOURCE_SHA256,
        PARENT_RELEASE: PARENT_RELEASE_SHA256,
        PARENT_CONSUMED: PARENT_CONSUMED_SHA256,
        PARENT_FAILURE: PARENT_FAILURE_SHA256,
        RUNTIME: RUNTIME_SHA256,
    }
    raw_by_path = {path: read_regular(root / path) for path in expected}
    if any(sha256(raw_by_path[path]) != digest for path, digest in expected.items()):
        raise PermissionError("parent execution evidence drifted")
    source = parse_object(raw_by_path[PARENT_SOURCE], "parent source commitment")
    release = parse_object(raw_by_path[PARENT_RELEASE], "parent execution release")
    consumed = parse_object(raw_by_path[PARENT_CONSUMED], "parent consumption")
    failure = parse_object(raw_by_path[PARENT_FAILURE], "parent failure")
    if not (
        source.get("candidate_id") == CANDIDATE_ID
        and source.get("status") == "FROZEN_COMMITTED_BOUNDED_SMOKE_ONLY"
        and release.get("entities") == [ENTITY_UID]
        and release.get("source_commitment_sha256") == PARENT_SOURCE_SHA256
        and consumed.get("execution_release_sha256") == PARENT_RELEASE_SHA256
        and consumed.get("source_commitment_sha256") == PARENT_SOURCE_SHA256
        and consumed.get("state") == "CONSUMED_BEFORE_TARGET_UNREAD_PDB_ACCESS"
        and failure.get("execution_release_sha256") == PARENT_RELEASE_SHA256
        and failure.get("source_commitment_sha256") == PARENT_SOURCE_SHA256
        and failure.get("state") == "FAILED_AFTER_ONE_SHOT_RELEASE_CONSUMPTION"
    ):
        raise PermissionError("parent recovery identity drifted")
    files = source.get("files")
    if not isinstance(files, dict):
        raise ValueError("parent source manifest is invalid")
    for relative, expected_hash in files.items():
        staged = (
            root
            / STAGE
            / ("inputs" if relative.endswith(".json.gz") else "scripts")
            / Path(relative).name
        )
        if sha256(read_regular(staged, 20_000_000)) != expected_hash:
            raise PermissionError(f"staged parent source drifted: {relative}")
    review_raw = read_regular(root / PARENT_REVIEW, 1_000_000)
    if sha256(review_raw) != release.get("review_receipt_sha256"):
        raise PermissionError("parent source-review receipt drifted")
    return source, raw_by_path[PARENT_RELEASE], raw_by_path[PARENT_CONSUMED], review_raw


def validate_recovery_release(
    root: Path, expected_hash: str, git_commit: str
) -> tuple[dict[str, Any], bytes]:
    if SHA256_RE.fullmatch(expected_hash) is None:
        raise PermissionError("external recovery-release hash is invalid")
    raw = read_regular(root / RECOVERY_RELEASE, 1_000_000)
    if sha256(raw) != expected_hash:
        raise PermissionError("external recovery-release hash drifted")
    release = parse_object(raw, "recovery release")
    review_raw = read_regular(root / RECOVERY_REVIEW, 1_000_000)
    review = parse_object(review_raw, "recovery review")
    fields = {
        "bioemu_root",
        "candidate_id",
        "closed_capabilities",
        "contract",
        "entity_uid",
        "git_commit",
        "mount_fix",
        "parent_consumed_sha256",
        "parent_failure_sha256",
        "parent_release_sha256",
        "parent_source_sha256",
        "recovery_review_sha256",
        "recovery_script_sha256",
        "runtime_sif_sha256",
        "state",
    }
    script_hash = sha256(read_regular(root / SCRIPT, 2_000_000))
    if set(release) != fields or not (
        release["candidate_id"] == RECOVERY_ID
        and release["closed_capabilities"] == CLOSED_CAPABILITIES
        and release["contract"]
        == "atypemu_ff15ipq_all_support_bounded_smoke_recovery_release_v1"
        and release["entity_uid"] == ENTITY_UID
        and release["git_commit"] == git_commit
        and release["mount_fix"]
        == "bind scripts and projection separately so writable nested outputs are not created beneath a read-only /work bind"
        and release["parent_consumed_sha256"] == PARENT_CONSUMED_SHA256
        and release["parent_failure_sha256"] == PARENT_FAILURE_SHA256
        and release["parent_release_sha256"] == PARENT_RELEASE_SHA256
        and release["parent_source_sha256"] == PARENT_SOURCE_SHA256
        and release["recovery_review_sha256"] == sha256(review_raw)
        and release["recovery_script_sha256"] == script_hash
        and release["runtime_sif_sha256"] == RUNTIME_SHA256
        and release["state"] == "EXTERNALLY_RELEASED_ONCE_FOR_MOUNT_ONLY_RECOVERY"
    ):
        raise PermissionError("recovery release identity drifted")
    if set(review) != {
        "candidate_id",
        "closed_capabilities",
        "contract",
        "git_commit",
        "parent_failure_sha256",
        "parent_source_sha256",
        "recovery_script_sha256",
        "reviews",
        "state",
    } or not (
        review["candidate_id"] == RECOVERY_ID
        and review["closed_capabilities"] == CLOSED_CAPABILITIES
        and review["contract"]
        == "atypemu_ff15ipq_all_support_bounded_smoke_recovery_review_v1"
        and review["git_commit"] == git_commit
        and review["parent_failure_sha256"] == PARENT_FAILURE_SHA256
        and review["parent_source_sha256"] == PARENT_SOURCE_SHA256
        and review["recovery_script_sha256"] == script_hash
        and review["state"] == "GO_MOUNT_ONLY_RECOVERY"
        and isinstance(review["reviews"], list)
        and len(review["reviews"]) == 3
        and {row.get("scope") for row in review["reviews"]}
        == {"operational", "scientific_leakage", "security_provenance"}
        and all(
            isinstance(row, dict)
            and set(row) == {"review_session", "scope", "verdict"}
            and isinstance(row["review_session"], str)
            and row["review_session"]
            and row["verdict"] == "GO"
            for row in review["reviews"]
        )
    ):
        raise PermissionError("recovery review receipt drifted")
    return release, raw


def command(
    root: Path,
    output: Path,
    bioemu: Path,
    script: str,
) -> list[str]:
    stage = root / STAGE
    released = output / "release"
    checker = script.startswith("check_")
    result = [
        "/usr/bin/singularity",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{stage / 'scripts'}:/work/scripts:ro",
        "--bind",
        f"{stage / 'inputs' / PROJECTION}:/work/inputs/{PROJECTION}:ro",
        "--bind",
        f"{output / 'entity_archives'}:/work/outputs:{'ro' if checker else 'rw'}",
    ]
    if checker:
        result.extend(["--bind", f"{output / 'checker_receipts'}:/work/checks:rw"])
    result.extend(
        [
            "--bind",
            f"{bioemu}:/work/inputs/data/BioEmu:ro",
            "--bind",
            f"{released}:/work/release:ro",
            "--env",
            f"ATYPEMU_STAGE_ROOT=/work,ATYPEMU_SOURCE_COMMITMENT_SHA256={PARENT_SOURCE_SHA256},"
            f"ATYPEMU_RUNTIME_SIF_SHA256={RUNTIME_SHA256},"
            f"ATYPEMU_GIT_COMMIT=ed3df1021c2cbd5c672a24e286bf15e8ace3c2ad,"
            f"ATYPEMU_EXECUTION_RELEASE_SHA256={PARENT_RELEASE_SHA256},"
            f"ATYPEMU_RELEASED_ENTITY_UID={ENTITY_UID}",
            str(stage / "runtime.sif"),
            "python",
            f"/work/scripts/{script}",
            "--entity-uid",
            ENTITY_UID,
        ]
    )
    return result


def execute(command_line: list[str]) -> None:
    process = subprocess.Popen(
        command_line,
        cwd="/tmp",
        env=HOST_ENV,
        start_new_session=True,
    )
    try:
        result = process.wait(timeout=21_600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise TimeoutError("bounded recovery worker exceeded six hours") from None
    if result != 0:
        raise subprocess.CalledProcessError(result, command_line)


def run(root: Path, expected_release_hash: str) -> None:
    executing = Path(__file__).absolute()
    canonical_script = (root / SCRIPT).absolute()
    if (
        executing != canonical_script
        or executing.is_symlink()
        or executing.resolve(strict=True) != executing
    ):
        raise PermissionError("recovery must execute the canonical direct Git script")
    git_commit = git_head(root)
    _, parent_release, parent_consumed, parent_review = validate_parent(root)
    release, release_raw = validate_recovery_release(
        root, expected_release_hash, git_commit
    )
    if (
        release["bioemu_root"]
        != parse_object(parent_release, "parent execution release")["bioemu_root"]
    ):
        raise PermissionError("recovery BioEmu root differs from the parent release")
    bioemu = Path(release["bioemu_root"])
    if not bioemu.is_absolute() or bioemu.is_symlink() or not bioemu.is_dir():
        raise PermissionError("recovery BioEmu root is absent or indirect")
    bioemu = bioemu.resolve(strict=True)
    if str(bioemu) != release["bioemu_root"]:
        raise PermissionError("recovery BioEmu root is noncanonical")
    for path in (root / RECOVERY_CONSUMED, root / RECOVERY_FAILURE, root / OUTPUT):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"recovery terminal path already exists: {path}")
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    (temporary / "entity_archives").mkdir(mode=0o700)
    (temporary / "checker_receipts").mkdir(mode=0o700)
    (temporary / "release").mkdir(mode=0o700)
    exclusive(temporary / "release/execution_release.json", parent_release)
    exclusive(temporary / "release/execution_consumed.json", parent_consumed)
    exclusive(temporary / "release/source_cold_review.json", parent_review)
    consumed = {
        "candidate_id": RECOVERY_ID,
        "contract": "atypemu_ff15ipq_all_support_bounded_smoke_recovery_consumption_v1",
        "parent_consumed_sha256": PARENT_CONSUMED_SHA256,
        "parent_failure_sha256": PARENT_FAILURE_SHA256,
        "parent_release_sha256": PARENT_RELEASE_SHA256,
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "recovery_release_sha256": expected_release_hash,
        "state": "CONSUMED_BEFORE_RECOVERY_PDB_ACCESS",
    }
    consumed_raw = canonical(consumed) + b"\n"
    consumed_written = False
    try:
        exclusive(root / RECOVERY_CONSUMED, consumed_raw)
        consumed_written = True
        exclusive(temporary / "release/recovery_consumed.json", consumed_raw)
        exclusive(temporary / "release/recovery_release.json", release_raw)
        publish_directory(temporary, output)
        execute(
            command(
                root,
                output,
                bioemu,
                "materialize_ff15ipq_all_support_entity.py",
            )
        )
        execute(
            command(
                root,
                output,
                bioemu,
                "check_ff15ipq_all_support_entity.py",
            )
        )
        result = {
            "candidate_id": RECOVERY_ID,
            "closed_capabilities": CLOSED_CAPABILITIES,
            "entity_uid": ENTITY_UID,
            "parent_failure_sha256": PARENT_FAILURE_SHA256,
            "parent_source_sha256": PARENT_SOURCE_SHA256,
            "recovery_release_sha256": expected_release_hash,
            "state": "PASS_MOUNT_ONLY_RECOVERY_BOUNDED_SMOKE",
        }
        exclusive(output / "recovery_result.json", canonical(result) + b"\n")
    except BaseException as error:
        if consumed_written:
            failure = {
                "candidate_id": RECOVERY_ID,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "parent_failure_sha256": PARENT_FAILURE_SHA256,
                "parent_source_sha256": PARENT_SOURCE_SHA256,
                "recovery_release_sha256": expected_release_hash,
                "state": "FAILED_AFTER_RECOVERY_CONSUMPTION",
            }
            try:
                exclusive(root / RECOVERY_FAILURE, canonical(failure) + b"\n")
            finally:
                raise
        raise


def self_test() -> int:
    for raw in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b"[]\n"):
        try:
            parse_object(raw, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON accepted")
    if OUTPUT == Path(
        ".auto/atypemu_nested_support_count_v1_ff15ipq_all_support_output_v1"
    ):
        raise AssertionError("recovery output aliases parent output")
    synthetic = command(Path("/repo"), Path("/output"), Path("/bioemu"), "producer.py")
    if (
        "/repo/.auto/staging/atypemu_nested_support_count_v1_ff15ipq_all_support_stage_v1:/work:ro"
        in synthetic
    ):
        raise AssertionError("failed read-only parent mount survived recovery")
    return 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--recovery-release-sha256")
    args = parser.parse_args()
    if args.self_test == args.run:
        parser.error("select exactly one of --self-test or --run")
    if args.self_test:
        print(f"STATUS PASS_FF15IPQ_RECOVERY_SELF_TEST checks={self_test()}")
        return 0
    if args.recovery_release_sha256 is None:
        parser.error("--recovery-release-sha256 is required")
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            env=HOST_ENV,
        ).stdout.strip()
    ).resolve(strict=True)
    run(root, args.recovery_release_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
