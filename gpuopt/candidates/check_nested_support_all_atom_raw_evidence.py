#!/usr/bin/env python3
"""Verify the sealed target-unread raw-PDB replay execution evidence."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_all_atom_raw_evidence_v2.tar.gz"
)
ARCHIVE_SHA256 = "e0df66a4046b9aef255f0d4992a732fdc2c7ebb4e1e5dccb7132a0ba511584f2"
CORRECTION_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_all_atom_raw_replay_correction_receipt.json"
)
RAW_CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_nested_support_all_atom_recount_raw.py"
)
RAW_CHECKER_SHA256 = "2df2bacf45bdf6fa64e5d521575f531006dbd1f65db573207274622a7e2ccd8c"
RAW_CHECKER_V1_SHA256 = "f9ae6413ef41dfe508c2e9b4bc386c93b0193f3bc099d104d061ff7035f75846"
CORRECT_COMMIT = "7021381b80f11521e0b6c701a142a6afdd15c783"
INVALID_COMMIT = "7021381c2c76aa630a0b7f800900e7d22b6bb850"
RAW_CHECKER_GIT_BLOB = "87b54968722ecfda8c0c49ba80d6b99c95ff31be"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SENTINELS = (
    "authorization_consumed",
    "outer_or_formal_metrics_opened",
    "science_executed",
    "source_scores_read",
    "target_values_read",
)
EXPECTED_LEVELS = {
    "32": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 32,
        "total_shortfall": 32,
    },
    "128": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 128,
        "total_shortfall": 128,
    },
    "768": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 134,
        "maximum_entity_shortfall": 768,
        "total_shortfall": 768,
    },
    "1536": {
        "all_entities_count_feasible": False,
        "entity_count_feasible": 0,
        "maximum_entity_shortfall": 1536,
        "total_shortfall": 73510,
    },
}
V1 = ".auto/staging/atypemu_nested_support_count_v1_all_atom_raw_launch_v1"
V2 = ".auto/staging/atypemu_nested_support_count_v1_all_atom_raw_launch_v2"
RAW = ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_raw_v2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(data: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=no_duplicates)
    assert isinstance(value, dict), f"{label}: expected JSON object"
    return value


def _expected_files() -> set[str]:
    files = {
        "gpuopt/candidates/check_nested_support_all_atom_recount_raw.py",
        "gpuopt/candidates/check_nested_support_all_atom_recount_raw."
        f"v1_failed_{RAW_CHECKER_V1_SHA256}.py",
        *(f"{V1}/{name}" for name in (
            "started_at_utc.txt", "started_epoch.txt", "ended_at_utc.txt",
            "ended_epoch.txt", "failure_receipt.json",
        )),
        *(f"{V2}/{name}" for name in (
            "started_at_utc.txt", "started_epoch.txt", "ended_at_utc.txt",
            "ended_epoch.txt", "shard_execution_receipt.json",
            "aggregate_started_at_utc.txt", "aggregate_started_epoch.txt",
            "aggregate_ended_at_utc.txt", "aggregate_ended_epoch.txt",
            "aggregate.pid", "aggregate.stdout", "aggregate.stderr",
            "aggregate.rc", "complete_execution_receipt.json",
        )),
        f"{RAW}/aggregate_receipt.json",
    }
    for shard in range(27):
        for suffix in ("rc", "stdout", "stderr"):
            files.add(f"{V1}/shard_{shard}.{suffix}")
            files.add(f"{V2}/shard_{shard}.{suffix}")
        files.add(f"{RAW}/o_excl_shard_receipts/shard_{shard}.json")
    return files


def _open_archive(archive: Path) -> dict[str, bytes]:
    assert archive.is_file() and not archive.is_symlink()
    assert _sha256(archive.read_bytes()) == ARCHIVE_SHA256
    files: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not member.issym() and not member.islnk()
            if not member.isfile():
                assert member.isdir()
                continue
            assert member.name not in files
            stream = handle.extractfile(member)
            assert isinstance(stream, io.BufferedReader)
            files[member.name] = stream.read()
    assert set(files) == _expected_files()
    return files


def _check_no_science(payload: dict[str, Any]) -> None:
    for key in SENTINELS:
        assert payload[key] is False, f"{key} must be false"


def _check_git_binding(root: Path, correction: dict[str, Any]) -> None:
    checker = correction["checker"]
    assert checker == {
        "git_blob": RAW_CHECKER_GIT_BLOB,
        "path": RAW_CHECKER_RELATIVE.as_posix(),
        "sha256": RAW_CHECKER_SHA256,
    }
    assert correction["correct_source_git_commit"] == CORRECT_COMMIT
    assert correction["invalid_source_git_commit_claim"] == INVALID_COMMIT
    assert correction["raw_replay_result_changed"] is False
    assert correction["status"] == (
        "CORRECTED_METADATA_RAW_REPLAY_PASS_ALL_LEVELS_REMAIN_HOLD"
    )
    assert correction["v2_evidence_archive"] == {
        "path": ARCHIVE_RELATIVE.as_posix(), "sha256": ARCHIVE_SHA256,
    }
    _check_no_science(correction)
    correct_bytes = subprocess.check_output(
        ["git", "show", f"{CORRECT_COMMIT}:{RAW_CHECKER_RELATIVE.as_posix()}"],
        cwd=root,
    )
    assert _sha256(correct_bytes) == RAW_CHECKER_SHA256
    blob = subprocess.check_output(
        ["git", "hash-object", "--stdin"], cwd=root, input=correct_bytes, text=False,
    ).decode().strip()
    assert blob == RAW_CHECKER_GIT_BLOB
    invalid = subprocess.run(
        ["git", "cat-file", "-e", f"{INVALID_COMMIT}^{{commit}}"], cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    assert invalid.returncode != 0


def _check_failed_v1(files: dict[str, bytes]) -> None:
    failure = _load_json(files[f"{V1}/failure_receipt.json"], "v1 failure")
    assert failure["contract"] == (
        "atypemu_nested_support_count_v1_all_atom_raw_launch_failure_v1"
    )
    assert failure["checker_sha256"] == RAW_CHECKER_V1_SHA256
    assert failure["failure_class"] == "PATH_EXPRESSION_PRE_PDB_REPLAY"
    assert failure["failed_shard_count"] == 27
    assert failure["shard_receipt_count"] == 0
    assert failure["catalog_read"] is True
    assert failure["identity_columns_read"] == [
        "entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "support_id",
    ]
    for key in ("mask_replay_started", "pdb_read_started", *SENTINELS):
        assert failure[key] is False
    assert len(failure["shards"]) == 27
    for shard, observed in enumerate(failure["shards"]):
        stdout = files[f"{V1}/shard_{shard}.stdout"]
        stderr = files[f"{V1}/shard_{shard}.stderr"]
        assert observed == {
            "rc": 1, "shard_index": shard,
            "stderr_sha256": _sha256(stderr), "stdout_sha256": _sha256(stdout),
        }
        assert stdout == b"" and b"unsupported operand type(s) for %" in stderr
        assert files[f"{V1}/shard_{shard}.rc"] == b"1\n"


def _check_v2(files: dict[str, bytes]) -> int:
    launch = _load_json(files[f"{V2}/shard_execution_receipt.json"], "v2 launch")
    assert launch["contract"] == (
        "atypemu_nested_support_count_v1_all_atom_raw_launch_receipt_v2"
    )
    assert launch["source_git_commit"] == INVALID_COMMIT
    assert launch["checker_sha256"] == RAW_CHECKER_SHA256
    assert launch["failed_v1_receipt_sha256"] == _sha256(
        files[f"{V1}/failure_receipt.json"]
    )
    assert launch["shard_count"] == launch["successful_shard_count"] == 27
    assert launch["raw_replay_entity_count"] == 135
    _check_no_science(launch)
    assert len(launch["shards"]) == 27
    total = 0
    for shard, observed in enumerate(launch["shards"]):
        prefix = f"{V2}/shard_{shard}"
        receipt_name = f"{RAW}/o_excl_shard_receipts/shard_{shard}.json"
        receipt_bytes = files[receipt_name]
        receipt = _load_json(receipt_bytes, f"raw shard {shard}")
        assert observed["shard_index"] == receipt["shard_index"] == shard
        assert observed["rc"] == 0
        assert observed["receipt_sha256"] == _sha256(receipt_bytes)
        assert observed["stdout_sha256"] == _sha256(files[f"{prefix}.stdout"])
        assert observed["stderr_sha256"] == EMPTY_SHA256
        assert files[f"{prefix}.stderr"] == b""
        assert files[f"{prefix}.rc"] == b"0\n"
        assert receipt["contract"] == (
            "atypemu_nested_support_count_v1_raw_replay_shard_receipt_v2"
        )
        assert receipt["checker_source_sha256"] == RAW_CHECKER_SHA256
        assert receipt["shard_count"] == 27
        assert receipt["raw_replay_entity_count"] == receipt["sealed_result_entity_count"]
        assert receipt["raw_replay_entities_sha256"] == (
            receipt["sealed_result_entities_sha256"]
        )
        assert receipt["exact_raw_replay_equals_sealed_result"] is True
        _check_no_science(receipt)
        total += receipt["raw_replay_entity_count"]
    assert total == 135
    return total


def _check_aggregate(files: dict[str, bytes]) -> None:
    aggregate_bytes = files[f"{RAW}/aggregate_receipt.json"]
    aggregate = _load_json(aggregate_bytes, "aggregate")
    assert aggregate["contract"] == (
        "atypemu_nested_support_count_v1_raw_replay_aggregate_receipt_v2"
    )
    assert aggregate["checker_source_sha256"] == RAW_CHECKER_SHA256
    assert aggregate["raw_replay_entity_count"] == aggregate["sealed_result_entity_count"] == 135
    assert aggregate["raw_replay_entities_sha256"] == aggregate["sealed_result_entities_sha256"]
    assert aggregate["catalog_support_count"] == 134850
    assert aggregate["exact_raw_replay_equals_sealed_result"] is True
    assert aggregate["level_count_feasibility"] == EXPECTED_LEVELS
    _check_no_science(aggregate)
    complete = _load_json(files[f"{V2}/complete_execution_receipt.json"], "complete")
    assert complete["contract"] == (
        "atypemu_nested_support_count_v1_all_atom_raw_complete_execution_receipt_v2"
    )
    assert complete["source_git_commit"] == INVALID_COMMIT
    assert complete["checker_sha256"] == RAW_CHECKER_SHA256
    assert complete["status"] == "PASS_EXACT_RAW_REPLAY_ALL_LEVELS_REMAIN_HOLD"
    assert complete["v2_aggregate_receipt"]["sha256"] == _sha256(aggregate_bytes)
    assert complete["v2_shard_execution_receipt"]["sha256"] == _sha256(
        files[f"{V2}/shard_execution_receipt.json"]
    )
    assert complete["failed_v1_receipt"]["sha256"] == _sha256(
        files[f"{V1}/failure_receipt.json"]
    )
    assert complete["raw_replay_entity_count"] == 135
    assert complete["raw_replay_catalog_support_count"] == 134850
    assert complete["level_count_feasibility"] == EXPECTED_LEVELS
    assert complete["exact_raw_replay_equals_sealed_result"] is True
    assert complete["aggregate_wall_seconds"] == 2128
    _check_no_science(complete)
    assert files[f"{V2}/aggregate.rc"] == b"0\n"
    assert files[f"{V2}/aggregate.stdout"] == b"RAW_REPLAY_AGGREGATE_OK 135\n"
    assert files[f"{V2}/aggregate.stderr"] == b""


def verify(root: Path) -> int:
    root = root.resolve()
    archive = root / ARCHIVE_RELATIVE
    files = _open_archive(archive)
    assert _sha256(files[RAW_CHECKER_RELATIVE.as_posix()]) == RAW_CHECKER_SHA256
    assert _sha256(files[
        "gpuopt/candidates/check_nested_support_all_atom_recount_raw."
        f"v1_failed_{RAW_CHECKER_V1_SHA256}.py"
    ]) == RAW_CHECKER_V1_SHA256
    correction = _load_json((root / CORRECTION_RELATIVE).read_bytes(), "correction")
    _check_git_binding(root, correction)
    _check_failed_v1(files)
    entity_count = _check_v2(files)
    _check_aggregate(files)
    return 134850 + entity_count + len(_expected_files())


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    checks = verify(root)
    print(f"RAW_EVIDENCE_OK {checks}")
    print(f"METRIC support_count_plan_checks={checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
