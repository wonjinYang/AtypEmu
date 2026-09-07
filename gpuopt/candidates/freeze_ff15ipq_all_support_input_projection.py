#!/usr/bin/env python3
"""Freeze the minimal target-unread ff15ipq all-support runtime projection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
SCRIPT_RELATIVE = Path(
    "gpuopt/candidates/freeze_ff15ipq_all_support_input_projection.py"
)
OUTPUT_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_"
    "ff15ipq_all_support_input_projection_v1.json.gz"
)
ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/"
    "catalog_v3_shards.tar.gz"
)
CONDITION_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json"
)
ARCHIVE_SHA256 = "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2"
CONDITION_SHA256 = "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618"
MIDPOINTS = ["2.2", "5.45", "7.5", "9.25", "12.0"]
SHA = re.compile(r"[0-9a-f]{64}\Z")
UID = re.compile(r"bmrb:([0-9]+):entity:1\Z")
CAPABILITIES = {
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


def reject_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(
        raw, object_pairs_hook=reject_dupes, parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def root_file(root: Path, relative: Path) -> bytes:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"invalid canonical input: {relative}")
    directory_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    file_fd: int | None = None
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {relative}")
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(file_fd)
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
            raise ValueError(f"file changed during read: {relative}")
        return b"".join(chunks)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def catalog_entities(raw: bytes) -> list[dict[str, Any]]:
    shards: dict[int, list[dict[str, Any]]] = {}
    expected_files = {
        "./SHA256SUMS",
        "./catalog_summary.json",
        *(f"./shard_{index}.json" for index in range(27)),
    }
    seen: set[str] = set()
    root_seen = False
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.name == "." and member.isdir():
                if root_seen:
                    raise ValueError("duplicate catalog archive root")
                root_seen = True
                continue
            if (
                member.name not in expected_files
                or member.name in seen
                or not member.isfile()
            ):
                raise ValueError("catalog archive member contract failed")
            seen.add(member.name)
            match = re.fullmatch(r"\./shard_([0-9]+)\.json", member.name)
            if match is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("catalog shard cannot be read")
            shard = json_object(handle.read(), member.name)
            index = int(match.group(1))
            rows = shard.get("entities")
            if (
                index in shards
                or shard.get("shard_index") != index
                or shard.get("shard_count") != 27
                or not isinstance(rows, list)
                or shard.get("entity_count") != len(rows)
                or any(not isinstance(row, dict) for row in rows)
            ):
                raise ValueError("catalog shard schema failed")
            shards[index] = rows
    if not root_seen or seen != expected_files or set(shards) != set(range(27)):
        raise ValueError("catalog archive is incomplete")
    return [row for index in range(27) for row in shards[index]]


def checked_missing_indices(value: Any, present: list[int]) -> list[int]:
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        raise TypeError("missing-support indexes are not exact integers")
    expected = [index for index in range(1, 1001) if index not in set(present)]
    if value != expected:
        raise ValueError("catalog missing-index complement mismatch")
    return expected


def build_projection(root: Path) -> bytes:
    archive_raw = root_file(root, ARCHIVE_RELATIVE)
    condition_raw = root_file(root, CONDITION_RELATIVE)
    if (
        digest(archive_raw) != ARCHIVE_SHA256
        or digest(condition_raw) != CONDITION_SHA256
    ):
        raise ValueError("upstream byte hash mismatch")
    conditions = json_object(condition_raw, "condition manifest").get("entities")
    if not isinstance(conditions, list) or len(conditions) != 135:
        raise ValueError("condition roster is invalid")
    condition_by_uid: dict[str, dict[str, Any]] = {}
    for row in conditions:
        if not isinstance(row, dict):
            raise TypeError("condition row is not an object")
        uid = row.get("entity_uid")
        if not isinstance(uid, str) or uid in condition_by_uid:
            raise ValueError("condition entity identity is invalid")
        condition_by_uid[uid] = row

    entries: list[dict[str, Any]] = []
    supports = candidate_states = 0
    for row in catalog_entities(archive_raw):
        uid, bmrb_id, files = (
            row.get("entity_uid"),
            row.get("bmrb_id"),
            row.get("files"),
        )
        if (
            not isinstance(uid, str)
            or not isinstance(bmrb_id, str)
            or not isinstance(files, list)
            or uid not in condition_by_uid
        ):
            raise ValueError("catalog projection identity is invalid")
        match = UID.fullmatch(uid)
        if match is None or bmrb_id != f"bmr{match.group(1)}":
            raise ValueError("catalog BMRB identity is invalid")
        condition = condition_by_uid[uid]
        if condition.get("bmrb_id") != bmrb_id:
            raise ValueError("catalog/condition identity mismatch")
        state = condition.get("condition_state")
        if state == "observed":
            ph = condition.get("deposited_ph")
            if not isinstance(ph, str):
                raise ValueError("observed pH is absent")
            branches = [{"condition_branch_id": "observed-0", "proposal_pH": ph}]
        elif state in {"state_missing", "state_ambiguous"}:
            if condition.get("deposited_ph") is not None:
                raise ValueError("unresolved pH was imputed")
            branches = [
                {"condition_branch_id": f"regime-{index}", "proposal_pH": ph}
                for index, ph in enumerate(MIDPOINTS)
            ]
        else:
            raise ValueError("unknown condition state")
        projected_supports: list[dict[str, Any]] = []
        indexes: list[int] = []
        for item in files:
            if not isinstance(item, dict):
                raise TypeError("catalog support is not an object")
            index, pdb_sha = item.get("support_index"), item.get("pdb_sha256")
            if type(index) is not int or not 1 <= index <= 1000:
                raise ValueError("support index is invalid")
            if not isinstance(pdb_sha, str) or SHA.fullmatch(pdb_sha) is None:
                raise ValueError("support PDB hash is invalid")
            indexes.append(index)
            projected_supports.append(
                {"parent_raw_pdb_sha256": pdb_sha, "support_index": index}
            )
        if indexes != sorted(set(indexes)):
            raise ValueError("support indexes are not unique and ordered")
        missing = checked_missing_indices(row.get("missing_indices_1_to_1000"), indexes)
        entries.append(
            {
                "bmrb_id": bmrb_id,
                "condition_branches": branches,
                "condition_state": state,
                "entity_uid": uid,
                "missing_support_indices": missing,
                "supports": projected_supports,
            }
        )
        supports += len(projected_supports)
        candidate_states += len(projected_supports) * len(branches)
    entries.sort(key=lambda item: item["entity_uid"])
    if (
        len(entries) != 135
        or len({item["entity_uid"] for item in entries}) != 135
        or supports != 134850
        or candidate_states != 198758
        or set(condition_by_uid) != {item["entity_uid"] for item in entries}
    ):
        raise ValueError("projection cardinality mismatch")
    payload = {
        "artifact_kind": "target_unread_minimal_all_support_runtime_input_projection",
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CAPABILITIES,
        "contract": "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1",
        "counts": {
            "entity_count": 135,
            "qualification_candidate_state_count": candidate_states,
            "support_count": supports,
        },
        "entries": entries,
        "producer": {
            "path": SCRIPT_RELATIVE.as_posix(),
            "sha256": digest(root_file(root, SCRIPT_RELATIVE)),
        },
        "runtime_allowed_fields": [
            "bmrb_id",
            "condition_branch_id",
            "condition_state",
            "entity_uid",
            "missing_support_indices",
            "parent_raw_pdb_sha256",
            "proposal_pH",
            "support_index",
        ],
        "source_path_template": (
            "data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb"
        ),
        "state": "HOLD_RUNTIME_INPUT_ONLY_IMPLEMENTATION_UNRUN",
        "upstream": {
            "catalog_shard_archive_sha256": ARCHIVE_SHA256,
            "condition_manifest_sha256": CONDITION_SHA256,
        },
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    actual = Path(__file__).absolute()
    root = actual.parents[2]
    if actual != root / SCRIPT_RELATIVE or actual.is_symlink():
        raise ValueError("producer must run at its canonical repository path")
    output = root / OUTPUT_RELATIVE
    proposed = build_projection(root)
    if args.verify:
        current = root_file(root, OUTPUT_RELATIVE)
        if current != proposed:
            raise ValueError("runtime input projection replay mismatch")
        print(
            "STATUS PASS_FF15IPQ_ALL_SUPPORT_INPUT_PROJECTION_REPLAY "
            f"sha256={digest(current)}"
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(proposed)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(
        "STATUS WROTE_FF15IPQ_ALL_SUPPORT_INPUT_PROJECTION "
        f"bytes={len(proposed)} sha256={digest(proposed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
