#!/usr/bin/env python3
"""Replay the immutable, target-unread recovery-v6 pH evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Dict, List, Tuple
import zipfile


ARCHIVE_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_evidence_v1.zip"
)
RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_evidence_receipt.json"
)
ARCHIVE_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"
ARCHIVE_PREFIX = (
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_evidence_v1/"
)
COMMIT = "ad13d0276533c0d8acc90703f1c44c6c5fab6ae2"
INTENT_SHA256 = "41152c6f559e7711cffa9d033324ed89bcb325270ae3f6fb4a47bf2bc4f316e9"
EXECUTION_SHA256 = "b817fef58b3bc1a64ae546c993cc550752eecc132844c498a43174c2eff943c3"
OUTPUT_SHA256 = "4050e642509cb90c5120eccc459ebaf7299b237fee487365d45a6664ee7ec213"
STARTED_SHA256 = "6d382b6401c822e79828a6b9212ca3336de214c9ce31da56b86a9690d7d753ff"
MAX_ARCHIVE_BYTES = 2_000_000
MAX_MEMBER_BYTES = 1_000_000
MAX_TOTAL_BYTES = 5_000_000
MAX_GIT_BYTES = 2_000_000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[1-9][0-9]*\Z")
ASSIGNED_KEYS = (
    "_Assigned_chem_shift_list.Entry_ID",
    "_Assigned_chem_shift_list.ID",
    "_Assigned_chem_shift_list.Sample_condition_list_ID",
)
CLOSED = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
OUTPUT_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "api_v6_recovery"
)
CONSUMED_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_launch_intent.consumed.json"
)
LOCK_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_execution_lock.json"
)
EXECUTION_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "recovery_v6_execution_receipt.json"
)
SOURCE_HASHES = {
    "checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v6.py",
        "sha256": "e841d7c81fb823b1939dd0b74e5ae0f610dd387fd17c4456d286fc9a2aaa99c5",
    },
    "handler": {
        "path": "gpuopt/candidates/run_openmm86_deposited_ph_recovery_v6_if_intent.py",
        "sha256": "c077226516b3b0aa97029d9ac453b57134e7fc04ece69805305460d09caf4034",
    },
    "parent_plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json",
        "sha256": "7345d94120445f75a59a3e5251ff76bb4d7fb2fe8a172039de43a18e6170d67c",
    },
    "parent_validator": {
        "path": "gpuopt/candidates/nested_support_count_plan.py",
        "sha256": "21ff8c836f6604cb49c474533455acd8e37bb5b8ddfe252aede255608fc018e8",
    },
    "plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6_plan.json",
        "sha256": "c12873f74e56db94d0a44d14b1edb339331d8cf4870c1361bcb353bfc538a81f",
    },
    "producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v6.py",
        "sha256": "6a7fff25f6cfb520511b269ca0c06e80b092eed35107b7e8cd740c8122e0d623",
    },
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON in %s: %s" % (label, error)) from error


def _read_fixed(root: Path, relative: Path, maximum: int) -> bytes:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe fixed path")
    current = root
    for part in relative.parts:
        current = current / part
        details = os.lstat(current)
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("fixed evidence path is indirect")
    details = os.lstat(current)
    if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
        raise ValueError("fixed evidence is not a bounded regular file")
    descriptor = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(opened.st_size + 1)
        if len(raw) != opened.st_size:
            raise ValueError("fixed evidence changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _expected_receipt() -> Dict[str, Any]:
    return {
        "archive": {
            "path": ARCHIVE_RELATIVE.as_posix(),
            "sha256": ARCHIVE_SHA256,
            "source_member_count": 140,
            "zip_member_count": 141,
        },
        "artifact_kind": "hold_only_target_unread_openmm86_deposited_ph_recovery_v6_evidence_receipt",
        "candidate_id": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6",
        "closed_capabilities": CLOSED,
        "contract": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1",
        "execution": {
            "consumed_intent_sha256": INTENT_SHA256,
            "execution_lock_sha256": INTENT_SHA256,
            "execution_receipt_sha256": EXECUTION_SHA256,
            "git_commit": COMMIT,
            "outcome": "local_hold_verified",
        },
        "independent_semantic_checker": {
            "path": "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py",
            "raw_derived_record_count": 135,
            "sha256": "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84",
            "verification_check_count": 810,
        },
        "result": {
            "all_entities_ph_feasible": False,
            "entity_count": 135,
            "hold_entity_count": 20,
            "output_receipt_sha256": OUTPUT_SHA256,
            "ph_feasible_entity_count": 115,
            "started_at_utc_sha256": STARTED_SHA256,
            "status": "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS",
        },
        "scope": "artifact-integrity and target-unread metadata HOLD evidence only; not all-atom support feasibility, source construction, science authorization, or score evidence",
    }


def _git_blob(root: Path, relative: str) -> bytes:
    process = subprocess.run(
        ["git", "show", "%s:%s" % (COMMIT, relative)],
        cwd=str(root),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if process.returncode != 0 or len(process.stdout) > MAX_GIT_BYTES:
        raise ValueError("cannot read bounded committed source: %s" % relative)
    return process.stdout


def _member_name(relative: Path) -> str:
    return ARCHIVE_PREFIX + "artifacts/" + relative.as_posix()


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name.startswith(ARCHIVE_PREFIX)
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("unsafe archive member name")


def _read_zip(info: zipfile.ZipInfo, archive: zipfile.ZipFile) -> bytes:
    if info.is_dir() or info.file_size > MAX_MEMBER_BYTES:
        raise ValueError("archive member is not a bounded file")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError("archive member uses an unapproved compression type")
    raw = archive.read(info)
    if len(raw) != info.file_size:
        raise ValueError("archive member size drifted")
    return raw


def _assigned_response(raw: bytes, bmrb_id: str) -> None:
    entry_id = bmrb_id[3:]
    value = _json(raw, bmrb_id + " assigned-list response")
    if not isinstance(value, dict) or set(value) != {entry_id}:
        raise ValueError("assigned-list entry envelope drifted")
    columns = value[entry_id]
    if not isinstance(columns, dict) or set(columns) != set(ASSIGNED_KEYS):
        raise ValueError("assigned-list response is not exactly metadata-only")
    values = [columns[key] for key in ASSIGNED_KEYS]
    if (
        not all(isinstance(column, list) for column in values)
        or len({len(column) for column in values}) != 1
        or not all(type(item) is str for column in values for item in column)
        or any(item != entry_id for item in values[0])
    ):
        raise ValueError("assigned-list metadata columns are malformed")


def _verify_archive(raw: bytes, root: Path) -> int:
    if _sha256(raw) != ARCHIVE_SHA256:
        raise ValueError("recovery-v6 evidence archive SHA-256 drifted")
    checks = 1
    import io

    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(infos) != 141 or len(set(names)) != 141:
            raise ValueError("archive inventory count or uniqueness drifted")
        for name in names:
            _safe_member(name)
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise ValueError("archive uncompressed total exceeds limit")
        by_name = dict(zip(names, infos))
        manifest_name = ARCHIVE_PREFIX + "archive_manifest.json"
        manifest = _json(
            _read_zip(by_name[manifest_name], archive), "archive manifest"
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            "artifact_kind", "authorization_consumed", "candidate_id", "contract",
            "member_count", "members", "outer_or_formal_metrics_opened",
            "science_executed", "source_scores_read", "target_atom_identities_read",
            "target_values_read",
        }:
            raise ValueError("archive manifest schema drifted")
        if (
            manifest["artifact_kind"]
            != "target_unread_openmm86_deposited_ph_recovery_v6_archive_manifest"
            or manifest["candidate_id"]
            != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
            or manifest["contract"]
            != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_archive_v1"
            or manifest["member_count"] != 140
            or any(manifest[name] is not False for name in CLOSED if name in manifest)
            or not isinstance(manifest["members"], list)
            or len(manifest["members"]) != 140
        ):
            raise ValueError("archive manifest semantics drifted")
        member_records = manifest["members"]
        paths: List[str] = []
        contents: Dict[str, bytes] = {}
        for item in member_records:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size"}
                or not isinstance(item["path"], str)
                or not isinstance(item["sha256"], str)
                or SHA256_RE.fullmatch(item["sha256"]) is None
                or type(item["size"]) is not int
                or item["size"] < 0
            ):
                raise ValueError("archive manifest member record drifted")
            name = ARCHIVE_PREFIX + "artifacts/" + item["path"]
            _safe_member(name)
            if name not in by_name:
                raise ValueError("manifest-bound archive member is absent")
            body = _read_zip(by_name[name], archive)
            if len(body) != item["size"] or _sha256(body) != item["sha256"]:
                raise ValueError("manifest-bound archive member bytes drifted")
            paths.append(item["path"])
            contents[item["path"]] = body
            checks += 1
        if paths != sorted(paths) or len(set(paths)) != 140:
            raise ValueError("manifest member paths are not sorted and unique")
        if set(names) != {manifest_name} | {
            ARCHIVE_PREFIX + "artifacts/" + path for path in paths
        }:
            raise ValueError("archive has unmanifested members")

    consumed = contents[CONSUMED_RELATIVE.as_posix()]
    lock = contents[LOCK_RELATIVE.as_posix()]
    execution_raw = contents[EXECUTION_RELATIVE.as_posix()]
    output_path = (OUTPUT_RELATIVE / "receipt.json").as_posix()
    started_path = (OUTPUT_RELATIVE / "started_at_utc.txt").as_posix()
    if (
        consumed != lock
        or _sha256(consumed) != INTENT_SHA256
        or _sha256(execution_raw) != EXECUTION_SHA256
        or _sha256(contents[output_path]) != OUTPUT_SHA256
        or _sha256(contents[started_path]) != STARTED_SHA256
    ):
        raise ValueError("primary archived evidence binding drifted")
    intent = _json(consumed, "consumed launch intent")
    execution = _json(execution_raw, "execution receipt")
    output = _json(contents[output_path], "output receipt")
    if (
        not isinstance(intent, dict)
        or intent.get("git_commit") != COMMIT
        or intent.get("candidate_id")
        != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
        or intent.get("contract")
        != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_launch_v1"
        or any(intent.get(name) is not False for name in CLOSED)
        or intent.get("checker") != SOURCE_HASHES["checker"]
        or intent.get("plan") != SOURCE_HASHES["plan"]
        or intent.get("producer") != SOURCE_HASHES["producer"]
    ):
        raise ValueError("consumed launch intent semantics drifted")
    if (
        not isinstance(execution, dict)
        or execution.get("git_commit") != COMMIT
        or execution.get("contract")
        != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_execution_v1"
        or execution.get("closed_controls") != CLOSED
        or execution.get("outcome") != "local_hold_verified"
        or execution.get("status")
        != "HOLD_LOCAL_ONLY_PENDING_SEPARATELY_EXACT_BOUND_IMMUTABLE_ARCHIVE_AND_EXECUTION_RECEIPT"
        or execution.get("error") is not None
        or execution.get("source_hashes") != SOURCE_HASHES
        or execution.get("consumed_intent", {}).get("sha256") != INTENT_SHA256
        or execution.get("execution_lock", {}).get("sha256") != INTENT_SHA256
        or execution.get("output_binding", {}).get("sha256") != OUTPUT_SHA256
        or execution.get("subprocesses", {}).get("producer", {}).get("returncode") != 4
        or execution.get("subprocesses", {}).get("checker", {}).get("returncode") != 4
    ):
        raise ValueError("execution receipt semantics drifted")
    for name, binding in SOURCE_HASHES.items():
        if _sha256(_git_blob(root, binding["path"])) != binding["sha256"]:
            raise ValueError("historical execution source drifted: %s" % name)
        checks += 1

    if (
        not isinstance(output, dict)
        or output.get("candidate_id")
        != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
        or output.get("contract")
        != "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6"
        or output.get("status")
        != "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
        or output.get("entity_count") != 135
        or output.get("ph_feasible_entity_count") != 115
        or output.get("all_entities_ph_feasible") is not False
        or any(output.get(name) is not False for name in CLOSED)
        or output.get("source_producer_git_commit") != COMMIT
        or output.get("source_producer_sha256") != SOURCE_HASHES["producer"]["sha256"]
        or not isinstance(output.get("entities"), list)
        or len(output["entities"]) != 135
        or not isinstance(output.get("new_response_manifest"), list)
        or len(output["new_response_manifest"]) != 135
    ):
        raise ValueError("output receipt semantics drifted")
    manifest_bindings = {
        item.get("bmrb_id"): item for item in output["new_response_manifest"]
        if isinstance(item, dict)
    }
    if len(manifest_bindings) != 135:
        raise ValueError("output response manifest identities drifted")
    identities = set()
    bmrb_ids = set()
    feasible = 0
    for entity in output["entities"]:
        if not isinstance(entity, dict):
            raise ValueError("output entity record is malformed")
        bmrb_id = entity.get("bmrb_id")
        entity_uid = entity.get("entity_uid")
        binding = entity.get("new_response_binding")
        if (
            not isinstance(bmrb_id, str)
            or not bmrb_id.startswith("bmr")
            or ID_RE.fullmatch(bmrb_id[3:]) is None
        ):
            raise ValueError("output BMRB identity drifted")
        expected_url = (
            "https://api.bmrb.io/v2/entry/%s?"
            "tag=_Assigned_chem_shift_list.Entry_ID&"
            "tag=_Assigned_chem_shift_list.ID&"
            "tag=_Assigned_chem_shift_list.Sample_condition_list_ID"
        ) % bmrb_id[3:]
        expected_relative = "raw_api_responses/%s.Assigned_chem_shift_list.json" % bmrb_id
        if (
            not isinstance(entity_uid, str)
            or entity_uid in identities
            or bmrb_id in bmrb_ids
            or not isinstance(binding, dict)
            or binding != manifest_bindings.get(bmrb_id)
            or set(binding) != {"bmrb_id", "final_url", "loop", "path", "sha256", "url"}
            or binding.get("loop") != "Assigned_chem_shift_list"
            or binding.get("path") != expected_relative
            or binding.get("url") != expected_url
            or binding.get("final_url") != expected_url
            or not isinstance(binding.get("sha256"), str)
            or SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("output entity identity or response binding drifted")
        identities.add(entity_uid)
        bmrb_ids.add(bmrb_id)
        relative = OUTPUT_RELATIVE / binding["path"]
        raw_response = contents.get(relative.as_posix())
        if raw_response is None or _sha256(raw_response) != binding.get("sha256"):
            raise ValueError("archived assigned-list response binding drifted")
        _assigned_response(raw_response, bmrb_id)
        if entity.get("ph_feasible") is True:
            feasible += 1
            if entity.get("hold_reasons") != []:
                raise ValueError("feasible entity retains HOLD reasons")
        elif entity.get("ph_feasible") is False:
            if not isinstance(entity.get("hold_reasons"), list) or not entity["hold_reasons"]:
                raise ValueError("held entity lacks reasons")
        else:
            raise ValueError("entity pH feasibility is not Boolean")
        checks += 2
    if (
        len(identities) != 135
        or bmrb_ids != set(manifest_bindings)
        or feasible != 115
    ):
        raise ValueError("output entity arithmetic drifted")
    expected_paths = {
        CONSUMED_RELATIVE.as_posix(), LOCK_RELATIVE.as_posix(),
        EXECUTION_RELATIVE.as_posix(), output_path, started_path,
    } | {
        (OUTPUT_RELATIVE / item["path"]).as_posix()
        for item in output["new_response_manifest"]
    }
    if set(contents) != expected_paths:
        raise ValueError("archive source-member inventory drifted")
    return checks + 5


def verify(root: Path) -> int:
    receipt_raw = _read_fixed(root, RECEIPT_RELATIVE, MAX_MEMBER_BYTES)
    expected_raw = (
        json.dumps(_expected_receipt(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if receipt_raw != expected_raw:
        raise ValueError("committed evidence receipt drifted")
    archive_raw = _read_fixed(root, ARCHIVE_RELATIVE, MAX_ARCHIVE_BYTES)
    return _verify_archive(archive_raw, root) + 1


def self_test(root: Path) -> int:
    checks = verify(root)
    raw = _read_fixed(root, ARCHIVE_RELATIVE, MAX_ARCHIVE_BYTES)
    tampered = bytearray(raw)
    tampered[len(tampered) // 2] ^= 1
    try:
        _verify_archive(bytes(tampered), root)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("tampered archive was accepted")
    try:
        _json(b'{"x":1,"x":2}', "duplicate-key smoke")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("duplicate JSON key was accepted")
    try:
        _safe_member(ARCHIVE_PREFIX + "../escape")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("traversal archive member was accepted")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-bound-evidence", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_hold_only or not (args.check_bound_evidence or args.self_test):
        parser.error("explicit HOLD-only evidence check is required")
    source = Path(__file__).absolute()
    root = source.parents[2]
    if source.is_symlink() or source.resolve(strict=True) != source:
        raise ValueError("checker source path is indirect")
    checks = self_test(root) if args.self_test else verify(root)
    print("METRIC openmm86_deposited_ph_recovery_v6_evidence_checks=%d" % checks)
    print("METRIC deposited_ph_feasible_entities=115")
    print("METRIC deposited_ph_hold_entities=20")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
