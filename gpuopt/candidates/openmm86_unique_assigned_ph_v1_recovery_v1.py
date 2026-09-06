#!/usr/bin/env python3
"""HOLD-only Run 143 recovery with corrected immutable archive bindings.

This is a separately named recovery, not an in-place retry.  It preserves the
v1 fallback rule by calling the committed v1 implementation verbatim.  A
metadata result is never protonation, all-atom, support, science, score, or
authorization evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    # Do not duplicate or alter the v1 fallback semantics.
    from openmm86_unique_assigned_ph_v1 import (
        _fallback_entity as _v1_fallback_entity,
        self_test as _v1_self_test,
    )
    from check_openmm86_deposited_ph_recovery_v6_semantic_independent import (
        PRIOR_NAME,
        PRIOR_SHA256 as CHECKER_PRIOR_ARCHIVE_SHA256,
        V6_NAME,
        V6_SHA256 as CHECKER_V6_ARCHIVE_SHA256,
        _check_receipt_semantics,
        _derive_entity,
        _validate_prior,
        _validate_v6,
    )
except ImportError:  # Package import for external synthetic test harnesses.
    from gpuopt.candidates.openmm86_unique_assigned_ph_v1 import (
        _fallback_entity as _v1_fallback_entity,
        self_test as _v1_self_test,
    )
    from gpuopt.candidates.check_openmm86_deposited_ph_recovery_v6_semantic_independent import (
        PRIOR_NAME,
        PRIOR_SHA256 as CHECKER_PRIOR_ARCHIVE_SHA256,
        V6_NAME,
        V6_SHA256 as CHECKER_V6_ARCHIVE_SHA256,
        _check_receipt_semantics,
        _derive_entity,
        _validate_prior,
        _validate_v6,
    )


CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v1"
CONTRACT = CANDIDATE_ID
CANDIDATE_RELATIVE = Path(
    "gpuopt/candidates/openmm86_unique_assigned_ph_v1_recovery_v1.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v1_plan.json"
)
PARENT_PLAN_RELATIVE = Path("gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json")
SEMANTIC_CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py"
)
V6_EVIDENCE_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json"
)
FAILED_SOURCE_RELATIVE = Path("gpuopt/candidates/openmm86_unique_assigned_ph_v1.py")
FAILED_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_plan.json"
)
FAILED_INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_execution_intent.json"
)
FAILED_RECEIPT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_failure_receipt.json"
)
RUN143_LOG_RELATIVE = Path(".auto/log.jsonl")
RECOVERY_INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v1_execution_intent.json"
)
OUTPUT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v1/receipt.json"
)

FAILED_CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1"
FAILED_SOURCE_SHA256 = "29974dbb0b41a0e3282d67171dfd939b212dc983d5db4cc90be08f533fcf0752"
FAILED_PLAN_SHA256 = "c35724112714e3e5de474f08538540e121241ebef3458717b034e712dd5b083e"
FAILED_INTENT_SHA256 = "c5ba9c469e2a34a25bf6140614578993baa6355a06bc85ecbe83839ba9e25400"
FAILED_RECEIPT_SHA256 = "db5f93fc1515fceee3aaedf0f269de996bd1f96d22561e41f2a48f47ee834752"
RUN143_RAW_LINE_SHA256 = "e746ddc9c4ae49c7b27436490ca7cb2cb2c4c3669ea996e79aa9ef0b061d0e04"
PARENT_PLAN_SHA256 = "9244f880c7a67e97c5a370013b3cfc81cccc8cb9d7267b2fff0b0a1b5dddab91"
V6_EVIDENCE_RECEIPT_SHA256 = "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36"
SEMANTIC_CHECKER_SHA256 = "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84"

# These are the corrected immutable bindings.  The failed v1 plan used the
# stale values below, which are admission evidence only and never read as data.
PRIOR_ARCHIVE_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
V6_ARCHIVE_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"
STALE_PRIOR_ARCHIVE_SHA256 = "cca8e8f94bb812d226f19055f4b6377da9f070520f21fc3ab76bde490a5b0b8a"
STALE_V6_ARCHIVE_SHA256 = "0fc60731b7f6cc053c9cf623ecb8e033ce8058f0f42791ea894414e98143d7b8"

CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
MAX_REGULAR_BYTES = 2_000_000
MAX_LOG_BYTES = 5_000_000
MAX_LOG_LINES = 10_000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

# The receipt is sealed by its full raw SHA-256 and additionally checked as a
# strict object so an unknown JSON field cannot be silently accepted.
FAILED_RECEIPT_CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
FAILED_RECEIPT = {
    "artifact_kind": "hold_only_metadata_reinterpretation_prearchive_failure_receipt",
    "candidate_id": FAILED_CANDIDATE_ID,
    "closed_capabilities": FAILED_RECEIPT_CLOSED_CAPABILITIES,
    "contract": "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_failure_v1",
    "execution_intent_path": FAILED_INTENT_RELATIVE.as_posix(),
    "execution_intent_sha256": FAILED_INTENT_SHA256,
    "expected_prior_archive_sha256": STALE_PRIOR_ARCHIVE_SHA256,
    "failure_stage": "prior_archive_whole_file_sha256_before_zip_open_or_member_parse",
    "observed_prior_archive_sha256": PRIOR_ARCHIVE_SHA256,
    "output_receipt_absent": True,
    "run": 143,
    "run_log_line_sha256": RUN143_RAW_LINE_SHA256,
    "state": "TERMINAL_FAILED_NO_REUSE",
    "v6_archive_sha256": V6_ARCHIVE_SHA256,
}


class RecoveryError(ValueError):
    """A fail-closed recovery admission or metadata ambiguity."""


def _fail(message: str) -> None:
    raise RecoveryError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("duplicate JSON key: %s" % key)
        value[key] = item
    return value


def _constant(value: str) -> None:
    _fail("non-finite JSON constant: %s" % value)


def _loads(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecoveryError) as error:
        _fail("invalid JSON in %s: %s" % (label, error))


def _safe_relative(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        _fail("unsafe repository-relative path")


def _read_regular(root: Path, relative: Path, label: str, maximum: int) -> bytes:
    """Read one bounded, direct regular file without following symlinks."""
    _safe_relative(relative)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise RecoveryError("missing %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode):
            _fail("indirect %s" % label)
    if not stat.S_ISREG(os.lstat(current).st_mode):
        _fail("non-regular %s" % label)
    descriptor = os.open(str(current), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
            _fail("unbounded or non-regular %s" % label)
        raw = b""
        while len(raw) <= maximum:
            block = os.read(descriptor, min(131072, maximum + 1 - len(raw)))
            if not block:
                break
            raw += block
        if len(raw) != opened.st_size or len(raw) > maximum:
            _fail("%s changed while reading" % label)
        return raw
    finally:
        os.close(descriptor)


def _read_bound_regular(
    root: Path, relative: Path, digest: str, label: str, maximum: int = MAX_REGULAR_BYTES
) -> bytes:
    raw = _read_regular(root, relative, label, maximum)
    if _sha256(raw) != digest:
        _fail("%s raw SHA-256 drifted" % label)
    return raw


def _exact(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _exact(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact(item, wanted) for item, wanted in zip(value, expected)
        )
    return value == expected


def _validate_failed_intent(intent: Any) -> None:
    """Validate the sealed v1 intent, including its stale (not corrected) hashes."""
    keys = {
        "artifact_kind", "candidate_id", "candidate_source_git_commit",
        "candidate_source_raw_sha256", "closed_capabilities", "contract",
        "plan_raw_sha256", "prior_archive_raw_sha256", "state",
        "v6_archive_raw_sha256",
    }
    if not isinstance(intent, dict) or set(intent) != keys:
        _fail("failed v1 intent schema has unknown or missing fields")
    if not (
        intent["artifact_kind"] == "hold_only_metadata_reinterpretation_execution_intent"
        and intent["candidate_id"] == FAILED_CANDIDATE_ID
        and type(intent["candidate_source_git_commit"]) is str
        and COMMIT_RE.fullmatch(intent["candidate_source_git_commit"]) is not None
        and intent["candidate_source_raw_sha256"] == FAILED_SOURCE_SHA256
        and intent["closed_capabilities"] == CLOSED_CAPABILITIES
        and intent["contract"] == FAILED_CANDIDATE_ID
        and intent["plan_raw_sha256"] == FAILED_PLAN_SHA256
        and intent["prior_archive_raw_sha256"] == STALE_PRIOR_ARCHIVE_SHA256
        and intent["state"] == "ACTIVE_ONCE"
        and intent["v6_archive_raw_sha256"] == STALE_V6_ARCHIVE_SHA256
    ):
        _fail("failed v1 intent values drifted")


def _validate_failed_receipt(receipt: Any) -> None:
    if not _exact(receipt, FAILED_RECEIPT):
        _fail("failed v1 receipt schema or values drifted")


def _validate_run143_line(raw: bytes) -> None:
    if _sha256(raw) != RUN143_RAW_LINE_SHA256:
        _fail("Run 143 raw log-line SHA-256 drifted")
    record = _loads(raw, "Run 143 log line")
    if not isinstance(record, dict) or type(record.get("run")) is not int or record.get("run") != 143:
        _fail("Run 143 log-line schema drifted")


def _read_run143_log(root: Path) -> None:
    raw_log = _read_regular(root, RUN143_LOG_RELATIVE, "append-only operational log", MAX_LOG_BYTES)
    lines = raw_log.splitlines(keepends=True)
    if len(lines) > MAX_LOG_LINES:
        _fail("append-only operational log exceeds the line limit")
    matches = [line for line in lines if _sha256(line) == RUN143_RAW_LINE_SHA256]
    if len(matches) != 1:
        _fail("Run 143 exact raw log line is absent or duplicated")
    _validate_run143_line(matches[0])


def _expected_plan(source_sha256: str) -> Dict[str, Any]:
    """The entire separately named plan, with only this source hash dynamic."""
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_recovery_plan_not_authorization",
        "bound_failure": {
            "failed_execution_intent": {
                "path": FAILED_INTENT_RELATIVE.as_posix(),
                "raw_sha256": FAILED_INTENT_SHA256,
            },
            "failed_failure_receipt": {
                "path": FAILED_RECEIPT_RELATIVE.as_posix(),
                "raw_sha256": FAILED_RECEIPT_SHA256,
                "exact_json": FAILED_RECEIPT,
            },
            "failed_plan": {
                "path": FAILED_PLAN_RELATIVE.as_posix(), "raw_sha256": FAILED_PLAN_SHA256,
            },
            "failed_source": {
                "path": FAILED_SOURCE_RELATIVE.as_posix(), "raw_sha256": FAILED_SOURCE_SHA256,
            },
            "run_143_log": {
                "path": RUN143_LOG_RELATIVE.as_posix(),
                "raw_line_sha256": RUN143_RAW_LINE_SHA256,
                "semantic_subset": {"run": 143},
            },
            "stale_archive_bindings": {
                "prior_archive_raw_sha256": STALE_PRIOR_ARCHIVE_SHA256,
                "recovery_v6_archive_raw_sha256": STALE_V6_ARCHIVE_SHA256,
            },
        },
        "bound_inputs": {
            "parent_plan": {
                "path": PARENT_PLAN_RELATIVE.as_posix(), "raw_sha256": PARENT_PLAN_SHA256,
            },
            "prior_condition_archive": {
                "path": ".auto/staging/" + PRIOR_NAME, "raw_sha256": PRIOR_ARCHIVE_SHA256,
            },
            "recovery_v6_evidence_archive": {
                "path": ".auto/staging/" + V6_NAME, "raw_sha256": V6_ARCHIVE_SHA256,
            },
            "recovery_v6_evidence_receipt": {
                "path": V6_EVIDENCE_RECEIPT_RELATIVE.as_posix(),
                "raw_sha256": V6_EVIDENCE_RECEIPT_SHA256,
            },
            "recovery_v6_semantic_checker": {
                "path": SEMANTIC_CHECKER_RELATIVE.as_posix(),
                "raw_sha256": SEMANTIC_CHECKER_SHA256,
            },
        },
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "execution": {
            "acknowledgement": "--acknowledge-hold-only",
            "default_mode": "synthetic in-memory self-test; no .auto reads",
            "external_execution_intent": RECOVERY_INTENT_RELATIVE.as_posix(),
            "execute_flag": "--execute",
            "output_receipt": OUTPUT_RELATIVE.as_posix(),
            "receipt_creation": "O_EXCL; one separately named receipt only",
            "reviewed_binding": "A separately created exact-schema intent must bind committed Git HEAD, recovery source SHA-256, recovery plan SHA-256, and both corrected archive SHA-256 values.",
        },
        "metadata_resolution_limit": "Metadata cannot establish protonation feasibility, all-atom feasibility, support feasibility, or science feasibility.",
        "recovery_source": {
            "path": CANDIDATE_RELATIVE.as_posix(), "raw_sha256": source_sha256,
        },
        "recovery_v6_result": {
            "hold_identity_count": 20, "metadata_resolved_entity_count": 115,
        },
        "state": "HOLD_ONLY_STATIC_RECOVERY",
        "warnings": [
            "The only operational correction is replacement of the stale archive bindings with corrected immutable archive bindings.",
            "The v1 fallback entity semantics are imported unchanged from openmm86_unique_assigned_ph_v1._fallback_entity.",
            "No metadata result is protonation, all-atom, support, science, score, target, or authorization evidence.",
        ],
    }


def _validate_plan(plan: Any, source_sha256: str) -> None:
    if not _exact(plan, _expected_plan(source_sha256)):
        _fail("recovery plan schema, identities, or bindings drifted")


def _git(root: Path, arguments: Sequence[str]) -> bytes:
    try:
        process = subprocess.run(
            list(arguments), cwd=str(root), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryError("cannot inspect committed recovery bytes: %s" % error) from error
    if process.returncode != 0 or len(process.stdout) > MAX_REGULAR_BYTES:
        _fail("cannot inspect committed recovery bytes")
    return process.stdout


def _canonical_root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / CANDIDATE_RELATIVE
    if (
        source != expected
        or source.is_symlink()
        or source.resolve(strict=True) != expected.resolve(strict=True)
        or not root.is_dir()
    ):
        _fail("recovery source path is indirect or misplaced")
    return root


def _validate_recovery_intent(
    intent: Any, revision: str, source_sha256: str, plan_sha256: str,
) -> None:
    expected = {
        "artifact_kind": "hold_only_unique_assigned_ph_recovery_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": plan_sha256,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    if not _exact(intent, expected):
        _fail("recovery execution intent schema or reviewed bindings drifted")


def _prearchive_validation(root: Path) -> Dict[str, str]:
    """Validate all failure and intent evidence before either archive is opened."""
    if (
        CHECKER_PRIOR_ARCHIVE_SHA256 != PRIOR_ARCHIVE_SHA256
        or CHECKER_V6_ARCHIVE_SHA256 != V6_ARCHIVE_SHA256
        or PRIOR_ARCHIVE_SHA256 == STALE_PRIOR_ARCHIVE_SHA256
        or V6_ARCHIVE_SHA256 == STALE_V6_ARCHIVE_SHA256
    ):
        _fail("committed semantic checker or corrected archive bindings drifted")

    _read_bound_regular(root, FAILED_SOURCE_RELATIVE, FAILED_SOURCE_SHA256, "failed v1 source")
    _read_bound_regular(root, FAILED_PLAN_RELATIVE, FAILED_PLAN_SHA256, "failed v1 plan")
    _read_bound_regular(root, PARENT_PLAN_RELATIVE, PARENT_PLAN_SHA256, "parent plan")
    _read_bound_regular(root, SEMANTIC_CHECKER_RELATIVE, SEMANTIC_CHECKER_SHA256, "semantic checker")
    _read_bound_regular(
        root, V6_EVIDENCE_RECEIPT_RELATIVE, V6_EVIDENCE_RECEIPT_SHA256,
        "recovery-v6 evidence receipt",
    )

    failed_intent_raw = _read_bound_regular(
        root, FAILED_INTENT_RELATIVE, FAILED_INTENT_SHA256, "failed v1 intent"
    )
    _validate_failed_intent(_loads(failed_intent_raw, "failed v1 intent"))
    failed_receipt_raw = _read_bound_regular(
        root, FAILED_RECEIPT_RELATIVE, FAILED_RECEIPT_SHA256, "failed v1 receipt"
    )
    _validate_failed_receipt(_loads(failed_receipt_raw, "failed v1 receipt"))
    _read_run143_log(root)

    source_raw = _read_regular(root, CANDIDATE_RELATIVE, "recovery source", MAX_REGULAR_BYTES)
    source_sha256 = _sha256(source_raw)
    plan_raw = _read_regular(root, PLAN_RELATIVE, "recovery plan", MAX_REGULAR_BYTES)
    plan_sha256 = _sha256(plan_raw)
    _validate_plan(_loads(plan_raw, "recovery plan"), source_sha256)

    revision = _git(root, ("git", "rev-parse", "HEAD")).decode("ascii").strip()
    if COMMIT_RE.fullmatch(revision) is None:
        _fail("committed Git HEAD is malformed")
    for relative, current, label in (
        (CANDIDATE_RELATIVE, source_raw, "recovery source"),
        (PLAN_RELATIVE, plan_raw, "recovery plan"),
    ):
        committed = _git(root, ("git", "show", "%s:%s" % (revision, relative.as_posix())))
        if committed != current:
            _fail("%s bytes differ from committed Git HEAD" % label)

    intent_raw = _read_regular(root, RECOVERY_INTENT_RELATIVE, "recovery execution intent", MAX_REGULAR_BYTES)
    _validate_recovery_intent(
        _loads(intent_raw, "recovery execution intent"), revision, source_sha256, plan_sha256
    )
    return {
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "execution_intent_path": RECOVERY_INTENT_RELATIVE.as_posix(),
        "execution_intent_raw_sha256": _sha256(intent_raw),
        "plan_raw_sha256": plan_sha256,
    }


def _receipt(
    records: Sequence[Dict[str, Any]], provenance: Mapping[str, str],
) -> Dict[str, Any]:
    recovered = sum(1 for item in records if item.get("fallback_metadata_resolved") is True)
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_recovery_receipt",
        "bound_inputs": {
            "failed_failure_receipt_raw_sha256": FAILED_RECEIPT_SHA256,
            "parent_plan_raw_sha256": PARENT_PLAN_SHA256,
            "prior_archive": {
                "path": ".auto/staging/" + PRIOR_NAME, "raw_sha256": PRIOR_ARCHIVE_SHA256,
            },
            "recovery_v6_archive": {
                "path": ".auto/staging/" + V6_NAME, "raw_sha256": V6_ARCHIVE_SHA256,
            },
            "recovery_v6_evidence_receipt_raw_sha256": V6_EVIDENCE_RECEIPT_SHA256,
            "recovery_v6_semantic_checker_raw_sha256": SEMANTIC_CHECKER_SHA256,
            "run_143_raw_line_sha256": RUN143_RAW_LINE_SHA256,
        },
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "combined_metadata_resolved_entity_count": 115 + recovered,
        "contract": CONTRACT,
        "entities": list(records),
        "entity_count": 135,
        "execution_provenance": dict(provenance),
        "fallback_candidate_entity_count": len(records),
        "fallback_metadata_resolved_entity_count": recovered,
        "metadata_resolution_limit": "Metadata resolution cannot establish protonation feasibility, all-atom feasibility, support feasibility, or science feasibility.",
        "output": {"path": OUTPUT_RELATIVE.as_posix(), "receipt_creation": "O_EXCL"},
        "recovery_v6_hold_entity_count": 20,
        "recovery_v6_metadata_resolved_entity_count": 115,
        "status": "HOLD_METADATA_REINTERPRETATION_ONLY",
    }


def _staging_directory(root: Path) -> Path:
    current = root
    for part in (".auto", "staging"):
        current = current / part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise RecoveryError("missing direct output parent: %s" % error) from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            _fail("output parent is indirect or non-directory")
    return current


def _write_receipt_once(root: Path, payload: Dict[str, Any]) -> None:
    """Create the separately named output and its receipt with O_EXCL only."""
    staging = _staging_directory(root)
    output = staging / OUTPUT_RELATIVE.parent.name
    try:
        os.mkdir(str(output), 0o700)
    except FileExistsError as error:
        raise RecoveryError("recovery output already exists; no in-place retry") from error
    details = os.lstat(output)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        _fail("new recovery output is indirect")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = output / "receipt.json"
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute(root: Path) -> Dict[str, Any]:
    """Perform the one HOLD-only recovery after all non-archive gates pass."""
    provenance = _prearchive_validation(root)

    # These imported checker functions own the corrected fixed SHA-256 checks.
    # No archive is opened until every Run 143 failure gate above has passed.
    staging = _staging_directory(root)
    roster, prior, prior_bindings = _validate_prior(staging / PRIOR_NAME)
    v6_receipt, assigned, v6_bindings = _validate_v6(staging / V6_NAME, roster)
    derived = [
        _derive_entity(
            bmrb, uid, prior, assigned[bmrb], prior_bindings, v6_bindings[bmrb]
        )
        for bmrb, uid in roster
    ]
    _check_receipt_semantics(v6_receipt, derived, roster, v6_bindings)

    held = [
        (item["bmrb_id"], item["entity_uid"])
        for item in derived
        if item.get("ph_feasible") is False
    ]
    if (
        len(roster) != 135
        or len(derived) != 135
        or len(held) != 20
        or sum(1 for item in derived if item.get("ph_feasible") is True) != 115
        or len(set(held)) != 20
        or set(held) - set(roster)
    ):
        _fail("independently validated recovery-v6 HOLD identities drifted")

    # This is deliberately the exact committed v1 function, not a copy.
    records = [
        _v1_fallback_entity(bmrb, uid, prior, assigned[bmrb])
        for bmrb, uid in roster
        if (bmrb, uid) in set(held)
    ]
    if len(records) != 20 or {(item.get("bmrb_id"), item.get("entity_uid")) for item in records} != set(held):
        _fail("v1 fallback recovery identity set drifted")
    payload = _receipt(records, provenance)
    _write_receipt_once(root, payload)
    return payload


def _expect_recovery_error(callback: Any, label: str) -> None:
    try:
        callback()
    except RecoveryError:
        return
    raise AssertionError("%s was accepted" % label)


def self_test() -> int:
    """Synthetic-only checks.  This function never reads .auto or an archive."""
    v1_checks = _v1_self_test()
    if v1_checks != 12:
        raise AssertionError("v1 fallback semantic check count drifted")
    checks = v1_checks

    if PRIOR_ARCHIVE_SHA256 == STALE_PRIOR_ARCHIVE_SHA256:
        raise AssertionError("stale prior hash equals corrected hash")
    if V6_ARCHIVE_SHA256 == STALE_V6_ARCHIVE_SHA256:
        raise AssertionError("stale v6 hash equals corrected hash")
    checks += 2

    failed_intent = {
        "artifact_kind": "hold_only_metadata_reinterpretation_execution_intent",
        "candidate_id": FAILED_CANDIDATE_ID,
        "candidate_source_git_commit": "a" * 40,
        "candidate_source_raw_sha256": FAILED_SOURCE_SHA256,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": FAILED_CANDIDATE_ID,
        "plan_raw_sha256": FAILED_PLAN_SHA256,
        "prior_archive_raw_sha256": STALE_PRIOR_ARCHIVE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": STALE_V6_ARCHIVE_SHA256,
    }
    _validate_failed_intent(failed_intent)
    _validate_failed_receipt(FAILED_RECEIPT)
    receipt_extra = dict(FAILED_RECEIPT)
    receipt_extra["unexpected"] = True
    _expect_recovery_error(lambda: _validate_failed_receipt(receipt_extra), "failure receipt extra field")
    checks += 3

    if CANDIDATE_ID == FAILED_CANDIDATE_ID or OUTPUT_RELATIVE.name == "receipt.json" and OUTPUT_RELATIVE.parent.name == "openmm86_unique_assigned_ph_v1":
        raise AssertionError("recovery identity or output is not separately named")
    checks += 1

    recovery_intent = {
        "artifact_kind": "hold_only_unique_assigned_ph_recovery_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": "b" * 40,
        "candidate_source_raw_sha256": "c" * 64,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": "d" * 64,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    _validate_recovery_intent(recovery_intent, "b" * 40, "c" * 64, "d" * 64)
    altered = dict(recovery_intent)
    altered["unexpected"] = False
    _expect_recovery_error(
        lambda: _validate_recovery_intent(altered, "b" * 40, "c" * 64, "d" * 64),
        "recovery intent unknown field",
    )
    checks += 2

    if (
        CHECKER_PRIOR_ARCHIVE_SHA256 != PRIOR_ARCHIVE_SHA256
        or CHECKER_V6_ARCHIVE_SHA256 != V6_ARCHIVE_SHA256
    ):
        raise AssertionError("committed semantic checker fixed hashes drifted")
    checks += 1
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test or not (args.execute and args.acknowledge_hold_only):
        checks = self_test()
        print("METRIC openmm86_unique_assigned_ph_v1_recovery_v1_v1_semantic_checks=12")
        print("METRIC openmm86_unique_assigned_ph_v1_recovery_v1_synthetic_checks=%d" % checks)
        print("METRIC target_values_read=0")
        print("METRIC source_scores_read=0")
        print("METRIC science_executed=0")
        print("METRIC authorization_consumed=0")
        print("STATUS HOLD_SELF_TEST_ONLY")
        return 0
    try:
        payload = execute(_canonical_root())
    except (RecoveryError, OSError, ValueError) as error:
        print("REFUSAL HOLD-only Run 143 recovery: %s" % error)
        return 3
    print("METRIC fallback_metadata_resolved_entity_count=%d" % payload["fallback_metadata_resolved_entity_count"])
    print("METRIC combined_metadata_resolved_entity_count=%d" % payload["combined_metadata_resolved_entity_count"])
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_METADATA_REINTERPRETATION_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
