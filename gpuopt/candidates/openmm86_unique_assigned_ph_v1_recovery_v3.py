#!/usr/bin/env python3
"""HOLD-only recovery-v3 for recovery-v2's terminal hash-framing failure.

This separately named candidate preserves recovery-v2's sole operational
repair: each frozen-v1 fallback call receives exactly the three-loop mapping
for its BMRB entity.  It additionally binds recovery-v2's terminal failure,
which resulted only from hashing a line-number-prefixed Run 146 JSONL record.
Metadata remains metadata only; it is not protonation, support, science,
score, target, or authorization evidence.
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
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    # Keep the frozen v1 fallback implementation and its loop vocabulary.
    import openmm86_unique_assigned_ph_v1 as v1
    # This is recovery-v1's independently authored recovery-v6 identity path.
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
    from gpuopt.candidates import openmm86_unique_assigned_ph_v1 as v1
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


CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v3"
CONTRACT = CANDIDATE_ID
CANDIDATE_RELATIVE = Path(
    "gpuopt/candidates/openmm86_unique_assigned_ph_v1_recovery_v3.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v3_plan.json"
)
PARENT_PLAN_RELATIVE = Path("gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json")
SEMANTIC_CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py"
)
V6_EVIDENCE_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json"
)

V1_SOURCE_RELATIVE = Path("gpuopt/candidates/openmm86_unique_assigned_ph_v1.py")
V1_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_plan.json"
)
RECOVERY_V1_SOURCE_RELATIVE = Path(
    "gpuopt/candidates/openmm86_unique_assigned_ph_v1_recovery_v1.py"
)
RECOVERY_V1_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v1_plan.json"
)
RECOVERY_V1_INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v1_execution_intent.json"
)
RECOVERY_V1_OUTPUT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v1/receipt.json"
)
RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v1_postoutput_failure_receipt.json"
)
RUN146_LOG_RELATIVE = Path(".auto/log.jsonl")
RUN148_LOG_RELATIVE = RUN146_LOG_RELATIVE
RECOVERY_V2_SOURCE_RELATIVE = Path(
    "gpuopt/candidates/openmm86_unique_assigned_ph_v1_recovery_v2.py"
)
RECOVERY_V2_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v2_plan.json"
)
RECOVERY_V2_INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v2_execution_intent.json"
)
RECOVERY_V2_FAILURE_RECEIPT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v2_failure_receipt.json"
)
RECOVERY_V2_OUTPUT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v2/receipt.json"
)
RECOVERY_V3_INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3_execution_intent.json"
)
OUTPUT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json"
)

PARENT_PLAN_SHA256 = "9244f880c7a67e97c5a370013b3cfc81cccc8cb9d7267b2fff0b0a1b5dddab91"
V1_SOURCE_SHA256 = "29974dbb0b41a0e3282d67171dfd939b212dc983d5db4cc90be08f533fcf0752"
V1_PLAN_SHA256 = "c35724112714e3e5de474f08538540e121241ebef3458717b034e712dd5b083e"
RECOVERY_V1_SOURCE_SHA256 = "1534df3eb693c8d7eaa228349ee001c4a330f3b36c6a35efb25b83f966dab2d4"
RECOVERY_V1_PLAN_SHA256 = "368cf599cbe6df757f40d3f214e4e5691511daf13ed0a7b36c59e5e55c20a575"
RECOVERY_V1_SOURCE_GIT_COMMIT = "8142433b41ba8d60636b81f8c82f0fc081d5e9b0"
RECOVERY_V1_INTENT_SHA256 = "ef7a18d2c34cccf3cc69fa100e623786ab8cbda984e65b4ee1e3dbacda48a4b3"

RECOVERY_V1_OUTPUT_SHA256 = "e3bc6ae75a79cff1d3580ea366d3631dfdad87f60dd1b228e069edad770f99dc"
# This is the SHA-256 of the raw JSONL bytes, including the final newline.
RUN146_RAW_LINE_SHA256 = "57063cba97bc9e7f925d037239c6acd41835e69af9c2d7f75462a270b689b4a5"
# The immutable recovery-v1 failure receipt records its then-wrong derivation.
HISTORICAL_RUN146_PREFIXED_LINE_SHA256 = "184f3a4c600741ee2d56ea986d29d092c969f97a4b8b1a02192a3df9856ce86f"
RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_SHA256 = "1c095c3d3bb61b6b6ea8859c723053d3293137799f70e5a4911110daa067a910"

RECOVERY_V2_SOURCE_GIT_COMMIT = "2866c1c1ac8bfbe26b171075c0fcebf14aeb6e93"
RECOVERY_V2_SOURCE_SHA256 = "9b104d59b7dc9aaea1283a863c091ee81bf32bda045b0a9e04412f808a214d81"
RECOVERY_V2_PLAN_SHA256 = "d54b369bf03942cad787efdbeb60583fdd13ec461bb7cfcc68f44184b568c821"
RECOVERY_V2_INTENT_SHA256 = "98768c7174a2fb30ce9d4561fca012026884d0cedeecf6522ac696ce65f1379d"
RECOVERY_V2_FAILURE_RECEIPT_SHA256 = "433839ecd75345ac240dda67b5ba91ab4329a9ec08a5d540c455c5eca7c659b7"
RUN148_RAW_LINE_SHA256 = "8efb4c6490095f9a39459371f3fce0243ce4e8c3b6846d9f6d68045c08d03433"

V6_EVIDENCE_RECEIPT_SHA256 = "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36"
SEMANTIC_CHECKER_SHA256 = "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84"
PRIOR_ARCHIVE_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
V6_ARCHIVE_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"

CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
RECOVERY_V1_FAILURE_CLOSED_CAPABILITIES = CLOSED_CAPABILITIES
MAX_REGULAR_BYTES = 2_000_000
MAX_LOG_BYTES = 5_000_000
MAX_LOG_LINES = 10_000
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

RECOVERY_V1_INTENT = {
    "artifact_kind": "hold_only_unique_assigned_ph_recovery_execution_intent",
    "candidate_id": "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v1",
    "candidate_source_git_commit": RECOVERY_V1_SOURCE_GIT_COMMIT,
    "candidate_source_raw_sha256": RECOVERY_V1_SOURCE_SHA256,
    "closed_capabilities": CLOSED_CAPABILITIES,
    "contract": "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v1",
    "plan_raw_sha256": RECOVERY_V1_PLAN_SHA256,
    "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
    "state": "ACTIVE_ONCE",
    "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
}

# The post-output receipt is a terminal failure binding, not a data source.
# Its historical line hash remains exact inside that immutable receipt.  The
# separately validated actual Run 146 bytes below use RUN146_RAW_LINE_SHA256.
RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_FIXED = {
    "artifact_kind": "hold_only_unique_assigned_ph_recovery_postoutput_failure_receipt",
    "candidate_id": RECOVERY_V1_INTENT["candidate_id"],
    "candidate_source_git_commit": RECOVERY_V1_SOURCE_GIT_COMMIT,
    "candidate_source_raw_sha256": RECOVERY_V1_SOURCE_SHA256,
    "closed_capabilities": RECOVERY_V1_FAILURE_CLOSED_CAPABILITIES,
    "error_class": "TypeError",
    "error_message": "_require() missing 2 required positional arguments: 'checks' and 'name'",
    "execution_intent_raw_sha256": RECOVERY_V1_INTENT_SHA256,
    "failed_plan_raw_sha256": RECOVERY_V1_PLAN_SHA256,
    "failure_stage": "after_O_EXCL_candidate_receipt_before_parent_metric_emission",
    "frozen_call_scope_defect": "full 135-entry prior loop mapping was passed to an entity-local v1 fallback that requires exactly the three loops for one BMRB entry",
    "output_receipt_raw_sha256": RECOVERY_V1_OUTPUT_SHA256,
    "output_semantically_eligible": False,
    "repair_limit": "scope the prior mapping to the current BMRB entry only; preserve all predeclared v1 fallback semantics, the 20 independently derived HOLD identities, and all closed capabilities",
    "run_146_raw_line_sha256": HISTORICAL_RUN146_PREFIXED_LINE_SHA256,
    "run_number": 146,
    "state": "TERMINAL_FAILED_NO_REUSE",
}

# This exact object is the terminal v2 prearchive failure binding.  It is
# admission evidence only; its corrected actual-line hash is independently
# matched against raw JSONL bytes before either ZIP is opened.
RECOVERY_V2_FAILURE_RECEIPT = {
    "artifact_kind": "hold_only_unique_assigned_ph_recovery_v2_prearchive_failure_receipt",
    "candidate_id": "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v2",
    "candidate_source_git_commit": RECOVERY_V2_SOURCE_GIT_COMMIT,
    "candidate_source_raw_sha256": RECOVERY_V2_SOURCE_SHA256,
    "closed_capabilities": CLOSED_CAPABILITIES,
    "error_class": "RecoveryError",
    "error_message": "Run 146 exact raw log line is absent or duplicated",
    "execution_intent_raw_sha256": RECOVERY_V2_INTENT_SHA256,
    "failure_stage": "prearchive_Run146_exact_raw_log_validation",
    "incorrect_run_146_raw_line_sha256": HISTORICAL_RUN146_PREFIXED_LINE_SHA256,
    "actual_run_146_raw_line_sha256": RUN146_RAW_LINE_SHA256,
    "incorrect_hash_derivation": "grep -n prefixed the selected JSONL line with its line number before hashing",
    "output_receipt_absent": True,
    "plan_raw_sha256": RECOVERY_V2_PLAN_SHA256,
    "run_148_raw_line_sha256": RUN148_RAW_LINE_SHA256,
    "run_number": 148,
    "state": "TERMINAL_FAILED_NO_REUSE",
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


def _safe_relative(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        _fail("unsafe repository-relative path")


def _read_regular(root: Path, relative: Path, label: str, maximum: int) -> bytes:
    """Read one bounded direct regular file without following symlinks."""
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


def _git(root: Path, arguments: Sequence[str]) -> bytes:
    try:
        process = subprocess.run(
            list(arguments), cwd=str(root), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryError("cannot inspect committed bytes: %s" % error) from error
    if process.returncode != 0 or len(process.stdout) > MAX_REGULAR_BYTES:
        _fail("cannot inspect committed bytes")
    return process.stdout


def _validate_committed_source(
    root: Path, relative: Path, digest: str, commit: str, label: str,
) -> None:
    raw = _read_bound_regular(root, relative, digest, label)
    committed = _git(root, ("git", "show", "%s:%s" % (commit, relative.as_posix())))
    if committed != raw or _sha256(committed) != digest:
        _fail("%s differs from its bound source commit" % label)


def _validate_recovery_v1_source(root: Path) -> None:
    _validate_committed_source(
        root, RECOVERY_V1_SOURCE_RELATIVE, RECOVERY_V1_SOURCE_SHA256,
        RECOVERY_V1_SOURCE_GIT_COMMIT, "recovery-v1 source",
    )


def _validate_recovery_v2_source(root: Path) -> None:
    _validate_committed_source(
        root, RECOVERY_V2_SOURCE_RELATIVE, RECOVERY_V2_SOURCE_SHA256,
        RECOVERY_V2_SOURCE_GIT_COMMIT, "recovery-v2 source",
    )


def _require_absent(root: Path, relative: Path, label: str) -> None:
    """Reject an existing artifact and indirect paths; a missing parent is absent."""
    _safe_relative(relative)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise RecoveryError("cannot inspect %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode):
            _fail("indirect %s" % label)
        if index != len(relative.parts) - 1 and not stat.S_ISDIR(details.st_mode):
            _fail("non-directory parent for %s" % label)
    _fail("%s is present" % label)


def _validate_recovery_v1_intent(intent: Any) -> None:
    if not _exact(intent, RECOVERY_V1_INTENT):
        _fail("recovery-v1 execution intent schema or values drifted")


def _validate_postoutput_failure_receipt(
    receipt: Any, sealed_output_sha256: str,
) -> None:
    if (
        not _exact(receipt, RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_FIXED)
        or sealed_output_sha256 != RECOVERY_V1_OUTPUT_SHA256
    ):
        _fail("recovery-v1 post-output failure receipt drifted")


def _validate_recovery_v2_failure_receipt(receipt: Any) -> None:
    if not _exact(receipt, RECOVERY_V2_FAILURE_RECEIPT):
        _fail("recovery-v2 prearchive failure receipt drifted")


def _validate_exact_raw_log_line(
    raw_line: bytes, expected_sha256: str, run_number: int, label: str,
) -> None:
    """Require raw JSONL framing; never hash a display-prefixed representation."""
    if not raw_line.endswith(b"\n") or _sha256(raw_line) != expected_sha256:
        _fail("%s raw log-line SHA-256 or newline framing drifted" % label)
    record = _loads(raw_line, label)
    if (
        not isinstance(record, dict)
        or type(record.get("run")) is not int
        or record.get("run") != run_number
    ):
        _fail("%s schema drifted" % label)


def _read_exact_log_line(
    root: Path, relative: Path, expected_sha256: str, run_number: int, label: str,
) -> str:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        _fail("%s reviewed SHA-256 binding is malformed" % label)
    raw_log = _read_regular(root, relative, "append-only operational log", MAX_LOG_BYTES)
    lines = raw_log.splitlines(keepends=True)
    if len(lines) > MAX_LOG_LINES:
        _fail("append-only operational log exceeds the line limit")
    matches = [line for line in lines if _sha256(line) == expected_sha256]
    if len(matches) != 1:
        _fail("%s exact raw log line is absent or duplicated" % label)
    _validate_exact_raw_log_line(matches[0], expected_sha256, run_number, label)
    return expected_sha256


def _read_run146_log(root: Path, expected_sha256: str) -> str:
    if expected_sha256 != RUN146_RAW_LINE_SHA256:
        _fail("Run 146 reviewed SHA-256 binding is malformed")
    return _read_exact_log_line(
        root, RUN146_LOG_RELATIVE, expected_sha256, 146, "Run 146 log line",
    )


def _read_run148_log(root: Path) -> str:
    return _read_exact_log_line(
        root, RUN148_LOG_RELATIVE, RUN148_RAW_LINE_SHA256, 148, "Run 148 log line",
    )


def _expected_plan(source_sha256: str) -> Dict[str, Any]:
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_recovery_v3_plan_not_authorization",
        "bound_failure": {
            "recovery_v1_execution_intent": {
                "path": RECOVERY_V1_INTENT_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V1_INTENT_SHA256,
            },
            "recovery_v1_postoutput_failure_receipt": {
                "path": RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_SHA256,
            },
            "recovery_v1_plan": {
                "path": RECOVERY_V1_PLAN_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V1_PLAN_SHA256,
            },
            "recovery_v1_sealed_output": {
                "eligible_for_reuse": False,
                "path": RECOVERY_V1_OUTPUT_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V1_OUTPUT_SHA256,
            },
            "recovery_v1_source": {
                "git_commit": RECOVERY_V1_SOURCE_GIT_COMMIT,
                "path": RECOVERY_V1_SOURCE_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V1_SOURCE_SHA256,
            },
            "recovery_v2_execution_intent": {
                "path": RECOVERY_V2_INTENT_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V2_INTENT_SHA256,
            },
            "recovery_v2_prearchive_failure_receipt": {
                "exact_json": RECOVERY_V2_FAILURE_RECEIPT,
                "path": RECOVERY_V2_FAILURE_RECEIPT_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V2_FAILURE_RECEIPT_SHA256,
            },
            "recovery_v2_plan": {
                "path": RECOVERY_V2_PLAN_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V2_PLAN_SHA256,
            },
            "recovery_v2_output": {
                "must_be_absent": True,
                "path": RECOVERY_V2_OUTPUT_RELATIVE.as_posix(),
            },
            "recovery_v2_source": {
                "git_commit": RECOVERY_V2_SOURCE_GIT_COMMIT,
                "path": RECOVERY_V2_SOURCE_RELATIVE.as_posix(),
                "raw_sha256": RECOVERY_V2_SOURCE_SHA256,
            },
            "run_146_log": {
                "path": RUN146_LOG_RELATIVE.as_posix(),
                "raw_line_sha256": RUN146_RAW_LINE_SHA256,
                "semantic_subset": {"run": 146},
            },
            "run_148_log": {
                "path": RUN148_LOG_RELATIVE.as_posix(),
                "raw_line_sha256": RUN148_RAW_LINE_SHA256,
                "semantic_subset": {"run": 148},
            },
            "v1_plan": {
                "path": V1_PLAN_RELATIVE.as_posix(), "raw_sha256": V1_PLAN_SHA256,
            },
            "v1_source": {
                "path": V1_SOURCE_RELATIVE.as_posix(), "raw_sha256": V1_SOURCE_SHA256,
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
            "external_execution_intent": RECOVERY_V3_INTENT_RELATIVE.as_posix(),
            "execute_flag": "--execute",
            "output_receipt": OUTPUT_RELATIVE.as_posix(),
            "receipt_creation": "O_EXCL; one separately named receipt only",
            "reviewed_binding": "A separately created exact-schema intent must bind committed Git HEAD, recovery-v3 source SHA-256, recovery-v3 plan SHA-256, both corrected archive SHA-256 values, recovery-v1 terminal output/failure SHA-256 values, recovery-v2 source/plan/intent/failure SHA-256 values, the corrected Run 146 raw-line SHA-256, and the Run 148 raw-line SHA-256.",
        },
        "metadata_resolution_limit": "Metadata-only qualification cannot establish protonation feasibility, support feasibility, or science feasibility.",
        "recovery_source": {
            "path": CANDIDATE_RELATIVE.as_posix(), "raw_sha256": source_sha256,
        },
        "repair": {
            "frozen_v1_fallback": "openmm86_unique_assigned_ph_v1._fallback_entity",
            "only_change": "pass {(bmrb, loop): prior[(bmrb, loop)] for loop in v1.LOOPS} for each held entity",
            "preserve_closed_capabilities": True,
            "preserve_independently_derived_held_identity_count": 20,
            "preserve_v1_fallback_semantics": True,
        },
        "recovery_v6_result": {
            "hold_identity_count": 20, "metadata_resolved_entity_count": 115,
        },
        "state": "HOLD_ONLY_STATIC_RECOVERY",
        "warnings": [
            "Recovery-v1 sealed output and recovery-v2 terminal failure are binding-only and are never reused as evidence.",
            "The corrected Run 146 binding hashes the raw newline-terminated JSONL line, never a line-number-prefixed shell rendering.",
            "The frozen v1 fallback is invoked unchanged with only its required entity-local three-loop mapping.",
            "No metadata result is protonation, support, science, score, target, or authorization evidence.",
        ],
    }

def _validate_plan(plan: Any, source_sha256: str) -> None:
    if not _exact(plan, _expected_plan(source_sha256)):
        _fail("recovery-v3 plan schema, identities, or bindings drifted")


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
        _fail("recovery-v3 source path is indirect or misplaced")
    return root


def _validate_recovery_v3_intent(
    intent: Any, revision: str, source_sha256: str, plan_sha256: str,
) -> Dict[str, str]:
    expected = {
        "artifact_kind": "hold_only_unique_assigned_ph_recovery_v3_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": plan_sha256,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "recovery_v1_postoutput_failure_receipt_raw_sha256": RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_SHA256,
        "recovery_v1_sealed_output_raw_sha256": RECOVERY_V1_OUTPUT_SHA256,
        "recovery_v2_failure_receipt_raw_sha256": RECOVERY_V2_FAILURE_RECEIPT_SHA256,
        "recovery_v2_intent_raw_sha256": RECOVERY_V2_INTENT_SHA256,
        "recovery_v2_output_receipt_absent": True,
        "recovery_v2_plan_raw_sha256": RECOVERY_V2_PLAN_SHA256,
        "recovery_v2_source_git_commit": RECOVERY_V2_SOURCE_GIT_COMMIT,
        "recovery_v2_source_raw_sha256": RECOVERY_V2_SOURCE_SHA256,
        "run_146_raw_line_sha256": RUN146_RAW_LINE_SHA256,
        "run_148_raw_line_sha256": RUN148_RAW_LINE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    if not _exact(intent, expected):
        _fail("recovery-v3 execution intent schema or values drifted")
    return {
        "recovery_v1_postoutput_failure_receipt_raw_sha256": RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_SHA256,
        "recovery_v1_sealed_output_raw_sha256": RECOVERY_V1_OUTPUT_SHA256,
        "recovery_v2_failure_receipt_raw_sha256": RECOVERY_V2_FAILURE_RECEIPT_SHA256,
        "recovery_v2_intent_raw_sha256": RECOVERY_V2_INTENT_SHA256,
        "recovery_v2_plan_raw_sha256": RECOVERY_V2_PLAN_SHA256,
        "recovery_v2_source_git_commit": RECOVERY_V2_SOURCE_GIT_COMMIT,
        "recovery_v2_source_raw_sha256": RECOVERY_V2_SOURCE_SHA256,
        "run_146_raw_line_sha256": RUN146_RAW_LINE_SHA256,
        "run_148_raw_line_sha256": RUN148_RAW_LINE_SHA256,
    }

def _prearchive_validation(root: Path) -> Dict[str, str]:
    """Bind every predecessor and terminal failure before either ZIP is opened."""
    if (
        CHECKER_PRIOR_ARCHIVE_SHA256 != PRIOR_ARCHIVE_SHA256
        or CHECKER_V6_ARCHIVE_SHA256 != V6_ARCHIVE_SHA256
        or tuple(v1.LOOPS) != (
            "Chem_shift_experiment", "Experiment", "Sample_condition_variable"
        )
    ):
        _fail("frozen fallback loop vocabulary or corrected archive bindings drifted")

    _read_bound_regular(root, PARENT_PLAN_RELATIVE, PARENT_PLAN_SHA256, "parent plan")
    _read_bound_regular(root, V1_SOURCE_RELATIVE, V1_SOURCE_SHA256, "v1 source")
    _read_bound_regular(root, V1_PLAN_RELATIVE, V1_PLAN_SHA256, "v1 plan")
    _validate_recovery_v1_source(root)
    _read_bound_regular(root, RECOVERY_V1_PLAN_RELATIVE, RECOVERY_V1_PLAN_SHA256, "recovery-v1 plan")
    _read_bound_regular(root, SEMANTIC_CHECKER_RELATIVE, SEMANTIC_CHECKER_SHA256, "semantic checker")
    _read_bound_regular(
        root, V6_EVIDENCE_RECEIPT_RELATIVE, V6_EVIDENCE_RECEIPT_SHA256,
        "recovery-v6 evidence receipt",
    )

    v1_intent_raw = _read_bound_regular(
        root, RECOVERY_V1_INTENT_RELATIVE, RECOVERY_V1_INTENT_SHA256,
        "recovery-v1 execution intent",
    )
    _validate_recovery_v1_intent(_loads(v1_intent_raw, "recovery-v1 execution intent"))

    # Bind recovery-v2's source and terminal prearchive receipt as immutable
    # admission evidence.  Its output directory must not exist: it failed
    # before archive opening and never produced a reusable receipt.
    _validate_recovery_v2_source(root)
    _read_bound_regular(root, RECOVERY_V2_PLAN_RELATIVE, RECOVERY_V2_PLAN_SHA256, "recovery-v2 plan")
    _read_bound_regular(root, RECOVERY_V2_INTENT_RELATIVE, RECOVERY_V2_INTENT_SHA256, "recovery-v2 execution intent")
    v2_failure_raw = _read_bound_regular(
        root, RECOVERY_V2_FAILURE_RECEIPT_RELATIVE, RECOVERY_V2_FAILURE_RECEIPT_SHA256,
        "recovery-v2 prearchive failure receipt",
    )
    _validate_recovery_v2_failure_receipt(
        _loads(v2_failure_raw, "recovery-v2 prearchive failure receipt")
    )
    _require_absent(root, RECOVERY_V2_OUTPUT_RELATIVE.parent, "recovery-v2 output")
    run148_sha256 = _read_run148_log(root)

    source_raw = _read_regular(root, CANDIDATE_RELATIVE, "recovery-v3 source", MAX_REGULAR_BYTES)
    source_sha256 = _sha256(source_raw)
    plan_raw = _read_regular(root, PLAN_RELATIVE, "recovery-v3 plan", MAX_REGULAR_BYTES)
    plan_sha256 = _sha256(plan_raw)
    _validate_plan(_loads(plan_raw, "recovery-v3 plan"), source_sha256)

    revision = _git(root, ("git", "rev-parse", "HEAD")).decode("ascii").strip()
    if COMMIT_RE.fullmatch(revision) is None:
        _fail("committed Git HEAD is malformed")
    for relative, current, label in (
        (CANDIDATE_RELATIVE, source_raw, "recovery-v3 source"),
        (PLAN_RELATIVE, plan_raw, "recovery-v3 plan"),
    ):
        committed = _git(root, ("git", "show", "%s:%s" % (revision, relative.as_posix())))
        if committed != current:
            _fail("%s bytes differ from committed Git HEAD" % label)

    intent_raw = _read_regular(
        root, RECOVERY_V3_INTENT_RELATIVE, "recovery-v3 execution intent", MAX_REGULAR_BYTES
    )
    terminal_bindings = _validate_recovery_v3_intent(
        _loads(intent_raw, "recovery-v3 execution intent"), revision, source_sha256, plan_sha256
    )
    sealed_output_sha256 = terminal_bindings["recovery_v1_sealed_output_raw_sha256"]
    _read_bound_regular(
        root, RECOVERY_V1_OUTPUT_RELATIVE, sealed_output_sha256,
        "recovery-v1 sealed output",
    )
    # The old failure receipt remains an exact terminal artifact, while the
    # active v3 gate uses the corrected raw, newline-terminated Run 146 bytes.
    run146_sha256 = _read_run146_log(root, terminal_bindings["run_146_raw_line_sha256"])
    failure_sha256 = terminal_bindings[
        "recovery_v1_postoutput_failure_receipt_raw_sha256"
    ]
    failure_raw = _read_bound_regular(
        root, RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_RELATIVE, failure_sha256,
        "recovery-v1 post-output failure receipt",
    )
    _validate_postoutput_failure_receipt(
        _loads(failure_raw, "recovery-v1 post-output failure receipt"),
        sealed_output_sha256,
    )
    return {
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "execution_intent_path": RECOVERY_V3_INTENT_RELATIVE.as_posix(),
        "execution_intent_raw_sha256": _sha256(intent_raw),
        "plan_raw_sha256": plan_sha256,
        "recovery_v1_postoutput_failure_receipt_raw_sha256": failure_sha256,
        "recovery_v1_sealed_output_raw_sha256": sealed_output_sha256,
        "recovery_v2_failure_receipt_raw_sha256": terminal_bindings[
            "recovery_v2_failure_receipt_raw_sha256"
        ],
        "recovery_v2_intent_raw_sha256": terminal_bindings[
            "recovery_v2_intent_raw_sha256"
        ],
        "recovery_v2_plan_raw_sha256": terminal_bindings["recovery_v2_plan_raw_sha256"],
        "recovery_v2_source_git_commit": terminal_bindings[
            "recovery_v2_source_git_commit"
        ],
        "recovery_v2_source_raw_sha256": terminal_bindings[
            "recovery_v2_source_raw_sha256"
        ],
        "run_146_raw_line_sha256": run146_sha256,
        "run_148_raw_line_sha256": run148_sha256,
    }

def _receipt(
    records: Sequence[Dict[str, Any]], provenance: Mapping[str, str],
) -> Dict[str, Any]:
    recovered = sum(1 for item in records if item.get("fallback_metadata_resolved") is True)
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_recovery_v3_receipt",
        "bound_inputs": {
            "parent_plan_raw_sha256": PARENT_PLAN_SHA256,
            "prior_archive": {
                "path": ".auto/staging/" + PRIOR_NAME, "raw_sha256": PRIOR_ARCHIVE_SHA256,
            },
            "recovery_v1_postoutput_failure_receipt_raw_sha256": provenance[
                "recovery_v1_postoutput_failure_receipt_raw_sha256"
            ],
            "recovery_v1_sealed_output_raw_sha256": provenance[
                "recovery_v1_sealed_output_raw_sha256"
            ],
            "recovery_v2_failure_receipt_raw_sha256": provenance[
                "recovery_v2_failure_receipt_raw_sha256"
            ],
            "recovery_v2_intent_raw_sha256": provenance[
                "recovery_v2_intent_raw_sha256"
            ],
            "recovery_v2_plan_raw_sha256": provenance["recovery_v2_plan_raw_sha256"],
            "recovery_v2_source_git_commit": provenance[
                "recovery_v2_source_git_commit"
            ],
            "recovery_v2_source_raw_sha256": provenance[
                "recovery_v2_source_raw_sha256"
            ],
            "recovery_v6_archive": {
                "path": ".auto/staging/" + V6_NAME, "raw_sha256": V6_ARCHIVE_SHA256,
            },
            "recovery_v6_evidence_receipt_raw_sha256": V6_EVIDENCE_RECEIPT_SHA256,
            "recovery_v6_semantic_checker_raw_sha256": SEMANTIC_CHECKER_SHA256,
            "run_146_raw_line_sha256": provenance["run_146_raw_line_sha256"],
            "run_148_raw_line_sha256": provenance["run_148_raw_line_sha256"],
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
        "metadata_resolution_limit": "Metadata-only qualification cannot establish protonation feasibility, support feasibility, or science feasibility.",
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


def _write_json_once(directory: Path, name: str, payload: Dict[str, Any]) -> None:
    """Write one deterministic receipt using O_EXCL, without retry semantics."""
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = directory / name
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


def _write_receipt_once(root: Path, payload: Dict[str, Any]) -> None:
    staging = _staging_directory(root)
    output = staging / OUTPUT_RELATIVE.parent.name
    try:
        os.mkdir(str(output), 0o700)
    except FileExistsError as error:
        raise RecoveryError("recovery-v3 output already exists; no in-place retry") from error
    details = os.lstat(output)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        _fail("new recovery-v3 output is indirect")
    _write_json_once(output, "receipt.json", payload)


def execute(root: Path) -> Dict[str, Any]:
    """Perform the one HOLD-only repair only after every pre-archive gate passes."""
    provenance = _prearchive_validation(root)

    # No archive is opened before _prearchive_validation returns.  These are
    # the same independently derived recovery-v6 identities used by v1.
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
    held = {
        (item["bmrb_id"], item["entity_uid"])
        for item in derived
        if item.get("ph_feasible") is False
    }
    if (
        len(roster) != 135
        or len(derived) != 135
        or len(held) != 20
        or sum(1 for item in derived if item.get("ph_feasible") is True) != 115
        or not held.issubset(set(roster))
    ):
        _fail("independently validated recovery-v6 HOLD identities drifted")

    # This is the sole repair.  The imported frozen v1 implementation receives
    # exactly the three loops for the current entity, never the 135-entry map.
    records = []
    for bmrb, uid in roster:
        if (bmrb, uid) not in held:
            continue
        entity_prior = {(bmrb, loop): prior[(bmrb, loop)] for loop in v1.LOOPS}
        records.append(v1._fallback_entity(bmrb, uid, entity_prior, assigned[bmrb]))
    if (
        len(records) != 20
        or {(item.get("bmrb_id"), item.get("entity_uid")) for item in records} != held
    ):
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
    """Synthetic, in-memory checks only; this function never reads .auto."""
    checks = 0
    if v1.self_test() != 12:
        raise AssertionError("frozen v1 fallback semantic check count drifted")
    checks += 12

    bmrb, uid = "bmr1", "bmrb:1:entity:1"
    chem = {name: None for name in v1.CHEM_TAGS}
    chem["Assigned_chem_shift_list_ID"] = "9"
    condition = {name: None for name in v1.CONDITION_TAGS}
    condition.update({
        "Sample_condition_list_ID": "7", "Type": "pH", "Val": "7.0", "Val_units": "pH",
    })
    local_prior = {
        (bmrb, "Chem_shift_experiment"): [chem],
        (bmrb, "Experiment"): [],
        (bmrb, "Sample_condition_variable"): [condition],
    }
    full_prior = dict(local_prior)
    full_prior.update({("bmr2", loop): [] for loop in v1.LOOPS})
    assigned = [{
        v1.ASSIGNED_TAGS[0]: "1", v1.ASSIGNED_TAGS[1]: "9", v1.ASSIGNED_TAGS[2]: "7",
    }]
    entity_prior = {(bmrb, loop): full_prior[(bmrb, loop)] for loop in v1.LOOPS}
    result = v1._fallback_entity(bmrb, uid, entity_prior, assigned)
    if not (result["fallback_metadata_resolved"] is True and result["fallback_ph"] == 7.0):
        raise AssertionError("entity-local frozen-v1 fallback did not resolve synthetic pH")
    if v1._fallback_entity(bmrb, uid, full_prior, assigned)["fallback_metadata_resolved"]:
        raise AssertionError("synthetic full mapping did not demonstrate the repaired defect")
    checks += 2

    _validate_recovery_v1_intent(RECOVERY_V1_INTENT)
    intent_extra = dict(RECOVERY_V1_INTENT)
    intent_extra["unexpected"] = False
    _expect_recovery_error(
        lambda: _validate_recovery_v1_intent(intent_extra), "recovery-v1 intent unknown field"
    )
    checks += 2

    _validate_recovery_v2_failure_receipt(RECOVERY_V2_FAILURE_RECEIPT)
    v2_failure_extra = dict(RECOVERY_V2_FAILURE_RECEIPT)
    v2_failure_extra["unexpected"] = False
    _expect_recovery_error(
        lambda: _validate_recovery_v2_failure_receipt(v2_failure_extra),
        "recovery-v2 prearchive receipt unknown field",
    )
    if (
        SHA256_RE.fullmatch(RUN146_RAW_LINE_SHA256) is None
        or SHA256_RE.fullmatch(RUN148_RAW_LINE_SHA256) is None
        or RUN146_RAW_LINE_SHA256 == HISTORICAL_RUN146_PREFIXED_LINE_SHA256
    ):
        raise AssertionError("corrected Run 146 or Run 148 full SHA-256 binding drifted")
    checks += 3

    # The positive case has exactly raw JSONL bytes and the trailing newline.
    # A grep -n rendering is neither those bytes nor their SHA-256 preimage.
    raw_run146_line = b'{"run":146,"state":"TERMINAL"}\n'
    raw_run146_sha256 = _sha256(raw_run146_line)
    _validate_exact_raw_log_line(
        raw_run146_line, raw_run146_sha256, 146, "synthetic corrected Run 146 line"
    )
    _expect_recovery_error(
        lambda: _validate_exact_raw_log_line(
            b"1:" + raw_run146_line, raw_run146_sha256, 146,
            "line-number-prefixed Run 146 variant",
        ),
        "line-number-prefixed Run 146 variant",
    )
    _expect_recovery_error(
        lambda: _validate_exact_raw_log_line(
            raw_run146_line[:-1], _sha256(raw_run146_line[:-1]), 146,
            "unterminated Run 146 variant",
        ),
        "unterminated Run 146 variant",
    )
    checks += 3

    recovery_v3_intent = {
        "artifact_kind": "hold_only_unique_assigned_ph_recovery_v3_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": "a" * 40,
        "candidate_source_raw_sha256": "b" * 64,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": "c" * 64,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "recovery_v1_postoutput_failure_receipt_raw_sha256": RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_SHA256,
        "recovery_v1_sealed_output_raw_sha256": RECOVERY_V1_OUTPUT_SHA256,
        "recovery_v2_failure_receipt_raw_sha256": RECOVERY_V2_FAILURE_RECEIPT_SHA256,
        "recovery_v2_intent_raw_sha256": RECOVERY_V2_INTENT_SHA256,
        "recovery_v2_output_receipt_absent": True,
        "recovery_v2_plan_raw_sha256": RECOVERY_V2_PLAN_SHA256,
        "recovery_v2_source_git_commit": RECOVERY_V2_SOURCE_GIT_COMMIT,
        "recovery_v2_source_raw_sha256": RECOVERY_V2_SOURCE_SHA256,
        "run_146_raw_line_sha256": RUN146_RAW_LINE_SHA256,
        "run_148_raw_line_sha256": RUN148_RAW_LINE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    bindings = _validate_recovery_v3_intent(
        recovery_v3_intent, "a" * 40, "b" * 64, "c" * 64
    )
    if bindings["recovery_v2_failure_receipt_raw_sha256"] != RECOVERY_V2_FAILURE_RECEIPT_SHA256:
        raise AssertionError("recovery-v3 terminal binding extraction drifted")
    malformed_v3_intent = dict(recovery_v3_intent)
    malformed_v3_intent["run_146_raw_line_sha256"] = "0" * 64
    _expect_recovery_error(
        lambda: _validate_recovery_v3_intent(
            malformed_v3_intent, "a" * 40, "b" * 64, "c" * 64
        ),
        "recovery-v3 corrected Run 146 binding",
    )
    checks += 3

    failure = dict(RECOVERY_V1_POSTOUTPUT_FAILURE_RECEIPT_FIXED)
    _validate_postoutput_failure_receipt(failure, RECOVERY_V1_OUTPUT_SHA256)
    failure_extra = dict(failure)
    failure_extra["unexpected"] = True
    _expect_recovery_error(
        lambda: _validate_postoutput_failure_receipt(
            failure_extra, RECOVERY_V1_OUTPUT_SHA256
        ),
        "post-output receipt unknown field",
    )
    checks += 2

    synthetic_receipt = {
        "closed_capabilities": CLOSED_CAPABILITIES,
        "output": {"receipt_creation": "O_EXCL"},
        "status": "HOLD_METADATA_REINTERPRETATION_ONLY",
    }
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _write_json_once(directory, "receipt.json", synthetic_receipt)
        if _loads((directory / "receipt.json").read_bytes(), "synthetic receipt") != synthetic_receipt:
            raise AssertionError("synthetic O_EXCL receipt schema drifted")
        try:
            _write_json_once(directory, "receipt.json", synthetic_receipt)
        except FileExistsError:
            pass
        else:
            raise AssertionError("O_EXCL receipt overwrite was accepted")
        _require_absent(directory, Path("absent-output"), "synthetic absent output")
        (directory / "present-output").mkdir()
        _expect_recovery_error(
            lambda: _require_absent(directory, Path("present-output"), "synthetic present output"),
            "present recovery-v2 output",
        )
    checks += 3

    if (
        CLOSED_CAPABILITIES != {key: False for key in CLOSED_CAPABILITIES}
        or CANDIDATE_ID.endswith("recovery_v2")
        or OUTPUT_RELATIVE.parent == RECOVERY_V2_OUTPUT_RELATIVE.parent
    ):
        raise AssertionError("recovery-v3 capability, identity, or output drift")
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
        print("METRIC openmm86_unique_assigned_ph_v1_recovery_v3_v1_semantic_checks=12")
        print("METRIC openmm86_unique_assigned_ph_v1_recovery_v3_synthetic_checks=%d" % checks)
        print("METRIC target_values_read=0")
        print("METRIC source_scores_read=0")
        print("METRIC science_executed=0")
        print("METRIC authorization_consumed=0")
        print("STATUS HOLD_SELF_TEST_ONLY")
        return 0
    try:
        payload = execute(_canonical_root())
    except (RecoveryError, OSError, ValueError) as error:
        print("REFUSAL HOLD-only recovery-v3: %s" % error)
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
