#!/usr/bin/env python3
# ruff: noqa: UP045, TRY004
"""Independently replay the fixed local deposited-pH recovery-v3 output.

This is a local-consistency checker only.  It reads fixed repository paths,
never imports or executes the producer, and does not fetch data.  A local PASS
or HOLD remains HOLD pending a separately exact-bound immutable archive and
execution receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import urllib.parse
import zipfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v3.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v3_plan.json"
)
PRODUCER_RELATIVE = Path(
    "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v3.py"
)
ROSTER_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
)
ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip"
)
V1_FAILURE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v1"
)
ARTIFACT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v3_recovery"
)
RECEIPT_RELATIVE = ARTIFACT_RELATIVE / "receipt.json"
RAW_DIRECTORY = "raw_api_responses"
START_MARKER = "started_at_utc.txt"

PLAN_RAW_SHA256 = "f8eadb6a4dde415d1af016c2843f867ee32e9546a7e4a272ff211fbfe531ba1e"
PRODUCER_RAW_SHA256 = "8c2549c18b60e6eb69812102a14f40e063458540ef0abf9191bdf7c3f4a59a49"
ROSTER_SHA256 = "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
ARCHIVE_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"

CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v3"
CONTRACT = "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v3"
PLAN_CONTRACT = "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v3_plan"
API_BASE = "https://api.bmrb.io/v2"
APPLICATION_HEADER = "AtypEmu openmm86-deposited-ph-catalog-recovery-v3"
ENTITY_COUNT = 135
MAX_SOURCE_BYTES = 2_000_000
MAX_JSON_BYTES = 10_000_000
MAX_ARCHIVE_BYTES = 100_000_000
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 500_000_000
MAX_LOG_BYTES = 5_000_000
MAX_LOG_LINES = 10_000
V4_RAW_PREFIX = (
    "atypemu_nested_support_count_v1_solution_conditions_api_v2_v4_recovery/"
    "raw_api_responses"
)
PRIOR_LOOPS = (
    "Chem_shift_experiment",
    "Experiment",
    "Sample_condition_variable",
)
NEW_LOOP = "Assigned_chem_shift_list"
ALL_LOOPS = PRIOR_LOOPS + (NEW_LOOP,)
MISSING = frozenset((None, "", ".", "?"))
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
ENTITY_RE = re.compile(r"bmrb:([1-9][0-9]*):entity:1\Z")
ID_RE = re.compile(r"[1-9][0-9]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

CLOSED_FLAGS = {
    "target_values_read": False,
    "target_atom_identities_read": False,
    "source_scores_read": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "authorization_consumed": False,
    "source_construction_executed": False,
}

# Copied from the raw-byte-bound recovery-v3 plan.  These are fixed local
# admission facts, not caller-selected evidence locations.
V2_OUTPUT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v2_recovery"
)
V2_LAUNCH_INTENT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v2_launch_intent.json"
)
LOG_RELATIVE = Path(".auto/log.jsonl")
V1_STARTED_AT_UTC = "2026-09-06T04:46:51.599115Z"
V1_FAILURE_RECEIPT = {
    "artifact_kind": "target_unread_openmm86_deposited_ph_catalog_failure_receipt",
    "candidate_id": "atypemu_nested_support_count_v1_openmm86_deposited_ph_v1",
    "contract": "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_v1",
    "error": "ValueError: bound roster raw SHA-256 drifted",
    "new_response_count_written": 0,
    "new_response_manifest": [],
    "started_at_utc": V1_STARTED_AT_UTC,
    **CLOSED_FLAGS,
    "plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1.json",
        "sha256": "e3bc877da2c3a0c3cac90399a99aac93817bd16646f29abc15ed575b0a95d913",
        "git_commit": "3d3eb017106729bbbf7539436cf1a799b99b9328",
    },
    "source_producer_relative_path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog.py",
    "source_producer_sha256": "1a449a2e966347d5bd607dcb22c5b52a59cfb2034aa6db49ae295232ed38f0ef",
    "source_producer_git_commit": "3d3eb017106729bbbf7539436cf1a799b99b9328",
}
V1_FAILURE_BINDING = {
    "failure_directory": V1_FAILURE_RELATIVE.as_posix(),
    "failure_inventory": {
        "regular_files": ["failure_receipt.json", "started_at_utc.txt"],
        "directories": [{"path": "raw_api_responses", "inventory": []}],
    },
    "failure_receipt": {
        "path": "failure_receipt.json",
        "sha256": "a8cbd988789c6bde3e774d625ab4e6583476dfd7ae1a1f4a641b9548edbda8c5",
        "exact_fields": V1_FAILURE_RECEIPT,
    },
    "started_at_utc": {
        "path": "started_at_utc.txt",
        "sha256": "d741789e11f995c3c034152fd7e1594a34faecdf6dd970ce4bc73309fee29141",
        "value": V1_STARTED_AT_UTC,
    },
}
V2_LAUNCH_INTENT = {
    "artifact_kind": "target_unread_metadata_fetch_launch_intent_not_authorization",
    "candidate_id": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v2",
    "contract": "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v2_launch_v1",
    "git_commit": "7ce913df82b8d9a2b32ecac7c4bebcd2f72d9041",
    "plan": {
        "path": "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v2_plan.json",
        "sha256": "5d6547c21cdf4c1dc6380340375084c53de0e137f10cee47f74f6683150227e3",
    },
    "producer": {
        "path": "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v2.py",
        "sha256": "aad9a1b2f20a46853b2b78dfcf46f85b2912ac8b67540dec8492764e74d1e4ae",
    },
    "checker": {
        "path": "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog_recovery_v2.py",
        "sha256": "d43093869b37f29a521cdc102008a10ab87da933a12cc9417d5d8160ab79842f",
    },
    "v1_failure_receipt_sha256": "a8cbd988789c6bde3e774d625ab4e6583476dfd7ae1a1f4a641b9548edbda8c5",
    "output": V2_OUTPUT_RELATIVE.as_posix(),
    "target_values_read": False,
    "target_atom_identities_read": False,
    "source_scores_read": False,
    "science_executed": False,
    "source_construction_executed": False,
    "authorization_consumed": False,
}
V2_LAUNCH_INTENT_BINDING = {
    "path": V2_LAUNCH_INTENT_RELATIVE.as_posix(),
    "sha256": "ffdcebcaa4cf73d1e6b3d0834c8775836ba2c0f14edcd1bd10f21c2e5ab7aa9f",
    "exact_fields": V2_LAUNCH_INTENT,
}
RUN122_LOG_BINDING = {
    "path": LOG_RELATIVE.as_posix(),
    "raw_line_sha256": "df01b57177613b0779b6b3519e43112ae021c29c58b5f8ac83a929c695402389",
    "semantic_subset": {
        "run": 122,
        "commit": "7ce913d",
        "metric": 0,
        "status": "crash",
        "description": "Run 122: recovery-v2 launch failed closed on omitted empty raw-response directory",
        "asi_required_fields": ["error", "root_cause", "failure_stage", "observed_controls"],
        "observed_controls": "all leaves are false or integer zero",
    },
}
PRE_OUTPUT_GATES = {
    "actual_v1_failure_tree": V1_FAILURE_BINDING,
    "active_v2_launch_intent": V2_LAUNCH_INTENT_BINDING,
    "run_122_log_record": RUN122_LOG_BINDING,
    "v2_output_must_be_absent": V2_OUTPUT_RELATIVE.as_posix(),
}
RECOVERY_EVIDENCE = {
    "v1_failure_tree": V1_FAILURE_BINDING,
    "v2_launch_intent": V2_LAUNCH_INTENT_BINDING,
    "run_122_log_record": RUN122_LOG_BINDING,
    "v2_output_absent_before_v3_creation": V2_OUTPUT_RELATIVE.as_posix(),
}

REQUIRED_TAGS = {
    "Chem_shift_experiment": frozenset(
        ("Entry_ID", "Experiment_ID", "Assigned_chem_shift_list_ID")
    ),
    "Experiment": frozenset(
        ("Entry_ID", "ID", "Sample_ID", "Sample_condition_list_ID")
    ),
    "Sample_condition_variable": frozenset(
        ("Entry_ID", "Type", "Val", "Val_units", "Sample_condition_list_ID")
    ),
    "Assigned_chem_shift_list": frozenset(
        ("Entry_ID", "ID", "Sample_condition_list_ID")
    ),
}
ALLOWED_TAGS = {
    "Chem_shift_experiment": frozenset(
        (
            "Experiment_ID", "Experiment_name", "Sample_ID", "Sample_label",
            "Sample_state", "Entry_ID", "Assigned_chem_shift_list_ID",
        )
    ),
    "Experiment": frozenset(
        (
            "ID", "Name", "Raw_data_flag", "NUS_flag", "Interleaved_flag",
            "NMR_spec_expt_ID", "NMR_spec_expt_label", "MS_expt_ID",
            "MS_expt_label", "SAXS_expt_ID", "SAXS_expt_label", "FRET_expt_ID",
            "FRET_expt_label", "EMR_expt_ID", "EMR_expt_label", "Sample_ID",
            "Sample_label", "Sample_state", "Sample_volume", "Sample_volume_units",
            "Sample_condition_list_ID", "Sample_condition_list_label",
            "Sample_spinning_rate", "Sample_angle", "NMR_tube_type",
            "NMR_spectrometer_ID", "NMR_spectrometer_label",
            "NMR_spectrometer_probe_ID", "NMR_spectrometer_probe_label",
            "NMR_spectral_processing_ID", "NMR_spectral_processing_label",
            "Mass_spectrometer_ID", "Mass_spectrometer_label", "Xray_instrument_ID",
            "Xray_instrument_label", "Fluorescence_instrument_ID",
            "Fluorescence_instrument_label", "EMR_instrument_ID",
            "EMR_instrument_label", "Chromatographic_system_ID",
            "Chromatographic_system_ID", "Chromatographic_system_label", "Details",
            "Entry_ID", "Experiment_list_ID",
        )
    ),
    "Sample_condition_variable": frozenset(
        ("Type", "Val", "Val_err", "Val_units", "Entry_ID", "Sample_condition_list_ID")
    ),
    "Assigned_chem_shift_list": REQUIRED_TAGS["Assigned_chem_shift_list"],
}

ROSTER_FIELDS = frozenset(
    (
        "artifact_kind", "authorization_consumed", "contract", "entities",
        "entity_count", "outer_or_formal_metrics_opened",
        "source_commitment_relative_path", "source_commitment_sha256",
        "source_scores_read", "study_id", "target_values_read",
    )
)
ROSTER_ENTITY_FIELDS = frozenset(
    (
        "bmrb_id", "canonical_all_atom_topology_sha256", "canonical_atom_count",
        "canonical_heavy_atom_count", "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256", "canonical_reference_relative_path",
        "canonical_reference_support_index", "entity_uid", "observer_fold", "split",
    )
)
RECEIPT_FIELDS = frozenset(
    (
        "artifact_kind", "candidate_id", "contract", "recovery_evidence",
        "metadata_scope", "api_base", "application_header", "archive", "roster",
        "plan", "started_at_utc", "ended_at_utc", "new_response_count",
        "new_response_manifest", "entity_count", "ph_feasible_entity_count",
        "all_entities_ph_feasible", "entities", "status", "future_consumer",
        "warnings", "target_values_read", "target_atom_identities_read",
        "source_scores_read", "outer_or_formal_metrics_opened", "science_executed",
        "authorization_consumed", "source_construction_executed",
        "source_producer_relative_path", "source_producer_sha256",
        "source_producer_git_commit",
    )
)
ENTITY_FIELDS = frozenset(
    (
        "entity_uid", "bmrb_id", "route", "complete_chem_shift_experiment_link_count",
        "experiment_ids", "assigned_chem_shift_list_ids", "sample_condition_list_ids",
        "deposited_ph", "deposited_ph_record_count", "temperature_ionic_diagnostics",
        "ph_feasible", "hold_reasons", "prior_archive_response_bindings",
        "new_response_binding",
    )
)
NEW_BINDING_FIELDS = frozenset(("bmrb_id", "loop", "path", "sha256", "url", "final_url"))
PRIOR_BINDING_FIELDS = frozenset(("archive_member", "loop", "sha256"))
DIAGNOSTIC_FIELDS = frozenset(("type", "value", "units"))
FUTURE_CONSUMER = {
    "declarative_only": True,
    "openmm_version": "8.6",
    "method": "Modeller.addHydrogens",
    "force_field_xml": "amber14/protein.ff14SB.xml",
    "pH_source": "deposited pH selected by this receipt",
}
WARNINGS = [
    "heuristic only; not a pKa calculation",
    "heuristic only; not a protonation ensemble calculation",
    "temperature and ionic diagnostics do not establish physicality",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _duplicate_free(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _nonfinite_json(value: str) -> None:
    raise ValueError("non-finite JSON value: %s" % value)


def _decode(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_free,
            parse_constant=_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON in %s: %s" % (label, error)) from error


def _exact(value: Any, expected: Any) -> bool:
    """JSON equality which does not equate False with 0."""
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


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid fixed path: %s" % label)
    pieces = value.split("/")
    if value.startswith("/") or "\\" in value or any(
        piece in ("", ".", "..") for piece in pieces
    ):
        raise ValueError("unsafe fixed path: %s" % label)
    return value


def _regular_path(root: Path, relative: Path, label: str) -> Path:
    text = _safe_relative(relative.as_posix(), label)
    try:
        current = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root is unavailable") from error
    for piece in text.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError("missing %s" % label) from error
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("symlinked %s" % label)
    if not stat.S_ISREG(os.lstat(current).st_mode):
        raise ValueError("non-regular %s" % label)
    return current


def _directory_path(root: Path, relative: Path, label: str) -> Path:
    text = _safe_relative(relative.as_posix(), label)
    try:
        current = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root is unavailable") from error
    for piece in text.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError("missing %s" % label) from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("indirect or non-directory %s" % label)
    return current


def _read_regular(root: Path, relative: Path, label: str, maximum: int) -> bytes:
    path = _regular_path(root, relative, label)
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("non-regular %s" % label)
        if details.st_size < 0 or details.st_size > maximum:
            raise ValueError("%s exceeds the byte limit" % label)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(details.st_size + 1)
        if len(data) != details.st_size:
            raise ValueError("%s changed while being read" % label)
        return data
    finally:
        os.close(descriptor)


def _bound_root() -> Path:
    checker = Path(__file__).absolute()
    root = checker.parents[2]
    expected = root / CHECKER_RELATIVE
    if (
        checker != expected
        or checker.is_symlink()
        or checker.resolve(strict=True) != checker
        or stat.S_ISLNK(os.lstat(root).st_mode)
        or not root.is_dir()
    ):
        raise ValueError("checker is indirect or outside its fixed repository path")
    return root


def _git(root: Path, arguments: Sequence[str], label: str) -> bytes:
    """Read fixed Git metadata/source with the same hard byte limit as files."""
    process: Optional[subprocess.Popen[bytes]] = None
    try:
        process = subprocess.Popen(
            list(arguments),
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        output = process.stdout.read(MAX_SOURCE_BYTES + 1)
        if len(output) > MAX_SOURCE_BYTES:
            process.kill()
            process.wait(timeout=30)
            raise ValueError("committed %s exceeds the byte limit" % label)
        if process.wait(timeout=30) != 0:
            raise ValueError("cannot inspect committed %s" % label)
        return output
    except (OSError, subprocess.TimeoutExpired) as error:
        if process is not None:
            process.kill()
            process.wait()
        raise ValueError("cannot inspect committed %s" % label) from error


def _head(root: Path) -> str:
    head = _git(root, ("git", "rev-parse", "HEAD"), "HEAD").decode("ascii").strip()
    if COMMIT_RE.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full commit ID")
    return head


def _committed_exact(root: Path, relative: Path, digest: str, label: str) -> str:
    current = _read_regular(root, relative, label, MAX_SOURCE_BYTES)
    if _sha256(current) != digest:
        raise ValueError("%s raw SHA-256 drifted" % label)
    head = _head(root)
    committed = _git(root, ("git", "show", "%s:%s" % (head, relative.as_posix())), label)
    if committed != current:
        raise ValueError("%s bytes do not match committed HEAD" % label)
    return head


def _require_committed_checker(root: Path) -> str:
    """The checker's external raw hash is deliberately parent-plan-bound."""
    current = _read_regular(root, CHECKER_RELATIVE, "checker source", MAX_SOURCE_BYTES)
    head = _head(root)
    committed = _git(
        root, ("git", "show", "%s:%s" % (head, CHECKER_RELATIVE.as_posix())), "checker source"
    )
    if current != committed:
        raise ValueError("checker source bytes do not match committed HEAD")
    return head


def _commit_has_bytes(
    root: Path, revision: Any, relative: Path, digest: str, label: str
) -> str:
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise ValueError("invalid %s revision" % label)
    if _git(root, ("git", "cat-file", "-t", revision), label).decode().strip() != "commit":
        raise ValueError("%s revision is not a commit" % label)
    source = _git(root, ("git", "show", "%s:%s" % (revision, relative.as_posix())), label)
    if _sha256(source) != digest:
        raise ValueError("recorded %s commit lacks bound bytes" % label)
    return revision


def _require_ancestor(root: Path, older: str, head: str, label: str) -> None:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, head],
            cwd=str(root),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("recorded %s commit is not reachable from HEAD" % label) from error


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError("invalid SHA-256: %s" % label)
    return value


def _load_plan_and_producer(root: Path) -> tuple[dict[str, Any], str, str]:
    plan_head = _committed_exact(root, PLAN_RELATIVE, PLAN_RAW_SHA256, "recovery-v3 plan")
    producer_head = _committed_exact(
        root, PRODUCER_RELATIVE, PRODUCER_RAW_SHA256, "recovery-v3 producer"
    )
    plan = _decode(
        _read_regular(root, PLAN_RELATIVE, "recovery-v3 plan", MAX_SOURCE_BYTES),
        "recovery-v3 plan",
    )
    if not isinstance(plan, dict):
        raise ValueError("recovery-v3 plan root is not an object")
    inputs = plan.get("bound_inputs")
    output = plan.get("output")
    archive = inputs.get("canonical_condition_archive") if isinstance(inputs, dict) else None
    roster = inputs.get("roster") if isinstance(inputs, dict) else None
    if (
        plan.get("artifact_kind")
        != "hold_only_target_unread_deposited_ph_catalog_recovery_plan_not_authorization"
        or plan.get("candidate_id") != CANDIDATE_ID
        or plan.get("contract") != PLAN_CONTRACT
        or plan.get("state") != "HOLD_PENDING_TARGET_UNREAD_DEPOSITED_PH_CATALOG_RECOVERY"
        or plan.get("closed_capabilities") != CLOSED_FLAGS
        or not isinstance(output, dict)
        or output.get("fresh_directory") != ARTIFACT_RELATIVE.as_posix()
        or output.get("receipt") != "receipt.json"
        or not isinstance(archive, dict)
        or archive.get("path") != ARCHIVE_RELATIVE.as_posix()
        or archive.get("sha256") != ARCHIVE_SHA256
        or archive.get("v4_raw_response_prefix") != V4_RAW_PREFIX
        or not isinstance(roster, dict)
        or roster.get("path") != ROSTER_RELATIVE.as_posix()
        or roster.get("sha256") != ROSTER_SHA256
        or roster.get("entity_count") != ENTITY_COUNT
        or not _exact(plan.get("recovery_of"), V1_FAILURE_BINDING)
        or not _exact(plan.get("pre_output_gates"), PRE_OUTPUT_GATES)
    ):
        raise ValueError("recovery-v3 plan semantic binding drifted")
    return plan, plan_head, producer_head


def _validate_v1_failure(root: Path, binding: Any = V1_FAILURE_BINDING) -> dict[str, Any]:
    """Read the actual v1 tree, including its required empty raw directory."""
    if not _exact(binding, V1_FAILURE_BINDING):
        raise ValueError("recovery-v3 v1 failure binding drifted")
    directory = _directory_path(root, V1_FAILURE_RELATIVE, "sealed v1 failure directory")
    inventory = binding["failure_inventory"]
    expected_files = set(inventory["regular_files"])
    expected_directories = {item["path"]: item["inventory"] for item in inventory["directories"]}
    if (
        len(expected_files) != len(inventory["regular_files"])
        or len(expected_directories) != len(inventory["directories"])
        or expected_files & set(expected_directories)
    ):
        raise ValueError("sealed v1 inventory binding is malformed")
    try:
        with os.scandir(directory) as entries:
            actual_files: set[str] = set()
            actual_directories: set[str] = set()
            for entry in entries:
                if len(actual_files) + len(actual_directories) >= len(expected_files) + len(expected_directories):
                    raise ValueError("sealed v1 failure inventory exceeds its bound")
                if entry.is_symlink():
                    raise ValueError("sealed v1 failure inventory has a symlink")
                if entry.is_file(follow_symlinks=False):
                    actual_files.add(entry.name)
                elif entry.is_dir(follow_symlinks=False):
                    actual_directories.add(entry.name)
                else:
                    raise ValueError("sealed v1 failure inventory has an indirect entry")
    except OSError as error:
        raise ValueError("cannot inspect sealed v1 failure directory") from error
    if actual_files != expected_files or actual_directories != set(expected_directories):
        raise ValueError("sealed v1 failure inventory drifted")
    for name, expected_inventory in expected_directories.items():
        if not isinstance(name, str) or "/" in name or name in ("", ".", "..") or expected_inventory != []:
            raise ValueError("sealed v1 raw-directory binding is malformed")
        raw_directory = directory / name
        try:
            details = os.lstat(raw_directory)
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise ValueError("sealed v1 raw-response directory is indirect")
            with os.scandir(raw_directory) as entries:
                if any(True for _ in entries):
                    raise ValueError("sealed v1 raw-response directory is not exactly empty")
        except OSError as error:
            raise ValueError("cannot inspect sealed v1 raw-response directory") from error
    receipt_spec = binding["failure_receipt"]
    marker_spec = binding["started_at_utc"]
    receipt_raw = _read_regular(
        root, V1_FAILURE_RELATIVE / receipt_spec["path"], "sealed v1 failure receipt", MAX_JSON_BYTES
    )
    marker_raw = _read_regular(
        root, V1_FAILURE_RELATIVE / marker_spec["path"], "sealed v1 start marker", 1024
    )
    if _sha256(receipt_raw) != receipt_spec["sha256"]:
        raise ValueError("sealed v1 failure receipt raw SHA-256 drifted")
    if _sha256(marker_raw) != marker_spec["sha256"]:
        raise ValueError("sealed v1 start marker raw SHA-256 drifted")
    if marker_raw != (marker_spec["value"] + "\n").encode("utf-8"):
        raise ValueError("sealed v1 start marker value drifted")
    if not _exact(_decode(receipt_raw, "sealed v1 failure receipt"), receipt_spec["exact_fields"]):
        raise ValueError("sealed v1 failure receipt schema or values drifted")
    return V1_FAILURE_BINDING


def _validate_v2_launch_intent(
    root: Path, binding: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Validate the fixed v2 intent; an injected binding is synthetic-test-only."""
    fixed = binding is None
    binding = V2_LAUNCH_INTENT_BINDING if binding is None else binding
    if not (
        isinstance(binding, dict)
        and set(binding) == {"path", "sha256", "exact_fields"}
        and binding.get("path") == V2_LAUNCH_INTENT_RELATIVE.as_posix()
        and isinstance(binding.get("sha256"), str)
        and SHA256_RE.fullmatch(binding["sha256"])
        and _exact(binding.get("exact_fields"), V2_LAUNCH_INTENT)
        and (not fixed or _exact(binding, V2_LAUNCH_INTENT_BINDING))
    ):
        raise ValueError("active recovery-v2 launch-intent binding drifted")
    raw = _read_regular(
        root, V2_LAUNCH_INTENT_RELATIVE, "active recovery-v2 launch intent", MAX_SOURCE_BYTES
    )
    if _sha256(raw) != binding["sha256"]:
        raise ValueError("active recovery-v2 launch intent raw SHA-256 drifted")
    if not _exact(_decode(raw, "active recovery-v2 launch intent"), binding["exact_fields"]):
        raise ValueError("active recovery-v2 launch intent schema or values drifted")
    return V2_LAUNCH_INTENT_BINDING if fixed else binding


def _closed_control_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(_closed_control_tree(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_closed_control_tree(item) for item in value)
    return value is False or (type(value) is int and value == 0)


def _validate_run122_log(
    root: Path, binding: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Validate the fixed Run 122 byte; an injected binding is synthetic-test-only."""
    fixed = binding is None
    binding = RUN122_LOG_BINDING if binding is None else binding
    if not (
        isinstance(binding, dict)
        and set(binding) == {"path", "raw_line_sha256", "semantic_subset"}
        and binding.get("path") == LOG_RELATIVE.as_posix()
        and isinstance(binding.get("raw_line_sha256"), str)
        and SHA256_RE.fullmatch(binding["raw_line_sha256"])
        and _exact(binding.get("semantic_subset"), RUN122_LOG_BINDING["semantic_subset"])
        and (not fixed or _exact(binding, RUN122_LOG_BINDING))
    ):
        raise ValueError("Run 122 log binding drifted")
    raw_log = _read_regular(root, LOG_RELATIVE, "append-only operational log", MAX_LOG_BYTES)
    lines = raw_log.splitlines(keepends=True)
    if len(lines) > MAX_LOG_LINES:
        raise ValueError("append-only operational log exceeds the line limit")
    matches: list[tuple[bytes, bytes]] = []
    for line in lines:
        json_raw = line.rstrip(b"\r\n")
        candidates = [line] if json_raw == line else [line, json_raw]
        hit = [candidate for candidate in candidates if candidate and _sha256(candidate) == binding["raw_line_sha256"]]
        if len(hit) > 1:
            raise ValueError("Run 122 raw log-line representation is ambiguous")
        if hit:
            matches.append((hit[0], json_raw))
    if len(matches) != 1:
        raise ValueError("Run 122 exact raw log line is absent or duplicated")
    record = _decode(matches[0][1], "Run 122 log line")
    semantic = binding["semantic_subset"]
    if not isinstance(record, dict) or not (
        type(record.get("run")) is int and record["run"] == semantic["run"]
        and record.get("commit") == semantic["commit"]
        and type(record.get("metric")) is int and record["metric"] == semantic["metric"]
        and record.get("status") == semantic["status"]
        and record.get("description") == semantic["description"]
    ):
        raise ValueError("Run 122 log-line semantic subset drifted")
    required = set(semantic["asi_required_fields"])
    asi_matches: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if required.issubset(value):
                asi_matches.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record)
    if len(asi_matches) != 1:
        raise ValueError("Run 122 ASI schema is absent or ambiguous")
    asi = asi_matches[0]
    if any(not isinstance(asi[name], str) or not asi[name] for name in ("error", "root_cause", "failure_stage")):
        raise ValueError("Run 122 ASI text fields drifted")
    if not isinstance(asi["observed_controls"], dict) or not _closed_control_tree(asi["observed_controls"]):
        raise ValueError("Run 122 ASI observed controls are not all closed")
    return RUN122_LOG_BINDING if fixed else binding


def _require_v2_output_absent(root: Path) -> None:
    """Reject a pre-existing v2 output without following any path component."""
    current = root.resolve(strict=True)
    for index, piece in enumerate(_safe_relative(V2_OUTPUT_RELATIVE.as_posix(), "v2 output").split("/")):
        current = current / piece
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ValueError("cannot inspect recovery-v2 output absence") from error
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("recovery-v2 output absence path is indirect")
        if index == len(V2_OUTPUT_RELATIVE.parts) - 1:
            raise ValueError("recovery-v2 output exists; in-place retry is forbidden")
        if not stat.S_ISDIR(details.st_mode):
            raise ValueError("recovery-v2 output parent is not a directory")


def _validate_pre_output_gates(root: Path) -> dict[str, Any]:
    """Validate every fixed v3 admission fact before any recovery artifact read."""
    evidence = {
        "v1_failure_tree": _validate_v1_failure(root),
        "v2_launch_intent": _validate_v2_launch_intent(root),
        "run_122_log_record": _validate_run122_log(root),
        "v2_output_absent_before_v3_creation": V2_OUTPUT_RELATIVE.as_posix(),
    }
    _require_v2_output_absent(root)
    if not _exact(evidence, RECOVERY_EVIDENCE):
        raise ValueError("recovery-v3 pre-output evidence drifted")
    return evidence


def _missing(value: Any) -> bool:
    return value in MISSING or (isinstance(value, str) and not value.strip())


def _valid_id(value: Any) -> Optional[str]:
    if _missing(value) or isinstance(value, bool):
        return None
    candidate = str(value).strip()
    return candidate if ID_RE.fullmatch(candidate) else None


def _identity(bmrb_id: Any, entity_uid: Any, label: str) -> tuple[str, str]:
    if not isinstance(bmrb_id, str) or not isinstance(entity_uid, str):
        raise ValueError("non-string identity: %s" % label)
    bmrb = BMRB_RE.fullmatch(bmrb_id)
    entity = ENTITY_RE.fullmatch(entity_uid)
    if bmrb is None or entity is None or bmrb.group(1) != entity.group(1):
        raise ValueError("noncanonical identity: %s" % label)
    return bmrb_id, entity_uid


def _rows(payload: bytes, entry_id: str, loop: str) -> list[dict[str, Any]]:
    """Parse one strictly allowlisted metadata category, never a target tag."""
    if loop not in ALL_LOOPS:
        raise ValueError("nonallowlisted response category")
    decoded = _decode(payload, "BMRB %s response" % loop)
    if not isinstance(decoded, dict) or set(decoded) != {entry_id}:
        raise ValueError("BMRB entry envelope drifted: %s" % loop)
    entry = decoded[entry_id]
    if entry == {}:
        return []
    if not isinstance(entry, dict) or set(entry) != {loop}:
        raise ValueError("BMRB category envelope drifted: %s" % loop)
    instances = entry[loop]
    if not isinstance(instances, list):
        raise ValueError("BMRB category instances are not a list: %s" % loop)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != {"category", "tags", "data"}:
            raise ValueError("BMRB instance schema drifted: %s" % loop)
        if instance["category"] != "_" + loop:
            raise ValueError("BMRB category identity drifted: %s" % loop)
        tags = instance["tags"]
        if (
            not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or len(tags) != len(set(tags))
            or not REQUIRED_TAGS[loop].issubset(tags)
            or not set(tags).issubset(ALLOWED_TAGS[loop])
            or (loop == NEW_LOOP and set(tags) != REQUIRED_TAGS[NEW_LOOP])
        ):
            raise ValueError("BMRB tags are incomplete or target-bearing: %s" % loop)
        data = instance["data"]
        if not isinstance(data, list):
            raise ValueError("BMRB category data are not a list: %s" % loop)
        for values in data:
            if (
                not isinstance(values, list)
                or len(values) != len(tags)
                or any(isinstance(value, (dict, list)) for value in values)
            ):
                raise ValueError("BMRB row schema drifted: %s" % loop)
            row = dict(zip(tags, values))
            if str(row["Entry_ID"]).strip() != entry_id:
                raise ValueError("BMRB row entry identity drifted: %s" % loop)
            rows.append(row)
    return rows


def _number(value: Any) -> Optional[float]:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _type_key(value: Any) -> str:
    return "".join(str(value).casefold().split())


def _condition(
    rows: Sequence[dict[str, Any]], condition_id: Optional[str]
) -> tuple[Optional[float], int, list[dict[str, str]], list[str]]:
    reasons: list[str] = []
    diagnostics: list[dict[str, str]] = []
    values: list[float] = []
    ph_records = 0
    if condition_id is None:
        return None, ph_records, diagnostics, ["assigned-list condition pointer is invalid"]
    for row in rows:
        if _valid_id(row["Sample_condition_list_ID"]) != condition_id:
            continue
        kind = _type_key(row["Type"])
        if kind in ("temperature", "ionicstrength"):
            diagnostics.append(
                {
                    "type": kind,
                    "value": "" if row["Val"] is None else str(row["Val"]),
                    "units": "" if row["Val_units"] is None else str(row["Val_units"]),
                }
            )
            continue
        if kind == "pd":
            reasons.append("pD record is not deposited pH")
            continue
        if kind != "ph":
            continue
        ph_records += 1
        units = "" if _missing(row["Val_units"]) else str(row["Val_units"]).strip().casefold()
        value = _number(row["Val"])
        if units not in ("", "ph", "p.h."):
            reasons.append("deposited pH units are invalid")
            continue
        if value is None or not 0.0 <= value <= 14.0:
            reasons.append("deposited pH value is invalid, nonfinite, or a range")
            continue
        values.append(value)
    unique = sorted(set(values))
    if not ph_records:
        reasons.append("missing deposited pH record")
    if len(unique) > 1:
        reasons.append("contradictory deposited pH values")
    if ph_records and not unique:
        reasons.append("no valid deposited pH value")
    return (unique[0] if len(unique) == 1 else None), ph_records, diagnostics, reasons


def summarize_entity(
    entity_uid: str, bmrb_id: str, prior: dict[str, bytes], assigned: bytes
) -> dict[str, Any]:
    """Independently apply the v1 exact-route/fallback metadata semantics."""
    _identity(bmrb_id, entity_uid, "summary entity")
    if set(prior) != set(PRIOR_LOOPS):
        raise ValueError("prior response set is not exactly the sealed three loops")
    entry_id = bmrb_id[3:]
    chem_rows = _rows(prior["Chem_shift_experiment"], entry_id, "Chem_shift_experiment")
    experiment_rows = _rows(prior["Experiment"], entry_id, "Experiment")
    condition_rows = _rows(
        prior["Sample_condition_variable"], entry_id, "Sample_condition_variable"
    )
    assigned_rows = _rows(assigned, entry_id, NEW_LOOP)

    experiments: dict[str, list[dict[str, Any]]] = {}
    for row in experiment_rows:
        identifier = _valid_id(row["ID"])
        if identifier is not None:
            experiments.setdefault(identifier, []).append(row)
    valid_assigned: list[tuple[str, str]] = []
    for row in assigned_rows:
        identifier = _valid_id(row["ID"])
        condition_id = _valid_id(row["Sample_condition_list_ID"])
        if identifier is not None and condition_id is not None:
            valid_assigned.append((identifier, condition_id))

    reasons: list[str] = []
    complete: list[tuple[str, str]] = []
    wholly_absent = 0
    malformed = 0
    for row in chem_rows:
        experiment_id = _valid_id(row["Experiment_ID"])
        assigned_id = _valid_id(row["Assigned_chem_shift_list_ID"])
        if experiment_id is not None and assigned_id is not None:
            complete.append((experiment_id, assigned_id))
        elif _missing(row["Experiment_ID"]) and _missing(row["Assigned_chem_shift_list_ID"]):
            wholly_absent += 1
        else:
            malformed += 1
    if malformed:
        reasons.append("Chem_shift_experiment row has only one valid or malformed linkage ID")

    route = "unavailable"
    selected_experiments: list[str] = []
    selected_assigned: list[str] = []
    selected_conditions: list[str] = []
    if not chem_rows or wholly_absent == len(chem_rows):
        route = "direct_fallback"
        if len(assigned_rows) != 1 or len(valid_assigned) != 1:
            reasons.append("direct fallback requires exactly one valid Assigned_chem_shift_list row")
        else:
            assigned_id, condition_id = valid_assigned[0]
            selected_assigned = [assigned_id]
            selected_conditions = [condition_id]
    elif complete and len(complete) == len(chem_rows):
        route = "exact_route"
        selected_experiments = sorted({item[0] for item in complete})
        selected_assigned = sorted({item[1] for item in complete})
        if len(selected_assigned) != 1:
            reasons.append("complete exact links do not resolve to one assigned list")
        experiment_conditions: list[str] = []
        for experiment_id in selected_experiments:
            matches = experiments.get(experiment_id, [])
            if len(matches) != 1:
                reasons.append("Chem_shift_experiment link does not resolve to one Experiment")
                continue
            condition_id = _valid_id(matches[0]["Sample_condition_list_ID"])
            if condition_id is None:
                reasons.append("linked Experiment condition pointer is missing or invalid")
            else:
                experiment_conditions.append(condition_id)
        selected_conditions = sorted(set(experiment_conditions))
        if len(selected_conditions) != 1:
            reasons.append("complete exact links do not resolve to one condition pointer")
        if len(selected_assigned) == 1 and len(selected_conditions) == 1:
            selected_rows = [
                row for row in assigned_rows if _valid_id(row["ID"]) == selected_assigned[0]
            ]
            if len(selected_rows) != 1:
                reasons.append("selected assigned list does not resolve to one fetched row")
            else:
                direct_condition = _valid_id(selected_rows[0]["Sample_condition_list_ID"])
                if direct_condition is None:
                    reasons.append("selected assigned-list condition pointer is missing or invalid")
                elif direct_condition != selected_conditions[0]:
                    reasons.append("direct assigned-list condition pointer contradicts Experiment")
    elif chem_rows:
        reasons.append("Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")

    deposited_ph: Optional[float] = None
    ph_record_count = 0
    diagnostics: list[dict[str, str]] = []
    if len(selected_conditions) == 1:
        deposited_ph, ph_record_count, diagnostics, condition_reasons = _condition(
            condition_rows, selected_conditions[0]
        )
        reasons.extend(condition_reasons)
    else:
        reasons.append("one condition-list pointer was not resolved")
    if len(selected_conditions) != 1 or deposited_ph is None:
        reasons.append("one deposited pH was not resolved")
    return {
        "entity_uid": entity_uid,
        "bmrb_id": bmrb_id,
        "route": route,
        "complete_chem_shift_experiment_link_count": len(complete),
        "experiment_ids": selected_experiments,
        "assigned_chem_shift_list_ids": selected_assigned,
        "sample_condition_list_ids": selected_conditions,
        "deposited_ph": deposited_ph,
        "deposited_ph_record_count": ph_record_count,
        "temperature_ionic_diagnostics": diagnostics,
        "ph_feasible": not reasons,
        "hold_reasons": sorted(set(reasons)),
    }


def _validate_roster(raw: bytes) -> dict[str, dict[str, Any]]:
    if _sha256(raw) != ROSTER_SHA256:
        raise ValueError("bound roster raw SHA-256 drifted")
    roster = _decode(raw, "bound roster")
    if not isinstance(roster, dict) or set(roster) != ROSTER_FIELDS:
        raise ValueError("bound roster schema drifted")
    if (
        roster.get("artifact_kind") != "target_unread_structural_entity_roster_not_authorization"
        or roster.get("contract") != "atypemu_nested_support_count_v1_entity_roster_v3"
        or roster.get("study_id") != "atypemu_nested_support_count_v1"
        or type(roster.get("entity_count")) is not int
        or roster["entity_count"] != ENTITY_COUNT
        or not isinstance(roster.get("entities"), list)
        or len(roster["entities"]) != ENTITY_COUNT
    ):
        raise ValueError("bound roster identity/count drifted")
    for flag in (
        "authorization_consumed", "outer_or_formal_metrics_opened", "source_scores_read", "target_values_read"
    ):
        if type(roster.get(flag)) is not bool or roster[flag] is not False:
            raise ValueError("bound roster protected flag opened: %s" % flag)
    if _safe_relative(roster.get("source_commitment_relative_path"), "roster source") != roster[
        "source_commitment_relative_path"
    ]:
        raise ValueError("unreachable roster source path")
    _sha(roster.get("source_commitment_sha256"), "roster source")

    by_uid: dict[str, dict[str, Any]] = {}
    seen_bmrb: set[str] = set()
    for index, entity in enumerate(roster["entities"]):
        label = "roster entity %d" % index
        if not isinstance(entity, dict) or set(entity) != ROSTER_ENTITY_FIELDS:
            raise ValueError("bound roster entity schema drifted: %s" % label)
        bmrb_id, entity_uid = _identity(entity.get("bmrb_id"), entity.get("entity_uid"), label)
        if (
            type(entity.get("canonical_atom_count")) is not int
            or entity["canonical_atom_count"] < 1
            or type(entity.get("canonical_heavy_atom_count")) is not int
            or entity["canonical_heavy_atom_count"] < 1
            or type(entity.get("canonical_reference_support_index")) is not int
            or entity["canonical_reference_support_index"] < 1
            or entity.get("split") != "train"
            or entity.get("observer_fold") not in ("A", "B")
        ):
            raise ValueError("bound roster entity metadata drifted: %s" % label)
        for name in (
            "canonical_all_atom_topology_sha256", "canonical_heavy_topology_sha256",
            "canonical_reference_pdb_sha256",
        ):
            _sha(entity.get(name), label + "." + name)
        reference = _safe_relative(entity.get("canonical_reference_relative_path"), label)
        if reference != entity["canonical_reference_relative_path"]:
            raise ValueError("unreachable roster reference path")
        if entity_uid in by_uid or bmrb_id in seen_bmrb:
            raise ValueError("duplicate bound roster identity")
        by_uid[entity_uid] = entity
        seen_bmrb.add(bmrb_id)
    if len(by_uid) != ENTITY_COUNT or len(seen_bmrb) != ENTITY_COUNT:
        raise ValueError("bound roster is incomplete")
    return by_uid


def _load_prior(raw: bytes, roster: dict[str, dict[str, Any]]) -> dict[str, dict[str, bytes]]:
    if _sha256(raw) != ARCHIVE_SHA256:
        raise ValueError("canonical condition archive raw SHA-256 drifted")
    result: dict[str, dict[str, bytes]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(infos) > MAX_ARCHIVE_MEMBERS
                or len(names) != len(set(names))
                or sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED_BYTES
            ):
                raise ValueError("canonical archive member inventory is unsafe")
            for info in infos:
                name = info.filename
                mode = info.external_attr >> 16
                if (
                    not name
                    or "\\" in name
                    or name.startswith("/")
                    or name.endswith("//")
                    or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError("canonical archive has an indirect member")
            by_name = {info.filename: info for info in infos}
            for entity_uid in sorted(roster):
                bmrb_id = str(roster[entity_uid]["bmrb_id"])
                selected: dict[str, bytes] = {}
                for loop in PRIOR_LOOPS:
                    member = "%s/%s.%s.json" % (V4_RAW_PREFIX, bmrb_id, loop)
                    info = by_name.get(member)
                    if (
                        info is None or info.filename != member or info.is_dir()
                        or info.flag_bits & 1 or info.file_size > MAX_JSON_BYTES
                    ):
                        raise ValueError("sealed archive response is absent or unsafe: %s" % member)
                    with archive.open(info, "r") as member_file:
                        payload = member_file.read(MAX_JSON_BYTES + 1)
                    if len(payload) != info.file_size or len(payload) > MAX_JSON_BYTES:
                        raise ValueError("sealed archive response length drifted: %s" % member)
                    _rows(payload, bmrb_id[3:], loop)
                    selected[loop] = payload
                result[entity_uid] = selected
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("canonical condition archive is unreadable") from error
    return result


def _requested_url(bmrb_id: str) -> str:
    tags = ",".join("%s.%s" % (NEW_LOOP, tag) for tag in sorted(REQUIRED_TAGS[NEW_LOOP]))
    return "%s/entry/%s/%s?%s" % (
        API_BASE, bmrb_id[3:], NEW_LOOP, urllib.parse.urlencode({"tag_list": tags})
    )


def _final_url(value: Any, bmrb_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError("final response URL is not a string")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("final response URL is malformed") from error
    requested = urllib.parse.urlsplit(_requested_url(bmrb_id))
    if (
        parsed.scheme != "https" or parsed.hostname != "api.bmrb.io"
        or parsed.username is not None or parsed.password is not None
        or port not in (None, 443) or parsed.fragment
        or parsed.path != requested.path or parsed.query != requested.query
    ):
        raise ValueError("final response URL leaves the exact official API route")
    return value


def _expected_new_binding(bmrb_id: str) -> dict[str, str]:
    return {
        "bmrb_id": bmrb_id,
        "loop": NEW_LOOP,
        "path": "%s/%s.%s.json" % (RAW_DIRECTORY, bmrb_id, NEW_LOOP),
        "url": _requested_url(bmrb_id),
    }


def _new_binding(value: Any, bmrb_id: str, label: str) -> dict[str, str]:
    expected = _expected_new_binding(bmrb_id)
    if not isinstance(value, dict) or set(value) != NEW_BINDING_FIELDS:
        raise ValueError("new response binding schema drifted: %s" % label)
    for name in ("bmrb_id", "loop", "path", "url"):
        if value[name] != expected[name]:
            raise ValueError("new response binding %s drifted: %s" % (name, label))
    _sha(value.get("sha256"), label + ".sha256")
    _final_url(value.get("final_url"), bmrb_id)
    return value  # type: ignore[return-value]


def _response_hash(payload: bytes, binding: dict[str, str], label: str) -> None:
    if _sha256(payload) != binding["sha256"]:
        raise ValueError("raw response SHA-256 mismatch: %s" % label)


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("invalid UTC timestamp: %s" % label)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("invalid UTC timestamp: %s" % label) from error


def _require_receipt_fields(receipt: Any) -> None:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("receipt schema drifted")


def _validate_receipt_static(
    receipt: Any, root: Path, plan_head: str, producer_head: str, recovery_evidence: dict[str, Any]
) -> None:
    _require_receipt_fields(receipt)
    assert isinstance(receipt, dict)
    if (
        receipt["artifact_kind"]
        != "hold_only_target_unread_openmm86_deposited_ph_catalog_recovery_receipt"
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["contract"] != CONTRACT
        or not _exact(receipt["recovery_evidence"], recovery_evidence)
        or receipt["metadata_scope"] != "target-unread metadata feasibility only"
        or receipt["api_base"] != API_BASE
        or receipt["application_header"] != APPLICATION_HEADER
        or not _exact(
            receipt["archive"], {"path": ARCHIVE_RELATIVE.as_posix(), "sha256": ARCHIVE_SHA256}
        )
        or not _exact(
            receipt["roster"], {"path": ROSTER_RELATIVE.as_posix(), "sha256": ROSTER_SHA256}
        )
    ):
        raise ValueError("receipt recovery identity or input binding drifted")
    for flag in CLOSED_FLAGS:
        if type(receipt[flag]) is not bool or receipt[flag] is not False:
            raise ValueError("receipt closed capability opened: %s" % flag)
    plan = receipt["plan"]
    if (
        not isinstance(plan, dict) or set(plan) != {"path", "sha256", "git_commit"}
        or plan.get("path") != PLAN_RELATIVE.as_posix()
        or plan.get("sha256") != PLAN_RAW_SHA256
        or receipt["source_producer_relative_path"] != PRODUCER_RELATIVE.as_posix()
        or receipt["source_producer_sha256"] != PRODUCER_RAW_SHA256
    ):
        raise ValueError("receipt plan or producer binding drifted")
    plan_revision = _commit_has_bytes(
        root, plan.get("git_commit"), PLAN_RELATIVE, PLAN_RAW_SHA256, "recovery-v3 plan"
    )
    producer_revision = _commit_has_bytes(
        root, receipt.get("source_producer_git_commit"), PRODUCER_RELATIVE,
        PRODUCER_RAW_SHA256, "recovery-v3 producer",
    )
    if plan_revision != producer_revision:
        raise ValueError("receipt plan and producer commits differ")
    _require_ancestor(root, plan_revision, plan_head, "recovery-v3 plan")
    _require_ancestor(root, producer_revision, producer_head, "recovery-v3 producer")
    started = _timestamp(receipt["started_at_utc"], "started_at_utc")
    ended = _timestamp(receipt["ended_at_utc"], "ended_at_utc")
    if ended < started:
        raise ValueError("receipt end time precedes start time")
    if not _exact(receipt["future_consumer"], FUTURE_CONSUMER) or not _exact(receipt["warnings"], WARNINGS):
        raise ValueError("receipt future-consumer declaration or warnings drifted")
    if (
        type(receipt["entity_count"]) is not int or receipt["entity_count"] != ENTITY_COUNT
        or type(receipt["new_response_count"]) is not int
        or receipt["new_response_count"] != ENTITY_COUNT
        or not isinstance(receipt["new_response_manifest"], list)
        or len(receipt["new_response_manifest"]) != ENTITY_COUNT
        or not isinstance(receipt["entities"], list) or len(receipt["entities"]) != ENTITY_COUNT
        or type(receipt["ph_feasible_entity_count"]) is not int
        or not 0 <= receipt["ph_feasible_entity_count"] <= ENTITY_COUNT
        or type(receipt["all_entities_ph_feasible"]) is not bool
        or not isinstance(receipt["status"], str)
    ):
        raise ValueError("receipt count/status schema drifted")


def _summary_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != ENTITY_FIELDS:
        raise ValueError("entity summary schema drifted: %s" % label)
    _identity(value.get("bmrb_id"), value.get("entity_uid"), label)
    if (
        value.get("route") not in ("unavailable", "direct_fallback", "exact_route")
        or type(value.get("complete_chem_shift_experiment_link_count")) is not int
        or value["complete_chem_shift_experiment_link_count"] < 0
        or type(value.get("deposited_ph_record_count")) is not int
        or value["deposited_ph_record_count"] < 0
        or type(value.get("ph_feasible")) is not bool
        or not isinstance(value.get("experiment_ids"), list)
        or not isinstance(value.get("assigned_chem_shift_list_ids"), list)
        or not isinstance(value.get("sample_condition_list_ids"), list)
        or not isinstance(value.get("temperature_ionic_diagnostics"), list)
        or not isinstance(value.get("hold_reasons"), list)
        or not isinstance(value.get("prior_archive_response_bindings"), list)
    ):
        raise ValueError("entity summary types drifted: %s" % label)
    deposited = value["deposited_ph"]
    if deposited is not None and (
        type(deposited) is not float or not math.isfinite(deposited) or not 0.0 <= deposited <= 14.0
    ):
        raise ValueError("entity deposited pH type/range drifted: %s" % label)
    for name in ("experiment_ids", "assigned_chem_shift_list_ids", "sample_condition_list_ids"):
        identifiers = value[name]
        if (
            not all(isinstance(item, str) and ID_RE.fullmatch(item) for item in identifiers)
            or identifiers != sorted(set(identifiers))
        ):
            raise ValueError("entity ID summary drifted: %s" % label)
    reasons = value["hold_reasons"]
    if not all(isinstance(reason, str) and reason for reason in reasons) or reasons != sorted(set(reasons)):
        raise ValueError("entity HOLD reasons drifted: %s" % label)
    for diagnostic in value["temperature_ionic_diagnostics"]:
        if (
            not isinstance(diagnostic, dict) or set(diagnostic) != DIAGNOSTIC_FIELDS
            or diagnostic.get("type") not in ("temperature", "ionicstrength")
            or not isinstance(diagnostic.get("value"), str)
            or not isinstance(diagnostic.get("units"), str)
        ):
            raise ValueError("entity diagnostic schema drifted: %s" % label)
    prior = value["prior_archive_response_bindings"]
    if len(prior) != len(PRIOR_LOOPS):
        raise ValueError("prior response binding count drifted: %s" % label)
    seen_loops = set()
    for binding in prior:
        if not isinstance(binding, dict) or set(binding) != PRIOR_BINDING_FIELDS:
            raise ValueError("prior response binding schema drifted: %s" % label)
        loop = binding.get("loop")
        if loop not in PRIOR_LOOPS or loop in seen_loops:
            raise ValueError("prior response binding category drifted: %s" % label)
        seen_loops.add(loop)
        _sha(binding.get("sha256"), label + ".prior")
    _new_binding(value["new_response_binding"], value["bmrb_id"], label + ".new")


def _validate_artifact_tree(root: Path, started: str, expected_files: set[str]) -> None:
    artifact = _directory_path(root, ARTIFACT_RELATIVE, "recovery-v3 output")
    try:
        with os.scandir(artifact) as entries:
            names = set()
            for entry in entries:
                if len(names) >= 3:
                    raise ValueError("recovery-v3 output inventory exceeds its bound")
                if entry.is_symlink():
                    raise ValueError("symlink in recovery-v3 output")
                names.add(entry.name)
    except OSError as error:
        raise ValueError("cannot inspect recovery-v3 output") from error
    if names != {RAW_DIRECTORY, "receipt.json", START_MARKER}:
        raise ValueError("recovery-v3 output inventory drifted")
    raw_directory = _directory_path(root, ARTIFACT_RELATIVE / RAW_DIRECTORY, "new response directory")
    try:
        with os.scandir(raw_directory) as entries:
            names = set()
            for entry in entries:
                if len(names) >= len(expected_files):
                    raise ValueError("new response directory exceeds its manifest bound")
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("non-regular new response entry")
                names.add(entry.name)
    except OSError as error:
        raise ValueError("cannot inspect new response directory") from error
    if names != expected_files:
        raise ValueError("new response directory is not the exact 135-file manifest")
    marker = _read_regular(root, ARTIFACT_RELATIVE / START_MARKER, "recovery start marker", 1024)
    if marker != (started + "\n").encode("utf-8"):
        raise ValueError("recovery start marker drifted")


def verify_artifact() -> tuple[int, str]:
    """Check only local consistency; this never establishes immutable evidence."""
    root = _bound_root()
    # These fixed gates must precede checker, plan, producer, roster, archive,
    # output receipt, and response reads.  They are independently checked once
    # more below through the raw-byte-bound parsed plan.
    recovery_evidence = _validate_pre_output_gates(root)
    _require_committed_checker(root)
    plan, plan_head, producer_head = _load_plan_and_producer(root)
    if not _exact(plan["pre_output_gates"], PRE_OUTPUT_GATES) or not _exact(
        plan["recovery_of"], V1_FAILURE_BINDING
    ):
        raise ValueError("recovery-v3 plan pre-output bindings drifted")

    receipt = _decode(
        _read_regular(root, RECEIPT_RELATIVE, "recovery-v3 receipt", MAX_JSON_BYTES),
        "recovery-v3 receipt",
    )
    _validate_receipt_static(receipt, root, plan_head, producer_head, recovery_evidence)

    roster = _validate_roster(
        _read_regular(root, ROSTER_RELATIVE, "bound roster", MAX_JSON_BYTES)
    )
    prior = _load_prior(
        _read_regular(root, ARCHIVE_RELATIVE, "canonical condition archive", MAX_ARCHIVE_BYTES),
        roster,
    )
    expected_uids = sorted(roster)
    expected_files = {
        "%s.%s.json" % (roster[uid]["bmrb_id"], NEW_LOOP) for uid in expected_uids
    }
    _validate_artifact_tree(root, receipt["started_at_utc"], expected_files)

    manifest = receipt["new_response_manifest"]
    responses: dict[str, bytes] = {}
    for index, entity_uid in enumerate(expected_uids):
        bmrb_id = str(roster[entity_uid]["bmrb_id"])
        binding = _new_binding(manifest[index], bmrb_id, "new_response_manifest[%d]" % index)
        response = _read_regular(
            root, ARTIFACT_RELATIVE / binding["path"], "new response %d" % index, MAX_JSON_BYTES
        )
        _response_hash(response, binding, str(index))
        _rows(response, bmrb_id[3:], NEW_LOOP)
        responses[entity_uid] = response

    expected_entities: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_bmrbs: set[str] = set()
    for index, entity_uid in enumerate(expected_uids):
        observed = receipt["entities"][index]
        label = "entities[%d]" % index
        _summary_shape(observed, label)
        bmrb_id = str(roster[entity_uid]["bmrb_id"])
        if observed["entity_uid"] != entity_uid or observed["bmrb_id"] != bmrb_id:
            raise ValueError("receipt entity roster identity drifted: %s" % label)
        if entity_uid in seen_uids or bmrb_id in seen_bmrbs:
            raise ValueError("duplicate receipt entity identity: %s" % label)
        seen_uids.add(entity_uid)
        seen_bmrbs.add(bmrb_id)
        replayed = summarize_entity(entity_uid, bmrb_id, prior[entity_uid], responses[entity_uid])
        replayed["prior_archive_response_bindings"] = [
            {
                "archive_member": "%s/%s.%s.json" % (V4_RAW_PREFIX, bmrb_id, loop),
                "loop": loop,
                "sha256": _sha256(prior[entity_uid][loop]),
            }
            for loop in PRIOR_LOOPS
        ]
        replayed["new_response_binding"] = manifest[index]
        _require_replayed_summary(observed, replayed, label)
        expected_entities.append(replayed)
    if seen_uids != set(expected_uids) or len(seen_bmrbs) != ENTITY_COUNT:
        raise ValueError("receipt entity roster is incomplete")

    feasible = sum(1 for entity in expected_entities if entity["ph_feasible"])
    if receipt["ph_feasible_entity_count"] != feasible:
        raise ValueError("receipt pH-feasible aggregate differs")
    if receipt["all_entities_ph_feasible"] is not (feasible == ENTITY_COUNT):
        raise ValueError("receipt all-entities pH aggregate differs")
    status = (
        "PASS_DEPOSITED_PH_METADATA_COMPLETE"
        if feasible == ENTITY_COUNT
        else "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
    )
    if receipt["status"] != status:
        raise ValueError("receipt status differs")
    return 9 + ENTITY_COUNT * (len(PRIOR_LOOPS) + 4), status


def _require_replayed_summary(
    observed: dict[str, Any], replayed: dict[str, Any], label: str
) -> None:
    if observed != replayed:
        raise ValueError("replayed entity summary differs: %s" % label)


# Synthetic fixtures only.  self_test does not open repository evidence or use a network.
def _fixture(
    *,
    chem: Optional[list[list[Any]]] = None,
    assigned: Optional[list[list[Any]]] = None,
    conditions: Optional[list[list[Any]]] = None,
    experiments: Optional[list[list[Any]]] = None,
) -> tuple[dict[str, bytes], bytes]:
    entry_id = "42"

    def wrapped(loop: str, tags: list[str], values: list[list[Any]]) -> bytes:
        return json.dumps(
            {entry_id: {loop: [{"category": "_" + loop, "tags": tags, "data": values}]}},
            sort_keys=True,
        ).encode("utf-8")

    chem = chem if chem is not None else [["1", "7", entry_id]]
    experiments = experiments if experiments is not None else [["1", "1", "10", entry_id]]
    conditions = conditions if conditions is not None else [
        [" P H ", "6.5", "p.h.", entry_id, "10"],
        ["temperature", "not-a-number", "K", entry_id, "10"],
        ["ionic strength", "n/a", "M", entry_id, "10"],
    ]
    assigned = assigned if assigned is not None else [[entry_id, "7", "10"]]
    return {
        "Chem_shift_experiment": wrapped(
            "Chem_shift_experiment", ["Experiment_ID", "Assigned_chem_shift_list_ID", "Entry_ID"], chem
        ),
        "Experiment": wrapped(
            "Experiment", ["ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"], experiments
        ),
        "Sample_condition_variable": wrapped(
            "Sample_condition_variable", ["Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"], conditions
        ),
    }, wrapped(NEW_LOOP, ["Entry_ID", "ID", "Sample_condition_list_ID"], assigned)


def _summary(prior: dict[str, bytes], assigned: bytes) -> dict[str, Any]:
    return summarize_entity("bmrb:42:entity:1", "bmr42", prior, assigned)


def _reject(callback: Any, label: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("adversarial case was accepted: %s" % label)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _v1_fixture(root: Path) -> Path:
    directory = root / V1_FAILURE_RELATIVE
    directory.mkdir(parents=True)
    (directory / "raw_api_responses").mkdir()
    (directory / "failure_receipt.json").write_bytes(_json_bytes(V1_FAILURE_RECEIPT))
    (directory / "started_at_utc.txt").write_bytes((V1_STARTED_AT_UTC + "\n").encode("utf-8"))
    return directory


def _intent_fixture(root: Path, raw: Optional[bytes] = None) -> dict[str, Any]:
    raw = _json_bytes(V2_LAUNCH_INTENT) if raw is None else raw
    path = root / V2_LAUNCH_INTENT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": V2_LAUNCH_INTENT_RELATIVE.as_posix(),
        "sha256": _sha256(raw),
        "exact_fields": V2_LAUNCH_INTENT,
    }


def _run122_raw(controls: Optional[dict[str, Any]] = None) -> bytes:
    semantic = RUN122_LOG_BINDING["semantic_subset"]
    record = {
        "run": semantic["run"], "commit": semantic["commit"], "metric": semantic["metric"],
        "status": semantic["status"], "description": semantic["description"],
        "ASI": {
            "error": "ValueError: synthetic inventory drift",
            "root_cause": "synthetic omission",
            "failure_stage": "pre_output_gate",
            "observed_controls": controls if controls is not None else {
                "target_values_read": False, "network_requests": 0, "new_response_count_written": 0,
            },
        },
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _run122_fixture(root: Path, raw: Optional[bytes] = None) -> dict[str, Any]:
    raw = _run122_raw() if raw is None else raw
    path = root / LOG_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw + b"\n")
    return {
        "path": LOG_RELATIVE.as_posix(),
        "raw_line_sha256": _sha256(raw),
        "semantic_subset": RUN122_LOG_BINDING["semantic_subset"],
    }


def self_test() -> int:
    """Exercise synthetic local gates and replay negatives; never read the repository."""
    checks = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _v1_fixture(root)
        assert _validate_v1_failure(root) == V1_FAILURE_BINDING
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = _v1_fixture(root)
        (directory / "raw_api_responses" / "unexpected.json").write_bytes(b"x")
        _reject(lambda: _validate_v1_failure(root), "nonempty v1 raw directory")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = _v1_fixture(root)
        raw_directory = directory / "raw_api_responses"
        raw_directory.rmdir()
        os.symlink("failure_receipt.json", raw_directory)
        _reject(lambda: _validate_v1_failure(root), "symlinked v1 raw directory")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = _v1_fixture(root)
        (directory / "failure_receipt.json").write_bytes(b"tampered")
        _reject(lambda: _validate_v1_failure(root), "v1 receipt hash tamper")
        checks += 1

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        intent = _intent_fixture(root)
        assert _validate_v2_launch_intent(root, intent) == intent
        _reject(lambda: _validate_v2_launch_intent(root), "fixed v2 intent SHA-256")
        checks += 2
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        intent = _intent_fixture(root)
        path = root / V2_LAUNCH_INTENT_RELATIVE
        path.write_bytes(b"intent hash tamper")
        _reject(lambda: _validate_v2_launch_intent(root, intent), "v2 intent hash")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        malformed = dict(V2_LAUNCH_INTENT)
        del malformed["output"]
        intent = _intent_fixture(root, _json_bytes(malformed))
        _reject(lambda: _validate_v2_launch_intent(root, intent), "v2 intent output field")
        checks += 1

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        log = _run122_fixture(root)
        assert _validate_run122_log(root, log) == log
        _reject(lambda: _validate_run122_log(root), "fixed Run 122 SHA-256")
        checks += 2
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = _run122_raw()
        log = _run122_fixture(root, raw)
        (root / LOG_RELATIVE).write_bytes(raw + b" \n")
        _reject(lambda: _validate_run122_log(root, log), "Run 122 raw hash")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = _run122_raw()
        log = _run122_fixture(root, raw)
        (root / LOG_RELATIVE).write_bytes(raw + b"\n" + raw + b"\n")
        _reject(lambda: _validate_run122_log(root, log), "Run 122 duplicate")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = _run122_raw({"target_values_read": True})
        log = _run122_fixture(root, raw)
        _reject(lambda: _validate_run122_log(root, log), "Run 122 opened control")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = _decode(_run122_raw(), "synthetic Run 122")
        del raw["status"]
        log = _run122_fixture(root, _json_bytes(raw).rstrip(b"\n"))
        _reject(lambda: _validate_run122_log(root, log), "Run 122 schema")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / LOG_RELATIVE
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * (MAX_LOG_BYTES + 1))
        _reject(lambda: _validate_run122_log(root), "bounded log read")
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / V2_OUTPUT_RELATIVE).mkdir(parents=True)
        _reject(lambda: _require_v2_output_absent(root), "v2 output presence")
        checks += 1

    prior, assigned = _fixture()
    complete = _summary(prior, assigned)
    assert complete["ph_feasible"] is True and complete["deposited_ph"] == 6.5
    assert complete["temperature_ionic_diagnostics"]
    checks += 1
    for kwargs in (
        {"chem": [["1", ".", "42"]]},
        {"chem": [["1", "7", "42"], [".", ".", "42"]]},
        {"assigned": []},
        {"assigned": [["42", "7", "11"]]},
        {"experiments": [["2", "1", "10", "42"]]},
        {"chem": [["1", "7", "42"], ["1", "8", "42"]]},
        {"conditions": [["pH", "6.5", "mM", "42", "10"]]},
        {"conditions": [["pH", "6.0-7.0", "pH", "42", "10"]]},
        {"conditions": [["pD", "6.5", "", "42", "10"]]},
        {"conditions": [["pH", "6.5", "pH", "42", "10"], ["pH", "7.0", "pH", "42", "10"]]},
    ):
        assert _summary(*_fixture(**kwargs))["ph_feasible"] is False
        checks += 1
    fallback = _summary(*_fixture(chem=[]))
    assert fallback["route"] == "direct_fallback" and fallback["ph_feasible"] is True
    checks += 1
    invalid_units = _summary(
        *_fixture(conditions=[["pH", "6.5", "mM", "42", "10"]])
    )
    assert invalid_units["deposited_ph"] is None
    mixed_units = _summary(
        *_fixture(
            conditions=[
                ["pH", "6.5", "pH", "42", "10"],
                ["pH", "7.0", "mM", "42", "10"],
            ]
        )
    )
    assert mixed_units["deposited_ph"] == 6.5 and mixed_units["ph_feasible"] is False
    checks += 2

    parsed = _decode(assigned, "synthetic response")
    parsed["42"][NEW_LOOP][0]["tags"].append("Atom_ID")
    parsed["42"][NEW_LOOP][0]["data"][0].append("CA")
    _reject(lambda: _rows(json.dumps(parsed).encode("utf-8"), "42", NEW_LOOP), "target-bearing tag")
    checks += 1
    binding = _expected_new_binding("bmr42")
    binding.update({"sha256": _sha256(assigned), "final_url": _requested_url("bmr42")})
    _reject(lambda: _response_hash(assigned + b" ", binding, "synthetic response"), "response tamper")
    _reject(lambda: _final_url(_requested_url("bmr42") + "&extra=1", "bmr42"), "URL query extension")
    checks += 2

    entity = dict(complete)
    entity["prior_archive_response_bindings"] = [
        {"archive_member": "%s/bmr42.%s.json" % (V4_RAW_PREFIX, loop), "loop": loop, "sha256": "0" * 64}
        for loop in PRIOR_LOOPS
    ]
    entity["new_response_binding"] = binding
    altered = dict(entity)
    altered["ph_feasible"] = False
    _reject(lambda: _require_replayed_summary(altered, entity, "synthetic"), "replayed summary tamper")
    unknown = dict(entity)
    unknown["unknown"] = False
    _reject(lambda: _summary_shape(unknown, "synthetic"), "unknown entity field")
    checks += 2

    receipt = {field: None for field in RECEIPT_FIELDS}
    receipt["candidate_id"] = "tampered"
    with tempfile.TemporaryDirectory() as temporary:
        _reject(
            lambda: _validate_receipt_static(
                receipt, Path(temporary), "0" * 40, "0" * 40, RECOVERY_EVIDENCE
            ),
            "receipt tamper",
        )
    exact_fields = {field: None for field in RECEIPT_FIELDS}
    _require_receipt_fields(exact_fields)
    exact_fields["unknown"] = False
    _reject(lambda: _require_receipt_fields(exact_fields), "unknown receipt field")
    checks += 2

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact = root / ARTIFACT_RELATIVE
        raw_directory = artifact / RAW_DIRECTORY
        raw_directory.mkdir(parents=True)
        expected = "bmr42.%s.json" % NEW_LOOP
        (raw_directory / expected).write_bytes(assigned)
        (artifact / "receipt.json").write_bytes(b"{}")
        (artifact / START_MARKER).write_bytes(b"2026-01-01T00:00:00Z\n")
        _validate_artifact_tree(root, "2026-01-01T00:00:00Z", {expected})
        (artifact / "unexpected").write_bytes(b"x")
        _reject(lambda: _validate_artifact_tree(root, "2026-01-01T00:00:00Z", {expected}), "output inventory tamper")
        checks += 1

    _reject(lambda: _decode(b'{"42": {}, "42": {}}', "duplicate keys"), "duplicate JSON keys")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "response.json").write_bytes(assigned)
        _reject(lambda: _read_regular(root, Path("response.json"), "bounded fixture", 1), "bounded read")
        checks += 2
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="run synthetic tests only")
    verify = commands.add_parser(
        "verify-local-consistency",
        help="replay fixed local evidence; it is not immutable evidence",
    )
    verify.add_argument("--acknowledge-target-unread-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        checks = self_test()
        print("METRIC openmm86_deposited_ph_recovery_v3_checker_self_tests=%d" % checks)
        print("STATUS PASS")
        return 0
    if not args.acknowledge_target_unread_metadata_only:
        print("REFUSAL target-unread metadata-only acknowledgement is required")
        return 3
    try:
        checks, status = verify_artifact()
    except ValueError as error:
        print("REFUSAL fixed local recovery-v3 evidence is absent or invalid: %s" % error)
        print("STATUS HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS")
        return 3
    print("METRIC openmm86_deposited_ph_recovery_v3_checker_checks=%d" % checks)
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC authorization_consumed=0")
    print("LOCAL_SEMANTIC_STATUS %s" % status)
    print("STATUS HOLD_LOCAL_ONLY_PENDING_SEPARATELY_EXACT_BOUND_IMMUTABLE_ARCHIVE_AND_EXECUTION_RECEIPT")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
