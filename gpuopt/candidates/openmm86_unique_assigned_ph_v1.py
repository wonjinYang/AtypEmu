#!/usr/bin/env python3
"""HOLD-only fallback reinterpretation of fixed recovery-v6 metadata.

The direct Assigned_chem_shift_list condition pointer is deliberately weaker
than the Experiment-loop pointer.  This candidate uses it only when every
complete Experiment link is absent or lacks a valid condition pointer.  A
resolved pH is metadata only; it is never a protonation, all-atom, support, or
science feasibility result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# These are deliberately the archive and response-schema helpers from the
# committed recovery-v6 independent semantic checker.  This candidate does
# not import its old linkage or pH derivation logic.
try:
    from check_openmm86_deposited_ph_recovery_v6_semantic_independent import (
        ASSIGNED_TAGS,
        CHEM_TAGS,
        CONDITION_TAGS,
        EXPERIMENT_TAGS,
        LOOPS,
        PRIOR_NAME,
        PRIOR_RECEIPT,
        PRIOR_ROOT,
        ROSTER_MEMBER,
        V6_API,
        V6_MANIFEST,
        V6_NAME,
        V6_ROOT,
        _audit_json_members,
        _canonical_roster,
        _load_json,
        _manifest_entry,
        _open_fixed_zip,
        _parse_assigned_response,
        _parse_prior_response,
        _safe_member_name,
        _sha256_bytes,
        _validate_v6,
        verify as verify_v6_semantics,
    )
except ImportError:  # Package import for external synthetic test harnesses.
    from gpuopt.candidates.check_openmm86_deposited_ph_recovery_v6_semantic_independent import (
        ASSIGNED_TAGS,
        CHEM_TAGS,
        CONDITION_TAGS,
        EXPERIMENT_TAGS,
        LOOPS,
        PRIOR_NAME,
        PRIOR_RECEIPT,
        PRIOR_ROOT,
        ROSTER_MEMBER,
        V6_API,
        V6_MANIFEST,
        V6_NAME,
        V6_ROOT,
        _audit_json_members,
        _canonical_roster,
        _load_json,
        _manifest_entry,
        _open_fixed_zip,
        _parse_assigned_response,
        _parse_prior_response,
        _safe_member_name,
        _sha256_bytes,
        _validate_v6,
        verify as verify_v6_semantics,
    )


CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1"
CONTRACT = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1"
CANDIDATE_RELATIVE = Path("gpuopt/candidates/openmm86_unique_assigned_ph_v1.py")
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_plan.json"
)
PLAN_RAW_SHA256 = "c35724112714e3e5de474f08538540e121241ebef3458717b034e712dd5b083e"
PARENT_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json"
)
V6_EVIDENCE_RECEIPT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_receipt.json"
)
V6_SEMANTIC_CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py"
)
OUTPUT_RELATIVE = Path(".auto/staging/openmm86_unique_assigned_ph_v1/receipt.json")
INTENT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_execution_intent.json"
)

# These validation bindings are declared in the companion static plan.
# Production hash-checks them before reading the two metadata archives.
PARENT_PLAN_RAW_SHA256 = "9244f880c7a67e97c5a370013b3cfc81cccc8cb9d7267b2fff0b0a1b5dddab91"
V6_EVIDENCE_RECEIPT_RAW_SHA256 = "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36"
V6_SEMANTIC_CHECKER_RAW_SHA256 = "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84"
PRIOR_ARCHIVE_SHA256 = "cca8e8f94bb812d226f19055f4b6377da9f070520f21fc3ab76bde490a5b0b8a"
V6_ARCHIVE_SHA256 = "0fc60731b7f6cc053c9cf623ecb8e033ce8058f0f42791ea894414e98143d7b8"

CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
ID_RE = re.compile(r"[1-9][0-9]*\Z")
EXPERIMENT_KEYSETS = frozenset(frozenset(tags) for tags in EXPERIMENT_TAGS)
MAX_SOURCE_BYTES = 2_000_000


class CandidateError(ValueError):
    """A structural or metadata ambiguity that remains HOLD-only."""


def _fail(message: str) -> None:
    raise CandidateError(message)


def _missing(value: Any) -> bool:
    return value is None or (
        type(value) is str and value.strip() in ("", ".", "?")
    )


def _valid_id(value: Any) -> Optional[str]:
    if type(value) is str and ID_RE.fullmatch(value) is not None:
        return value
    return None


def _text_or_none(value: Any) -> bool:
    return value is None or type(value) is str


def _normal_type(value: Any) -> str:
    if type(value) is not str:
        return ""
    return re.sub(r"\s+", "", value.casefold())


def _numeric_ph(value: Any) -> Optional[float]:
    """Return only a finite, in-range JSON scalar pH value."""
    if isinstance(value, bool) or _missing(value):
        return None
    if type(value) not in (int, float, str):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 14.0:
        return None
    return 0.0 if number == 0.0 else number


def _row_has_exact_keys(row: Any, keys: Sequence[str], label: str) -> Dict[str, Any]:
    if not isinstance(row, dict) or set(row) != set(keys):
        _fail("%s row has malformed or extra fields" % label)
    return row


def _validate_input_schema(
    prior: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    bmrb: str,
    assigned_rows: List[Dict[str, Any]],
) -> None:
    """Reject schema drift before applying the independent fallback rule."""
    expected = {(bmrb, loop) for loop in LOOPS}
    if not isinstance(prior, Mapping) or set(prior) != expected:
        _fail("prior loop mapping has malformed or extra fields")
    for loop in LOOPS:
        rows = prior[(bmrb, loop)]
        if not isinstance(rows, list):
            _fail("%s rows are not a list" % loop)
        for row in rows:
            if loop == "Chem_shift_experiment":
                checked = _row_has_exact_keys(row, CHEM_TAGS, loop)
                if not all(_text_or_none(item) for item in checked.values()):
                    _fail("Chem_shift_experiment row has non-text fields")
            elif loop == "Experiment":
                if not isinstance(row, dict) or frozenset(row) not in EXPERIMENT_KEYSETS:
                    _fail("Experiment row has malformed or extra fields")
                if not all(_text_or_none(item) for item in row.values()):
                    _fail("Experiment row has non-text fields")
            else:
                checked = _row_has_exact_keys(row, CONDITION_TAGS, loop)
                for name, item in checked.items():
                    if name == "Val":
                        if isinstance(item, bool) or type(item) not in (
                            type(None), int, float, str,
                        ):
                            _fail("Sample_condition_variable Val has an invalid type")
                    elif not _text_or_none(item):
                        _fail("Sample_condition_variable row has non-text fields")
    if not isinstance(assigned_rows, list):
        _fail("Assigned_chem_shift_list rows are not a list")
    for row in assigned_rows:
        checked = _row_has_exact_keys(row, ASSIGNED_TAGS, "Assigned_chem_shift_list")
        if not all(_text_or_none(item) for item in checked.values()):
            _fail("Assigned_chem_shift_list row has non-text fields")


def _held_result(
    bmrb: str, uid: str, reason: str, assigned_id: Optional[str] = None,
    condition_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "assigned_chem_shift_list_id": assigned_id,
        "bmrb_id": bmrb,
        "direct_sample_condition_list_id": condition_id,
        "entity_uid": uid,
        "fallback_metadata_resolved": False,
        "fallback_ph": None,
        "hold_reasons": [reason, "metadata resolution is not feasibility evidence"],
    }


def _fallback_entity(
    bmrb: str,
    uid: str,
    prior: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    assigned_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply only this candidate's direct-pointer fallback semantics.

    In particular, no route from the prior semantic checker is imported or
    reused here.  A valid preferred Experiment condition pointer is sufficient
    to forbid this fallback, even if that preferred route is otherwise not
    useful for a pH result.
    """
    try:
        _validate_input_schema(prior, bmrb, assigned_rows)
        if not (type(bmrb) is str and bmrb.startswith("bmr") and _valid_id(bmrb[3:])):
            _fail("BMRB identity is invalid")
        entry_number = bmrb[3:]
        direct: List[Tuple[str, str]] = []
        for row in assigned_rows:
            entry_id = _valid_id(row[ASSIGNED_TAGS[0]])
            assigned_id = _valid_id(row[ASSIGNED_TAGS[1]])
            condition_id = _valid_id(row[ASSIGNED_TAGS[2]])
            # The v6 endpoint is per entry.  A mismatched or incomplete row is
            # not harmless ambiguity: it cannot establish a direct fallback.
            if entry_id != entry_number or assigned_id is None or condition_id is None:
                _fail("direct Assigned-list row is missing, invalid, or foreign")
            direct.append((assigned_id, condition_id))
        if len(direct) != 1:
            _fail("fallback requires exactly one valid direct Assigned-list row")
        assigned_id, condition_id = direct[0]

        complete_experiment_ids: List[str] = []
        matching_assigned_count = 0
        for row in prior[(bmrb, "Chem_shift_experiment")]:
            experiment_id = row["Experiment_ID"]
            row_assigned_id = row["Assigned_chem_shift_list_ID"]
            valid_experiment_id = _valid_id(experiment_id)
            valid_assigned_id = _valid_id(row_assigned_id)
            if not (_missing(experiment_id) or valid_experiment_id is not None):
                _fail("Chem_shift_experiment Experiment_ID is malformed")
            if not (_missing(row_assigned_id) or valid_assigned_id is not None):
                _fail("Chem_shift_experiment Assigned-list ID is malformed")
            if valid_assigned_id is not None:
                if valid_assigned_id != assigned_id:
                    _fail("a Chem_shift_experiment Assigned-list ID differs from direct ID")
                matching_assigned_count += 1
            # A valid Experiment pointer must be complete and agree with the
            # direct list.  This explicitly rejects the old one-sided form in
            # the direction that could hide preferred linkage.
            if valid_experiment_id is not None:
                if valid_assigned_id != assigned_id:
                    _fail("valid Experiment_ID has missing or other Assigned-list ID")
                complete_experiment_ids.append(valid_experiment_id)
        if matching_assigned_count == 0:
            _fail("no Chem_shift_experiment Assigned-list ID matches direct ID")

        experiment_rows = prior[(bmrb, "Experiment")]
        for experiment_id in sorted(set(complete_experiment_ids)):
            matches = [row for row in experiment_rows if row["ID"] == experiment_id]
            if len(matches) > 1:
                _fail("complete Experiment_ID resolves to duplicate Experiment rows")
            if len(matches) == 1 and _valid_id(matches[0]["Sample_condition_list_ID"]) is not None:
                _fail("preferred Experiment condition linkage is present; fallback is forbidden")
            # Zero matches, missing condition pointers, and invalid condition
            # pointers are deliberately the only states that reach fallback.

        ph_rows: List[Dict[str, Any]] = []
        for row in prior[(bmrb, "Sample_condition_variable")]:
            if row["Sample_condition_list_ID"] == condition_id and _normal_type(row["Type"]) == "ph":
                ph_rows.append(row)
        if len(ph_rows) != 1:
            _fail("direct condition does not have exactly one pH row")
        ph_row = ph_rows[0]
        units = ph_row["Val_units"]
        if not (units is None or units == "" or units == "pH"):
            _fail("direct pH row units are neither pH nor empty")
        ph = _numeric_ph(ph_row["Val"])
        if ph is None:
            _fail("direct pH row is not a numeric value in [0,14]")
        return {
            "assigned_chem_shift_list_id": assigned_id,
            "bmrb_id": bmrb,
            "direct_sample_condition_list_id": condition_id,
            "entity_uid": uid,
            "fallback_metadata_resolved": True,
            "fallback_ph": ph,
            "hold_reasons": ["metadata resolution is not feasibility evidence"],
        }
    except CandidateError as error:
        return _held_result(bmrb, uid, str(error))


def _read_prior_archive(
    path: Path,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """Load the fixed predecessor archive using the committed schema helpers."""
    archive, infos = _open_fixed_zip(path, PRIOR_ARCHIVE_SHA256)
    try:
        _audit_json_members(archive, infos, "prior")
        if PRIOR_RECEIPT not in infos or ROSTER_MEMBER not in infos:
            _fail("prior archive envelope is incomplete")
        receipt = _load_json(archive.read(PRIOR_RECEIPT), PRIOR_RECEIPT)
        if not isinstance(receipt, dict) or (
            receipt.get("contract") != "atypemu_solution_state_condition_catalog_v1"
            or receipt.get("entity_count") != 135
            or receipt.get("response_count") != 405
        ):
            _fail("prior archive receipt contract/count mismatch")
        roster_binding = receipt.get("roster")
        if not isinstance(roster_binding, dict) or set(roster_binding) != {"path", "sha256"}:
            _fail("prior archive roster binding is malformed")
        roster_raw = archive.read(ROSTER_MEMBER)
        if _sha256_bytes(roster_raw) != roster_binding.get("sha256"):
            _fail("prior archive roster hash mismatch")
        roster = _canonical_roster(_load_json(roster_raw, ROSTER_MEMBER))
        expected = {(bmrb, loop) for bmrb, _ in roster for loop in LOOPS}
        manifest = receipt.get("response_manifest")
        if not isinstance(manifest, list) or len(manifest) != len(expected):
            _fail("prior archive response manifest count mismatch")
        responses: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for entry in manifest:
            if not isinstance(entry, dict):
                _fail("prior archive response manifest entry is malformed")
            bmrb, loop = entry.get("bmrb_id"), entry.get("loop")
            if (bmrb, loop) not in expected or (bmrb, loop) in responses:
                _fail("prior archive response manifest is duplicate or unallowlisted")
            relative = "raw_api_responses/%s.%s.json" % (bmrb, loop)
            _manifest_entry(entry, bmrb, loop, relative)
            member = PRIOR_ROOT + "/" + relative
            if member not in infos:
                _fail("prior archive response member is absent")
            raw = archive.read(member)
            if _sha256_bytes(raw) != entry["sha256"]:
                _fail("prior archive response member hash mismatch")
            responses[(bmrb, loop)] = _parse_prior_response(_load_json(raw, member), bmrb, loop)
        if set(responses) != expected:
            _fail("prior archive response manifest is incomplete")
        return roster, responses
    finally:
        archive.close()


def _read_v6_assigned_archive(
    path: Path, roster: Sequence[Tuple[str, str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Read only v6 Assigned-list response members and their manifest bindings."""
    archive, infos = _open_fixed_zip(path, V6_ARCHIVE_SHA256)
    try:
        _audit_json_members(archive, infos, "v6")
        if V6_MANIFEST not in infos:
            _fail("v6 archive manifest is absent")
        manifest = _load_json(archive.read(V6_MANIFEST), V6_MANIFEST)
        manifest_keys = {
            "artifact_kind", "authorization_consumed", "candidate_id", "contract",
            "member_count", "members", "outer_or_formal_metrics_opened",
            "science_executed", "source_scores_read", "target_atom_identities_read",
            "target_values_read",
        }
        if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
            _fail("v6 archive manifest schema is malformed")
        if (
            manifest["artifact_kind"]
            != "target_unread_openmm86_deposited_ph_recovery_v6_archive_manifest"
            or manifest["candidate_id"]
            != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
            or manifest["contract"]
            != "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_archive_v1"
            or any(manifest[name] is not False for name in manifest_keys if name in CLOSED_CAPABILITIES)
            or type(manifest["member_count"]) is not int
            or not isinstance(manifest["members"], list)
            or manifest["member_count"] != 140
            or manifest["member_count"] != len(manifest["members"])
        ):
            _fail("v6 archive manifest values are malformed")
        listed: Dict[str, Dict[str, Any]] = {}
        for item in manifest["members"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
                _fail("v6 archive manifest member schema is malformed")
            relative, digest, size = item["path"], item["sha256"], item["size"]
            if (
                type(relative) is not str
                or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int
                or size < 0
                or relative in listed
            ):
                _fail("v6 archive manifest member values are malformed")
            _safe_member_name(relative)
            member = V6_ROOT + "/artifacts/" + relative
            if member not in infos or infos[member].file_size != size:
                _fail("v6 archive manifest member is absent or has the wrong size")
            raw = archive.read(member)
            if _sha256_bytes(raw) != digest:
                _fail("v6 archive manifest member hash mismatch")
            listed[relative] = item
        expected_names = {V6_MANIFEST} | {
            V6_ROOT + "/artifacts/" + relative for relative in listed
        }
        if set(infos) != expected_names:
            _fail("v6 archive has unmanifested members")

        assigned: Dict[str, List[Dict[str, Any]]] = {}
        for bmrb, _ in roster:
            relative = "%s/raw_api_responses/%s.Assigned_chem_shift_list.json" % (V6_API, bmrb)
            if relative not in listed:
                _fail("v6 Assigned-list response member is absent")
            member = V6_ROOT + "/artifacts/" + relative
            raw = archive.read(member)
            if _sha256_bytes(raw) != listed[relative]["sha256"]:
                _fail("v6 Assigned-list response hash mismatch")
            assigned[bmrb] = _parse_assigned_response(_load_json(raw, member), bmrb)
        if set(assigned) != {bmrb for bmrb, _ in roster}:
            _fail("v6 Assigned-list response identities are incomplete")
        return assigned
    finally:
        archive.close()


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
        _fail("candidate source path is indirect or misplaced")
    return root


def _read_regular(root: Path, relative: Path) -> bytes:
    path = root / relative
    details = os.lstat(path)
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        _fail("bound source is not a direct regular file: %s" % relative)
    if details.st_size > MAX_SOURCE_BYTES:
        _fail("bound source is unexpectedly too large: %s" % relative)
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        raw = b""
        while len(raw) <= MAX_SOURCE_BYTES:
            block = os.read(descriptor, min(131072, MAX_SOURCE_BYTES + 1 - len(raw)))
            if not block:
                break
            raw += block
    finally:
        os.close(descriptor)
    if len(raw) > MAX_SOURCE_BYTES:
        _fail("bound source is unexpectedly too large: %s" % relative)
    return raw


def _read_bound_regular(root: Path, relative: Path, digest: str) -> bytes:
    raw = _read_regular(root, relative)
    if hashlib.sha256(raw).hexdigest() != digest:
        _fail("bound source bytes drifted: %s" % relative)
    return raw


def _git(root: Path, arguments: Sequence[str]) -> bytes:
    process = subprocess.run(
        list(arguments), cwd=str(root), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if process.returncode != 0 or len(process.stdout) > MAX_SOURCE_BYTES:
        _fail("cannot inspect committed candidate bytes")
    return process.stdout


def _validate_intent(
    intent: Any, revision: str, source_sha256: str,
) -> None:
    expected = {
        "artifact_kind": "hold_only_metadata_reinterpretation_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": PLAN_RAW_SHA256,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    if intent != expected:
        _fail("external execution intent schema or reviewed bindings drifted")


def _execution_provenance(root: Path) -> Dict[str, str]:
    bound = (
        (PLAN_RELATIVE, PLAN_RAW_SHA256),
        (PARENT_PLAN_RELATIVE, PARENT_PLAN_RAW_SHA256),
        (V6_EVIDENCE_RECEIPT_RELATIVE, V6_EVIDENCE_RECEIPT_RAW_SHA256),
        (V6_SEMANTIC_CHECKER_RELATIVE, V6_SEMANTIC_CHECKER_RAW_SHA256),
    )
    for relative, digest in bound:
        _read_bound_regular(root, relative, digest)
    source = root / CANDIDATE_RELATIVE
    source_raw = source.read_bytes()
    if len(source_raw) > MAX_SOURCE_BYTES:
        _fail("candidate source is too large")
    revision = _git(root, ("git", "rev-parse", "HEAD")).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        _fail("committed HEAD is malformed")
    for relative, current in (
        (PLAN_RELATIVE, _read_bound_regular(root, PLAN_RELATIVE, PLAN_RAW_SHA256)),
        (CANDIDATE_RELATIVE, source_raw),
    ):
        committed = _git(
            root, ("git", "show", "%s:%s" % (revision, relative.as_posix()))
        )
        if committed != current:
            _fail("candidate execution bytes differ from committed HEAD")
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    intent_raw = _read_regular(root, INTENT_RELATIVE)
    intent = _load_json(intent_raw, INTENT_RELATIVE.as_posix())
    _validate_intent(intent, revision, source_sha256)
    return {
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "execution_intent_path": INTENT_RELATIVE.as_posix(),
        "execution_intent_raw_sha256": hashlib.sha256(intent_raw).hexdigest(),
        "plan_raw_sha256": PLAN_RAW_SHA256,
    }


def _receipt(
    records: Sequence[Dict[str, Any]], provenance: Mapping[str, str],
) -> Dict[str, Any]:
    resolved = sum(1 for item in records if item["fallback_metadata_resolved"])
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_metadata_receipt",
        "bound_inputs": {
            "parent_plan_raw_sha256": PARENT_PLAN_RAW_SHA256,
            "prior_archive": {"path": ".auto/staging/" + PRIOR_NAME, "raw_sha256": PRIOR_ARCHIVE_SHA256},
            "recovery_v6_archive": {"path": ".auto/staging/" + V6_NAME, "raw_sha256": V6_ARCHIVE_SHA256},
            "recovery_v6_evidence_receipt_raw_sha256": V6_EVIDENCE_RECEIPT_RAW_SHA256,
            "recovery_v6_semantic_checker_raw_sha256": V6_SEMANTIC_CHECKER_RAW_SHA256,
        },
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "combined_ph_feasible_entity_count": 115 + resolved,
        "contract": CONTRACT,
        "entities": list(records),
        "entity_count": 135,
        "execution_provenance": dict(provenance),
        "fallback_candidate_entity_count": len(records),
        "fallback_metadata_resolved_entity_count": resolved,
        "recovery_v6_ph_feasible_entity_count": 115,
        "recovery_v6_ph_hold_entity_count": 20,
        "metadata_resolution_limit": "Metadata resolution cannot establish protonation feasibility, all-atom feasibility, support feasibility, or science feasibility.",
        "output": {"path": OUTPUT_RELATIVE.as_posix(), "receipt_creation": "O_EXCL"},
        "status": "HOLD_METADATA_REINTERPRETATION_ONLY",
    }


def _ensure_output_parent(root: Path) -> Path:
    """Create only direct directory parents; reject any existing indirection."""
    current = root
    parent_relative = OUTPUT_RELATIVE.parent.parent
    for part in parent_relative.parts:
        current = current / part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            continue
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            _fail("output parent is indirect or non-directory")
    return current


def _write_receipt_once(root: Path, payload: Dict[str, Any]) -> None:
    """Create one new directory and one new deterministic receipt, never retrying."""
    parent = _ensure_output_parent(root)
    output = parent / OUTPUT_RELATIVE.parent.name
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise CandidateError("O_EXCL output directory already exists") from error
    details = os.lstat(output)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        _fail("new output directory is indirect or non-directory")
    receipt = output / "receipt.json"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        str(receipt),
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
    """Read only the two fixed archives and seal the one permitted receipt."""
    provenance = _execution_provenance(root)
    prior_path = root / ".auto" / "staging" / PRIOR_NAME
    v6_path = root / ".auto" / "staging" / V6_NAME
    roster, prior = _read_prior_archive(prior_path)
    assigned = _read_v6_assigned_archive(v6_path, roster)
    semantic_checks = verify_v6_semantics(root)
    _, v6_output = _validate_v6(v6_path, roster)
    if (
        semantic_checks != 810
        or v6_output.get("ph_feasible_entity_count") != 115
        or v6_output.get("hold_entity_count") != 20
    ):
        _fail("recovery-v6 independent semantic result drifted")
    held = {
        (item.get("bmrb_id"), item.get("entity_uid"))
        for item in v6_output.get("entities", [])
        if isinstance(item, dict) and item.get("ph_feasible") is False
    }
    if len(held) != 20:
        _fail("recovery-v6 HOLD identity set drifted")
    records = [
        _fallback_entity(bmrb, uid, prior, assigned[bmrb])
        for bmrb, uid in roster
        if (bmrb, uid) in held
    ]
    if len(records) != 20:
        _fail("fallback candidate identity set is incomplete")
    payload = _receipt(records, provenance)
    _write_receipt_once(root, payload)
    return payload


def _chem_row(experiment_id: Any, assigned_id: Any) -> Dict[str, Any]:
    row = {name: None for name in CHEM_TAGS}
    row["Experiment_ID"] = experiment_id
    row["Assigned_chem_shift_list_ID"] = assigned_id
    return row


def _experiment_row(experiment_id: Any, condition_id: Any) -> Dict[str, Any]:
    keys = min(EXPERIMENT_TAGS)
    row = {name: None for name in keys}
    row["ID"] = experiment_id
    row["Sample_condition_list_ID"] = condition_id
    return row


def _condition_row(condition_id: Any, value: Any, units: Any = "pH") -> Dict[str, Any]:
    row = {name: None for name in CONDITION_TAGS}
    row["Sample_condition_list_ID"] = condition_id
    row["Type"] = "pH"
    row["Val"] = value
    row["Val_units"] = units
    return row


def _assigned_row(entry_id: Any, assigned_id: Any, condition_id: Any) -> Dict[str, Any]:
    return {
        ASSIGNED_TAGS[0]: entry_id,
        ASSIGNED_TAGS[1]: assigned_id,
        ASSIGNED_TAGS[2]: condition_id,
    }


def _fixture(
    chem: Sequence[Dict[str, Any]],
    experiments: Sequence[Dict[str, Any]],
    variables: Sequence[Dict[str, Any]],
    assigned: Sequence[Dict[str, Any]],
) -> Tuple[str, str, Dict[Tuple[str, str], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    bmrb, uid = "bmr1", "bmrb:1:entity:1"
    prior = {
        (bmrb, "Chem_shift_experiment"): list(chem),
        (bmrb, "Experiment"): list(experiments),
        (bmrb, "Sample_condition_variable"): list(variables),
    }
    return bmrb, uid, prior, list(assigned)


def _assert_held(result: Dict[str, Any], label: str) -> None:
    if result["fallback_metadata_resolved"]:
        raise AssertionError("%s incorrectly authorized fallback" % label)


def self_test() -> int:
    """Synthetic, in-memory adversarial tests; this function never reads archives."""
    checks = 0

    # Old one-sided form: no Experiment pointer, but a direct matching list.
    fixture = _fixture(
        [_chem_row(None, "9")], [], [_condition_row("7", "7.0")],
        [_assigned_row("1", "9", "7")],
    )
    result = _fallback_entity(*fixture)
    assert result["fallback_metadata_resolved"] and result["fallback_ph"] == 7.0
    checks += 1

    # A complete link may still fall back only when its Experiment condition is
    # absent or invalid.  Here it is absent.
    fixture = _fixture(
        [_chem_row("1", "9")], [_experiment_row("1", None)],
        [_condition_row("7", 6.5)], [_assigned_row("1", "9", "7")],
    )
    result = _fallback_entity(*fixture)
    assert result["fallback_metadata_resolved"] and result["fallback_ph"] == 6.5
    checks += 1

    fixture = _fixture(
        [_chem_row("1", "9")], [_experiment_row("1", "8")],
        [_condition_row("7", "7")], [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "valid preferred condition")
    checks += 1

    fixture = _fixture(
        [_chem_row(None, "9")], [], [_condition_row("7", "7")],
        [_assigned_row("1", "9", "7"), _assigned_row("1", "10", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "multiple direct list rows")
    checks += 1

    fixture = _fixture(
        [_chem_row("1", "9")],
        [_experiment_row("1", None), _experiment_row("1", None)],
        [_condition_row("7", "7")], [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "duplicate Experiment rows")
    checks += 1

    malformed = _chem_row(None, "9")
    malformed["unexpected"] = "field"
    fixture = _fixture(
        [malformed], [], [_condition_row("7", "7")], [_assigned_row("1", "9", "7")]
    )
    _assert_held(_fallback_entity(*fixture), "extra Chem_shift_experiment field")
    checks += 1

    fixture = _fixture(
        [_chem_row(None, "not-an-id")], [], [_condition_row("7", "7")],
        [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "malformed Chem_shift_experiment field")
    checks += 1

    fixture = _fixture(
        [_chem_row(None, "9")], [],
        [_condition_row("7", "7"), _condition_row("7", "7")],
        [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "ambiguous pH rows")
    checks += 1

    fixture = _fixture(
        [_chem_row(None, "9")], [], [], [_assigned_row("1", "9", "7")]
    )
    _assert_held(_fallback_entity(*fixture), "missing pH row")
    checks += 1

    fixture = _fixture(
        [_chem_row(None, "9")], [], [_condition_row("7", "7", ".")],
        [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "BMRB missing-token pH units")
    checks += 1

    # One-sided linkage in the opposite direction could hide preferred data.
    fixture = _fixture(
        [_chem_row("1", None)], [], [_condition_row("7", "7")],
        [_assigned_row("1", "9", "7")],
    )
    _assert_held(_fallback_entity(*fixture), "one-sided valid Experiment link")
    checks += 1

    revision, source_sha256 = "a" * 40, "b" * 64
    intent = {
        "artifact_kind": "hold_only_metadata_reinterpretation_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": revision,
        "candidate_source_raw_sha256": source_sha256,
        "closed_capabilities": CLOSED_CAPABILITIES,
        "contract": CONTRACT,
        "plan_raw_sha256": PLAN_RAW_SHA256,
        "prior_archive_raw_sha256": PRIOR_ARCHIVE_SHA256,
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_ARCHIVE_SHA256,
    }
    _validate_intent(intent, revision, source_sha256)
    tampered = dict(intent)
    tampered["unexpected"] = True
    try:
        _validate_intent(tampered, revision, source_sha256)
    except CandidateError:
        pass
    else:
        raise AssertionError("execution intent accepted an unknown field")
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
        print("METRIC openmm86_unique_assigned_ph_v1_synthetic_checks=%d" % checks)
        print("METRIC target_values_read=0")
        print("METRIC source_scores_read=0")
        print("METRIC science_executed=0")
        print("METRIC authorization_consumed=0")
        print("STATUS HOLD_SELF_TEST_ONLY")
        return 0
    try:
        payload = execute(_canonical_root())
    except (CandidateError, OSError, ValueError) as error:
        print("REFUSAL HOLD-only metadata reinterpretation: %s" % error)
        return 3
    print("METRIC fallback_metadata_resolved_entity_count=%d" % payload["fallback_metadata_resolved_entity_count"])
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_METADATA_REINTERPRETATION_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
