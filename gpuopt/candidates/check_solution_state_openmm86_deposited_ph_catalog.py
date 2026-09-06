# ruff: noqa: UP045, TRY004
"""Independently replay a local HOLD-only deposited-pH catalog artifact.

This checker reads only the bound roster, archive metadata loops, and saved
Assigned_chem_shift_list responses.  It never imports or executes the catalog
producer, fetches data, opens targets, or performs scientific work.  A passing
local replay remains HOLD until a separately exact-bound immutable evidence
archive and execution receipt are independently checked.
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
    "gpuopt/candidates/check_solution_state_openmm86_deposited_ph_catalog.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1.json"
)
PRODUCER_RELATIVE = Path(
    "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog.py"
)
ARCHIVE_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip"
)
ROSTER_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
)
ARTIFACT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v1"
)
RECEIPT_RELATIVE = ARTIFACT_RELATIVE / "receipt.json"
RAW_DIRECTORY = "raw_api_responses"
START_MARKER = "started_at_utc.txt"

PLAN_RAW_SHA256 = "e3bc877da2c3a0c3cac90399a99aac93817bd16646f29abc15ed575b0a95d913"
PRODUCER_RAW_SHA256 = "1a449a2e966347d5bd607dcb22c5b52a59cfb2034aa6db49ae295232ed38f0ef"
ARCHIVE_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
ROSTER_SHA256 = "1a2d71529c20179f06977105a573539132feb25a13adc6e1d7dc0e0e1b4c1cb9"

CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_deposited_ph_v1"
CONTRACT = "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_v1"
PLAN_CONTRACT = "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1"
API_BASE = "https://api.bmrb.io/v2"
APPLICATION_HEADER = "AtypEmu openmm86-deposited-ph-catalog-v1"
ENTITY_COUNT = 135
MAX_RESPONSE_BYTES = 10_000_000
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
            "Experiment_ID",
            "Experiment_name",
            "Sample_ID",
            "Sample_label",
            "Sample_state",
            "Entry_ID",
            "Assigned_chem_shift_list_ID",
        )
    ),
    "Experiment": frozenset(
        (
            "ID",
            "Name",
            "Raw_data_flag",
            "NUS_flag",
            "Interleaved_flag",
            "NMR_spec_expt_ID",
            "NMR_spec_expt_label",
            "MS_expt_ID",
            "MS_expt_label",
            "SAXS_expt_ID",
            "SAXS_expt_label",
            "FRET_expt_ID",
            "FRET_expt_label",
            "EMR_expt_ID",
            "EMR_expt_label",
            "Sample_ID",
            "Sample_label",
            "Sample_state",
            "Sample_volume",
            "Sample_volume_units",
            "Sample_condition_list_ID",
            "Sample_condition_list_label",
            "Sample_spinning_rate",
            "Sample_angle",
            "NMR_tube_type",
            "NMR_spectrometer_ID",
            "NMR_spectrometer_label",
            "NMR_spectrometer_probe_ID",
            "NMR_spectrometer_probe_label",
            "NMR_spectral_processing_ID",
            "NMR_spectral_processing_label",
            "Mass_spectrometer_ID",
            "Mass_spectrometer_label",
            "Xray_instrument_ID",
            "Xray_instrument_label",
            "Fluorescence_instrument_ID",
            "Fluorescence_instrument_label",
            "EMR_instrument_ID",
            "EMR_instrument_label",
            "Chromatographic_system_ID",
            "Chromatographic_system_label",
            "Details",
            "Entry_ID",
            "Experiment_list_ID",
        )
    ),
    "Sample_condition_variable": frozenset(
        ("Type", "Val", "Val_err", "Val_units", "Entry_ID", "Sample_condition_list_ID")
    ),
    "Assigned_chem_shift_list": REQUIRED_TAGS["Assigned_chem_shift_list"],
}

ROSTER_FIELDS = frozenset(
    (
        "artifact_kind",
        "authorization_consumed",
        "contract",
        "entities",
        "entity_count",
        "outer_or_formal_metrics_opened",
        "source_commitment_relative_path",
        "source_commitment_sha256",
        "source_scores_read",
        "study_id",
        "target_values_read",
    )
)
ROSTER_ENTITY_FIELDS = frozenset(
    (
        "bmrb_id",
        "canonical_all_atom_topology_sha256",
        "canonical_atom_count",
        "canonical_heavy_atom_count",
        "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256",
        "canonical_reference_relative_path",
        "canonical_reference_support_index",
        "entity_uid",
        "observer_fold",
        "split",
    )
)
RECEIPT_FIELDS = frozenset(
    (
        "artifact_kind",
        "candidate_id",
        "contract",
        "metadata_scope",
        "api_base",
        "application_header",
        "archive",
        "roster",
        "plan",
        "started_at_utc",
        "ended_at_utc",
        "new_response_count",
        "new_response_manifest",
        "entity_count",
        "ph_feasible_entity_count",
        "all_entities_ph_feasible",
        "entities",
        "status",
        "future_consumer",
        "warnings",
        "target_values_read",
        "target_atom_identities_read",
        "source_scores_read",
        "outer_or_formal_metrics_opened",
        "science_executed",
        "authorization_consumed",
        "source_construction_executed",
        "source_producer_relative_path",
        "source_producer_sha256",
        "source_producer_git_commit",
    )
)
ENTITY_FIELDS = frozenset(
    (
        "entity_uid",
        "bmrb_id",
        "route",
        "complete_chem_shift_experiment_link_count",
        "experiment_ids",
        "assigned_chem_shift_list_ids",
        "sample_condition_list_ids",
        "deposited_ph",
        "deposited_ph_record_count",
        "temperature_ionic_diagnostics",
        "ph_feasible",
        "hold_reasons",
        "prior_archive_response_bindings",
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


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _decode(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid relative path: {label}")
    pieces = value.split("/")
    if value.startswith("/") or "\\" in value or any(
        piece in ("", ".", "..") for piece in pieces
    ):
        raise ValueError(f"unsafe relative path: {label}")
    return value


def _regular_path(root: Path, relative: Path, label: str) -> Path:
    """Resolve a fixed relative regular file while refusing all symlinks."""
    relative_text = _safe_relative(relative.as_posix(), label)
    root_real = root.resolve(strict=True)
    current = root_real
    for piece in relative_text.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError(f"missing {label}: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"symlinked {label}")
    if not stat.S_ISREG(os.lstat(current).st_mode):
        raise ValueError(f"non-regular {label}")
    return current


def _directory_path(root: Path, relative: Path, label: str) -> Path:
    """Resolve a fixed relative directory while refusing all symlinks."""
    relative_text = _safe_relative(relative.as_posix(), label)
    current = root.resolve(strict=True)
    for piece in relative_text.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError(f"missing {label}: {error}") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"indirect or non-directory {label}")
    return current


def _read_regular(
    root: Path, relative: Path, label: str, max_bytes: Optional[int] = None
) -> bytes:
    path = _regular_path(root, relative, label)
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        size = os.fstat(descriptor).st_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"{label} exceeds the byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(size + 1)
        if len(data) != size:
            raise ValueError(f"{label} changed while being read")
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
        or not root.is_dir()
    ):
        raise ValueError("checker is indirect or outside its fixed repository path")
    return root


def _git_output(root: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        return subprocess.run(
            list(arguments),
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot read committed {label}") from error


def _committed_source(root: Path, relative: Path, expected_hash: str, label: str) -> str:
    current = _read_regular(root, relative, label)
    if _sha256(current) != expected_hash:
        raise ValueError(f"{label} raw SHA-256 drifted")
    head = _git_output(root, ("git", "rev-parse", "HEAD"), "HEAD").decode().strip()
    if COMMIT_RE.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full commit ID")
    committed = _git_output(
        root, ("git", "show", f"{head}:{relative.as_posix()}"), label
    )
    if committed != current:
        raise ValueError(f"{label} bytes do not match committed HEAD")
    return head


def _require_committed_checker(root: Path) -> str:
    """Require this checker to be the exact regular file committed at HEAD.

    Its external raw hash is bound by the parent feasibility plan; self-hashing
    here would create a circular source constant.
    """
    current = _read_regular(root, CHECKER_RELATIVE, "checker source")
    head = _git_output(root, ("git", "rev-parse", "HEAD"), "HEAD").decode().strip()
    if COMMIT_RE.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full commit ID")
    committed = _git_output(
        root, ("git", "show", f"{head}:{CHECKER_RELATIVE.as_posix()}"), "checker source"
    )
    if committed != current:
        raise ValueError("checker source bytes do not match committed HEAD")
    return head


def _commit_has_source(
    root: Path, commit: Any, relative: Path, expected_hash: str, label: str
) -> str:
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"invalid committed {label} revision")
    kind = _git_output(root, ("git", "cat-file", "-t", commit), label).decode().strip()
    if kind != "commit":
        raise ValueError(f"{label} revision is not a commit")
    source = _git_output(
        root, ("git", "show", f"{commit}:{relative.as_posix()}"), label
    )
    if _sha256(source) != expected_hash:
        raise ValueError(f"recorded {label} commit does not contain bound bytes")
    return commit


def _require_ancestor(root: Path, commit: str, head: str, label: str) -> None:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"recorded {label} commit is not reachable from HEAD") from error


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid SHA-256: {label}")
    return value


def _missing(value: Any) -> bool:
    return value in MISSING or (isinstance(value, str) and not value.strip())


def _valid_id(value: Any) -> Optional[str]:
    if _missing(value) or isinstance(value, bool):
        return None
    candidate = str(value).strip()
    return candidate if ID_RE.fullmatch(candidate) is not None else None


def _canonical_identity(bmrb_id: Any, entity_uid: Any, label: str) -> tuple[str, str]:
    if not isinstance(bmrb_id, str) or not isinstance(entity_uid, str):
        raise ValueError(f"non-string identity: {label}")
    bmrb = BMRB_RE.fullmatch(bmrb_id)
    uid = ENTITY_RE.fullmatch(entity_uid)
    if bmrb is None or uid is None or bmrb.group(1) != uid.group(1):
        raise ValueError(f"noncanonical identity: {label}")
    return bmrb_id, entity_uid


def _load_plan_and_producer(root: Path) -> tuple[str, str]:
    plan_head = _committed_source(root, PLAN_RELATIVE, PLAN_RAW_SHA256, "route plan")
    producer_head = _committed_source(
        root, PRODUCER_RELATIVE, PRODUCER_RAW_SHA256, "route producer"
    )
    plan = _decode(_read_regular(root, PLAN_RELATIVE, "route plan"), "route plan")
    if not isinstance(plan, dict):
        raise ValueError("route plan root is not an object")
    bound_inputs = plan.get("bound_inputs")
    output = plan.get("output")
    if (
        plan.get("artifact_kind")
        != "hold_only_target_unread_deposited_ph_catalog_plan_not_authorization"
        or plan.get("candidate_id") != CANDIDATE_ID
        or plan.get("contract") != PLAN_CONTRACT
        or plan.get("state") != "HOLD_PENDING_TARGET_UNREAD_DEPOSITED_PH_CATALOG"
        or plan.get("closed_capabilities") != CLOSED_FLAGS
        or not isinstance(bound_inputs, dict)
        or not isinstance(output, dict)
    ):
        raise ValueError("route plan semantic binding drifted")
    archive = bound_inputs.get("canonical_condition_archive")
    roster = bound_inputs.get("roster")
    if (
        not isinstance(archive, dict)
        or archive.get("path") != ARCHIVE_RELATIVE.as_posix()
        or archive.get("sha256") != ARCHIVE_SHA256
        or archive.get("v4_raw_response_prefix") != V4_RAW_PREFIX
        or not isinstance(roster, dict)
        or roster.get("path") != ROSTER_RELATIVE.as_posix()
        or roster.get("sha256") != ROSTER_SHA256
        or roster.get("entity_count") != ENTITY_COUNT
        or output.get("fresh_directory") != ARTIFACT_RELATIVE.as_posix()
    ):
        raise ValueError("route plan fixed input/output binding drifted")
    return plan_head, producer_head


def _rows(payload: bytes, entry_id: str, loop: str) -> list[dict[str, Any]]:
    """Parse exactly one target-unread BMRB category response."""
    if loop not in ALL_LOOPS:
        raise ValueError("nonallowlisted response category")
    decoded = _decode(payload, f"BMRB {loop} response")
    if not isinstance(decoded, dict) or set(decoded) != {entry_id}:
        raise ValueError(f"BMRB entry envelope drifted: {loop}")
    entry = decoded[entry_id]
    if entry == {}:
        return []
    if not isinstance(entry, dict) or set(entry) != {loop}:
        raise ValueError(f"BMRB category envelope drifted: {loop}")
    instances = entry[loop]
    if not isinstance(instances, list):
        raise ValueError(f"BMRB category instances are not a list: {loop}")
    result: list[dict[str, Any]] = []
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != {"category", "tags", "data"}:
            raise ValueError(f"BMRB instance schema drifted: {loop}")
        if instance["category"] != "_" + loop:
            raise ValueError(f"BMRB category identity drifted: {loop}")
        tags = instance["tags"]
        if (
            not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or len(tags) != len(set(tags))
            or not REQUIRED_TAGS[loop].issubset(tags)
            or not set(tags).issubset(ALLOWED_TAGS[loop])
        ):
            raise ValueError(f"BMRB tags are incomplete or target-bearing: {loop}")
        if loop == NEW_LOOP and set(tags) != REQUIRED_TAGS[NEW_LOOP]:
            raise ValueError("new BMRB category tags are not exact")
        data = instance["data"]
        if not isinstance(data, list):
            raise ValueError(f"BMRB category data are not a list: {loop}")
        for values in data:
            if (
                not isinstance(values, list)
                or len(values) != len(tags)
                or any(isinstance(value, (dict, list)) for value in values)
            ):
                raise ValueError(f"BMRB row schema drifted: {loop}")
            row = dict(zip(tags, values))
            if str(row["Entry_ID"]).strip() != entry_id:
                raise ValueError(f"BMRB row entry identity drifted: {loop}")
            result.append(row)
    return result


def _number(value: Any) -> Optional[float]:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _type_key(value: Any) -> str:
    return "".join(str(value).casefold().split())


def _condition_evaluation(
    condition_rows: Sequence[dict[str, Any]], condition_id: Optional[str]
) -> tuple[Optional[float], int, list[dict[str, str]], list[str]]:
    """Return deposited pH and non-gating temperature/ionic diagnostics."""
    reasons: list[str] = []
    diagnostics: list[dict[str, str]] = []
    values: list[float] = []
    ph_records = 0
    if condition_id is None:
        return None, ph_records, diagnostics, ["assigned-list condition pointer is invalid"]
    for row in condition_rows:
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
        unit = "" if _missing(row["Val_units"]) else str(row["Val_units"]).strip().casefold()
        value = _number(row["Val"])
        if unit not in ("", "ph", "p.h."):
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
    entity_uid: str,
    bmrb_id: str,
    prior_responses: dict[str, bytes],
    assigned_response: bytes,
) -> dict[str, Any]:
    """Replay the plan's complete-route/fallback pH decision independently."""
    _canonical_identity(bmrb_id, entity_uid, "summary entity")
    if set(prior_responses) != set(PRIOR_LOOPS):
        raise ValueError("prior response set is not exactly the sealed three loops")
    entry_id = bmrb_id[3:]
    chem_rows = _rows(prior_responses["Chem_shift_experiment"], entry_id, PRIOR_LOOPS[0])
    experiment_rows = _rows(prior_responses["Experiment"], entry_id, PRIOR_LOOPS[1])
    condition_rows = _rows(
        prior_responses["Sample_condition_variable"], entry_id, PRIOR_LOOPS[2]
    )
    assigned_rows = _rows(assigned_response, entry_id, NEW_LOOP)

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
    complete_links: list[tuple[str, str]] = []
    both_missing = 0
    malformed = 0
    for row in chem_rows:
        experiment_id = _valid_id(row["Experiment_ID"])
        assigned_id = _valid_id(row["Assigned_chem_shift_list_ID"])
        both_are_missing = _missing(row["Experiment_ID"]) and _missing(
            row["Assigned_chem_shift_list_ID"]
        )
        if experiment_id is not None and assigned_id is not None:
            complete_links.append((experiment_id, assigned_id))
        elif both_are_missing:
            both_missing += 1
        else:
            malformed += 1
    if malformed:
        reasons.append("Chem_shift_experiment row has only one valid or malformed linkage ID")

    route = "unavailable"
    selected_experiments: list[str] = []
    selected_assigned: list[str] = []
    selected_conditions: list[str] = []
    if not chem_rows or both_missing == len(chem_rows):
        route = "direct_fallback"
        if len(assigned_rows) != 1 or len(valid_assigned) != 1:
            reasons.append("direct fallback requires exactly one valid Assigned_chem_shift_list row")
        else:
            assigned_id, condition_id = valid_assigned[0]
            selected_assigned = [assigned_id]
            selected_conditions = [condition_id]
    elif complete_links and len(complete_links) == len(chem_rows):
        route = "exact_route"
        selected_experiments = sorted({item[0] for item in complete_links})
        selected_assigned = sorted({item[1] for item in complete_links})
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
                    reasons.append(
                        "direct assigned-list condition pointer contradicts Experiment"
                    )
    elif chem_rows:
        reasons.append(
            "Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable"
        )

    deposited_ph: Optional[float] = None
    ph_record_count = 0
    diagnostics: list[dict[str, str]] = []
    if len(selected_conditions) == 1:
        deposited_ph, ph_record_count, diagnostics, ph_reasons = _condition_evaluation(
            condition_rows, selected_conditions[0]
        )
        reasons.extend(ph_reasons)
    else:
        reasons.append("one condition-list pointer was not resolved")
    if len(selected_conditions) != 1 or deposited_ph is None:
        reasons.append("one deposited pH was not resolved")
    return {
        "entity_uid": entity_uid,
        "bmrb_id": bmrb_id,
        "route": route,
        "complete_chem_shift_experiment_link_count": len(complete_links),
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
        roster.get("artifact_kind")
        != "target_unread_structural_entity_roster_not_authorization"
        or roster.get("contract") != "atypemu_nested_support_count_v1_entity_roster_v3"
        or roster.get("study_id") != "atypemu_nested_support_count_v1"
        or type(roster.get("entity_count")) is not int
        or roster["entity_count"] != ENTITY_COUNT
        or not isinstance(roster.get("entities"), list)
        or len(roster["entities"]) != ENTITY_COUNT
    ):
        raise ValueError("bound roster identity/count drifted")
    for field in (
        "authorization_consumed",
        "outer_or_formal_metrics_opened",
        "source_scores_read",
        "target_values_read",
    ):
        if type(roster.get(field)) is not bool or roster[field] is not False:
            raise ValueError(f"bound roster protected flag opened: {field}")
    if (
        _safe_relative(roster.get("source_commitment_relative_path"), "roster source")
        != roster["source_commitment_relative_path"]
    ):
        raise ValueError("unreachable roster source path")
    _require_sha256(roster.get("source_commitment_sha256"), "roster source")

    by_uid: dict[str, dict[str, Any]] = {}
    bmrb_ids = set()
    for index, row in enumerate(roster["entities"]):
        label = f"roster entity {index}"
        if not isinstance(row, dict) or set(row) != ROSTER_ENTITY_FIELDS:
            raise ValueError(f"bound roster entity schema drifted: {label}")
        bmrb_id, entity_uid = _canonical_identity(row.get("bmrb_id"), row.get("entity_uid"), label)
        if (
            type(row.get("canonical_atom_count")) is not int
            or row["canonical_atom_count"] < 1
            or type(row.get("canonical_heavy_atom_count")) is not int
            or row["canonical_heavy_atom_count"] < 1
            or type(row.get("canonical_reference_support_index")) is not int
            or row["canonical_reference_support_index"] < 1
            or row.get("split") != "train"
            or row.get("observer_fold") not in ("A", "B")
        ):
            raise ValueError(f"bound roster entity metadata drifted: {label}")
        for field in (
            "canonical_all_atom_topology_sha256",
            "canonical_heavy_topology_sha256",
            "canonical_reference_pdb_sha256",
        ):
            _require_sha256(row.get(field), f"{label}.{field}")
        path = _safe_relative(row.get("canonical_reference_relative_path"), label)
        if path != row["canonical_reference_relative_path"]:
            raise ValueError("unreachable roster reference path")
        if entity_uid in by_uid or bmrb_id in bmrb_ids:
            raise ValueError("duplicate bound roster identity")
        by_uid[entity_uid] = row
        bmrb_ids.add(bmrb_id)
    if len(by_uid) != ENTITY_COUNT or len(bmrb_ids) != ENTITY_COUNT:
        raise ValueError("bound roster is incomplete")
    return by_uid


def _load_archive(raw: bytes, roster: dict[str, dict[str, Any]]) -> dict[str, dict[str, bytes]]:
    if _sha256(raw) != ARCHIVE_SHA256:
        raise ValueError("canonical v2 evidence archive raw SHA-256 drifted")
    result: dict[str, dict[str, bytes]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise ValueError("canonical archive has duplicate member names")
            for entity_uid in sorted(roster):
                bmrb_id = str(roster[entity_uid]["bmrb_id"])
                responses: dict[str, bytes] = {}
                for loop in PRIOR_LOOPS:
                    member = f"{V4_RAW_PREFIX}/{bmrb_id}.{loop}.json"
                    try:
                        info = archive.getinfo(member)
                    except KeyError as error:
                        raise ValueError(f"sealed archive response is absent: {member}") from error
                    if (
                        info.is_dir()
                        or info.flag_bits & 1
                        or info.file_size > MAX_RESPONSE_BYTES
                        or info.filename != member
                    ):
                        raise ValueError(f"sealed archive response is unsafe: {member}")
                    payload = archive.read(info)
                    if len(payload) != info.file_size:
                        raise ValueError(f"sealed archive response length drifted: {member}")
                    _rows(payload, bmrb_id[3:], loop)
                    responses[loop] = payload
                result[entity_uid] = responses
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("canonical v2 evidence archive is unreadable") from error
    return result


def _requested_url(bmrb_id: str) -> str:
    entry_id = bmrb_id[3:]
    tags = ",".join(
        f"{NEW_LOOP}.{tag}" for tag in sorted(REQUIRED_TAGS[NEW_LOOP])
    )
    return "{}/entry/{}/{}?{}".format(
        API_BASE,
        entry_id,
        NEW_LOOP,
        urllib.parse.urlencode({"tag_list": tags}),
    )


def _validate_final_url(value: Any, bmrb_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError("final response URL is not a string")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("final response URL is malformed") from error
    expected_path = f"/v2/entry/{bmrb_id[3:]}/{NEW_LOOP}"
    expected_query = urllib.parse.urlsplit(_requested_url(bmrb_id)).query
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.bmrb.io"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
        or parsed.path != expected_path
        or parsed.query != expected_query
    ):
        raise ValueError("final response URL leaves the official API route")
    return value


def _expected_binding(bmrb_id: str) -> dict[str, str]:
    return {
        "bmrb_id": bmrb_id,
        "loop": NEW_LOOP,
        "path": f"{RAW_DIRECTORY}/{bmrb_id}.{NEW_LOOP}.json",
        "url": _requested_url(bmrb_id),
    }


def _validate_new_binding(value: Any, bmrb_id: str, label: str) -> dict[str, str]:
    expected = _expected_binding(bmrb_id)
    if not isinstance(value, dict) or set(value) != NEW_BINDING_FIELDS:
        raise ValueError(f"new response binding schema drifted: {label}")
    for field in ("bmrb_id", "loop", "path", "url"):
        if value[field] != expected[field]:
            raise ValueError(f"new response binding {field} drifted: {label}")
    _require_sha256(value.get("sha256"), label + ".sha256")
    _validate_final_url(value.get("final_url"), bmrb_id)
    return value  # type: ignore[return-value]


def _validate_payload_hash(payload: bytes, binding: dict[str, str], label: str) -> None:
    if _sha256(payload) != binding["sha256"]:
        raise ValueError(f"raw response SHA-256 mismatch: {label}")


def _validate_artifact_tree(root: Path, started_at_utc: str, expected_names: set) -> None:
    artifact = _directory_path(root, ARTIFACT_RELATIVE, "canonical catalog output")
    try:
        with os.scandir(artifact) as entries:
            observed = set()
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("symlink in canonical catalog output")
                observed.add(entry.name)
    except OSError as error:
        raise ValueError("cannot inspect canonical catalog output") from error
    if observed != {RAW_DIRECTORY, "receipt.json", START_MARKER}:
        raise ValueError("canonical catalog output inventory drifted")
    raw_directory = _directory_path(
        root, ARTIFACT_RELATIVE / RAW_DIRECTORY, "new response directory"
    )
    try:
        with os.scandir(raw_directory) as entries:
            actual = set()
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("non-regular new response entry")
                actual.add(entry.name)
    except OSError as error:
        raise ValueError("cannot inspect new response directory") from error
    if actual != expected_names:
        raise ValueError("new response directory is not the exact 135-file manifest")
    marker = _read_regular(root, ARTIFACT_RELATIVE / START_MARKER, "start marker")
    if marker != (started_at_utc + "\n").encode("utf-8"):
        raise ValueError("catalog start marker drifted")


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"invalid UTC timestamp: {label}")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"invalid UTC timestamp: {label}") from error
    return value


def _validate_receipt_static(
    receipt: Any, root: Path, plan_head: str, producer_head: str
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("receipt schema drifted")
    if (
        receipt["artifact_kind"]
        != "hold_only_target_unread_openmm86_deposited_ph_catalog_receipt"
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["contract"] != CONTRACT
        or receipt["metadata_scope"] != "target-unread metadata feasibility only"
        or receipt["api_base"] != API_BASE
        or receipt["application_header"] != APPLICATION_HEADER
    ):
        raise ValueError("receipt identity/scope drifted")
    for field in CLOSED_FLAGS:
        if type(receipt[field]) is not bool or receipt[field] is not False:
            raise ValueError(f"receipt closed capability opened: {field}")
    archive = receipt["archive"]
    roster = receipt["roster"]
    plan = receipt["plan"]
    if (
        not isinstance(archive, dict)
        or archive != {"path": ARCHIVE_RELATIVE.as_posix(), "sha256": ARCHIVE_SHA256}
        or not isinstance(roster, dict)
        or roster != {"path": ROSTER_RELATIVE.as_posix(), "sha256": ROSTER_SHA256}
        or not isinstance(plan, dict)
        or set(plan) != {"path", "sha256", "git_commit"}
        or plan.get("path") != PLAN_RELATIVE.as_posix()
        or plan.get("sha256") != PLAN_RAW_SHA256
    ):
        raise ValueError("receipt bound input binding drifted")
    if receipt["source_producer_relative_path"] != PRODUCER_RELATIVE.as_posix():
        raise ValueError("receipt producer path drifted")
    if receipt["source_producer_sha256"] != PRODUCER_RAW_SHA256:
        raise ValueError("receipt producer raw SHA-256 drifted")
    plan_commit = _commit_has_source(
        root, plan["git_commit"], PLAN_RELATIVE, PLAN_RAW_SHA256, "route plan"
    )
    producer_commit = _commit_has_source(
        root,
        receipt["source_producer_git_commit"],
        PRODUCER_RELATIVE,
        PRODUCER_RAW_SHA256,
        "route producer",
    )
    if plan_commit != producer_commit:
        raise ValueError("receipt plan and producer commits differ")
    _require_ancestor(root, plan_commit, plan_head, "route plan")
    _require_ancestor(root, producer_commit, producer_head, "route producer")
    started = _validate_timestamp(receipt["started_at_utc"], "started_at_utc")
    ended = _validate_timestamp(receipt["ended_at_utc"], "ended_at_utc")
    if ended < started:
        raise ValueError("receipt end time precedes start time")
    if receipt["future_consumer"] != FUTURE_CONSUMER or receipt["warnings"] != WARNINGS:
        raise ValueError("receipt future-consumer declaration or warnings drifted")
    if (
        type(receipt["entity_count"]) is not int
        or receipt["entity_count"] != ENTITY_COUNT
        or type(receipt["new_response_count"]) is not int
        or receipt["new_response_count"] != ENTITY_COUNT
        or not isinstance(receipt["entities"], list)
        or len(receipt["entities"]) != ENTITY_COUNT
        or not isinstance(receipt["new_response_manifest"], list)
        or len(receipt["new_response_manifest"]) != ENTITY_COUNT
        or type(receipt["ph_feasible_entity_count"]) is not int
        or not 0 <= receipt["ph_feasible_entity_count"] <= ENTITY_COUNT
        or type(receipt["all_entities_ph_feasible"]) is not bool
        or not isinstance(receipt["status"], str)
    ):
        raise ValueError("receipt counts/status schema drifted")


def _validate_summary_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != ENTITY_FIELDS:
        raise ValueError(f"entity summary schema drifted: {label}")
    _canonical_identity(value.get("bmrb_id"), value.get("entity_uid"), label)
    if value.get("route") not in ("unavailable", "direct_fallback", "exact_route"):
        raise ValueError(f"entity route drifted: {label}")
    if (
        type(value.get("complete_chem_shift_experiment_link_count")) is not int
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
        raise ValueError(f"entity summary types drifted: {label}")
    deposited = value["deposited_ph"]
    if deposited is not None and (
        type(deposited) is not float or not math.isfinite(deposited) or not 0.0 <= deposited <= 14.0
    ):
        raise ValueError(f"entity deposited pH type/range drifted: {label}")
    for name in (
        "experiment_ids",
        "assigned_chem_shift_list_ids",
        "sample_condition_list_ids",
    ):
        values = value[name]
        if (
            not all(isinstance(item, str) and ID_RE.fullmatch(item) for item in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"entity ID summary drifted: {label}")
    reasons = value["hold_reasons"]
    if (
        not all(isinstance(item, str) and item for item in reasons)
        or reasons != sorted(set(reasons))
    ):
        raise ValueError(f"entity HOLD reasons drifted: {label}")
    for diagnostic in value["temperature_ionic_diagnostics"]:
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != DIAGNOSTIC_FIELDS
            or diagnostic.get("type") not in ("temperature", "ionicstrength")
            or not isinstance(diagnostic.get("value"), str)
            or not isinstance(diagnostic.get("units"), str)
        ):
            raise ValueError(f"entity diagnostic schema drifted: {label}")
    bindings = value["prior_archive_response_bindings"]
    if len(bindings) != len(PRIOR_LOOPS):
        raise ValueError(f"prior response binding count drifted: {label}")
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != PRIOR_BINDING_FIELDS:
            raise ValueError(f"prior response binding schema drifted: {label}")
        if binding.get("loop") not in PRIOR_LOOPS:
            raise ValueError(f"prior response binding category drifted: {label}")
        _require_sha256(binding.get("sha256"), label + ".prior")
    _validate_new_binding(value["new_response_binding"], value["bmrb_id"], label + ".new")


def verify_artifact() -> tuple[int, str]:
    """Replay only the fixed local output; this does not establish immutability."""
    root = _bound_root()
    _require_committed_checker(root)
    plan_head, producer_head = _load_plan_and_producer(root)
    receipt_raw = _read_regular(root, RECEIPT_RELATIVE, "canonical catalog receipt")
    receipt = _decode(receipt_raw, "canonical catalog receipt")
    _validate_receipt_static(receipt, root, plan_head, producer_head)

    roster_raw = _read_regular(root, ROSTER_RELATIVE, "bound roster")
    roster = _validate_roster(roster_raw)
    archive_raw = _read_regular(root, ARCHIVE_RELATIVE, "canonical v2 evidence archive")
    prior_by_uid = _load_archive(archive_raw, roster)

    expected_uids = sorted(roster)
    expected_names = {
        "{}.{}.json".format(roster[uid]["bmrb_id"], NEW_LOOP) for uid in expected_uids
    }
    _validate_artifact_tree(root, receipt["started_at_utc"], expected_names)

    new_by_uid: dict[str, bytes] = {}
    manifest = receipt["new_response_manifest"]
    for index, entity_uid in enumerate(expected_uids):
        bmrb_id = str(roster[entity_uid]["bmrb_id"])
        binding = _validate_new_binding(
            manifest[index], bmrb_id, f"new_response_manifest[{index}]"
        )
        relative = ARTIFACT_RELATIVE / binding["path"]
        payload = _read_regular(
            root, relative, f"new response {index}", MAX_RESPONSE_BYTES
        )
        _validate_payload_hash(payload, binding, f"new response {index}")
        _rows(payload, bmrb_id[3:], NEW_LOOP)
        new_by_uid[entity_uid] = payload

    observed_uids = set()
    observed_bmrbs = set()
    recomputed: list[dict[str, Any]] = []
    for index, entity_uid in enumerate(expected_uids):
        observed = receipt["entities"][index]
        label = f"entities[{index}]"
        _validate_summary_shape(observed, label)
        bmrb_id = str(roster[entity_uid]["bmrb_id"])
        if observed["entity_uid"] != entity_uid or observed["bmrb_id"] != bmrb_id:
            raise ValueError(f"receipt entity roster identity drifted: {label}")
        if entity_uid in observed_uids or bmrb_id in observed_bmrbs:
            raise ValueError(f"duplicate receipt entity identity: {label}")
        observed_uids.add(entity_uid)
        observed_bmrbs.add(bmrb_id)
        expected = summarize_entity(entity_uid, bmrb_id, prior_by_uid[entity_uid], new_by_uid[entity_uid])
        expected["prior_archive_response_bindings"] = [
            {
                "archive_member": f"{V4_RAW_PREFIX}/{bmrb_id}.{loop}.json",
                "loop": loop,
                "sha256": _sha256(prior_by_uid[entity_uid][loop]),
            }
            for loop in PRIOR_LOOPS
        ]
        expected["new_response_binding"] = manifest[index]
        if observed != expected:
            raise ValueError(f"replayed entity summary differs: {label}")
        recomputed.append(expected)
    if observed_uids != set(expected_uids) or len(observed_bmrbs) != ENTITY_COUNT:
        raise ValueError("receipt entity roster is incomplete")

    feasible = sum(1 for entity in recomputed if entity["ph_feasible"])
    if receipt["ph_feasible_entity_count"] != feasible:
        raise ValueError("receipt pH-feasible aggregate differs")
    if receipt["all_entities_ph_feasible"] is not (feasible == ENTITY_COUNT):
        raise ValueError("receipt all-entities pH aggregate differs")
    expected_status = (
        "PASS_DEPOSITED_PH_METADATA_COMPLETE"
        if feasible == ENTITY_COUNT
        else "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
    )
    if receipt["status"] != expected_status:
        raise ValueError("receipt status differs")
    checks = 8 + ENTITY_COUNT * (len(PRIOR_LOOPS) + 4)
    return checks, expected_status


# Synthetic fixtures only.  self_test never reads the repository evidence.
def _fixture(
    *,
    chem: Optional[list[list[Any]]] = None,
    assigned: Optional[list[list[Any]]] = None,
    conditions: Optional[list[list[Any]]] = None,
    experiments: Optional[list[list[Any]]] = None,
) -> tuple[dict[str, bytes], bytes]:
    entry_id = "42"

    def wrapped(loop: str, tags: list[str], rows: list[list[Any]]) -> bytes:
        return json.dumps(
            {entry_id: {loop: [{"category": "_" + loop, "tags": tags, "data": rows}]}},
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
    prior = {
        "Chem_shift_experiment": wrapped(
            "Chem_shift_experiment",
            ["Experiment_ID", "Assigned_chem_shift_list_ID", "Entry_ID"],
            chem,
        ),
        "Experiment": wrapped(
            "Experiment", ["ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"], experiments
        ),
        "Sample_condition_variable": wrapped(
            "Sample_condition_variable",
            ["Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"],
            conditions,
        ),
    }
    new = wrapped(NEW_LOOP, ["Entry_ID", "ID", "Sample_condition_list_ID"], assigned)
    return prior, new


def _summary(prior: dict[str, bytes], assigned: bytes) -> dict[str, Any]:
    return summarize_entity("bmrb:42:entity:1", "bmr42", prior, assigned)


def _expect_rejected(callback: Any, label: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(f"adversarial case was accepted: {label}")


def self_test() -> int:
    """Run synthetic linkage, schema, and byte-integrity adversarial checks."""
    checks = 0
    prior, assigned = _fixture()
    complete = _summary(prior, assigned)
    assert complete["ph_feasible"] is True and complete["deposited_ph"] == 6.5
    assert complete["temperature_ionic_diagnostics"]
    checks += 1

    prior, assigned = _fixture(chem=[["1", ".", "42"]])
    assert _summary(prior, assigned)["ph_feasible"] is False  # partial cannot fall back
    checks += 1
    prior, assigned = _fixture(chem=[])
    fallback = _summary(prior, assigned)
    assert fallback["ph_feasible"] is True and fallback["route"] == "direct_fallback"
    checks += 1
    prior, assigned = _fixture(chem=[[".", ".", "42"]])
    fallback = _summary(prior, assigned)
    assert fallback["ph_feasible"] is True and fallback["route"] == "direct_fallback"
    checks += 1
    prior, assigned = _fixture(assigned=[])
    assert _summary(prior, assigned)["ph_feasible"] is False  # fetched selected row absent
    checks += 1
    prior, assigned = _fixture(assigned=[["42", "7", "11"]])
    assert _summary(prior, assigned)["ph_feasible"] is False  # pointer contradiction
    checks += 1
    prior, assigned = _fixture(conditions=[["pH", "14.1", "pH", "42", "10"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(conditions=[["pD", "6.5", "", "42", "10"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(
        conditions=[
            ["pH", "6.5", "pH", "42", "10"],
            ["pH", "7.0", "pH", "42", "10"],
        ]
    )
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1

    prior, assigned = _fixture()
    target_tag = _decode(assigned, "synthetic target tag")
    target_tag["42"][NEW_LOOP][0]["tags"].append("Atom_ID")
    target_tag["42"][NEW_LOOP][0]["data"][0].append("CA")
    _expect_rejected(
        lambda: _rows(json.dumps(target_tag).encode("utf-8"), "42", NEW_LOOP),
        "target-bearing new tag",
    )
    checks += 1

    response_binding = _expected_binding("bmr42")
    response_binding.update({"sha256": _sha256(assigned), "final_url": _requested_url("bmr42")})
    _expect_rejected(
        lambda: _validate_payload_hash(
            assigned + b" ", response_binding, "synthetic response"
        ),
        "response byte tampering",
    )
    checks += 1

    _expect_rejected(
        lambda: _validate_final_url(_requested_url("bmr42") + "&extra=1", "bmr42"),
        "final URL query extension",
    )
    checks += 1

    entity = dict(complete)
    entity["prior_archive_response_bindings"] = [
        {
            "archive_member": f"{V4_RAW_PREFIX}/bmr42.{loop}.json",
            "loop": loop,
            "sha256": "0" * 64,
        }
        for loop in PRIOR_LOOPS
    ]
    entity["new_response_binding"] = response_binding
    entity_with_unknown = dict(entity)
    entity_with_unknown["unexpected"] = False
    _expect_rejected(
        lambda: _validate_summary_shape(entity_with_unknown, "synthetic"),
        "entity unknown field",
    )
    checks += 1

    tampered_entity = dict(entity)
    tampered_entity["ph_feasible"] = False

    def entity_byte_tamper() -> None:
        _validate_summary_shape(tampered_entity, "synthetic")
        if tampered_entity != entity:
            raise ValueError("entity replay differs after byte tampering")

    _expect_rejected(entity_byte_tamper, "entity byte tampering")
    checks += 1

    receipt = {field: None for field in RECEIPT_FIELDS}
    receipt["candidate_id"] = "tampered"
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    _expect_rejected(
        lambda: _validate_receipt_static(
            _decode(receipt_bytes, "synthetic tampered receipt"),
            Path("."),
            "0" * 40,
            "0" * 40,
        ),
        "receipt byte tampering",
    )
    checks += 1

    receipt["unexpected"] = False
    _expect_rejected(
        lambda: _validate_receipt_static(receipt, Path("."), "0" * 40, "0" * 40),
        "receipt unknown field",
    )
    checks += 1

    _expect_rejected(
        lambda: _decode(b'{"42": {}, "42": {}}', "synthetic duplicate keys"),
        "duplicate response keys",
    )
    checks += 1

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "response.json"
        path.write_bytes(assigned)
        assert _sha256(path.read_bytes()) == response_binding["sha256"]
        root = Path(temporary)
        _expect_rejected(
            lambda: _read_regular(root, Path("response.json"), "oversize fixture", 1),
            "oversize response",
        )
        checks += 1
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="run synthetic tests only")
    verify = subparsers.add_parser(
        "verify-local-consistency",
        help="replay the one canonical local output without trusting it as immutable evidence",
    )
    verify.add_argument("--acknowledge-target-unread-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        checks = self_test()
        print(f"METRIC openmm86_deposited_ph_catalog_checker_self_tests={checks}")
        print("STATUS PASS")
        return 0
    if not args.acknowledge_target_unread_metadata_only:
        print("REFUSAL target-unread metadata-only acknowledgement is required")
        return 3
    try:
        checks, status = verify_artifact()
    except ValueError as error:
        print(f"REFUSAL canonical deposited-pH catalog evidence is absent or invalid: {error}")
        print("STATUS HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS")
        return 3
    print(f"METRIC openmm86_deposited_ph_catalog_checker_checks={checks}")
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC authorization_consumed=0")
    print(f"LOCAL_SEMANTIC_STATUS {status}")
    print("STATUS HOLD_PENDING_EXTERNAL_EVIDENCE_BINDING")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
