#!/usr/bin/env python3
"""Build a HOLD-only, target-unread deposited-pH metadata catalog.

This producer intentionally does not import OpenMM or construct coordinates.  It
only joins deposited BMRB metadata under the immutable companion plan.
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
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_deposited_ph_v1"
CONTRACT = "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_v1"
API_BASE = "https://api.bmrb.io/v2"
APPLICATION_HEADER = "AtypEmu openmm86-deposited-ph-catalog-v1"
ENTITY_COUNT = 135
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1.json"
)
PLAN_RAW_SHA256 = "caefd2517fe2a133204432f20e799e56504e8ebfa103aeaa1a715c1a8992ed15"
PRODUCER_RELATIVE = Path(
    "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog.py"
)
ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip"
)
ARCHIVE_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
ROSTER_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
)
ROSTER_SHA256 = "1a2d71529c20179f06977105a573539132feb25a13adc6e1d7dc0e0e1b4c1cb9"
OUTPUT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v1"
)
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
ID_RE = re.compile(r"[1-9][0-9]*\Z")
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
MISSING = frozenset((None, "", ".", "?"))
MAX_RESPONSE_BYTES = 10_000_000

# The archived loops were collected by the predecessor catalog.  Their broad
# metadata allowlists are copied here so a newly appearing target-bearing tag
# (or an unrelated category) is never silently accepted.
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
CLOSED_FLAGS = {
    "target_values_read": False,
    "target_atom_identities_read": False,
    "source_scores_read": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "authorization_consumed": False,
    "source_construction_executed": False,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _loads(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON in %s: %s" % (label, error)) from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing(value: Any) -> bool:
    return value in MISSING or (isinstance(value, str) and not value.strip())


def _valid_id(value: Any) -> Optional[str]:
    if _missing(value):
        return None
    candidate = str(value).strip()
    return candidate if ID_RE.fullmatch(candidate) is not None else None


def _safe_relative(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fixed repository-relative path is unsafe")


def _read_regular(root: Path, relative: Path, label: str) -> bytes:
    _safe_relative(relative)
    current = root
    for piece in relative.parts:
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError("missing %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("indirect %s" % label)
    if not stat.S_ISREG(os.lstat(current).st_mode):
        raise ValueError("non-regular %s" % label)
    descriptor = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _bound_root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / PRODUCER_RELATIVE
    if (
        source != expected
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or not root.is_dir()
    ):
        raise ValueError("producer is indirect or outside its fixed repository path")
    return root


def _require_safe_parents(path: Path, root: Path) -> None:
    _safe_relative(path.relative_to(root))
    current = path.parent
    while current != root:
        if current.exists():
            details = os.lstat(current)
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise ValueError("indirect output parent: %s" % current)
        current = current.parent


def _create_output(root: Path) -> Path:
    output = root / OUTPUT_RELATIVE
    _require_safe_parents(output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)  # atomic fresh directory; never reuse an old catalog
    raw = output / "raw_api_responses"
    os.mkdir(raw, 0o700)
    return output


def _committed_bytes(root: Path, relative: Path, current: bytes, label: str) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root), check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        committed = subprocess.run(
            ["git", "show", "%s:%s" % (revision, relative.as_posix())],
            cwd=str(root), check=True, capture_output=True, timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("cannot bind %s to committed HEAD" % label) from error
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or committed != current:
        raise ValueError("%s bytes are not identical to committed HEAD" % label)
    return revision


def _load_plan(root: Path) -> Tuple[bytes, Dict[str, Any], str]:
    raw = _read_regular(root, PLAN_RELATIVE, "immutable plan")
    if _sha256(raw) != PLAN_RAW_SHA256:
        raise ValueError("immutable plan raw-byte hash drifted")
    plan = _loads(raw, "immutable plan")
    if not isinstance(plan, dict) or not (
        plan.get("artifact_kind")
        == "hold_only_target_unread_deposited_ph_catalog_plan_not_authorization"
        and plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("contract")
        == "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_plan_v1"
        and plan.get("state") == "HOLD_PENDING_TARGET_UNREAD_DEPOSITED_PH_CATALOG"
        and plan.get("closed_capabilities") == CLOSED_FLAGS
        and plan.get("output", {}).get("fresh_directory") == OUTPUT_RELATIVE.as_posix()
        and plan.get("bound_inputs", {}).get("canonical_condition_archive", {}).get("sha256")
        == ARCHIVE_SHA256
        and plan.get("bound_inputs", {}).get("roster", {}).get("sha256") == ROSTER_SHA256
    ):
        raise ValueError("immutable plan semantics drifted")
    revision = _committed_bytes(root, PLAN_RELATIVE, raw, "immutable plan")
    return raw, plan, revision


def _producer_provenance(root: Path) -> Dict[str, str]:
    raw = _read_regular(root, PRODUCER_RELATIVE, "producer source")
    revision = _committed_bytes(root, PRODUCER_RELATIVE, raw, "producer source")
    return {
        "source_producer_relative_path": PRODUCER_RELATIVE.as_posix(),
        "source_producer_sha256": _sha256(raw),
        "source_producer_git_commit": revision,
    }


def _rows(payload: bytes, entry_id: str, loop: str) -> List[Dict[str, Any]]:
    """Decode one response while rejecting all nonallowlisted/target-bearing tags."""
    if loop not in ALL_LOOPS:
        raise ValueError("nonallowlisted BMRB category")
    decoded = _loads(payload, "BMRB %s response" % loop)
    if not isinstance(decoded, dict) or set(decoded) != {entry_id}:
        raise ValueError("BMRB entry envelope drifted: %s" % loop)
    entry = decoded[entry_id]
    if entry == {}:
        return []
    if not isinstance(entry, dict) or set(entry) != {loop}:
        raise ValueError("BMRB category envelope is not exactly allowlisted: %s" % loop)
    instances = entry[loop]
    if not isinstance(instances, list):
        raise ValueError("BMRB category instances are not a list: %s" % loop)
    result: List[Dict[str, Any]] = []
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != {"category", "tags", "data"}:
            raise ValueError("BMRB loop-instance schema drifted: %s" % loop)
        if instance["category"] != "_" + loop:
            raise ValueError("BMRB category drifted: %s" % loop)
        tags = instance["tags"]
        if (
            not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or len(tags) != len(set(tags))
            or not REQUIRED_TAGS[loop].issubset(tags)
            or not set(tags).issubset(ALLOWED_TAGS[loop])
        ):
            raise ValueError("BMRB tags are incomplete, nonallowlisted, or target-bearing: %s" % loop)
        if loop == NEW_LOOP and set(tags) != REQUIRED_TAGS[NEW_LOOP]:
            raise ValueError("new BMRB category does not have exactly its three allowlisted tags")
        data = instance["data"]
        if not isinstance(data, list):
            raise ValueError("BMRB category data is not a list: %s" % loop)
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
            result.append(row)
    return result


def _number(value: Any) -> Optional[float]:
    if _missing(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _type_key(value: Any) -> str:
    return "".join(str(value).casefold().split())


def _condition_evaluation(
    condition_rows: Sequence[Dict[str, Any]], condition_id: Optional[str]
) -> Tuple[Optional[float], int, List[Dict[str, str]], List[str]]:
    """Return pH plus non-gating temperature/ionic diagnostics for one list."""
    reasons: List[str] = []
    diagnostics: List[Dict[str, str]] = []
    ph_values: List[float] = []
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
        ph_values.append(value)
    unique = sorted(set(ph_values))
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
    prior_responses: Dict[str, bytes],
    assigned_response: bytes,
) -> Dict[str, Any]:
    """Fail closed on incomplete/ambiguous linkage, but retain a HOLD summary."""
    bmrb_match = BMRB_RE.fullmatch(bmrb_id)
    if bmrb_match is None or entity_uid != "bmrb:%s:entity:1" % bmrb_match.group(1):
        raise ValueError("noncanonical roster entity identity")
    if set(prior_responses) != set(PRIOR_LOOPS):
        raise ValueError("prior response set is not exactly the three sealed loops")
    entry_id = bmrb_match.group(1)
    chem_rows = _rows(prior_responses["Chem_shift_experiment"], entry_id, "Chem_shift_experiment")
    experiment_rows = _rows(prior_responses["Experiment"], entry_id, "Experiment")
    condition_rows = _rows(
        prior_responses["Sample_condition_variable"], entry_id, "Sample_condition_variable"
    )
    assigned_rows = _rows(assigned_response, entry_id, NEW_LOOP)

    experiments: Dict[str, List[Dict[str, Any]]] = {}
    for row in experiment_rows:
        identifier = _valid_id(row["ID"])
        if identifier is not None:
            experiments.setdefault(identifier, []).append(row)

    valid_assigned: List[Tuple[str, str]] = []
    for row in assigned_rows:
        identifier = _valid_id(row["ID"])
        condition_id = _valid_id(row["Sample_condition_list_ID"])
        if identifier is not None and condition_id is not None:
            valid_assigned.append((identifier, condition_id))

    reasons: List[str] = []
    complete_links: List[Tuple[str, str]] = []
    both_missing = 0
    malformed = 0
    for row in chem_rows:
        experiment_id = _valid_id(row["Experiment_ID"])
        assigned_id = _valid_id(row["Assigned_chem_shift_list_ID"])
        raw_both_missing = _missing(row["Experiment_ID"]) and _missing(
            row["Assigned_chem_shift_list_ID"]
        )
        if experiment_id is not None and assigned_id is not None:
            complete_links.append((experiment_id, assigned_id))
        elif raw_both_missing:
            both_missing += 1
        else:
            malformed += 1
    if malformed:
        reasons.append("Chem_shift_experiment row has only one valid or malformed linkage ID")

    route = "unavailable"
    selected_assigned_ids: List[str] = []
    selected_condition_ids: List[str] = []
    selected_experiment_ids: List[str] = []
    if not chem_rows or both_missing == len(chem_rows):
        route = "direct_fallback"
        if len(assigned_rows) != 1 or len(valid_assigned) != 1:
            reasons.append("direct fallback requires exactly one valid Assigned_chem_shift_list row")
        else:
            assigned_id, condition_id = valid_assigned[0]
            selected_assigned_ids = [assigned_id]
            selected_condition_ids = [condition_id]
    elif complete_links and len(complete_links) == len(chem_rows):
        route = "exact_route"
        selected_experiment_ids = sorted({item[0] for item in complete_links})
        selected_assigned_ids = sorted({item[1] for item in complete_links})
        if len(selected_assigned_ids) != 1:
            reasons.append("complete exact links do not resolve to one assigned list")
        experiment_condition_ids: List[str] = []
        for experiment_id in selected_experiment_ids:
            matches = experiments.get(experiment_id, [])
            if len(matches) != 1:
                reasons.append("Chem_shift_experiment link does not resolve to one Experiment")
            else:
                condition_id = _valid_id(matches[0]["Sample_condition_list_ID"])
                if condition_id is None:
                    reasons.append("linked Experiment condition pointer is missing or invalid")
                else:
                    experiment_condition_ids.append(condition_id)
        selected_condition_ids = sorted(set(experiment_condition_ids))
        if len(selected_condition_ids) != 1:
            reasons.append("complete exact links do not resolve to one condition pointer")
        if len(selected_assigned_ids) == 1 and len(selected_condition_ids) == 1:
            selected_rows = [
                row
                for row in assigned_rows
                if _valid_id(row["ID"]) == selected_assigned_ids[0]
            ]
            if len(selected_rows) != 1:
                reasons.append("selected assigned list does not resolve to one fetched row")
            else:
                direct_condition_id = _valid_id(
                    selected_rows[0]["Sample_condition_list_ID"]
                )
                if direct_condition_id is None:
                    reasons.append("selected assigned-list condition pointer is missing or invalid")
                elif direct_condition_id != selected_condition_ids[0]:
                    reasons.append(
                        "direct assigned-list condition pointer contradicts Experiment"
                    )
    elif chem_rows:
        reasons.append("Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")

    deposited_ph: Optional[float] = None
    ph_record_count = 0
    diagnostics: List[Dict[str, str]] = []
    if len(selected_condition_ids) == 1:
        deposited_ph, ph_record_count, diagnostics, ph_reasons = _condition_evaluation(
            condition_rows, selected_condition_ids[0]
        )
        reasons.extend(ph_reasons)
    else:
        reasons.append("one condition-list pointer was not resolved")

    # A pH is usable only after every complete exact link selected the same
    # assigned list and therefore the same one condition-list pointer.
    if len(selected_condition_ids) != 1 or deposited_ph is None:
        reasons.append("one deposited pH was not resolved")
    return {
        "entity_uid": entity_uid,
        "bmrb_id": bmrb_id,
        "route": route,
        "complete_chem_shift_experiment_link_count": len(complete_links),
        "experiment_ids": selected_experiment_ids,
        "assigned_chem_shift_list_ids": selected_assigned_ids,
        "sample_condition_list_ids": selected_condition_ids,
        "deposited_ph": deposited_ph,
        "deposited_ph_record_count": ph_record_count,
        "temperature_ionic_diagnostics": diagnostics,
        "ph_feasible": not reasons,
        "hold_reasons": sorted(set(reasons)),
    }


def _validate_roster(raw: bytes) -> List[Dict[str, Any]]:
    if _sha256(raw) != ROSTER_SHA256:
        raise ValueError("bound roster raw SHA-256 drifted")
    roster = _loads(raw, "bound roster")
    if not isinstance(roster, dict) or set(roster) != ROSTER_FIELDS:
        raise ValueError("bound roster schema drifted")
    if not (
        roster.get("artifact_kind")
        == "target_unread_structural_entity_roster_not_authorization"
        and roster.get("contract") == "atypemu_nested_support_count_v1_entity_roster_v3"
        and roster.get("study_id") == "atypemu_nested_support_count_v1"
        and roster.get("entity_count") == ENTITY_COUNT
        and isinstance(roster.get("entities"), list)
        and len(roster["entities"]) == ENTITY_COUNT
        and all(roster.get(flag) is False for flag in (
            "authorization_consumed", "outer_or_formal_metrics_opened",
            "source_scores_read", "target_values_read",
        ))
    ):
        raise ValueError("bound roster identity/count/protected flags drifted")
    entities: List[Dict[str, Any]] = []
    seen_uids = set()
    seen_bmrb = set()
    for row in roster["entities"]:
        if not isinstance(row, dict) or set(row) != ROSTER_ENTITY_FIELDS:
            raise ValueError("bound roster entity schema drifted")
        bmrb_id = row.get("bmrb_id")
        entity_uid = row.get("entity_uid")
        match = BMRB_RE.fullmatch(bmrb_id) if isinstance(bmrb_id, str) else None
        if (
            match is None
            or entity_uid != "bmrb:%s:entity:1" % match.group(1)
            or row.get("split") != "train"
            or row.get("observer_fold") not in ("A", "B")
            or entity_uid in seen_uids
            or bmrb_id in seen_bmrb
        ):
            raise ValueError("bound roster entity identity drifted")
        seen_uids.add(entity_uid)
        seen_bmrb.add(bmrb_id)
        entities.append(row)
    return sorted(entities, key=lambda row: str(row["entity_uid"]))


def _archive_prior_responses(
    archive: zipfile.ZipFile, bmrb_id: str
) -> Tuple[Dict[str, bytes], List[Dict[str, str]]]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("canonical archive has duplicate member names")
    info_by_name = {info.filename: info for info in archive.infolist()}
    responses: Dict[str, bytes] = {}
    bindings: List[Dict[str, str]] = []
    for loop in PRIOR_LOOPS:
        member = "%s/%s.%s.json" % (V4_RAW_PREFIX, bmrb_id, loop)
        info = info_by_name.get(member)
        if info is None or info.is_dir() or info.flag_bits & 1 or info.file_size > MAX_RESPONSE_BYTES:
            raise ValueError("sealed v4 response member is absent or unsafe: %s" % member)
        payload = archive.read(info)
        if len(payload) != info.file_size:
            raise ValueError("sealed v4 response member length drifted")
        responses[loop] = payload
        bindings.append(
            {
                "archive_member": member,
                "loop": loop,
                "sha256": _sha256(payload),
            }
        )
    return responses, bindings


def _fetch_assigned_list(entry_id: str) -> Tuple[bytes, str, str]:
    tag_list = ",".join("%s.%s" % (NEW_LOOP, tag) for tag in sorted(REQUIRED_TAGS[NEW_LOOP]))
    requested_url = "%s/entry/%s/%s?%s" % (
        API_BASE,
        entry_id,
        NEW_LOOP,
        urllib.parse.urlencode({"tag_list": tag_list}),
    )
    request = urllib.request.Request(
        requested_url,
        headers={"Application": APPLICATION_HEADER, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise ValueError("BMRB API status is %s" % response.status)
        final_url = response.geturl()
        final = urllib.parse.urlsplit(final_url)
        if final.scheme != "https" or final.hostname != "api.bmrb.io":
            raise ValueError("BMRB API redirect leaves the official host")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("BMRB API response exceeds size limit")
    return payload, requested_url, final_url


def _json_bytes(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _failure_receipt(
    output: Path,
    started: str,
    error: BaseException,
    plan_binding: Optional[Dict[str, str]],
    producer: Optional[Dict[str, str]],
    response_manifest: Sequence[Dict[str, str]],
) -> None:
    receipt: Dict[str, Any] = {
        "artifact_kind": "target_unread_openmm86_deposited_ph_catalog_failure_receipt",
        "candidate_id": CANDIDATE_ID,
        "contract": CONTRACT,
        "error": "%s: %s" % (type(error).__name__, error),
        "new_response_count_written": len(response_manifest),
        "new_response_manifest": list(response_manifest),
        "started_at_utc": started,
        **CLOSED_FLAGS,
    }
    if plan_binding is not None:
        receipt["plan"] = plan_binding
    if producer is not None:
        receipt.update(producer)
    _write_new(output / "failure_receipt.json", _json_bytes(receipt))


def fetch_catalog() -> Dict[str, Any]:
    root = _bound_root()
    output = _create_output(root)
    started = _utc_now()
    _write_new(output / "started_at_utc.txt", (started + "\n").encode("utf-8"))
    plan_binding: Optional[Dict[str, str]] = None
    producer: Optional[Dict[str, str]] = None
    response_manifest: List[Dict[str, str]] = []
    try:
        plan_raw, _plan, plan_commit = _load_plan(root)
        plan_binding = {
            "path": PLAN_RELATIVE.as_posix(),
            "sha256": _sha256(plan_raw),
            "git_commit": plan_commit,
        }
        producer = _producer_provenance(root)
        archive_raw = _read_regular(root, ARCHIVE_RELATIVE, "canonical condition archive")
        if _sha256(archive_raw) != ARCHIVE_SHA256:
            raise ValueError("canonical condition archive raw SHA-256 drifted")
        roster_raw = _read_regular(root, ROSTER_RELATIVE, "bound roster")
        roster = _validate_roster(roster_raw)
        summaries: List[Dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
            for entity in roster:
                bmrb_id = str(entity["bmrb_id"])
                entry_id = bmrb_id[3:]
                prior, prior_bindings = _archive_prior_responses(archive, bmrb_id)
                assigned, requested_url, final_url = _fetch_assigned_list(entry_id)
                # Validate before persistence so target-bearing responses are never stored.
                _rows(assigned, entry_id, NEW_LOOP)
                relative = Path("raw_api_responses") / (
                    "%s.%s.json" % (bmrb_id, NEW_LOOP)
                )
                _write_new(output / relative, assigned)
                new_binding = {
                    "bmrb_id": bmrb_id,
                    "loop": NEW_LOOP,
                    "path": relative.as_posix(),
                    "sha256": _sha256(assigned),
                    "url": requested_url,
                    "final_url": final_url,
                }
                response_manifest.append(new_binding)
                summary = summarize_entity(
                    str(entity["entity_uid"]), bmrb_id, prior, assigned
                )
                summary["prior_archive_response_bindings"] = prior_bindings
                summary["new_response_binding"] = new_binding
                summaries.append(summary)
                time.sleep(0.05)
    except Exception as error:
        _failure_receipt(output, started, error, plan_binding, producer, response_manifest)
        raise
    feasible = sum(1 for summary in summaries if summary["ph_feasible"])
    receipt: Dict[str, Any] = {
        "artifact_kind": "hold_only_target_unread_openmm86_deposited_ph_catalog_receipt",
        "candidate_id": CANDIDATE_ID,
        "contract": CONTRACT,
        "metadata_scope": "target-unread metadata feasibility only",
        "api_base": API_BASE,
        "application_header": APPLICATION_HEADER,
        "archive": {"path": ARCHIVE_RELATIVE.as_posix(), "sha256": ARCHIVE_SHA256},
        "roster": {"path": ROSTER_RELATIVE.as_posix(), "sha256": _sha256(roster_raw)},
        "plan": plan_binding,
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "new_response_count": len(response_manifest),
        "new_response_manifest": response_manifest,
        "entity_count": len(summaries),
        "ph_feasible_entity_count": feasible,
        "all_entities_ph_feasible": feasible == ENTITY_COUNT,
        "entities": summaries,
        "status": (
            "PASS_DEPOSITED_PH_METADATA_COMPLETE"
            if feasible == ENTITY_COUNT
            else "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS"
        ),
        "future_consumer": {
            "declarative_only": True,
            "openmm_version": "8.6",
            "method": "Modeller.addHydrogens",
            "force_field_xml": "amber14/protein.ff14SB.xml",
            "pH_source": "deposited pH selected by this receipt",
        },
        "warnings": [
            "heuristic only; not a pKa calculation",
            "heuristic only; not a protonation ensemble calculation",
            "temperature and ionic diagnostics do not establish physicality",
        ],
        **CLOSED_FLAGS,
        **producer,
    }
    _write_new(output / "receipt.json", _json_bytes(receipt))
    return receipt


# Synthetic fixtures only: self-test never reads a roster/archive or makes a request.
def _fixture(
    *,
    chem: Optional[List[List[Any]]] = None,
    assigned: Optional[List[List[Any]]] = None,
    conditions: Optional[List[List[Any]]] = None,
    experiments: Optional[List[List[Any]]] = None,
) -> Tuple[Dict[str, bytes], bytes]:
    entry_id = "42"

    def wrapped(loop: str, tags: List[str], rows: List[List[Any]]) -> bytes:
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
            ["Experiment_ID", "Assigned_chem_shift_list_ID", "Entry_ID"], chem,
        ),
        "Experiment": wrapped(
            "Experiment",
            ["ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"], experiments,
        ),
        "Sample_condition_variable": wrapped(
            "Sample_condition_variable",
            ["Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"],
            conditions,
        ),
    }
    new = wrapped(
        NEW_LOOP, ["Entry_ID", "ID", "Sample_condition_list_ID"], assigned
    )
    return prior, new


def _summary(prior: Dict[str, bytes], assigned: bytes) -> Dict[str, Any]:
    return summarize_entity("bmrb:42:entity:1", "bmr42", prior, assigned)


def self_test() -> int:
    checks = 0
    prior, assigned = _fixture()
    complete = _summary(prior, assigned)
    assert complete["ph_feasible"] is True and complete["deposited_ph"] == 6.5
    assert complete["temperature_ionic_diagnostics"]  # explicitly non-gating bad diagnostics
    checks += 1

    # Planner/linkage negatives: partial link, mixed absent link, missing
    # Experiment, multiple lists, and invalid pH unit must all become HOLD.
    prior, assigned = _fixture(chem=[["1", ".", "42"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(chem=[["1", "7", "42"], [".", ".", "42"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(experiments=[["2", "1", "10", "42"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(chem=[["1", "7", "42"], ["1", "8", "42"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(conditions=[["pH", "6.5", "mM", "42", "10"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1

    prior, assigned = _fixture(conditions=[["pH", "6.0-7.0", "pH", "42", "10"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(conditions=[["p D", "6.5", "", "42", "10"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(conditions=[
        ["pH", "6.5", "pH", "42", "10"], ["pH", "7.0", "pH", "42", "10"]
    ])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1

    prior, assigned = _fixture(chem=[[".", ".", "42"]])
    fallback = _summary(prior, assigned)
    assert fallback["ph_feasible"] is True and fallback["route"] == "direct_fallback"
    checks += 1
    prior, assigned = _fixture(chem=[])
    assert _summary(prior, assigned)["ph_feasible"] is True
    checks += 1
    prior, assigned = _fixture(
        chem=[[".", ".", "42"]], assigned=[["42", "7", "10"], ["42", "8", "10"]]
    )
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1

    prior, assigned = _fixture(assigned=[["42", "7", "11"]])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(assigned=[])
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1
    prior, assigned = _fixture(
        chem=[["1", "7", "42"], ["2", "7", "42"]],
        experiments=[["1", "1", "10", "42"], ["2", "1", "11", "42"]],
        conditions=[
            ["pH", "6.5", "pH", "42", "10"],
            ["pH", "6.5", "pH", "42", "11"],
        ],
    )
    assert _summary(prior, assigned)["ph_feasible"] is False
    checks += 1

    prior, assigned = _fixture()
    parsed = _loads(assigned, "synthetic assigned response")
    instance = parsed["42"][NEW_LOOP][0]
    instance["tags"].append("Atom_ID")
    instance["data"][0].append("CA")
    try:
        _rows(json.dumps(parsed).encode("utf-8"), "42", NEW_LOOP)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("target-bearing tag was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "fresh.json"
        _write_new(path, b"first")
        try:
            _write_new(path, b"second")
        except FileExistsError:
            checks += 1
        else:
            raise AssertionError("O_EXCL output overwrite was accepted")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--acknowledge-target-unread-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        checks = self_test()
        print("METRIC openmm86_deposited_ph_catalog_checks=%d" % checks)
        print("STATUS PASS")
        return 0
    if not args.acknowledge_target_unread_metadata_only:
        print("REFUSAL target-unread metadata-only acknowledgement is required")
        return 3
    try:
        receipt = fetch_catalog()
    except Exception as error:
        print("STATUS CRASH_OPENMM86_DEPOSITED_PH_CATALOG")
        print("ERROR %s: %s" % (type(error).__name__, error))
        return 1
    print("METRIC deposited_ph_feasible_entities=%d/%d" % (
        receipt["ph_feasible_entity_count"], ENTITY_COUNT
    ))
    print("STATUS %s" % receipt["status"])
    return 0 if receipt["all_entities_ph_feasible"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
