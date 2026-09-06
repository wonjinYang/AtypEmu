#!/usr/bin/env python3
"""Independently and fail-closed validate the solution-condition catalog evidence.

This checker consumes only the sealed condition receipt, its bound target-free
entity roster, and the 405 saved BMRB loop responses.  It deliberately does not
import the producer and never opens coordinate targets, scores, or
authorization material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ARTIFACT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_v4_recovery"
)
RECEIPT_RELATIVE = ARTIFACT_RELATIVE / "receipt.json"
RAW_DIRECTORY_NAME = "raw_api_responses"
ROSTER_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
)
ROSTER_SHA256 = "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
PARENT_FAILURE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_v3_recovery/failure_receipt.json"
)
PARENT_FAILURE_SHA256 = "e406ba9b96ea10f3a052457c85b5e506cb9dfd267db61d5bdb9f87af2567a704"
GRANDPARENT_FAILURE_SHA256 = (
    "e77a1c7601d385b804404f391a27e5ed85fb3c30d99f2760a9345acca9aee6a3"
)
PRODUCER_RELATIVE = "gpuopt/candidates/solution_state_condition_catalog.py"
API_BASE = "https://api.bmrb.io/v2"
APPLICATION_HEADER = "AtypEmu solution-condition-catalog-v1"
CONTRACT = "atypemu_solution_state_condition_catalog_v1"
ARTIFACT_KIND = "hold_only_target_unread_solution_condition_catalog_receipt"
CANDIDATE_ID = "atypemu_nested_support_count_v1_solution_state_protonation_support_v1"
ENTITY_COUNT = 135
LOOPS = (
    "Chem_shift_experiment",
    "Experiment",
    "Sample_condition_variable",
)

# These are deliberately local copies, not an import of the producer's policy.
REQUIRED_TAGS = {
    "Chem_shift_experiment": frozenset(
        {"Experiment_ID", "Assigned_chem_shift_list_ID", "Entry_ID"}
    ),
    "Experiment": frozenset(
        {"ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"}
    ),
    "Sample_condition_variable": frozenset(
        {"Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"}
    ),
}
ALLOWED_TAGS = {
    "Chem_shift_experiment": frozenset(
        {
            "Experiment_ID",
            "Experiment_name",
            "Sample_ID",
            "Sample_label",
            "Sample_state",
            "Entry_ID",
            "Assigned_chem_shift_list_ID",
        }
    ),
    "Experiment": frozenset(
        {
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
            "Chromatographic_column_ID",
            "Chromatographic_column_label",
            "Details",
            "Entry_ID",
            "Experiment_list_ID",
        }
    ),
    "Sample_condition_variable": frozenset(
        {
            "Type",
            "Val",
            "Val_err",
            "Val_units",
            "Entry_ID",
            "Sample_condition_list_ID",
        }
    ),
}

RECEIPT_BASE_FIELDS = frozenset(
    {
        "artifact_kind",
        "contract",
        "candidate_id",
        "api_base",
        "application_header",
        "roster",
        "started_at_utc",
        "ended_at_utc",
        "response_count",
        "response_manifest",
        "recovery_parent_failure",
        "entity_count",
        "condition_feasible_entity_count",
        "all_entities_condition_feasible",
        "entities",
        "status",
        "target_values_read",
        "target_atom_identities_read",
        "source_scores_read",
        "outer_or_formal_metrics_opened",
        "science_executed",
        "authorization_consumed",
        "source_construction_executed",
    }
)
PROVENANCE_FIELDS = (
    "source_producer_relative_path",
    "source_producer_sha256",
    "source_producer_git_commit",
)
RECEIPT_ENTITY_FIELDS = frozenset(
    {
        "entity_uid",
        "bmrb_id",
        "assigned_shift_lists",
        "experiment_links",
        "condition_feasible",
        "unique_condition_signatures",
        "hold_reasons",
        "source_response_bindings",
    }
)
SHIFT_LIST_FIELDS = frozenset(
    {
        "assigned_chem_shift_list_id",
        "sample_condition_list_ids",
        "complete_unique_condition_signature",
        "condition_signatures",
    }
)
LINK_FIELDS = frozenset(
    {
        "assigned_chem_shift_list_id",
        "experiment_id",
        "sample_condition_list_id",
        "sample_id",
    }
)
BINDING_FIELDS = frozenset({"bmrb_id", "loop", "path", "sha256", "url", "final_url"})
ROSTER_FIELDS = frozenset(
    {
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
    }
)
ROSTER_ENTITY_FIELDS = frozenset(
    {
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
    }
)
CLOSED_RECEIPT_FLAGS = (
    "target_values_read",
    "target_atom_identities_read",
    "source_scores_read",
    "outer_or_formal_metrics_opened",
    "science_executed",
    "authorization_consumed",
    "source_construction_executed",
)
MISSING = frozenset({None, "", ".", "?"})
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
ENTITY_RE = re.compile(r"bmrb:([1-9][0-9]*):entity:1\Z")
ID_RE = re.compile(r"[1-9][0-9]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class Checks:
    """A small counter which increments only after a successful invariant."""

    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
        self.count += 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _decode_json(data: bytes, label: str) -> Dict[str, Any]:
    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON in %s: %s" % (label, error)) from error
    if not isinstance(decoded, dict):
        raise ValueError("JSON root must be an object: %s" % label)
    return decoded


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid relative path: %s" % label)
    pieces = value.split("/")
    if value.startswith("/") or "\\" in value or any(
        piece in {"", ".", ".."} for piece in pieces
    ):
        raise ValueError("unsafe relative path: %s" % label)
    return value


def _regular_file(root: Path, relative: str, label: str) -> Path:
    """Resolve a known-relative regular file without following symlinks."""
    relative = _safe_relative(relative, label)
    candidate = root.joinpath(*relative.split("/"))
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root is unavailable: %s" % error) from error
    current = root_resolved
    for piece in relative.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError("missing %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("symlinked %s" % label)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("unreadable %s: %s" % (label, error)) from error
    if resolved.parent != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("path escapes repository: %s" % label)
    if not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise ValueError("not a regular file: %s" % label)
    return resolved


def _read_bound(root: Path, relative: str, label: str) -> bytes:
    path = _regular_file(root, relative, label)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError("unable to read %s: %s" % (label, error)) from error


def _safe_directory(root: Path, relative: str, label: str) -> Path:
    """Return a direct child directory only after rejecting every symlink."""
    relative = _safe_relative(relative, label)
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root is unavailable: %s" % error) from error
    current = root_resolved
    for piece in relative.split("/"):
        current = current / piece
        try:
            details = os.lstat(current)
        except OSError as error:
            raise ValueError("missing %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("indirect or non-directory %s" % label)
    return current


def _canonical_identity(bmrb_id: Any, entity_uid: Any, label: str) -> Tuple[str, str]:
    if not isinstance(bmrb_id, str) or not isinstance(entity_uid, str):
        raise ValueError("non-string entity identity: %s" % label)
    bmrb = BMRB_RE.fullmatch(bmrb_id)
    uid = ENTITY_RE.fullmatch(entity_uid)
    if bmrb is None or uid is None or bmrb.group(1) != uid.group(1):
        raise ValueError("noncanonical entity identity: %s" % label)
    return bmrb_id, entity_uid


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError("invalid SHA-256: %s" % label)
    return value


def _valid_id(value: Any, label: str) -> str:
    # Values arrive as scalar JSON values; str() matches BMRB's string-valued
    # loop fields while refusing zero, signs, and fabricated decimal IDs.
    if value in MISSING or ID_RE.fullmatch(str(value)) is None:
        raise ValueError("missing or invalid ID: %s" % label)
    return str(value)


def _number(value: Any) -> Optional[float]:
    if value in MISSING:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_condition(kind: Any, value: Any, units: Any) -> Tuple[str, float]:
    name = " ".join(str(kind).strip().lower().replace("_", " ").split())
    if name == "pressure":
        raise KeyError(name)
    if name not in {"ph", "temperature", "ionic strength"}:
        raise ValueError("nonallowlisted sample-condition type: %s" % name)
    unit = "" if units in MISSING else str(units).strip().lower()
    number = _number(value)
    if number is None:
        raise ValueError("condition value is missing or non-finite")
    if name == "ph":
        if unit not in {"", "ph"} or not 0.0 <= number <= 14.0:
            raise ValueError("invalid pH condition")
        return "solution_ph", number
    if name == "temperature":
        if unit in {"k", "kelvin"}:
            kelvin = number
        elif unit in {"c", "celsius", "deg c", "°c"}:
            kelvin = number + 273.15
        else:
            raise ValueError("invalid temperature unit")
        if not 0.0 < kelvin < 1000.0:
            raise ValueError("temperature outside physical range")
        return "solution_temperature_k", kelvin
    if unit in {"m", "mol/l", "mol/liter", "mol/litre"}:
        millimolar = number * 1000.0
    elif unit in {"mm", "mmol/l", "mmol/liter", "mmol/litre"}:
        millimolar = number
    else:
        raise ValueError("invalid ionic-strength unit")
    if not 0.0 <= millimolar < 10000.0:
        raise ValueError("ionic strength outside physical range")
    return "solution_ionic_strength_mm", millimolar


def _rows(payload: bytes, entry_id: str, loop: str) -> List[Dict[str, Any]]:
    """Decode one raw BMRB response under the independently sealed envelope."""
    if loop not in LOOPS:
        raise ValueError("nonallowlisted BMRB loop")
    decoded = _decode_json(payload, "BMRB %s response" % loop)
    if set(decoded) != {entry_id}:
        raise ValueError("BMRB entry envelope drifted: %s" % loop)
    entry = decoded[entry_id]
    if entry == {}:
        return []
    if not isinstance(entry, dict) or set(entry) != {loop}:
        raise ValueError("BMRB loop envelope drifted: %s" % loop)
    instances = entry[loop]
    if not isinstance(instances, list):
        raise ValueError("BMRB loop instances are not a list: %s" % loop)
    rows: List[Dict[str, Any]] = []
    for instance_index, instance in enumerate(instances):
        if not isinstance(instance, dict) or set(instance) != {"category", "tags", "data"}:
            raise ValueError("BMRB loop-instance schema drifted: %s" % loop)
        if instance["category"] != "_" + loop:
            raise ValueError("BMRB loop category drifted: %s" % loop)
        tags = instance["tags"]
        if (
            not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or len(tags) != len(set(tags))
            or not REQUIRED_TAGS[loop].issubset(tags)
            or not set(tags).issubset(ALLOWED_TAGS[loop])
        ):
            raise ValueError("BMRB tags are incomplete or outside the exact allowlist: %s" % loop)
        data = instance["data"]
        if not isinstance(data, list):
            raise ValueError("BMRB loop data is not a list: %s" % loop)
        for row_index, values in enumerate(data):
            if not isinstance(values, list) or len(values) != len(tags):
                raise ValueError("BMRB row width drifted: %s[%d,%d]" % (loop, instance_index, row_index))
            if any(isinstance(value, (dict, list)) for value in values):
                raise ValueError("BMRB row has a nested value: %s" % loop)
            row = dict(zip(tags, values))
            if str(row["Entry_ID"]) != entry_id:
                raise ValueError("BMRB row entry identity drifted: %s" % loop)
            rows.append(row)
    return rows


def _condition_lists(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    result: Dict[str, Dict[str, float]] = {}
    invalid_condition_ids: set = set()
    for row_index, row in enumerate(rows):
        try:
            condition_list_id = _valid_id(
                row["Sample_condition_list_ID"], "Sample_condition_list_ID"
            )
        except ValueError:
            invalid_condition_ids.add("invalid-row-%d" % row_index)
            continue
        name = " ".join(str(row["Type"]).strip().lower().replace("_", " ").split())
        if name not in {"ph", "temperature", "ionic strength", "pressure"}:
            raise ValueError("nonallowlisted sample-condition type: %s" % name)
        values = result.setdefault(condition_list_id, {})
        try:
            field, value = _normalize_condition(row["Type"], row["Val"], row["Val_units"])
        except KeyError:
            # Non-condition variables are intentionally irrelevant, but their
            # list identity is still checked above.
            continue
        except ValueError:
            invalid_condition_ids.add(condition_list_id)
            continue
        if field in values and values[field] != value:
            raise ValueError("contradictory duplicate condition value")
        values[field] = value
    return result, sorted(invalid_condition_ids)


def _signature(values: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
    fields = (
        "solution_ph",
        "solution_temperature_k",
        "solution_ionic_strength_mm",
    )
    if any(field not in values for field in fields):
        return None
    return (values[fields[0]], values[fields[1]], values[fields[2]])


def summarize_entity(entity_uid: str, bmrb_id: str, responses: Dict[str, bytes]) -> Dict[str, Any]:
    """Reconstruct the producer's condition feasibility decision from raw data."""
    _canonical_identity(bmrb_id, entity_uid, "summary entity")
    if set(responses) != set(LOOPS):
        raise ValueError("entity does not have exactly the three BMRB responses")
    entry_id = bmrb_id[3:]
    chem_rows = _rows(responses["Chem_shift_experiment"], entry_id, "Chem_shift_experiment")
    experiment_rows = _rows(responses["Experiment"], entry_id, "Experiment")
    condition_rows = _rows(
        responses["Sample_condition_variable"], entry_id, "Sample_condition_variable"
    )

    experiments: Dict[str, Dict[str, Any]] = {}
    invalid_linkage_rows = 0
    for row in experiment_rows:
        try:
            experiment_id = _valid_id(row["ID"], "Experiment.ID")
        except ValueError:
            invalid_linkage_rows += 1
            continue
        if experiment_id in experiments:
            raise ValueError("duplicate Experiment.ID")
        experiments[experiment_id] = row
    conditions, invalid_condition_ids = _condition_lists(condition_rows)

    by_shift_list: Dict[str, set] = {}
    unresolved: List[str] = []
    links: List[Dict[str, str]] = []
    seen_chem_links: set = set()
    for row in chem_rows:
        try:
            experiment_id = _valid_id(
                row["Experiment_ID"], "Chem_shift_experiment.Experiment_ID"
            )
            shift_list_id = _valid_id(
                row["Assigned_chem_shift_list_ID"],
                "Chem_shift_experiment.Assigned_chem_shift_list_ID",
            )
        except ValueError:
            invalid_linkage_rows += 1
            continue
        raw_link = (shift_list_id, experiment_id)
        if raw_link in seen_chem_links:
            raise ValueError("duplicate Chem_shift_experiment linkage")
        seen_chem_links.add(raw_link)
        experiment = experiments.get(experiment_id)
        if experiment is None:
            unresolved.append(experiment_id)
            continue
        try:
            condition_list_id = _valid_id(
                experiment["Sample_condition_list_ID"], "Experiment.Sample_condition_list_ID"
            )
            sample_id = _valid_id(experiment["Sample_ID"], "Experiment.Sample_ID")
        except ValueError:
            unresolved.append(experiment_id)
            continue
        if condition_list_id not in conditions:
            unresolved.append(experiment_id)
            continue
        by_shift_list.setdefault(shift_list_id, set()).add(condition_list_id)
        links.append(
            {
                "assigned_chem_shift_list_id": shift_list_id,
                "experiment_id": experiment_id,
                "sample_condition_list_id": condition_list_id,
                "sample_id": sample_id,
            }
        )

    shift_lists: List[Dict[str, Any]] = []
    signatures: set = set()
    for shift_list_id in sorted(by_shift_list):
        condition_list_ids = sorted(by_shift_list[shift_list_id])
        resolved = [_signature(conditions[item]) for item in condition_list_ids]
        local = {item for item in resolved if item is not None}
        complete_unique = len(resolved) == len(condition_list_ids) and all(
            item is not None for item in resolved
        ) and len(local) == 1
        signatures.update(local)
        shift_lists.append(
            {
                "assigned_chem_shift_list_id": shift_list_id,
                "sample_condition_list_ids": condition_list_ids,
                "complete_unique_condition_signature": complete_unique,
                "condition_signatures": [list(item) for item in sorted(local)],
            }
        )

    reasons: List[str] = []
    if not chem_rows:
        reasons.append("no Chem_shift_experiment rows")
    if unresolved or invalid_linkage_rows:
        reasons.append("unresolved experiment-to-condition linkage")
    if invalid_condition_ids:
        reasons.append("invalid numeric condition value or units")
    if not shift_lists:
        reasons.append("no linked assigned-shift-list condition")
    if len(shift_lists) != 1:
        reasons.append("entry does not have exactly one assigned shift list")
    if any(not item["complete_unique_condition_signature"] for item in shift_lists):
        reasons.append("shift list lacks one complete condition signature")
    if len(signatures) != 1:
        reasons.append("entry does not have one cohort-usable condition signature")
    return {
        "entity_uid": entity_uid,
        "bmrb_id": bmrb_id,
        "assigned_shift_lists": shift_lists,
        "experiment_links": sorted(
            links,
            key=lambda item: (item["assigned_chem_shift_list_id"], item["experiment_id"]),
        ),
        "condition_feasible": not reasons,
        "unique_condition_signatures": [list(item) for item in sorted(signatures)],
        "hold_reasons": sorted(set(reasons)),
    }


def _validate_roster(roster: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if set(roster) != ROSTER_FIELDS:
        raise ValueError("bound entity-roster schema drifted")
    if (
        roster["artifact_kind"] != "target_unread_structural_entity_roster_not_authorization"
        or roster["contract"] != "atypemu_nested_support_count_v1_entity_roster_v3"
        or roster["study_id"] != "atypemu_nested_support_count_v1"
        or type(roster["entity_count"]) is not int
        or roster["entity_count"] != ENTITY_COUNT
        or not isinstance(roster["entities"], list)
        or len(roster["entities"]) != ENTITY_COUNT
    ):
        raise ValueError("bound entity-roster identity/count drifted")
    for field in (
        "authorization_consumed",
        "outer_or_formal_metrics_opened",
        "source_scores_read",
        "target_values_read",
    ):
        if type(roster[field]) is not bool or roster[field] is not False:
            raise ValueError("bound entity-roster protected flag opened: %s" % field)

    by_uid: Dict[str, Dict[str, Any]] = {}
    bmrb_ids: set = set()
    for index, row in enumerate(roster["entities"]):
        label = "entity-roster row %d" % index
        if not isinstance(row, dict) or set(row) != ROSTER_ENTITY_FIELDS:
            raise ValueError("bound entity-roster row schema drifted: %s" % label)
        bmrb_id, entity_uid = _canonical_identity(row["bmrb_id"], row["entity_uid"], label)
        if row["split"] != "train" or row["observer_fold"] not in {"A", "B"}:
            raise ValueError("bound entity-roster split/fold drifted: %s" % label)
        if entity_uid in by_uid or bmrb_id in bmrb_ids:
            raise ValueError("duplicate entity in bound entity roster")
        by_uid[entity_uid] = row
        bmrb_ids.add(bmrb_id)
    if len(by_uid) != ENTITY_COUNT or len(bmrb_ids) != ENTITY_COUNT:
        raise ValueError("incomplete entity roster")
    return by_uid


def _load_roster(root: Path, receipt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    roster_binding = receipt["roster"]
    if not isinstance(roster_binding, dict) or set(roster_binding) != {"path", "sha256"}:
        raise ValueError("receipt roster binding schema drifted")
    if roster_binding["path"] != ROSTER_RELATIVE.as_posix():
        raise ValueError("receipt roster path drifted")
    if roster_binding["sha256"] != ROSTER_SHA256:
        raise ValueError("receipt roster SHA-256 drifted")
    raw = _read_bound(root, ROSTER_RELATIVE.as_posix(), "bound entity roster")
    if _sha256(raw) != ROSTER_SHA256:
        raise ValueError("bound entity-roster raw SHA-256 drifted")
    return _validate_roster(_decode_json(raw, "bound entity roster"))


def _validate_recovery_parent(root: Path, receipt: Dict[str, Any]) -> None:
    binding = receipt["recovery_parent_failure"]
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("recovery-parent binding schema drifted")
    if (
        binding["path"] != PARENT_FAILURE_RELATIVE.as_posix()
        or binding["sha256"] != PARENT_FAILURE_SHA256
    ):
        raise ValueError("recovery-parent binding identity drifted")
    raw = _read_bound(root, binding["path"], "recovery parent failure receipt")
    if _sha256(raw) != PARENT_FAILURE_SHA256:
        raise ValueError("recovery parent failure receipt hash drifted")
    parent = _decode_json(raw, "recovery parent failure receipt")
    if (
        parent.get("artifact_kind")
        != "target_unread_condition_catalog_failure_receipt"
        or parent.get("response_count_written") != 264
        or parent.get("source_producer_git_commit")
        != "16f6d700ae02bab8e8003ab448a743812e379948"
        or parent.get("recovery_parent_failure")
        != {
            "path": ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_v2_recovery/failure_receipt.json",
            "sha256": GRANDPARENT_FAILURE_SHA256,
        }
        or any(
            parent.get(field) is not False
            for field in (
                "target_values_read",
                "source_scores_read",
                "science_executed",
                "authorization_consumed",
            )
        )
    ):
        raise ValueError("recovery parent failure receipt semantics drifted")


def _provenance_fields(receipt: Dict[str, Any]) -> Tuple[str, str, str]:
    fields = set(receipt) - set(RECEIPT_BASE_FIELDS)
    if fields != set(PROVENANCE_FIELDS):
        raise ValueError("receipt lacks exact producer provenance fields")
    values = tuple(receipt[field] for field in PROVENANCE_FIELDS)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("producer provenance values must be strings")
    return values  # type: ignore[return-value]


def _validate_provenance(root: Path, receipt: Dict[str, Any]) -> None:
    provenance = _provenance_fields(receipt)
    relative, digest, commit = provenance
    if relative != PRODUCER_RELATIVE:
        raise ValueError("producer source path drifted")
    _valid_sha256(digest, "producer source hash")
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("producer source commit is not a full Git revision")
    # The path is fixed and the receipt's source hash must bind the exact
    # checked-out producer bytes.  No producer code is imported or executed.
    if _sha256(_read_bound(root, relative, "producer source")) != digest:
        raise ValueError("producer source hash does not match checked-out bytes")
    try:
        object_type = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
        committed = subprocess.run(
            ["git", "show", "%s:%s" % (commit, relative)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("producer Git source cannot be replayed") from error
    if object_type != "commit":
        raise ValueError("producer revision does not name a Git commit")
    if _sha256(committed) != digest:
        raise ValueError("producer source hash does not match its recorded commit")


def _binding_for(bmrb_id: str, loop: str) -> Dict[str, str]:
    entry_id = bmrb_id[3:]
    return {
        "bmrb_id": bmrb_id,
        "loop": loop,
        "path": "%s/%s.%s.json" % (RAW_DIRECTORY_NAME, bmrb_id, loop),
        "sha256": "",  # populated only after the exact raw file is read
        "url": "%s/entry/%s?loop=%s" % (API_BASE, entry_id, loop),
        "final_url": "%s/entry/%s?loop=%s" % (API_BASE, entry_id, loop),
    }


def _validate_binding_shape(binding: Any, expected: Dict[str, str], label: str) -> Dict[str, str]:
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        raise ValueError("response binding schema drifted: %s" % label)
    if binding["bmrb_id"] != expected["bmrb_id"] or binding["loop"] != expected["loop"]:
        raise ValueError("response binding identity drifted: %s" % label)
    if binding["path"] != expected["path"]:
        raise ValueError("response binding path drifted: %s" % label)
    if binding["url"] != expected["url"] or binding["final_url"] != expected["final_url"]:
        raise ValueError("response binding URL/redirect drifted: %s" % label)
    _valid_sha256(binding["sha256"], label + ".sha256")
    return binding  # type: ignore[return-value]


def _validate_raw_tree(root: Path, expected_paths: set) -> None:
    raw_relative = ARTIFACT_RELATIVE / RAW_DIRECTORY_NAME
    raw_root = _safe_directory(root, raw_relative.as_posix(), "raw response directory")
    observed: set = set()
    try:
        with os.scandir(raw_root) as entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("unexpected non-regular raw response entry")
                observed.add(entry.name)
    except OSError as error:
        raise ValueError("unable to inspect raw response directory: %s" % error) from error
    expected_names = {path.rsplit("/", 1)[1] for path in expected_paths}
    if observed != expected_names:
        raise ValueError("raw response directory does not contain exactly 405 bound files")


def _validate_artifact_root(root: Path, started_at_utc: str) -> None:
    artifact_root = _safe_directory(
        root, ARTIFACT_RELATIVE.as_posix(), "condition-catalog artifact root"
    )
    expected = {"raw_api_responses", "receipt.json", "started_at_utc.txt"}
    observed: set = set()
    with os.scandir(artifact_root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError("symlink in condition-catalog artifact root")
            observed.add(entry.name)
    if observed != expected:
        raise ValueError("condition-catalog artifact root inventory drifted")
    started = _read_bound(
        root,
        (ARTIFACT_RELATIVE / "started_at_utc.txt").as_posix(),
        "condition-catalog start marker",
    )
    if started != (started_at_utc + "\n").encode("utf-8"):
        raise ValueError("condition-catalog start marker drifted")


def _validate_entity_summary_shape(summary: Any, label: str) -> None:
    if not isinstance(summary, dict) or set(summary) != RECEIPT_ENTITY_FIELDS:
        raise ValueError("entity summary schema drifted: %s" % label)
    if not isinstance(summary["assigned_shift_lists"], list):
        raise ValueError("assigned-shift-list summary is not a list: %s" % label)
    if not isinstance(summary["experiment_links"], list):
        raise ValueError("experiment-link summary is not a list: %s" % label)
    if not isinstance(summary["unique_condition_signatures"], list):
        raise ValueError("condition-signature summary is not a list: %s" % label)
    if not isinstance(summary["hold_reasons"], list):
        raise ValueError("hold-reason summary is not a list: %s" % label)
    if type(summary["condition_feasible"]) is not bool:
        raise ValueError("condition-feasible flag is not boolean: %s" % label)
    for shift in summary["assigned_shift_lists"]:
        if not isinstance(shift, dict) or set(shift) != SHIFT_LIST_FIELDS:
            raise ValueError("assigned-shift-list schema drifted: %s" % label)
    for link in summary["experiment_links"]:
        if not isinstance(link, dict) or set(link) != LINK_FIELDS:
            raise ValueError("experiment-link schema drifted: %s" % label)


def validate_receipt(
    receipt: Dict[str, Any], root: Path, roster_by_uid: Dict[str, Dict[str, Any]], checks: Checks
) -> None:
    if set(receipt) != RECEIPT_BASE_FIELDS | set(PROVENANCE_FIELDS):
        raise ValueError("receipt schema drifted")
    if (
        receipt["artifact_kind"] != ARTIFACT_KIND
        or receipt["contract"] != CONTRACT
        or receipt["candidate_id"] != CANDIDATE_ID
        or receipt["api_base"] != API_BASE
        or receipt["application_header"] != APPLICATION_HEADER
    ):
        raise ValueError("receipt contract/identity drifted")
    checks.count += 2
    for flag in CLOSED_RECEIPT_FLAGS:
        checks.require(
            type(receipt[flag]) is bool and receipt[flag] is False,
            "receipt protected flag opened: %s" % flag,
        )
    if not isinstance(receipt["started_at_utc"], str) or not isinstance(receipt["ended_at_utc"], str):
        raise ValueError("receipt timestamps are not strings")
    if receipt["ended_at_utc"] < receipt["started_at_utc"]:
        raise ValueError("receipt end timestamp precedes start timestamp")
    checks.count += 1
    _validate_artifact_root(root, receipt["started_at_utc"])
    checks.count += 1
    if type(receipt["entity_count"]) is not int or receipt["entity_count"] != ENTITY_COUNT:
        raise ValueError("receipt entity count is not exactly 135")
    if not isinstance(receipt["entities"], list) or len(receipt["entities"]) != ENTITY_COUNT:
        raise ValueError("receipt entity list is not exactly 135 rows")
    if type(receipt["response_count"]) is not int or receipt["response_count"] != 3 * ENTITY_COUNT:
        raise ValueError("receipt response count is not exactly 405")
    if not isinstance(receipt["response_manifest"], list) or len(receipt["response_manifest"]) != 3 * ENTITY_COUNT:
        raise ValueError("receipt response manifest is not exactly 405 rows")
    checks.count += 2
    _validate_recovery_parent(root, receipt)
    checks.count += 1
    _validate_provenance(root, receipt)

    roster_order = sorted(roster_by_uid)
    expected_bindings: List[Dict[str, str]] = []
    for entity_uid in roster_order:
        bmrb_id = str(roster_by_uid[entity_uid]["bmrb_id"])
        expected_bindings.extend(_binding_for(bmrb_id, loop) for loop in LOOPS)
    expected_paths = {binding["path"] for binding in expected_bindings}
    _validate_raw_tree(root, expected_paths)

    responses_by_uid: Dict[str, Dict[str, bytes]] = {}
    actual_manifest = receipt["response_manifest"]
    for index, (raw_binding, expected) in enumerate(zip(actual_manifest, expected_bindings)):
        binding = _validate_binding_shape(raw_binding, expected, "response_manifest[%d]" % index)
        payload = _read_bound(
            root,
            (ARTIFACT_RELATIVE / binding["path"]).as_posix(),
            "raw response %d" % index,
        )
        if _sha256(payload) != binding["sha256"]:
            raise ValueError("raw response SHA-256 mismatch: response_manifest[%d]" % index)
        # Parsing here closes both duplicate-key and API-envelope/category/tag
        # validation before any outcome is reconstructed.
        _rows(payload, binding["bmrb_id"][3:], binding["loop"])
        entity_uid = "bmrb:%s:entity:1" % binding["bmrb_id"][3:]
        responses_by_uid.setdefault(entity_uid, {})[binding["loop"]] = payload
        checks.count += 3

    if set(responses_by_uid) != set(roster_by_uid) or any(
        set(item) != set(LOOPS) for item in responses_by_uid.values()
    ):
        raise ValueError("response manifest does not bind exactly three loops per roster entity")
    checks.count += 1

    entity_uids: set = set()
    bmrb_ids: set = set()
    recomputed: List[Dict[str, Any]] = []
    for index, (observed, entity_uid) in enumerate(zip(receipt["entities"], roster_order)):
        label = "entities[%d]" % index
        _validate_entity_summary_shape(observed, label)
        roster_row = roster_by_uid[entity_uid]
        bmrb_id, observed_uid = _canonical_identity(observed["bmrb_id"], observed["entity_uid"], label)
        if observed_uid != entity_uid or bmrb_id != roster_row["bmrb_id"]:
            raise ValueError("receipt entity is duplicate, fake, or not in bound roster: %s" % label)
        if observed_uid in entity_uids or bmrb_id in bmrb_ids:
            raise ValueError("duplicate receipt entity identity: %s" % label)
        entity_uids.add(observed_uid)
        bmrb_ids.add(bmrb_id)
        expected = summarize_entity(observed_uid, bmrb_id, responses_by_uid[observed_uid])
        start = index * len(LOOPS)
        expected["source_response_bindings"] = actual_manifest[start : start + len(LOOPS)]
        if observed != expected:
            raise ValueError("recomputed condition summary differs: %s" % label)
        recomputed.append(expected)
        checks.count += 2
    if entity_uids != set(roster_by_uid) or len(bmrb_ids) != ENTITY_COUNT:
        raise ValueError("receipt entity roster is incomplete or fake")

    feasible = sum(bool(entity["condition_feasible"]) for entity in recomputed)
    if (
        type(receipt["condition_feasible_entity_count"]) is not int
        or receipt["condition_feasible_entity_count"] != feasible
    ):
        raise ValueError("condition-feasible entity aggregate differs")
    if receipt["all_entities_condition_feasible"] is not (feasible == ENTITY_COUNT):
        raise ValueError("all-entities condition-feasible aggregate differs")
    expected_status = (
        "PASS_CONDITION_MANIFEST_INPUTS_COMPLETE"
        if feasible == ENTITY_COUNT
        else "HOLD_CONDITION_MANIFEST_INPUTS_INCOMPLETE_OR_AMBIGUOUS"
    )
    if receipt["status"] != expected_status:
        raise ValueError("receipt condition-manifest status differs")
    checks.count += 3


def verify_artifact() -> Tuple[int, str]:
    """Validate only the fixed sealed artifact and its target-free dependencies."""
    checker = Path(__file__).absolute()
    root = checker.parents[2]
    expected_checker = root / "gpuopt/candidates/check_solution_state_condition_catalog.py"
    if checker != expected_checker or checker.is_symlink() or checker.resolve(strict=True) != checker:
        raise ValueError("checker is indirect or not at its committed relative path")
    receipt_raw = _read_bound(root, RECEIPT_RELATIVE.as_posix(), "condition receipt")
    receipt = _decode_json(receipt_raw, "condition receipt")
    checks = Checks()
    roster = _load_roster(root, receipt)
    checks.count += 2
    validate_receipt(receipt, root, roster, checks)
    return checks.count, str(receipt["status"])


# Synthetic-only fixtures for self_test; no live artifact or roster is opened.
def _fixture_responses(
    *,
    missing_ionic: bool = False,
    ambiguous: bool = False,
    celsius: bool = False,
    multiple_shift_lists: bool = False,
    invalid_ionic_units: bool = False,
) -> Dict[str, bytes]:
    entry_id = "42"

    def wrapped(loop: str, tags: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
        return json.dumps(
            {entry_id: {loop: [{"category": "_" + loop, "tags": list(tags), "data": list(rows)}]}},
            sort_keys=True,
        ).encode("utf-8")

    chem = [
        ["1", entry_id, "7"],
        ["2", entry_id, "8" if multiple_shift_lists else "7"],
    ]
    experiments = [
        ["1", "10", "1", entry_id],
        ["2", "10", "2" if ambiguous else "1", entry_id],
    ]
    conditions: List[List[Any]] = [
        ["pH", "6.5", "pH", entry_id, "1"],
        ["temperature", "26.85" if celsius else "300", "C" if celsius else "K", entry_id, "1"],
    ]
    if not missing_ionic:
        conditions.append(
            [
                "ionic strength",
                "0.15",
                "Not defined" if invalid_ionic_units else "M",
                entry_id,
                "1",
            ]
        )
    if ambiguous:
        conditions.extend(
            [
                ["pH", "5.0", "pH", entry_id, "2"],
                ["temperature", "300", "K", entry_id, "2"],
                ["ionic strength", "150", "mM", entry_id, "2"],
            ]
        )
    return {
        "Chem_shift_experiment": wrapped(
            "Chem_shift_experiment",
            ["Experiment_ID", "Entry_ID", "Assigned_chem_shift_list_ID"],
            chem,
        ),
        "Experiment": wrapped(
            "Experiment",
            ["ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"],
            experiments,
        ),
        "Sample_condition_variable": wrapped(
            "Sample_condition_variable",
            ["Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"],
            conditions,
        ),
    }


def _synthetic_roster() -> Dict[str, Any]:
    entities: List[Dict[str, Any]] = []
    digest = "0" * 64
    for number in range(1, ENTITY_COUNT + 1):
        bmrb_id = "bmr%d" % number
        entities.append(
            {
                "bmrb_id": bmrb_id,
                "canonical_all_atom_topology_sha256": digest,
                "canonical_atom_count": 1,
                "canonical_heavy_atom_count": 1,
                "canonical_heavy_topology_sha256": digest,
                "canonical_reference_pdb_sha256": digest,
                "canonical_reference_relative_path": "synthetic/%s.pdb" % bmrb_id,
                "canonical_reference_support_index": 1,
                "entity_uid": "bmrb:%d:entity:1" % number,
                "observer_fold": "A",
                "split": "train",
            }
        )
    return {
        "artifact_kind": "target_unread_structural_entity_roster_not_authorization",
        "authorization_consumed": False,
        "contract": "atypemu_nested_support_count_v1_entity_roster_v3",
        "entities": entities,
        "entity_count": ENTITY_COUNT,
        "outer_or_formal_metrics_opened": False,
        "source_commitment_relative_path": "synthetic/commitment.json",
        "source_commitment_sha256": digest,
        "source_scores_read": False,
        "study_id": "atypemu_nested_support_count_v1",
        "target_values_read": False,
    }


def self_test() -> int:
    """Exercise negative cases using only in-memory synthetic JSON fixtures."""
    checks = 0
    complete = summarize_entity("bmrb:42:entity:1", "bmr42", _fixture_responses(celsius=True))
    assert complete["condition_feasible"] is True
    assert complete["unique_condition_signatures"] == [[6.5, 300.0, 150.0]]
    checks += 1

    missing = summarize_entity(
        "bmrb:42:entity:1", "bmr42", _fixture_responses(missing_ionic=True)
    )
    assert missing["condition_feasible"] is False
    checks += 1

    missing_experiment_id = _fixture_responses()
    parsed_missing_id = _decode_json(
        missing_experiment_id["Chem_shift_experiment"],
        "synthetic missing experiment ID",
    )
    parsed_missing_id["42"]["Chem_shift_experiment"][0]["data"][0][0] = "."
    missing_experiment_id["Chem_shift_experiment"] = json.dumps(
        parsed_missing_id
    ).encode("utf-8")
    missing_link = summarize_entity(
        "bmrb:42:entity:1", "bmr42", missing_experiment_id
    )
    assert missing_link["condition_feasible"] is False
    checks += 1

    invalid_units = summarize_entity(
        "bmrb:42:entity:1",
        "bmr42",
        _fixture_responses(invalid_ionic_units=True),
    )
    assert invalid_units["condition_feasible"] is False
    assert "invalid numeric condition value or units" in invalid_units["hold_reasons"]
    checks += 1

    ambiguous = summarize_entity(
        "bmrb:42:entity:1", "bmr42", _fixture_responses(ambiguous=True)
    )
    assert ambiguous["condition_feasible"] is False
    checks += 1

    multiple = summarize_entity(
        "bmrb:42:entity:1",
        "bmr42",
        _fixture_responses(multiple_shift_lists=True),
    )
    assert multiple["condition_feasible"] is False
    checks += 1

    target_tag = _fixture_responses()
    parsed = _decode_json(target_tag["Experiment"], "synthetic target-tag response")
    instance = parsed["42"]["Experiment"][0]
    instance["tags"].append("Atom_ID")
    for row in instance["data"]:
        row.append("CA")
    try:
        _rows(json.dumps(parsed).encode("utf-8"), "42", "Experiment")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("extra target-bearing BMRB tag was accepted")

    try:
        _decode_json(b'{"42": {}, "42": {}}', "synthetic duplicate-key response")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("duplicate JSON key was accepted")

    payload = _fixture_responses()["Chem_shift_experiment"]
    expected = _binding_for("bmr42", "Chem_shift_experiment")
    tampered = dict(expected)
    tampered["sha256"] = "0" * 64
    try:
        binding = _validate_binding_shape(tampered, expected, "synthetic hash")
        if _sha256(payload) != binding["sha256"]:
            raise ValueError("synthetic raw hash mismatch")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("tampered raw response hash was accepted")

    roster = _synthetic_roster()
    duplicate = _decode_json(json.dumps(roster).encode("utf-8"), "synthetic roster")
    duplicate["entities"][1] = dict(duplicate["entities"][0])
    try:
        _validate_roster(duplicate)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("duplicate roster entity was accepted")

    fake = _decode_json(json.dumps(roster).encode("utf-8"), "synthetic roster")
    fake["entities"][0]["entity_uid"] = "bmrb:999:entity:1"
    try:
        _validate_roster(fake)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("fake roster entity was accepted")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests only")
    args = parser.parse_args()
    if args.self_test:
        checks = self_test()
        print("METRIC solution_state_condition_catalog_checker_self_tests=%d" % checks)
        print("STATUS PASS")
        return 0
    try:
        checks, condition_status = verify_artifact()
    except ValueError as error:
        # A missing receipt/raw response is a closed failure, never a reason to
        # fetch BMRB or inspect any target-bearing material.
        print("REFUSAL condition-catalog evidence is absent or invalid: %s" % error)
        print("STATUS HOLD_CONDITION_MANIFEST_INPUTS_INCOMPLETE_OR_AMBIGUOUS")
        return 3
    print("METRIC solution_state_condition_catalog_checker_checks=%d" % checks)
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC authorization_consumed=0")
    print("STATUS %s" % condition_status)
    return 0 if condition_status == "PASS_CONDITION_MANIFEST_INPUTS_COMPLETE" else 4


if __name__ == "__main__":
    raise SystemExit(main())
