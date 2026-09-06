#!/usr/bin/env python3
"""Read-only, independent audit of the recovery-v3 HOLD-only receipt.

This checker deliberately contains its own ZIP, JSON, response-schema, pH, and
fallback implementations.  It does not import candidate, checker, or parser
code from this repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import zipfile


SOURCE_RELATIVE = Path(
    "gpuopt/candidates/check_openmm86_unique_assigned_ph_recovery_v3_independent.py"
)
STAGING_RELATIVE = Path(".auto/staging")
PRIOR_NAME = "atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip"
V6_NAME = "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1.zip"
INTENT_NAME = "openmm86_unique_assigned_ph_v1_recovery_v3_execution_intent.json"
SEALED_RECEIPT_RELATIVE = Path(
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json"
)

PRIOR_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
V6_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"
INTENT_SHA256 = "44915b9125a734b739ed1262632688dab2b2cf833cc997a27b5c852ededce346"
SEALED_RECEIPT_SHA256 = "2d8a255add821950e0e381401afc8c27a97d37cffb8ead827f5d8bd500bc3b8b"

CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_unique_assigned_ph_v1_recovery_v3"
CONTRACT = CANDIDATE_ID
PRIOR_ROOT = "atypemu_nested_support_count_v1_solution_conditions_api_v2_v4_recovery"
PRIOR_RECEIPT = PRIOR_ROOT + "/receipt.json"
ROSTER_MEMBER = "atypemu_nested_support_count_v1_entity_roster_v3.json"
V6_ROOT = V6_NAME[:-4]
V6_MANIFEST = V6_ROOT + "/archive_manifest.json"
V6_API = ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v6_recovery"
V6_RECEIPT_RELATIVE = V6_API + "/receipt.json"
V6_RECEIPT = V6_ROOT + "/artifacts/" + V6_RECEIPT_RELATIVE

LOOPS = (
    "Chem_shift_experiment",
    "Experiment",
    "Sample_condition_variable",
)
CHEM_TAGS = (
    "Experiment_ID",
    "Experiment_name",
    "Sample_ID",
    "Sample_label",
    "Sample_state",
    "Entry_ID",
    "Assigned_chem_shift_list_ID",
)
CONDITION_TAGS = (
    "Type",
    "Val",
    "Val_err",
    "Val_units",
    "Entry_ID",
    "Sample_condition_list_ID",
)
ASSIGNED_TAGS = (
    "_Assigned_chem_shift_list.Entry_ID",
    "_Assigned_chem_shift_list.ID",
    "_Assigned_chem_shift_list.Sample_condition_list_ID",
)

# BMRB exposed two observed Experiment tag layouts.  Keeping both complete
# layouts makes raw-response parsing independent of any project parser.
EXPERIMENT_TAG_LAYOUTS = (
    (
        "ID", "Name", "Raw_data_flag", "NMR_spec_expt_ID", "NMR_spec_expt_label",
        "MS_expt_ID", "MS_expt_label", "SAXS_expt_ID", "SAXS_expt_label",
        "FRET_expt_ID", "FRET_expt_label", "EMR_expt_ID", "EMR_expt_label",
        "Sample_ID", "Sample_label", "Sample_state", "Sample_volume",
        "Sample_volume_units", "Sample_condition_list_ID",
        "Sample_condition_list_label", "Sample_spinning_rate", "Sample_angle",
        "NMR_tube_type", "NMR_spectrometer_ID", "NMR_spectrometer_label",
        "NMR_spectrometer_probe_ID", "NMR_spectrometer_probe_label",
        "NMR_spectral_processing_ID", "NMR_spectral_processing_label",
        "Mass_spectrometer_ID", "Mass_spectrometer_label", "Xray_instrument_ID",
        "Xray_instrument_label", "Fluorescence_instrument_ID",
        "Fluorescence_instrument_label", "EMR_instrument_ID", "EMR_instrument_label",
        "Chromatographic_system_ID", "Chromatographic_system_label",
        "Chromatographic_column_ID", "Chromatographic_column_label", "Entry_ID",
        "Experiment_list_ID",
    ),
    (
        "ID", "Name", "Raw_data_flag", "NUS_flag", "Interleaved_flag",
        "NMR_spec_expt_ID", "NMR_spec_expt_label", "MS_expt_ID", "MS_expt_label",
        "SAXS_expt_ID", "SAXS_expt_label", "FRET_expt_ID", "FRET_expt_label",
        "EMR_expt_ID", "EMR_expt_label", "Sample_ID", "Sample_label",
        "Sample_state", "Sample_volume", "Sample_volume_units",
        "Sample_condition_list_ID", "Sample_condition_list_label",
        "Sample_spinning_rate", "Sample_angle", "NMR_tube_type",
        "NMR_spectrometer_ID", "NMR_spectrometer_label",
        "NMR_spectrometer_probe_ID", "NMR_spectrometer_probe_label",
        "NMR_spectral_processing_ID", "NMR_spectral_processing_label",
        "Mass_spectrometer_ID", "Mass_spectrometer_label", "Xray_instrument_ID",
        "Xray_instrument_label", "Fluorescence_instrument_ID",
        "Fluorescence_instrument_label", "EMR_instrument_ID", "EMR_instrument_label",
        "Chromatographic_system_ID", "Chromatographic_system_label",
        "Chromatographic_column_ID", "Chromatographic_column_label", "Details",
        "Entry_ID", "Experiment_list_ID",
    ),
)
EXPERIMENT_LAYOUT_SET = frozenset(EXPERIMENT_TAG_LAYOUTS)
EXPERIMENT_KEY_SET = frozenset(frozenset(item) for item in EXPERIMENT_TAG_LAYOUTS)

CLOSED = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}

ROSTER_SHA256 = "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
PARENT_PLAN_SHA256 = "9244f880c7a67e97c5a370013b3cfc81cccc8cb9d7267b2fff0b0a1b5dddab91"
V6_EVIDENCE_RECEIPT_SHA256 = "c6fe9398b0ea44668da5cc1f8c8bd8e8b799f1659e501f4fb2b887d047462f36"
V6_SEMANTIC_CHECKER_SHA256 = "f12eb66b24c4696562e9c1833836c240cac4c30299b963761614b70903e46d84"

MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_INTENT_BYTES = 256 * 1024
MAX_ZIP_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 16 * 1024 * 1024
ID_RE = re.compile(r"[1-9][0-9]*\Z")
BMRB_RE = re.compile(r"bmr([1-9][0-9]*)\Z")
UID_RE = re.compile(r"bmrb:([1-9][0-9]*):entity:([1-9][0-9]*)\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    """A malformed, noncanonical, or semantically inconsistent artifact."""


def _fail(message: str) -> None:
    raise AuditError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _json_constant(value: str) -> None:
    _fail("non-finite JSON constant: %s" % value)


def _loads(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AuditError) as error:
        _fail("invalid JSON in %s: %s" % (label, error))


def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact(item, wanted) for item, wanted in zip(actual, expected)
        )
    return actual == expected


def _check_closed(value: Any, label: str) -> None:
    """No nested artifact may claim a closed capability was opened."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in CLOSED and item is not False:
                _fail("closed capability is not false at %s.%s" % (label, key))
            _check_closed(item, label + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_closed(item, "%s[%d]" % (label, index))


def _safe_relative(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        _fail("unsafe fixed repository path")


def _read_regular(root: Path, relative: Path, label: str, maximum: int) -> bytes:
    """Read a bounded regular file without following a path-component symlink."""
    _safe_relative(relative)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            details = os.lstat(current)
        except OSError as error:
            raise AuditError("missing %s: %s" % (label, error)) from error
        if stat.S_ISLNK(details.st_mode):
            _fail("indirect %s" % label)
    details = os.lstat(current)
    if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
        _fail("non-regular or oversized %s" % label)
    descriptor = os.open(str(current), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
            _fail("opened %s is not a bounded regular file" % label)
        chunks: List[bytes] = []
        left = opened.st_size + 1
        while left:
            block = os.read(descriptor, min(131072, left))
            if not block:
                break
            chunks.append(block)
            left -= len(block)
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            _fail("%s changed while reading" % label)
        return raw
    finally:
        os.close(descriptor)


def _read_bound(root: Path, relative: Path, digest: str, label: str, maximum: int) -> bytes:
    raw = _read_regular(root, relative, label, maximum)
    if _sha256(raw) != digest:
        _fail("%s SHA-256 drifted" % label)
    return raw


def _safe_zip_name(name: str) -> str:
    _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
             "unsafe ZIP member name")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), "absolute ZIP member name")
    parts = name.split("/")
    _require(all(part not in ("", ".", "..") for part in parts), "traversal ZIP member name")
    return name


def _zip_infos(raw: bytes, label: str, expected_count: int) -> Tuple[zipfile.ZipFile, Dict[str, zipfile.ZipInfo]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), "r")
    except zipfile.BadZipFile as error:
        raise AuditError("invalid %s ZIP: %s" % (label, error)) from error
    try:
        infos = archive.infolist()
        _require(len(infos) == expected_count, "%s ZIP member count drifted" % label)
        names: List[str] = []
        total = 0
        for info in infos:
            _safe_zip_name(info.filename)
            _require(not info.is_dir() and not (info.flag_bits & 1),
                     "%s has a directory or encrypted member" % label)
            # POSIX symlink bit in the high Unix mode bits.
            mode = (info.external_attr >> 16) & 0xFFFF
            _require(not stat.S_ISLNK(mode), "%s has a symlink member" % label)
            _require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                     "%s has an unapproved ZIP compression method" % label)
            _require(0 <= info.file_size <= MAX_ZIP_MEMBER_BYTES,
                     "%s has an oversized ZIP member" % label)
            _require(info.file_size <= max(1024, info.compress_size * 200),
                     "%s ZIP compression ratio is unsafe" % label)
            total += info.file_size
            names.append(info.filename)
        _require(total <= MAX_ZIP_TOTAL_BYTES and len(names) == len(set(names)),
                 "%s ZIP total size or member uniqueness drifted" % label)
        bad = archive.testzip()
        _require(bad is None, "%s ZIP CRC failed at %s" % (label, bad))
        return archive, {item.filename: item for item in infos}
    except BaseException:
        archive.close()
        raise


def _zip_read(archive: zipfile.ZipFile, info: zipfile.ZipInfo, label: str) -> bytes:
    try:
        raw = archive.read(info)
    except (OSError, zipfile.BadZipFile) as error:
        raise AuditError("cannot read %s: %s" % (label, error)) from error
    _require(len(raw) == info.file_size, "ZIP member size drifted: %s" % label)
    return raw


def _require_inventory(infos: Mapping[str, zipfile.ZipInfo], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    if set(infos) != expected_set:
        unknown = sorted(set(infos) - expected_set)
        missing = sorted(expected_set - set(infos))
        _fail("%s ZIP inventory drifted (unknown=%r missing=%r)" % (label, unknown[:2], missing[:2]))


def _missing(value: Any) -> bool:
    return value is None or (type(value) is str and value.strip() in ("", ".", "?"))


def _valid_id(value: Any) -> Optional[str]:
    return value if type(value) is str and ID_RE.fullmatch(value) is not None else None


def _bmrb_number(bmrb: Any) -> str:
    match = BMRB_RE.fullmatch(bmrb) if type(bmrb) is str else None
    if match is None:
        _fail("noncanonical BMRB identity: %r" % (bmrb,))
    return match.group(1)


def _normal_type(value: Any) -> str:
    return re.sub(r"\s+", "", value.casefold()) if type(value) is str else ""


def _numeric_ph(value: Any) -> Optional[float]:
    if isinstance(value, bool) or _missing(value) or type(value) not in (int, float, str):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 14.0:
        return None
    return 0.0 if number == 0.0 else number


def _response_url(bmrb: str, loop: str) -> str:
    return "https://api.bmrb.io/v2/entry/%s?loop=%s" % (_bmrb_number(bmrb), loop)


def _assigned_url(bmrb: str) -> str:
    return (
        "https://api.bmrb.io/v2/entry/%s?tag=_Assigned_chem_shift_list.Entry_ID"
        "&tag=_Assigned_chem_shift_list.ID"
        "&tag=_Assigned_chem_shift_list.Sample_condition_list_ID"
    ) % _bmrb_number(bmrb)


def _parse_prior_response(payload: Any, bmrb: str, loop: str) -> List[Dict[str, Any]]:
    number = _bmrb_number(bmrb)
    _require(isinstance(payload, dict) and set(payload) == {number},
             "unexpected %s response root" % loop)
    envelope = payload[number]
    _require(isinstance(envelope, dict) and set(envelope) == {loop},
             "unexpected %s response envelope" % loop)
    records = envelope[loop]
    _require(isinstance(records, list), "%s response records are not a list" % loop)
    expected_tags: Optional[Tuple[str, ...]]
    expected_tags = CHEM_TAGS if loop == "Chem_shift_experiment" else CONDITION_TAGS
    rows: List[Dict[str, Any]] = []
    for record in records:
        _require(isinstance(record, dict) and set(record) == {"category", "tags", "data"},
                 "unexpected %s response record" % loop)
        _require(record["category"] == "_" + loop and isinstance(record["tags"], list)
                 and isinstance(record["data"], list), "unexpected %s response shape" % loop)
        tags = tuple(record["tags"])
        if loop == "Experiment":
            _require(tags in EXPERIMENT_LAYOUT_SET, "unallowlisted Experiment response tags")
        else:
            _require(tags == expected_tags, "unallowlisted %s response tags" % loop)
        _require(not any("target" in _normal_type(tag) for tag in tags),
                 "target-bearing response tag")
        for row in record["data"]:
            _require(isinstance(row, list) and len(row) == len(tags),
                     "malformed %s response row" % loop)
            rows.append(dict(zip(tags, row)))
    return rows


def _parse_assigned_response(payload: Any, bmrb: str) -> List[Dict[str, Any]]:
    number = _bmrb_number(bmrb)
    _require(isinstance(payload, dict) and set(payload) == {number},
             "unexpected Assigned-list response root")
    columns = payload[number]
    _require(isinstance(columns, dict) and set(columns) == set(ASSIGNED_TAGS),
             "Assigned-list response is not the exact three-column schema")
    values = [columns[key] for key in ASSIGNED_TAGS]
    _require(all(isinstance(column, list) for column in values),
             "Assigned-list response columns are not lists")
    _require(len({len(column) for column in values}) == 1,
             "Assigned-list response columns have unequal lengths")
    _require(all(type(item) in (str, type(None)) for column in values for item in column),
             "Assigned-list response has non-text values")
    return [dict(zip(ASSIGNED_TAGS, row)) for row in zip(*values)]


def _parse_roster(payload: Any) -> List[Tuple[str, str]]:
    required_top = {
        "artifact_kind", "authorization_consumed", "contract", "entities", "entity_count",
        "outer_or_formal_metrics_opened", "source_commitment_relative_path",
        "source_commitment_sha256", "source_scores_read", "study_id", "target_values_read",
    }
    required_entity = {
        "bmrb_id", "canonical_all_atom_topology_sha256", "canonical_atom_count",
        "canonical_heavy_atom_count", "canonical_heavy_topology_sha256",
        "canonical_reference_pdb_sha256", "canonical_reference_relative_path",
        "canonical_reference_support_index", "entity_uid", "observer_fold", "split",
    }
    _require(isinstance(payload, dict) and set(payload) == required_top,
             "roster top-level schema drifted")
    _require(payload["artifact_kind"] == "target_unread_structural_entity_roster_not_authorization"
             and payload["contract"] == "atypemu_nested_support_count_v1_entity_roster_v3"
             and payload["entity_count"] == 135 and isinstance(payload["entities"], list)
             and len(payload["entities"]) == 135, "135-entity roster count/contract drifted")
    roster: List[Tuple[str, str]] = []
    seen_bmrb, seen_uid = set(), set()
    for item in payload["entities"]:
        _require(isinstance(item, dict) and set(item) == required_entity,
                 "roster entity schema drifted")
        bmrb, uid = item["bmrb_id"], item["entity_uid"]
        number = _bmrb_number(bmrb)
        match = UID_RE.fullmatch(uid) if type(uid) is str else None
        _require(match is not None and match.group(1) == number,
                 "roster BMRB/entity identity mismatch")
        _require(bmrb not in seen_bmrb and uid not in seen_uid,
                 "roster has duplicate canonical identity")
        seen_bmrb.add(bmrb)
        seen_uid.add(uid)
        roster.append((bmrb, uid))
    return roster


def _manifest_binding(entry: Any, bmrb: str, loop: str, path: str) -> Dict[str, Any]:
    expected = {"bmrb_id", "final_url", "loop", "path", "sha256", "url"}
    _require(isinstance(entry, dict) and set(entry) == expected,
             "response-manifest entry schema drifted")
    _require(entry["bmrb_id"] == bmrb and entry["loop"] == loop and entry["path"] == path,
             "response-manifest identity drifted")
    _require(entry["url"] == _response_url(bmrb, loop)
             and entry["final_url"] == _response_url(bmrb, loop),
             "response-manifest URL drifted")
    _require(type(entry["sha256"]) is str and SHA_RE.fullmatch(entry["sha256"]) is not None,
             "response-manifest SHA-256 is malformed")
    return entry


def _audit_all_json(archive: zipfile.ZipFile, infos: Mapping[str, zipfile.ZipInfo], label: str) -> None:
    """Detect duplicate keys/non-finite values and opened flags in every JSON member."""
    for name, info in infos.items():
        if name.endswith(".json"):
            payload = _loads(_zip_read(archive, info, label + ":" + name), label + ":" + name)
            _check_closed(payload, label + ":" + name)


def _validate_historical_inventory(
    archive: zipfile.ZipFile, infos: Mapping[str, zipfile.ZipInfo]
) -> set[str]:
    """Validate the three failed predecessor subtrees and return their exact names."""
    output: set[str] = set()
    for version, expected_count in (("v1", 39), ("v2_recovery", 261), ("v3_recovery", 264)):
        root = "atypemu_nested_support_count_v1_solution_conditions_api_v2_" + version
        receipt_name = root + "/failure_receipt.json"
        started_name = root + "/started_at_utc.txt"
        _require(receipt_name in infos and started_name in infos,
                 "historical %s inventory lacks receipt or timestamp" % version)
        receipt = _loads(_zip_read(archive, infos[receipt_name], receipt_name), receipt_name)
        _require(isinstance(receipt, dict)
                 and receipt.get("artifact_kind") == "target_unread_condition_catalog_failure_receipt"
                 and receipt.get("response_count_written") == expected_count,
                 "historical %s failure receipt drifted" % version)
        prefix = root + "/raw_api_responses/"
        raw_names = [name for name in infos if name.startswith(prefix)]
        _require(len(raw_names) == expected_count and len(raw_names) == len(set(raw_names)),
                 "historical %s response inventory drifted" % version)
        seen = set()
        for name in raw_names:
            match = re.fullmatch(
                re.escape(prefix) + r"(bmr[1-9][0-9]*)\.(Chem_shift_experiment|Experiment|Sample_condition_variable)\.json",
                name,
            )
            _require(match is not None and (match.group(1), match.group(2)) not in seen,
                     "historical %s response path drifted" % version)
            seen.add((match.group(1), match.group(2)))
        output.update(raw_names)
        output.update((receipt_name, started_name))
    return output


def _validate_prior_archive(raw: bytes) -> Tuple[
    List[Tuple[str, str]],
    Dict[Tuple[str, str], List[Dict[str, Any]]],
    Dict[Tuple[str, str], Dict[str, Any]],
]:
    _require(_sha256(raw) == PRIOR_SHA256, "prior archive SHA-256 drifted")
    archive, infos = _zip_infos(raw, "prior", 978)
    try:
        _audit_all_json(archive, infos, "prior")
        _require(PRIOR_RECEIPT in infos and ROSTER_MEMBER in infos,
                 "prior archive envelope is incomplete")
        roster_raw = _zip_read(archive, infos[ROSTER_MEMBER], ROSTER_MEMBER)
        _require(_sha256(roster_raw) == ROSTER_SHA256, "prior roster SHA-256 drifted")
        roster = _parse_roster(_loads(roster_raw, ROSTER_MEMBER))

        receipt = _loads(_zip_read(archive, infos[PRIOR_RECEIPT], PRIOR_RECEIPT), PRIOR_RECEIPT)
        prior_keys = {
            "all_entities_condition_feasible", "api_base", "application_header", "artifact_kind",
            "authorization_consumed", "candidate_id", "condition_feasible_entity_count", "contract",
            "ended_at_utc", "entities", "entity_count", "outer_or_formal_metrics_opened",
            "recovery_parent_failure", "response_count", "response_manifest", "roster",
            "science_executed", "source_construction_executed", "source_producer_git_commit",
            "source_producer_relative_path", "source_producer_sha256", "source_scores_read",
            "started_at_utc", "status", "target_atom_identities_read", "target_values_read",
        }
        _require(isinstance(receipt, dict) and set(receipt) == prior_keys,
                 "prior final receipt schema drifted")
        _require(receipt["artifact_kind"] == "hold_only_target_unread_solution_condition_catalog_receipt"
                 and receipt["contract"] == "atypemu_solution_state_condition_catalog_v1"
                 and receipt["entity_count"] == len(roster) == 135
                 and receipt["response_count"] == 405,
                 "prior final receipt contract/count drifted")
        _require(receipt["roster"] == {
            "path": ".auto/staging/" + ROSTER_MEMBER,
            "sha256": ROSTER_SHA256,
        }, "prior final receipt roster binding drifted")

        expected_pairs = {(bmrb, loop) for bmrb, _ in roster for loop in LOOPS}
        manifest = receipt["response_manifest"]
        _require(isinstance(manifest, list) and len(manifest) == len(expected_pairs),
                 "prior response manifest count drifted")
        responses: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        bindings: Dict[Tuple[str, str], Dict[str, Any]] = {}
        final_members = {ROSTER_MEMBER, PRIOR_RECEIPT, PRIOR_ROOT + "/started_at_utc.txt"}
        for entry in manifest:
            _require(isinstance(entry, dict), "prior response-manifest entry is not an object")
            bmrb, loop = entry.get("bmrb_id"), entry.get("loop")
            _require((bmrb, loop) in expected_pairs and (bmrb, loop) not in bindings,
                     "prior response-manifest identity is duplicate or unallowlisted")
            member_relative = "raw_api_responses/%s.%s.json" % (bmrb, loop)
            binding = _manifest_binding(entry, bmrb, loop, member_relative)
            member = PRIOR_ROOT + "/" + member_relative
            _require(member in infos, "prior manifest response member is absent")
            body = _zip_read(archive, infos[member], member)
            _require(_sha256(body) == binding["sha256"], "prior manifest response hash drifted")
            responses[(bmrb, loop)] = _parse_prior_response(_loads(body, member), bmrb, loop)
            bindings[(bmrb, loop)] = binding
            final_members.add(member)
        _require(set(bindings) == expected_pairs, "prior response manifest is incomplete")

        historical = _validate_historical_inventory(archive, infos)
        _require_inventory(infos, final_members | historical, "prior")
        return roster, responses, bindings
    finally:
        archive.close()


def _validate_v6_archive(raw: bytes, roster: Sequence[Tuple[str, str]]) -> Tuple[
    Dict[str, Any], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]
]:
    _require(_sha256(raw) == V6_SHA256, "recovery-v6 archive SHA-256 drifted")
    archive, infos = _zip_infos(raw, "recovery-v6", 141)
    try:
        _audit_all_json(archive, infos, "recovery-v6")
        _require(V6_MANIFEST in infos and V6_RECEIPT in infos,
                 "recovery-v6 archive envelope is incomplete")
        manifest = _loads(_zip_read(archive, infos[V6_MANIFEST], V6_MANIFEST), V6_MANIFEST)
        manifest_keys = {
            "artifact_kind", "authorization_consumed", "candidate_id", "contract", "member_count",
            "members", "outer_or_formal_metrics_opened", "science_executed", "source_scores_read",
            "target_atom_identities_read", "target_values_read",
        }
        _require(isinstance(manifest, dict) and set(manifest) == manifest_keys,
                 "recovery-v6 archive manifest schema drifted")
        _require(manifest["artifact_kind"] == "target_unread_openmm86_deposited_ph_recovery_v6_archive_manifest"
                 and manifest["candidate_id"] == "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
                 and manifest["contract"] == "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_archive_v1"
                 and manifest["member_count"] == 140 and isinstance(manifest["members"], list)
                 and len(manifest["members"]) == 140, "recovery-v6 archive manifest values drifted")
        listed: Dict[str, Dict[str, Any]] = {}
        for item in manifest["members"]:
            _require(isinstance(item, dict) and set(item) == {"path", "sha256", "size"},
                     "recovery-v6 manifest member schema drifted")
            relative, digest, size = item["path"], item["sha256"], item["size"]
            _require(type(relative) is str and type(digest) is str and SHA_RE.fullmatch(digest) is not None
                     and type(size) is int and size >= 0 and relative not in listed,
                     "recovery-v6 manifest member values drifted")
            _safe_zip_name(relative)
            member = V6_ROOT + "/artifacts/" + relative
            _require(member in infos and infos[member].file_size == size,
                     "recovery-v6 manifest member is absent or wrong-sized")
            body = _zip_read(archive, infos[member], member)
            _require(_sha256(body) == digest, "recovery-v6 manifest member hash drifted")
            listed[relative] = item
        _require(list(listed) == sorted(listed), "recovery-v6 manifest paths are not sorted")
        _require_inventory(infos, {V6_MANIFEST} | {
            V6_ROOT + "/artifacts/" + relative for relative in listed
        }, "recovery-v6")

        receipt_body = _zip_read(archive, infos[V6_RECEIPT], V6_RECEIPT)
        _require(_sha256(receipt_body) == listed[V6_RECEIPT_RELATIVE]["sha256"],
                 "recovery-v6 output receipt manifest binding drifted")
        receipt = _loads(receipt_body, V6_RECEIPT)
        receipt_keys = {
            "all_entities_ph_feasible", "api_base", "application_header", "archive", "artifact_kind",
            "authorization_consumed", "candidate_id", "contract", "ended_at_utc", "entities",
            "entity_count", "future_consumer", "metadata_scope", "new_response_count",
            "new_response_manifest", "outer_or_formal_metrics_opened", "ph_feasible_entity_count",
            "plan", "recovery_evidence", "roster", "science_executed", "source_construction_executed",
            "source_producer_git_commit", "source_producer_relative_path", "source_producer_sha256",
            "source_scores_read", "started_at_utc", "status", "target_atom_identities_read",
            "target_values_read", "warnings",
        }
        _require(isinstance(receipt, dict) and set(receipt) == receipt_keys,
                 "recovery-v6 output receipt schema drifted")
        _require(receipt["artifact_kind"] == "hold_only_target_unread_openmm86_deposited_ph_catalog_recovery_receipt"
                 and receipt["candidate_id"] == "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
                 and receipt["contract"] == "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6"
                 and receipt["archive"] == {
                     "path": ".auto/staging/" + PRIOR_NAME, "sha256": PRIOR_SHA256,
                 }
                 and receipt["roster"] == {
                     "path": ".auto/staging/" + ROSTER_MEMBER, "sha256": ROSTER_SHA256,
                 }
                 and receipt["entity_count"] == len(roster) == 135
                 and receipt["new_response_count"] == len(roster),
                 "recovery-v6 output receipt static binding/count drifted")

        expected_bmrbs = {bmrb for bmrb, _ in roster}
        output_manifest = receipt["new_response_manifest"]
        _require(isinstance(output_manifest, list) and len(output_manifest) == len(expected_bmrbs),
                 "recovery-v6 output response-manifest count drifted")
        assigned: Dict[str, List[Dict[str, Any]]] = {}
        bindings: Dict[str, Dict[str, Any]] = {}
        for entry in output_manifest:
            _require(isinstance(entry, dict) and set(entry) == {
                "bmrb_id", "final_url", "loop", "path", "sha256", "url"
            }, "recovery-v6 output response-manifest schema drifted")
            bmrb = entry.get("bmrb_id")
            _require(bmrb in expected_bmrbs and bmrb not in bindings
                     and entry.get("loop") == "Assigned_chem_shift_list",
                     "recovery-v6 output response-manifest identity drifted")
            relative = "raw_api_responses/%s.Assigned_chem_shift_list.json" % bmrb
            _require(entry.get("path") == relative and entry.get("url") == _assigned_url(bmrb)
                     and entry.get("final_url") == _assigned_url(bmrb)
                     and type(entry.get("sha256")) is str
                     and SHA_RE.fullmatch(entry["sha256"]) is not None,
                     "recovery-v6 output response-manifest binding drifted")
            archive_relative = V6_API + "/" + relative
            _require(archive_relative in listed, "recovery-v6 assigned response is unmanifested")
            body = _zip_read(
                archive, infos[V6_ROOT + "/artifacts/" + archive_relative], archive_relative
            )
            _require(_sha256(body) == entry["sha256"] == listed[archive_relative]["sha256"],
                     "recovery-v6 assigned response hash drifted")
            assigned[bmrb] = _parse_assigned_response(_loads(body, archive_relative), bmrb)
            bindings[bmrb] = entry
        _require(set(bindings) == expected_bmrbs, "recovery-v6 assigned response set is incomplete")
        return receipt, assigned, bindings
    finally:
        archive.close()


def _reason(reasons: List[str], text: str) -> None:
    if text not in reasons:
        reasons.append(text)


def _derive_v6_record(
    bmrb: str,
    uid: str,
    prior: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    assigned_rows: Sequence[Dict[str, Any]],
    prior_bindings: Mapping[Tuple[str, str], Dict[str, Any]],
    new_binding: Dict[str, Any],
) -> Dict[str, Any]:
    """Independently replay the preferred Experiment-condition pH route."""
    chem = prior[(bmrb, "Chem_shift_experiment")]
    experiments = prior[(bmrb, "Experiment")]
    variables = prior[(bmrb, "Sample_condition_variable")]
    statuses: List[str] = []
    complete: List[Tuple[str, str]] = []
    for row in chem:
        experiment_id, assigned_id = row.get("Experiment_ID"), row.get("Assigned_chem_shift_list_ID")
        if _valid_id(experiment_id) is not None and _valid_id(assigned_id) is not None:
            statuses.append("complete")
            complete.append((experiment_id, assigned_id))
        elif _missing(experiment_id) and _missing(assigned_id):
            statuses.append("absent")
        else:
            statuses.append("malformed")

    reasons: List[str] = []
    route = "unavailable"
    selected_condition: Optional[str] = None
    assigned_ids: List[str] = []
    experiment_ids: List[str] = []
    condition_ids: List[str] = []
    if not chem or all(status == "absent" for status in statuses):
        valid_direct = [
            row for row in assigned_rows
            if _valid_id(row.get(ASSIGNED_TAGS[0])) == _bmrb_number(bmrb)
            and _valid_id(row.get(ASSIGNED_TAGS[1])) is not None
            and _valid_id(row.get(ASSIGNED_TAGS[2])) is not None
        ]
        if len(valid_direct) == 1:
            route = "direct_fallback"
            assigned_ids = [valid_direct[0][ASSIGNED_TAGS[1]]]
            condition_ids = [valid_direct[0][ASSIGNED_TAGS[2]]]
            selected_condition = condition_ids[0]
        else:
            _reason(reasons, "Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")
    elif all(status == "complete" for status in statuses):
        route = "exact_route"
        assigned_ids = sorted({pair[1] for pair in complete})
        experiment_ids = sorted({pair[0] for pair in complete})
        if len(assigned_ids) != 1:
            _reason(reasons, "complete exact links do not resolve to one assigned list")
        invalid_preferred_pointer = False
        condition_set = set()
        for experiment_id in experiment_ids:
            matches = [row for row in experiments if row.get("ID") == experiment_id]
            if len(matches) == 1 and _valid_id(matches[0].get("Sample_condition_list_ID")) is not None:
                condition_set.add(matches[0]["Sample_condition_list_ID"])
            else:
                invalid_preferred_pointer = True
        condition_ids = sorted(condition_set)
        if len(condition_ids) != 1:
            _reason(reasons, "complete exact links do not resolve to one condition pointer")
        if invalid_preferred_pointer:
            _reason(reasons, "linked Experiment condition pointer is missing or invalid")
        if len(assigned_ids) == 1 and len(condition_ids) == 1:
            matches = [row for row in assigned_rows if row.get(ASSIGNED_TAGS[1]) == assigned_ids[0]]
            if (len(matches) == 1
                    and _valid_id(matches[0].get(ASSIGNED_TAGS[0])) == _bmrb_number(bmrb)
                    and _valid_id(matches[0].get(ASSIGNED_TAGS[2])) == condition_ids[0]):
                selected_condition = condition_ids[0]
            else:
                _reason(reasons, "selected assigned-list condition pointer is missing, ambiguous, or inconsistent")
    else:
        _reason(reasons, "Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")
        _reason(reasons, "Chem_shift_experiment row has only one valid or malformed linkage ID")

    diagnostics: List[Dict[str, Any]] = []
    ph_values: List[float] = []
    ph_count = 0
    if selected_condition is None:
        _reason(reasons, "one condition-list pointer was not resolved")
        _reason(reasons, "one deposited pH was not resolved")
    else:
        invalid_ph = False
        for row in variables:
            if row.get("Sample_condition_list_ID") != selected_condition:
                continue
            kind = _normal_type(row.get("Type"))
            if kind in ("temperature", "ionicstrength"):
                diagnostics.append({
                    "type": kind, "units": row.get("Val_units"), "value": row.get("Val"),
                })
            if kind == "ph":
                ph_count += 1
                units = row.get("Val_units")
                if not (_missing(units) or _normal_type(units) in ("ph", "p.h.")):
                    invalid_ph = True
                    continue
                number = _numeric_ph(row.get("Val"))
                if number is None:
                    invalid_ph = True
                else:
                    ph_values.append(number)
        if ph_count == 0:
            _reason(reasons, "selected condition has no pH record")
        elif invalid_ph:
            _reason(reasons, "selected pH record is missing, invalid, or out of range")
        elif len(set(ph_values)) != 1:
            _reason(reasons, "selected condition has contradictory deposited pH values")
        if reasons:
            _reason(reasons, "one deposited pH was not resolved")
    deposited_ph = next(iter(set(ph_values))) if not reasons and ph_values else None
    source_bindings = [
        {
            "archive_member": PRIOR_ROOT + "/" + prior_bindings[(bmrb, loop)]["path"],
            "loop": loop,
            "sha256": prior_bindings[(bmrb, loop)]["sha256"],
        }
        for loop in LOOPS
    ]
    return {
        "assigned_chem_shift_list_ids": assigned_ids if route != "unavailable" else [],
        "bmrb_id": bmrb,
        "complete_chem_shift_experiment_link_count": len(complete),
        "deposited_ph": deposited_ph,
        "deposited_ph_record_count": ph_count,
        "entity_uid": uid,
        "experiment_ids": experiment_ids if route == "exact_route" else [],
        "hold_reasons": reasons,
        "new_response_binding": new_binding,
        "ph_feasible": not reasons,
        "prior_archive_response_bindings": source_bindings,
        "route": route,
        "sample_condition_list_ids": condition_ids if route != "unavailable" else [],
        "temperature_ionic_diagnostics": diagnostics if selected_condition is not None else [],
    }


def _validate_v6_semantics(
    receipt: Dict[str, Any],
    roster: Sequence[Tuple[str, str]],
    derived: Sequence[Dict[str, Any]],
    bindings: Mapping[str, Dict[str, Any]],
) -> List[Tuple[str, str]]:
    _require(receipt["entities"] == list(derived),
             "recovery-v6 output records do not equal independent derivation")
    feasible = sum(1 for item in derived if item["ph_feasible"] is True)
    held = [(item["bmrb_id"], item["entity_uid"]) for item in derived if item["ph_feasible"] is False]
    _require(len(roster) == len(derived) == 135 and len(bindings) == 135,
             "recovery-v6 independent roster/record arithmetic drifted")
    _require(feasible == 115 and len(held) == 20 and feasible + len(held) == len(roster),
             "recovery-v6 independently derived HOLD arithmetic drifted")
    _require(receipt["ph_feasible_entity_count"] == feasible
             and receipt["all_entities_ph_feasible"] is (len(held) == 0)
             and receipt["status"] == "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS",
             "recovery-v6 receipt aggregate/status drifted")
    _require(len(set(held)) == len(held) and set(held).issubset(set(roster)),
             "recovery-v6 HOLD identity set is malformed")
    return held


def _text_or_none(value: Any) -> bool:
    return value is None or type(value) is str


def _validate_local_fallback_schema(
    bmrb: str,
    local_prior: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    assigned_rows: Sequence[Dict[str, Any]],
) -> None:
    expected = {(bmrb, loop) for loop in LOOPS}
    _require(isinstance(local_prior, Mapping) and set(local_prior) == expected,
             "prior loop mapping has malformed or extra fields")
    for loop in LOOPS:
        rows = local_prior[(bmrb, loop)]
        _require(isinstance(rows, list), "%s rows are not a list" % loop)
        for row in rows:
            if loop == "Chem_shift_experiment":
                _require(isinstance(row, dict) and set(row) == set(CHEM_TAGS)
                         and all(_text_or_none(value) for value in row.values()),
                         "Chem_shift_experiment row has malformed or extra fields")
            elif loop == "Experiment":
                _require(isinstance(row, dict) and frozenset(row) in EXPERIMENT_KEY_SET
                         and all(_text_or_none(value) for value in row.values()),
                         "Experiment row has malformed or extra fields")
            else:
                _require(isinstance(row, dict) and set(row) == set(CONDITION_TAGS),
                         "Sample_condition_variable row has malformed or extra fields")
                for key, value in row.items():
                    if key == "Val":
                        _require(not isinstance(value, bool) and type(value) in (type(None), int, float, str),
                                 "Sample_condition_variable Val has invalid type")
                    else:
                        _require(_text_or_none(value),
                                 "Sample_condition_variable row has non-text fields")
    _require(isinstance(assigned_rows, Sequence), "Assigned-list rows are not a sequence")
    for row in assigned_rows:
        _require(isinstance(row, dict) and set(row) == set(ASSIGNED_TAGS)
                 and all(_text_or_none(value) for value in row.values()),
                 "Assigned-list row has malformed or extra fields")


def _held_fallback_record(bmrb: str, uid: str, reason: str) -> Dict[str, Any]:
    return {
        "assigned_chem_shift_list_id": None,
        "bmrb_id": bmrb,
        "direct_sample_condition_list_id": None,
        "entity_uid": uid,
        "fallback_metadata_resolved": False,
        "fallback_ph": None,
        "hold_reasons": [reason, "metadata resolution is not feasibility evidence"],
    }


def _fallback_record(
    bmrb: str,
    uid: str,
    local_prior: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    assigned_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply the predeclared direct-pointer fallback, conservatively and locally."""
    try:
        _validate_local_fallback_schema(bmrb, local_prior, assigned_rows)
        entry_number = _bmrb_number(bmrb)
        direct: List[Tuple[str, str]] = []
        for row in assigned_rows:
            entry_id = _valid_id(row[ASSIGNED_TAGS[0]])
            assigned_id = _valid_id(row[ASSIGNED_TAGS[1]])
            condition_id = _valid_id(row[ASSIGNED_TAGS[2]])
            if entry_id != entry_number or assigned_id is None or condition_id is None:
                _fail("direct Assigned-list row is missing, invalid, or foreign")
            direct.append((assigned_id, condition_id))
        if len(direct) != 1:
            _fail("fallback requires exactly one valid direct Assigned-list row")
        assigned_id, condition_id = direct[0]

        complete_experiment_ids: List[str] = []
        matching_assigned_count = 0
        for row in local_prior[(bmrb, "Chem_shift_experiment")]:
            experiment_id = row["Experiment_ID"]
            chem_assigned_id = row["Assigned_chem_shift_list_ID"]
            valid_experiment = _valid_id(experiment_id)
            valid_assigned = _valid_id(chem_assigned_id)
            if not (_missing(experiment_id) or valid_experiment is not None):
                _fail("Chem_shift_experiment Experiment_ID is malformed")
            if not (_missing(chem_assigned_id) or valid_assigned is not None):
                _fail("Chem_shift_experiment Assigned-list ID is malformed")
            if valid_assigned is not None:
                if valid_assigned != assigned_id:
                    _fail("a Chem_shift_experiment Assigned-list ID differs from direct ID")
                matching_assigned_count += 1
            if valid_experiment is not None:
                if valid_assigned != assigned_id:
                    _fail("valid Experiment_ID has missing or other Assigned-list ID")
                complete_experiment_ids.append(valid_experiment)
        if matching_assigned_count == 0:
            _fail("no Chem_shift_experiment Assigned-list ID matches direct ID")

        for experiment_id in sorted(set(complete_experiment_ids)):
            matches = [
                row for row in local_prior[(bmrb, "Experiment")]
                if row["ID"] == experiment_id
            ]
            if len(matches) > 1:
                _fail("complete Experiment_ID resolves to duplicate Experiment rows")
            if len(matches) == 1 and _valid_id(matches[0]["Sample_condition_list_ID"]) is not None:
                _fail("preferred Experiment condition linkage is present; fallback is forbidden")

        ph_rows = [
            row for row in local_prior[(bmrb, "Sample_condition_variable")]
            if row["Sample_condition_list_ID"] == condition_id and _normal_type(row["Type"]) == "ph"
        ]
        if len(ph_rows) != 1:
            _fail("direct condition does not have exactly one pH row")
        ph_row = ph_rows[0]
        if ph_row["Val_units"] not in (None, "", "pH"):
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
    except AuditError as error:
        return _held_fallback_record(bmrb, uid, str(error))


def _expected_intent() -> Dict[str, Any]:
    return {
        "artifact_kind": "hold_only_unique_assigned_ph_recovery_v3_execution_intent",
        "candidate_id": CANDIDATE_ID,
        "candidate_source_git_commit": "1fb25950768f75ed7790175e8b13bb18f6d90024",
        "candidate_source_raw_sha256": "0e62678d95bbdf32c5e472866b143fc57a3ea9bb18edf9f65c727dcdb4e46892",
        "closed_capabilities": CLOSED,
        "contract": CONTRACT,
        "plan_raw_sha256": "4361a594365709386d952a301bb4262a3dbf278c2bb43208a3ea3eecc344a5fc",
        "prior_archive_raw_sha256": PRIOR_SHA256,
        "recovery_v1_postoutput_failure_receipt_raw_sha256": "1c095c3d3bb61b6b6ea8859c723053d3293137799f70e5a4911110daa067a910",
        "recovery_v1_sealed_output_raw_sha256": "e3bc6ae75a79cff1d3580ea366d3631dfdad87f60dd1b228e069edad770f99dc",
        "recovery_v2_failure_receipt_raw_sha256": "433839ecd75345ac240dda67b5ba91ab4329a9ec08a5d540c455c5eca7c659b7",
        "recovery_v2_intent_raw_sha256": "98768c7174a2fb30ce9d4561fca012026884d0cedeecf6522ac696ce65f1379d",
        "recovery_v2_output_receipt_absent": True,
        "recovery_v2_plan_raw_sha256": "d54b369bf03942cad787efdbeb60583fdd13ec461bb7cfcc68f44184b568c821",
        "recovery_v2_source_git_commit": "2866c1c1ac8bfbe26b171075c0fcebf14aeb6e93",
        "recovery_v2_source_raw_sha256": "9b104d59b7dc9aaea1283a863c091ee81bf32bda045b0a9e04412f808a214d81",
        "run_146_raw_line_sha256": "57063cba97bc9e7f925d037239c6acd41835e69af9c2d7f75462a270b689b4a5",
        "run_148_raw_line_sha256": "8efb4c6490095f9a39459371f3fce0243ce4e8c3b6846d9f6d68045c08d03433",
        "state": "ACTIVE_ONCE",
        "v6_archive_raw_sha256": V6_SHA256,
    }


def _validate_intent(raw: bytes) -> Dict[str, Any]:
    _require(_sha256(raw) == INTENT_SHA256, "execution intent SHA-256 drifted")
    intent = _loads(raw, "recovery-v3 execution intent")
    _require(_exact(intent, _expected_intent()), "execution intent schema or binding drifted")
    return intent


def _expected_provenance(intent: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_source_git_commit": intent["candidate_source_git_commit"],
        "candidate_source_raw_sha256": intent["candidate_source_raw_sha256"],
        "execution_intent_path": (STAGING_RELATIVE / INTENT_NAME).as_posix(),
        "execution_intent_raw_sha256": INTENT_SHA256,
        "plan_raw_sha256": intent["plan_raw_sha256"],
        "recovery_v1_postoutput_failure_receipt_raw_sha256": intent[
            "recovery_v1_postoutput_failure_receipt_raw_sha256"
        ],
        "recovery_v1_sealed_output_raw_sha256": intent["recovery_v1_sealed_output_raw_sha256"],
        "recovery_v2_failure_receipt_raw_sha256": intent["recovery_v2_failure_receipt_raw_sha256"],
        "recovery_v2_intent_raw_sha256": intent["recovery_v2_intent_raw_sha256"],
        "recovery_v2_plan_raw_sha256": intent["recovery_v2_plan_raw_sha256"],
        "recovery_v2_source_git_commit": intent["recovery_v2_source_git_commit"],
        "recovery_v2_source_raw_sha256": intent["recovery_v2_source_raw_sha256"],
        "run_146_raw_line_sha256": intent["run_146_raw_line_sha256"],
        "run_148_raw_line_sha256": intent["run_148_raw_line_sha256"],
    }


def _expected_bound_inputs(intent: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "parent_plan_raw_sha256": PARENT_PLAN_SHA256,
        "prior_archive": {
            "path": (STAGING_RELATIVE / PRIOR_NAME).as_posix(), "raw_sha256": PRIOR_SHA256,
        },
        "recovery_v1_postoutput_failure_receipt_raw_sha256": intent[
            "recovery_v1_postoutput_failure_receipt_raw_sha256"
        ],
        "recovery_v1_sealed_output_raw_sha256": intent["recovery_v1_sealed_output_raw_sha256"],
        "recovery_v2_failure_receipt_raw_sha256": intent["recovery_v2_failure_receipt_raw_sha256"],
        "recovery_v2_intent_raw_sha256": intent["recovery_v2_intent_raw_sha256"],
        "recovery_v2_plan_raw_sha256": intent["recovery_v2_plan_raw_sha256"],
        "recovery_v2_source_git_commit": intent["recovery_v2_source_git_commit"],
        "recovery_v2_source_raw_sha256": intent["recovery_v2_source_raw_sha256"],
        "recovery_v6_archive": {
            "path": (STAGING_RELATIVE / V6_NAME).as_posix(), "raw_sha256": V6_SHA256,
        },
        "recovery_v6_evidence_receipt_raw_sha256": V6_EVIDENCE_RECEIPT_SHA256,
        "recovery_v6_semantic_checker_raw_sha256": V6_SEMANTIC_CHECKER_SHA256,
        "run_146_raw_line_sha256": intent["run_146_raw_line_sha256"],
        "run_148_raw_line_sha256": intent["run_148_raw_line_sha256"],
    }


def _receipt_template(
    intent: Mapping[str, Any], roster: Sequence[Tuple[str, str]], held: Sequence[Tuple[str, str]],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """The exact receipt object expected after raw independent derivation."""
    resolved = sum(1 for record in records if record["fallback_metadata_resolved"] is True)
    feasible = len(roster) - len(held)
    return {
        "artifact_kind": "hold_only_target_unread_unique_assigned_ph_recovery_v3_receipt",
        "bound_inputs": _expected_bound_inputs(intent),
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": CLOSED,
        "combined_metadata_resolved_entity_count": feasible + resolved,
        "contract": CONTRACT,
        "entities": list(records),
        "entity_count": len(roster),
        "execution_provenance": _expected_provenance(intent),
        "fallback_candidate_entity_count": len(records),
        "fallback_metadata_resolved_entity_count": resolved,
        "metadata_resolution_limit": "Metadata-only qualification cannot establish protonation feasibility, support feasibility, or science feasibility.",
        "output": {
            "path": SEALED_RECEIPT_RELATIVE.as_posix(), "receipt_creation": "O_EXCL",
        },
        "recovery_v6_hold_entity_count": len(held),
        "recovery_v6_metadata_resolved_entity_count": feasible,
        "status": "HOLD_METADATA_REINTERPRETATION_ONLY",
    }


def _validate_sealed_receipt(
    raw: bytes,
    intent: Mapping[str, Any],
    roster: Sequence[Tuple[str, str]],
    held: Sequence[Tuple[str, str]],
    records: Sequence[Dict[str, Any]],
    verify_hash: bool = True,
) -> None:
    if verify_hash:
        _require(_sha256(raw) == SEALED_RECEIPT_SHA256, "sealed recovery-v3 receipt SHA-256 drifted")
    receipt = _loads(raw, "sealed recovery-v3 receipt")
    expected = _receipt_template(intent, roster, held, records)
    _require(_exact(receipt, expected),
             "sealed recovery-v3 receipt schema, records, provenance, or arithmetic drifted")
    _require(len(roster) == 135 and len(held) == 20 and len(records) == len(held),
             "sealed receipt independent roster/HOLD arithmetic drifted")
    _require(sum(1 for item in records if item["fallback_metadata_resolved"] is True)
             == receipt["fallback_metadata_resolved_entity_count"],
             "sealed receipt fallback-resolution arithmetic drifted")


def _canonical_root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / SOURCE_RELATIVE
    try:
        source_mode = os.lstat(source).st_mode
        root_mode = os.lstat(root).st_mode
    except OSError as error:
        raise AuditError("cannot inspect canonical checker path: %s" % error) from error
    _require(not stat.S_ISLNK(source_mode) and stat.S_ISREG(source_mode)
             and not stat.S_ISLNK(root_mode) and stat.S_ISDIR(root_mode)
             and source == expected and source.resolve(strict=True) == expected.resolve(strict=True),
             "independent checker source path is indirect or noncanonical")
    return root


def verify(root: Path) -> Dict[str, int]:
    """Production replay.  All paths are fixed beneath the canonical repository root."""
    prior_raw = _read_bound(
        root, STAGING_RELATIVE / PRIOR_NAME, PRIOR_SHA256, "prior metadata archive", MAX_ARCHIVE_BYTES
    )
    v6_raw = _read_bound(
        root, STAGING_RELATIVE / V6_NAME, V6_SHA256, "recovery-v6 metadata archive", MAX_ARCHIVE_BYTES
    )
    intent_raw = _read_bound(
        root, STAGING_RELATIVE / INTENT_NAME, INTENT_SHA256, "recovery-v3 execution intent", MAX_INTENT_BYTES
    )
    sealed_raw = _read_bound(
        root, SEALED_RECEIPT_RELATIVE, SEALED_RECEIPT_SHA256, "sealed recovery-v3 receipt", MAX_RECEIPT_BYTES
    )
    intent = _validate_intent(intent_raw)
    roster, prior, prior_bindings = _validate_prior_archive(prior_raw)
    v6_receipt, assigned, new_bindings = _validate_v6_archive(v6_raw, roster)
    derived = [
        _derive_v6_record(bmrb, uid, prior, assigned[bmrb], prior_bindings, new_bindings[bmrb])
        for bmrb, uid in roster
    ]
    held = _validate_v6_semantics(v6_receipt, roster, derived, new_bindings)
    held_set = set(held)
    records = [
        _fallback_record(
            bmrb, uid, {(bmrb, loop): prior[(bmrb, loop)] for loop in LOOPS}, assigned[bmrb]
        )
        for bmrb, uid in roster if (bmrb, uid) in held_set
    ]
    _require([(item["bmrb_id"], item["entity_uid"]) for item in records] == held,
             "fallback records do not preserve independently derived HOLD roster order")
    _validate_sealed_receipt(sealed_raw, intent, roster, held, records)
    return {
        "checks": len(roster) + len(prior_bindings) + len(new_bindings) + len(derived) + len(records),
        "roster": len(roster),
        "hold": len(held),
        "resolved": sum(1 for record in records if record["fallback_metadata_resolved"] is True),
    }


def _expect_rejection(callback: Any, label: str) -> None:
    try:
        callback()
    except AuditError:
        return
    raise AssertionError("%s was accepted" % label)


def _chem_row(experiment_id: Any, assigned_id: Any) -> Dict[str, Any]:
    row = {key: None for key in CHEM_TAGS}
    row["Experiment_ID"] = experiment_id
    row["Assigned_chem_shift_list_ID"] = assigned_id
    return row


def _experiment_row(experiment_id: Any, condition_id: Any) -> Dict[str, Any]:
    row = {key: None for key in EXPERIMENT_TAG_LAYOUTS[0]}
    row["ID"] = experiment_id
    row["Sample_condition_list_ID"] = condition_id
    return row


def _condition_row(condition_id: Any, value: Any, units: Any = "pH") -> Dict[str, Any]:
    row = {key: None for key in CONDITION_TAGS}
    row.update({"Sample_condition_list_ID": condition_id, "Type": "pH", "Val": value, "Val_units": units})
    return row


def _assigned_row(entry: Any, assigned: Any, condition: Any) -> Dict[str, Any]:
    return dict(zip(ASSIGNED_TAGS, (entry, assigned, condition)))


def _local_fixture(
    chem: Sequence[Dict[str, Any]], experiments: Sequence[Dict[str, Any]],
    conditions: Sequence[Dict[str, Any]], assigned: Sequence[Dict[str, Any]],
) -> Tuple[str, str, Dict[Tuple[str, str], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    bmrb, uid = "bmr1", "bmrb:1:entity:1"
    return bmrb, uid, {
        (bmrb, "Chem_shift_experiment"): list(chem),
        (bmrb, "Experiment"): list(experiments),
        (bmrb, "Sample_condition_variable"): list(conditions),
    }, list(assigned)


def _synthetic_receipt_records() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Dict[str, Any]]]:
    roster = [("bmr%d" % index, "bmrb:%d:entity:1" % index) for index in range(1, 136)]
    held = roster[:20]
    records = [_held_fallback_record(bmrb, uid, "synthetic HOLD") for bmrb, uid in held]
    return roster, held, records


def self_test() -> int:
    """Pure synthetic/in-memory negative tests; this function never reads .auto."""
    checks = 0
    _expect_rejection(lambda: _loads(b'{"x": 1, "x": 2}', "duplicate-key test"), "duplicate JSON keys")
    checks += 1

    # Inventory validation is intentionally separate from global archive hashes:
    # an extra member remains a rejection even in this synthetic ZIP exercise.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("expected.json", b"{}")
        archive.writestr("extra.json", b"{}")
    archive, infos = _zip_infos(stream.getvalue(), "synthetic", 2)
    try:
        _expect_rejection(lambda: _require_inventory(infos, {"expected.json"}, "synthetic"),
                          "extra ZIP member")
    finally:
        archive.close()
    checks += 1

    baseline = _local_fixture(
        [_chem_row(None, "9")], [], [_condition_row("7", "7.0")],
        [_assigned_row("1", "9", "7")],
    )
    result = _fallback_record(*baseline)
    if not (result["fallback_metadata_resolved"] is True and result["fallback_ph"] == 7.0):
        raise AssertionError("synthetic direct fallback baseline did not resolve")
    checks += 1

    unit_ambiguous = _local_fixture(
        [_chem_row(None, "9")], [], [_condition_row("7", "7.0", "pD")],
        [_assigned_row("1", "9", "7")],
    )
    if _fallback_record(*unit_ambiguous)["fallback_metadata_resolved"]:
        raise AssertionError("ambiguous/non-pH direct units were accepted")
    checks += 1

    preferred_present = _local_fixture(
        [_chem_row("1", "9")], [_experiment_row("1", "8")], [_condition_row("7", 7)],
        [_assigned_row("1", "9", "7")],
    )
    if _fallback_record(*preferred_present)["fallback_metadata_resolved"]:
        raise AssertionError("present preferred Experiment condition pointer permitted fallback")
    checks += 1

    # The fallback input is explicitly entity-local.  The old full-cohort map
    # fails closed instead of silently reading an unrelated entity's loops.
    bmrb, uid, local_prior, assigned = baseline
    full_cohort = dict(local_prior)
    full_cohort.update({("bmr2", loop): [] for loop in LOOPS})
    if _fallback_record(bmrb, uid, full_cohort, assigned)["fallback_metadata_resolved"]:
        raise AssertionError("full-cohort loop map was accepted as entity-local")
    if not _fallback_record(bmrb, uid, local_prior, assigned)["fallback_metadata_resolved"]:
        raise AssertionError("entity-local loop map did not remain usable")
    checks += 2

    roster, held, records = _synthetic_receipt_records()
    intent = _expected_intent()
    payload = _receipt_template(intent, roster, held, records)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _validate_sealed_receipt(raw, intent, roster, held, records, verify_hash=False)
    result_tamper = copy.deepcopy(payload)
    result_tamper["entities"][0]["fallback_ph"] = 6.0
    _expect_rejection(
        lambda: _validate_sealed_receipt(
            json.dumps(result_tamper).encode("utf-8"), intent, roster, held, records, verify_hash=False
        ), "sealed result tamper",
    )
    target_tamper = copy.deepcopy(payload)
    target_tamper["target_value"] = 1
    _expect_rejection(
        lambda: _validate_sealed_receipt(
            json.dumps(target_tamper).encode("utf-8"), intent, roster, held, records, verify_hash=False
        ), "sealed target-field tamper",
    )
    checks += 3
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="replay only the canonical sealed metadata artifacts")
    parser.add_argument("--acknowledge-hold-only", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run synthetic in-memory checks (default)")
    args = parser.parse_args(argv)

    if args.self_test or not args.check:
        if args.check and args.self_test:
            parser.error("--self-test and --check are mutually exclusive")
        checks = self_test()
        print("METRIC openmm86_unique_assigned_ph_recovery_v3_independent_checks=%d" % checks)
        print("METRIC target_values_read=0")
        print("METRIC source_scores_read=0")
        print("METRIC science_executed=0")
        print("METRIC authorization_consumed=0")
        print("STATUS HOLD_SELF_TEST_ONLY")
        return 0
    if not args.acknowledge_hold_only:
        parser.error("--check requires --acknowledge-hold-only")
    try:
        result = verify(_canonical_root())
    except (AuditError, OSError, ValueError, zipfile.BadZipFile) as error:
        print("REFUSAL HOLD-only independent recovery-v3 audit: %s" % error)
        return 3
    print("METRIC openmm86_unique_assigned_ph_recovery_v3_independent_checks=%d" % result["checks"])
    print("METRIC recovery_v6_roster_entities=%d" % result["roster"])
    print("METRIC recovery_v6_hold_entities=%d" % result["hold"])
    print("METRIC fallback_metadata_resolved_entity_count=%d" % result["resolved"])
    print("METRIC target_values_read=0")
    print("METRIC source_scores_read=0")
    print("METRIC science_executed=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_METADATA_REINTERPRETATION_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
