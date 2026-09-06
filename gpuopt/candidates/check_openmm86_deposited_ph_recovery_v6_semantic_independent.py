#!/usr/bin/env python3
"""Independent, read-only semantic audit of the deposited-pH v6 evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

PRIOR_NAME = "atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip"
V6_NAME = "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_v1.zip"
PRIOR_SHA256 = "cca8b6612757005cbc62693ec6aaf433b4cb345919080a31f5492c2eb5349c70"
V6_SHA256 = "cc962fef0297aad433979372020343abfb593715690a39dda114c016f4ee0c36"
PRIOR_ROOT = "atypemu_nested_support_count_v1_solution_conditions_api_v2_v4_recovery"
PRIOR_RECEIPT = PRIOR_ROOT + "/receipt.json"
ROSTER_MEMBER = "atypemu_nested_support_count_v1_entity_roster_v3.json"
V6_ROOT = V6_NAME[:-4]
V6_API = ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_api_v6_recovery"
V6_RECEIPT_REL = V6_API + "/receipt.json"
V6_RECEIPT = V6_ROOT + "/artifacts/" + V6_RECEIPT_REL
V6_MANIFEST = V6_ROOT + "/archive_manifest.json"
LOOPS = ("Chem_shift_experiment", "Experiment", "Sample_condition_variable")
CLOSED = frozenset((
    "authorization_consumed", "outer_or_formal_metrics_opened", "science_executed",
    "source_construction_executed", "source_scores_read", "target_atom_identities_read",
    "target_values_read",
))
CHEM_TAGS = ("Experiment_ID", "Experiment_name", "Sample_ID", "Sample_label",
             "Sample_state", "Entry_ID", "Assigned_chem_shift_list_ID")
CONDITION_TAGS = ("Type", "Val", "Val_err", "Val_units", "Entry_ID",
                  "Sample_condition_list_ID")
EXPERIMENT_TAGS = frozenset((
    ("ID", "Name", "Raw_data_flag", "NMR_spec_expt_ID", "NMR_spec_expt_label",
     "MS_expt_ID", "MS_expt_label", "SAXS_expt_ID", "SAXS_expt_label",
     "FRET_expt_ID", "FRET_expt_label", "EMR_expt_ID", "EMR_expt_label",
     "Sample_ID", "Sample_label", "Sample_state", "Sample_volume",
     "Sample_volume_units", "Sample_condition_list_ID", "Sample_condition_list_label",
     "Sample_spinning_rate", "Sample_angle", "NMR_tube_type",
     "NMR_spectrometer_ID", "NMR_spectrometer_label", "NMR_spectrometer_probe_ID",
     "NMR_spectrometer_probe_label", "NMR_spectral_processing_ID",
     "NMR_spectral_processing_label", "Mass_spectrometer_ID",
     "Mass_spectrometer_label", "Xray_instrument_ID", "Xray_instrument_label",
     "Fluorescence_instrument_ID", "Fluorescence_instrument_label", "EMR_instrument_ID",
     "EMR_instrument_label", "Chromatographic_system_ID",
     "Chromatographic_system_label", "Chromatographic_column_ID",
     "Chromatographic_column_label", "Entry_ID", "Experiment_list_ID"),
    ("ID", "Name", "Raw_data_flag", "NUS_flag", "Interleaved_flag",
     "NMR_spec_expt_ID", "NMR_spec_expt_label", "MS_expt_ID", "MS_expt_label",
     "SAXS_expt_ID", "SAXS_expt_label", "FRET_expt_ID", "FRET_expt_label",
     "EMR_expt_ID", "EMR_expt_label", "Sample_ID", "Sample_label", "Sample_state",
     "Sample_volume", "Sample_volume_units", "Sample_condition_list_ID",
     "Sample_condition_list_label", "Sample_spinning_rate", "Sample_angle",
     "NMR_tube_type", "NMR_spectrometer_ID", "NMR_spectrometer_label",
     "NMR_spectrometer_probe_ID", "NMR_spectrometer_probe_label",
     "NMR_spectral_processing_ID", "NMR_spectral_processing_label",
     "Mass_spectrometer_ID", "Mass_spectrometer_label", "Xray_instrument_ID",
     "Xray_instrument_label", "Fluorescence_instrument_ID",
     "Fluorescence_instrument_label", "EMR_instrument_ID", "EMR_instrument_label",
     "Chromatographic_system_ID", "Chromatographic_system_label",
     "Chromatographic_column_ID", "Chromatographic_column_label", "Details",
     "Entry_ID", "Experiment_list_ID"),
))
ASSIGNED_TAGS = (
    "_Assigned_chem_shift_list.Entry_ID", "_Assigned_chem_shift_list.ID",
    "_Assigned_chem_shift_list.Sample_condition_list_ID",
)


class EvidenceError(ValueError):
    pass


def _fail(message):
    raise EvidenceError(message)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON key: %r" % (key,))
        result[key] = value
    return result


def _load_json(raw, where):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as exc:
        _fail("invalid JSON in %s: %s" % (where, exc))


def _require(condition, message):
    if not condition:
        _fail(message)


def _closed_false(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in CLOSED and item is not False:
                _fail("closed flag is not false at %s.%s" % (where, key))
            _closed_false(item, where + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _closed_false(item, "%s[%d]" % (where, index))


def _safe_member_name(name):
    _require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
             "unsafe ZIP member name")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), "absolute ZIP member")
    parts = name.split("/")
    _require(all(part not in ("", ".", "..") for part in parts), "traversal ZIP member")
    return name


def _open_fixed_zip(path, expected_sha):
    try:
        mode = os.lstat(str(path)).st_mode
    except OSError as exc:
        _fail("cannot stat archive %s: %s" % (path, exc))
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), "archive is not a regular non-symlink")
    _require(path.stat().st_size <= 8 * 1024 * 1024, "archive exceeds size bound")
    _require(_sha256_file(path) == expected_sha, "fixed archive SHA-256 mismatch: %s" % path.name)
    try:
        archive = zipfile.ZipFile(str(path), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        _fail("invalid ZIP %s: %s" % (path.name, exc))
    infos = archive.infolist()
    _require(0 < len(infos) <= 2000, "ZIP member count bound failed")
    names = []
    total = 0
    for info in infos:
        _safe_member_name(info.filename)
        _require(not info.is_dir() and not (info.flag_bits & 1), "directory or encrypted ZIP member")
        _require(info.file_size <= 2 * 1024 * 1024, "ZIP member exceeds size bound")
        _require(info.file_size <= max(1024, info.compress_size * 200), "ZIP member compression bound failed")
        total += info.file_size
        names.append(info.filename)
    _require(total <= 16 * 1024 * 1024 and len(set(names)) == len(names),
             "ZIP total-size or duplicate-member bound failed")
    try:
        bad = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        archive.close()
        _fail("ZIP CRC failure: %s" % exc)
    _require(bad is None, "ZIP CRC failure in %s" % bad)
    return archive, {info.filename: info for info in infos}


def _audit_json_members(archive, infos, label):
    for name in infos:
        if name.endswith(".json"):
            payload = _load_json(archive.read(name), "%s:%s" % (label, name))
            _closed_false(payload, "%s:%s" % (label, name))


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip() in ("", ".", "?"))


def _valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None


def _bmrb_number(bmrb):
    match = re.fullmatch(r"bmr([1-9][0-9]*)", bmrb if isinstance(bmrb, str) else "")
    if not match:
        _fail("noncanonical BMRB identity: %r" % (bmrb,))
    return match.group(1)


def _normal_type(value):
    return re.sub(r"\s+", "", value.casefold()) if isinstance(value, str) else ""


def _response_url(bmrb, loop):
    return "https://api.bmrb.io/v2/entry/%s?loop=%s" % (_bmrb_number(bmrb), loop)


def _assigned_url(bmrb):
    return ("https://api.bmrb.io/v2/entry/%s?tag=_Assigned_chem_shift_list.Entry_ID"
            "&tag=_Assigned_chem_shift_list.ID"
            "&tag=_Assigned_chem_shift_list.Sample_condition_list_ID") % _bmrb_number(bmrb)


def _parse_prior_response(payload, bmrb, loop):
    number = _bmrb_number(bmrb)
    _require(isinstance(payload, dict) and set(payload) == {number}, "unexpected %s response root" % loop)
    inside = payload[number]
    _require(isinstance(inside, dict) and set(inside) == {loop}, "unexpected %s response envelope" % loop)
    records = inside[loop]
    _require(isinstance(records, list), "%s records are not a list" % loop)
    expected_tags = CHEM_TAGS if loop == "Chem_shift_experiment" else CONDITION_TAGS
    rows = []
    for record in records:
        _require(isinstance(record, dict) and set(record) == {"category", "tags", "data"},
                 "unexpected %s record" % loop)
        _require(record["category"] == "_" + loop and isinstance(record["tags"], list)
                 and isinstance(record["data"], list), "unexpected %s record shape" % loop)
        tags = tuple(record["tags"])
        if loop == "Experiment":
            _require(tags in EXPERIMENT_TAGS, "unallowlisted Experiment tags")
        else:
            _require(tags == expected_tags, "unallowlisted %s tags" % loop)
        _require(not any("target" in _normal_type(tag) for tag in tags if isinstance(tag, str)),
                 "target-bearing response tag")
        for row in record["data"]:
            _require(isinstance(row, list) and len(row) == len(tags), "malformed %s row" % loop)
            rows.append(dict(zip(tags, row)))
    return rows


def _parse_assigned_response(payload, bmrb):
    number = _bmrb_number(bmrb)
    _require(isinstance(payload, dict) and set(payload) == {number}, "unexpected Assigned response root")
    inside = payload[number]
    _require(isinstance(inside, dict) and set(inside) == set(ASSIGNED_TAGS),
             "Assigned response is not the exact three-tag response")
    _require(not any("target" in _normal_type(tag) for tag in inside), "target-bearing response tag")
    columns = [inside[tag] for tag in ASSIGNED_TAGS]
    _require(all(isinstance(column, list) for column in columns), "Assigned response columns are not lists")
    _require(len({len(column) for column in columns}) == 1, "Assigned response columns have unequal lengths")
    return [dict(zip(ASSIGNED_TAGS, values)) for values in zip(*columns)]


def _canonical_roster(payload):
    _require(isinstance(payload, dict) and payload.get("entity_count") == 135
             and isinstance(payload.get("entities"), list) and len(payload["entities"]) == 135,
             "invalid 135-entity roster")
    records = []
    seen_bmrb, seen_uid = set(), set()
    for item in payload["entities"]:
        _require(isinstance(item, dict), "invalid roster entity")
        bmrb, uid = item.get("bmrb_id"), item.get("entity_uid")
        number = _bmrb_number(bmrb)
        match = re.fullmatch(r"bmrb:([1-9][0-9]*):entity:([1-9][0-9]*)", uid if isinstance(uid, str) else "")
        _require(match is not None and match.group(1) == number, "noncanonical roster entity UID")
        _require(bmrb not in seen_bmrb and uid not in seen_uid, "duplicate canonical roster identity")
        seen_bmrb.add(bmrb)
        seen_uid.add(uid)
        records.append((bmrb, uid))
    return records


def _manifest_entry(entry, bmrb, loop, path):
    expected = {"bmrb_id", "final_url", "loop", "path", "sha256", "url"}
    _require(isinstance(entry, dict) and set(entry) == expected, "invalid response-manifest entry")
    _require(entry["bmrb_id"] == bmrb and entry["loop"] == loop and entry["path"] == path,
             "response-manifest identity mismatch")
    expected_url = _response_url(bmrb, loop)
    _require(entry["url"] == expected_url and entry["final_url"] == expected_url,
             "response-manifest URL is not exact")
    _require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
             "invalid response-manifest SHA-256")


def _validate_prior(path):
    archive, infos = _open_fixed_zip(path, PRIOR_SHA256)
    try:
        _audit_json_members(archive, infos, "prior")
        _require(PRIOR_RECEIPT in infos and ROSTER_MEMBER in infos, "prior archive envelope is incomplete")
        receipt = _load_json(archive.read(PRIOR_RECEIPT), PRIOR_RECEIPT)
        _require(receipt.get("contract") == "atypemu_solution_state_condition_catalog_v1"
                 and receipt.get("entity_count") == 135 and receipt.get("response_count") == 405,
                 "prior receipt contract/count mismatch")
        roster_binding = receipt.get("roster")
        _require(isinstance(roster_binding, dict) and set(roster_binding) == {"path", "sha256"}
                 and roster_binding["path"] == ".auto/staging/" + ROSTER_MEMBER,
                 "prior receipt roster binding mismatch")
        roster_raw = archive.read(ROSTER_MEMBER)
        _require(_sha256_bytes(roster_raw) == roster_binding["sha256"], "prior roster hash mismatch")
        roster = _canonical_roster(_load_json(roster_raw, ROSTER_MEMBER))
        expected = {(bmrb, loop) for bmrb, _ in roster for loop in LOOPS}
        manifest = receipt.get("response_manifest")
        _require(isinstance(manifest, list) and len(manifest) == len(expected), "prior response count mismatch")
        bindings, raw = {}, {}
        for entry in manifest:
            _require(isinstance(entry, dict), "non-object prior response-manifest entry")
            bmrb, loop = entry.get("bmrb_id"), entry.get("loop")
            _require((bmrb, loop) in expected and (bmrb, loop) not in bindings,
                     "duplicate or unallowlisted prior response")
            member_rel = "raw_api_responses/%s.%s.json" % (bmrb, loop)
            _manifest_entry(entry, bmrb, loop, member_rel)
            member = PRIOR_ROOT + "/" + member_rel
            _require(member in infos, "missing allowlisted prior raw response")
            raw_bytes = archive.read(member)
            _require(_sha256_bytes(raw_bytes) == entry["sha256"], "prior raw member hash mismatch")
            raw[(bmrb, loop)] = _parse_prior_response(_load_json(raw_bytes, member), bmrb, loop)
            bindings[(bmrb, loop)] = entry
        _require(set(bindings) == expected, "prior response allowlist is not exact")
        return roster, raw, bindings
    finally:
        archive.close()


def _validate_v6(path, roster):
    archive, infos = _open_fixed_zip(path, V6_SHA256)
    try:
        _audit_json_members(archive, infos, "v6")
        _require(V6_MANIFEST in infos and V6_RECEIPT in infos, "v6 archive envelope is incomplete")
        manifest = _load_json(archive.read(V6_MANIFEST), V6_MANIFEST)
        _require(isinstance(manifest, dict)
                 and manifest.get("artifact_kind") == "target_unread_openmm86_deposited_ph_recovery_v6_archive_manifest"
                 and manifest.get("candidate_id") == "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6"
                 and manifest.get("contract") == "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v6_evidence_archive_v1",
                 "v6 manifest contract mismatch")
        members = manifest.get("members")
        _require(isinstance(members, list) and manifest.get("member_count") == len(members) == 140,
                 "v6 manifest count mismatch")
        listed = set()
        for item in members:
            _require(isinstance(item, dict) and set(item) == {"path", "sha256", "size"}
                     and isinstance(item["path"], str) and isinstance(item["size"], int)
                     and re.fullmatch(r"[0-9a-f]{64}", item["sha256"] or "") is not None,
                     "invalid v6 manifest member")
            _safe_member_name(item["path"])
            _require(item["path"] not in listed, "duplicate v6 manifest member")
            listed.add(item["path"])
            name = V6_ROOT + "/artifacts/" + item["path"]
            _require(name in infos and infos[name].file_size == item["size"], "v6 manifest size/path mismatch")
            _require(_sha256_bytes(archive.read(name)) == item["sha256"], "v6 manifest member hash mismatch")
        _require(set(infos) == {V6_MANIFEST} | {V6_ROOT + "/artifacts/" + item for item in listed},
                 "v6 archive has unmanifested members")
        receipt = _load_json(archive.read(V6_RECEIPT), V6_RECEIPT)
        _require(receipt.get("contract") == "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_recovery_v6"
                 and receipt.get("archive") == {"path": ".auto/staging/" + PRIOR_NAME, "sha256": PRIOR_SHA256},
                 "v6 receipt contract/archive binding mismatch")
        expected = {bmrb for bmrb, _ in roster}
        responses = receipt.get("new_response_manifest")
        _require(isinstance(responses, list) and len(responses) == len(expected), "v6 response count mismatch")
        bindings, raw = {}, {}
        for entry in responses:
            _require(isinstance(entry, dict) and set(entry) == {"bmrb_id", "final_url", "loop", "path", "sha256", "url"},
                     "invalid v6 response-manifest entry")
            bmrb = entry.get("bmrb_id")
            _require(bmrb in expected and bmrb not in bindings and entry.get("loop") == "Assigned_chem_shift_list",
                     "duplicate or unallowlisted v6 response")
            rel = "raw_api_responses/%s.Assigned_chem_shift_list.json" % bmrb
            _require(entry.get("path") == rel and entry.get("url") == _assigned_url(bmrb)
                     and entry.get("final_url") == _assigned_url(bmrb)
                     and isinstance(entry.get("sha256"), str)
                     and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
                     "v6 response binding mismatch")
            member_rel = V6_API + "/" + rel
            member = V6_ROOT + "/artifacts/" + member_rel
            _require(member_rel in listed and member in infos, "missing allowlisted v6 raw response")
            raw_bytes = archive.read(member)
            _require(_sha256_bytes(raw_bytes) == entry["sha256"], "v6 raw member hash mismatch")
            raw[bmrb] = _parse_assigned_response(_load_json(raw_bytes, member), bmrb)
            bindings[bmrb] = entry
        _require(set(bindings) == expected, "v6 response allowlist is not exact")
        return receipt, raw, bindings
    finally:
        archive.close()


def _reason(reasons, text):
    if text not in reasons:
        reasons.append(text)


def _ph_number(value):
    if isinstance(value, bool) or _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 14.0 else None


def _derive_entity(bmrb, uid, prior, assigned_rows, prior_bindings, new_binding):
    chem = prior[(bmrb, "Chem_shift_experiment")]
    experiments = prior[(bmrb, "Experiment")]
    variables = prior[(bmrb, "Sample_condition_variable")]
    statuses, complete = [], []
    for row in chem:
        experiment_id, assigned_id = row.get("Experiment_ID"), row.get("Assigned_chem_shift_list_ID")
        if _valid_id(experiment_id) and _valid_id(assigned_id):
            statuses.append("complete")
            complete.append((experiment_id, assigned_id))
        elif _missing(experiment_id) and _missing(assigned_id):
            statuses.append("absent")
        else:
            statuses.append("malformed")
    reasons = []
    route = "unavailable"
    selected_condition = None
    assigned_ids, experiment_ids, condition_ids = [], [], []
    if not chem or all(status == "absent" for status in statuses):
        valid_rows = [row for row in assigned_rows if _valid_id(row.get(ASSIGNED_TAGS[0]))
                      and row.get(ASSIGNED_TAGS[0]) == _bmrb_number(bmrb)
                      and _valid_id(row.get(ASSIGNED_TAGS[1])) and _valid_id(row.get(ASSIGNED_TAGS[2]))]
        if len(valid_rows) == 1:
            route = "direct_fallback"
            assigned_ids = [valid_rows[0][ASSIGNED_TAGS[1]]]
            condition_ids = [valid_rows[0][ASSIGNED_TAGS[2]]]
            selected_condition = condition_ids[0]
        else:
            _reason(reasons, "Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")
    elif all(status == "complete" for status in statuses):
        route = "exact_route"
        assigned_ids = sorted({pair[1] for pair in complete})
        experiment_ids = sorted({pair[0] for pair in complete})
        if len(assigned_ids) != 1:
            _reason(reasons, "complete exact links do not resolve to one assigned list")
        bad_experiment_pointer = False
        condition_set = set()
        for experiment_id in experiment_ids:
            matches = [row for row in experiments if row.get("ID") == experiment_id]
            if len(matches) == 1 and _valid_id(matches[0].get("Sample_condition_list_ID")):
                condition_set.add(matches[0]["Sample_condition_list_ID"])
            else:
                bad_experiment_pointer = True
        condition_ids = sorted(condition_set)
        if len(condition_ids) != 1:
            _reason(reasons, "complete exact links do not resolve to one condition pointer")
        if bad_experiment_pointer:
            _reason(reasons, "linked Experiment condition pointer is missing or invalid")
        if len(assigned_ids) == 1 and len(condition_ids) == 1:
            matched = [row for row in assigned_rows if row.get(ASSIGNED_TAGS[1]) == assigned_ids[0]]
            if (len(matched) == 1 and _valid_id(matched[0].get(ASSIGNED_TAGS[0]))
                    and matched[0][ASSIGNED_TAGS[0]] == _bmrb_number(bmrb)
                    and _valid_id(matched[0].get(ASSIGNED_TAGS[2]))
                    and matched[0][ASSIGNED_TAGS[2]] == condition_ids[0]):
                selected_condition = condition_ids[0]
            else:
                _reason(reasons, "selected assigned-list condition pointer is missing, ambiguous, or inconsistent")
    else:
        _reason(reasons, "Chem_shift_experiment linkage is missing or mixed; direct fallback is unavailable")
        _reason(reasons, "Chem_shift_experiment row has only one valid or malformed linkage ID")
    diagnostics = []
    ph_values = []
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
                diagnostics.append({"type": kind, "units": row.get("Val_units"), "value": row.get("Val")})
            if kind == "ph":
                ph_count += 1
                units = row.get("Val_units")
                if not (_missing(units) or _normal_type(units) in ("ph", "p.h.")):
                    invalid_ph = True
                    continue
                number = _ph_number(row.get("Val"))
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
    source_bindings = []
    for loop in LOOPS:
        binding = prior_bindings[(bmrb, loop)]
        source_bindings.append({"archive_member": PRIOR_ROOT + "/" + binding["path"],
                                "loop": loop, "sha256": binding["sha256"]})
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


def _check_receipt_semantics(receipt, derived, roster, bindings):
    required = {
        "all_entities_ph_feasible", "api_base", "application_header", "archive", "artifact_kind",
        "authorization_consumed", "candidate_id", "contract", "ended_at_utc", "entities", "entity_count",
        "future_consumer", "metadata_scope", "new_response_count", "new_response_manifest",
        "outer_or_formal_metrics_opened", "ph_feasible_entity_count", "plan", "recovery_evidence", "roster",
        "science_executed", "source_construction_executed", "source_producer_git_commit",
        "source_producer_relative_path", "source_producer_sha256", "source_scores_read", "started_at_utc",
        "status", "target_atom_identities_read", "target_values_read", "warnings",
    }
    _require(isinstance(receipt, dict) and set(receipt) == required, "unexpected v6 receipt schema")
    _require(receipt["entities"] == derived, "v6 receipt entity semantic mismatch")
    feasible = sum(1 for entity in derived if entity["ph_feasible"])
    held = len(derived) - feasible
    # Historical evidence assertions occur after (and never influence) raw derivation.
    _require(feasible == 115 and held == 20, "historical derived feasible/HOLD count mismatch")
    _require(receipt["entity_count"] == len(roster) == 135
             and receipt["new_response_count"] == len(bindings) == 135
             and receipt["ph_feasible_entity_count"] == feasible
             and receipt["all_entities_ph_feasible"] == (held == 0),
             "v6 receipt aggregate mismatch")
    _require(receipt["status"] == "HOLD_DEPOSITED_PH_METADATA_INCOMPLETE_OR_AMBIGUOUS",
             "v6 HOLD aggregate label mismatch")
    _require(receipt["roster"] == {"path": ".auto/staging/" + ROSTER_MEMBER,
                                    "sha256": "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"},
             "v6 roster binding mismatch")


def _production_paths(root):
    root = Path(root)
    try:
        root_mode = os.lstat(str(root)).st_mode
        auto = root / ".auto"
        staging = auto / "staging"
        auto_mode = os.lstat(str(auto)).st_mode
        staging_mode = os.lstat(str(staging)).st_mode
    except OSError as exc:
        _fail("missing production .auto/staging: %s" % exc)
    _require(stat.S_ISDIR(root_mode) and not stat.S_ISLNK(root_mode)
             and stat.S_ISDIR(auto_mode) and not stat.S_ISLNK(auto_mode)
             and stat.S_ISDIR(staging_mode) and not stat.S_ISLNK(staging_mode),
             "production root or staging is a symlink/non-directory")
    return staging / PRIOR_NAME, staging / V6_NAME


def verify(root):
    """Validate both fixed archives and return the semantic replay check count."""
    prior_path, v6_path = _production_paths(root)
    roster, prior, prior_bindings = _validate_prior(prior_path)
    receipt, assigned, new_bindings = _validate_v6(v6_path, roster)
    derived = [
        _derive_entity(
            bmrb, uid, prior, assigned[bmrb], prior_bindings, new_bindings[bmrb]
        )
        for bmrb, uid in roster
    ]
    _check_receipt_semantics(receipt, derived, roster, new_bindings)
    return len(roster) + len(prior_bindings) + len(new_bindings) + len(derived)


def self_test(root):
    """Small no-network negative tests for parser and semantic fail-closed behavior."""
    checks = 0
    try:
        _load_json(b'{"x": 1, "x": 2}', "duplicate-test")
        _fail("duplicate-key test did not fail")
    except EvidenceError:
        checks += 1
    try:
        _parse_assigned_response({"1": {"_Assigned_chem_shift_list.Entry_ID": ["1"],
                                         "_Assigned_chem_shift_list.ID": ["1"],
                                         "_Assigned_chem_shift_list.Target": ["x"]}}, "bmr1")
        _fail("target-tag test did not fail")
    except EvidenceError:
        checks += 1
    try:
        _safe_member_name("../escape.json")
        _fail("traversal test did not fail")
    except EvidenceError:
        checks += 1
    bmrb, uid = "bmr1", "bmrb:1:entity:1"
    prior = {
        (bmrb, "Chem_shift_experiment"): [{"Experiment_ID": "1", "Assigned_chem_shift_list_ID": "1"}],
        (bmrb, "Experiment"): [{"ID": "1", "Sample_condition_list_ID": "1"}],
        (bmrb, "Sample_condition_variable"): [{"Type": "pH", "Val": "7.0", "Val_units": "pH", "Sample_condition_list_ID": "1"}],
    }
    bindings = {(bmrb, loop): {"path": "raw_api_responses/%s.%s.json" % (bmrb, loop), "sha256": "0" * 64}
                for loop in LOOPS}
    assigned = [{ASSIGNED_TAGS[0]: "1", ASSIGNED_TAGS[1]: "1", ASSIGNED_TAGS[2]: "1"}]
    baseline = _derive_entity(bmrb, uid, prior, assigned, bindings, {"test": True})
    _require(baseline["ph_feasible"] and baseline["deposited_ph"] == 7.0, "semantic baseline failed")
    prior[(bmrb, "Sample_condition_variable")][0]["Val_units"] = "pD"
    changed = _derive_entity(bmrb, uid, prior, assigned, bindings, {"test": True})
    _require(not changed["ph_feasible"] and changed["deposited_ph"] is None,
             "raw-response semantic perturbation was accepted")
    return checks + 1


def _canonical_root():
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / "gpuopt/candidates/check_openmm86_deposited_ph_recovery_v6_semantic_independent.py"
    if source.is_symlink() or source.resolve(strict=True) != expected.resolve(strict=True):
        _fail("independent checker source path is indirect or misplaced")
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only independent deposited-pH v6 semantic checker")
    parser.add_argument("--acknowledge-hold-only", action="store_true", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    root = _canonical_root()
    checks = verify(root)
    if args.self_test:
        checks += self_test(root)
    print("METRIC openmm86_deposited_ph_recovery_v6_independent_semantic_checks=%d" % checks)
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
