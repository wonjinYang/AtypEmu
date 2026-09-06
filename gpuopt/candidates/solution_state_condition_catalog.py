"""Build a target-unread BMRB solution-condition feasibility catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

API_BASE = "https://api.bmrb.io/v2"
APPLICATION_HEADER = "AtypEmu solution-condition-catalog-v1"
ROSTER_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
)
ROSTER_SHA256 = "1a2d08e2cce23932996c8534ba710088dc05488cab350e628133926cec5c1cb9"
OUTPUT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_v1"
)
LOOPS = (
    "Chem_shift_experiment",
    "Experiment",
    "Sample_condition_variable",
)
REQUIRED_TAGS = {
    "Chem_shift_experiment": {
        "Experiment_ID",
        "Assigned_chem_shift_list_ID",
        "Entry_ID",
    },
    "Experiment": {"ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"},
    "Sample_condition_variable": {
        "Type",
        "Val",
        "Val_units",
        "Entry_ID",
        "Sample_condition_list_ID",
    },
}
ALLOWED_TAGS = {
    "Chem_shift_experiment": {
        "Experiment_ID",
        "Experiment_name",
        "Sample_ID",
        "Sample_label",
        "Sample_state",
        "Entry_ID",
        "Assigned_chem_shift_list_ID",
    },
    "Experiment": {
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
    },
    "Sample_condition_variable": {
        "Type",
        "Val",
        "Val_err",
        "Val_units",
        "Entry_ID",
        "Sample_condition_list_ID",
    },
}
MISSING = {None, "", ".", "?"}
ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
ROSTER_TOP_LEVEL_FIELDS = {
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
ROSTER_ENTITY_FIELDS = {
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _read_newline_safe(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError("bound input is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _require_safe_parents(path: Path, root: Path) -> None:
    current = path.parent
    while current != root:
        if root not in current.parents:
            raise ValueError("output path escapes repository root")
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlinked path parent: {current}")
        current = current.parent


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(payload: bytes) -> Any:
    return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rows(payload: bytes, entry_id: str, loop_name: str) -> list[dict[str, Any]]:
    if loop_name not in LOOPS:
        raise ValueError("loop category is not allowlisted")
    parsed = _loads(payload)
    if not isinstance(parsed, dict) or set(parsed) != {entry_id}:
        raise ValueError("API response entry envelope drifted")
    entry = parsed[entry_id]
    if entry == {}:
        return []
    if not isinstance(entry, dict) or set(entry) != {loop_name}:
        raise ValueError("API response loop envelope drifted")
    instances = entry[loop_name]
    if not isinstance(instances, list):
        raise ValueError("API loop instances are not a list")
    result: list[dict[str, Any]] = []
    for instance in instances:
        if not isinstance(instance, dict) or set(instance) != {
            "category",
            "tags",
            "data",
        }:
            raise ValueError("API loop instance schema drifted")
        if instance["category"] != f"_{loop_name}":
            raise ValueError("API loop category drifted")
        tags = instance["tags"]
        if (
            not isinstance(tags, list)
            or len(tags) != len(set(tags))
            or not REQUIRED_TAGS[loop_name].issubset(tags)
            or not set(tags).issubset(ALLOWED_TAGS[loop_name])
        ):
            raise ValueError("API loop tags are incomplete or target-bearing")
        for values in instance["data"]:
            if not isinstance(values, list) or len(values) != len(tags):
                raise ValueError("API loop row width drifted")
            if any(isinstance(value, (dict, list)) for value in values):
                raise ValueError("API loop row contains nested values")
            row = dict(zip(tags, values))
            if str(row["Entry_ID"]) != entry_id:
                raise ValueError("API loop row entry identity drifted")
            result.append(row)
    return result


def _number(value: Any) -> Optional[float]:
    if value in MISSING:
        return None
    try:
        result = float(str(value).strip())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _id(value: Any, name: str) -> str:
    if value in MISSING or not ID_PATTERN.fullmatch(str(value)):
        raise ValueError(f"{name} is missing or invalid")
    return str(value)


def _normalize_condition(kind: Any, value: Any, units: Any) -> tuple[str, float]:
    name = " ".join(str(kind).strip().lower().replace("_", " ").split())
    if name not in {"ph", "temperature", "ionic strength"}:
        raise KeyError(name)
    unit = "" if units in MISSING else str(units).strip().lower()
    number = _number(value)
    if number is None:
        raise ValueError("condition value is missing or nonfinite")
    if name == "ph":
        if unit not in {"", "ph"} or not 0.0 <= number <= 14.0:
            raise ValueError("pH value or unit is invalid")
        return "solution_ph", number
    if name == "temperature":
        if unit in {"k", "kelvin"}:
            kelvin = number
        elif unit in {"c", "celsius", "deg c", "°c"}:
            kelvin = number + 273.15
        else:
            raise ValueError("temperature unit is invalid")
        if not 0.0 < kelvin < 1000.0:
            raise ValueError("temperature is outside the physical range")
        return "solution_temperature_k", kelvin
    if name == "ionic strength":
        if unit in {"m", "mol/l", "mol/liter", "mol/litre"}:
            millimolar = number * 1000.0
        elif unit in {"mm", "mmol/l", "mmol/liter", "mmol/litre"}:
            millimolar = number
        else:
            raise ValueError("ionic-strength unit is invalid")
        if not 0.0 <= millimolar < 10000.0:
            raise ValueError("ionic strength is outside the physical range")
        return "solution_ionic_strength_mm", millimolar
    raise AssertionError("unreachable condition type")


def _condition_lists(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        list_id = _id(row["Sample_condition_list_ID"], "sample-condition-list ID")
        try:
            field, value = _normalize_condition(
                row["Type"], row["Val"], row["Val_units"]
            )
        except KeyError:
            continue
        values = result.setdefault(list_id, {})
        if field in values and values[field] != value:
            raise ValueError("condition list has contradictory duplicate values")
        values[field] = value
    return result


def _signature(values: dict[str, float]) -> Optional[tuple[float, float, float]]:
    fields = (
        "solution_ph",
        "solution_temperature_k",
        "solution_ionic_strength_mm",
    )
    if any(field not in values for field in fields):
        return None
    return tuple(values[field] for field in fields)  # type: ignore[return-value]


def summarize_entity(
    *,
    entity_uid: str,
    bmrb_id: str,
    responses: dict[str, bytes],
) -> dict[str, Any]:
    entry_id = bmrb_id.removeprefix("bmr")
    chem_rows = _rows(responses["Chem_shift_experiment"], entry_id, LOOPS[0])
    experiment_rows = _rows(responses["Experiment"], entry_id, LOOPS[1])
    condition_rows = _rows(
        responses["Sample_condition_variable"], entry_id, LOOPS[2]
    )
    experiments: dict[str, dict[str, Any]] = {}
    for row in experiment_rows:
        experiment_id = _id(row["ID"], "Experiment.ID")
        if experiment_id in experiments:
            raise ValueError("duplicate Experiment.ID")
        experiments[experiment_id] = row
    condition_lists = _condition_lists(condition_rows)
    by_shift_list: dict[str, set[str]] = {}
    unresolved_experiments: list[str] = []
    experiment_links: list[dict[str, str]] = []
    for row in chem_rows:
        experiment_id = _id(row["Experiment_ID"], "Chem_shift_experiment ID")
        shift_list_id = _id(
            row["Assigned_chem_shift_list_ID"], "assigned-shift-list ID"
        )
        if experiment_id not in experiments:
            unresolved_experiments.append(experiment_id)
            continue
        experiment = experiments[experiment_id]
        try:
            condition_id = _id(
                experiment["Sample_condition_list_ID"], "sample-condition-list ID"
            )
            sample_id = _id(experiment["Sample_ID"], "sample ID")
        except ValueError:
            unresolved_experiments.append(experiment_id)
            continue
        if condition_id not in condition_lists:
            unresolved_experiments.append(experiment_id)
            continue
        by_shift_list.setdefault(shift_list_id, set()).add(condition_id)
        experiment_links.append(
            {
                "assigned_chem_shift_list_id": shift_list_id,
                "experiment_id": experiment_id,
                "sample_condition_list_id": condition_id,
                "sample_id": sample_id,
            }
        )
    shift_lists: list[dict[str, Any]] = []
    signatures: set[tuple[float, float, float]] = set()
    for shift_list_id in sorted(by_shift_list):
        condition_ids = sorted(by_shift_list[shift_list_id])
        resolved = [_signature(condition_lists[condition_id]) for condition_id in condition_ids]
        local = {signature for signature in resolved if signature is not None}
        all_complete = len(resolved) == len(condition_ids) and all(
            signature is not None for signature in resolved
        )
        signatures.update(local)
        shift_lists.append(
            {
                "assigned_chem_shift_list_id": shift_list_id,
                "sample_condition_list_ids": condition_ids,
                "complete_unique_condition_signature": all_complete and len(local) == 1,
                "condition_signatures": [list(item) for item in sorted(local)],
            }
        )
    reasons: list[str] = []
    if not chem_rows:
        reasons.append("no Chem_shift_experiment rows")
    if unresolved_experiments:
        reasons.append("unresolved experiment-to-condition linkage")
    if not shift_lists:
        reasons.append("no linked assigned-shift-list condition")
    if any(not item["complete_unique_condition_signature"] for item in shift_lists):
        reasons.append("shift list lacks one complete condition signature")
    if len(signatures) != 1:
        reasons.append("entry does not have one cohort-usable condition signature")
    return {
        "entity_uid": entity_uid,
        "bmrb_id": bmrb_id,
        "assigned_shift_lists": shift_lists,
        "experiment_links": sorted(
            experiment_links,
            key=lambda item: (
                item["assigned_chem_shift_list_id"],
                item["experiment_id"],
            ),
        ),
        "condition_feasible": not reasons,
        "unique_condition_signatures": [list(item) for item in sorted(signatures)],
        "hold_reasons": sorted(set(reasons)),
    }


def _fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"Application": APPLICATION_HEADER, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise ValueError(f"BMRB API status is {response.status}")
        final_url = response.geturl()
        if final_url != url:
            raise ValueError("BMRB API request redirected")
        payload = response.read(10_000_001)
    if len(payload) > 10_000_000:
        raise ValueError("BMRB API response exceeds size limit")
    return payload, final_url


def _bound_root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    if source.resolve(strict=True) != source or source.is_symlink():
        raise ValueError("producer source path is indirect")
    return root


def _validate_roster(roster: Any, roster_raw: bytes) -> list[dict[str, Any]]:
    if _sha256(roster_raw) != ROSTER_SHA256:
        raise ValueError("bound roster raw hash drifted")
    if not isinstance(roster, dict) or set(roster) != ROSTER_TOP_LEVEL_FIELDS:
        raise ValueError("bound roster schema drifted")
    if (
        roster["artifact_kind"]
        != "target_unread_structural_entity_roster_not_authorization"
        or roster["contract"] != "atypemu_nested_support_count_v1_entity_roster_v3"
        or roster["study_id"] != "atypemu_nested_support_count_v1"
        or roster["entity_count"] != 135
        or len(roster["entities"]) != 135
        or any(
            roster[name] is not False
            for name in (
                "authorization_consumed",
                "outer_or_formal_metrics_opened",
                "source_scores_read",
                "target_values_read",
            )
        )
    ):
        raise ValueError("bound roster identity or closed state drifted")
    entities = roster["entities"]
    if not isinstance(entities, list):
        raise ValueError("bound roster entities are not a list")
    entity_uids: set[str] = set()
    bmrb_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != ROSTER_ENTITY_FIELDS:
            raise ValueError("bound roster entity schema drifted")
        bmrb_id = str(entity["bmrb_id"])
        entry_id = bmrb_id.removeprefix("bmr")
        entity_uid = str(entity["entity_uid"])
        if (
            not bmrb_id.startswith("bmr")
            or not ID_PATTERN.fullmatch(entry_id)
            or entity_uid != f"bmrb:{entry_id}:entity:1"
            or entity["split"] != "train"
            or entity["observer_fold"] not in {"A", "B"}
            or entity_uid in entity_uids
            or bmrb_id in bmrb_ids
        ):
            raise ValueError("bound roster entity identity drifted")
        entity_uids.add(entity_uid)
        bmrb_ids.add(bmrb_id)
    return entities


def fetch_catalog() -> dict[str, Any]:
    root = _bound_root()
    roster_path = root / ROSTER_RELATIVE
    output = root / OUTPUT_RELATIVE
    _require_safe_parents(roster_path, root)
    if not roster_path.is_file() or roster_path.is_symlink():
        raise ValueError("bound roster path is unavailable or indirect")
    roster_raw = _read_newline_safe(roster_path)
    roster = _loads(roster_raw)
    roster_entities = _validate_roster(roster, roster_raw)
    _require_safe_parents(output, root)
    output.mkdir(parents=True, exist_ok=False)
    raw_root = output / "raw_api_responses"
    raw_root.mkdir()
    started = _utc_now()
    _write_new(output / "started_at_utc.txt", f"{started}\n".encode())
    entities: list[dict[str, Any]] = []
    response_manifest: list[dict[str, Any]] = []
    try:
        for entity in sorted(roster_entities, key=lambda item: item["entity_uid"]):
            bmrb_id = str(entity["bmrb_id"])
            entry_id = bmrb_id.removeprefix("bmr")
            responses: dict[str, bytes] = {}
            response_bindings: list[dict[str, str]] = []
            for loop_name in LOOPS:
                url = f"{API_BASE}/entry/{entry_id}?loop={loop_name}"
                payload, final_url = _fetch(url)
                _rows(payload, entry_id, loop_name)
                relative = Path("raw_api_responses") / f"{bmrb_id}.{loop_name}.json"
                _write_new(output / relative, payload)
                responses[loop_name] = payload
                binding = {
                    "bmrb_id": bmrb_id,
                    "loop": loop_name,
                    "path": relative.as_posix(),
                    "sha256": _sha256(payload),
                    "url": url,
                    "final_url": final_url,
                }
                response_manifest.append(binding)
                response_bindings.append(binding)
                time.sleep(0.05)
            summary = summarize_entity(
                entity_uid=str(entity["entity_uid"]),
                bmrb_id=bmrb_id,
                responses=responses,
            )
            summary["source_response_bindings"] = response_bindings
            entities.append(
                summary
            )
    except Exception as error:
        failure = {
            "artifact_kind": "target_unread_condition_catalog_failure_receipt",
            "error": f"{type(error).__name__}: {error}",
            "response_count_written": len(response_manifest),
            "started_at_utc": started,
            "target_values_read": False,
            "source_scores_read": False,
            "science_executed": False,
            "authorization_consumed": False,
        }
        _write_new(
            output / "failure_receipt.json",
            (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
        )
        raise
    feasible = sum(item["condition_feasible"] for item in entities)
    receipt = {
        "artifact_kind": "hold_only_target_unread_solution_condition_catalog_receipt",
        "contract": "atypemu_solution_state_condition_catalog_v1",
        "candidate_id": (
            "atypemu_nested_support_count_v1_solution_state_protonation_support_v1"
        ),
        "api_base": API_BASE,
        "application_header": APPLICATION_HEADER,
        "roster": {
            "path": ROSTER_RELATIVE.as_posix(),
            "sha256": _sha256(roster_raw),
        },
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "response_count": len(response_manifest),
        "response_manifest": response_manifest,
        "entity_count": len(entities),
        "condition_feasible_entity_count": feasible,
        "all_entities_condition_feasible": feasible == len(entities) == 135,
        "entities": entities,
        "status": (
            "PASS_CONDITION_MANIFEST_INPUTS_COMPLETE"
            if feasible == 135
            else "HOLD_CONDITION_MANIFEST_INPUTS_INCOMPLETE_OR_AMBIGUOUS"
        ),
        "target_values_read": False,
        "target_atom_identities_read": False,
        "source_scores_read": False,
        "outer_or_formal_metrics_opened": False,
        "science_executed": False,
        "authorization_consumed": False,
        "source_construction_executed": False,
    }
    _write_new(
        output / "receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    return receipt


def _fixture(
    *, ionic_units: str = "M", ambiguous: bool = False, linked: bool = True
) -> dict[str, bytes]:
    entry_id = "42"

    def wrap(name: str, tags: list[str], data: list[list[Any]]) -> bytes:
        value = {
            entry_id: {
                name: [{"category": f"_{name}", "tags": tags, "data": data}]
            }
        }
        return json.dumps(value).encode()

    chem = [
        ["1", entry_id, "7"],
        ["2", entry_id, "7"],
    ]
    experiment = [
        ["1", "10", "1", entry_id],
        ["2", "10", "2" if ambiguous else ("1" if linked else "."), entry_id],
    ]
    conditions = [
        ["pH", "6.5", "pH", entry_id, "1"],
        ["temperature", "300", "K", entry_id, "1"],
        ["ionic strength", "0.15", ionic_units, entry_id, "1"],
    ]
    if ambiguous:
        conditions.extend(
            [
                ["pH", "5.0", "pH", entry_id, "2"],
                ["temperature", "300", "K", entry_id, "2"],
                ["ionic strength", "0.15", "M", entry_id, "2"],
            ]
        )
    return {
        "Chem_shift_experiment": wrap(
            "Chem_shift_experiment",
            ["Experiment_ID", "Entry_ID", "Assigned_chem_shift_list_ID"],
            chem,
        ),
        "Experiment": wrap(
            "Experiment",
            ["ID", "Sample_ID", "Sample_condition_list_ID", "Entry_ID"],
            experiment,
        ),
        "Sample_condition_variable": wrap(
            "Sample_condition_variable",
            ["Type", "Val", "Val_units", "Entry_ID", "Sample_condition_list_ID"],
            conditions,
        ),
    }


def self_test() -> int:
    complete = summarize_entity(
        entity_uid="bmrb:42:entity:1", bmrb_id="bmr42", responses=_fixture()
    )
    assert complete["condition_feasible"] is True
    assert complete["unique_condition_signatures"] == [[6.5, 300.0, 150.0]]
    assert summarize_entity(
        entity_uid="bmrb:42:entity:1",
        bmrb_id="bmr42",
        responses=_fixture(linked=False),
    )["condition_feasible"] is False
    partial = _fixture(ambiguous=True)
    parsed_partial = _loads(partial["Sample_condition_variable"])
    parsed_partial["42"]["Sample_condition_variable"][0]["data"].pop()
    partial["Sample_condition_variable"] = json.dumps(parsed_partial).encode()
    assert summarize_entity(
        entity_uid="bmrb:42:entity:1", bmrb_id="bmr42", responses=partial
    )["condition_feasible"] is False
    unsupported = _fixture()
    parsed_unsupported = _loads(unsupported["Sample_condition_variable"])
    parsed_unsupported["42"]["Sample_condition_variable"][0]["data"].append(
        ["buffer", "Tris", ".", "42", "1"]
    )
    unsupported["Sample_condition_variable"] = json.dumps(parsed_unsupported).encode()
    assert summarize_entity(
        entity_uid="bmrb:42:entity:1", bmrb_id="bmr42", responses=unsupported
    )["condition_feasible"] is True
    assert summarize_entity(
        entity_uid="bmrb:42:entity:1",
        bmrb_id="bmr42",
        responses=_fixture(ambiguous=True),
    )["condition_feasible"] is False
    bad_units = _fixture(ionic_units="kg")
    try:
        summarize_entity(
            entity_uid="bmrb:42:entity:1", bmrb_id="bmr42", responses=bad_units
        )
    except ValueError:
        pass
    else:
        raise AssertionError("bad condition unit was accepted")
    target_loop = _fixture()
    parsed = _loads(target_loop["Experiment"])
    parsed["42"]["Experiment"][0]["tags"].append("Atom_ID")
    for row in parsed["42"]["Experiment"][0]["data"]:
        row.append("CA")
    try:
        _rows(json.dumps(parsed).encode(), "42", "Experiment")
    except ValueError:
        pass
    else:
        raise AssertionError("target-bearing API tags were accepted")
    try:
        _loads(b'{"42": {}, "42": {}}')
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON key was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "receipt.json"
        _write_new(destination, b"first")
        try:
            _write_new(destination, b"second")
        except FileExistsError:
            pass
        else:
            raise AssertionError("O_EXCL output reuse was accepted")
        outside = Path(temporary) / "outside"
        dangling = Path(temporary) / "root" / "linked"
        dangling.parent.mkdir()
        dangling.symlink_to(outside, target_is_directory=True)
        try:
            _require_safe_parents(dangling / "receipt.json", dangling.parent)
        except ValueError:
            pass
        else:
            raise AssertionError("dangling symlink parent was accepted")
    return 13


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--acknowledge-target-unread-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        checks = self_test()
        print(f"METRIC solution_condition_catalog_checks={checks}")
        print("STATUS PASS")
        return 0
    if not args.acknowledge_target_unread_metadata_only:
        print("REFUSAL target-unread metadata-only acknowledgement is required")
        return 3
    result = fetch_catalog()
    print(f"STATUS {result['status']}")
    print(f"CONDITION_FEASIBLE_ENTITIES {result['condition_feasible_entity_count']}/135")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
