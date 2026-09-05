#!/usr/bin/env python3
"""Independently raw-replay all-atom masks/topology, with O_EXCL receipts.
This checker intentionally does not import the recount producer.  It reads only
six identity columns from each source feature Parquet and reimplements the PDB,
HIS-tautomer, and target-mask rules before comparing every sealed entity.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any
STUDY_ID = "atypemu_nested_support_count_v1"
CHECKER_RELATIVE = "gpuopt/candidates/check_nested_support_all_atom_recount_raw_v3.py"
POLICY_RELATIVE = "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_policy.json"
CATALOG_RELATIVE = ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz"
SOURCE_RELATIVE = ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
SEALED_RELATIVE = ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1_yulab/recount_results.tar.gz"
OUTPUT_RELATIVE = ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_raw_v3"
PDB_ROOT_RELATIVE = "data/BioEmu"
# Per-PDB and per-Parquet hashes are bound through these committed inputs.
FIXED_SHA256 = {
    POLICY_RELATIVE: "f792ea47639f19d0376088b2c2e2fd5e9a161f33e4871205664bc9972e1b35ae",
    CATALOG_RELATIVE: "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2",
    SOURCE_RELATIVE: "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    SEALED_RELATIVE: "790a5cd77fdc10b46e5df85d80013938bb48a81732f5f9db00a7ef66cd483f16",
}
SEALED_SUMMARY_SHA256 = "07fb86fc5ec4602d8e9d93856c8ff26af40dd0f45fe8754e398198cac502350c"
SHARD_COUNT = 27
ENTITY_COUNT = 135
CATALOG_SUPPORT_COUNT = 134850
LEVELS = (32, 128, 768, 1536)
ELEMENTS = ("H", "C", "N", "O", "S")
IDENTITY_COLUMNS = ("entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "support_id")
SOURCE_SUPPORT_IDS = tuple("BioEmu_%d" % number for number in (1, 126, 251, 376, 501, 626, 751, 876))
SENTINELS = ("authorization_consumed", "outer_or_formal_metrics_opened", "science_executed", "source_scores_read", "target_values_read")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BMRB_RE = re.compile(r"bmr[1-9][0-9]*\Z")
PREFIX = "atypemu_nested_support_count_v1_all_atom_recount_v1"
def _fields(names: str) -> frozenset[str]:
    return frozenset(names.split())
SEALED_SHARD_FIELDS = _fields("artifact_kind authorization_consumed catalog_archive_sha256 contract entities entity_count outer_or_formal_metrics_opened policy_sha256 science_executed shard_count shard_index source_commitment_sha256 source_scores_read study_id target_values_read")
SEALED_ENTITY_FIELDS = _fields("bmrb_id catalog_support_count entity_uid observer_fold policy_compliant_support_count rejected_reason_counts rejected_support_count rejections split supports target_count target_identity_sha256")
SUPPORT_FIELDS = _fields("all_atom_topology_sha256 distance_available_count_by_element distance_mask_sha256 histidine_tautomer_change_count pdb_sha256 status support_index target_available_count target_mask_sha256 target_missing_by_atom target_missing_count")
REJECTION_FIELDS = _fields("pdb_sha256 reason status support_index")
SUMMARY_FIELDS = _fields("artifact_kind authorization_consumed catalog_archive_sha256 contract entities entity_count full_recount_digest level_count_feasibility outer_or_formal_metrics_opened policy_sha256 policy_compliant_support_count rejected_support_count science_executed source_commitment_sha256 source_scores_read study_id target_values_read catalog_support_count")
SHARD_RECEIPT_FIELDS = _fields("artifact_kind authorization_consumed bindings checker_relative_path checker_source_sha256 contract exact_raw_replay_equals_sealed_result outer_or_formal_metrics_opened raw_replay_entity_count raw_replay_entities_sha256 science_executed sealed_result_entity_count sealed_result_entities_sha256 shard_count shard_index source_scores_read study_id target_values_read")
AGGREGATE_RECEIPT_FIELDS = _fields("artifact_kind authorization_consumed bindings catalog_support_count checker_relative_path checker_source_sha256 contract exact_raw_replay_equals_sealed_result level_count_feasibility outer_or_formal_metrics_opened raw_replay_entity_count raw_replay_entities_sha256 science_executed sealed_result_entity_count sealed_result_entities_sha256 shard_count source_scores_read statement study_id target_values_read")
def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode())
def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result
def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)
def _json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode(), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise ValueError("JSON object required: %s" % label)
    return value
def _root() -> Path:
    path = Path(__file__).absolute()
    root = path.parents[2]
    expected = root / CHECKER_RELATIVE
    if path != expected or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("checker is not at its committed direct path")
    return root
def _bound(root: Path, relative: str) -> bytes:
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("indirect or missing bound path: %s" % relative)
    raw = path.read_bytes()
    if _sha256_bytes(raw) != FIXED_SHA256[relative]:
        raise ValueError("bound SHA-256 mismatch: %s" % relative)
    return raw
def _bound_path(root: Path, relative: str, expected: str, label: str) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("indirect or missing %s: %s" % (label, relative))
    if _sha256_file(path) != expected:
        raise ValueError("SHA-256 mismatch for %s: %s" % (label, relative))
    return path
def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError("invalid SHA-256: %s" % label)
    return value
def _require_false(value: dict[str, Any], label: str) -> None:
    for key in SENTINELS:
        if type(value.get(key)) is not bool or value[key] is not False:
            raise ValueError("scope sentinel is not false: %s.%s" % (label, key))
def _bindings() -> dict[str, dict[str, str]]:
    return {
        relative: {"path": relative, "sha256": digest}
        for relative, digest in sorted(FIXED_SHA256.items())
    }
def _safe_tar_members(archive: tarfile.TarFile, expected: set[str], label: str) -> None:
    names = {member.name for member in archive.getmembers()}
    if names != expected:
        raise ValueError("%s member roster mismatch" % label)
    for member in archive.getmembers():
        if member.name == ".":
            if not member.isdir():
                raise ValueError("%s root member is not a directory" % label)
        elif (
            not member.isfile()
            or member.issym()
            or member.islnk()
            or member.name.startswith("/")
            or ".." in Path(member.name).parts
        ):
            raise ValueError("unsafe %s member: %s" % (label, member.name))
def _tar_json(archive: tarfile.TarFile, name: str) -> tuple[dict[str, Any], bytes]:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError("unreadable archive member: %s" % name)
    with handle:
        raw = handle.read()
    return _json(raw, name), raw
def _validate_policy(policy: dict[str, Any]) -> None:
    if (
        policy.get("state") != "FROZEN_POLICY_RECOUNT_NOT_RUN"
        or policy.get("study_id") != STUDY_ID
        or policy.get("qualification", {}).get(
            "non_histidine_missing_target_invalidates_support"
        )
        is not True
        or policy.get("qualification", {}).get("deuterium_invalidates_support")
        is not True
        or policy.get("qualification", {}).get(
            "future_support_rosters_may_use_target_availability_for_selection"
        )
        is not False
        or tuple(policy.get("qualification", {}).get("levels", ())) != LEVELS
        or tuple(policy.get("qualification", {}).get("target_identity_input_columns", ()))
        != IDENTITY_COLUMNS
    ):
        raise ValueError("frozen all-atom policy mismatch")
def _catalog_entities(raw: bytes) -> list[dict[str, Any]]:
    expected = {".", "./catalog_summary.json", "./SHA256SUMS"} | {
        "./shard_%d.json" % index for index in range(SHARD_COUNT)
    }
    entities: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        _safe_tar_members(archive, expected, "catalog archive")
        for index in range(SHARD_COUNT):
            shard, _ = _tar_json(archive, "./shard_%d.json" % index)
            _require_false(shard, "catalog shard %d" % index)
            if shard.get("shard_index") != index or shard.get("shard_count") != SHARD_COUNT:
                raise ValueError("catalog shard identity mismatch: %d" % index)
            rows = shard.get("entities")
            if not isinstance(rows, list):
                raise ValueError("catalog entities are not a list: %d" % index)
            entities.extend(rows)
    entities.sort(key=lambda row: row["entity_uid"])
    if len(entities) != ENTITY_COUNT or len({row.get("entity_uid") for row in entities}) != ENTITY_COUNT:
        raise ValueError("catalog does not contain exactly 135 entities")
    total = 0
    for entity in entities:
        uid, bmrb = entity.get("entity_uid"), entity.get("bmrb_id")
        files = entity.get("files")
        if (
            not isinstance(uid, str)
            or not isinstance(bmrb, str)
            or BMRB_RE.fullmatch(bmrb) is None
            or not isinstance(files, list)
            or entity.get("split") != "train"
            or entity.get("observer_fold") not in {"A", "B"}
        ):
            raise ValueError("invalid catalog entity: %r" % uid)
        indexes: list[int] = []
        for record in files:
            if not isinstance(record, dict) or type(record.get("support_index")) is not int:
                raise ValueError("invalid catalog support: %s" % uid)
            _require_hash(record.get("pdb_sha256"), "catalog PDB %s" % uid)
            _require_hash(record.get("all_atom_topology_sha256"), "catalog topology %s" % uid)
            indexes.append(record["support_index"])
        if indexes != sorted(indexes) or len(set(indexes)) != len(indexes):
            raise ValueError("catalog support ordering mismatch: %s" % uid)
        total += len(files)
    if total != CATALOG_SUPPORT_COUNT:
        raise ValueError("catalog support total is not 134850")
    return entities
def _source_features(source: dict[str, Any], catalogs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if (
        source.get("contract") != "k32_dynamic_distance_cache_source_commitment_v1"
        or source.get("target_values_read") is not False
        or source.get("source_gate_authorized") is not False
        or source.get("formal_evaluation_authorized") is not False
    ):
        raise ValueError("source commitment scope mismatch")
    features = source.get("features")
    if not isinstance(features, list) or len(features) != ENTITY_COUNT:
        raise ValueError("source feature roster is incomplete")
    expected = {row["entity_uid"]: row for row in catalogs}
    result: dict[str, dict[str, Any]] = {}
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != {
            "bmrb_id", "entity_uid", "relative_path", "row_count", "sha256"
        }:
            raise ValueError("source feature schema mismatch")
        uid = feature["entity_uid"]
        catalog = expected.get(uid)
        if catalog is None or uid in result or feature["bmrb_id"] != catalog["bmrb_id"]:
            raise ValueError("source feature/catalog roster mismatch")
        expected_path = "data/all_atom_observer_v1/features/%s.parquet" % catalog["bmrb_id"]
        if (
            feature["relative_path"] != expected_path
            or type(feature["row_count"]) is not int
            or feature["row_count"] <= 0
        ):
            raise ValueError("source feature path/count mismatch: %s" % uid)
        _require_hash(feature["sha256"], "source feature %s" % uid)
        result[uid] = feature
    if set(result) != set(expected):
        raise ValueError("source feature set differs from catalog")
    return result
def _read_targets(root: Path, entity: dict[str, Any], feature: dict[str, Any]) -> list[dict[str, Any]]:
    """Read no columns other than the frozen identity-column allowlist."""
    path = _bound_path(
        root, feature["relative_path"], feature["sha256"], "source feature Parquet"
    )
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - execution environment check
        raise RuntimeError("raw replay requires the Parquet codec (pyarrow)") from error
    table = parquet.read_table(path, columns=list(IDENTITY_COLUMNS), use_threads=False)
    if tuple(table.column_names) != IDENTITY_COLUMNS:
        raise ValueError("Parquet identity-column projection changed")
    rows = table.to_pylist()
    if len(rows) != feature["row_count"]:
        raise ValueError("Parquet row count differs from source commitment")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if set(row) != set(IDENTITY_COLUMNS):
            raise ValueError("Parquet identity row schema mismatch")
        if row["entity_uid"] != entity["entity_uid"] or any(
            row[key] is None for key in IDENTITY_COLUMNS
        ):
            raise ValueError("null or foreign Parquet identity row")
        target_id = str(row["target_id"])
        grouped.setdefault(target_id, []).append(row)
    targets: list[dict[str, Any]] = []
    for target_id in sorted(grouped):
        group = grouped[target_id]
        support_ids = [str(row["support_id"]) for row in group]
        if len(group) != len(SOURCE_SUPPORT_IDS) or set(support_ids) != set(SOURCE_SUPPORT_IDS):
            raise ValueError("target/support roster mismatch: %s" % entity["entity_uid"])
        canonical = {
            "entity_uid": entity["entity_uid"],
            "target_id": target_id,
            "seq_id": int(group[0]["seq_id"]),
            "comp_id": str(group[0]["comp_id"]).strip().upper(),
            "atom_id": str(group[0]["atom_id"]).strip().upper(),
        }
        for row in group[1:]:
            observed = (
                int(row["seq_id"]),
                str(row["comp_id"]).strip().upper(),
                str(row["atom_id"]).strip().upper(),
            )
            if observed != (
                canonical["seq_id"],
                canonical["comp_id"],
                canonical["atom_id"],
            ):
                raise ValueError("target identity drift: %s" % entity["entity_uid"])
        targets.append(canonical)
    if not targets:
        raise ValueError("no targets in feature Parquet: %s" % entity["entity_uid"])
    return targets
# An atom identity deliberately includes every policy-preserved PDB field.
Atom = tuple[str, str, str, int, str, int, str, str]
Residue = tuple[int, str, int, str, str]
def _parse_pdb(raw: bytes) -> dict[str, Any]:
    atoms: set[Atom] = set()
    topology = hashlib.sha256()
    lookup: dict[tuple[int, str, str], list[Residue]] = {}
    element_residues: dict[str, set[Residue]] = {element: set() for element in ELEMENTS}
    segment = 0
    for line in raw.decode("ascii", errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        try:
            seq_id = int(line[22:26])
            coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        except ValueError as error:
            raise ValueError("invalid PDB atom fields") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("nonfinite PDB coordinate")
        element = line[76:78].strip().upper() or next(
            (character for character in name if character.isalpha()), ""
        )
        atom = (
            line[:6].strip(),
            name,
            resname,
            segment,
            chain,
            seq_id,
            line[26:27].strip(),
            element,
        )
        if atom in atoms:
            raise ValueError("duplicate PDB atom identity: %r" % (atom,))
        atoms.add(atom)
        topology.update(json.dumps(atom, separators=(",", ":")).encode("utf-8"))
        topology.update(b"\n")
        residue = (segment, chain, seq_id, atom[6], resname)
        lookup.setdefault((seq_id, resname, name), []).append(residue)
        if element in element_residues:
            element_residues[element].add(residue)
    if not atoms:
        raise ValueError("PDB contains no atoms")
    return {
        "all_atom_topology_sha256": topology.hexdigest(),
        "atoms": atoms,
        "lookup": lookup,
        "element_residues": element_residues,
    }
def _his_changes(reference: dict[str, Any], support: dict[str, Any]) -> int:
    reference_atoms: set[Atom] = reference["atoms"]
    atoms: set[Atom] = support["atoms"]
    if any(atom[-1] == "D" for atom in atoms):
        raise ValueError("deuterium is outside the frozen hydrogen policy")
    reference_heavy = {atom for atom in reference_atoms if atom[-1] not in {"H", "D"}}
    heavy = {atom for atom in atoms if atom[-1] not in {"H", "D"}}
    if heavy != reference_heavy:
        raise ValueError("heavy-atom topology differs from reference")
    def donor(atom: Atom) -> bool:
        return atom[2] == "HIS" and atom[1] in {"HD1", "HE2"} and atom[-1] == "H"
    if {atom for atom in reference_atoms if not donor(atom)} != {
        atom for atom in atoms if not donor(atom)
    }:
        raise ValueError("non-HIS-donor atom identity differs from reference")
    residues = {
        (atom[3], atom[4], atom[5], atom[6], atom[2])
        for atom in heavy
        if atom[2] == "HIS"
    }
    for atom in reference_atoms | atoms:
        if donor(atom) and (
            atom[0] != "ATOM" or (atom[3], atom[4], atom[5], atom[6], atom[2]) not in residues
        ):
            raise ValueError("HIS donor identity has no exact ATOM heavy residue")
    changed = 0
    for residue in residues:
        before = {
            atom[1]
            for atom in reference_atoms
            if donor(atom) and (atom[3], atom[4], atom[5], atom[6], atom[2]) == residue
        }
        after = {
            atom[1]
            for atom in atoms
            if donor(atom) and (atom[3], atom[4], atom[5], atom[6], atom[2]) == residue
        }
        if len(before) != 1 or len(after) != 1:
            raise ValueError("HIS must have exactly one emitted HD1/HE2 donor hydrogen")
        changed += before != after
    return changed
def _masks(support: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    target_digest = hashlib.sha256()
    distance_digest = hashlib.sha256()
    available_count = 0
    missing = Counter()
    distance_counts = Counter({element: 0 for element in ELEMENTS})
    for target in targets:
        source_atom = target["atom_id"]
        atom_name = "H" if source_atom == "HN" else source_atom
        matches = support["lookup"].get(
            (target["seq_id"], target["comp_id"], atom_name), []
        )
        if len(matches) > 1:
            raise ValueError("ambiguous target atom identity: %s" % target["target_id"])
        available = len(matches) == 1
        if not available and "%s:%s" % (target["comp_id"], atom_name) not in {
            "HIS:HD1",
            "HIS:HE2",
        }:
            raise ValueError("non-HIS target atom is unavailable: %s" % target["target_id"])
        if available:
            available_count += 1
        else:
            missing["%s:%s" % (target["comp_id"], atom_name)] += 1
        residue = matches[0] if available else None
        channels = tuple(
            bool(available and (support["element_residues"][element] - {residue}))
            for element in ELEMENTS
        )
        for element, enabled in zip(ELEMENTS, channels):
            distance_counts[element] += enabled
        identity = json.dumps(
            [
                target["entity_uid"],
                target["target_id"],
                target["seq_id"],
                target["comp_id"],
                source_atom,
            ],
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        target_digest.update(identity + b"\0" + bytes([available]))
        distance_digest.update(identity + b"\0" + bytes(channels))
    return {
        "target_available_count": available_count,
        "target_missing_count": len(targets) - available_count,
        "target_missing_by_atom": dict(sorted(missing.items())),
        "target_mask_sha256": target_digest.hexdigest(),
        "distance_available_count_by_element": dict(distance_counts),
        "distance_mask_sha256": distance_digest.hexdigest(),
    }
def _pdb_relative(bmrb_id: str, support_index: int) -> str:
    if BMRB_RE.fullmatch(bmrb_id) is None or type(support_index) is not int or support_index < 1:
        raise ValueError("invalid PDB path identity")
    return "%s/%s/%s_BioEmu_%d.pdb" % (
        PDB_ROOT_RELATIVE,
        bmrb_id,
        bmrb_id,
        support_index,
    )
def _replay_entity(root: Path, entity: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    directory = root / PDB_ROOT_RELATIVE / entity["bmrb_id"]
    if directory.is_symlink() or directory.resolve(strict=True) != directory:
        raise ValueError("indirect PDB directory: %s" % entity["bmrb_id"])
    records = entity["files"]
    reference_record = next(
        (row for row in records if row["support_index"] == 1), None
    )
    if reference_record is None:
        raise ValueError("catalog reference support 1 is absent: %s" % entity["entity_uid"])
    reference_path = _bound_path(
        root,
        _pdb_relative(entity["bmrb_id"], 1),
        reference_record["pdb_sha256"],
        "reference PDB",
    )
    reference = _parse_pdb(reference_path.read_bytes())
    supports: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reasons = Counter()
    for record in records:
        support_index = record["support_index"]
        relative = _pdb_relative(entity["bmrb_id"], support_index)
        path = _bound_path(root, relative, record["pdb_sha256"], "support PDB")
        raw = path.read_bytes()
        try:
            parsed = _parse_pdb(raw)
            if parsed["all_atom_topology_sha256"] != record["all_atom_topology_sha256"]:
                raise ValueError("raw PDB topology hash differs from catalog")
            support = {
                "support_index": support_index,
                "pdb_sha256": _sha256_bytes(raw),
                "all_atom_topology_sha256": parsed["all_atom_topology_sha256"],
                "histidine_tautomer_change_count": _his_changes(reference, parsed),
                "status": "POLICY_COMPLIANT_MASK_EMITTED",
            }
            support.update(_masks(parsed, targets))
            supports.append(support)
        except ValueError as error:
            reason = str(error)
            reasons[reason] += 1
            rejections.append(
                {
                    "support_index": support_index,
                    "pdb_sha256": record["pdb_sha256"],
                    "reason": reason,
                    "status": "POLICY_REJECTED",
                }
            )
    return {
        "entity_uid": entity["entity_uid"],
        "bmrb_id": entity["bmrb_id"],
        "observer_fold": entity["observer_fold"],
        "split": entity["split"],
        "target_count": len(targets),
        "target_identity_sha256": _canonical_sha256(targets),
        "catalog_support_count": len(records),
        "policy_compliant_support_count": len(supports),
        "rejected_support_count": len(rejections),
        "rejected_reason_counts": dict(sorted(reasons.items())),
        "rejections": rejections,
        "supports": supports,
    }
def _validate_sealed_entity(
    output: object, catalog: dict[str, Any], targets: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(output, dict) or set(output) != SEALED_ENTITY_FIELDS:
        raise ValueError("sealed entity schema mismatch: %s" % catalog["entity_uid"])
    uid = catalog["entity_uid"]
    for field in ("entity_uid", "bmrb_id", "observer_fold", "split"):
        if output[field] != catalog[field]:
            raise ValueError("sealed entity identity mismatch: %s.%s" % (uid, field))
    if (
        type(output["target_count"]) is not int
        or output["target_count"] != len(targets)
        or output["target_identity_sha256"] != _canonical_sha256(targets)
        or type(output["catalog_support_count"]) is not int
        or output["catalog_support_count"] != len(catalog["files"])
    ):
        raise ValueError("sealed target/catalog count mismatch: %s" % uid)
    supports, rejections = output["supports"], output["rejections"]
    if not isinstance(supports, list) or not isinstance(rejections, list):
        raise ValueError("sealed support lists mismatch: %s" % uid)
    if (
        type(output["policy_compliant_support_count"]) is not int
        or type(output["rejected_support_count"]) is not int
        or output["policy_compliant_support_count"] != len(supports)
        or output["rejected_support_count"] != len(rejections)
    ):
        raise ValueError("sealed support count mismatch: %s" % uid)
    records = {row["support_index"]: row for row in catalog["files"]}
    accepted: list[int] = []
    rejected: list[int] = []
    reasons = Counter()
    for row in supports:
        if not isinstance(row, dict) or set(row) != SUPPORT_FIELDS:
            raise ValueError("sealed support schema mismatch: %s" % uid)
        index = row["support_index"]
        if type(index) is not int or index not in records:
            raise ValueError("sealed support index mismatch: %s" % uid)
        record = records[index]
        if (
            row["status"] != "POLICY_COMPLIANT_MASK_EMITTED"
            or row["pdb_sha256"] != record["pdb_sha256"]
            or row["all_atom_topology_sha256"] != record["all_atom_topology_sha256"]
            or any(_require_hash(row[field], "sealed support") is None for field in (
                "target_mask_sha256", "distance_mask_sha256"
            ))
            or type(row["histidine_tautomer_change_count"]) is not int
            or row["histidine_tautomer_change_count"] < 0
        ):
            raise ValueError("sealed support provenance mismatch: %s" % uid)
        available, missing = row["target_available_count"], row["target_missing_count"]
        if (
            type(available) is not int
            or type(missing) is not int
            or available < 0
            or missing < 0
            or available + missing != len(targets)
        ):
            raise ValueError("sealed target arithmetic mismatch: %s" % uid)
        missing_by_atom = row["target_missing_by_atom"]
        distances = row["distance_available_count_by_element"]
        if (
            not isinstance(missing_by_atom, dict)
            or set(missing_by_atom) - {"HIS:HD1", "HIS:HE2"}
            or any(type(value) is not int or value <= 0 for value in missing_by_atom.values())
            or sum(missing_by_atom.values()) != missing
            or not isinstance(distances, dict)
            or set(distances) != set(ELEMENTS)
            or any(type(value) is not int or not 0 <= value <= available for value in distances.values())
        ):
            raise ValueError("sealed mask schema mismatch: %s" % uid)
        accepted.append(index)
    for row in rejections:
        if not isinstance(row, dict) or set(row) != REJECTION_FIELDS:
            raise ValueError("sealed rejection schema mismatch: %s" % uid)
        index = row["support_index"]
        if (
            type(index) is not int
            or index not in records
            or row["status"] != "POLICY_REJECTED"
            or row["pdb_sha256"] != records[index]["pdb_sha256"]
            or not isinstance(row["reason"], str)
            or not row["reason"]
        ):
            raise ValueError("sealed rejection mismatch: %s" % uid)
        rejected.append(index)
        reasons[row["reason"]] += 1
    if (
        accepted != sorted(accepted)
        or rejected != sorted(rejected)
        or len(set(accepted)) != len(accepted)
        or len(set(rejected)) != len(rejected)
        or set(accepted).intersection(rejected)
        or set(accepted + rejected) != set(records)
        or output["rejected_reason_counts"] != dict(sorted(reasons.items()))
    ):
        raise ValueError("sealed support partition mismatch: %s" % uid)
    return output
def _feasibility(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        shortfalls = [max(0, level - row["policy_compliant_support_count"]) for row in entities]
        result[str(level)] = {
            "all_entities_count_feasible": all(value == 0 for value in shortfalls),
            "entity_count_feasible": sum(value == 0 for value in shortfalls),
            "maximum_entity_shortfall": max(shortfalls),
            "total_shortfall": sum(shortfalls),
        }
    return result
def _sealed_results(raw: bytes, catalogs: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    expected = {"%s/summary.json" % PREFIX} | {
        "%s/shards/shard_%d.json" % (PREFIX, index) for index in range(SHARD_COUNT)
    }
    by_index: dict[int, list[dict[str, Any]]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members if member.isfile()}
        if names != expected or any(
            not (member.isfile() or member.isdir())
            or member.issym()
            or member.islnk()
            or member.name.startswith("/")
            or ".." in Path(member.name).parts
            for member in members
        ):
            raise ValueError("sealed result archive roster is unsafe or incomplete")
        summary, summary_raw = _tar_json(archive, "%s/summary.json" % PREFIX)
        if _sha256_bytes(summary_raw) != SEALED_SUMMARY_SHA256:
            raise ValueError("sealed summary SHA-256 mismatch")
        for index in range(SHARD_COUNT):
            shard, _ = _tar_json(archive, "%s/shards/shard_%d.json" % (PREFIX, index))
            if (
                set(shard) != SEALED_SHARD_FIELDS
                or shard.get("artifact_kind")
                != "hold_only_target_unread_all_atom_recount_shard_not_authorization"
                or shard.get("contract") != "atypemu_nested_support_count_v1_all_atom_recount_shard_v1"
                or shard.get("study_id") != STUDY_ID
                or shard.get("policy_sha256") != FIXED_SHA256[POLICY_RELATIVE]
                or shard.get("catalog_archive_sha256") != FIXED_SHA256[CATALOG_RELATIVE]
                or shard.get("source_commitment_sha256") != FIXED_SHA256[SOURCE_RELATIVE]
                or shard.get("shard_index") != index
                or shard.get("shard_count") != SHARD_COUNT
            ):
                raise ValueError("sealed shard header mismatch: %d" % index)
            _require_false(shard, "sealed shard %d" % index)
            rows = shard.get("entities")
            expected_uids = {
                row["entity_uid"]
                for position, row in enumerate(catalogs)
                if position % SHARD_COUNT == index
            }
            if (
                not isinstance(rows, list)
                or type(shard.get("entity_count")) is not int
                or shard["entity_count"] != len(rows)
                or {row.get("entity_uid") for row in rows if isinstance(row, dict)} != expected_uids
                or len(rows) != len(expected_uids)
            ):
                raise ValueError("sealed shard assignment mismatch: %d" % index)
            by_index[index] = rows
    _validate_sealed_summary_header(summary)
    return by_index, summary
def _validate_sealed_summary_header(summary: dict[str, Any]) -> None:
    if (
        set(summary) != SUMMARY_FIELDS
        or summary.get("artifact_kind") != "hold_only_target_unread_all_atom_recount_not_authorization"
        or summary.get("contract") != "atypemu_nested_support_count_v1_all_atom_recount_v1"
        or summary.get("study_id") != STUDY_ID
        or summary.get("policy_sha256") != FIXED_SHA256[POLICY_RELATIVE]
        or summary.get("catalog_archive_sha256") != FIXED_SHA256[CATALOG_RELATIVE]
        or summary.get("source_commitment_sha256") != FIXED_SHA256[SOURCE_RELATIVE]
    ):
        raise ValueError("sealed summary header mismatch")
    _require_false(summary, "sealed summary")
def _validate_sealed_summary(summary: dict[str, Any], entities: list[dict[str, Any]]) -> None:
    expected = {
        "entity_count": len(entities),
        "catalog_support_count": sum(row["catalog_support_count"] for row in entities),
        "policy_compliant_support_count": sum(row["policy_compliant_support_count"] for row in entities),
        "rejected_support_count": sum(row["rejected_support_count"] for row in entities),
    }
    if (
        not isinstance(summary["entities"], list)
        or summary["entities"] != entities
        or any(type(summary[key]) is not int or summary[key] != value for key, value in expected.items())
        or summary["full_recount_digest"] != _canonical_sha256(entities)
        or summary["level_count_feasibility"] != _feasibility(entities)
    ):
        raise ValueError("sealed summary content mismatch")
def _load(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    policy = _json(_bound(root, POLICY_RELATIVE), POLICY_RELATIVE)
    _validate_policy(policy)
    catalogs = _catalog_entities(_bound(root, CATALOG_RELATIVE))
    features = _source_features(_json(_bound(root, SOURCE_RELATIVE), SOURCE_RELATIVE), catalogs)
    sealed, summary = _sealed_results(_bound(root, SEALED_RELATIVE), catalogs)
    return catalogs, features, sealed, summary
def _replay_compare_shard(
    root: Path,
    index: int,
    catalogs: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    sealed: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for position, row in enumerate(catalogs) if position % SHARD_COUNT == index]
    sealed_rows = sealed[index]
    sealed_by_uid = {row["entity_uid"]: row for row in sealed_rows}
    raw_rows: list[dict[str, Any]] = []
    for catalog in selected:
        targets = _read_targets(root, catalog, features[catalog["entity_uid"]])
        sealed_row = _validate_sealed_entity(
            sealed_by_uid.get(catalog["entity_uid"]), catalog, targets
        )
        raw_row = _replay_entity(root, catalog, targets)
        if raw_row != sealed_row:
            raise ValueError("raw replay differs from sealed result: %s" % catalog["entity_uid"])
        raw_rows.append(raw_row)
    raw_rows.sort(key=lambda row: row["entity_uid"])
    return raw_rows, sorted(sealed_rows, key=lambda row: row["entity_uid"])
def _checker_sha256(root: Path) -> str:
    return _sha256_file(root / CHECKER_RELATIVE)
def _receipt_path(root: Path, index: int) -> Path:
    return root / OUTPUT_RELATIVE / "o_excl_shard_receipts" / ("shard_%d.json" % index)
def _aggregate_path(root: Path) -> Path:
    return root / OUTPUT_RELATIVE / "aggregate_receipt.json"
def _write_o_excl(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("O_EXCL receipt already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
def _shard_receipt(root: Path, index: int, raw_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if raw_rows != sealed_rows:
        raise ValueError("receipt requires exact raw/sealed entity comparison")
    return {
        "artifact_kind": "hold_only_raw_replay_shard_execution_evidence_not_authorization",
        "contract": "atypemu_nested_support_count_v1_raw_replay_shard_receipt_v3",
        "study_id": STUDY_ID,
        "bindings": _bindings(),
        "checker_relative_path": CHECKER_RELATIVE,
        "checker_source_sha256": _checker_sha256(root),
        "shard_index": index,
        "shard_count": SHARD_COUNT,
        "raw_replay_entity_count": len(raw_rows),
        "raw_replay_entities_sha256": _canonical_sha256(raw_rows),
        "sealed_result_entity_count": len(sealed_rows),
        "sealed_result_entities_sha256": _canonical_sha256(sealed_rows),
        "exact_raw_replay_equals_sealed_result": True,
        **{key: False for key in SENTINELS},
    }
def _validate_receipt(receipt: dict[str, Any], index: int, root: Path) -> None:
    if set(receipt) != SHARD_RECEIPT_FIELDS:
        raise ValueError("shard receipt schema mismatch: %d" % index)
    if (
        receipt.get("artifact_kind")
        != "hold_only_raw_replay_shard_execution_evidence_not_authorization"
        or receipt.get("contract")
        != "atypemu_nested_support_count_v1_raw_replay_shard_receipt_v3"
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("bindings") != _bindings()
        or receipt.get("checker_relative_path") != CHECKER_RELATIVE
        or receipt.get("checker_source_sha256") != _checker_sha256(root)
        or receipt.get("shard_index") != index
        or receipt.get("shard_count") != SHARD_COUNT
        or receipt.get("exact_raw_replay_equals_sealed_result") is not True
    ):
        raise ValueError("shard receipt binding mismatch: %d" % index)
    _require_false(receipt, "shard receipt %d" % index)
    for count, digest in (
        ("raw_replay_entity_count", "raw_replay_entities_sha256"),
        ("sealed_result_entity_count", "sealed_result_entities_sha256"),
    ):
        if type(receipt.get(count)) is not int or receipt[count] < 0:
            raise ValueError("invalid receipt count: %d.%s" % (index, count))
        _require_hash(receipt.get(digest), "receipt %d.%s" % (index, digest))
def run_shard(index: int) -> dict[str, Any]:
    if not 0 <= index < SHARD_COUNT:
        raise ValueError("shard index is outside 0..26")
    root = _root()
    catalogs, features, sealed, _summary = _load(root)
    raw_rows, sealed_rows = _replay_compare_shard(root, index, catalogs, features, sealed)
    receipt = _shard_receipt(root, index, raw_rows, sealed_rows)
    _write_o_excl(_receipt_path(root, index), receipt)
    return receipt
def aggregate() -> dict[str, Any]:
    root = _root()
    catalogs, features, sealed, summary = _load(root)
    raw_entities: list[dict[str, Any]] = []
    sealed_entities: list[dict[str, Any]] = []
    for index in range(SHARD_COUNT):
        path = _receipt_path(root, index)
        if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError("indirect or absent shard receipt: %d" % index)
        receipt = _json(path.read_bytes(), "shard receipt %d" % index)
        _validate_receipt(receipt, index, root)
        # Fresh replay gives aggregate an exact sealed entity schema/content check,
        # rather than treating a shard-receipt digest as a substitute for content.
        raw_rows, sealed_rows = _replay_compare_shard(root, index, catalogs, features, sealed)
        expected = (
            len(raw_rows),
            _canonical_sha256(raw_rows),
            len(sealed_rows),
            _canonical_sha256(sealed_rows),
        )
        observed = (
            receipt["raw_replay_entity_count"],
            receipt["raw_replay_entities_sha256"],
            receipt["sealed_result_entity_count"],
            receipt["sealed_result_entities_sha256"],
        )
        if observed != expected:
            raise ValueError("shard receipt digest/count mismatch: %d" % index)
        raw_entities.extend(raw_rows)
        sealed_entities.extend(sealed_rows)
    raw_entities.sort(key=lambda row: row["entity_uid"])
    sealed_entities.sort(key=lambda row: row["entity_uid"])
    _validate_sealed_summary(summary, sealed_entities)
    if (
        len(raw_entities) != ENTITY_COUNT
        or len(sealed_entities) != ENTITY_COUNT
        or len({row["entity_uid"] for row in raw_entities}) != ENTITY_COUNT
        or raw_entities != sealed_entities
        or sum(row["catalog_support_count"] for row in raw_entities) != CATALOG_SUPPORT_COUNT
    ):
        raise ValueError("aggregate entity coverage/content mismatch")
    receipt = {
        "artifact_kind": "hold_only_raw_replay_aggregate_execution_evidence_not_authorization",
        "contract": "atypemu_nested_support_count_v1_raw_replay_aggregate_receipt_v3",
        "study_id": STUDY_ID,
        "statement": "This receipt is execution evidence, not a cryptographic attestation.",
        "bindings": _bindings(),
        "checker_relative_path": CHECKER_RELATIVE,
        "checker_source_sha256": _checker_sha256(root),
        "shard_count": SHARD_COUNT,
        "raw_replay_entity_count": len(raw_entities),
        "raw_replay_entities_sha256": _canonical_sha256(raw_entities),
        "sealed_result_entity_count": len(sealed_entities),
        "sealed_result_entities_sha256": _canonical_sha256(sealed_entities),
        "exact_raw_replay_equals_sealed_result": True,
        "catalog_support_count": CATALOG_SUPPORT_COUNT,
        "level_count_feasibility": _feasibility(raw_entities),
        **{key: False for key in SENTINELS},
    }
    _write_o_excl(_aggregate_path(root), receipt)
    return receipt
def _pdb(atoms: list[tuple[str, str, str, int, str]]) -> bytes:
    lines = [f"{record:<6}{serial:5d} {name:>4s} {residue:>3s} A{sequence:4d}    {serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}" for serial, (name, residue, element, sequence, record) in enumerate(atoms, 1)]
    return ("\n".join(lines) + "\nEND\n").encode("ascii")
def self_test() -> int:
    reference = _pdb([("N", "ALA", "N", 1, "ATOM"), ("H", "ALA", "H", 1, "ATOM"), ("CA", "ALA", "C", 1, "ATOM"), ("N", "HIS", "N", 2, "ATOM"), ("CA", "HIS", "C", 2, "ATOM"), ("HE2", "HIS", "H", 2, "ATOM")])
    support = reference.replace(b" HE2", b" HD1")
    targets = [{"entity_uid": "bmrb:1:entity:1", "target_id": name, "seq_id": sequence, "comp_id": residue, "atom_id": atom} for name, sequence, residue, atom in (("a", 1, "ALA", "HN"), ("b", 2, "HIS", "HD1"), ("c", 2, "HIS", "HE2"))]
    before, after = _parse_pdb(reference), _parse_pdb(support)
    masks = _masks(after, targets)
    assert _his_changes(before, after) == 1 and masks["target_available_count"] == 2
    assert before["all_atom_topology_sha256"] != after["all_atom_topology_sha256"]
    assert masks["target_missing_by_atom"] == {"HIS:HE2": 1}
    assert masks["distance_available_count_by_element"] == {"H": 2, "C": 2, "N": 2, "O": 0, "S": 0}
    renamed = [dict(row) for row in targets]
    renamed[0]["target_id"] = "renamed"
    assert _masks(after, renamed)["target_mask_sha256"] != masks["target_mask_sha256"]
    for invalid, operation in (
        (_parse_pdb(reference.replace(b"   H ALA", b"  H1 ALA")), lambda value: _masks(value, targets)),
        (_parse_pdb(reference.replace(b" HE2", b" DD1")), lambda value: _his_changes(before, value)),
    ):
        try:
            operation(invalid)
        except ValueError:
            continue
        raise AssertionError("self-test accepted forbidden PDB change")
    assert SHARD_RECEIPT_FIELDS >= {"raw_replay_entity_count", "raw_replay_entities_sha256", "sealed_result_entity_count", "sealed_result_entities_sha256"}
    assert "statement" in AGGREGATE_RECEIPT_FIELDS
    assert _pdb_relative("bmr1", 1) == "data/BioEmu/bmr1/bmr1_BioEmu_1.pdb"
    try:
        _pdb_relative("bmr0", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("self-test accepted invalid generic PDB identity")
    return 11
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    shard = commands.add_parser("shard")
    shard.add_argument("--shard-index", required=True, type=int)
    commands.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "self-test":
        print("RAW_REPLAY_SELF_TESTS=%d" % self_test())
    elif args.command == "shard":
        receipt = run_shard(args.shard_index)
        print("RAW_REPLAY_SHARD_OK %d" % receipt["shard_index"])
    else:
        receipt = aggregate()
        print("RAW_REPLAY_AGGREGATE_OK %d" % receipt["raw_replay_entity_count"])
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
