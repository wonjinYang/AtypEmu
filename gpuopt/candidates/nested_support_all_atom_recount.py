#!/usr/bin/env python3
"""Target-unread all-atom mask recount for the exact-nested support study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

POLICY_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_policy.json"
)
POLICY_SHA256 = "f792ea47639f19d0376088b2c2e2fd5e9a161f33e4871205664bc9972e1b35ae"
ARCHIVE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/"
    "catalog_v3_shards.tar.gz"
)
ARCHIVE_SHA256 = "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2"
SOURCE_RELATIVE = Path(
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
)
SOURCE_SHA256 = "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8"
PDB_ROOT_RELATIVE = Path("data/BioEmu")
OUTPUT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1"
)
SHARD_COUNT = 27
ENTITY_COUNT = 135
LEVELS = (32, 128, 768, 1536)
ELEMENTS = ("H", "C", "N", "O", "S")
SOURCE_SUPPORT_IDS = tuple(
    f"BioEmu_{index}" for index in (1, 32, 63, 94, 126, 157, 188, 221)
)
FALSE_SENTINELS = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_scores_read": False,
    "target_values_read": False,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHARD_FIELDS = frozenset(
    {
        "artifact_kind",
        "authorization_consumed",
        "catalog_archive_sha256",
        "contract",
        "entities",
        "entity_count",
        "outer_or_formal_metrics_opened",
        "policy_sha256",
        "science_executed",
        "shard_count",
        "shard_index",
        "source_commitment_sha256",
        "source_scores_read",
        "study_id",
        "target_values_read",
    }
)
ENTITY_FIELDS = frozenset(
    {
        "bmrb_id",
        "catalog_support_count",
        "entity_uid",
        "observer_fold",
        "policy_compliant_support_count",
        "rejected_reason_counts",
        "rejected_support_count",
        "rejections",
        "split",
        "supports",
        "target_count",
        "target_identity_sha256",
    }
)
SUPPORT_FIELDS = frozenset(
    {
        "all_atom_topology_sha256",
        "distance_available_count_by_element",
        "distance_mask_sha256",
        "histidine_tautomer_change_count",
        "pdb_sha256",
        "status",
        "support_index",
        "target_available_count",
        "target_mask_sha256",
        "target_missing_by_atom",
        "target_missing_count",
    }
)
REJECTION_FIELDS = frozenset({"pdb_sha256", "reason", "status", "support_index"})

Atom = tuple[str, str, str, int, str, int, str, str]
Residue = tuple[int, str, int, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _root() -> Path:
    script = Path(__file__).absolute()
    root = script.parents[2]
    if script != root / "gpuopt/candidates/nested_support_all_atom_recount.py":
        raise ValueError("recount producer is not at its committed relative path")
    if script.is_symlink() or script.resolve(strict=True) != script:
        raise ValueError("recount producer path is indirect")
    return root


def _bound_file(root: Path, relative: Path, expected_sha256: str) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"bound input path is indirect or absent: {relative}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"bound input hash mismatch: {relative}")
    return path


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_pdb(raw: bytes) -> dict[str, Any]:
    identities: set[Atom] = set()
    lookup: dict[tuple[int, str, str], list[Residue]] = {}
    element_residues: dict[str, set[Residue]] = {element: set() for element in ELEMENTS}
    segment = 0
    for line in raw.decode("ascii", errors="strict").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("TER"):
            segment += 1
            continue
        if not line.startswith(("ATOM  ", "HETATM")) or line[16:17] not in {
            " ",
            "A",
        }:
            continue
        name = line[12:16].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        seq_id = int(line[22:26])
        insertion = line[26:27].strip()
        element = line[76:78].strip().upper() or next(
            (character for character in name if character.isalpha()), ""
        )
        coordinates = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("nonfinite PDB coordinate")
        atom = (
            line[:6].strip(),
            name,
            resname,
            segment,
            chain,
            seq_id,
            insertion,
            element,
        )
        if atom in identities:
            raise ValueError(f"duplicate PDB atom identity: {atom}")
        identities.add(atom)
        residue = (segment, chain, seq_id, insertion, resname)
        lookup.setdefault((seq_id, resname, name), []).append(residue)
        if element in element_residues:
            element_residues[element].add(residue)
    if not identities:
        raise ValueError("PDB contains no atoms")
    return {
        "identities": identities,
        "lookup": lookup,
        "element_residues": element_residues,
    }


def _hydrogen_policy(reference: dict[str, Any], support: dict[str, Any]) -> int:
    ref_atoms: set[Atom] = reference["identities"]
    atoms: set[Atom] = support["identities"]
    if any(atom[-1] == "D" for atom in atoms):
        raise ValueError("deuterium is outside the frozen hydrogen policy")
    ref_heavy = {atom for atom in ref_atoms if atom[-1] not in {"H", "D"}}
    heavy = {atom for atom in atoms if atom[-1] not in {"H", "D"}}
    if heavy != ref_heavy:
        raise ValueError("heavy-atom topology differs from reference")

    def donor(atom: Atom) -> bool:
        return atom[2] == "HIS" and atom[1] in {"HD1", "HE2"} and atom[-1] == "H"

    if {atom for atom in ref_atoms if not donor(atom)} != {
        atom for atom in atoms if not donor(atom)
    }:
        raise ValueError("non-HIS-donor atom identity differs from reference")
    his_residues = {
        (atom[3], atom[4], atom[5], atom[6], atom[2])
        for atom in heavy
        if atom[2] == "HIS"
    }
    for atom in ref_atoms | atoms:
        if donor(atom):
            residue = (atom[3], atom[4], atom[5], atom[6], atom[2])
            if atom[0] != "ATOM" or residue not in his_residues:
                raise ValueError("HIS donor identity has no exact ATOM heavy residue")
    changed = 0
    for residue in his_residues:
        ref_names = {
            atom[1]
            for atom in ref_atoms
            if donor(atom) and (atom[3], atom[4], atom[5], atom[6], atom[2]) == residue
        }
        names = {
            atom[1]
            for atom in atoms
            if donor(atom) and (atom[3], atom[4], atom[5], atom[6], atom[2]) == residue
        }
        if len(ref_names) != 1 or len(names) != 1:
            raise ValueError("HIS must have exactly one emitted HD1/HE2 donor hydrogen")
        changed += ref_names != names
    return changed


def _target_masks(
    support: dict[str, Any], targets: list[dict[str, Any]]
) -> dict[str, Any]:
    target_digest = hashlib.sha256()
    distance_digest = hashlib.sha256()
    available_count = 0
    distance_counts = Counter({element: 0 for element in ELEMENTS})
    missing = Counter()
    lookup = support["lookup"]
    element_residues = support["element_residues"]
    for target in targets:
        target_id = str(target["target_id"])
        entity_uid = str(target["entity_uid"])
        seq_id = int(target["seq_id"])
        comp_id = str(target["comp_id"]).strip().upper()
        source_atom_id = str(target["atom_id"]).strip().upper()
        atom_id = "H" if source_atom_id == "HN" else source_atom_id
        matches = lookup.get((seq_id, comp_id, atom_id), [])
        if len(matches) > 1:
            raise ValueError(f"ambiguous target atom identity: {target_id}")
        available = len(matches) == 1
        if not available and f"{comp_id}:{atom_id}" not in {"HIS:HD1", "HIS:HE2"}:
            raise ValueError(f"non-HIS target atom is unavailable: {target_id}")
        available_count += available
        if not available:
            missing[f"{comp_id}:{atom_id}"] += 1
        residue = matches[0] if available else None
        channels = tuple(
            bool(available and (element_residues[element] - {residue}))
            for element in ELEMENTS
        )
        for element, channel in zip(ELEMENTS, channels):
            distance_counts[element] += channel
        identity = json.dumps(
            [entity_uid, target_id, seq_id, comp_id, source_atom_id],
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
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


def evaluate_support(
    raw: bytes,
    reference: dict[str, Any],
    catalog_record: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != catalog_record["pdb_sha256"]:
        raise ValueError("PDB bytes differ from the bound catalog")
    support = parse_pdb(raw)
    changes = _hydrogen_policy(reference, support)
    return {
        "support_index": int(catalog_record["support_index"]),
        "pdb_sha256": raw_sha256,
        "all_atom_topology_sha256": catalog_record["all_atom_topology_sha256"],
        "histidine_tautomer_change_count": changes,
        "status": "POLICY_COMPLIANT_MASK_EMITTED",
        **_target_masks(support, targets),
    }


def _catalog_entities(archive: Path) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    with tarfile.open(archive, "r:gz") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.isfile()
            and member.name.removeprefix("./").startswith("shard_")
            and member.name.endswith(".json")
        ]
        if len(members) != SHARD_COUNT:
            raise ValueError("catalog archive does not contain 27 shard files")
        for member in members:
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read catalog member: {member.name}")
            shard = json.load(stream)
            if any(shard.get(key) is not False for key in FALSE_SENTINELS):
                raise ValueError("catalog shard scope sentinel changed")
            entities.extend(shard["entities"])
    entities.sort(key=lambda row: row["entity_uid"])
    if (
        len(entities) != ENTITY_COUNT
        or len({row["entity_uid"] for row in entities}) != ENTITY_COUNT
    ):
        raise ValueError("catalog entity roster is not exactly 135 unique entities")
    return entities


def _target_rows(
    root: Path, source: dict[str, Any], entity: dict[str, Any]
) -> list[dict[str, Any]]:
    import pandas as pd

    feature = next(
        row for row in source["features"] if row["bmrb_id"] == entity["bmrb_id"]
    )
    path = root / feature["relative_path"]
    if (
        path.is_symlink()
        or path.resolve(strict=True) != path
        or _sha256(path) != feature["sha256"]
    ):
        raise ValueError(f"feature provenance mismatch: {entity['entity_uid']}")
    columns = ["entity_uid", "target_id", "seq_id", "comp_id", "atom_id", "support_id"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["entity_uid"].astype(str).eq(entity["entity_uid"])]
    if set(frame["support_id"].astype(str)) != set(SOURCE_SUPPORT_IDS):
        raise ValueError(f"feature support roster mismatch: {entity['entity_uid']}")
    grouped = frame.groupby("target_id", sort=False)
    if not grouped.size().eq(len(SOURCE_SUPPORT_IDS)).all():
        raise ValueError(f"target/support cardinality mismatch: {entity['entity_uid']}")
    for column in ("seq_id", "comp_id", "atom_id"):
        if not grouped[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"target identity drift: {entity['entity_uid']}:{column}")
    unique = (
        frame.loc[:, ["target_id", "seq_id", "comp_id", "atom_id"]]
        .drop_duplicates("target_id")
        .sort_values("target_id", kind="stable")
    )
    if unique.isna().any().any():
        raise ValueError(f"null target identity: {entity['entity_uid']}")
    return [
        {
            "entity_uid": entity["entity_uid"],
            "target_id": str(row.target_id),
            "seq_id": int(row.seq_id),
            "comp_id": str(row.comp_id).strip().upper(),
            "atom_id": str(row.atom_id).strip().upper(),
        }
        for row in unique.itertuples(index=False)
    ]


def _require_source_scope(source: dict[str, Any]) -> None:
    for field in (
        "target_values_read",
        "source_gate_authorized",
        "formal_evaluation_authorized",
    ):
        if source.get(field) is not False:
            raise ValueError(f"source commitment scope sentinel changed: {field}")


def _evaluate_entity_supports(
    root: Path, entity: dict[str, Any], targets: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    directory = root / PDB_ROOT_RELATIVE / entity["bmrb_id"]
    if directory.is_symlink() or directory.resolve(strict=True) != directory:
        raise ValueError(f"PDB entity directory is indirect: {entity['bmrb_id']}")
    reference_record = next(
        row for row in entity["files"] if int(row["support_index"]) == 1
    )
    reference_path = directory / f"{entity['bmrb_id']}_BioEmu_1.pdb"
    if (
        reference_path.is_symlink()
        or reference_path.resolve(strict=True) != reference_path
    ):
        raise ValueError(f"reference PDB path is indirect: {entity['entity_uid']}")
    reference_raw = reference_path.read_bytes()
    if hashlib.sha256(reference_raw).hexdigest() != reference_record["pdb_sha256"]:
        raise ValueError(f"reference PDB provenance mismatch: {entity['entity_uid']}")
    reference = parse_pdb(reference_raw)
    supports: list[dict[str, Any]] = []
    rejected = Counter()
    rejections: list[dict[str, Any]] = []
    for record in entity["files"]:
        support_index = int(record["support_index"])
        path = directory / f"{entity['bmrb_id']}_BioEmu_{support_index}.pdb"
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError(
                f"support PDB path is indirect: {entity['entity_uid']}:{support_index}"
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["pdb_sha256"]:
            raise ValueError(
                f"support PDB provenance mismatch: {entity['entity_uid']}:{support_index}"
            )
        try:
            supports.append(evaluate_support(raw, reference, record, targets))
        except ValueError as error:
            rejected[str(error)] += 1
            rejections.append(
                {
                    "support_index": support_index,
                    "pdb_sha256": record["pdb_sha256"],
                    "reason": str(error),
                    "status": "POLICY_REJECTED",
                }
            )
    return supports, rejections, dict(sorted(rejected.items()))


def run_shard(index: int) -> dict[str, Any]:
    if not 0 <= index < SHARD_COUNT:
        raise ValueError("shard index is outside 0..26")
    root = _root()
    _bound_file(root, POLICY_RELATIVE, POLICY_SHA256)
    archive = _bound_file(root, ARCHIVE_RELATIVE, ARCHIVE_SHA256)
    source_path = _bound_file(root, SOURCE_RELATIVE, SOURCE_SHA256)
    source = json.loads(source_path.read_bytes())
    _require_source_scope(source)
    entities = _catalog_entities(archive)
    selected = [
        row for position, row in enumerate(entities) if position % SHARD_COUNT == index
    ]
    outputs: list[dict[str, Any]] = []
    for entity in selected:
        targets = _target_rows(root, source, entity)
        supports, rejections, rejected = _evaluate_entity_supports(
            root, entity, targets
        )
        outputs.append(
            {
                "entity_uid": entity["entity_uid"],
                "bmrb_id": entity["bmrb_id"],
                "observer_fold": entity["observer_fold"],
                "split": entity["split"],
                "target_count": len(targets),
                "target_identity_sha256": _canonical_sha256(targets),
                "catalog_support_count": len(entity["files"]),
                "policy_compliant_support_count": len(supports),
                "rejected_support_count": len(rejections),
                "rejected_reason_counts": rejected,
                "rejections": rejections,
                "supports": supports,
            }
        )
    result = {
        "artifact_kind": "hold_only_target_unread_all_atom_recount_shard_not_authorization",
        "contract": "atypemu_nested_support_count_v1_all_atom_recount_shard_v1",
        "study_id": "atypemu_nested_support_count_v1",
        "policy_sha256": POLICY_SHA256,
        "catalog_archive_sha256": ARCHIVE_SHA256,
        "source_commitment_sha256": SOURCE_SHA256,
        "shard_index": index,
        "shard_count": SHARD_COUNT,
        "entity_count": len(outputs),
        "entities": outputs,
        **FALSE_SENTINELS,
    }
    output = root / OUTPUT_RELATIVE / "shards" / f"shard_{index}.json"
    _write_new(output, result)
    return result


def _validate_entity_output(
    output: dict[str, Any],
    catalog: dict[str, Any],
    targets: list[dict[str, Any]],
) -> None:
    if set(output) != ENTITY_FIELDS:
        raise ValueError(f"recount entity schema mismatch: {catalog['entity_uid']}")
    for field in ("entity_uid", "bmrb_id", "observer_fold", "split"):
        if output[field] != catalog[field]:
            raise ValueError(
                f"recount entity identity mismatch: {catalog['entity_uid']}:{field}"
            )
    if (
        type(output["target_count"]) is not int
        or output["target_count"] != len(targets)
        or output["target_identity_sha256"] != _canonical_sha256(targets)
    ):
        raise ValueError(f"target identity receipt mismatch: {catalog['entity_uid']}")
    catalog_by_index = {int(row["support_index"]): row for row in catalog["files"]}
    supports = output["supports"]
    rejections = output["rejections"]
    if not isinstance(supports, list) or not isinstance(rejections, list):
        raise ValueError(
            f"support/rejection rows must be lists: {catalog['entity_uid']}"
        )
    if output["catalog_support_count"] != len(catalog_by_index):
        raise ValueError(f"catalog count mismatch: {catalog['entity_uid']}")
    if output["policy_compliant_support_count"] != len(supports):
        raise ValueError(f"compliant count mismatch: {catalog['entity_uid']}")
    if output["rejected_support_count"] != len(rejections):
        raise ValueError(f"rejected count mismatch: {catalog['entity_uid']}")
    support_indexes: list[int] = []
    for support in supports:
        if not isinstance(support, dict) or set(support) != SUPPORT_FIELDS:
            raise ValueError(f"support schema mismatch: {catalog['entity_uid']}")
        support_index = support["support_index"]
        if type(support_index) is not int or support_index not in catalog_by_index:
            raise ValueError(f"support index mismatch: {catalog['entity_uid']}")
        record = catalog_by_index[support_index]
        if (
            support["status"] != "POLICY_COMPLIANT_MASK_EMITTED"
            or support["pdb_sha256"] != record["pdb_sha256"]
            or support["all_atom_topology_sha256"] != record["all_atom_topology_sha256"]
            or SHA256_RE.fullmatch(support["target_mask_sha256"]) is None
            or SHA256_RE.fullmatch(support["distance_mask_sha256"]) is None
        ):
            raise ValueError(
                f"support provenance/status mismatch: {catalog['entity_uid']}"
            )
        available = support["target_available_count"]
        missing = support["target_missing_count"]
        if (
            type(available) is not int
            or type(missing) is not int
            or available < 0
            or missing < 0
            or available + missing != len(targets)
            or type(support["histidine_tautomer_change_count"]) is not int
            or support["histidine_tautomer_change_count"] < 0
        ):
            raise ValueError(f"support target count mismatch: {catalog['entity_uid']}")
        missing_by_atom = support["target_missing_by_atom"]
        if (
            not isinstance(missing_by_atom, dict)
            or set(missing_by_atom) - {"HIS:HD1", "HIS:HE2"}
            or any(
                type(value) is not int or value <= 0
                for value in missing_by_atom.values()
            )
            or sum(missing_by_atom.values()) != missing
        ):
            raise ValueError(
                f"support missing-target schema mismatch: {catalog['entity_uid']}"
            )
        distance_counts = support["distance_available_count_by_element"]
        if (
            not isinstance(distance_counts, dict)
            or set(distance_counts) != set(ELEMENTS)
            or any(
                type(value) is not int or not 0 <= value <= available
                for value in distance_counts.values()
            )
        ):
            raise ValueError(
                f"support distance count mismatch: {catalog['entity_uid']}"
            )
        support_indexes.append(support_index)
    rejected_indexes: list[int] = []
    reason_counts = Counter()
    for rejection in rejections:
        if not isinstance(rejection, dict) or set(rejection) != REJECTION_FIELDS:
            raise ValueError(f"rejection schema mismatch: {catalog['entity_uid']}")
        support_index = rejection["support_index"]
        if type(support_index) is not int or support_index not in catalog_by_index:
            raise ValueError(f"rejection index mismatch: {catalog['entity_uid']}")
        if (
            rejection["status"] != "POLICY_REJECTED"
            or rejection["pdb_sha256"] != catalog_by_index[support_index]["pdb_sha256"]
            or not isinstance(rejection["reason"], str)
            or not rejection["reason"]
        ):
            raise ValueError(
                f"rejection provenance/status mismatch: {catalog['entity_uid']}"
            )
        rejected_indexes.append(support_index)
        reason_counts[rejection["reason"]] += 1
    if (
        support_indexes != sorted(support_indexes)
        or rejected_indexes != sorted(rejected_indexes)
        or len(set(support_indexes)) != len(support_indexes)
        or len(set(rejected_indexes)) != len(rejected_indexes)
        or len(support_indexes) + len(rejected_indexes) != len(catalog_by_index)
    ):
        raise ValueError(f"support outputs are not ordered: {catalog['entity_uid']}")
    if set(support_indexes).intersection(rejected_indexes) or set(
        support_indexes + rejected_indexes
    ) != set(catalog_by_index):
        raise ValueError(f"support output coverage mismatch: {catalog['entity_uid']}")
    if output["rejected_reason_counts"] != dict(sorted(reason_counts.items())):
        raise ValueError(
            f"rejection reason arithmetic mismatch: {catalog['entity_uid']}"
        )


def aggregate() -> dict[str, Any]:
    root = _root()
    _bound_file(root, POLICY_RELATIVE, POLICY_SHA256)
    _bound_file(root, ARCHIVE_RELATIVE, ARCHIVE_SHA256)
    source_path = _bound_file(root, SOURCE_RELATIVE, SOURCE_SHA256)
    source = json.loads(source_path.read_bytes())
    _require_source_scope(source)
    catalogs = _catalog_entities(root / ARCHIVE_RELATIVE)
    catalog_by_uid = {row["entity_uid"]: row for row in catalogs}
    entities: list[dict[str, Any]] = []
    for index in range(SHARD_COUNT):
        path = root / OUTPUT_RELATIVE / "shards" / f"shard_{index}.json"
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError(f"recount shard path is indirect: {index}")
        shard = json.loads(path.read_bytes())
        if (
            set(shard) != SHARD_FIELDS
            or shard.get("artifact_kind")
            != "hold_only_target_unread_all_atom_recount_shard_not_authorization"
            or shard.get("study_id") != "atypemu_nested_support_count_v1"
            or shard.get("contract")
            != "atypemu_nested_support_count_v1_all_atom_recount_shard_v1"
            or shard.get("shard_index") != index
            or shard.get("shard_count") != SHARD_COUNT
            or shard.get("policy_sha256") != POLICY_SHA256
            or shard.get("catalog_archive_sha256") != ARCHIVE_SHA256
            or shard.get("source_commitment_sha256") != SOURCE_SHA256
            or any(shard.get(key) is not False for key in FALSE_SENTINELS)
        ):
            raise ValueError(f"invalid recount shard: {index}")
        expected_uids = {
            row["entity_uid"]
            for position, row in enumerate(catalogs)
            if position % SHARD_COUNT == index
        }
        if (
            type(shard.get("entity_count")) is not int
            or shard["entity_count"] != len(shard.get("entities", []))
            or {row.get("entity_uid") for row in shard.get("entities", [])}
            != expected_uids
        ):
            raise ValueError(f"recount shard assignment mismatch: {index}")
        for output in shard["entities"]:
            catalog = catalog_by_uid[output["entity_uid"]]
            targets = _target_rows(root, source, catalog)
            _validate_entity_output(output, catalog, targets)
            expected_supports, expected_rejections, expected_reasons = (
                _evaluate_entity_supports(root, catalog, targets)
            )
            if (
                output["supports"] != expected_supports
                or output["rejections"] != expected_rejections
                or output["rejected_reason_counts"] != expected_reasons
            ):
                raise ValueError(
                    f"recount shard differs from raw-byte replay: {catalog['entity_uid']}"
                )
        entities.extend(shard["entities"])
    entities.sort(key=lambda row: row["entity_uid"])
    if (
        len(entities) != ENTITY_COUNT
        or len({row["entity_uid"] for row in entities}) != ENTITY_COUNT
    ):
        raise ValueError("recount does not cover exactly 135 unique entities")
    feasibility = {}
    for level in LEVELS:
        counts = [int(row["policy_compliant_support_count"]) for row in entities]
        feasibility[str(level)] = {
            "all_entities_count_feasible": all(count >= level for count in counts),
            "entity_count_feasible": sum(count >= level for count in counts),
            "maximum_entity_shortfall": max(max(0, level - count) for count in counts),
            "total_shortfall": sum(max(0, level - count) for count in counts),
        }
    result = {
        "artifact_kind": "hold_only_target_unread_all_atom_recount_not_authorization",
        "contract": "atypemu_nested_support_count_v1_all_atom_recount_v1",
        "study_id": "atypemu_nested_support_count_v1",
        "policy_sha256": POLICY_SHA256,
        "catalog_archive_sha256": ARCHIVE_SHA256,
        "source_commitment_sha256": SOURCE_SHA256,
        "entity_count": len(entities),
        "catalog_support_count": sum(
            int(row["catalog_support_count"]) for row in entities
        ),
        "policy_compliant_support_count": sum(
            int(row["policy_compliant_support_count"]) for row in entities
        ),
        "rejected_support_count": sum(
            int(row["rejected_support_count"]) for row in entities
        ),
        "level_count_feasibility": feasibility,
        "entities": entities,
        **FALSE_SENTINELS,
    }
    result["full_recount_digest"] = _canonical_sha256(entities)
    _write_new(root / OUTPUT_RELATIVE / "summary.json", result)
    return result


def _pdb(atoms: list[tuple[str, str, str, int, str]], chain: str = "A") -> bytes:
    lines = []
    for serial, (name, resname, element, seq_id, record) in enumerate(atoms, 1):
        lines.append(
            f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{seq_id:4d}    "
            f"{serial:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}"
        )
    return ("\n".join(lines) + "\nEND\n").encode()


def self_test() -> int:
    reference_raw = _pdb(
        [
            ("N", "ALA", "N", 1, "ATOM"),
            ("H", "ALA", "H", 1, "ATOM"),
            ("CA", "ALA", "C", 1, "ATOM"),
            ("N", "HIS", "N", 2, "ATOM"),
            ("CA", "HIS", "C", 2, "ATOM"),
            ("HE2", "HIS", "H", 2, "ATOM"),
        ]
    )
    support_raw = reference_raw.replace(b" HE2", b" HD1")
    catalog = {
        "support_index": 2,
        "pdb_sha256": hashlib.sha256(support_raw).hexdigest(),
        "all_atom_topology_sha256": "0" * 64,
    }
    targets = [
        {
            "entity_uid": "bmrb:1:entity:1",
            "target_id": "a",
            "seq_id": 1,
            "comp_id": "ALA",
            "atom_id": "HN",
        },
        {
            "entity_uid": "bmrb:1:entity:1",
            "target_id": "b",
            "seq_id": 2,
            "comp_id": "HIS",
            "atom_id": "HD1",
        },
        {
            "entity_uid": "bmrb:1:entity:1",
            "target_id": "c",
            "seq_id": 2,
            "comp_id": "HIS",
            "atom_id": "HE2",
        },
    ]
    result = evaluate_support(support_raw, parse_pdb(reference_raw), catalog, targets)
    assert result["histidine_tautomer_change_count"] == 1
    assert result["target_available_count"] == 2
    assert result["target_missing_by_atom"] == {"HIS:HE2": 1}
    assert dict(zip(ELEMENTS, (True, True, True, True, False))) == {
        "H": True,
        "C": True,
        "N": True,
        "O": True,
        "S": False,
    }
    renamed_targets = [dict(row) for row in targets]
    renamed_targets[0]["target_id"] = "renamed"
    assert (
        _target_masks(parse_pdb(support_raw), renamed_targets)["target_mask_sha256"]
        != result["target_mask_sha256"]
    )
    catalog_entity = {
        "entity_uid": "bmrb:1:entity:1",
        "bmrb_id": "bmr1",
        "observer_fold": "A",
        "split": "train",
        "files": [catalog],
    }
    entity_output = {
        "entity_uid": "bmrb:1:entity:1",
        "bmrb_id": "bmr1",
        "observer_fold": "A",
        "split": "train",
        "target_count": len(targets),
        "target_identity_sha256": _canonical_sha256(targets),
        "catalog_support_count": 1,
        "policy_compliant_support_count": 1,
        "rejected_support_count": 0,
        "rejected_reason_counts": {},
        "rejections": [],
        "supports": [result],
    }
    _validate_entity_output(entity_output, catalog_entity, targets)
    tampered_output = dict(entity_output)
    tampered_output["policy_compliant_support_count"] = 2
    try:
        _validate_entity_output(tampered_output, catalog_entity, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("self-test accepted forged compliant-support count")

    deuterium = _pdb(
        [
            ("N", "ALA", "N", 1, "ATOM"),
            ("H", "ALA", "H", 1, "ATOM"),
            ("CA", "ALA", "C", 1, "ATOM"),
            ("N", "HIS", "N", 2, "ATOM"),
            ("CA", "HIS", "C", 2, "ATOM"),
            ("DD1", "HIS", "D", 2, "ATOM"),
        ]
    )
    non_his_change = support_raw.replace(b"   H ALA", b"  H1 ALA")
    ala_h_line = next(
        line for line in reference_raw.splitlines(keepends=True) if b"   H ALA" in line
    )
    no_ala_h = reference_raw.replace(ala_h_line, b"")
    cases = {
        "deuterium": (deuterium, parse_pdb(reference_raw)),
        "non-HIS hydrogen": (non_his_change, parse_pdb(reference_raw)),
        "missing non-HIS target": (no_ala_h, parse_pdb(no_ala_h)),
    }
    for name, (raw, case_reference) in cases.items():
        bad_catalog = dict(catalog, pdb_sha256=hashlib.sha256(raw).hexdigest())
        try:
            evaluate_support(raw, case_reference, bad_catalog, targets)
        except ValueError:
            pass
        else:
            raise AssertionError(f"self-test accepted {name}")
    ambiguous = support_raw + _pdb([("H", "ALA", "H", 1, "ATOM")], chain="B")
    bad_catalog = dict(catalog, pdb_sha256=hashlib.sha256(ambiguous).hexdigest())
    try:
        evaluate_support(ambiguous, parse_pdb(reference_raw), bad_catalog, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("self-test accepted ambiguous target identity")
    return 9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    shard = subparsers.add_parser("shard")
    shard.add_argument("--shard-index", type=int, required=True)
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "self-test":
        print(f"RECOUNT_POLICY_CHECKS={self_test()}")
    elif args.command == "shard":
        result = run_shard(args.shard_index)
        print("RECOUNT_SHARD_OK", result["shard_index"], result["entity_count"])
    else:
        result = aggregate()
        print("RECOUNT_AGGREGATE_OK", result["entity_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
