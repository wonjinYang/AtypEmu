"""Inventory target-unread BioEmu PDB catalogs for the support-count HOLD study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any


LEVELS = (32, 128, 768, 1536)
ROSTER_FIELDS = {
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
ENTITY_FIELDS = {
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)


def load_roster(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = path.read_bytes()
    roster = json.loads(raw)
    if set(roster) != ROSTER_FIELDS:
        raise ValueError("entity roster schema mismatch")
    expected = {
        "artifact_kind": "target_unread_structural_entity_roster_not_authorization",
        "authorization_consumed": False,
        "contract": "atypemu_nested_support_count_v1_entity_roster_v3",
        "outer_or_formal_metrics_opened": False,
        "source_scores_read": False,
        "study_id": "atypemu_nested_support_count_v1",
        "target_values_read": False,
    }
    for key, value in expected.items():
        if roster.get(key) != value:
            raise ValueError(f"unsafe entity roster field: {key}")
    entities = roster.get("entities")
    if not isinstance(entities, list) or roster.get("entity_count") != 135:
        raise ValueError("entity roster count mismatch")
    normalized: list[dict[str, Any]] = []
    for row in entities:
        if not isinstance(row, dict) or set(row) != ENTITY_FIELDS:
            raise ValueError("entity row schema mismatch")
        bmrb_id = str(row["bmrb_id"])
        entity_uid = str(row["entity_uid"])
        match = re.fullmatch(r"bmr(\d+)", bmrb_id)
        if (
            match is None
            or entity_uid != f"bmrb:{match.group(1)}:entity:1"
            or row["split"] != "train"
            or row["observer_fold"] not in {"A", "B"}
            or row["canonical_reference_support_index"] != 1
            or int(row["canonical_atom_count"]) < 1
            or int(row["canonical_heavy_atom_count"]) < 1
            or int(row["canonical_heavy_atom_count"]) > int(row["canonical_atom_count"])
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row["canonical_all_atom_topology_sha256"])
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row["canonical_heavy_topology_sha256"])
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row["canonical_reference_pdb_sha256"])
            )
            is None
            or row["canonical_reference_relative_path"]
            != f"data/k32_complete_coordinate_supports_v4/{bmrb_id}/"
            f"{bmrb_id}_BioEmu_1.pdb"
        ):
            raise ValueError(f"invalid target-unread entity identity: {entity_uid}")
        normalized.append(
            {
                **{key: str(row[key]) for key in (
                    "bmrb_id",
                    "canonical_all_atom_topology_sha256",
                    "canonical_heavy_topology_sha256",
                    "canonical_reference_pdb_sha256",
                    "canonical_reference_relative_path",
                    "entity_uid",
                    "observer_fold",
                    "split",
                )},
                "canonical_atom_count": int(row["canonical_atom_count"]),
                "canonical_heavy_atom_count": int(row["canonical_heavy_atom_count"]),
                "canonical_reference_support_index": int(
                    row["canonical_reference_support_index"]
                ),
            }
        )
    normalized.sort(key=lambda row: row["entity_uid"])
    if (
        len(normalized) != 135
        or len({row["entity_uid"] for row in normalized}) != 135
        or len({row["bmrb_id"] for row in normalized}) != 135
    ):
        raise ValueError("duplicate or incomplete entity roster")
    return {
        "roster_sha256": sha256_bytes(raw),
        "source_commitment_relative_path": roster[
            "source_commitment_relative_path"
        ],
        "source_commitment_sha256": roster["source_commitment_sha256"],
    }, normalized


def parse_coordinate_fingerprints(raw: bytes) -> dict[str, Any]:
    topology = hashlib.sha256()
    coordinates = hashlib.sha256()
    heavy_topology = hashlib.sha256()
    heavy_coordinates = hashlib.sha256()
    seen: set[tuple[str, str, str, int, str, int, str, str]] = set()
    segment = 0
    atom_count = 0
    heavy_atom_count = 0
    for line in raw.decode("ascii", errors="replace").splitlines():
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
        identity = (
            line[:6].strip(),
            name,
            resname,
            segment,
            chain,
            seq_id,
            insertion,
            element,
        )
        if identity in seen:
            raise ValueError(f"duplicate atom identity: {identity}")
        seen.add(identity)
        xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("nonfinite coordinate")
        identity_bytes = json.dumps(identity, separators=(",", ":")).encode()
        topology.update(identity_bytes)
        topology.update(b"\n")
        coordinates.update(identity_bytes)
        coordinates.update(struct.pack(">ddd", *xyz))
        atom_count += 1
        if element not in {"H", "D"}:
            heavy_topology.update(identity_bytes)
            heavy_topology.update(b"\n")
            heavy_coordinates.update(identity_bytes)
            heavy_coordinates.update(struct.pack(">ddd", *xyz))
            heavy_atom_count += 1
    if atom_count == 0:
        raise ValueError("PDB contains no canonical atoms")
    if heavy_atom_count == 0:
        raise ValueError("PDB contains no canonical heavy atoms")
    return {
        "all_atom_coordinate_sha256": coordinates.hexdigest(),
        "all_atom_topology_sha256": topology.hexdigest(),
        "atom_count": atom_count,
        "heavy_atom_coordinate_sha256": heavy_coordinates.hexdigest(),
        "heavy_atom_count": heavy_atom_count,
        "heavy_atom_topology_sha256": heavy_topology.hexdigest(),
    }


def parse_coordinate_hashes(raw: bytes) -> tuple[str, str, int]:
    """Compatibility helper for existing callers of the all-atom fingerprints."""
    fingerprints = parse_coordinate_fingerprints(raw)
    return (
        str(fingerprints["all_atom_topology_sha256"]),
        str(fingerprints["all_atom_coordinate_sha256"]),
        int(fingerprints["atom_count"]),
    )


def _contained_directory(root: Path, bmrb_id: str) -> Path:
    if root.is_symlink():
        raise ValueError("PDB root may not be a symlink")
    resolved_root = root.resolve(strict=True)
    directory = root / bmrb_id
    if directory.is_symlink():
        raise ValueError(f"entity directory may not be a symlink: {bmrb_id}")
    resolved = directory.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"entity directory escapes PDB root: {bmrb_id}")
    return resolved


def scan_entity(root: Path, entity: dict[str, Any]) -> dict[str, Any]:
    bmrb_id = entity["bmrb_id"]
    directory = _contained_directory(root, bmrb_id)
    pattern = re.compile(rf"{re.escape(bmrb_id)}_BioEmu_(\d+)\.pdb")
    indexed: dict[int, Path] = {}
    unexpected: list[str] = []
    for entry in os.scandir(directory):
        match = pattern.fullmatch(entry.name)
        if not match or not entry.is_file(follow_symlinks=False) or entry.is_symlink():
            unexpected.append(entry.name)
            continue
        index = int(match.group(1))
        if not 1 <= index <= 1000 or index in indexed:
            unexpected.append(entry.name)
            continue
        indexed[index] = Path(entry.path)
    files: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    heavy_topology_reference = str(entity["canonical_heavy_topology_sha256"])
    heavy_coordinate_hashes: set[str] = set()
    all_atom_topology_variants: set[str] = set()
    for index, path in sorted(indexed.items()):
        raw = path.read_bytes()
        raw_hash = sha256_bytes(raw)
        try:
            fingerprints = parse_coordinate_fingerprints(raw)
            if fingerprints["heavy_atom_topology_sha256"] != heavy_topology_reference:
                raise ValueError("heavy-atom topology differs from committed K32 reference")
            heavy_coordinate_hash = str(fingerprints["heavy_atom_coordinate_sha256"])
            if heavy_coordinate_hash in heavy_coordinate_hashes:
                raise ValueError("exact heavy-atom coordinate duplicate")
            heavy_coordinate_hashes.add(heavy_coordinate_hash)
            all_atom_topology_variants.add(
                str(fingerprints["all_atom_topology_sha256"])
            )
        except (UnicodeError, ValueError) as error:
            invalid.append(
                {
                    "error": str(error),
                    "pdb_sha256": raw_hash,
                    "support_index": index,
                }
            )
            continue
        files.append(
            {
                **fingerprints,
                "all_atom_topology_matches_reference": fingerprints[
                    "all_atom_topology_sha256"
                ]
                == entity["canonical_all_atom_topology_sha256"],
                "pdb_sha256": raw_hash,
                "support_index": index,
            }
        )
    valid_indices = {int(row["support_index"]) for row in files}
    return {
        **entity,
        "canonical_filename_count": len(indexed),
        "catalog_digest": canonical_json_sha256(files),
        "files": files,
        "all_atom_topology_variant_count": len(all_atom_topology_variants),
        "hydrogen_topology_variation_present": len(all_atom_topology_variants) > 1,
        "invalid": invalid,
        "invalid_count": len(invalid),
        "missing_indices_1_to_1000": sorted(set(range(1, 1001)) - set(indexed)),
        "heavy_topology_compatible_unique_coordinate_count": len(files),
        "unexpected_entries": sorted(unexpected),
        "valid_index_digest": canonical_json_sha256(sorted(valid_indices)),
    }


def make_shard(
    roster_path: Path,
    pdb_root: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard specification")
    roster_binding, entities = load_roster(roster_path)
    selected = [row for index, row in enumerate(entities) if index % shard_count == shard_index]
    rows = [scan_entity(pdb_root, row) for row in selected]
    write_json_exclusive(
        output,
        {
            "artifact_kind": "target_unread_coordinate_catalog_shard_not_authorization",
            "authorization_consumed": False,
            "contract": "atypemu_nested_support_count_v1_catalog_shard_v2",
            "entities": rows,
            "entity_count": len(rows),
            "outer_or_formal_metrics_opened": False,
            **roster_binding,
            "science_executed": False,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "source_scores_read": False,
            "study_id": "atypemu_nested_support_count_v1",
            "target_values_read": False,
        },
    )


def aggregate_shards(roster_path: Path, shard_dir: Path, output: Path) -> None:
    roster_binding, expected_entities = load_roster(roster_path)
    expected_ids = {row["entity_uid"] for row in expected_entities}
    paths = sorted(shard_dir.glob("shard_*.json"))
    if not paths:
        raise ValueError("no catalog shards")
    rows: list[dict[str, Any]] = []
    shard_count: int | None = None
    seen_shards: set[int] = set()
    for path in paths:
        shard = json.loads(path.read_bytes())
        if (
            shard.get("contract")
            != "atypemu_nested_support_count_v1_catalog_shard_v2"
            or shard.get("artifact_kind")
            != "target_unread_coordinate_catalog_shard_not_authorization"
            or any(
                shard.get(field) is not False
                for field in (
                    "authorization_consumed",
                    "outer_or_formal_metrics_opened",
                    "science_executed",
                    "source_scores_read",
                    "target_values_read",
                )
            )
            or any(shard.get(field) != value for field, value in roster_binding.items())
        ):
            raise ValueError(f"unsafe or mismatched catalog shard: {path}")
        current_count = int(shard["shard_count"])
        if shard_count is None:
            shard_count = current_count
        index = int(shard["shard_index"])
        if current_count != shard_count or index in seen_shards:
            raise ValueError("shard identity mismatch")
        seen_shards.add(index)
        rows.extend(shard["entities"])
    if seen_shards != set(range(int(shard_count or 0))):
        raise ValueError("catalog shard set incomplete")
    rows.sort(key=lambda row: str(row["entity_uid"]))
    if (
        len(rows) != 135
        or {str(row["entity_uid"]) for row in rows} != expected_ids
        or len({str(row["bmrb_id"]) for row in rows}) != 135
    ):
        raise ValueError("catalog entity coverage mismatch")
    level_status: dict[str, Any] = {}
    for level in LEVELS:
        counts = [
            int(row["heavy_topology_compatible_unique_coordinate_count"])
            for row in rows
        ]
        level_status[str(level)] = {
            "all_entities_count_feasible": all(count >= level for count in counts),
            "entity_count_feasible": sum(count >= level for count in counts),
            "maximum_entity_shortfall": max(max(0, level - count) for count in counts),
            "total_shortfall": sum(max(0, level - count) for count in counts),
        }
    summary_rows = [
        {
            key: row[key]
            for key in (
                "bmrb_id",
                "all_atom_topology_variant_count",
                "canonical_filename_count",
                "catalog_digest",
                "entity_uid",
                "heavy_topology_compatible_unique_coordinate_count",
                "hydrogen_topology_variation_present",
                "invalid_count",
                "missing_indices_1_to_1000",
                "observer_fold",
                "split",
                "unexpected_entries",
                "valid_index_digest",
            )
        }
        for row in rows
    ]
    write_json_exclusive(
        output,
        {
            "artifact_kind": "target_unread_coordinate_catalog_receipt_not_authorization",
            "authorization_consumed": False,
            "catalog_shard_count": shard_count,
            "contract": "atypemu_nested_support_count_v1_catalog_receipt_v2",
            "entities": summary_rows,
            "entity_count": len(rows),
            "full_catalog_digest": canonical_json_sha256(rows),
            "level_count_feasibility": level_status,
            "outer_or_formal_metrics_opened": False,
            **roster_binding,
            "science_executed": False,
            "source_method_composition": {"BioEmu": 1.0},
            "source_scores_read": False,
            "study_id": "atypemu_nested_support_count_v1",
            "target_values_read": False,
        },
    )


def _test_pdb(
    offset: float = 0.0,
    atom_name: str = "CA",
    hydrogen_name: str = "H",
) -> bytes:
    lines = [
        f"ATOM      1 {atom_name:>4s} ALA A   1    "
        f"{offset:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          C \n",
        f"ATOM      2 {hydrogen_name:>4s} ALA A   1    "
        f"{offset:8.3f}{1.0:8.3f}{0.0:8.3f}  1.00  0.00          H \n",
    ]
    return ("".join(lines) + "END\n").encode()


def self_test() -> int:
    checks = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "BioEmu"
        directory = root / "bmr1"
        directory.mkdir(parents=True)
        (directory / "bmr1_BioEmu_1.pdb").write_bytes(_test_pdb())
        (directory / "bmr1_BioEmu_2.pdb").write_bytes(_test_pdb(1.0))
        reference = parse_coordinate_fingerprints(_test_pdb())
        entity = {
            "bmrb_id": "bmr1",
            "canonical_all_atom_topology_sha256": reference[
                "all_atom_topology_sha256"
            ],
            "canonical_atom_count": reference["atom_count"],
            "canonical_heavy_atom_count": reference["heavy_atom_count"],
            "canonical_heavy_topology_sha256": reference[
                "heavy_atom_topology_sha256"
            ],
            "canonical_reference_pdb_sha256": sha256_bytes(_test_pdb()),
            "canonical_reference_relative_path": (
                "data/k32_complete_coordinate_supports_v4/bmr1/"
                "bmr1_BioEmu_1.pdb"
            ),
            "canonical_reference_support_index": 1,
            "entity_uid": "bmrb:1:entity:1",
            "observer_fold": "A",
            "split": "train",
        }
        row = scan_entity(root, entity)
        if row["heavy_topology_compatible_unique_coordinate_count"] != 2:
            raise AssertionError("valid synthetic catalog count mismatch")
        checks += 1
        if row["missing_indices_1_to_1000"][:2] != [3, 4]:
            raise AssertionError("missing-index inventory mismatch")
        checks += 1
        (directory / "bmr1_BioEmu_3.pdb").write_bytes(_test_pdb(1.0))
        row = scan_entity(root, entity)
        if row["invalid_count"] != 1 or "duplicate" not in row["invalid"][0]["error"]:
            raise AssertionError("canonical-coordinate duplicate accepted")
        checks += 1
        (directory / "bmr1_BioEmu_3.pdb").write_bytes(_test_pdb(2.0, "N"))
        row = scan_entity(root, entity)
        if row["invalid_count"] != 1 or "topology" not in row["invalid"][0]["error"]:
            raise AssertionError("topology mismatch accepted")
        checks += 1
        (directory / "bmr1_BioEmu_3.pdb").write_bytes(
            _test_pdb(2.0, hydrogen_name="H1")
        )
        row = scan_entity(root, entity)
        if (
            row["heavy_topology_compatible_unique_coordinate_count"] != 3
            or row["all_atom_topology_variant_count"] != 2
            or not row["hydrogen_topology_variation_present"]
        ):
            raise AssertionError("physical hydrogen-topology variation was rejected")
        checks += 1
        (directory / "unexpected.pdb").write_bytes(_test_pdb(3.0))
        row = scan_entity(root, entity)
        if row["unexpected_entries"] != ["unexpected.pdb"]:
            raise AssertionError("unexpected PDB was hidden")
        checks += 1
        symlink = directory / "bmr1_BioEmu_4.pdb"
        symlink.symlink_to(directory / "bmr1_BioEmu_1.pdb")
        row = scan_entity(root, entity)
        if "bmr1_BioEmu_4.pdb" not in row["unexpected_entries"]:
            raise AssertionError("symlinked support was accepted")
        checks += 1
        hashes = parse_coordinate_hashes(_test_pdb())
        if len(hashes[0]) != 64 or len(hashes[1]) != 64 or hashes[2] != 2:
            raise AssertionError("canonical PDB hashes invalid")
        checks += 1
        output = Path(temporary) / "receipt.json"
        write_json_exclusive(output, {"safe": True})
        try:
            write_json_exclusive(output, {"safe": True})
        except FileExistsError:
            checks += 1
        else:
            raise AssertionError("exclusive receipt write allowed overwrite")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--roster", type=Path, required=True)
    shard.add_argument("--pdb-root", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--roster", type=Path, required=True)
    aggregate.add_argument("--shard-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "shard":
        make_shard(
            args.roster,
            args.pdb_root,
            args.output,
            args.shard_index,
            args.shard_count,
        )
    elif args.command == "aggregate":
        aggregate_shards(args.roster, args.shard_dir, args.output)
    else:
        checks = self_test()
        print(f"METRIC support_catalog_checks={checks}")
        print("METRIC source_target_values_read=0")
        print("METRIC outer_or_formal_metrics_opened=0")
        print("METRIC authorization_consumed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
