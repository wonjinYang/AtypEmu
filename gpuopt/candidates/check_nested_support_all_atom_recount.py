#!/usr/bin/env python3
"""Independently replay the sealed HOLD-only all-atom recount evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

SHA256 = {
    "catalog": "69fee89d20588cbeb2a15cc1c4a4f002f63a50928f2028f5835c5b3b871061c2",
    "execution_receipt": "70fe86bc0d3f28b6902a3ea29a457aff6cb388aff2926f0c23b4fa4a3e08f541",
    "policy": "f792ea47639f19d0376088b2c2e2fd5e9a161f33e4871205664bc9972e1b35ae",
    "producer": "41cd2261292d47c724138f8a671d50423245fa19eff0e71f1a5a95b19215e90c",
    "receipt": "7f119c216ebfcfe4bfe2dfb5a607d1cd3689f6ecbca7ca69bafab4b51a985f1c",
    "result": "790a5cd77fdc10b46e5df85d80013938bb48a81732f5f9db00a7ef66cd483f16",
    "source": "af8ae50e7b704181471be6d86794cc45d99562136d50152e65c9fbe8df5b1ca8",
    "summary": "07fb86fc5ec4602d8e9d93856c8ff26af40dd0f45fe8754e398198cac502350c",
}
PATHS = {
    "catalog": ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz",
    "execution_receipt": ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1_yulab/execution_receipt.json",
    "policy": "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_policy.json",
    "producer": "gpuopt/candidates/nested_support_all_atom_recount.py",
    "receipt": "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_recount_receipt.json",
    "result": ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1_yulab/recount_results.tar.gz",
    "source": ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json",
}
RECEIPT_EVIDENCE_KEYS = {
    "catalog_shard_archive",
    "execution_receipt",
    "policy",
    "producer",
    "result_archive",
    "source_commitment",
}
FALSE_SENTINELS = {
    "authorization_consumed",
    "outer_or_formal_metrics_opened",
    "science_executed",
    "source_scores_read",
    "target_values_read",
}
LEVELS = (32, 128, 768, 1536)
PREFIX = "atypemu_nested_support_count_v1_all_atom_recount_v1"
BLOCKED_UID = "bmrb:50238:entity:1"
BLOCKED_REASON = (
    "non-HIS target atom is unavailable: "
    "bmrb:50238:entity:1:target:cs:1:328:1:1:_:36:GLU:HE2"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _read_bound(root: Path, name: str) -> bytes:
    path = root / PATHS[name]
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"indirect evidence path: {name}")
    raw = path.read_bytes()
    if name in SHA256 and _sha256(raw) != SHA256[name]:
        raise ValueError(f"evidence hash mismatch: {name}")
    return raw


def _json_members(
    raw: bytes, names: set[str]
) -> tuple[list[tarfile.TarInfo], dict[str, tuple[dict[str, Any], str]]]:
    """Read wanted members in one forward gzip pass instead of repeated seeks."""
    members = []
    values = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as archive:
        for member in archive:
            members.append(member)
            if member.name not in names or not member.isfile():
                continue
            if member.name in values:
                raise ValueError(f"duplicate archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"missing archive member: {member.name}")
            member_raw = stream.read()
            value = json.loads(member_raw)
            if not isinstance(value, dict):
                raise ValueError(f"non-object archive member: {member.name}")
            values[member.name] = (value, _sha256(member_raw))
    if set(values) != names:
        raise ValueError(f"missing archive members: {sorted(names - set(values))}")
    return members, values


def _assert_false_sentinels(value: dict[str, Any], where: str) -> None:
    if any(value.get(key) is not False for key in FALSE_SENTINELS):
        raise ValueError(f"forbidden capability in {where}")


def _validate_receipt_evidence(receipt: dict[str, Any]) -> None:
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != RECEIPT_EVIDENCE_KEYS:
        raise ValueError("published evidence key set mismatch")
    for name, binding in evidence.items():
        key = {
            "catalog_shard_archive": "catalog",
            "result_archive": "result",
            "source_commitment": "source",
        }.get(name, name)
        if binding != {"path": PATHS[key], "sha256": SHA256[key]}:
            raise ValueError(f"published evidence binding mismatch: {name}")


def _validate_support_partition(
    uid: str,
    catalog_indexes: set[int],
    accepted_indexes: list[int],
    rejected_indexes: list[int],
) -> None:
    if (
        accepted_indexes != sorted(accepted_indexes)
        or rejected_indexes != sorted(rejected_indexes)
        or len(accepted_indexes) != len(set(accepted_indexes))
        or len(rejected_indexes) != len(set(rejected_indexes))
        or len(accepted_indexes) + len(rejected_indexes) != len(catalog_indexes)
        or set(accepted_indexes).intersection(rejected_indexes)
        or set(accepted_indexes + rejected_indexes) != catalog_indexes
    ):
        raise ValueError(f"support partition mismatch: {uid}")


def _catalog_entities(raw: bytes) -> list[dict[str, Any]]:
    shard_names = {f"./shard_{index}.json" for index in range(27)}
    members, values = _json_members(raw, shard_names)
    names = {member.name for member in members}
    expected = {".", "./catalog_summary.json", "./SHA256SUMS"} | shard_names
    if names != expected or any(
        not (member.isdir() if member.name == "." else member.isfile())
        for member in members
    ):
        raise ValueError("catalog archive member roster mismatch")
    entities = []
    for index in range(27):
        shard, _ = values[f"./shard_{index}.json"]
        _assert_false_sentinels(shard, f"catalog shard {index}")
        if shard.get("shard_index") != index or shard.get("shard_count") != 27:
            raise ValueError(f"catalog shard identity mismatch: {index}")
        entities.extend(shard.get("entities", []))
    entities.sort(key=lambda row: row["entity_uid"])
    if len(entities) != 135 or len({row["entity_uid"] for row in entities}) != 135:
        raise ValueError("catalog does not contain 135 unique entities")
    return entities


def _expected_level_counts(entities: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [row["policy_compliant_support_count"] for row in entities]
    return {
        str(level): {
            "all_entities_count_feasible": all(count >= level for count in counts),
            "entity_count_feasible": sum(count >= level for count in counts),
            "maximum_entity_shortfall": max(max(0, level - count) for count in counts),
            "total_shortfall": sum(max(0, level - count) for count in counts),
        }
        for level in LEVELS
    }


def verify(root: Path) -> int:
    raw = {name: _read_bound(root, name) for name in SHA256 if name != "summary"}
    execution = json.loads(raw["execution_receipt"])
    policy = json.loads(raw["policy"])
    receipt = json.loads(raw["receipt"])
    _validate_receipt_evidence(receipt)
    _assert_false_sentinels(execution, "execution receipt")
    _assert_false_sentinels(receipt, "published receipt")
    if (
        policy.get("state") != "FROZEN_POLICY_RECOUNT_NOT_RUN"
        or policy.get("qualification", {}).get(
            "non_histidine_missing_target_invalidates_support"
        )
        is not True
        or policy.get("qualification", {}).get(
            "future_support_rosters_may_use_target_availability_for_selection"
        )
        is not False
    ):
        raise ValueError("frozen policy was relaxed")

    catalogs = _catalog_entities(raw["catalog"])
    catalog_by_uid = {row["entity_uid"]: row for row in catalogs}
    expected_files = {f"{PREFIX}/summary.json"} | {
        f"{PREFIX}/shards/shard_{index}.json" for index in range(27)
    }
    members, values = _json_members(raw["result"], expected_files)
    actual_files = {member.name for member in members if member.isfile()}
    if actual_files != expected_files or any(
        not (member.isfile() or member.isdir())
        or member.issym()
        or member.islnk()
        or member.name.startswith("/")
        or ".." in Path(member.name).parts
        for member in members
    ):
        raise ValueError("result archive member roster is unsafe or incomplete")
    summary, summary_sha256 = values[f"{PREFIX}/summary.json"]
    if summary_sha256 != SHA256["summary"]:
        raise ValueError("summary hash mismatch")
    _assert_false_sentinels(summary, "summary")
    summary_by_uid = {row["entity_uid"]: row for row in summary["entities"]}
    output_entities = []
    for index in range(27):
        shard, _ = values[f"{PREFIX}/shards/shard_{index}.json"]
        _assert_false_sentinels(shard, f"result shard {index}")
        expected_uids = {
            row["entity_uid"]
            for position, row in enumerate(catalogs)
            if position % 27 == index
        }
        if (
            shard.get("shard_index") != index
            or shard.get("shard_count") != 27
            or shard.get("entity_count") != len(shard.get("entities", []))
            or {row["entity_uid"] for row in shard.get("entities", [])}
            != expected_uids
        ):
            raise ValueError(f"result shard identity mismatch: {index}")
        for output in shard["entities"]:
            uid = output["entity_uid"]
            if output != summary_by_uid.get(uid):
                raise ValueError(f"shard/summary entity mismatch: {uid}")
            catalog = catalog_by_uid[uid]
            for field in ("entity_uid", "bmrb_id", "observer_fold", "split"):
                if output[field] != catalog[field]:
                    raise ValueError(f"entity identity mismatch: {uid}:{field}")
            catalog_by_index = {
                row["support_index"]: row for row in catalog["files"]
            }
            accepted = output["supports"]
            rejected = output["rejections"]
            accepted_indexes = [row["support_index"] for row in accepted]
            rejected_indexes = [row["support_index"] for row in rejected]
            if (
                output["catalog_support_count"] != len(catalog_by_index)
                or output["policy_compliant_support_count"] != len(accepted)
                or output["rejected_support_count"] != len(rejected)
            ):
                raise ValueError(f"support-count mismatch: {uid}")
            _validate_support_partition(
                uid, set(catalog_by_index), accepted_indexes, rejected_indexes
            )
            reasons = Counter()
            for row in accepted:
                source = catalog_by_index[row["support_index"]]
                if (
                    row["status"] != "POLICY_COMPLIANT_MASK_EMITTED"
                    or row["pdb_sha256"] != source["pdb_sha256"]
                    or row["all_atom_topology_sha256"]
                    != source["all_atom_topology_sha256"]
                    or row["target_available_count"] + row["target_missing_count"]
                    != output["target_count"]
                    or set(row["distance_available_count_by_element"])
                    != {"H", "C", "N", "O", "S"}
                    or any(
                        len(row[key]) != 64
                        or any(char not in "0123456789abcdef" for char in row[key])
                        for key in ("target_mask_sha256", "distance_mask_sha256")
                    )
                ):
                    raise ValueError(f"accepted support mismatch: {uid}")
            for row in rejected:
                source = catalog_by_index[row["support_index"]]
                if (
                    row["status"] != "POLICY_REJECTED"
                    or row["pdb_sha256"] != source["pdb_sha256"]
                    or not row["reason"]
                ):
                    raise ValueError(f"rejected support mismatch: {uid}")
                reasons[row["reason"]] += 1
            if output["rejected_reason_counts"] != dict(sorted(reasons.items())):
                raise ValueError(f"rejection arithmetic mismatch: {uid}")
            output_entities.append(output)
    output_entities.sort(key=lambda row: row["entity_uid"])
    blocked = [row for row in output_entities if row["rejected_support_count"]]
    if (
        len(output_entities) != 135
        or len(blocked) != 1
        or blocked[0]["entity_uid"] != BLOCKED_UID
        or blocked[0]["catalog_support_count"] != 1000
        or blocked[0]["policy_compliant_support_count"] != 0
        or blocked[0]["rejected_reason_counts"] != {BLOCKED_REASON: 1000}
    ):
        raise ValueError("blocking-entity evidence mismatch")
    levels = _expected_level_counts(output_entities)
    expected_counts = {
        "entity_count": 135,
        "catalog_support_count": 134850,
        "policy_compliant_support_count": 133850,
        "rejected_support_count": 1000,
    }
    if (
        any(summary[key] != value for key, value in expected_counts.items())
        or summary["level_count_feasibility"] != levels
        or summary["full_recount_digest"] != _canonical_sha256(output_entities)
        or execution["level_count_feasibility"] != levels
        or receipt["level_count_feasibility"] != levels
        or any(execution[key] != value for key, value in expected_counts.items())
        or any(receipt["execution"][key] != value for key, value in expected_counts.items())
        or receipt["status"]
        != "ALL_ATOM_COUNT_BLOCKED_ONE_ENTITY_NON_HIS_TARGET_UNAVAILABLE"
        or receipt["claims"]["all_atom_count_feasible_levels"] != []
        or receipt["claims"]["post_result_policy_relaxation_allowed"] is not False
    ):
        raise ValueError("published count, digest, claim, or level arithmetic mismatch")
    return 134850 + 135 + 27 + 8


def self_test(root: Path) -> int:
    checks = verify(root)
    execution = json.loads(_read_bound(root, "execution_receipt"))
    execution["target_values_read"] = True
    try:
        _assert_false_sentinels(execution, "tampered receipt")
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden-capability tamper was accepted")
    receipt = json.loads(_read_bound(root, "receipt"))
    receipt["evidence"]["unexpected"] = {"path": "unexpected", "sha256": "0" * 64}
    try:
        _validate_receipt_evidence(receipt)
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected receipt evidence key was accepted")
    try:
        _validate_support_partition("duplicate", {1, 2}, [1, 1, 2], [])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate support index was accepted")
    return checks + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).absolute().parents[2]
    checks = self_test(root) if args.self_test else verify(root)
    print(f"METRIC all_atom_recount_checks={checks}")
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    print("METRIC authorization_consumed=0")
    print("STATUS HOLD_ALL_ATOM_COUNT_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
