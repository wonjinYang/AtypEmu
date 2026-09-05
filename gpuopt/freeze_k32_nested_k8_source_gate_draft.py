#!/usr/bin/env python3
"""Create a target-value-unread draft input commitment for the K32 source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONTRACT = "k32_nested_k8_source_gate_draft_commitment_v1"
STATIC_FILES = (
    ".auto/checks.sh",
    ".auto/frozen/all_label_inventory.json",
    ".auto/runs/k32_dynamic_distance_cache_independent_check_v1r1.json",
    ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json",
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json",
    "data/all_atom_observer_v1/commitment.json",
    "gpuopt/candidates/k32_nested_k8_adapter.py",
    "gpuopt/freeze_k32_nested_k8_source_gate_draft.py",
    "gpuopt/k32_source_gate_authorization.py",
    "gpuopt/preunblind/k32_nested_k8_source_gate_plan_v1.json",
    "gpuopt/run_k32_nested_k8_source_gate.py",
    "gpuopt/source_gate_eligibility.py",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/receipt.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/A/summary.json",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/predictions.parquet",
    "gpuopt/assets/sequence_final_atom_macro_ensemble3_v0/B/summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> tuple[list[Path], list[str]]:
    parent = json.loads((root / "data/all_atom_observer_v1/commitment.json").read_text())
    entities = sorted(
        str(row["entity_uid"])
        for row in parent["entities"]
        if row.get("split") == "train" and row.get("observer_fold") in {"A", "B"}
    )
    if len(entities) != 135 or len(set(entities)) != 135:
        raise ValueError("source entity roster is not the frozen 135-entity cohort")
    receipt = json.loads(
        (root / ".auto/runs/k32_dynamic_distance_cache_receipt_v1.json").read_text()
    )
    cache_rows = {
        str(row["entity_uid"]): str(row["output_relative_path"])
        for row in receipt["outputs"]
    }
    if set(cache_rows) != set(entities):
        raise ValueError("dynamic cache and source entity rosters differ")
    paths = {root / relative for relative in STATIC_FILES}
    for entity in entities:
        bmrb_id = f"bmr{entity.split(':')[1]}"
        paths.add(root / f"data/all_atom_observer_v1/targets/{bmrb_id}.parquet")
        paths.add(root / cache_rows[entity])
    root = root.resolve()
    ordered = sorted(path.resolve() for path in paths)
    if any(root not in path.parents for path in ordered):
        raise ValueError("draft commitment input escapes project root")
    missing = [path for path in ordered if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing draft source input: {missing[0]}")
    return ordered, entities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths, entities = collect(root)
    with ThreadPoolExecutor(max_workers=16) as executor:
        hashes = list(executor.map(sha256, paths))
    files = {
        str(path.relative_to(root)): digest
        for path, digest in zip(paths, hashes, strict=True)
    }
    target_files = [path for path in files if "/targets/" in path]
    cache_files = [
        path for path in files if path.startswith("data/k32_dynamic_distance_cache_v1/")
    ]
    if len(target_files) != 135 or len(cache_files) != 135:
        raise ValueError("draft commitment input cardinality mismatch")
    payload = {
        "authorization_allowed": False,
        "cache_file_count": len(cache_files),
        "contract": CONTRACT,
        "draft_only": True,
        "files": files,
        "formal_or_outer_access": False,
        "runner_and_checker_pending": True,
        "source_entity_count": len(entities),
        "source_entity_uids": entities,
        "target_file_count": len(target_files),
        "target_values_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("METRIC draft_source_entities=135")
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
