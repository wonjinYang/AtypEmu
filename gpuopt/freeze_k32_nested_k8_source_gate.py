#!/usr/bin/env python3
"""Freeze the final target-value-unread K32 source-gate commitment."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gpuopt.freeze_k32_nested_k8_source_gate_draft import (
    STATIC_FILES as DRAFT_STATIC_FILES,
)
from gpuopt.freeze_k32_nested_k8_source_gate_draft import collect as collect_draft
from gpuopt.freeze_k32_nested_k8_source_gate_draft import sha256

CONTRACT = "k32_nested_k8_source_gate_source_commitment_v1"
IMAGE_SHA256 = "67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531"
STATIC_FILES = (
    *DRAFT_STATIC_FILES,
    "gpuopt/freeze_k32_nested_k8_source_gate.py",
    "gpuopt/slurm/run_k32_nested_k8_source_gate_l40s.sbatch",
)


def collect(root: Path) -> tuple[list[Path], list[str]]:
    paths, entities = collect_draft(root)
    additions = [
        root / relative
        for relative in STATIC_FILES
        if relative not in DRAFT_STATIC_FILES
    ]
    paths = sorted({*paths, *(path.resolve() for path in additions)})
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"missing final source input: {missing[0]}")
    return paths, entities


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
    targets = [path for path in files if "/targets/" in path]
    caches = [
        path for path in files
        if path.startswith("data/k32_dynamic_distance_cache_v1/")
    ]
    if len(targets) != 135 or len(caches) != 135:
        raise ValueError("final source commitment cardinality mismatch")
    payload = {
        "assimilation_steps": 100,
        "authorization_allowed": True,
        "cache_file_count": len(caches),
        "container_image_sha256": IMAGE_SHA256,
        "contract": CONTRACT,
        "draft_only": False,
        "files": files,
        "formal_or_outer_access": False,
        "observer_epochs": 1024,
        "required_node": "iREMB-C-08",
        "required_partition": "l40sq",
        "source_entity_count": len(entities),
        "source_entity_uids": entities,
        "target_file_count": len(targets),
        "target_values_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("METRIC source_commitment_entities=135")
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
