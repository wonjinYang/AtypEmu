#!/usr/bin/env python3
"""Preflight and eventually execute the matched K32 source-only gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DRAFT_CONTRACT = "k32_nested_k8_source_gate_draft_commitment_v1"
PREFLIGHT_CONTRACT = "k32_nested_k8_source_gate_call_graph_preflight_v1"
PRODUCTION_READY = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_draft_commitment(root: Path, path: Path) -> dict[str, Any]:
    draft = json.loads(path.read_text())
    required = {
        "contract": DRAFT_CONTRACT,
        "draft_only": True,
        "authorization_allowed": False,
        "formal_or_outer_access": False,
        "runner_and_checker_pending": True,
        "source_entity_count": 135,
        "target_file_count": 135,
        "cache_file_count": 135,
        "target_values_opened": False,
    }
    if any(draft.get(key) != value for key, value in required.items()):
        raise ValueError("draft source commitment metadata mismatch")
    files = draft.get("files")
    if not isinstance(files, dict):
        raise ValueError("draft source commitment has no file manifest")
    root = root.resolve()
    for relative, expected in files.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents or len(str(expected)) != 64:
            raise ValueError("draft source commitment path or hash is malformed")
        if sha256(candidate) != expected:
            raise ValueError(f"draft source commitment hash mismatch: {relative}")
    return draft


def preflight_science_call_graph(root: Path) -> dict[str, Any]:
    """Reach every nested pre-science adapter with target reads guarded."""
    from unittest import mock

    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from gpuopt.candidates import k32_nested_k8_adapter as adapter

    original_reader = adapter.pd.read_parquet
    target_reader_calls = 0

    def guarded_reader(path: Path | str, *args: Any, **kwargs: Any) -> Any:
        nonlocal target_reader_calls
        if "/targets/" in Path(path).as_posix():
            target_reader_calls += 1
            raise AssertionError("source target reader reached during preflight")
        return original_reader(path, *args, **kwargs)

    with mock.patch.object(adapter.pd, "read_parquet", side_effect=guarded_reader):
        plan = adapter.source_crossfit_plan(root)
        fold = plan[0].fold
        entity_uid = plan[0].train_entities[0]
        surface = adapter.load(root, entity_uid)
        anchor = adapter.sequence_anchor(root, fold, surface)
        try:
            adapter.load_source_entity_targets(
                root,
                surface,
                entity_uid=entity_uid,
                fold=fold,
                expected_sha256="0" * 64,
                access=None,
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("target loader accepted an absent consumed capability")
    if target_reader_calls:
        raise AssertionError("target-bearing Parquet was opened during preflight")
    return {
        "anchor_rows": int(anchor.shape[0]),
        "crossfit_cells": len(plan),
        "source_target_reader_calls": target_reader_calls,
        "support_count": len(surface.support_ids),
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draft-commitment", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only or PRODUCTION_READY:
        raise PermissionError("production K32 source-gate execution is not authorized")
    root = args.root.resolve()
    draft = verify_draft_commitment(root, args.draft_commitment)
    call_graph = preflight_science_call_graph(root)
    receipt = {
        "authorization_consumed": False,
        "contract": PREFLIGHT_CONTRACT,
        "draft_commitment_sha256": sha256(args.draft_commitment),
        "formal_or_outer_access": False,
        "production_ready": False,
        "source_entity_count": draft["source_entity_count"],
        "source_target_values_read": False,
        **call_graph,
    }
    write_json_new(args.preflight_output, receipt)
    print("METRIC source_target_values_read=0")
    print("METRIC outer_or_formal_metrics_opened=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
