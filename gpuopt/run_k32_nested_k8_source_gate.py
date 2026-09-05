#!/usr/bin/env python3
"""Preflight and eventually execute the matched K32 source-only gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DRAFT_CONTRACT = "k32_nested_k8_source_gate_draft_commitment_v1"
FINAL_CONTRACT = "k32_nested_k8_source_gate_source_commitment_v1"
PREFLIGHT_CONTRACT = "k32_nested_k8_source_gate_call_graph_preflight_v1"
EXECUTION_CONTRACT = "k32_nested_k8_source_gate_execution_v1"
PRODUCTION_READY = True


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


def verify_source_commitment(root: Path, path: Path) -> dict[str, Any]:
    commitment = json.loads(path.read_text())
    required = {
        "authorization_allowed": True,
        "contract": FINAL_CONTRACT,
        "draft_only": False,
        "formal_or_outer_access": False,
        "observer_epochs": 1024,
        "assimilation_steps": 100,
        "source_entity_count": 135,
        "target_file_count": 135,
        "cache_file_count": 135,
        "target_values_opened": False,
    }
    if any(commitment.get(key) != value for key, value in required.items()):
        raise ValueError("final source commitment metadata mismatch")
    files = commitment.get("files")
    if not isinstance(files, dict):
        raise ValueError("final source commitment has no file manifest")
    root = root.resolve()
    for relative, expected in files.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents or len(str(expected)) != 64:
            raise ValueError("final source commitment path or hash is malformed")
        if sha256(candidate) != expected:
            raise ValueError(f"final source commitment hash mismatch: {relative}")
    return commitment


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


def target_hashes_from_commitment(
    commitment: dict[str, Any], entities: set[str]
) -> dict[str, str]:
    output = {}
    for entity_uid in entities:
        relative = (
            "data/all_atom_observer_v1/targets/"
            f"bmr{entity_uid.split(':')[1]}.parquet"
        )
        if relative not in commitment["files"]:
            raise ValueError(f"source commitment omits target: {entity_uid}")
        output[entity_uid] = str(commitment["files"][relative])
    return output


def execute_source_cells(
    root: Path,
    commitment: dict[str, Any],
    consumed: Any,
    output_dir: Path,
    *,
    epochs: int,
    steps: int,
    device_name: str,
    adapter_module: Any | None = None,
) -> dict[str, Any]:
    """Execute exactly four source cells after authorization; do not score."""
    if output_dir.exists():
        raise FileExistsError("source-gate output directory already exists")
    if (epochs, steps) != (1024, 100):
        raise ValueError("source-gate hyperparameters differ from frozen plan")
    if adapter_module is None:
        from gpuopt.candidates import k32_nested_k8_adapter as adapter_module
    import torch

    inventory = json.loads((root / ".auto/frozen/all_label_inventory.json").read_text())
    atom_ids = tuple(str(value) for value in inventory["eligible_atom_ids"])
    if len(atom_ids) != 62 or len(set(atom_ids)) != 62:
        raise ValueError("source-gate Atom_ID inventory mismatch")
    plan = adapter_module.source_crossfit_plan(root)
    expected_cells = {(fold, half) for fold in ("A", "B") for half in (0, 1)}
    if len(plan) != 4 or {(cell.fold, cell.held_half) for cell in plan} != expected_cells:
        raise ValueError("source-gate crossfit plan mismatch")
    entities = set().union(
        *(set(cell.train_entities) | set(cell.evaluation_entities) for cell in plan)
    )
    if len(entities) != 135:
        raise ValueError("source-gate plan does not cover 135 entities")
    target_sha256 = target_hashes_from_commitment(commitment, entities)
    access = adapter_module.ConsumedAuthorization(
        consumed.source_commitment_sha256, consumed.authorization_sha256
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir()
    cell_receipts = {}
    device = torch.device(device_name)
    for cell in plan:
        common = {
            "fold": cell.fold,
            "held_half": cell.held_half,
            "frozen_atom_ids": atom_ids,
            "access": access,
        }
        train = adapter_module.assemble_source_half(
            root,
            cell.train_entities,
            role="train",
            target_sha256={entity: target_sha256[entity] for entity in cell.train_entities},
            **common,
        )
        evaluation = adapter_module.assemble_source_half(
            root,
            cell.evaluation_entities,
            role="evaluation",
            target_sha256={
                entity: target_sha256[entity] for entity in cell.evaluation_entities
            },
            normalization=train.targets.normalization,
            **common,
        )
        result = adapter_module.run_source_crossfit_cell(
            train,
            evaluation,
            atom_inventory=atom_ids,
            epochs=epochs,
            steps=steps,
            seed=20260905 + (0 if cell.fold == "A" else 2) + cell.held_half,
            device=device,
        )
        physicality = adapter_module.audit_source_cell_coordinates(root, result)
        cell_path = cells_dir / f"cell_{cell.fold}_{cell.held_half}"
        cell_receipts[f"{cell.fold}_{cell.held_half}"] = (
            adapter_module.write_source_cell_outputs_new(
                cell_path, result, physicality
            )
        )
    execution = {
        "authorization_sha256": consumed.authorization_sha256,
        "cell_receipt_sha256": {
            identity: sha256(cells_dir / f"cell_{identity}" / "receipt.json")
            for identity in sorted(cell_receipts)
        },
        "contract": EXECUTION_CONTRACT,
        "formal_or_outer_metrics_opened": False,
        "source_commitment_sha256": consumed.source_commitment_sha256,
        "source_entity_count": len(entities),
        "source_target_values_opened": True,
    }
    write_json_new(output_dir / "execution_receipt.json", execution)
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draft-commitment", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--source-commitment", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--consumed", type=Path)
    parser.add_argument("--external-claim", type=Path)
    parser.add_argument("--authorization-git-dir", type=Path)
    parser.add_argument("--authorization-ref")
    parser.add_argument("--authorization-git-blob")
    parser.add_argument("--container-image-sha256")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--epochs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.preflight_only:
        if args.draft_commitment is None or args.preflight_output is None:
            raise ValueError("preflight paths are required")
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
    production_arguments = (
        args.source_commitment,
        args.authorization,
        args.consumed,
        args.external_claim,
        args.authorization_git_dir,
        args.authorization_ref,
        args.authorization_git_blob,
        args.container_image_sha256,
        args.slurm_job_id,
        args.output_dir,
    )
    if not PRODUCTION_READY or any(value is None for value in production_arguments):
        raise PermissionError("production K32 source-gate arguments are incomplete")
    commitment = verify_source_commitment(root, args.source_commitment)
    preflight_science_call_graph(root)
    if args.output_dir.exists():
        raise FileExistsError("source-gate output directory already exists")
    from gpuopt.k32_source_gate_authorization import (
        AuthorizationSpec,
        consume_authorization,
    )

    consumed = consume_authorization(
        args.authorization,
        args.source_commitment,
        args.consumed,
        args.external_claim,
        spec=AuthorizationSpec(
            sha256(args.source_commitment),
            args.epochs,
            args.steps,
            args.container_image_sha256,
            args.slurm_job_id,
            args.authorization_ref,
            args.authorization_git_blob,
        ),
        authorization_git_dir=args.authorization_git_dir,
    )
    execute_source_cells(
        root,
        commitment,
        consumed,
        args.output_dir,
        epochs=args.epochs,
        steps=args.steps,
        device_name=args.device,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "gpuopt/check_k32_nested_k8_source_gate.py"),
            "--root",
            str(root),
            "--run-dir",
            str(args.output_dir),
            "--output",
            str(args.output_dir / "decision.json"),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
