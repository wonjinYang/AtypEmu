#!/usr/bin/env python3
"""Time two exact train-only Phase-D epochs from a frozen resume state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch


def load_trainer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase_d_benchmark_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Phase-D trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps(
            ["tensor", str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value, key=lambda item: str(item)):
            hash_value(digest, key)
            hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            hash_value(digest, item)
        digest.update(b"]")
    else:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\n")


def object_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    hash_value(digest, value)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trainer", required=True, type=Path)
    result.add_argument("--train-manifest", required=True, type=Path)
    result.add_argument("--q-checkpoint", required=True, type=Path)
    result.add_argument("--q-transfer-receipt", required=True, type=Path)
    result.add_argument("--fixed-support-source", required=True, type=Path)
    result.add_argument("--stage-a-source", required=True, type=Path)
    result.add_argument("--stage-a-checkpoint", required=True, type=Path)
    result.add_argument("--resume-state", required=True, type=Path)
    result.add_argument("--result", required=True, type=Path)
    result.add_argument("--timing-ready", required=True, type=Path)
    result.add_argument("--timing-done", required=True, type=Path)
    result.add_argument("--start-signal", type=Path)
    result.add_argument("--epochs", default=2, choices=(2,), type=int)
    result.add_argument("--cohort-size", default=32, choices=(32,), type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    trainer = load_trainer(args.trainer)
    device = torch.device("cuda")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("benchmark requires exactly remapped CUDA ordinal 0")

    trainer.V6.configure_deterministic_runtime(device)
    trainer.V6.seed_runtime(3339, device)
    all_rows = trainer.read_manifest(args.train_manifest, "inner_train", 1, 8)
    # Keep the local 8-GiB test bounded while preserving a fixed mix of 32
    # production entities. Candidate code cannot influence cohort selection.
    rows = sorted(
        all_rows,
        key=lambda row: (Path(row["batch_path"]).stat().st_size, row["entity_uid"]),
    )[: args.cohort_size]
    stage_a, _ = trainer.V6.load_frozen_stage_a(
        args.stage_a_source,
        trainer.STAGE_A_SOURCE_SHA256,
        args.stage_a_checkpoint,
        trainer.EXPECTED_STAGE_A_SHA256[1],
        outer_fold=1,
        device=device,
    )
    model, _, _ = trainer.initialize_model(
        args.q_checkpoint,
        args.q_transfer_receipt,
        args.fixed_support_source,
        route="legacy5",
        fold=1,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    resume = torch.load(args.resume_state, map_location=device, weights_only=True)
    if (
        resume.get("artifact_kind") != "v3339_phase_d_atom27_epoch_resume_v1"
        or resume.get("binding", {}).get("route") != "legacy5"
        or int(resume.get("binding", {}).get("outer_fold", -1)) != 1
    ):
        raise RuntimeError("benchmark resume identity drift")
    start_epoch = int(resume["next_epoch"])
    model.load_state_dict(resume["model_state_dict"], strict=True)
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    del resume

    torch.cuda.synchronize()
    args.timing_ready.touch()
    if args.start_signal is not None:
        for _ in range(360_000):
            if args.start_signal.exists():
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("benchmark start signal was not issued")
    started = time.perf_counter()
    prepared = False
    work = rows
    if hasattr(trainer, "prepare_entities"):
        work = trainer.prepare_entities(rows, stage_a, route="legacy5", device=device)
        prepared = True
    history = []
    for epoch in range(start_epoch, start_epoch + args.epochs):
        trainer.V6.seed_runtime(3339 + epoch, device)
        history.append(trainer.one_epoch(
            model,
            stage_a,
            work,
            route="legacy5",
            device=device,
            optimizer=optimizer,
            seed=3339 + epoch,
        ))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.timing_done.touch()

    payload = {
        "artifact_kind": "v3339_phase_d_gpu_benchmark_v1",
        "train_only": True,
        "outer_held_accessed": False,
        "fold": 1,
        "route": "legacy5",
        "start_epoch": start_epoch,
        "epochs": args.epochs,
        "entity_updates": len(rows) * args.epochs,
        "cohort_size": len(rows),
        "cohort_sha256": object_sha256([row["entity_uid"] for row in rows]),
        "prepared_entity_cache": prepared,
        "elapsed_seconds": elapsed,
        "updates_per_second": len(rows) * args.epochs / elapsed,
        "trainer_sha256": trainer.sha256_file(args.trainer),
        "model_state_sha256": object_sha256(model.state_dict()),
        "optimizer_state_sha256": object_sha256(optimizer.state_dict()),
        "history_sha256": object_sha256(history),
        "history": history,
    }
    args.result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.result.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
