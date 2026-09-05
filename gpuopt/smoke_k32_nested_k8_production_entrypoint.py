#!/usr/bin/env python3
"""Exercise the production CLI through authorization dispatch without science reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import gpuopt.k32_source_gate_authorization as authorization_module
import gpuopt.run_k32_nested_k8_source_gate as runner


class SmokeStop(RuntimeError):
    """Expected stop at the target-reading executor boundary."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_new(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def smoke(root: Path, source_commitment: Path, output: Path) -> dict[str, object]:
    """Run the real parser/main path and stop exactly before source-target loading."""
    if output.exists():
        raise FileExistsError("production-entrypoint smoke output already exists")
    root, source_commitment = root.resolve(), source_commitment.resolve()
    source_sha = sha256(source_commitment)
    scratch = output.resolve().parent
    job = "900000"
    ref = f"refs/atypemu-authorizations/k32-source/job-{job}"
    image_hash = "67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531"
    paths = {
        name: scratch / f"{output.stem}.synthetic_{name}.json"
        for name in ("authorization", "consumed", "external_claim")
    }
    calls = {"authorization": 0, "executor": 0, "source_targets": 0}
    original_reader = pd.read_parquet

    def guarded_reader(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if "/data/all_atom_observer_v1/targets/" in Path(path).resolve().as_posix():
            calls["source_targets"] += 1
            raise AssertionError("production-entrypoint smoke attempted a source-target read")
        return original_reader(path, *args, **kwargs)

    def synthetic_consume(
        *args: object, spec: object, **kwargs: object
    ) -> SimpleNamespace:
        del args, kwargs
        calls["authorization"] += 1
        if getattr(spec, "source_commitment_sha256") != source_sha:
            raise AssertionError("smoke authorization did not receive the final commitment")
        authorization = {
            "authorization_ref": ref,
            "authorized": True,
            "container_image_sha256": image_hash,
            "contract": "k32_nested_k8_source_gate_authorization_v1",
            "epochs": 1024,
            "slurm_job_id": job,
            "source_commitment_sha256": source_sha,
            "steps": 100,
        }
        write_json_new(paths["authorization"], authorization)
        authorization_sha = sha256(paths["authorization"])
        claim = {
            "authorization_git_blob": "a" * 40,
            "authorization_ref": ref,
            "authorization_sha256": authorization_sha,
            "container_image_sha256": image_hash,
            "contract": "k32_nested_k8_source_gate_external_claim_v1",
            "slurm_job_id": job,
            "source_commitment_sha256": source_sha,
        }
        write_json_new(paths["external_claim"], claim)
        write_json_new(
            paths["consumed"],
            {
                **claim,
                "contract": "k32_nested_k8_source_gate_consumed_v1",
                "external_claim_git_blob": "b" * 40,
                "external_claim_sha256": sha256(paths["external_claim"]),
            },
        )
        return SimpleNamespace(
            authorization_sha256=authorization_sha,
            source_commitment_sha256=source_sha,
        )

    def stop_at_executor(*args: object, **kwargs: object) -> None:
        del args
        calls["executor"] += 1
        binding = kwargs["execution_binding"]
        if binding["consumed_authorization_sha256"] != sha256(paths["consumed"]):
            raise AssertionError("executor received the wrong consumed receipt")
        if binding["external_claim_sha256"] != sha256(paths["external_claim"]):
            raise AssertionError("executor received the wrong external claim")
        raise SmokeStop

    argv = [
        "run_k32_nested_k8_source_gate.py",
        "--root", str(root),
        "--source-commitment", str(source_commitment),
        "--authorization", str(paths["authorization"]),
        "--consumed", str(paths["consumed"]),
        "--external-claim", str(paths["external_claim"]),
        "--authorization-git-dir", str(scratch / "synthetic.git"),
        "--authorization-ref", ref,
        "--authorization-git-blob", "a" * 40,
        "--container-image-sha256", image_hash,
        "--slurm-job-id", job,
        "--output-dir", str(scratch / "synthetic_run"),
        "--epochs", "1024",
        "--steps", "100",
        "--device", "cpu",
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(pd, "read_parquet", side_effect=guarded_reader),
        mock.patch.object(
            authorization_module, "consume_authorization", side_effect=synthetic_consume
        ),
        mock.patch.object(runner, "execute_source_cells", side_effect=stop_at_executor),
    ):
        try:
            runner.main()
        except SmokeStop:
            pass
        else:
            raise AssertionError("production-entrypoint smoke did not reach executor boundary")
    if calls != {"authorization": 1, "executor": 1, "source_targets": 0}:
        raise AssertionError(f"unexpected production-entrypoint calls: {calls}")
    receipt: dict[str, object] = {
        "authorization_consumed": False,
        "container_entrypoint_reached": True,
        "contract": "k32_nested_k8_production_entrypoint_smoke_v1",
        "executor_boundary_reached": True,
        "final_source_commitment_sha256": source_sha,
        "formal_or_outer_metrics_opened": False,
        "source_target_values_read": 0,
        "synthetic_authorization_calls": 1,
    }
    write_json_new(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commitment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = smoke(args.root, args.source_commitment, args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
