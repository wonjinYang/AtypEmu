#!/usr/bin/env python3
"""Verify the receipt-bound backpressure recovery for corrected-K8 Job 135711."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCIENCE_JOB_ID = "135711"
CHECK_JOB_ID = "135887"
SOURCE_SHA = "9f6ed71e8a213c7f3cb95ba61687383a57369928cff2766de4ab0174363f168a"
SOURCE_REF = "refs/atypemu-source-commitments/corrected-k8/final-v7"
SOURCE_BLOB = "0ca9e9378b24feef3da67f96a4f3b8f6179e51f6"
RESULT_SHA = "a0750092171e837c3f881c33ffe64a1247b427e03d813994dd6753645f4cc206"
DECISION_SHA = "5e111f11a59fd1b8be78182d60ce6336726b103607f486d0496bb8a05cb1c6f6"
AUTH_SHA = "693b81203f9fe8737a848b9850340aad35ef0113ee4eba81ef6a10546cc4a29c"
AUTH_BLOB = "8c25f439f30c548ebdb10eefd1a1dae996a77d09"
AUTH_REF = "refs/atypemu-authorizations/corrected-k8/job-135711"
CLAIM_BLOB = "5c9aacf9f87d504176cb546bd14199fdc2d95004"
CHECK_SOURCE_SHA = "56ae0071f02f3f0225f3d5b6860a0e06d289aa16d456c317b300dde3ae799882"
CHECK_SOURCE_BLOB = "3cd7755c8197e2b191d893a2c9d961d09f796909"
CHECK_SOURCE_REF = (
    "refs/atypemu-check-recovery-sources/corrected-k8/job135711/final-v3"
)
RECEIPT_SHA = "4113acb58e7bbaca06862fe098105ac60784d6989f8e9b4a8a4c5cac27be827c"
RECEIPT_BLOB = "ba495c32255f1728f03e0e938c01e5a3d870db73"
RECEIPT_REF = (
    "refs/atypemu-check-recovery-decisions/corrected-k8/job135711/final-v1"
)
IMAGE_SHA = "67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531"

ARTIFACT_HASHES = {
    "checks.err": "ce72ddf5b97b3ac2d63137d23e491e9da5a6e444aeb0afa29c8d9c7ecbc91977",
    "checks.out": "14661e02543ad246b6e8b159fbfa1127d722e3bab6671e4841357b01f1932542",
    "checks.trace": "4ba08ae6d7a3529327f315eda14397dba179c7e895366986878a97202c1704b3",
    "executables.sha256": "088796d8ca9d596092127332927d1b2c97bc175dadace934eddfedf2c7f38b94",
    "input.after.controller.sha256": (
        "776cf38a0c132a04ce724b1b7c8c51f46a982e4c472fa8c5d8ae26bb4b9de472"
    ),
    "input.before.sha256": (
        "776cf38a0c132a04ce724b1b7c8c51f46a982e4c472fa8c5d8ae26bb4b9de472"
    ),
    "receipt.json": RECEIPT_SHA,
}
EXECUTABLE_HASHES = {
    "./.auto/checks.sh": "c8820483d3ba10ecbb6d5d08cdd8c119c229f491057e872c67d22c8d0a448ddd",
    "./gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py": (
        "32164f241edf99c15359ca23984628d90d582bb70dc0f62c1208a5c03908c7a5"
    ),
    "./gpuopt/test_all_label_one_shared_q_metric.py": (
        "4f72c8e2f615c6df5ace595998246c82519d85b4dab92d6a639a37a6f1081739"
    ),
    "./gpuopt/tests/test_corrected_k8_dynamic_coordinate_source_gate.py": (
        "7472e0ce39d09dad0b5decd369187227b4219e4f208dc84d48ebef8d150434ef"
    ),
    "./ruff": "29d26e3878b2a9c9c96a74b522e42b7037e958903bd33652a5315034f53fde5a",
}
INPUT_HASHES = {
    "./authorization.json": AUTH_SHA,
    "./check_mode.json": "2c7e5d432059ad1b1a2ae080a6c3be0f7fec763938c5ba1f06e2c4e9fe960e94",
    "./consumed_authorization.json": (
        "d2c97e218063063f9508ed8e546b776387172f69a44374dc45a86d9ff373978a"
    ),
    "./corrected_k8_source_commitment.json": SOURCE_SHA,
    "./decision.json": DECISION_SHA,
    "./external_claim.json": "1273b6ae18729d2a8afe3eeb11372784221289bb0885a2d31365d247d3334373",
    "./source_coordinate_A_half0.npz": (
        "c32581aaf05ff6ade1adc49b78e2df8d4908b965346638a33f80d2385b57c8f4"
    ),
    "./source_coordinate_A_half1.npz": (
        "33921b1c71a6b80bc059f32773c72ebb9100f915d1010d82a461a149ef1bf05f"
    ),
    "./source_coordinate_B_half0.npz": (
        "8e97d2499cc628cc5defa6027b01470f1082160f59bda9948d4b9c546bd1c490"
    ),
    "./source_coordinate_B_half1.npz": (
        "2446b7adef882a27e7e70bde6fe1f4fee89e2a33554ca99eafc4d448e942ee65"
    ),
    "./source_oof_A.parquet": (
        "3b9ef71d05508e7fcd7b6b4f54b4637a01d35fb68ec33a729b5cbeea9c62fb69"
    ),
    "./source_oof_B.parquet": (
        "a562cb53d160ef869e0bd93c94f0aea2d9eccade4f973341407e99afbf0522c6"
    ),
    "./unchecked_result.json": RESULT_SHA,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def parse_manifest(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        if name in parsed or len(digest) != 64:
            raise ValueError(f"invalid manifest: {path}")
        parsed[name] = digest
    return parsed


def git_blob(git_dir: Path, ref: str) -> tuple[str, bytes]:
    blob = subprocess.check_output(
        ["git", f"--git-dir={git_dir}", "rev-parse", f"{ref}^{{blob}}"],
        text=True,
    ).strip()
    content = subprocess.check_output(
        ["git", f"--git-dir={git_dir}", "cat-file", "blob", ref]
    )
    return blob, content


def require_ordered(text: str, fragments: list[str]) -> None:
    cursor = -1
    for fragment in fragments:
        cursor = text.find(fragment, cursor + 1)
        if cursor < 0:
            raise ValueError(f"missing ordered trace fragment: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization-git-dir", type=Path, required=True)
    parser.add_argument("--emit-metrics", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    recovery = run_dir / "backpressure_recovery"

    if set(path.name for path in recovery.iterdir()) != set(ARTIFACT_HASHES):
        raise ValueError("unexpected backpressure-recovery artifact inventory")
    for name, expected in ARTIFACT_HASHES.items():
        if sha256(recovery / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")

    receipt = load_json(recovery / "receipt.json")
    if receipt.get("contract") != (
        "corrected_k8_job135711_backpressure_recovery_receipt_v1"
    ):
        raise ValueError("receipt contract mismatch")
    if receipt.get("backpressure_checks_passed") is not True:
        raise ValueError("backpressure receipt is not PASS")
    if receipt.get("original_science_job_id") != SCIENCE_JOB_ID:
        raise ValueError("receipt science-job mismatch")
    if receipt.get("source_science_rerun") is not False:
        raise ValueError("receipt claims a science rerun")
    if receipt.get("original_results_mutated") is not False:
        raise ValueError("receipt does not preserve original results")
    if receipt.get("formal_metrics_opened") is not False:
        raise ValueError("receipt opened formal metrics")
    if receipt.get("stage_current_restored") is not True:
        raise ValueError("receipt did not restore the staged current run")
    if receipt.get("source_commitment_sha256") != SOURCE_SHA:
        raise ValueError("receipt source mismatch")
    if receipt.get("result_sha256") != RESULT_SHA:
        raise ValueError("receipt result mismatch")
    if receipt.get("decision_sha256") != DECISION_SHA:
        raise ValueError("receipt decision mismatch")
    if receipt.get("container_image_sha256") != IMAGE_SHA:
        raise ValueError("receipt image mismatch")
    if receipt.get("authorization") != {
        "authorization_git_blob": AUTH_BLOB,
        "authorization_sha256": AUTH_SHA,
        "external_claim_git_blob": CLAIM_BLOB,
        "external_claim_ref": AUTH_REF,
        "external_claim_sha256": INPUT_HASHES["./external_claim.json"],
    }:
        raise ValueError("receipt authorization mismatch")
    if receipt.get("check_recovery_source") != {
        "git_blob": CHECK_SOURCE_BLOB,
        "ref": CHECK_SOURCE_REF,
        "sha256": CHECK_SOURCE_SHA,
    }:
        raise ValueError("receipt check-source mismatch")
    execution = receipt.get("check_execution", {})
    if execution != {
        "end_utc": "2026-09-05T10:56:20",
        "exit_code": "1:0",
        "job_id": CHECK_JOB_ID,
        "node": "iREMB-C-08",
        "start_utc": "2026-09-05T10:55:59",
        "state": "FAILED",
    }:
        raise ValueError("check-execution identity mismatch")
    wrapper_failure = receipt.get("wrapper_failure", {})
    if wrapper_failure.get("after_exact_checks_exit_zero") is not True:
        raise ValueError("wrapper failure did not follow exact check success")
    if wrapper_failure.get("scientific_failure") is not False:
        raise ValueError("wrapper failure classified as scientific")
    checks = receipt.get("checks", {})
    expected_checks = {
        "checker_invoked": True,
        "checker_replay_matched_decision": True,
        "executables_manifest_sha256": ARTIFACT_HASHES["executables.sha256"],
        "input_after_sha256": ARTIFACT_HASHES["input.after.controller.sha256"],
        "input_before_sha256": ARTIFACT_HASHES["input.before.sha256"],
        "input_unchanged": True,
        "metric_test_passed": True,
        "ruff_passed": True,
        "stderr_sha256": ARTIFACT_HASHES["checks.err"],
        "stdout_sha256": ARTIFACT_HASHES["checks.out"],
        "trace_sha256": ARTIFACT_HASHES["checks.trace"],
        "unit_test_count": 23,
        "unit_tests_passed": True,
    }
    if checks != expected_checks:
        raise ValueError("receipt check evidence mismatch")

    out = (recovery / "checks.out").read_text(encoding="utf-8")
    if out != (
        "all-label macro one-shared-q metric checks: PASS\n"
        "All checks passed!\n"
        "corrected K8 source-gate backpressure checks: PASS\n"
    ):
        raise ValueError("unexpected check stdout")
    err = (recovery / "checks.err").read_text(encoding="utf-8")
    if re.fullmatch(
        r"\.{23}\n-{70}\nRan 23 tests in [0-9]+\.[0-9]{3}s\n\nOK\n",
        err,
    ) is None:
        raise ValueError("unexpected unittest stderr")

    trace = (recovery / "checks.trace").read_text(encoding="utf-8")
    require_ordered(
        trace,
        [
            "gpuopt/test_all_label_one_shared_q_metric.py",
            "gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py",
            "--slurm-job-id 135711",
            "cmp --silent ",
            "ruff check ",
            "-m unittest gpuopt.tests.test_corrected_k8_dynamic_coordinate_source_gate",
            "echo 'corrected K8 source-gate backpressure checks: PASS'",
            "+ exit 0",
            "rm -f /workspace/.auto/runs/current/decision.replay.",
        ],
    )
    if parse_manifest(recovery / "executables.sha256") != EXECUTABLE_HASHES:
        raise ValueError("executed-source manifest mismatch")
    before = parse_manifest(recovery / "input.before.sha256")
    after = parse_manifest(recovery / "input.after.controller.sha256")
    if before != INPUT_HASHES or after != INPUT_HASHES:
        raise ValueError("check input changed or has unexpected identity")

    live_files = {name.removeprefix("./"): digest for name, digest in INPUT_HASHES.items()}
    live_files.pop("check_mode.json")
    for name, expected in live_files.items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f"live result identity mismatch: {name}")

    decision = load_json(run_dir / "decision.json")
    if decision.get("contract") != "corrected_k8_dynamic_coordinate_source_oof_check_v1":
        raise ValueError("decision contract mismatch")
    if decision.get("slurm_job_id") != SCIENCE_JOB_ID:
        raise ValueError("decision science-job mismatch")
    if decision.get("selected_for_k32_followup") is not True:
        raise ValueError("decision did not select K32 follow-up")
    if decision.get("minimum_gain_each_fold") != 0.0005:
        raise ValueError("decision threshold mismatch")
    if decision.get("folds", {}).get("A", {}).get("gain") != 0.005707835928873806:
        raise ValueError("A-fold gain mismatch")
    if decision.get("folds", {}).get("B", {}).get("gain") != 0.005949596311624394:
        raise ValueError("B-fold gain mismatch")
    if decision.get("combined", {}).get("gain") != 0.00838304634138265:
        raise ValueError("combined gain mismatch")

    result = load_json(run_dir / "unchecked_result.json")
    if result.get("ccc_used_in_loss") is not False:
        raise ValueError("CCC entered the source loss")
    if result.get("slurm_job_id") != SCIENCE_JOB_ID:
        raise ValueError("result science-job mismatch")
    if result.get("source_commitment_sha256") != SOURCE_SHA:
        raise ValueError("result source mismatch")

    refs = {
        SOURCE_REF: (SOURCE_BLOB, SOURCE_SHA),
        CHECK_SOURCE_REF: (CHECK_SOURCE_BLOB, CHECK_SOURCE_SHA),
        RECEIPT_REF: (RECEIPT_BLOB, RECEIPT_SHA),
        AUTH_REF: (CLAIM_BLOB, INPUT_HASHES["./external_claim.json"]),
    }
    for ref, (expected_blob, expected_sha) in refs.items():
        blob, content = git_blob(args.authorization_git_dir, ref)
        if blob != expected_blob or hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError(f"external ref mismatch: {ref}")
    auth_content = subprocess.check_output(
        ["git", f"--git-dir={args.authorization_git_dir}", "cat-file", "blob", AUTH_BLOB]
    )
    if hashlib.sha256(auth_content).hexdigest() != AUTH_SHA:
        raise ValueError("authorization blob mismatch")
    _, claim_content = git_blob(args.authorization_git_dir, AUTH_REF)
    if (run_dir / "external_claim.json").read_bytes() != claim_content:
        raise ValueError("live external claim differs from external ref")
    claim = json.loads(claim_content)
    if claim != {
        "authorization_git_blob": AUTH_BLOB,
        "authorization_ref": AUTH_REF,
        "authorization_sha256": AUTH_SHA,
        "container_image_sha256": IMAGE_SHA,
        "contract": "corrected_k8_dynamic_coordinate_source_gate_external_claim_v1",
        "slurm_job_id": SCIENCE_JOB_ID,
        "source_commitment_sha256": SOURCE_SHA,
    }:
        raise ValueError("external claim does not bind the authorization")
    if (run_dir / "authorization.json").read_bytes() != auth_content:
        raise ValueError("live authorization differs from claimed Git blob")
    authorization = json.loads(auth_content)
    if authorization != {
        "authorization_ref": AUTH_REF,
        "authorized": True,
        "container_image_sha256": IMAGE_SHA,
        "contract": "corrected_k8_dynamic_coordinate_source_gate_authorization_v1",
        "epochs": 1024,
        "slurm_job_id": SCIENCE_JOB_ID,
        "source_commitment_sha256": SOURCE_SHA,
        "steps": 100,
    }:
        raise ValueError("authorization payload mismatch")

    print("corrected K8 Job 135711 backpressure recovery: PASS")
    if args.emit_metrics:
        print("METRIC corrected_k8_worst_fold_margin=0.005207835928873806")
        print("METRIC corrected_k8_min_direction_gain=0.005707835928873806")
        print("METRIC corrected_k8_mean_direction_gain=0.0058287161202491")
        print("METRIC corrected_k8_combined_gain=0.00838304634138265")
        print("METRIC corrected_k8_A_gain=0.005707835928873806")
        print("METRIC corrected_k8_B_gain=0.005949596311624394")
        print("METRIC source_science_rerun=0")
        print("METRIC outer_or_formal_metrics_opened=0")


if __name__ == "__main__":
    main()
