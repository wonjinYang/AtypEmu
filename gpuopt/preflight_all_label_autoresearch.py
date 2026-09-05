#!/usr/bin/env python3
"""No-science preflight for the corrected all-label autoresearch workspace."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CANONICAL_INVALIDATION_COMMIT = "ad099b8131a0e25fefe5b3f0fd85937d55c0288a"
RETAINED_CANDIDATE_COMMIT = "f68379d977b113fa13b1b2e05eb220f5d5ee16f9"
QUARANTINE_COMMIT = "95a8906efe7cfb7db5cc80666c2287004cccf284"
ACCIDENTAL_LOG_COMMIT = "badb416c5e0b151022d313354a1b7aa4cbeb0e45"
LOG_SUPERSESSION_BLOB = "d2c54145e4e868fa945532b2e8d6cee467d916cd"
EXPECTED_PYTHON = Path("/home/yang07/anaconda3/envs/atypemu/bin/python")
REQUIRED_MODULES = ("numpy", "pandas", "pyarrow", "scipy", "torch")
CANDIDATE_PATHS = (
    "gpuopt/run_all_label_candidate.py",
    "gpuopt/candidates/torsion_assimilator.py",
)
REPORT_ROOT = "reports/experiments/job134710_k8_dynamic_coordinate_source_gate"
ARCHIVE_RECORDS = {
    "authorization.json": "recovery_authorization.json",
    "consumed_authorization.json": "consumed_authorization.json",
    "decision.json": "decision.json",
    "failure_receipt.json": "initial_failure_receipt.json",
    "result.json": "result.json",
    "review_receipt.json": "recovery_review_receipt.json",
    "source_commitment.json": "source_commitment.json",
}


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=root, text=True, capture_output=True, check=False
    )


def git_show(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"missing immutable Git evidence: {commit}:{relative}")
    return result.stdout


def eligibility_precedes_fit(source: str) -> bool:
    """Conservative structural smoke only; this is not a dataflow proof."""
    required = {
        "restrict_source_subset_eligibility",
        "crossfit_anchor_selection",
        "normalization",
        "calibrate_reference_bounds",
        "train_observer",
        "optimize_assimilation",
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for function in (
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        calls: list[tuple[int, int, str]] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in required:
                calls.append((node.lineno, node.col_offset, name))
        first = {name: min(call[:2] for call in calls if call[2] == name) for name in required if any(call[2] == name for call in calls)}
        if set(first) == required and all(
            first["restrict_source_subset_eligibility"] < first[name]
            for name in required - {"restrict_source_subset_eligibility"}
        ):
            return True
    return False


def resolve_archive_member(archive: Path, relative: str) -> Path:
    archive_root = archive.resolve()
    path = (archive / relative).resolve()
    if archive_root not in path.parents:
        raise ValueError(f"archive checksum path escapes root: {relative}")
    return path


def verify_archive(root: Path) -> list[str]:
    archive = root / ".auto/consolidation/job134710_20260903"
    sums = archive / "SHA256SUMS"
    if not sums.is_file():
        return ["Job 134710 local SHA256SUMS archive is missing"]
    errors = []
    for line in sums.read_text().splitlines():
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append("malformed archive checksum line")
            continue
        try:
            path = resolve_archive_member(archive, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if not path.is_file():
            errors.append(f"archive file missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"archive hash mismatch: {relative}")
    try:
        handoff = json.loads(
            git_show(root, QUARANTINE_COMMIT, f"{REPORT_ROOT}/handoff_receipt.json")
        )
        for name, expected in handoff["verified_output_sha256"].items():
            path = archive / "remote_results" / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"archive differs from immutable output receipt: {name}")
        for remote_name, tracked_name in ARCHIVE_RECORDS.items():
            path = archive / "remote_results" / remote_name
            expected = git_show(
                root, QUARANTINE_COMMIT, f"{REPORT_ROOT}/{tracked_name}"
            )
            if not path.is_file() or path.read_bytes() != expected:
                errors.append(
                    f"archive record differs from quarantine commit: {remote_name}"
                )
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return errors


def collect_errors(
    root: Path,
    *,
    expect_retained_baseline: bool,
    corrected_runner: Path | None = None,
) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    config = json.loads((root / ".auto/config.json").read_text())
    configured = (root / config.get("workingDir", ".")).resolve()
    if configured != root:
        errors.append(f"autoresearch workingDir resolves outside checkout: {configured}")
    if config.get("maxIterations") != 100:
        errors.append("autoresearch maxIterations must remain 100")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        errors.append(f"wrong Python: {sys.executable}")
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            errors.append(f"missing Python module: {module}")
    if shutil.which("ruff") is None:
        errors.append("ruff is not on PATH")
    for relative in (".auto/measure.sh", ".auto/checks.sh"):
        path = root / relative
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"missing executable harness file: {relative}")

    audit = json.loads(
        git_show(
            root,
            CANONICAL_INVALIDATION_COMMIT,
            f"{REPORT_ROOT}/post_handoff_audit.json",
        )
    )
    if audit.get("verdict") != "INVALIDATED" or audit.get("effect", {}).get(
        "source_gate_evidence_valid"
    ) is not False:
        errors.append("Job 134710 canonical invalidation is absent")
    supersession_path = root / REPORT_ROOT / "log_supersession.json"
    supersession_blob = git(root, "hash-object", str(supersession_path))
    if (
        supersession_blob.returncode != 0
        or supersession_blob.stdout.strip() != LOG_SUPERSESSION_BLOB
    ):
        errors.append("Job 134710 log supersession differs from its Git-bound blob")
    supersession = json.loads(supersession_path.read_text())
    if supersession.get("invalidated_log_runs") != [7, 8]:
        errors.append("Job 134710 stale autoresearch log rows are not superseded")
    log_lines = (root / ".auto/log.jsonl").read_bytes().splitlines(keepends=True)
    for line_number, expected in supersession.get(
        "invalidated_log_line_sha256", {}
    ).items():
        index = int(line_number) - 1
        if index >= len(log_lines):
            errors.append(f"superseded autoresearch log line is missing: {line_number}")
        elif hashlib.sha256(log_lines[index]).hexdigest() != expected:
            errors.append(f"superseded autoresearch log line changed: {line_number}")
    ancestor = git(root, "merge-base", "--is-ancestor", CANONICAL_INVALIDATION_COMMIT, "HEAD")
    if ancestor.returncode != 0:
        errors.append("HEAD does not contain the canonical Job 134710 invalidation")
    if expect_retained_baseline:
        candidate_diff = git(
            root, "diff", "--quiet", RETAINED_CANDIDATE_COMMIT, "--", *CANDIDATE_PATHS
        )
        if candidate_diff.returncode != 0:
            errors.append("active candidate differs from retained Run 98 baseline")
        k32 = git(root, "grep", "-n", "TANGENT_QUADRATURE", "--", *CANDIDATE_PATHS)
        if k32.returncode == 0:
            errors.append("quarantined K32 tangent-quadrature code is active")
    for reference, expected in (
        ("quarantine/job134710-invalid-source-gate", QUARANTINE_COMMIT),
        ("quarantine/accidental-log-tool-badb416", ACCIDENTAL_LOG_COMMIT),
    ):
        actual = git(root, "rev-parse", "--verify", reference)
        if actual.returncode != 0 or actual.stdout.strip() != expected:
            errors.append(f"quarantine ref mismatch: {reference}")
    if corrected_runner is not None:
        runner = corrected_runner if corrected_runner.is_absolute() else root / corrected_runner
        if not runner.is_file():
            errors.append(f"corrected K8 runner is missing: {runner}")
        elif not eligibility_precedes_fit(runner.read_text()):
            errors.append(
                "corrected K8 runner lacks a direct eligibility-before-fit structural trace"
            )
    errors.extend(verify_archive(root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expect-retained-baseline", action="store_true")
    parser.add_argument("--corrected-runner", type=Path)
    args = parser.parse_args()
    errors = collect_errors(
        args.root,
        expect_retained_baseline=args.expect_retained_baseline,
        corrected_runner=args.corrected_runner,
    )
    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("PASS setup-only all-label preflight; NOT AUTHORIZATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
