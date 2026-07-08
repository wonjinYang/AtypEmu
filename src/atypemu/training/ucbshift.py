"""UCBShift2.0 sidecar and baseline generation helpers."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.adapters.cnnls_adapter import (
    load_candidate_shift_file,
    normalize_candidate_key,
)
from atypemu.datasets import IntegratedDataRegistry
from atypemu.training.config import UCBShiftGenerationConfig
from atypemu.training.materialize import resolve_existing_path
from atypemu.types import NMRTargetBundle, canonical_atom_name


@dataclass(slots=True)
class UCBShiftRuntime:
    """Resolved UCBShift2.0 runtime paths."""

    python_executable: Path
    ucbshift_root: Path
    ucbshift_script: Path
    models_dir: Path
    dssp_executable: Path | None
    reduce_executable: Path | None
    model_file_count: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the runtime description."""
        return {
            "python_executable": str(self.python_executable),
            "ucbshift_root": str(self.ucbshift_root),
            "ucbshift_script": str(self.ucbshift_script),
            "models_dir": str(self.models_dir),
            "dssp_executable": (
                None if self.dssp_executable is None else str(self.dssp_executable)
            ),
            "reduce_executable": (
                None if self.reduce_executable is None else str(self.reduce_executable)
            ),
            "model_file_count": self.model_file_count,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class UCBShiftEntityResult:
    """Per-entity UCBShift generation outcome."""

    entity_uid: str
    bmrb_id: str
    status: str
    message: str
    requested_candidates: int = 0
    written_candidates: int = 0
    skipped_existing_candidates: int = 0
    failed_candidates: int = 0
    baseline_rows: int = 0
    baseline_source: str | None = None
    source_candidate_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result payload."""
        return {
            "entity_uid": self.entity_uid,
            "bmrb_id": self.bmrb_id,
            "status": self.status,
            "message": self.message,
            "requested_candidates": self.requested_candidates,
            "written_candidates": self.written_candidates,
            "skipped_existing_candidates": self.skipped_existing_candidates,
            "failed_candidates": self.failed_candidates,
            "baseline_rows": self.baseline_rows,
            "baseline_source": self.baseline_source,
            "source_candidate_counts": dict(self.source_candidate_counts),
        }


@dataclass(frozen=True, slots=True)
class UCBShiftCandidateResult:
    """Per-candidate UCBShift subprocess outcome."""

    source: str
    pdb_path: Path
    output_path: Path
    status: str
    message: str = ""


def generate_ucbshift_sidecars(
    data_root: str | Path,
    integrated_root: str | Path,
    config: UCBShiftGenerationConfig,
) -> dict[str, Any]:
    """Generate per-candidate UCBShift2.0 sidecars and baseline tables.

    Args:
        data_root: Repository ``data`` root.
        integrated_root: Integrated workspace root such as ``data/integrated``.
        config: UCBShift2.0 generation configuration.

    Returns:
        Summary payload describing generated sidecars and baseline tables.
    """
    data_root_path = Path(data_root)
    integrated_root_path = Path(integrated_root)
    repo_root = data_root_path.parent

    runtime = resolve_ucbshift_runtime(
        config=config,
        data_root=data_root_path,
        repo_root=repo_root,
    )

    registry = IntegratedDataRegistry.from_data_root(data_root_path)
    teacher_examples = registry.load_teacher_examples()
    selected = teacher_examples.loc[
        teacher_examples["split"].isin(config.selected_splits)
    ].copy()
    selected_bmrb_ids = {str(value) for value in config.selected_bmrb_ids}
    if selected_bmrb_ids:
        selected = selected.loc[
            selected["bmrb_id"].astype(str).isin(selected_bmrb_ids)
        ].copy()
    if config.shard_count < 1:
        raise ValueError("UCBShift shard_count must be >= 1.")
    if not 0 <= config.shard_index < config.shard_count:
        raise ValueError("UCBShift shard_index must satisfy 0 <= index < count.")
    total_selected_examples = int(len(selected))
    if config.shard_count > 1:
        selected = selected.iloc[config.shard_index :: config.shard_count].reset_index(
            drop=True
        )
    if config.max_examples is not None:
        selected = selected.head(config.max_examples)

    summary_dir = integrated_root_path / "ucbshift"
    summary_dir.mkdir(parents=True, exist_ok=True)
    shard_label = _shard_label(config)
    summary_path = summary_dir / f"sidecar_generation_summary{shard_label}.json"

    runtime_errors: list[str] = []
    if not runtime.python_executable.exists():
        runtime_errors.append(
            f"Missing UCBShift python executable: {runtime.python_executable}"
        )
    if not runtime.ucbshift_script.exists():
        runtime_errors.append(f"Missing UCBShift script: {runtime.ucbshift_script}")
    if runtime.model_file_count < 141:
        runtime_errors.append(
            "UCBShift model directory is incomplete; expected at least 141 .sav files."
        )

    if runtime_errors:
        payload = {
            "status": "failed_runtime",
            "message": " ".join(runtime_errors),
            "total_selected_examples": total_selected_examples,
            "shard_count": config.shard_count,
            "shard_index": config.shard_index,
            "requested_examples": int(len(selected)),
            "requested_candidates": 0,
            "written_candidates": 0,
            "skipped_existing_candidates": 0,
            "failed_candidates": 0,
            "requested_candidate_examples": 0,
            "written_candidate_examples": 0,
            "baseline_examples": 0,
            "dependency_status": {
                "status": "failed_runtime",
                **runtime.as_dict(),
            },
            "results": [],
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return {"summary_path": str(summary_path), **payload}

    baseline_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    results: list[UCBShiftEntityResult] = []

    total_requested_candidates = 0
    total_written_candidates = 0
    total_skipped_existing = 0
    total_failed_candidates = 0

    for row in selected.to_dict(orient="records"):
        try:
            result, entity_baseline_rows = _generate_entity_sidecars(
                row=row,
                runtime=runtime,
                config=config,
                data_root=data_root_path,
                repo_root=repo_root,
            )
        except Exception as exc:
            if config.fail_fast:
                raise
            result = UCBShiftEntityResult(
                entity_uid=str(row["entity_uid"]),
                bmrb_id=str(row["bmrb_id"]),
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            entity_baseline_rows = []
        results.append(result)
        baseline_rows.extend(entity_baseline_rows)
        if entity_baseline_rows:
            reference_rows.append(
                {
                    "entity_uid": result.entity_uid,
                    "bmrb_id": result.bmrb_id,
                    "baseline_source": result.baseline_source,
                }
            )
        total_requested_candidates += result.requested_candidates
        total_written_candidates += result.written_candidates
        total_skipped_existing += result.skipped_existing_candidates
        total_failed_candidates += result.failed_candidates

    requested_candidate_examples = sum(
        int(result.requested_candidates > 0) for result in results
    )
    written_candidate_examples = sum(
        int(result.written_candidates + result.skipped_existing_candidates > 0)
        for result in results
    )
    baseline_examples = sum(int(result.baseline_rows > 0) for result in results)
    failed_example_count = sum(int(result.status == "failed") for result in results)
    partial_example_count = sum(int(result.status == "partial") for result in results)
    baseline_required = bool(
        config.baseline_output_path or config.reference_corpus_output_path
    )

    baseline_output_path = _resolve_output_path(
        config.baseline_output_path,
        data_root=data_root_path,
        repo_root=repo_root,
    )
    reference_output_path = _resolve_output_path(
        config.reference_corpus_output_path,
        data_root=data_root_path,
        repo_root=repo_root,
    )
    if config.shard_count > 1:
        baseline_output_path = _shard_output_path(baseline_output_path, config)
        reference_output_path = _shard_output_path(reference_output_path, config)

    if baseline_output_path is not None:
        baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(baseline_rows).to_csv(baseline_output_path, index=False)
    if reference_output_path is not None:
        reference_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(reference_rows).drop_duplicates().to_csv(
            reference_output_path,
            index=False,
        )

    status = "ok"
    message = "UCBShift sidecar generation completed successfully."
    if len(selected) > 0 and requested_candidate_examples == 0:
        status = "failed_generation"
        message = "No candidate structures were discovered for the requested examples."
    elif requested_candidate_examples > 0 and written_candidate_examples == 0:
        status = "failed_generation"
        message = "No usable UCBShift sidecars were generated for any example."
    elif (
        baseline_required
        and requested_candidate_examples > 0
        and baseline_examples == 0
    ):
        status = "failed_generation"
        message = "Representative UCBShift baseline rows were not generated."
    elif failed_example_count > 0 or partial_example_count > 0:
        status = "partial"
        message = "UCBShift sidecar generation completed with per-example failures."

    payload = {
        "status": status,
        "message": message,
        "total_selected_examples": total_selected_examples,
        "shard_count": config.shard_count,
        "shard_index": config.shard_index,
        "requested_examples": int(len(selected)),
        "requested_candidates": total_requested_candidates,
        "written_candidates": total_written_candidates,
        "skipped_existing_candidates": total_skipped_existing,
        "failed_candidates": total_failed_candidates,
        "requested_candidate_examples": requested_candidate_examples,
        "written_candidate_examples": written_candidate_examples,
        "baseline_examples": baseline_examples,
        "failed_examples": failed_example_count,
        "partial_examples": partial_example_count,
        "baseline_rows": len(baseline_rows),
        "reference_examples": len(reference_rows),
        "worker_count": max(1, config.worker_count),
        "ucbshift_worker_count": max(1, config.ucbshift_worker_count),
        "baseline_output_path": (
            None if baseline_output_path is None else str(baseline_output_path)
        ),
        "reference_corpus_output_path": (
            None if reference_output_path is None else str(reference_output_path)
        ),
        "dependency_status": {"status": "ok", **runtime.as_dict()},
        "results": [result.as_dict() for result in results],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"summary_path": str(summary_path), **payload}


def resolve_ucbshift_runtime(
    config: UCBShiftGenerationConfig,
    data_root: Path,
    repo_root: Path,
) -> UCBShiftRuntime:
    """Resolve one UCBShift2.0 runtime description."""
    ucbshift_root = _resolve_configured_path(
        config.ucbshift_root,
        data_root=data_root,
        repo_root=repo_root,
    )
    python_executable = _resolve_python_executable(
        config.python_executable,
        data_root=data_root,
        repo_root=repo_root,
    )
    raw_models_dir = (
        _resolve_configured_path(
            config.models_dir,
            data_root=data_root,
            repo_root=repo_root,
        )
        if config.models_dir
        else ucbshift_root / "models"
    )
    models_dir, model_file_count = _resolve_models_dir(raw_models_dir)
    dssp_executable = None
    bundled_dssp = ucbshift_root / "bins" / "mkdssp"
    if bundled_dssp.exists():
        dssp_executable = bundled_dssp
    reduce_executable = _which_path("reduce")
    warnings: list[str] = []
    if reduce_executable is None:
        warnings.append(
            "REDUCE is not currently in PATH; UCBShift feature extraction may fail "
            "for non-protonated structures."
        )
    if dssp_executable is None:
        warnings.append("mkdssp was not found under the configured UCBShift root.")
    if model_file_count < 141:
        warnings.append(
            "UCBShift model directory is incomplete; expected 141 .sav files."
        )
    return UCBShiftRuntime(
        python_executable=python_executable,
        ucbshift_root=ucbshift_root,
        ucbshift_script=ucbshift_root / "CSpred.py",
        models_dir=models_dir,
        dssp_executable=dssp_executable,
        reduce_executable=reduce_executable,
        model_file_count=model_file_count,
        warnings=warnings,
    )


def build_ucbshift_command(
    pdb_path: Path,
    output_path: Path,
    runtime: UCBShiftRuntime,
    config: UCBShiftGenerationConfig,
) -> list[str]:
    """Build one UCBShift2.0 command line."""
    pdb_path = Path(pdb_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    command = [
        str(runtime.python_executable),
        str(runtime.ucbshift_script),
        str(pdb_path),
        "--output",
        str(output_path),
        "--worker",
        str(max(1, config.ucbshift_worker_count)),
        "--pH",
        f"{config.ph:g}",
        "--models",
        _format_models_dir_argument(runtime.models_dir),
    ]
    if config.prediction_mode == "shiftx_only":
        command.append("--shiftx_only")
    elif config.prediction_mode == "shifty_only":
        command.append("--shifty_only")
    return command


def _generate_entity_sidecars(
    row: dict[str, Any],
    runtime: UCBShiftRuntime,
    config: UCBShiftGenerationConfig,
    data_root: Path,
    repo_root: Path,
) -> tuple[UCBShiftEntityResult, list[dict[str, Any]]]:
    """Generate UCBShift sidecars for one integrated teacher row."""
    entity_uid = str(row["entity_uid"])
    bmrb_id = str(row["bmrb_id"])
    bundle_path = resolve_existing_path(
        row.get("target_bundle_path"),
        data_root=data_root,
        repo_root=repo_root,
    )
    if bundle_path is None or not bundle_path.exists():
        return (
            UCBShiftEntityResult(
                entity_uid=entity_uid,
                bmrb_id=bmrb_id,
                status="skipped_missing_target",
                message=f"Missing target bundle: {row.get('target_bundle_path')}",
            ),
            [],
        )

    candidate_roots = json.loads(str(row["candidate_structure_roots"]))
    candidate_specs = _discover_candidate_specs(
        candidate_roots=candidate_roots,
        bmrb_id=bmrb_id,
        data_root=data_root,
        repo_root=repo_root,
        config=config,
    )
    if not candidate_specs:
        return (
            UCBShiftEntityResult(
                entity_uid=entity_uid,
                bmrb_id=bmrb_id,
                status="skipped_missing_pool",
                message="No candidate structures were discovered for this accession.",
            ),
            [],
        )

    source_counts: dict[str, int] = {}
    predictions_by_source: dict[str, list[tuple[str, Path]]] = {}
    written = 0
    skipped_existing = 0
    failed = 0

    env = _build_ucbshift_env(runtime)
    failure_messages: list[str] = []
    sequence_text = str(row.get("sequence") or "").strip()
    pending_specs: list[tuple[str, Path, Path]] = []
    for source, pdb_path, output_path in candidate_specs:
        source_counts[source] = source_counts.get(source, 0) + 1
        predictions_by_source.setdefault(source, [])
        if _ucbshift_sidecar_usable(output_path) and not config.refresh_outputs:
            skipped_existing += 1
            predictions_by_source[source].append(
                (normalize_candidate_key(pdb_path), output_path)
            )
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pending_specs.append((source, pdb_path, output_path))

    parallel_workers = min(max(1, config.worker_count), max(len(pending_specs), 1))
    if pending_specs:
        print(
            "[ucbshift] "
            f"{bmrb_id}: pending={len(pending_specs)} "
            f"existing={skipped_existing} "
            f"candidate_workers={parallel_workers} "
            f"ucbshift_worker={max(1, config.ucbshift_worker_count)}",
            flush=True,
        )

    candidate_results = _run_candidate_sidecars(
        pending_specs=pending_specs,
        runtime=runtime,
        config=config,
        env=env,
        sequence_text=sequence_text,
        max_workers=parallel_workers,
    )

    for candidate_result in candidate_results:
        source = candidate_result.source
        pdb_path = candidate_result.pdb_path
        output_path = candidate_result.output_path
        if candidate_result.status == "written":
            written += 1
            predictions_by_source[source].append(
                (normalize_candidate_key(pdb_path), output_path)
            )
            continue
        failed += 1
        if candidate_result.message:
            failure_messages.append(candidate_result.message)
        if config.fail_fast:
            raise RuntimeError(candidate_result.message)

    if pending_specs:
        print(
            "[ucbshift] "
            f"{bmrb_id}: written={written} existing={skipped_existing} "
            f"failed={failed}",
            flush=True,
        )

    baseline_rows, baseline_source = _build_baseline_rows_for_entity(
        entity_uid=entity_uid,
        bmrb_id=bmrb_id,
        bundle_path=bundle_path,
        predictions_by_source=predictions_by_source,
        config=config,
    )
    return (
        _finish_entity_result(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            candidate_specs=candidate_specs,
            source_counts=source_counts,
            written=written,
            skipped_existing=skipped_existing,
            failed=failed,
            failure_messages=failure_messages,
            baseline_rows=baseline_rows,
            baseline_source=baseline_source,
        ),
        baseline_rows,
    )


def _run_candidate_sidecars(
    pending_specs: list[tuple[str, Path, Path]],
    runtime: UCBShiftRuntime,
    config: UCBShiftGenerationConfig,
    env: dict[str, str],
    sequence_text: str,
    max_workers: int,
) -> list[UCBShiftCandidateResult]:
    """Run pending candidate sidecars with bounded subprocess parallelism."""

    if not pending_specs:
        return []
    if max_workers <= 1:
        return [
            _run_one_candidate_sidecar(
                spec=spec,
                runtime=runtime,
                config=config,
                env=env,
                sequence_text=sequence_text,
            )
            for spec in pending_specs
        ]

    results: list[UCBShiftCandidateResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one_candidate_sidecar,
                spec=spec,
                runtime=runtime,
                config=config,
                env=env,
                sequence_text=sequence_text,
            )
            for spec in pending_specs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: (result.source, str(result.pdb_path)))


def _run_one_candidate_sidecar(
    spec: tuple[str, Path, Path],
    runtime: UCBShiftRuntime,
    config: UCBShiftGenerationConfig,
    env: dict[str, str],
    sequence_text: str,
) -> UCBShiftCandidateResult:
    """Run UCBShift for one candidate structure."""

    source, pdb_path, output_path = spec
    if _ucbshift_sidecar_usable(output_path) and not config.refresh_outputs:
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="written",
        )
    tmp_dir = output_path.parent / ".tmp" / output_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_fd, lock_acquired = _acquire_sidecar_lock(
        lock_path=lock_path,
        output_path=output_path,
        refresh_outputs=bool(config.refresh_outputs),
        timeout_seconds=float(config.sidecar_lock_timeout_seconds),
        stale_seconds=float(config.sidecar_stale_lock_seconds),
    )
    if lock_fd is None:
        if _ucbshift_sidecar_usable(output_path) and not config.refresh_outputs:
            return UCBShiftCandidateResult(
                source=source,
                pdb_path=pdb_path,
                output_path=output_path,
                status="written",
            )
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="failed",
            message=f"Timed out waiting for sidecar lock: {lock_path}",
        )
    try:
        if _ucbshift_sidecar_usable(output_path) and not config.refresh_outputs:
            return UCBShiftCandidateResult(
                source=source,
                pdb_path=pdb_path,
                output_path=output_path,
                status="written",
            )
        tmp_output_path = tmp_dir / output_path.name
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        return _run_one_candidate_sidecar_locked(
            spec=spec,
            tmp_output_path=tmp_output_path,
            runtime=runtime,
            config=config,
            env=env,
            sequence_text=sequence_text,
        )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _run_one_candidate_sidecar_locked(
    spec: tuple[str, Path, Path],
    tmp_output_path: Path,
    runtime: UCBShiftRuntime,
    config: UCBShiftGenerationConfig,
    env: dict[str, str],
    sequence_text: str,
) -> UCBShiftCandidateResult:
    source, pdb_path, output_path = spec
    command = build_ucbshift_command(
        pdb_path=pdb_path,
        output_path=tmp_output_path,
        runtime=runtime,
        config=config,
    )
    candidate_env = dict(env)
    tmp_dir = tmp_output_path.parent
    candidate_env["TMPDIR"] = str(tmp_dir)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=runtime.ucbshift_root,
            env=candidate_env,
            timeout=_subprocess_timeout_seconds(config),
        )
    except subprocess.TimeoutExpired as exc:
        partial_stderr = _trim_subprocess_text(exc.stderr)
        partial_stdout = _trim_subprocess_text(exc.stdout)
        details = partial_stderr or partial_stdout
        message = f"UCBShift timed out after {exc.timeout:g}s for {pdb_path}."
        if details:
            message = f"{message} Last output: {details}"
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="failed",
            message=message,
        )
    except OSError as exc:
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="failed",
            message=stderr or f"UCBShift failed with exit code {completed.returncode}",
        )
    if not _ucbshift_sidecar_usable(tmp_output_path):
        return UCBShiftCandidateResult(
            source=source,
            pdb_path=pdb_path,
            output_path=output_path,
            status="failed",
            message=f"UCBShift did not write a non-empty output for {pdb_path}.",
        )
    _realign_ucbshift_sidecar(tmp_output_path, sequence_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path.replace(output_path)
    return UCBShiftCandidateResult(
        source=source,
        pdb_path=pdb_path,
        output_path=output_path,
        status="written",
    )


def _ucbshift_sidecar_usable(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _acquire_sidecar_lock(
    *,
    lock_path: Path,
    output_path: Path,
    refresh_outputs: bool,
    timeout_seconds: float = 7200.0,
    stale_seconds: float = 900.0,
    poll_seconds: float = 1.0,
) -> tuple[int | None, bool]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        if _ucbshift_sidecar_usable(output_path) and not refresh_outputs:
            return None, False
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, _sidecar_lock_payload(output_path).encode())
            return fd, True
        except FileExistsError:
            if _remove_stale_sidecar_lock(lock_path, stale_seconds=stale_seconds):
                continue
            if time.monotonic() - start >= timeout_seconds:
                return None, False
            time.sleep(poll_seconds)


def _subprocess_timeout_seconds(config: UCBShiftGenerationConfig) -> float | None:
    raw_value = config.subprocess_timeout_seconds
    if raw_value is None:
        return None
    value = float(raw_value)
    return value if value > 0 else None


def _sidecar_lock_payload(output_path: Path) -> str:
    return "\n".join(
        [
            f"pid={os.getpid()}",
            f"host={socket.gethostname()}",
            f"started_at={time.time():.6f}",
            f"output={output_path}",
            "",
        ]
    )


def _remove_stale_sidecar_lock(lock_path: Path, *, stale_seconds: float) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if age < stale_seconds:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _trim_subprocess_text(value: str | bytes | None, *, limit: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _build_baseline_rows_for_entity(
    entity_uid: str,
    bmrb_id: str,
    bundle_path: Path,
    predictions_by_source: dict[str, list[tuple[str, Path]]],
    config: UCBShiftGenerationConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build representative baseline rows for one entity."""

    for source in config.representative_source_priority:
        candidates = sorted(predictions_by_source.get(source, []))
        if not candidates:
            continue
        baseline_rows = _build_baseline_rows(
            entity_uid=entity_uid,
            bmrb_id=bmrb_id,
            bundle_path=bundle_path,
            shift_path=candidates[0][1],
        )
        if baseline_rows:
            return baseline_rows, source
    return [], None


def _finish_entity_result(
    entity_uid: str,
    bmrb_id: str,
    candidate_specs: list[tuple[str, Path, Path]],
    source_counts: dict[str, int],
    written: int,
    skipped_existing: int,
    failed: int,
    failure_messages: list[str],
    baseline_rows: list[dict[str, Any]],
    baseline_source: str | None,
) -> UCBShiftEntityResult:
    """Build the final entity result payload."""

    status = "written"
    message = "UCBShift sidecars generated."
    usable_candidates = written + skipped_existing
    if written == 0 and skipped_existing > 0 and failed == 0:
        status = "skipped_existing"
        message = "All requested UCBShift sidecars already existed."
    elif usable_candidates == 0 and failed > 0:
        status = "failed"
        message = (
            "; ".join(dict.fromkeys(failure_messages))
            if failure_messages
            else "Every requested UCBShift sidecar generation failed."
        )
    elif failed > 0:
        status = "partial"
        message = (
            "; ".join(dict.fromkeys(failure_messages))
            if failure_messages
            else "UCBShift sidecar generation completed with partial failures."
        )

    return UCBShiftEntityResult(
        entity_uid=entity_uid,
        bmrb_id=bmrb_id,
        status=status,
        message=message,
        requested_candidates=len(candidate_specs),
        written_candidates=written,
        skipped_existing_candidates=skipped_existing,
        failed_candidates=failed,
        baseline_rows=len(baseline_rows),
        baseline_source=baseline_source,
        source_candidate_counts=source_counts,
    )


def _discover_candidate_specs(
    candidate_roots: dict[str, str],
    bmrb_id: str,
    data_root: Path,
    repo_root: Path,
    config: UCBShiftGenerationConfig,
) -> list[tuple[str, Path, Path]]:
    """Discover candidate PDBs and target sidecar paths for one entity."""
    specs: list[tuple[str, Path, Path]] = []
    source_paths: dict[str, list[Path]] = {}
    for source in config.selected_sources:
        relative_root = candidate_roots.get(source)
        if not relative_root:
            continue
        structure_root = resolve_existing_path(
            relative_root,
            data_root=data_root,
            repo_root=repo_root,
        )
        if structure_root is None or not structure_root.exists():
            continue
        source_paths[source] = sorted(structure_root.rglob("*.pdb"))

    selected_paths = _cap_candidate_paths_by_example(
        source_paths=source_paths,
        max_candidates=config.max_candidates_per_example,
    )
    for source in config.selected_sources:
        output_root = _resolve_output_root(
            source=source,
            bmrb_id=bmrb_id,
            config=config,
            data_root=data_root,
            repo_root=repo_root,
        )
        for pdb_path in selected_paths.get(source, []):
            candidate_id = normalize_candidate_key(pdb_path)
            output_path = output_root / f"{candidate_id}.csv"
            specs.append((source, pdb_path, output_path))
    return specs


def _cap_candidate_paths_by_example(
    source_paths: dict[str, list[Path]],
    max_candidates: int | None,
) -> dict[str, list[Path]]:
    """Return a source-balanced total candidate cap for one accession."""

    if max_candidates is None:
        return {source: list(paths) for source, paths in source_paths.items()}
    capped: dict[str, list[Path]] = {source: [] for source in source_paths}
    remaining = {source: list(paths) for source, paths in source_paths.items()}
    source_order = sorted(source_paths)
    while sum(len(paths) for paths in capped.values()) < max_candidates:
        progressed = False
        for source in source_order:
            if sum(len(paths) for paths in capped.values()) >= max_candidates:
                break
            if not remaining[source]:
                continue
            capped[source].append(remaining[source].pop(0))
            progressed = True
        if not progressed:
            break
    return capped


def _resolve_output_root(
    source: str,
    bmrb_id: str,
    config: UCBShiftGenerationConfig,
    data_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve one per-source sidecar output directory."""
    template = config.output_dir_templates.get(source)
    if template is None:
        raise ValueError(f"Missing UCBShift output template for source {source}.")
    return _resolve_output_path(
        template.format(bmrb_id=bmrb_id),
        data_root=data_root,
        repo_root=repo_root,
    )


def _build_baseline_rows(
    entity_uid: str,
    bmrb_id: str,
    bundle_path: Path,
    shift_path: Path,
) -> list[dict[str, Any]]:
    """Build baseline rows by aligning one UCBShift sidecar to bundle targets."""
    bundle = NMRTargetBundle.from_json(bundle_path)
    predicted = load_candidate_shift_file(shift_path, "ucbshift")
    rows: list[dict[str, Any]] = []
    for target in bundle.chemical_shifts:
        target_key = (int(target.seq_id), canonical_atom_name(target.atom_id))
        if target_key not in predicted:
            continue
        rows.append(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "target_id": target.target_id(),
                "predicted_value": float(predicted[target_key]),
                "corpus_panel": "ucbshift2_reference_corpus",
            }
        )
    return rows


def _realign_ucbshift_sidecar(path: Path, sequence_text: str) -> None:
    """Rewrite one UCBShift CSV so RESNUM matches the teacher sequence."""
    if not sequence_text:
        return
    target_sequence = [token.strip() for token in sequence_text.split("-") if token]
    if not target_sequence:
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        return

    remapped_rows = _remap_ucbshift_rows(rows, target_sequence)
    if not remapped_rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remapped_rows)


def _remap_ucbshift_rows(
    rows: list[dict[str, str]],
    target_sequence: list[str],
) -> list[dict[str, str]]:
    """Map one UCBShift residue numbering scheme onto the teacher sequence."""
    predicted_residues: list[tuple[int, str]] = []
    seen: set[int] = set()
    for row in rows:
        try:
            resnum = int(str(row["RESNUM"]).strip())
        except (KeyError, TypeError, ValueError):
            continue
        if resnum in seen:
            continue
        seen.add(resnum)
        predicted_residues.append((resnum, str(row.get("RESNAME", "")).strip()))
    if not predicted_residues:
        return rows

    mapping = _infer_residue_numbering_map(
        predicted_residues=predicted_residues,
        target_sequence=target_sequence,
    )
    if not mapping:
        return rows

    remapped_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            resnum = int(str(row["RESNUM"]).strip())
        except (KeyError, TypeError, ValueError):
            continue
        mapped_seq_id = mapping.get(resnum)
        if mapped_seq_id is None:
            continue
        updated = dict(row)
        updated["RESNUM"] = str(mapped_seq_id)
        remapped_rows.append(updated)
    return remapped_rows or rows


def _infer_residue_numbering_map(
    predicted_residues: list[tuple[int, str]],
    target_sequence: list[str],
) -> dict[int, int]:
    """Infer one monotonic residue-number map by longest exact subsequence match."""
    best_pred_start = 0
    best_target_start = 0
    best_length = 0
    predicted_names = [resname for _, resname in predicted_residues]

    for pred_start, pred_name in enumerate(predicted_names):
        for target_start, target_name in enumerate(target_sequence):
            if pred_name != target_name:
                continue
            length = 0
            while (
                pred_start + length < len(predicted_names)
                and target_start + length < len(target_sequence)
                and predicted_names[pred_start + length]
                == target_sequence[target_start + length]
            ):
                length += 1
            if length > best_length:
                best_length = length
                best_pred_start = pred_start
                best_target_start = target_start

    if best_length == 0:
        return {}

    anchor_resnum = predicted_residues[best_pred_start][0]
    delta = (best_target_start + 1) - anchor_resnum
    mapping: dict[int, int] = {}
    for resnum, resname in predicted_residues:
        seq_id = resnum + delta
        if not (1 <= seq_id <= len(target_sequence)):
            continue
        if target_sequence[seq_id - 1] != resname:
            continue
        mapping[resnum] = seq_id
    return mapping


def _resolve_configured_path(
    value: str,
    data_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve one configured path against repo and data roots."""
    path = Path(value)
    if path.is_absolute():
        return path
    if len(path.parts) == 1 and path.name == value:
        return path
    if path.parts and path.parts[0] == "data":
        return repo_root / path
    return repo_root / path


def _resolve_output_path(
    value: str | None,
    data_root: Path,
    repo_root: Path,
) -> Path | None:
    """Resolve one optional output path."""
    if value is None:
        return None
    return _resolve_configured_path(value, data_root=data_root, repo_root=repo_root)


def _shard_label(config: UCBShiftGenerationConfig) -> str:
    """Return a filename suffix for sharded sidecar generation."""

    if config.shard_count <= 1:
        return ""
    return f".shard{config.shard_index:02d}-of-{config.shard_count:02d}"


def _shard_output_path(
    path: Path | None,
    config: UCBShiftGenerationConfig,
) -> Path | None:
    """Return a shard-specific output path without changing the canonical name."""

    if path is None or config.shard_count <= 1:
        return path
    return path.with_name(f"{path.stem}{_shard_label(config)}{path.suffix}")


def _resolve_python_executable(
    value: str,
    data_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve the Python executable used for UCBShift subprocesses."""
    normalized = str(value).strip()
    if normalized in {"", "python", "sys.executable"}:
        return Path(sys.executable)
    return _resolve_configured_path(
        normalized, data_root=data_root, repo_root=repo_root
    )


def _resolve_models_dir(models_dir: Path) -> tuple[Path, int]:
    """Resolve one canonical UCBShift model directory and file count."""
    candidates = [models_dir, models_dir / "models"]
    counts = [
        len(list(candidate.glob("*.sav"))) if candidate.exists() else 0
        for candidate in candidates
    ]
    for candidate, count in zip(candidates, counts, strict=True):
        if count >= 141:
            return candidate, count
    best_index = max(range(len(candidates)), key=lambda index: counts[index])
    return candidates[best_index], counts[best_index]


def _format_models_dir_argument(models_dir: Path) -> str:
    """Return one slash-safe model directory argument for UCBShift."""
    value = str(models_dir)
    return value if value.endswith(os.sep) else value + os.sep


def _build_ucbshift_env(runtime: UCBShiftRuntime) -> dict[str, str]:
    """Build one environment dictionary for subprocess execution."""
    env = os.environ.copy()
    path_entries = [str(runtime.ucbshift_root / "bins")]
    if runtime.reduce_executable is not None:
        path_entries.append(str(runtime.reduce_executable.parent))
    env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
    if runtime.dssp_executable is not None:
        env["DSSP"] = str(runtime.dssp_executable)
    env.setdefault("MPLBACKEND", "Agg")
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        env[key] = "1"
    return env


def _which_path(executable: str) -> Path | None:
    """Return the resolved executable path when it exists in PATH."""
    resolved = subprocess.run(
        ["which", executable],
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0:
        return None
    output = resolved.stdout.strip()
    return Path(output) if output else None
