"""Run fixed target-unread HOLD checks in an isolated parallel snapshot.

This optional preflight is not science, authorization, or a replacement for
``.auto/measure.sh``. Its exact committed inputs are copied to a private
snapshot before workers start; the authoritative measure remains serial.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
WORKERS_ENV = "ATYPEMU_HOLD_PREFLIGHT_WORKERS"
TIMEOUT_SECONDS = 300.0
PYTHON = sys.executable
PREFIX = "gpuopt/candidates/"
TASKS = (
    (
        "all-atom-recount",
        (PYTHON, PREFIX + "check_nested_support_all_atom_recount.py", "--self-test"),
    ),
    (
        "catalog-bound-check",
        (
            PYTHON,
            PREFIX + "check_nested_support_catalog.py",
            "--check-bound-evidence",
            "--self-test",
        ),
    ),
    (
        "catalog-self-test",
        (PYTHON, PREFIX + "nested_support_catalog.py", "self-test"),
    ),
    (
        "solution-state-plan",
        (
            PYTHON,
            PREFIX + "check_solution_state_protonation_support_plan.py",
            "--self-test",
            "--acknowledge-hold-only",
        ),
    ),
    (
        "condition-uncertainty-plan",
        (
            PYTHON,
            PREFIX + "check_condition_uncertainty_protonation_plan.py",
            "--self-test",
            "--acknowledge-hold-only",
        ),
    ),
    (
        "condition-uncertainty-inputs",
        (
            PYTHON,
            PREFIX + "check_condition_uncertainty_inputs.py",
            "--self-test",
        ),
    ),
)
EXPECTED_STDOUT = {
    "all-atom-recount": (
        "METRIC all_atom_recount_checks=135021",
        "METRIC source_target_values_read=0",
        "METRIC outer_or_formal_metrics_opened=0",
        "METRIC authorization_consumed=0",
        "STATUS HOLD_ALL_ATOM_COUNT_BLOCKED",
    ),
    "catalog-bound-check": (
        "METRIC support_catalog_checker_checks=2079812",
        "METRIC target_values_read=0",
        "METRIC source_scores_read=0",
        "METRIC outer_or_formal_metrics_opened=0",
        "METRIC science_executed=0",
        "METRIC authorization_consumed=0",
        "STATUS PASS",
    ),
    "catalog-self-test": (
        "METRIC support_catalog_checks=9",
        "METRIC source_target_values_read=0",
        "METRIC outer_or_formal_metrics_opened=0",
        "METRIC authorization_consumed=0",
    ),
    "solution-state-plan": (
        "METRIC solution_state_support_plan_checks=42",
        "METRIC target_values_read=0",
        "METRIC source_scores_read=0",
        "METRIC science_executed=0",
        "METRIC authorization_consumed=0",
        "STATUS HOLD_CONDITION_MANIFEST_INCOMPLETE",
    ),
    "condition-uncertainty-plan": (
        "METRIC condition_uncertainty_support_plan_checks=105",
        "METRIC target_values_read=0",
        "METRIC source_scores_read=0",
        "METRIC science_executed=0",
        "METRIC authorization_consumed=0",
        "STATUS HOLD_INPUTS_QUALIFIED_PROTONATION_SMOKE_UNRUN",
    ),
    "condition-uncertainty-inputs": (
        "METRIC condition_uncertainty_input_self_test_checks=16",
        "METRIC target_values_read=0",
        "METRIC source_scores_read=0",
        "METRIC science_executed=0",
        "METRIC authorization_consumed=0",
        "STATUS HOLD_TARGET_UNREAD_INPUT_SELF_TEST_PASS",
    ),
}
SOURCE_RELATIVES = (
    "gpuopt/candidates/check_condition_uncertainty_inputs.py",
    "gpuopt/candidates/check_condition_uncertainty_protonation_plan.py",
    "gpuopt/candidates/check_nested_support_all_atom_recount.py",
    "gpuopt/candidates/check_nested_support_catalog.py",
    "gpuopt/candidates/check_openmm86_unique_assigned_ph_recovery_v3_independent.py",
    "gpuopt/candidates/check_solution_state_protonation_support_plan.py",
    "gpuopt/candidates/hold_static_preflight.py",
    "gpuopt/candidates/nested_support_all_atom_recount.py",
    "gpuopt/candidates/nested_support_catalog.py",
    "gpuopt/candidates/nested_support_count_plan.py",
    "gpuopt/candidates/freeze_condition_uncertainty_inputs.py",
)
REQUIRED = SOURCE_RELATIVES + (
    ".auto/measure.sh",
    "gpuopt/candidates/openmm86_protonation_runtime.Dockerfile",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_inputs_check_receipt_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_protonation_plan_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_protein_sequence_manifest_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_solution_state_protonation_support_plan_v1.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_solution_state_condition_catalog_v1.json",
    "references/thesis_jeon.pdf",
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json",
    ".auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz",
    ".auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json",
    ".auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json",
    ".auto/staging/atypemu_nested_support_count_v1_solution_conditions_api_v2_evidence_v1.zip",
    ".auto/staging/openmm86_unique_assigned_ph_v1_recovery_v3/receipt.json",
    ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1_yulab/execution_receipt.json",
    ".auto/staging/atypemu_nested_support_count_v1_all_atom_recount_v1_yulab/recount_results.tar.gz",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_policy.json",
    "gpuopt/preunblind/atypemu_nested_support_count_v1_all_atom_recount_receipt.json",
)
TRACKED_REQUIRED = tuple(
    relative
    for relative in REQUIRED
    if not relative.startswith(".auto/staging/")
    and relative != "references/thesis_jeon.pdf"
)


class Result(NamedTuple):
    task_id: str
    returncode: Optional[int]
    seconds: float
    stdout: bytes
    stderr: bytes
    error: Optional[str]


def _child_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": tempfile.gettempdir(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "TMPDIR": tempfile.gettempdir(),
    }


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    # The session/process-group ID remains the leader PID even if the leader has
    # already exited.  Always signal the group: a descendant may still hold a pipe.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_one(payload: Tuple[str, Tuple[str, ...], str, float]) -> Result:
    task_id, command, root, timeout = payload
    started = time.monotonic()
    process: Optional[subprocess.Popen[bytes]] = None
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return Result(
            task_id,
            process.returncode,
            time.monotonic() - started,
            stdout,
            stderr,
            None,
        )
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            _terminate_process_group(process)
        return Result(
            task_id,
            None,
            time.monotonic() - started,
            exc.stdout or b"",
            exc.stderr or b"",
            "timeout",
        )
    except Exception as exc:  # pragma: no cover - fail-closed process boundary
        if process is not None:
            _terminate_process_group(process)
        return Result(
            task_id,
            None,
            time.monotonic() - started,
            b"",
            b"",
            f"{type(exc).__name__}: {exc}",
        )


def run_tasks(
    tasks: Sequence[Tuple[str, Tuple[str, ...]]],
    root: Path,
    workers: int,
    timeout: float,
) -> Tuple[Tuple[Result, ...], float]:
    if len({task_id for task_id, _ in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    payloads = [(task_id, command, str(root), timeout) for task_id, command in tasks]
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = tuple(executor.map(run_one, payloads))
    return tuple(sorted(results)), time.monotonic() - started


def worker_count(task_count: int) -> int:
    raw = os.environ.get(WORKERS_ENV)
    if raw is None:
        return min(4, os.cpu_count() or 1, task_count)
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError(f"{WORKERS_ENV} must be an integer") from exc
    if not 1 <= workers <= task_count:
        raise ValueError(f"{WORKERS_ENV} must be between 1 and {task_count}")
    return workers


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise RuntimeError("git executable is unavailable")
    environment = _child_environment()
    return subprocess.run(
        (git, "-c", "core.hooksPath=/dev/null", *arguments),
        cwd=str(root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
        check=False,
    )


def committed_head(root: Path) -> str:
    top = _git(root, "rev-parse", "--show-toplevel")
    if (
        top.returncode != 0
        or Path(top.stdout.decode().strip()).resolve() != root.resolve()
    ):
        raise RuntimeError("preflight root is not the Git worktree root")
    tracked = _git(root, "ls-files", "--error-unmatch", "--", *TRACKED_REQUIRED)
    if tracked.returncode != 0:
        raise RuntimeError("a required source or plan is not tracked")
    dirty = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *TRACKED_REQUIRED,
    )
    if dirty.returncode != 0 or dirty.stdout:
        raise RuntimeError("a required source or plan differs from committed HEAD")
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError("cannot resolve committed HEAD")
    return head.stdout.decode("ascii").strip()


def _read_once(root: Path, relative: str) -> Tuple[bytes, Tuple[int, ...]]:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"invalid required input path: {relative}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = os.open(root, directory_flags)
    descriptor = -1
    try:
        for component in pure.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(pure.parts[-1], file_flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"required input is not regular: {relative}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked = os.stat(pure.parts[-1], dir_fd=directory, follow_symlinks=False)

        def identity(value: os.stat_result) -> Tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(linked):
            raise RuntimeError(f"required input changed while reading: {relative}")
        return b"".join(chunks), identity(after)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def _read_regular(root: Path, relative: str) -> bytes:
    first, first_identity = _read_once(root, relative)
    second, second_identity = _read_once(root, relative)
    if first_identity != second_identity or first != second:
        raise RuntimeError(f"required input changed across reads: {relative}")
    return first


def _git_blob(root: Path, head: str, relative: str) -> bytes:
    result = _git(root, "cat-file", "blob", f"{head}:{relative}")
    if result.returncode != 0:
        raise RuntimeError(f"cannot read committed blob: {relative}")
    return result.stdout


def isolated_snapshot(root: Path, destination: Path, head: str) -> str:
    digest = hashlib.sha256(head.encode("ascii"))
    tracked = set(TRACKED_REQUIRED)
    for relative in sorted(REQUIRED):
        value = (
            _git_blob(root, head, relative)
            if relative in tracked
            else _read_regular(root, relative)
        )
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(value).to_bytes(8, "big") + value)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(value)
    return digest.hexdigest()


def process_passed(results: Sequence[Result], expected: int) -> bool:
    return len(results) == expected and all(
        result.returncode == 0 and result.error is None for result in results
    )


def outputs_match(results: Sequence[Result]) -> bool:
    if {result.task_id for result in results} != set(EXPECTED_STDOUT):
        return False
    return all(
        result.stderr == b""
        and tuple(result.stdout.decode("utf-8", errors="strict").splitlines())
        == EXPECTED_STDOUT[result.task_id]
        for result in results
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        sleepers = tuple(
            (task_id, (PYTHON, "-c", "import time; time.sleep(0.5)"))
            for task_id in ("charlie", "alpha", "bravo")
        )
        serial, serial_seconds = run_tasks(sleepers, Path(temporary), 1, 2.0)
        parallel, parallel_seconds = run_tasks(sleepers, Path(temporary), 3, 2.0)
        assert process_passed(serial, 3) and process_passed(parallel, 3)
        assert [result.task_id for result in parallel] == ["alpha", "bravo", "charlie"]
        assert parallel_seconds < serial_seconds * 0.8
        failed, _ = run_tasks(
            (("failure", (PYTHON, "-c", "raise SystemExit(7)")),),
            Path(temporary),
            1,
            2.0,
        )
        timed_out, _ = run_tasks(
            (("timeout", (PYTHON, "-c", "import time; time.sleep(1)")),),
            Path(temporary),
            1,
            0.05,
        )
        assert not process_passed(failed, 1) and failed[0].returncode == 7
        assert not process_passed(timed_out, 1) and timed_out[0].error == "timeout"
        child_pid_path = Path(temporary) / "descendant.pid"
        descendant_code = (
            "import pathlib,subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
            "time.sleep(30)"
        )
        descendant, _ = run_tasks(
            (("descendant", (PYTHON, "-c", descendant_code)),),
            Path(temporary),
            1,
            0.5,
        )
        assert descendant[0].error == "timeout" and child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text())
        for _ in range(40):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("timed-out descendant process survived")
        orphan_pid_path = Path(temporary) / "orphan.pid"
        orphan_code = (
            "import pathlib,subprocess,sys;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            f"pathlib.Path({str(orphan_pid_path)!r}).write_text(str(p.pid))"
        )
        orphan, _ = run_tasks(
            (("orphan", (PYTHON, "-c", orphan_code)),),
            Path(temporary),
            1,
            0.5,
        )
        assert orphan[0].error == "timeout" and orphan_pid_path.is_file()
        orphan_pid = int(orphan_pid_path.read_text())
        for _ in range(40):
            try:
                os.kill(orphan_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("exited leader left a timed-out descendant")
        valid_outputs = tuple(
            Result(
                task_id,
                0,
                0.0,
                ("\n".join(lines) + "\n").encode(),
                b"",
                None,
            )
            for task_id, lines in sorted(EXPECTED_STDOUT.items())
        )
        assert outputs_match(valid_outputs)
        tampered = list(valid_outputs)
        tampered[0] = tampered[0]._replace(
            stdout=tampered[0].stdout.replace(b"=0", b"=1", 1)
        )
        assert not outputs_match(tuple(tampered))


def emit(results: Sequence[Result]) -> None:
    for result in results:
        stdout_sha = hashlib.sha256(result.stdout).hexdigest()
        stderr_sha = hashlib.sha256(result.stderr).hexdigest()
        print(
            f"HOLD_PREFLIGHT_TASK id={result.task_id} returncode={result.returncode}"
            f" seconds={result.seconds:.3f} stdout={len(result.stdout)}:{stdout_sha}"
            f" stderr={len(result.stderr)}:{stderr_sha} error={result.error or 'none'}"
        )


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("HOLD_PREFLIGHT SELF_TEST_PASS")
        return 0
    if sys.argv[1:]:
        print("HOLD_PREFLIGHT FAIL error=only --self-test is accepted")
        return 1
    try:
        workers = worker_count(len(TASKS))
        head = committed_head(ROOT)
        with tempfile.TemporaryDirectory(prefix="atypemu-hold-preflight-") as temporary:
            snapshot_root = Path(temporary)
            snapshot_sha256 = isolated_snapshot(ROOT, snapshot_root, head)
            if head != committed_head(ROOT):
                raise RuntimeError("committed source changed while snapshotting")
            results, seconds = run_tasks(TASKS, snapshot_root, workers, TIMEOUT_SECONDS)
        unchanged = head == committed_head(ROOT)
    except Exception as exc:
        print(f"HOLD_PREFLIGHT FAIL error={type(exc).__name__}: {exc}")
        return 1
    emit(results)
    if (
        not process_passed(results, len(TASKS))
        or not outputs_match(results)
        or not unchanged
    ):
        print(
            f"HOLD_PREFLIGHT FAIL tasks={len(results)}/{len(TASKS)}"
            f" workers={workers} expected_outputs=false"
            f" committed_source_unchanged={str(unchanged).lower()}"
        )
        return 1
    print(
        f"HOLD_PREFLIGHT PASS tasks={len(results)} workers={workers}"
        f" seconds={seconds:.3f} git_head={head}"
        f" isolated_snapshot_sha256={snapshot_sha256}"
        " expected_outputs=true committed_source_unchanged=true"
        " official_serial_measure_required=true"
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
