#!/usr/bin/env python3
"""Consume one exact-bound recovery-v4 launch intent, or do nothing.

This launcher is deliberately a one-shot, local-only boundary.  It does not
create an output or start a subprocess until every fixed source and the active
intent have been byte-checked.  The intent digest is left unbound while this
source is committed.  The intent canonically binds the committed launcher and
all other execution sources without a self-referential source hash.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HANDLER_RELATIVE = Path(
    "gpuopt/candidates/run_openmm86_deposited_ph_recovery_v4_if_intent.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_catalog_"
    "recovery_v4_plan.json"
)
PRODUCER_RELATIVE = Path(
    "gpuopt/candidates/solution_state_openmm86_deposited_ph_catalog_recovery_v4.py"
)
CHECKER_RELATIVE = Path(
    "gpuopt/candidates/"
    "check_solution_state_openmm86_deposited_ph_catalog_recovery_v4.py"
)
PARENT_PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_plan.json"
)
PARENT_VALIDATOR_RELATIVE = Path("gpuopt/candidates/nested_support_count_plan.py")

ACTIVE_INTENT_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4_"
    "launch_intent.json"
)
CONSUMED_INTENT_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4_"
    "launch_intent.consumed.json"
)
EXECUTION_RECEIPT_RELATIVE = Path(
    ".auto/staging/"
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4_"
    "execution_receipt.json"
)
OUTPUT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_openmm86_deposited_ph_"
    "api_v4_recovery"
)
OUTPUT_RECEIPT_RELATIVE = OUTPUT_RELATIVE / "receipt.json"

CANDIDATE_ID = "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4"
INTENT_CONTRACT = (
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4_launch_v1"
)
EXECUTION_CONTRACT = (
    "atypemu_nested_support_count_v1_openmm86_deposited_ph_recovery_v4_execution_v1"
)

PLAN_RAW_SHA256 = "079f426a454cbfaacc8bdc86dd1e7a5b577fd0cfec6ae90f725dd6158738bff3"
PRODUCER_RAW_SHA256 = "3242f79ffff689ce8b6c10a9d69c53dde31f5f004c44931b47f8068d3cc07a9a"
CHECKER_RAW_SHA256 = "d417cdd32608bc1cfdf3a37d3fc4c7fae70a553fcdf896278e09f8dae42d145c"
FAILED_V3_INTENT_RAW_SHA256 = (
    "c75986bea0ce11684fec2d0aae6aa594ec65527a9814cd89525083f1bd9d7a54"
)
RUN126_RAW_LINE_SHA256 = (
    "0fe712c190d65653ff0521f3bb8c3518fef96296c80354be4a1b1c9c8cfdd15a"
)

CLOSED_CONTROLS = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
EXPECTED_HOLD_STATUS = (
    "HOLD_LOCAL_ONLY_PENDING_SEPARATELY_EXACT_BOUND_"
    "IMMUTABLE_ARCHIVE_AND_EXECUTION_RECEIPT"
)
MAX_INTENT_BYTES = 65_536
MAX_SOURCE_BYTES = 2_000_000
MAX_OUTPUT_RECEIPT_BYTES = 10_000_000
MAX_CAPTURE_BYTES = 16_384
MAX_SUBPROCESS_SECONDS = 780
MAX_GIT_SECONDS = 30
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

SourceRecord = Dict[str, str]
Runner = Callable[[Sequence[str], Path], Dict[str, Any]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON in %s: %s" % (label, error)) from error


def _read_regular(path: Path, label: str, maximum: int) -> bytes:
    """Read one stable, direct regular file without following its final link."""
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError("missing %s" % label) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("%s is indirect or non-regular" % label)
    if before.st_size > maximum:
        raise ValueError("%s exceeds the byte limit" % label)
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError("%s changed before reading" % label)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(before.st_size + 1)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or (after.st_dev, after.st_ino, after.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise ValueError("%s changed while reading" % label)
        return raw
    finally:
        os.close(descriptor)


def _require_direct_path(root: Path, relative: Path, label: str) -> Path:
    """Reject symlinks and non-directory parents on one fixed relative path."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe %s path" % label)
    current = root
    for index, piece in enumerate(relative.parts):
        current = current / piece
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            return root / relative
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("indirect %s path" % label)
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise ValueError("non-directory %s parent" % label)
    return current


def _write_new(path: Path, raw: bytes, mode: int = 0o600) -> None:
    """Write a complete file once; callers must not create a replacement."""
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _capture_pipe(pipe: Any, limit: int, result: Dict[str, Any]) -> None:
    digest = hashlib.sha256()
    captured = bytearray()
    total = 0
    try:
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if len(captured) < limit:
                captured.extend(chunk[: limit - len(captured)])
    except (OSError, ValueError):
        pass
    captured_bytes = bytes(captured)
    result.update(
        {
            "captured_bytes": len(captured_bytes),
            "captured_sha256": _sha256(captured_bytes),
            "captured_text": captured_bytes.decode("utf-8", "backslashreplace"),
            "sha256": digest.hexdigest(),
            "total_bytes": total,
            "truncated": total > len(captured_bytes),
        }
    )


def _run_bounded(command: Sequence[str], root: Path) -> Dict[str, Any]:
    """Run once while draining both streams without retaining unbounded output."""
    try:
        process = subprocess.Popen(
            list(command),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        return {
            "returncode": None,
            "spawn_error": "%s: %s" % (type(error).__name__, error),
            "stderr": _empty_capture(),
            "stdout": _empty_capture(),
        }
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: Dict[str, Any] = {}
    stderr: Dict[str, Any] = {}
    stdout_thread = threading.Thread(
        target=_capture_pipe,
        args=(process.stdout, MAX_CAPTURE_BYTES, stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_capture_pipe,
        args=(process.stderr, MAX_CAPTURE_BYTES, stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=MAX_SUBPROCESS_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        returncode = process.wait()
    stdout_thread.join(timeout=MAX_GIT_SECONDS)
    stderr_thread.join(timeout=MAX_GIT_SECONDS)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.stdout.close()
        process.stderr.close()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
    return {
        "returncode": returncode,
        "stderr": stderr,
        "stdout": stdout,
        "timed_out": timed_out,
    }


def _empty_capture() -> Dict[str, Any]:
    return {
        "captured_bytes": 0,
        "captured_sha256": _sha256(b""),
        "captured_text": "",
        "sha256": _sha256(b""),
        "total_bytes": 0,
        "truncated": False,
    }


def _git_bytes_exact(root: Path, arguments: Sequence[str], maximum: int) -> bytes:
    """Git output is source bytes, so retain it as bytes rather than text."""
    process = subprocess.Popen(
        ["git", *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert process.stdout is not None
    captured: Dict[str, Any] = {}

    def capture_raw() -> None:
        data = bytearray()
        total = 0
        try:
            while True:
                chunk = process.stdout.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if len(data) <= maximum:
                    data.extend(chunk[: maximum + 1 - len(data)])
        except (OSError, ValueError):
            pass
        captured.update({"raw": bytes(data), "total": total})

    reader = threading.Thread(target=capture_raw, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=MAX_GIT_SECONDS)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        reader.join(timeout=1)
        raise ValueError("git %s timed out" % " ".join(arguments)) from error
    reader.join(timeout=MAX_GIT_SECONDS)
    if reader.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.stdout.close()
        reader.join(timeout=1)
        raise ValueError("git %s pipe did not close" % " ".join(arguments))
    raw = captured.get("raw", b"")
    if captured.get("total", maximum + 1) > maximum:
        raise ValueError("git %s exceeded the byte limit" % " ".join(arguments))
    if returncode != 0:
        raise ValueError("git %s failed" % " ".join(arguments))
    return raw


def _head_commit(root: Path) -> str:
    raw = _git_bytes_exact(root, ["rev-parse", "HEAD"], 128)
    try:
        commit = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("git HEAD is not ASCII") from error
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("git HEAD is not a full lowercase commit")
    return commit


def _verify_source(
    root: Path, head: str, relative: Path, expected_sha256: Optional[str]
) -> SourceRecord:
    path = _require_direct_path(root, relative, "%s source" % relative)
    current = _read_regular(path, "%s source" % relative, MAX_SOURCE_BYTES)
    actual_sha256 = _sha256(current)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("%s source SHA-256 drifted" % relative)
    committed = _git_bytes_exact(
        root, ["show", "%s:%s" % (head, relative.as_posix())], MAX_SOURCE_BYTES
    )
    if current != committed:
        raise ValueError("%s source bytes do not match committed HEAD" % relative)
    return {"path": relative.as_posix(), "sha256": actual_sha256}


def _verify_sources(root: Path, head: str) -> Dict[str, SourceRecord]:
    return {
        "checker": _verify_source(root, head, CHECKER_RELATIVE, CHECKER_RAW_SHA256),
        "handler": _verify_source(root, head, HANDLER_RELATIVE, None),
        "parent_plan": _verify_source(root, head, PARENT_PLAN_RELATIVE, None),
        "parent_validator": _verify_source(root, head, PARENT_VALIDATOR_RELATIVE, None),
        "plan": _verify_source(root, head, PLAN_RELATIVE, PLAN_RAW_SHA256),
        "producer": _verify_source(root, head, PRODUCER_RELATIVE, PRODUCER_RAW_SHA256),
    }


def _expected_intent(
    head: str, sources: Dict[str, SourceRecord]
) -> Dict[str, Any]:
    return {
        "artifact_kind": "target_unread_metadata_fetch_launch_intent_not_authorization",
        "candidate_id": CANDIDATE_ID,
        "checker": {
            "path": CHECKER_RELATIVE.as_posix(),
            "sha256": CHECKER_RAW_SHA256,
        },
        "contract": INTENT_CONTRACT,
        "failed_v3_launch_intent_sha256": FAILED_V3_INTENT_RAW_SHA256,
        "git_commit": head,
        "handler": sources["handler"],
        "output": OUTPUT_RELATIVE.as_posix(),
        "parent_plan": sources["parent_plan"],
        "parent_validator": sources["parent_validator"],
        "plan": {"path": PLAN_RELATIVE.as_posix(), "sha256": PLAN_RAW_SHA256},
        "producer": {
            "path": PRODUCER_RELATIVE.as_posix(),
            "sha256": PRODUCER_RAW_SHA256,
        },
        "run_126_raw_line_sha256": RUN126_RAW_LINE_SHA256,
        **CLOSED_CONTROLS,
    }


def _validate_intent(
    raw: bytes, head: str, sources: Dict[str, SourceRecord]
) -> Dict[str, Any]:
    expected = _expected_intent(head, sources)
    canonical = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if raw != canonical:
        raise ValueError("active recovery-v4 launch intent raw bytes are not canonical")
    value = _decode_json(raw, "active recovery-v4 launch intent")
    if not isinstance(value, dict) or value != expected:
        raise ValueError("active recovery-v4 launch intent schema or fields drifted")
    return value


def _reject_prior_artifacts(root: Path) -> None:
    for relative, label in (
        (CONSUMED_INTENT_RELATIVE, "consumed recovery-v4 launch intent"),
        (EXECUTION_RECEIPT_RELATIVE, "recovery-v4 execution receipt"),
        (OUTPUT_RELATIVE, "recovery-v4 output"),
    ):
        if os.path.lexists(root / relative):
            raise ValueError(
                "%s already exists; one-shot launch will not retry" % label
            )


def _consume_intent(active: Path, consumed: Path, raw: bytes) -> None:
    """Make the immutable consumed marker before unlinking the active intent."""
    try:
        os.link(active, consumed, follow_symlinks=False)
    except OSError as link_error:
        if link_error.errno != errno.EXDEV:
            raise ValueError(
                "could not atomically link consumed launch intent"
            ) from link_error
        try:
            _write_new(consumed, raw, 0o444)
        except OSError as copy_error:
            raise ValueError(
                "could not O_EXCL-copy consumed launch intent"
            ) from copy_error
    consumed_raw = _read_regular(
        consumed, "consumed recovery-v4 launch intent", MAX_INTENT_BYTES
    )
    if consumed_raw != raw:
        raise ValueError("consumed recovery-v4 launch intent bytes drifted")
    active_raw = _read_regular(
        active, "active recovery-v4 launch intent", MAX_INTENT_BYTES
    )
    if active_raw != raw:
        raise ValueError("active recovery-v4 launch intent changed before consumption")
    os.chmod(consumed, 0o444, follow_symlinks=False)
    os.unlink(active)


def _output_receipt_binding(root: Path) -> Dict[str, Optional[str]]:
    binding: Dict[str, Optional[str]] = {
        "error": None,
        "path": OUTPUT_RECEIPT_RELATIVE.as_posix(),
        "sha256": None,
    }
    path = root / OUTPUT_RECEIPT_RELATIVE
    if not os.path.lexists(path):
        return binding
    try:
        raw = _read_regular(
            path, "recovery-v4 output receipt", MAX_OUTPUT_RECEIPT_BYTES
        )
    except (OSError, ValueError) as error:
        binding["error"] = "%s: %s" % (type(error).__name__, error)
        return binding
    binding["sha256"] = _sha256(raw)
    return binding


def _receipt_payload(
    head: str,
    intent_sha256: str,
    sources: Dict[str, SourceRecord],
    outcome: str,
    error: Optional[str],
    producer: Optional[Dict[str, Any]],
    checker: Optional[Dict[str, Any]],
    output_receipt: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    return {
        "artifact_kind": "target_unread_metadata_fetch_execution_receipt",
        "candidate_id": CANDIDATE_ID,
        "closed_controls": CLOSED_CONTROLS,
        "consumed_intent": {
            "path": CONSUMED_INTENT_RELATIVE.as_posix(),
            "sha256": intent_sha256,
        },
        "contract": EXECUTION_CONTRACT,
        "error": error,
        "git_commit": head,
        "outcome": outcome,
        "output_receipt": output_receipt,
        "status": (
            EXPECTED_HOLD_STATUS
            if outcome == "local_hold_verified"
            else "HOLD_LOCAL_ONLY_EXECUTION_FAILED"
        ),
        "subprocesses": {"checker": checker, "producer": producer},
        "source_hashes": sources,
    }


def _write_execution_receipt(root: Path, payload: Dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new(root / EXECUTION_RECEIPT_RELATIVE, raw, 0o600)


def _captured_has_expected_terminal_status(result: Dict[str, Any]) -> bool:
    stdout = result["stdout"]["captured_text"]
    return any(
        line == "STATUS %s" % EXPECTED_HOLD_STATUS for line in stdout.splitlines()
    )


def _failure_receipt(
    root: Path,
    head: str,
    intent_sha256: str,
    sources: Dict[str, SourceRecord],
    outcome: str,
    error: str,
    producer: Optional[Dict[str, Any]],
    checker: Optional[Dict[str, Any]],
) -> int:
    try:
        output_receipt = _output_receipt_binding(root)
        if output_receipt["error"] is not None or output_receipt["sha256"] is None:
            raise ValueError("verified output receipt binding is unavailable")
        _write_execution_receipt(
            root,
            _receipt_payload(
                head,
                intent_sha256,
                sources,
                outcome,
                error,
                producer,
                checker,
                output_receipt,
            ),
        )
    except Exception as receipt_error:
        print("REFUSAL could not seal execution failure: %s" % receipt_error)
        return 1
    print("STATUS HOLD_LOCAL_ONLY_EXECUTION_FAILED")
    return 1


def run_handler(root: Path, runner: Runner = _run_bounded) -> int:
    """Run the fixed one-shot protocol.  ``runner`` exists for synthetic tests."""
    active = _require_direct_path(
        root, ACTIVE_INTENT_RELATIVE, "active recovery-v4 launch intent"
    )
    if not os.path.lexists(active):
        print("NO_OP recovery-v4 active launch intent is absent")
        return 0
    _reject_prior_artifacts(root)
    raw = _read_regular(active, "active recovery-v4 launch intent", MAX_INTENT_BYTES)
    head = _head_commit(root)
    sources = _verify_sources(root, head)
    _validate_intent(raw, head, sources)
    intent_sha256 = _sha256(raw)

    if _head_commit(root) != head or _verify_sources(root, head) != sources:
        raise ValueError("HEAD or execution sources changed before intent consumption")

    try:
        _consume_intent(active, root / CONSUMED_INTENT_RELATIVE, raw)
    except Exception as error:
        return _failure_receipt(
            root,
            head,
            intent_sha256,
            sources,
            "intent_consumption_failed",
            "%s: %s" % (type(error).__name__, error),
            None,
            None,
        )

    if _head_commit(root) != head or _verify_sources(root, head) != sources:
        return _failure_receipt(
            root, head, intent_sha256, sources, "source_recheck_failed",
            "HEAD or execution sources changed before producer", None, None,
        )
    producer = runner(
        [
            sys.executable,
            PRODUCER_RELATIVE.as_posix(),
            "fetch",
            "--acknowledge-target-unread-metadata-only",
        ],
        root,
    )
    if producer["returncode"] != 0:
        return _failure_receipt(
            root,
            head,
            intent_sha256,
            sources,
            "producer_failed",
            "producer return code was %r" % producer["returncode"],
            producer,
            None,
        )

    if _head_commit(root) != head or _verify_sources(root, head) != sources:
        return _failure_receipt(
            root, head, intent_sha256, sources, "source_recheck_failed",
            "HEAD or execution sources changed before checker", producer, None,
        )
    checker = runner(
        [
            sys.executable,
            CHECKER_RELATIVE.as_posix(),
            "verify-local-consistency",
            "--acknowledge-target-unread-metadata-only",
        ],
        root,
    )
    if (
        checker["returncode"] != 4
        or not _captured_has_expected_terminal_status(checker)
    ):
        return _failure_receipt(
            root,
            head,
            intent_sha256,
            sources,
            "checker_failed",
            "checker did not return the expected local HOLD terminal status",
            producer,
            checker,
        )

    try:
        output_receipt = _output_receipt_binding(root)
        _write_execution_receipt(
            root,
            _receipt_payload(
                head,
                intent_sha256,
                sources,
                "local_hold_verified",
                None,
                producer,
                checker,
                output_receipt,
            ),
        )
    except Exception as error:
        print("REFUSAL could not seal local HOLD execution: %s" % error)
        return 1
    print("STATUS %s" % EXPECTED_HOLD_STATUS)
    return 0


def _synthetic_sources() -> Dict[str, SourceRecord]:
    return {
        name: {"path": name + ".py", "sha256": ("%x" % index) * 64}
        for index, name in enumerate(
            ("checker", "handler", "parent_plan", "parent_validator", "plan", "producer"),
            start=1,
        )
    }


def _synthetic_intent(head: str, sources: Dict[str, SourceRecord]) -> bytes:
    return (json.dumps(_expected_intent(head, sources), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _expect_value_error(action: Callable[[], Any], label: str) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("%s was accepted" % label)


def self_test() -> int:
    """Exercise synthetic parsing and process boundaries without producer access."""
    checks = 0
    head = "a" * 40
    sources = _synthetic_sources()
    raw = _synthetic_intent(head, sources)
    assert _validate_intent(raw, head, sources) == _expected_intent(head, sources)
    checks += 1
    _expect_value_error(
        lambda: _validate_intent(raw + b" ", head, sources), "noncanonical intent bytes"
    )
    duplicate = (
        b'{"git_commit":"'
        + b"a" * 40
        + b'","git_commit":"'
        + b"a" * 40
        + b'"}'
    )
    _expect_value_error(
        lambda: _validate_intent(duplicate, head, sources),
        "duplicate intent keys",
    )
    altered = _expected_intent(head, sources)
    altered["output"] = "not-the-fixed-output"
    altered_raw = json.dumps(altered, sort_keys=True).encode("utf-8")
    _expect_value_error(
        lambda: _validate_intent(altered_raw, head, sources),
        "intent output field drift",
    )
    checks += 3

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert run_handler(root) == 0
        checks += 1

    for relative, label, directory in (
        (CONSUMED_INTENT_RELATIVE, "existing consumed marker", False),
        (EXECUTION_RECEIPT_RELATIVE, "existing execution receipt", False),
        (OUTPUT_RELATIVE, "existing output", True),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / ACTIVE_INTENT_RELATIVE
            active.parent.mkdir(parents=True)
            active.write_bytes(raw)
            artifact = root / relative
            if directory:
                artifact.mkdir()
            else:
                artifact.write_bytes(b"already sealed")
            _expect_value_error(lambda: run_handler(root), label)
            assert active.read_bytes() == raw
            checks += 1

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "new.json"
        _write_new(path, b"first")
        try:
            _write_new(path, b"second")
        except FileExistsError:
            checks += 1
        else:
            raise AssertionError("O_EXCL receipt clobber was accepted")

    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 20000); sys.stderr.write('y' * 20000)",
    ]
    with tempfile.TemporaryDirectory() as temporary:
        result = _run_bounded(command, Path(temporary))
    assert result["returncode"] == 0
    assert result["stdout"]["truncated"] and result["stderr"]["truncated"]
    assert result["stdout"]["captured_bytes"] == MAX_CAPTURE_BYTES
    assert result["stderr"]["captured_bytes"] == MAX_CAPTURE_BYTES
    checks += 1
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        checks = self_test()
        print("METRIC recovery_v4_one_shot_launcher_self_tests=%d" % checks)
        print("STATUS PASS")
        return 0
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / HANDLER_RELATIVE
    if (
        source != expected
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or not root.is_dir()
    ):
        print("REFUSAL recovery-v4 one-shot launch: handler path is indirect")
        return 3
    try:
        return run_handler(root)
    except (OSError, ValueError) as error:
        print("REFUSAL recovery-v4 one-shot launch: %s" % error)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
