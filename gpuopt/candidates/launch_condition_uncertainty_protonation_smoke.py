#!/usr/bin/env python3
"""Launch the bounded protonation smoke with only its target-free inputs mounted."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Tuple


CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_v1"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_"
    "source_commitment_v1"
)
SIF = Path(
    "/home/yang07/.cache/atypemu_openmm86_runtime/openmm86_protonation_8.6.0.sif"
)
SIF_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
OUTPUT_RELATIVE = Path(
    ".auto/staging/condition_uncertainty_protonation_smoke_execution_v1"
)
SOURCES = {
    "checker": Path(
        "gpuopt/candidates/check_condition_uncertainty_protonation_smoke.py"
    ),
    "generator": Path("gpuopt/candidates/condition_uncertainty_protonation_smoke.py"),
    "launcher": Path(
        "gpuopt/candidates/launch_condition_uncertainty_protonation_smoke.py"
    ),
    "plan": Path(
        "gpuopt/preunblind/"
        "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_plan_v1.json"
    ),
}
COMMITMENT_RELATIVE = Path(
    "gpuopt/preunblind/"
    "atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_"
    "source_commitment_v1.json"
)
PARENTS = {
    "bmr10109_BioEmu_1.pdb": Path(
        "data/k32_complete_coordinate_supports_v4/bmr10109/bmr10109_BioEmu_1.pdb"
    ),
    "bmr4333_BioEmu_1.pdb": Path(
        "data/k32_complete_coordinate_supports_v4/bmr4333/bmr4333_BioEmu_1.pdb"
    ),
}
CLOSED_FIELDS = frozenset(
    (
        "authorization_consumed",
        "outer_or_formal_metrics_opened",
        "science_executed",
        "source_construction_executed",
        "source_scores_read",
        "target_atom_identities_read",
        "target_values_read",
    )
)


class LaunchError(ValueError):
    """Fail-closed launch error."""


def _pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise LaunchError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, limit: int) -> bytes:
    absolute = path.absolute()
    if absolute.resolve(strict=True) != absolute:
        raise LaunchError("source path or ancestor is indirect: %s" % path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(absolute), flags)
    with os.fdopen(descriptor, "rb") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise LaunchError("source is nonregular: %s" % path)
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise LaunchError("oversize source: %s" % path)
    return raw


def _schema(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        raise LaunchError("%s schema drifted" % label)


def _root() -> Path:
    source = Path(__file__).absolute()
    root = source.parents[2]
    expected = root / SOURCES["launcher"]
    if source.is_symlink() or source.resolve(strict=True) != expected.resolve(
        strict=True
    ):
        raise LaunchError("launcher path is indirect or misplaced")
    return root


def _commitment(root: Path) -> Tuple[Dict[str, Any], bytes]:
    raw = _read(root / COMMITMENT_RELATIVE, 100_000)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, LaunchError) as error:
        raise LaunchError("invalid source commitment: %s" % error) from error
    _schema(
        value,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "runtime_sif_sha256",
            "scope",
            "source_git_commit",
            "sources",
            "state",
        },
        "commitment",
    )
    if (
        value["artifact_kind"]
        != "hold_only_target_unread_bounded_smoke_source_commitment"
        or value["candidate_id"] != CANDIDATE_ID
        or value["contract"] != COMMITMENT_CONTRACT
        or value["runtime_sif_sha256"] != SIF_SHA256
        or value["state"] != "HOLD_SMOKE_SOURCE_FROZEN_UNRUN"
        or set(value["closed_capabilities"]) != CLOSED_FIELDS
        or any(
            value["closed_capabilities"][field] is not False for field in CLOSED_FIELDS
        )
        or set(value["sources"]) != set(SOURCES)
    ):
        raise LaunchError("source commitment semantics drifted")
    commit = value["source_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(ch not in "0123456789abcdef" for ch in commit)
    ):
        raise LaunchError("invalid source Git commit")
    for name, relative in SOURCES.items():
        record = value["sources"][name]
        _schema(record, {"repository_path", "sha256"}, "%s source" % name)
        current = _read(root / relative, 1_000_000)
        if record != {"repository_path": relative.as_posix(), "sha256": _sha(current)}:
            raise LaunchError("current %s source differs from commitment" % name)
        try:
            committed = subprocess.run(
                ["git", "show", "%s:%s" % (commit, relative.as_posix())],
                cwd=str(root),
                check=True,
                capture_output=True,
                timeout=20,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise LaunchError(
                "cannot replay committed %s source: %s" % (name, error)
            ) from error
        if committed != current:
            raise LaunchError("%s source differs from source Git commit" % name)
    if _canonical(value) != raw:
        raise LaunchError("source commitment is not canonical JSON")
    return value, raw


def _clean_head(root: Path, commitment_raw: bytes) -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        committed_commitment = subprocess.run(
            ["git", "show", "HEAD:%s" % COMMITMENT_RELATIVE.as_posix()],
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise LaunchError("clean Git HEAD check failed: %s" % error) from error
    if status:
        raise LaunchError("launch requires a clean worktree")
    if committed_commitment != commitment_raw:
        raise LaunchError("executed HEAD does not bind the source commitment")
    return head


def _write(path: Path, raw: bytes) -> None:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_inputs(root: Path, output: Path) -> Path:
    sealed = output / "sealed_inputs"
    sealed.mkdir(mode=0o700)
    for name, relative in SOURCES.items():
        _write(
            sealed / (name + Path(relative).suffix), _read(root / relative, 1_000_000)
        )
    _write(
        sealed / "source_commitment.json", _read(root / COMMITMENT_RELATIVE, 100_000)
    )
    for name, relative in PARENTS.items():
        _write(sealed / name, _read(root / relative, 2_000_000))
    _write(sealed / "runtime.sif", _read(SIF, 1_000_000_000))
    sealed.chmod(0o500)
    return sealed


def _command(sealed: Path, output: Path, program: str) -> list:
    singularity = shutil.which("singularity")
    if singularity != "/usr/bin/singularity":
        raise LaunchError("exact /usr/bin/singularity is unavailable")
    binds = [
        "%s:/work/generator.py:ro" % (sealed / "generator.py"),
        "%s:/work/checker.py:ro" % (sealed / "checker.py"),
        "%s:/work/launcher.py:ro" % (sealed / "launcher.py"),
        "%s:/work/plan.json:ro" % (sealed / "plan.json"),
        "%s:/work/source_commitment.json:ro" % (sealed / "source_commitment.json"),
        "%s:/inputs/bmr10109_BioEmu_1.pdb:ro" % (sealed / "bmr10109_BioEmu_1.pdb"),
        "%s:/inputs/bmr4333_BioEmu_1.pdb:ro" % (sealed / "bmr4333_BioEmu_1.pdb"),
        "%s:/out:rw" % output,
    ]
    tail = (
        ["python", "/work/generator.py"]
        if program == "generator"
        else ["python", "/work/checker.py", "--acknowledge-hold-only"]
    )
    return [
        singularity,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-mount",
        "home,cwd",
        "--net",
        "--network",
        "none",
        "--pwd",
        "/work",
        "--bind",
        ",".join(binds),
        str(sealed / "runtime.sif"),
    ] + tail


def _tree(root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "sha256": _sha(_read(path, 1_000_000_000)),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    root = _root()
    commitment, commitment_raw = _commitment(root)
    head = _clean_head(root, commitment_raw)
    if _sha(_read(SIF, 1_000_000_000)) != SIF_SHA256:
        raise LaunchError("runtime SIF hash drifted")
    for relative in PARENTS.values():
        _read(root / relative, 2_000_000)
    output = root / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise LaunchError("no-clobber output namespace already exists")
    output.mkdir(mode=0o700, parents=False)
    results = output / "results"
    results.mkdir(mode=0o700)
    runs = []
    try:
        sealed = _stage_inputs(root, output)
        for name in ("generator", "checker"):
            command = _command(sealed, results, name)
            completed = subprocess.run(
                command,
                cwd="/",
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            _write(output / (name + ".stdout"), completed.stdout.encode("utf-8"))
            _write(output / (name + ".stderr"), completed.stderr.encode("utf-8"))
            runs.append(
                {
                    "argv": command,
                    "returncode": completed.returncode,
                    "stderr_sha256": _sha(completed.stderr.encode("utf-8")),
                    "stdout_sha256": _sha(completed.stdout.encode("utf-8")),
                }
            )
            if completed.returncode != 0:
                raise LaunchError(
                    "%s failed with return code %d" % (name, completed.returncode)
                )
    except Exception as error:
        failure = {
            "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_failure_receipt",
            "candidate_id": CANDIDATE_ID,
            "closed_capabilities": {field: False for field in sorted(CLOSED_FIELDS)},
            "error": "%s: %s" % (type(error).__name__, error),
            "execution_git_head": head,
            "output_tree": _tree(output),
            "runs": runs,
            "source_commitment_sha256": _sha(commitment_raw),
            "status": "HOLD_BOUNDED_PROTONATION_SMOKE_FAILED_CLOSED",
        }
        _write(output / "failure_receipt.json", _canonical(failure))
        raise
    receipt = {
        "artifact_kind": "hold_only_target_unread_bounded_protonation_smoke_launch_receipt",
        "candidate_id": CANDIDATE_ID,
        "closed_capabilities": {field: False for field in sorted(CLOSED_FIELDS)},
        "execution_git_head": head,
        "output_tree": _tree(output),
        "runs": runs,
        "source_commitment_sha256": _sha(commitment_raw),
        "source_git_commit": commitment["source_git_commit"],
        "status": "HOLD_BOUNDED_PROTONATION_SMOKE_PASSED_NOT_ALL_SUPPORT_EVIDENCE",
    }
    _write(output / "launch_receipt.json", _canonical(receipt))
    print("METRIC condition_uncertainty_protonation_smoke_launches=2")
    print("STATUS HOLD_BOUNDED_PROTONATION_SMOKE_PASSED_NOT_ALL_SUPPORT_EVIDENCE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LaunchError, OSError, subprocess.SubprocessError) as error:
        print("ERROR %s" % error, file=sys.stderr)
        raise SystemExit(2)
