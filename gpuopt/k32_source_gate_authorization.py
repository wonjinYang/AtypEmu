"""Fail-closed external authorization consumption for the K32 source gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

AUTHORIZATION_CONTRACT = "k32_nested_k8_source_gate_authorization_v1"
CONSUMED_CONTRACT = "k32_nested_k8_source_gate_consumed_v1"
CLAIM_CONTRACT = "k32_nested_k8_source_gate_external_claim_v1"


@dataclass(frozen=True)
class AuthorizationSpec:
    source_commitment_sha256: str
    epochs: int
    steps: int
    container_image_sha256: str
    slurm_job_id: str
    authorization_ref: str
    authorization_git_blob: str


@dataclass(frozen=True)
class ConsumedAuthorization:
    source_commitment_sha256: str
    authorization_sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _git_blob(payload: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - this is a Git object identity
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _require_hex(value: str, length: int, field: str) -> None:
    if len(value) != length:
        raise ValueError(f"invalid {field}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"invalid {field}") from error


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def consume_authorization(
    authorization_path: Path,
    source_commitment_path: Path,
    consumed_path: Path,
    external_claim_path: Path,
    *,
    spec: AuthorizationSpec,
    authorization_git_dir: Path,
) -> ConsumedAuthorization:
    """Consume an immutable Git-bound authorization exactly once."""
    if consumed_path.exists() or external_claim_path.exists():
        raise FileExistsError("K32 source-gate authorization was already consumed")
    _require_hex(spec.source_commitment_sha256, 64, "source commitment hash")
    _require_hex(spec.container_image_sha256, 64, "container image hash")
    _require_hex(spec.authorization_git_blob, 40, "authorization Git blob")
    if not spec.slurm_job_id.isdigit() or spec.epochs <= 0 or spec.steps <= 0:
        raise ValueError("authorization execution parameters are invalid")
    expected_ref = (
        f"refs/atypemu-authorizations/k32-source/job-{spec.slurm_job_id}"
    )
    if spec.authorization_ref != expected_ref:
        raise ValueError("authorization ref and Slurm job differ")
    if sha256(source_commitment_path) != spec.source_commitment_sha256:
        raise ValueError("source commitment identity mismatch")

    expected = {
        "contract": AUTHORIZATION_CONTRACT,
        "source_commitment_sha256": spec.source_commitment_sha256,
        "epochs": spec.epochs,
        "steps": spec.steps,
        "authorized": True,
        "container_image_sha256": spec.container_image_sha256,
        "slurm_job_id": spec.slurm_job_id,
        "authorization_ref": spec.authorization_ref,
    }
    authorization = json.loads(authorization_path.read_text())
    if authorization != expected:
        raise ValueError("authorization schema or value mismatch")
    authorization_bytes = authorization_path.read_bytes()
    authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    if _git_blob(authorization_bytes) != spec.authorization_git_blob:
        raise ValueError("authorization Git blob mismatch")
    resolved = subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "rev-parse",
            f"{spec.authorization_ref}^{{blob}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != spec.authorization_git_blob:
        raise ValueError("external authorization ref is not fresh")

    claim = {
        "contract": CLAIM_CONTRACT,
        "authorization_git_blob": spec.authorization_git_blob,
        "authorization_ref": spec.authorization_ref,
        "authorization_sha256": authorization_sha256,
        "container_image_sha256": spec.container_image_sha256,
        "slurm_job_id": spec.slurm_job_id,
        "source_commitment_sha256": spec.source_commitment_sha256,
    }
    claim_bytes = _json_bytes(claim)
    claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
    claim_blob = _git_blob(claim_bytes)
    consumed = {
        **claim,
        "contract": CONSUMED_CONTRACT,
        "external_claim_git_blob": claim_blob,
        "external_claim_sha256": claim_sha256,
    }
    # This is the first irreversible action. Any later failure leaves a durable
    # no-clobber marker and therefore requires a separately authorized recovery.
    _write_new(consumed_path, _json_bytes(consumed))
    _write_new(external_claim_path, claim_bytes)
    written_blob = subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "hash-object",
            "-w",
            str(external_claim_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if written_blob != claim_blob:
        raise ValueError("external claim Git write mismatch")
    subprocess.run(
        [
            "git",
            f"--git-dir={authorization_git_dir}",
            "update-ref",
            spec.authorization_ref,
            claim_blob,
            spec.authorization_git_blob,
        ],
        check=True,
    )
    return ConsumedAuthorization(spec.source_commitment_sha256, authorization_sha256)
