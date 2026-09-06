# ruff: noqa: TRY004
"""Stage and later launch the HOLD support-1 preflight, fail-closed until frozen.

No invocation is currently possible: the plan deliberately records an absent source
commitment.  The static self-test does not inspect frozen cohort data or OpenMM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

CANDIDATE_ID = (
    "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_v1"
)
SCRIPT_RELATIVE = Path(
    "gpuopt/candidates/launch_cohort_support1_protonation_preflight.py"
)
PLAN_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_plan_v1.json"
)
GENERATOR_RELATIVE = Path("gpuopt/candidates/cohort_support1_protonation_preflight.py")
CHECKER_RELATIVE = Path(
    "gpuopt/candidates/check_cohort_support1_protonation_preflight.py"
)
COMMITMENT_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_source_commitment_v5.json"
)
V1_FAILURE_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_source_commitment_v1_failure_receipt.json"
)
V1_FAILURE_SHA256 = "a3472da3e9c7b7ae69a4233ce1d387db5b127b4192983ab9662a5f770f358037"
V2_FAILURE_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_launch_v2_failure_receipt.json"
)
V2_FAILURE_SHA256 = "81af40765dc4e70d501e115c0f834d2fed88930822d07ad6e17de82e19d0633b"
V3_FAILURE_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_launch_v3_failure_receipt.json"
)
V3_FAILURE_SHA256 = "0b9cbc316eea4dcf8612060a1091b4994a90b9e5858954e2016f3fc77117707b"
V4_FAILURE_RELATIVE = Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_launch_v4_failure_receipt.json"
)
V4_FAILURE_SHA256 = "de53a1880a9f1ab6c20615b9c026ea62c8dafc6e6799e84eb449f87502c3e6e3"
STAGE_RELATIVE = Path(
    ".auto/staging/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_stage_v5"
)
OUTPUT_RELATIVE = Path(
    ".auto/atypemu_nested_support_count_v1_cohort_support1_"
    "protonation_preflight_output_v4"
)
INPUTS = {
    "environment.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_openmm86_environment_manifest_v1.json"
        ),
        "6a5f3ff4a041d0fc055ee0a3b855ef2715a826812d11d6b2fdf1744a56fecd88",
    ),
    "condition.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_manifest_v1.json"
        ),
        "ac51d7a40259f3a61e5fec0b521964d85a86d54aa2b91b78d051f9a0cca22618",
    ),
    "sequence.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_protein_sequence_manifest_v1.json"
        ),
        "4faf799877e0387a90c9f00c641e958bd02585dde6b1b99bbdcaeb9f2e82012b",
    ),
    "parent.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_parent_heavy_coordinate_manifest_v1.json"
        ),
        "77d52e663e80e55d178ffa7994596292f1753ffe4a0b4e5fd75005f826a119b3",
    ),
    "input_check_receipt.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_inputs_check_receipt_v1.json"
        ),
        "3a95bb683385079a46d921b0e17aa56cd684578b25b34ee13e9e1fd541356f8e",
    ),
    "smoke_evidence.json": (
        Path(
            "gpuopt/preunblind/atypemu_nested_support_count_v1_condition_uncertainty_protonation_smoke_evidence_v3.json"
        ),
        "57bc3939e0deffdf4dce16eb5f3082089c78230733ceb05b64122a2a38e7163a",
    ),
}
RUNTIME_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
SHA = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=reject_dupes, parse_constant=reject_constant
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def entity_token(uid: str) -> str:
    if not isinstance(uid, str) or not uid or "\x00" in uid:
        raise ValueError("invalid entity UID")
    return (
        f"{SLUG.sub('-', uid).strip('.-')[:48] or 'entity'}-{digest(uid.encode())[:20]}"
    )


def checked_relative(relative: Path) -> Path:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe relative path: {relative}")
    return relative


def repo_regular(root: Path, relative: Path, limit: int = 5_000_000) -> bytes:
    relative = checked_relative(relative)
    current = root
    for part in relative.parts:
        current /= part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink rejected: {relative}")
    fd = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError(f"not a bounded regular file: {relative}")
        blocks: list[bytes] = []
        total = 0
        while block := os.read(fd, 1_048_576):
            total += len(block)
            if total > limit:
                raise ValueError(f"file exceeds limit: {relative}")
            blocks.append(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"file changed while reading: {relative}")
        return b"".join(blocks)
    finally:
        os.close(fd)


def absolute_regular(path: Path, limit: int = 1_000_000_000) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("runtime must be an absolute direct file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("invalid runtime SIF")
        blocks: list[bytes] = []
        total = 0
        while block := os.read(fd, 1_048_576):
            total += len(block)
            if total > limit:
                raise ValueError("runtime SIF too large")
            blocks.append(block)
        return b"".join(blocks)
    finally:
        os.close(fd)


def exclusive(path: Path, raw: bytes, mode: int = 0o444) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.chmod(path, mode)


def exact_keys(value: dict[str, Any], names: set[str], label: str) -> None:
    if set(value) != names:
        raise ValueError(f"{label} exact schema mismatch")


def hash_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"invalid SHA256 in {label}")
    return value


def validate_plan(plan: dict[str, Any]) -> None:
    exact_keys(
        plan,
        {
            "artifact_kind",
            "authorization",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "frozen_inputs",
            "limitations",
            "openmm",
            "output_contract",
            "physicality_scope",
            "roster",
            "run_contract",
            "source_commitment",
            "state",
            "study_id",
        },
        "plan",
    )
    if not (
        plan["candidate_id"] == CANDIDATE_ID
        and isinstance(plan["state"], str)
        and plan["roster"]["entity_count"] == 135
        and plan["roster"]["observed_entity_count"] == 119
        and plan["roster"]["unresolved_entity_count"] == 16
        and plan["roster"]["state_count"] == 199
        and plan["physicality_scope"]["representative_support_index"] == 1
        and plan["openmm"]
        == {
            "add_hydrogens_call": "Modeller.addHydrogens(forcefield, pH=branch_ph, variants=None, platform=Reference)",
            "container_sif_sha256": RUNTIME_SHA256,
            "forcefield": "amber14/protein.ff14SB.xml",
            "openmm_exact_version": "8.6.0.dev-c6173db",
            "platform": "Reference",
        }
        and plan["source_commitment"].get("canonical_relative_path")
        == COMMITMENT_RELATIVE.as_posix()
        and plan["source_commitment"].get("prior_v1_failure_receipt")
        == {
            "path": V1_FAILURE_RELATIVE.as_posix(),
            "sha256": V1_FAILURE_SHA256,
        }
        and plan["source_commitment"].get("prior_v2_launch_failure_receipt")
        == {
            "path": V2_FAILURE_RELATIVE.as_posix(),
            "sha256": V2_FAILURE_SHA256,
        }
        and plan["source_commitment"].get("prior_v3_launch_failure_receipt")
        == {
            "path": V3_FAILURE_RELATIVE.as_posix(),
            "sha256": V3_FAILURE_SHA256,
        }
        and plan["source_commitment"].get("prior_v4_launch_failure_receipt")
        == {
            "path": V4_FAILURE_RELATIVE.as_posix(),
            "sha256": V4_FAILURE_SHA256,
        }
        and plan["output_contract"].get("canonical_stage_path")
        == STAGE_RELATIVE.as_posix()
        and plan["output_contract"].get("canonical_output_path")
        == OUTPUT_RELATIVE.as_posix()
        and plan["source_commitment"].get("status")
        in {"ABSENT_UNFROZEN", "FROZEN_COMMITTED"}
        and not any(plan["closed_capabilities"].values())
    ):
        raise ValueError("HOLD plan contract drifted")
    if (
        plan["source_commitment"]["status"] == "ABSENT_UNFROZEN"
        and plan["state"] != "HOLD_PREFLIGHT_SOURCE_UNFROZEN_UNRUN"
    ):
        raise ValueError("unfrozen source commitment escaped its HOLD state")
    bindings = {
        "environment_manifest": INPUTS["environment.json"],
        "condition_manifest": INPUTS["condition.json"],
        "protein_sequence_manifest": INPUTS["sequence.json"],
        "parent_heavy_coordinate_manifest": INPUTS["parent.json"],
        "input_check_receipt": INPUTS["input_check_receipt.json"],
        "smoke_evidence": INPUTS["smoke_evidence.json"],
    }
    for name, (relative, expected) in bindings.items():
        row = plan["frozen_inputs"].get(name)
        if row != {"path": relative.as_posix(), "sha256": expected}:
            raise ValueError(f"plan frozen binding mismatch: {name}")


def canonical_root() -> Path:
    actual = Path(__file__).absolute()
    root = actual.parents[2]
    expected = root / SCRIPT_RELATIVE
    if (
        actual != expected
        or actual.is_symlink()
        or actual.resolve(strict=True) != actual
    ):
        raise ValueError("launcher must run at its canonical repository path")
    return root


def require_frozen_commitment(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    source = plan["source_commitment"]
    # This is intentionally first: no manifest, support coordinate, runtime, or
    # evidence is opened while the source commitment is absent.
    if source["status"] != "FROZEN_COMMITTED":
        raise PermissionError("HOLD: canonical source commitment is ABSENT_UNFROZEN")
    raw = repo_regular(root, Path(source["canonical_relative_path"]))
    commitment = json_object(raw, "source commitment")
    exact_keys(
        commitment,
        {"candidate_id", "contract", "files", "git_commit", "status"},
        "source commitment",
    )
    if (
        commitment["candidate_id"] != CANDIDATE_ID
        or commitment["contract"]
        != "atypemu_nested_support_count_v1_cohort_support1_protonation_preflight_source_commitment_v5"
        or commitment["status"] != "FROZEN_COMMITTED"
        or not isinstance(commitment["git_commit"], str)
        or GIT_SHA.fullmatch(commitment["git_commit"]) is None
    ):
        raise ValueError("source commitment identity mismatch")
    if digest(repo_regular(root, V1_FAILURE_RELATIVE)) != V1_FAILURE_SHA256:
        raise ValueError("v1 pre-data failure receipt binding mismatch")
    if digest(repo_regular(root, V2_FAILURE_RELATIVE)) != V2_FAILURE_SHA256:
        raise ValueError("v2 pre-container failure receipt binding mismatch")
    if digest(repo_regular(root, V3_FAILURE_RELATIVE)) != V3_FAILURE_SHA256:
        raise ValueError("v3 pre-generation failure receipt binding mismatch")
    if digest(repo_regular(root, V4_FAILURE_RELATIVE)) != V4_FAILURE_SHA256:
        raise ValueError("v4 first-worker failure receipt binding mismatch")
    files = commitment["files"]
    expected_paths = {
        PLAN_RELATIVE.as_posix(),
        GENERATOR_RELATIVE.as_posix(),
        CHECKER_RELATIVE.as_posix(),
        SCRIPT_RELATIVE.as_posix(),
    }
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise ValueError("source commitment file inventory mismatch")
    for name, expected in files.items():
        current = repo_regular(root, Path(name))
        if hash_text(expected, name) != digest(current):
            raise ValueError(f"source commitment raw binding mismatch: {name}")
        committed = subprocess.run(
            ["git", "show", f"{commitment['git_commit']}:{name}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        if committed != current:
            raise ValueError(f"source differs from committed Git object: {name}")
    return commitment


def require_clean_committed_tree(root: Path, commitment: dict[str, Any]) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise ValueError("repository is not a clean committed tree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if GIT_SHA.fullmatch(head) is None:
        raise ValueError("invalid current Git HEAD")
    current_commitment = repo_regular(root, COMMITMENT_RELATIVE)
    committed_commitment = subprocess.run(
        ["git", "show", f"HEAD:{COMMITMENT_RELATIVE.as_posix()}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if committed_commitment != current_commitment:
        raise ValueError("current Git HEAD does not bind source commitment")
    for relative in commitment["files"]:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=True,
            capture_output=True,
        )


def staged_pdb_records(raw: bytes) -> tuple[str, str]:
    """Return canonical parsed heavy topology/coordinate hashes for staging only."""
    topology = hashlib.sha256()
    heavy_count = 0
    coordinates: list[dict[str, str]] = []
    identities: set[tuple[str, ...]] = set()
    segment = 0
    for number, line in enumerate(raw.splitlines(), 1):
        if line.startswith(b"TER"):
            segment += 1
            continue
        if line[:6] not in {b"ATOM  ", b"HETATM"}:
            continue
        if line[:6] != b"ATOM  " or len(line) < 78 or line[16:17] != b" ":
            raise ValueError(f"unsupported support PDB record {number}")
        try:
            row = {
                "atom_name": line[12:16].decode("ascii").strip(),
                "chain_id": line[21:22].decode("ascii"),
                "element": line[76:78].decode("ascii").strip().upper(),
                "insertion_code": line[26:27].decode("ascii"),
                "residue_id": line[22:26].decode("ascii").strip(),
                "residue_name": line[17:20].decode("ascii").strip(),
                "x": format(Decimal(line[30:38].decode("ascii").strip()), ".3f"),
                "y": format(Decimal(line[38:46].decode("ascii").strip()), ".3f"),
                "z": format(Decimal(line[46:54].decode("ascii").strip()), ".3f"),
            }
        except (InvalidOperation, UnicodeDecodeError) as error:
            raise ValueError(f"malformed support PDB record {number}") from error
        if not row["atom_name"] or not row["element"]:
            raise ValueError("missing support PDB identity")
        identity = tuple(
            row[name]
            for name in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        if identity in identities:
            raise ValueError("duplicate support PDB atom")
        identities.add(identity)
        if row["element"] not in {"H", "D", "T"}:
            catalog_identity = (
                line[:6].decode("ascii").strip(),
                row["atom_name"].upper(),
                row["residue_name"].upper(),
                segment,
                row["chain_id"].strip() or "_",
                int(row["residue_id"]),
                row["insertion_code"].strip(),
                row["element"],
            )
            topology.update(
                json.dumps(catalog_identity, separators=(",", ":")).encode()
            )
            topology.update(b"\n")
            heavy_count += 1
            coordinates.append(row)
    if not heavy_count:
        raise ValueError("support PDB has no heavy atoms")
    return topology.hexdigest(), digest(canonical(coordinates))


def stage(root: Path, stage_dir: Path, runtime_sif: Path) -> None:
    if stage_dir.exists():
        raise FileExistsError("stage directory already exists; no clobber")
    os.mkdir(stage_dir, 0o700)
    inputs = stage_dir / "inputs"
    scripts = stage_dir / "scripts"
    supports = inputs / "support1"
    inputs.mkdir(0o700)
    scripts.mkdir(0o700)
    supports.mkdir(0o700)
    copied: dict[str, bytes] = {}
    for stage_name, (relative, expected) in INPUTS.items():
        raw = repo_regular(root, relative)
        if digest(raw) != expected:
            raise ValueError(f"bound input hash mismatch: {relative}")
        json_object(
            raw, stage_name
        )  # duplicate/non-finite rejection for every JSON input.
        exclusive(inputs / stage_name, raw)
        copied[stage_name] = raw
    sequence = json_object(copied["sequence.json"], "sequence manifest")
    exact_keys(
        sequence,
        {
            "artifact_kind",
            "candidate_id",
            "closed_capabilities",
            "contract",
            "entities",
            "entity_count",
            "freezer",
            "roster",
            "scope",
        },
        "sequence manifest",
    )
    if sequence["entity_count"] != 135 or len(sequence["entities"]) != 135:
        raise ValueError("sequence manifest count mismatch")
    inventory = []
    seen: set[str] = set()
    for row in sequence["entities"]:
        exact_keys(
            row,
            {
                "bmrb_id",
                "entity_uid",
                "reference_pdb",
                "residue_count",
                "residues",
                "sequence_one_letter",
                "sequence_sha256",
            },
            "sequence entity",
        )
        uid, bmrb, reference = row["entity_uid"], row["bmrb_id"], row["reference_pdb"]
        if (
            not isinstance(uid, str)
            or uid in seen
            or not isinstance(bmrb, str)
            or not isinstance(reference, dict)
        ):
            raise ValueError("sequence entity identity mismatch")
        seen.add(uid)
        exact_keys(reference, {"path", "sha256"}, "reference PDB")
        expected_path = (
            Path("data/k32_complete_coordinate_supports_v4")
            / bmrb
            / f"{bmrb}_BioEmu_1.pdb"
        )
        if reference["path"] != expected_path.as_posix():
            raise ValueError("sequence reference is not canonical support-index-1 path")
        source_pdb = repo_regular(root, expected_path)
        if digest(source_pdb) != hash_text(
            reference["sha256"], "sequence reference PDB"
        ):
            raise ValueError("canonical support-1 raw PDB hash mismatch")
        heavy_topology, heavy_coordinates = staged_pdb_records(source_pdb)
        destination = f"support1/{entity_token(uid)}.pdb"
        exclusive(inputs / destination, source_pdb)
        inventory.append(
            {
                "entity_uid": uid,
                "heavy_coordinate_sha256": heavy_coordinates,
                "heavy_topology_sha256": heavy_topology,
                "path": destination,
                "pdb_sha256": digest(source_pdb),
            }
        )
    if len(inventory) != 135:
        raise ValueError("support staging quota mismatch")
    exclusive(
        inputs / "support1_inventory.json",
        canonical(
            {
                "contract": "atypemu_nested_support_count_v1_cohort_support1_inventory_v1",
                "entities": sorted(inventory, key=lambda item: item["entity_uid"]),
                "entity_count": 135,
                "support_index": 1,
            }
        )
        + b"\n",
    )
    for relative in (
        PLAN_RELATIVE,
        GENERATOR_RELATIVE,
        CHECKER_RELATIVE,
        SCRIPT_RELATIVE,
        COMMITMENT_RELATIVE,
    ):
        exclusive(scripts / relative.name, repo_regular(root, relative))
    runtime_raw = absolute_regular(runtime_sif)
    if digest(runtime_raw) != RUNTIME_SHA256:
        raise ValueError("frozen SIF hash mismatch")
    exclusive(stage_dir / "runtime.sif", runtime_raw)
    for directory in (supports, inputs, scripts):
        os.chmod(directory, 0o555)
    os.chmod(stage_dir / "runtime.sif", 0o444)
    os.chmod(stage_dir, 0o555)


def launch(root: Path, stage_dir: Path, output: Path, runtime_sif: Path) -> None:
    expected_stage = (root / STAGE_RELATIVE).absolute()
    expected_output = (root / OUTPUT_RELATIVE).absolute()
    if stage_dir.absolute() != expected_stage or output.absolute() != expected_output:
        raise ValueError("launcher paths differ from the frozen recovery-v3 namespace")
    stage_dir, output = expected_stage, expected_output
    plan = json_object(repo_regular(root, PLAN_RELATIVE), "plan")
    validate_plan(plan)
    commitment = require_frozen_commitment(root, plan)
    require_clean_committed_tree(root, commitment)
    if output.exists():
        raise FileExistsError("output already exists; no clobber")
    stage(root, stage_dir, runtime_sif)
    os.mkdir(output, 0o700)
    common = [
        "singularity",
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{stage_dir}:/stage:ro",
        "--bind",
        f"{output}:/output",
        str(stage_dir / "runtime.sif"),
        "python",
    ]
    subprocess.run(
        common
        + [
            "/stage/scripts/cohort_support1_protonation_preflight.py",
            "--inputs",
            "/stage/inputs",
            "--output",
            "/output/generation",
        ],
        cwd="/tmp",
        check=True,
    )
    subprocess.run(
        common
        + [
            "/stage/scripts/check_cohort_support1_protonation_preflight.py",
            "--inputs",
            "/stage/inputs",
            "--generated",
            "/output/generation",
            "--output",
            "/output/checker",
        ],
        cwd="/tmp",
        check=True,
    )
    os.chmod(output, 0o555)


def self_test() -> int:
    assert entity_token("same/slash") != entity_token("same:slash")
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"):
        try:
            json_object(raw, "self-test")
        except ValueError:
            pass
        else:
            raise AssertionError("bad JSON was accepted")
    plan = {"source_commitment": {"status": "ABSENT_UNFROZEN"}}
    try:
        # No root access occurs before this HOLD refusal.
        if plan["source_commitment"]["status"] != "FROZEN_COMMITTED":
            raise PermissionError("HOLD")
    except PermissionError:
        pass
    else:
        raise AssertionError("absent commitment did not fail closed")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "real").mkdir()
        (root / "real" / "x").write_bytes(b"x")
        (root / "indirect").symlink_to(root / "real", target_is_directory=True)
        try:
            repo_regular(root, Path("indirect/x"))
        except ValueError:
            pass
        else:
            raise AssertionError("symlink source accepted")
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-sif", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(f"STATUS PASS_STATIC_LAUNCHER_SELF_TEST checks={self_test()}")
        return 0
    if args.stage_dir is None or args.output is None or args.runtime_sif is None:
        parser.error("--stage-dir, --output, and --runtime-sif are required")
    launch(canonical_root(), args.stage_dir, args.output, args.runtime_sif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
