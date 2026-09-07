# ruff: noqa: TRY004
"""Independently replay one frozen ff15ipq all-support entity archive.

This checker does not import the producer.  Production paths are derived from
the staged checker location; no caller-selected input or output path exists.
The self-test is OpenMM-free and reads no cohort input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import stat
import tarfile
from decimal import Decimal, InvalidOperation
from itertools import pairwise, product
from pathlib import Path
from typing import Any

CANDIDATE_ID = "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_v1"
COMMITMENT_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1.json"
)
COMMITMENT_CONTRACT = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_source_commitment_v1"
)
PLAN_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_qualification_plan_v1.json"
)
PROJECTION_NAME = (
    "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1.json.gz"
)
PROJECTION_SHA256 = "e9b4861216d99e8136df872958a50f568337fc8a6b4991b25df2b906dcee5046"
OPENMM_VERSION = "8.6.0.dev-c6173db"
FORCEFIELD_PATH = "amber14/protein.ff15ipq.xml"
RUNTIME_SIF_SHA256 = "a9f2df1d1f5fb1039af8ac791b15f4bfbbd62237dbd923ec4695114ec5d18bc5"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
STATE_FIELDS = {
    "bmrb_id",
    "condition_branch_id",
    "condition_state",
    "entity_uid",
    "parent_raw_pdb_sha256",
    "proposal_pH",
    "support_index",
}
METADATA_FIELDS = {
    "atom_count",
    "condition_branch_id",
    "entity_uid",
    "hydrogen_count",
    "parent_heavy_coordinate_sha256",
    "parent_heavy_topology_sha256",
    "parent_raw_pdb_sha256",
    "potential_energy_kj_per_mol",
    "proposal_pH",
    "protonated_gzip_sha256",
    "protonated_raw_pdb_sha256",
    "protonation_signature_sha256",
    "returned_variants_sha256",
    "support_index",
}
CLOSED_CAPABILITIES = {
    "authorization_consumed": False,
    "outer_or_formal_metrics_opened": False,
    "science_executed": False,
    "source_construction_executed": False,
    "source_scores_read": False,
    "target_atom_identities_read": False,
    "target_values_read": False,
}
RUNTIME_FILE_HASHES = {
    "data/amber14/protein.ff15ipq.xml": (
        "0085c9dc2818a28501f6074a0bb46b994a9787bf4010ff86417fe014b11ab622"
    ),
    "data/hydrogens.xml": (
        "413096cd3005ca5a638180e9cf623a8f6d574c81acf0e2c9d92b2bd26bb7658d"
    ),
    "modeller.py": "f61e61f1419fcc3c24e7096ab96e10f87f70951085a83941d6040390e8819ca3",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def parse_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def safe_token(entity_uid: str) -> str:
    if not isinstance(entity_uid, str) or not entity_uid or "\x00" in entity_uid:
        raise ValueError("invalid entity_uid")
    return sha256(entity_uid.encode("utf-8"))


def stage_paths() -> tuple[Path, Path, Path, Path]:
    scripts = Path(__file__).resolve().parent
    stage = scripts.parent
    return scripts, stage / "inputs", stage / "outputs", stage / "checks"


def read_regular(path: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"invalid regular file: {path.name}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1_048_576, maximum_bytes + 1)):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"file exceeds size ceiling: {path.name}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"file changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def safe_relative(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("unsafe relative path")
    return path.parts


def read_beneath(root: Path, relative: str, maximum_bytes: int = 8_000_000) -> bytes:
    parts = safe_relative(relative)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor
        )
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise ValueError("invalid projected PDB")
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(file_descriptor, min(1_048_576, maximum_bytes + 1)):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("projected PDB exceeds size ceiling")
                chunks.append(chunk)
            after = os.fstat(file_descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("projected PDB changed while reading")
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)


def require_source(scripts: Path, inputs: Path) -> None:
    commitment_raw = read_regular(scripts / COMMITMENT_NAME, 1_000_000)
    expected_commitment = os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
    if (
        expected_commitment is None
        or SHA256_RE.fullmatch(expected_commitment) is None
        or sha256(commitment_raw) != expected_commitment
        or os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256") != RUNTIME_SIF_SHA256
        or os.environ.get("ATYPEMU_STAGE_ROOT") != str(scripts.parent)
    ):
        raise PermissionError(
            "external source/runtime/stage binding is absent or wrong"
        )
    commitment = parse_object(commitment_raw, "source commitment")
    if set(commitment) != {"candidate_id", "contract", "files", "git_commit", "status"}:
        raise ValueError("source commitment schema drifted")
    if not (
        commitment["candidate_id"] == CANDIDATE_ID
        and commitment["contract"] == COMMITMENT_CONTRACT
        and commitment["status"] == "FROZEN_COMMITTED_BOUNDED_SMOKE_ONLY"
        and isinstance(commitment["git_commit"], str)
        and GIT_COMMIT_RE.fullmatch(commitment["git_commit"])
        and os.environ.get("ATYPEMU_GIT_COMMIT") == commitment["git_commit"]
    ):
        raise PermissionError("all-support source is not frozen")
    expected = {
        "gpuopt/candidates/materialize_ff15ipq_all_support_entity.py": "materialize_ff15ipq_all_support_entity.py",
        "gpuopt/candidates/check_ff15ipq_all_support_entity.py": Path(__file__).name,
        "gpuopt/candidates/ff15ipq_all_support_common.py": "ff15ipq_all_support_common.py",
        "gpuopt/candidates/check_ff15ipq_all_support_qualification_plan.py": "check_ff15ipq_all_support_qualification_plan.py",
        "gpuopt/candidates/launch_ff15ipq_all_support_qualification.py": "launch_ff15ipq_all_support_qualification.py",
        f"gpuopt/preunblind/{PLAN_NAME}": PLAN_NAME,
        f"gpuopt/preunblind/{PROJECTION_NAME}": PROJECTION_NAME,
        "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_all_support_implementation_cold_review_receipt_v1.json": "atypemu_nested_support_count_v1_ff15ipq_all_support_implementation_cold_review_receipt_v1.json",
        "gpuopt/preunblind/atypemu_nested_support_count_v1_ff15ipq_all_support_plan_validation_receipt_v1.json": "atypemu_nested_support_count_v1_ff15ipq_all_support_plan_validation_receipt_v1.json",
    }
    files = commitment["files"]
    if not isinstance(files, dict) or set(files) != set(expected):
        raise ValueError("source commitment inventory drifted")
    for repository_path, staged_name in expected.items():
        root = inputs if staged_name == PROJECTION_NAME else scripts
        raw = read_regular(root / staged_name, 8_000_000)
        wanted = files[repository_path]
        if not isinstance(wanted, str) or sha256(raw) != wanted:
            raise ValueError(f"source commitment hash mismatch: {repository_path}")
    plan = parse_object(read_regular(scripts / PLAN_NAME, 1_000_000), "plan")
    if not (
        plan.get("candidate_id") == CANDIDATE_ID
        and plan.get("state") == "HOLD_EXTERNAL_SOURCE_COMMITMENT_PENDING_UNRUN"
        and plan.get("execution_contract", {}).get("source_commitment")
        == "EXTERNAL_O_EXCL_PENDING"
    ):
        raise PermissionError("plan has not released independent replay")


def require_smoke_release(scripts: Path, entity_uid: str) -> None:
    release_root = scripts.parent / "release"
    release_raw = read_regular(release_root / "execution_release.json", 1_000_000)
    release_hash = sha256(release_raw)
    if os.environ.get("ATYPEMU_EXECUTION_RELEASE_SHA256") != release_hash:
        raise PermissionError("execution release environment binding drifted")
    release = parse_object(release_raw, "execution release")
    if release_raw != canonical_bytes(release) + b"\n" or set(release) != {
        "bioemu_root",
        "candidate_id",
        "closed_capabilities",
        "contract",
        "entities",
        "git_commit",
        "mode",
        "review_receipt_sha256",
        "runtime_sif_sha256",
        "source_commitment_sha256",
        "state",
    }:
        raise PermissionError("execution release schema or framing drifted")
    if not (
        release["candidate_id"] == CANDIDATE_ID
        and release["closed_capabilities"] == CLOSED_CAPABILITIES
        and release["contract"]
        == "atypemu_nested_support_count_v1_ff15ipq_all_support_execution_release_v1"
        and release["entities"] == [entity_uid]
        and release["git_commit"] == os.environ.get("ATYPEMU_GIT_COMMIT")
        and release["mode"] == "BOUNDED_SMOKE"
        and release["runtime_sif_sha256"]
        == os.environ.get("ATYPEMU_RUNTIME_SIF_SHA256")
        and release["source_commitment_sha256"]
        == os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
        and release["state"] == "EXTERNALLY_RELEASED_ONCE"
        and os.environ.get("ATYPEMU_RELEASED_ENTITY_UID") == entity_uid
    ):
        raise PermissionError("execution release identity or bounded scope drifted")
    review_raw = read_regular(release_root / "source_cold_review.json", 1_000_000)
    if sha256(review_raw) != release["review_receipt_sha256"]:
        raise PermissionError("source cold-review receipt hash drifted")
    consumed_raw = read_regular(release_root / "execution_consumed.json", 1_000_000)
    consumed = parse_object(consumed_raw, "execution consumption")
    if consumed_raw != canonical_bytes(consumed) + b"\n" or not (
        consumed.get("candidate_id") == CANDIDATE_ID
        and consumed.get("contract")
        == "atypemu_ff15ipq_all_support_execution_consumption_v1"
        and consumed.get("execution_release_sha256") == release_hash
        and consumed.get("mode") == "BOUNDED_SMOKE"
        and consumed.get("source_commitment_sha256")
        == os.environ.get("ATYPEMU_SOURCE_COMMITMENT_SHA256")
        and consumed.get("state") == "CONSUMED_BEFORE_TARGET_UNREAD_PDB_ACCESS"
    ):
        raise PermissionError("execution consumption binding drifted")


def load_projection(inputs: Path) -> dict[str, Any]:
    raw = read_regular(inputs / PROJECTION_NAME, 8_000_000)
    if sha256(raw) != PROJECTION_SHA256:
        raise ValueError("runtime input projection hash drifted")
    try:
        decoded = gzip.decompress(raw)
    except (EOFError, OSError) as error:
        raise ValueError("runtime input projection gzip is invalid") from error
    value = parse_object(decoded, "runtime input projection")
    if not (
        value.get("candidate_id") == CANDIDATE_ID
        and value.get("contract")
        == "atypemu_nested_support_count_v1_ff15ipq_all_support_input_projection_v1"
        and value.get("source_path_template")
        == "data/BioEmu/{bmrb_id}/{bmrb_id}_BioEmu_{support_index}.pdb"
    ):
        raise ValueError("runtime input projection contract drifted")
    return value


def entity_states(projection: dict[str, Any], entity_uid: str) -> list[dict[str, Any]]:
    entries = projection.get("entries")
    if not isinstance(entries, list):
        raise ValueError("projection entities are invalid")
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("entity_uid") == entity_uid
    ]
    if len(matches) != 1:
        raise ValueError("entity_uid does not select one projection entity")
    entity = matches[0]
    missing = entity.get("missing_support_indices")
    supports = entity.get("supports")
    branches = entity.get("condition_branches")
    if (
        not isinstance(missing, list)
        or any(type(index) is not int for index in missing)
        or missing != sorted(set(missing))
        or not isinstance(supports, list)
        or not isinstance(branches, list)
    ):
        raise ValueError("projection entity inventory drifted")
    output: list[dict[str, Any]] = []
    for support in supports:
        if (
            not isinstance(support, dict)
            or set(support) != {"parent_raw_pdb_sha256", "support_index"}
            or type(support["support_index"]) is not int
            or not 1 <= support["support_index"] <= 1000
            or not isinstance(support["parent_raw_pdb_sha256"], str)
            or SHA256_RE.fullmatch(support["parent_raw_pdb_sha256"]) is None
        ):
            raise ValueError("projection runtime state drifted")
        for branch in branches:
            if (
                not isinstance(branch, dict)
                or set(branch) != {"condition_branch_id", "proposal_pH"}
                or not isinstance(branch["condition_branch_id"], str)
                or not isinstance(branch["proposal_pH"], str)
            ):
                raise ValueError("projection condition branch drifted")
            output.append(
                {
                    "bmrb_id": entity["bmrb_id"],
                    "condition_branch_id": branch["condition_branch_id"],
                    "condition_state": entity["condition_state"],
                    "entity_uid": entity_uid,
                    "parent_raw_pdb_sha256": support["parent_raw_pdb_sha256"],
                    "proposal_pH": branch["proposal_pH"],
                    "support_index": support["support_index"],
                }
            )
    output.sort(key=lambda row: (row["support_index"], row["condition_branch_id"]))
    expected_supports = sorted(set(range(1, 1001)) - set(missing))
    expected_branches = (
        [
            {
                "condition_branch_id": "observed-0",
                "proposal_pH": branches[0]["proposal_pH"],
            }
        ]
        if entity.get("condition_state") == "observed" and len(branches) == 1
        else [
            {"condition_branch_id": f"regime-{index}", "proposal_pH": ph}
            for index, ph in enumerate(("2.2", "5.45", "7.5", "9.25", "12.0"))
        ]
    )
    if (
        branches != expected_branches
        or sorted({row["support_index"] for row in output}) != expected_supports
        or len(output) != len(expected_branches) * len(expected_supports)
        or len({(row["support_index"], row["condition_branch_id"]) for row in output})
        != len(output)
    ):
        raise ValueError("projection runtime state coverage drifted")
    return output


def seed_for(state: dict[str, Any]) -> int:
    material = "\0".join(
        (
            CANDIDATE_ID,
            state["entity_uid"],
            str(state["support_index"]),
            state["condition_branch_id"],
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _octal(field: bytes, label: str) -> int:
    if field[-1:] not in {b"\0", b" "} or any(
        byte not in b" 01234567\0" for byte in field
    ):
        raise ValueError(f"noncanonical tar numeric field: {label}")
    stripped = field.rstrip(b"\0 ").lstrip(b" ")
    return int(stripped or b"0", 8)


def strict_tar_members(archive_raw: bytes) -> list[tuple[str, bytes]]:
    raw = archive_raw
    if not raw or len(raw) % 512 != 0:
        raise ValueError("USTAR block framing is noncanonical")
    members: list[tuple[str, bytes]] = []
    offset = 0
    while offset + 512 <= len(raw):
        header = raw[offset : offset + 512]
        if header == bytes(512):
            if raw[offset:] != bytes(1024):
                raise ValueError("tar must end with exactly two zero blocks")
            break
        stored_checksum = _octal(header[148:156], "checksum")
        checksum_header = header[:148] + b" " * 8 + header[156:]
        if sum(checksum_header) != stored_checksum:
            raise ValueError("tar header checksum mismatch")
        if header[257:263] != b"ustar\0" or header[263:265] != b"00":
            raise ValueError("tar member is not USTAR")
        if header[156:157] not in {b"\0", b"0"}:
            raise ValueError("tar contains a non-regular member")
        if (
            _octal(header[100:108], "mode") != 0o444
            or _octal(header[108:116], "uid") != 0
            or _octal(header[116:124], "gid") != 0
            or _octal(header[136:148], "mtime") != 0
            or header[265:297].rstrip(b"\0")
            or header[297:329].rstrip(b"\0")
            or _octal(header[329:337], "device major") != 0
            or _octal(header[337:345], "device minor") != 0
            or header[157:257] != bytes(100)
            or header[500:512] != bytes(12)
        ):
            raise ValueError("tar member metadata is noncanonical")
        name_raw = header[:100].split(b"\0", 1)[0]
        prefix_raw = header[345:500].split(b"\0", 1)[0]
        try:
            name = (prefix_raw + (b"/" if prefix_raw else b"") + name_raw).decode(
                "ascii"
            )
        except UnicodeDecodeError as error:
            raise ValueError("tar member name is not ASCII") from error
        if "/".join(safe_relative(name)) != name:
            raise ValueError("tar member name is not canonical")
        size = _octal(header[124:136], "size")
        expected_info = tarfile.TarInfo(name)
        expected_info.size = size
        expected_info.mode = 0o444
        expected_info.uid = expected_info.gid = expected_info.mtime = 0
        expected_info.uname = expected_info.gname = ""
        expected_info.type = tarfile.REGTYPE
        if header != expected_info.tobuf(format=tarfile.USTAR_FORMAT):
            raise ValueError("tar header is not byte-canonical USTAR")
        start = offset + 512
        stop = start + size
        if stop > len(raw):
            raise ValueError("tar member is truncated")
        members.append((name, raw[start:stop]))
        offset = start + ((size + 511) // 512) * 512
    else:
        raise ValueError("tar terminal zero blocks are absent")
    names = [name for name, _ in members]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("tar member order or uniqueness drifted")
    return members


def parse_metadata(raw: bytes) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("metadata JSONL framing is invalid")
    output = []
    for line in raw.splitlines(keepends=True):
        value = parse_object(line, "metadata row")
        if line != canonical_bytes(value) + b"\n":
            raise ValueError("metadata JSONL is not canonical")
        output.append(value)
    return output


def bounded_gzip_decompress(raw: bytes, maximum_bytes: int) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            decoded = handle.read(maximum_bytes + 1)
            if len(decoded) > maximum_bytes or handle.read(1):
                raise ValueError("gzip member exceeds decompressed-size ceiling")
            return decoded
    except (EOFError, OSError) as error:
        raise ValueError("gzip member is invalid") from error


def pdb_records(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    segment = 0
    for number, line in enumerate(raw.splitlines(), 1):
        if line.startswith(b"TER"):
            segment += 1
            continue
        if line[:6] not in {b"ATOM  ", b"HETATM"}:
            continue
        if line[:6] != b"ATOM  " or len(line) < 78 or line[16:17] != b" ":
            raise ValueError(f"unsupported PDB record at line {number}")
        try:
            xyz = [
                Decimal(line[start:stop].decode("ascii").strip())
                for start, stop in ((30, 38), (38, 46), (46, 54))
            ]
            row = {
                "_record_type": line[:6].decode("ascii").strip(),
                "_segment": segment,
                "atom_name": line[12:16].decode("ascii").strip(),
                "chain_id": line[21:22].decode("ascii"),
                "element": line[76:78].decode("ascii").strip().upper(),
                "insertion_code": line[26:27].decode("ascii"),
                "residue_id": line[22:26].decode("ascii").strip(),
                "residue_name": line[17:20].decode("ascii").strip(),
                "x": format(xyz[0], ".3f"),
                "y": format(xyz[1], ".3f"),
                "z": format(xyz[2], ".3f"),
            }
        except (UnicodeDecodeError, InvalidOperation) as error:
            raise ValueError(f"invalid PDB record at line {number}") from error
        if (
            not row["atom_name"]
            or not row["element"]
            or not all(x.is_finite() for x in xyz)
        ):
            raise ValueError(f"invalid PDB atom fields at line {number}")
        records.append(row)
    identities = [
        tuple(
            row[key]
            for key in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for row in records
    ]
    if not records or len(identities) != len(set(identities)):
        raise ValueError("PDB atom inventory is empty or duplicate")
    return records


def heavy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in records
        if row["element"] not in {"H", "D", "T"}
    ]


def heavy_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    identity = (
        "atom_name",
        "chain_id",
        "element",
        "insertion_code",
        "residue_id",
        "residue_name",
    )
    return len(left) == len(right) and all(
        all(a[key] == b[key] for key in identity)
        and all(Decimal(a[key]) == Decimal(b[key]) for key in ("x", "y", "z"))
        for a, b in zip(left, right, strict=True)
    )


def heavy_hashes(records: list[dict[str, Any]]) -> tuple[str, str]:
    heavy = [row for row in records if row["element"] not in {"H", "D", "T"}]
    topology = hashlib.sha256()
    for row in heavy:
        identity = (
            row["_record_type"],
            row["atom_name"].upper(),
            row["residue_name"].upper(),
            row["_segment"],
            row["chain_id"].strip() or "_",
            int(row["residue_id"]),
            row["insertion_code"].strip(),
            row["element"],
        )
        topology.update(json.dumps(identity, separators=(",", ":")).encode())
        topology.update(b"\n")
    coordinates = [
        {
            key: row[key]
            for key in (
                "atom_name",
                "chain_id",
                "element",
                "insertion_code",
                "residue_id",
                "residue_name",
                "x",
                "y",
                "z",
            )
        }
        for row in heavy
    ]
    return topology.hexdigest(), sha256(canonical_bytes(coordinates))


def atom_key(atom: Any) -> tuple[str, str, str, str, str, str]:
    residue = atom.residue
    element = atom.element.symbol.upper() if atom.element is not None else ""
    return (
        residue.chain.id,
        str(residue.id),
        residue.insertionCode,
        residue.name,
        atom.name,
        element,
    )


def openmm_heavy_records(
    topology: Any, positions: Any, unit: Any
) -> list[dict[str, Any]]:
    output = []
    for atom, position in zip(topology.atoms(), positions):
        if atom.element is None or atom.element.atomic_number == 1:
            continue
        chain, residue_id, insertion, residue, name, element = atom_key(atom)
        xyz = position.value_in_unit(unit.angstrom)
        output.append(
            {
                "atom_name": name,
                "chain_id": chain,
                "element": element,
                "insertion_code": insertion,
                "residue_id": residue_id,
                "residue_name": residue,
                "x": format(Decimal(str(float(xyz[0]))), ".3f"),
                "y": format(Decimal(str(float(xyz[1]))), ".3f"),
                "z": format(Decimal(str(float(xyz[2]))), ".3f"),
            }
        )
    return output


def distance(left: Any, right: Any, unit: Any) -> float:
    vector = left - right
    return math.sqrt(
        sum(float(value * value) for value in vector.value_in_unit(unit.angstrom))
    )


def exact_minimum_distance(
    coordinates: list[tuple[float, float, float]], upper_bound: float
) -> float:
    if len(coordinates) < 2:
        return math.inf
    if not math.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("nearest-neighbor upper bound is invalid")
    cells: dict[tuple[int, int, int], list[int]] = {}
    minimum_sq = upper_bound * upper_bound
    for index, xyz in enumerate(coordinates):
        cell = tuple(math.floor(value / upper_bound) for value in xyz)
        for delta in product((-2, -1, 0, 1, 2), repeat=3):
            neighbor = tuple(cell[axis] + delta[axis] for axis in range(3))
            for prior_index in cells.get(neighbor, ()):
                prior = coordinates[prior_index]
                candidate = sum((xyz[axis] - prior[axis]) ** 2 for axis in range(3))
                minimum_sq = min(minimum_sq, candidate)
        cells.setdefault(cell, []).append(index)
    return math.sqrt(minimum_sq)


def signature(topology: Any) -> list[dict[str, str]]:
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in topology.atoms()}
    for left, right in topology.bonds():
        adjacency[left].append(right)
        adjacency[right].append(left)
    output = []
    for atom in topology.atoms():
        if atom.element is None or atom.element.atomic_number != 1:
            continue
        parents = [
            other
            for other in adjacency[atom]
            if other.element and other.element.atomic_number != 1
        ]
        if len(parents) != 1:
            raise ValueError("hydrogen signature parent count drifted")
        chain, residue_id, insertion, residue, name, _ = atom_key(atom)
        output.append(
            {
                "chain_id": chain,
                "hydrogen_name": name,
                "heavy_parent": parents[0].name,
                "insertion_code": insertion,
                "residue_id": residue_id,
                "residue_name": residue,
            }
        )
    return sorted(output, key=canonical_bytes)


def physicality(
    modeller: Any, forcefield: Any, platform: Any, unit: Any, openmm: Any
) -> dict[str, Any]:
    atoms, positions = list(modeller.topology.atoms()), list(modeller.positions)
    if not atoms or len(atoms) != len(positions):
        raise ValueError("atom-position inventory drifted")
    coordinates = [
        tuple(float(value) for value in position.value_in_unit(unit.angstrom))
        for position in positions
    ]
    if not all(math.isfinite(value) for xyz in coordinates for value in xyz):
        raise ValueError("non-finite coordinate")
    adjacency: dict[Any, list[Any]] = {atom: [] for atom in atoms}
    atom_index = {atom: index for index, atom in enumerate(atoms)}
    bond_distances = []
    for left, right in modeller.topology.bonds():
        adjacency[left].append(right)
        adjacency[right].append(left)
        current = distance(
            positions[atom_index[left]], positions[atom_index[right]], unit
        )
        bond_distances.append(current)
        left_element = left.element.symbol.upper() if left.element else ""
        right_element = right.element.symbol.upper() if right.element else ""
        if left_element != "H" and right_element != "H" and not 1.0 <= current <= 2.3:
            raise ValueError("heavy-heavy bond distance failed")
    if not bond_distances:
        raise ValueError("topology has no bonds")
    minimum = exact_minimum_distance(coordinates, min(bond_distances))
    if minimum < 0.5:
        raise ValueError("all-distinct-atom distance failed")
    limits = {"H": 1, "C": 4, "N": 4, "O": 2, "S": 6}
    hydrogen_count = 0
    for atom in atoms:
        element = atom.element.symbol.upper() if atom.element else ""
        if element not in limits or len(adjacency[atom]) > limits[element]:
            raise ValueError("element valence failed")
        if element != "H":
            continue
        hydrogen_count += 1
        parents = [
            other
            for other in adjacency[atom]
            if other.element and other.element.atomic_number != 1
        ]
        if len(parents) != 1 or len(adjacency[atom]) != 1:
            raise ValueError("hydrogen parent count failed")
        parent = parents[0]
        length = distance(
            positions[atom_index[atom]], positions[atom_index[parent]], unit
        )
        lower, upper = (
            (1.1, 1.5) if parent.element.symbol.upper() == "S" else (0.7, 1.3)
        )
        if not lower <= length <= upper:
            raise ValueError("hydrogen bond length failed")
        others = [
            other
            for other in adjacency[parent]
            if other.element and other.element.atomic_number != 1
        ]
        if not others:
            raise ValueError("hydrogen parent has no angle reference")
        for other in others:
            a = positions[atom_index[atom]] - positions[atom_index[parent]]
            b = positions[atom_index[other]] - positions[atom_index[parent]]
            av, bv = a.value_in_unit(unit.angstrom), b.value_in_unit(unit.angstrom)
            cosine = sum(float(x * y) for x, y in zip(av, bv, strict=True)) / math.sqrt(
                sum(float(x * x) for x in av) * sum(float(y * y) for y in bv)
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if not 55.0 <= angle <= 180.0:
                raise ValueError("hydrogen bond angle failed")
    if hydrogen_count == 0:
        raise ValueError("no hydrogen was emitted")
    for chain in modeller.topology.chains():
        for left, right in pairwise(list(chain.residues())):
            left_c = next((atom for atom in left.atoms() if atom.name == "C"), None)
            right_n = next((atom for atom in right.atoms() if atom.name == "N"), None)
            if left_c is None or right_n is None or right_n not in adjacency[left_c]:
                raise ValueError("backbone topology continuity failed")
            length = distance(
                positions[atom_index[left_c]], positions[atom_index[right_n]], unit
            )
            if not 1.0 <= length <= 1.8:
                raise ValueError("backbone coordinate continuity failed")
    system = forcefield.createSystem(modeller.topology)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPositions(modeller.positions)
        energy = float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context
        del integrator
    if not math.isfinite(energy):
        raise ValueError("potential energy is non-finite")
    return {
        "atom_count": len(atoms),
        "hydrogen_count": hydrogen_count,
        "minimum_distinct_atom_distance_A": minimum,
        "potential_energy_kj_per_mol": energy,
    }


def require_runtime(openmm: Any, app: Any) -> None:
    if openmm.version.full_version != OPENMM_VERSION:
        raise ValueError("OpenMM exact version drifted")
    root = Path(app.__file__).parent
    for relative, expected in RUNTIME_FILE_HASHES.items():
        if sha256((root / relative).read_bytes()) != expected:
            raise ValueError(f"OpenMM runtime file hash drifted: {relative}")


def state_path(state: dict[str, Any]) -> str:
    bmrb, index = state["bmrb_id"], state["support_index"]
    if (
        not isinstance(bmrb, str)
        or re.fullmatch(r"bmr[0-9]+", bmrb) is None
        or type(index) is not int
    ):
        raise ValueError("state source identity is invalid")
    return f"data/BioEmu/{bmrb}/{bmrb}_BioEmu_{index}.pdb"


def exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def replay_worker(
    task: tuple[str, int, dict[str, Any]],
) -> tuple[bytes, bytes]:
    entity_uid, ordinal, state = task
    if type(ordinal) is not int or not 0 <= ordinal < 5000:
        raise ValueError("replay ordinal is invalid")
    scripts, inputs, _, _ = stage_paths()
    require_source(scripts, inputs)
    require_smoke_release(scripts, entity_uid)
    if set(state) != STATE_FIELDS or state["entity_uid"] != entity_uid:
        raise ValueError("replay state identity drifted")
    seed = seed_for(state)
    random.seed(seed)
    raw_parent = read_beneath(inputs, state_path(state))
    if sha256(raw_parent) != state["parent_raw_pdb_sha256"]:
        raise ValueError("replay parent PDB hash drifted")
    parent_records = pdb_records(raw_parent)
    parent_heavy = heavy_records(parent_records)
    try:
        import numpy  # type: ignore[import-not-found]
        import openmm  # type: ignore[import-not-found]
        from openmm import Platform, app, unit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("frozen OpenMM runtime is required") from error
    numpy.random.seed(seed % (2**32))
    require_runtime(openmm, app)
    pdb = app.PDBFile(io.StringIO(raw_parent.decode("ascii")))
    if len(list(pdb.topology.atoms())) != len(parent_records) or not heavy_equal(
        openmm_heavy_records(pdb.topology, pdb.positions, unit), parent_heavy
    ):
        raise ValueError("OpenMM parent parse drifted")
    parent_keys = {
        tuple(
            row[key]
            for key in (
                "chain_id",
                "residue_id",
                "insertion_code",
                "residue_name",
                "atom_name",
                "element",
            )
        )
        for row in parent_heavy
    }
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.delete(
        [
            atom
            for atom in modeller.topology.atoms()
            if atom_key(atom) not in parent_keys
        ]
    )
    if not heavy_equal(
        openmm_heavy_records(modeller.topology, modeller.positions, unit), parent_heavy
    ):
        raise ValueError("replay input-hydrogen removal changed parent heavy records")
    forcefield = app.ForceField(FORCEFIELD_PATH)
    platform = Platform.getPlatformByName("Reference")
    variants = modeller.addHydrogens(
        forcefield, pH=float(state["proposal_pH"]), variants=None, platform=platform
    )
    if not heavy_equal(
        openmm_heavy_records(modeller.topology, modeller.positions, unit), parent_heavy
    ):
        raise ValueError("replay protonation changed parent heavy records")
    audit = physicality(modeller, forcefield, platform, unit, openmm)
    proton_signature = signature(modeller.topology)
    text = io.StringIO(newline="\n")
    app.PDBFile.writeModel(modeller.topology, modeller.positions, text, keepIds=True)
    app.PDBFile.writeFooter(modeller.topology, text)
    pdb_raw = text.getvalue().encode("ascii")
    if not heavy_equal(heavy_records(pdb_records(pdb_raw)), parent_heavy):
        raise ValueError("replayed PDB changed parent heavy records")
    topology_hash, coordinate_hash = heavy_hashes(parent_records)
    compressed_pdb = gzip.compress(pdb_raw, compresslevel=9, mtime=0)
    metadata = {
        "atom_count": audit["atom_count"],
        "condition_branch_id": state["condition_branch_id"],
        "entity_uid": entity_uid,
        "hydrogen_count": audit["hydrogen_count"],
        "parent_heavy_coordinate_sha256": coordinate_hash,
        "parent_heavy_topology_sha256": topology_hash,
        "parent_raw_pdb_sha256": state["parent_raw_pdb_sha256"],
        "potential_energy_kj_per_mol": audit["potential_energy_kj_per_mol"],
        "proposal_pH": state["proposal_pH"],
        "protonated_gzip_sha256": sha256(compressed_pdb),
        "protonated_raw_pdb_sha256": sha256(pdb_raw),
        "protonation_signature_sha256": sha256(canonical_bytes(proton_signature)),
        "returned_variants_sha256": sha256(canonical_bytes(list(variants))),
        "support_index": state["support_index"],
    }
    replay_result = {
        "metadata": metadata,
        "minimum_distinct_atom_distance_A": audit["minimum_distinct_atom_distance_A"],
    }
    return compressed_pdb, canonical_bytes(replay_result) + b"\n"


def check_entity(entity_uid: str) -> None:
    scripts, inputs, outputs, checks = stage_paths()
    require_source(scripts, inputs)
    require_smoke_release(scripts, entity_uid)
    states = entity_states(load_projection(inputs), entity_uid)
    if states[0]["condition_state"] != "observed":
        raise PermissionError("bounded smoke requires one observed-condition entity")
    token = safe_token(entity_uid)
    archive_raw = read_regular(outputs / f"{token}.tar", 2_000_000_000)
    members = strict_tar_members(archive_raw)
    member_map = dict(members)
    metadata_member = member_map.pop("metadata.jsonl", None)
    if metadata_member is None:
        raise ValueError("metadata member is absent")
    metadata = parse_metadata(metadata_member)
    expected_names = [
        f"states/{state['support_index']:04d}/{state['condition_branch_id']}.pdb.gz"
        for state in states
    ]
    if [name for name, _ in members] != ["metadata.jsonl", *expected_names] or len(
        metadata
    ) != len(states):
        raise ValueError("archive state inventory drifted")
    work = checks / f".{token}.work.{os.getpid()}"
    receipt_path = checks / f"{token}.checker.json"
    if receipt_path.exists():
        raise FileExistsError("checker output already exists")
    os.mkdir(work, 0o700)
    minimum_distance = math.inf
    minimum_energy = math.inf
    maximum_energy = -math.inf
    branch_signatures: dict[int, set[str]] = {}
    try:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=1, maxtasksperchild=1)
        for ordinal, (state, claimed) in enumerate(zip(states, metadata, strict=True)):
            if set(claimed) != METADATA_FIELDS:
                raise ValueError("state metadata schema drifted")
            member_name = expected_names[ordinal]
            integer_fields = ("atom_count", "hydrogen_count", "support_index")
            hash_fields = (
                "parent_heavy_coordinate_sha256",
                "parent_heavy_topology_sha256",
                "parent_raw_pdb_sha256",
                "protonated_gzip_sha256",
                "protonated_raw_pdb_sha256",
                "protonation_signature_sha256",
                "returned_variants_sha256",
            )
            if (
                any(type(claimed[field]) is not int for field in integer_fields)
                or claimed["atom_count"] <= 0
                or claimed["hydrogen_count"] <= 0
                or any(
                    not isinstance(claimed[field], str)
                    or SHA256_RE.fullmatch(claimed[field]) is None
                    for field in hash_fields
                )
                or type(claimed["potential_energy_kj_per_mol"]) not in {int, float}
                or not math.isfinite(float(claimed["potential_energy_kj_per_mol"]))
            ):
                raise ValueError("state metadata type or finite-value drifted")
            compressed_pdb = member_map[member_name]
            if not (
                len(compressed_pdb) >= 18
                and compressed_pdb[:4] == b"\x1f\x8b\x08\x00"
                and compressed_pdb[4:8] == b"\0\0\0\0"
            ):
                raise ValueError("state PDB gzip header is noncanonical")
            raw_pdb = bounded_gzip_decompress(compressed_pdb, 8_000_000)
            if gzip.compress(raw_pdb, compresslevel=9, mtime=0) != compressed_pdb:
                raise ValueError("state PDB gzip stream is not canonical")
            if not (
                claimed["entity_uid"] == entity_uid
                and claimed["condition_branch_id"] == state["condition_branch_id"]
                and claimed["proposal_pH"] == state["proposal_pH"]
                and claimed["support_index"] == state["support_index"]
                and claimed["parent_raw_pdb_sha256"] == state["parent_raw_pdb_sha256"]
                and claimed["protonated_gzip_sha256"] == sha256(compressed_pdb)
                and claimed["protonated_raw_pdb_sha256"] == sha256(raw_pdb)
            ):
                raise ValueError("state metadata identity or hash drifted")
            replay_pdb, replay_raw = pool.apply(
                replay_worker,
                ((entity_uid, ordinal, state),),
            )
            replay_result = parse_object(replay_raw, "replay metadata")
            if set(replay_result) != {
                "metadata",
                "minimum_distinct_atom_distance_A",
            } or not isinstance(replay_result["metadata"], dict):
                raise ValueError("independent replay result schema drifted")
            if replay_pdb != compressed_pdb or claimed != replay_result["metadata"]:
                raise ValueError("independent state replay differs from producer")
            replay_minimum = replay_result["minimum_distinct_atom_distance_A"]
            if type(replay_minimum) not in {int, float} or not math.isfinite(
                float(replay_minimum)
            ):
                raise ValueError("independent replay minimum distance is invalid")
            minimum_distance = min(minimum_distance, float(replay_minimum))
            energy = float(claimed["potential_energy_kj_per_mol"])
            minimum_energy, maximum_energy = (
                min(minimum_energy, energy),
                max(maximum_energy, energy),
            )
            branch_signatures.setdefault(state["support_index"], set()).add(
                claimed["protonation_signature_sha256"]
            )
        pool.close()
        pool.join()
        unresolved = states[0]["condition_state"] != "observed"
        distinct_supports = sum(
            len(values) > 1 for values in branch_signatures.values()
        )
        if unresolved and distinct_supports == 0:
            raise ValueError("unresolved entity has uniform-zero protonation response")
        receipt = {
            "archive_sha256": sha256(archive_raw),
            "candidate_id": CANDIDATE_ID,
            "closed_capabilities": CLOSED_CAPABILITIES,
            "distinct_signature_support_count": distinct_supports,
            "entity_token": token,
            "entity_uid": entity_uid,
            "maximum_potential_energy_kj_per_mol": maximum_energy,
            "minimum_distinct_atom_distance_A": minimum_distance,
            "minimum_potential_energy_kj_per_mol": minimum_energy,
            "replayed_state_count": len(states),
            "state": "PASS_INDEPENDENT_ENTITY_REPLAY",
        }
        exclusive(receipt_path, canonical_bytes(receipt) + b"\n")
    finally:
        if "pool" in locals():
            pool.terminate()
            pool.join()
        if work.exists():
            shutil.rmtree(work)


def self_test() -> int:
    random.seed(29)
    cases = [[], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]]
    cases.extend(
        [tuple(random.uniform(-4.0, 4.0) for _ in range(3)) for _ in range(size)]
        for size in range(2, 50)
    )
    for coordinates in cases:
        distances = [
            math.dist(left, right)
            for index, left in enumerate(coordinates)
            for right in coordinates[index + 1 :]
        ]
        wanted = min(distances, default=math.inf)
        got = exact_minimum_distance(
            coordinates, wanted if math.isfinite(wanted) else 1.0
        )
        if not (
            (math.isinf(got) and math.isinf(wanted))
            or math.isclose(got, wanted, rel_tol=1e-15, abs_tol=1e-15)
        ):
            raise AssertionError(
                "independent cell-list result differs from brute force"
            )
    tar_buffer = io.BytesIO()
    for name, raw in (("a", b"1"), ("b", b"2")):
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.uid, info.gid, info.mtime = (
            len(raw),
            0o444,
            0,
            0,
            0,
        )
        info.uname = info.gname = ""
        tar_buffer.write(info.tobuf(format=tarfile.USTAR_FORMAT))
        tar_buffer.write(raw)
        tar_buffer.write(bytes((-len(raw)) % 512))
    tar_buffer.write(bytes(1024))
    archive_raw = tar_buffer.getvalue()
    if strict_tar_members(archive_raw) != [("a", b"1"), ("b", b"2")]:
        raise AssertionError("strict USTAR parser rejected canonical bytes")
    tampered = archive_raw + bytes(512)
    try:
        strict_tar_members(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("extra terminal zero block accepted")
    for bad in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
        try:
            parse_object(bad, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("malformed JSON accepted")
    return len(cases) + 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-uid")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    modes = sum((args.entity_uid is not None, args.self_test))
    if modes != 1:
        parser.error("select exactly one checker mode")
    if args.self_test:
        print(f"STATUS PASS_FF15IPQ_ALL_SUPPORT_CHECKER_SELF_TEST checks={self_test()}")
    else:
        check_entity(args.entity_uid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
