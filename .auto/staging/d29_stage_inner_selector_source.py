#!/usr/bin/env python3
"""Freeze one immutable source bundle for an inner-selector experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


FILES = {
    "atom27": "analysis/v3339_atom27_explicit_h.py",
    "trainer": "analysis/v3339_train_atom27_phase_d.py",
    "selector": "analysis/v3339_inner_chart_selector.py",
    "runner": "run_inner_selector.sbatch",
    "menu": "inner_chart_menu.json",
}
PARENT_FILES = (
    "analysis/v3339_all_atom_e2e.py",
    "analysis/v3339_atom14_support.py",
    "analysis/v3339_train_all_bmrb_stage_a.py",
    "analysis/v3339_train_stage_bc.py",
    "deps/fixed_support_measure.py",
    "q_initialization_transfer_receipt_d4v2.json",
)
BINDING_KEYS = {
    "container_image_sha256",
    "input_manifest_sha256",
    "parent_checksums_sha256",
    "stage_script_sha256",
}
PARENT_MANIFEST = "parent_checksums.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_sha(hashes: dict[str, str], bindings: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"source_hashes": hashes, "bindings": bindings},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        item = Path(relative)
        normalized = str(item)
        if (
            separator != "  "
            or len(digest) != 64
            or item.is_absolute()
            or ".." in item.parts
            or normalized in result
        ):
            raise ValueError(f"invalid checksum manifest: {path}")
        result[normalized] = digest
    return result


def verify_bundle(
    root: Path,
    *,
    identifier: str,
    hashes: dict[str, str],
    bindings: dict[str, str],
) -> None:
    receipt_path = root / "inner_selector_source_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    parent_manifest = root / PARENT_MANIFEST
    if sha256_file(parent_manifest) != bindings["parent_checksums_sha256"]:
        raise ValueError("bundled parent checksum manifest drift")
    parent_hashes = read_checksums(parent_manifest)
    expected_parent = {relative: parent_hashes.get(relative) for relative in PARENT_FILES}
    if any(value is None for value in expected_parent.values()) or (
        receipt.get("artifact_kind") != "v3339_phase_d_inner_selector_source_bundle_v1"
        or receipt.get("bundle_sha256") != identifier
        or receipt.get("source_hashes") != hashes
        or receipt.get("bindings") != bindings
        or receipt.get("parent_file_hashes") != expected_parent
    ):
        raise ValueError("source bundle receipt drift")
    expected_files = set(PARENT_FILES) | set(FILES.values()) | {
        PARENT_MANIFEST,
        receipt_path.name,
    }
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != root / "checksums.sha256"
    }
    if actual_files != expected_files:
        raise ValueError("source bundle file inventory drift")
    for name, relative in FILES.items():
        if sha256_file(root / relative) != hashes[name]:
            raise ValueError(f"source bundle uploaded file drift: {name}")
    for relative, digest in expected_parent.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"source bundle parent file drift: {relative}")
    inventory = read_checksums(root / "checksums.sha256")
    if set(inventory) != expected_files or any(
        sha256_file(root / relative) != digest for relative, digest in inventory.items()
    ):
        raise ValueError("source bundle checksum inventory drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--upload", type=Path, required=True)
    parser.add_argument("--destination-base", type=Path, required=True)
    parser.add_argument("--hashes-json", required=True)
    parser.add_argument("--bindings-json", required=True)
    args = parser.parse_args()
    hashes = json.loads(args.hashes_json)
    bindings = json.loads(args.bindings_json)
    if set(hashes) != set(FILES) or any(len(value) != 64 for value in hashes.values()):
        raise ValueError("source hash inventory drift")
    if set(bindings) != BINDING_KEYS or any(len(value) != 64 for value in bindings.values()):
        raise ValueError("source binding inventory drift")
    if sha256_file(Path(__file__)) != bindings["stage_script_sha256"]:
        raise ValueError("source staging script SHA256 drift")
    if sha256_file(args.parent / "checksums.sha256") != bindings["parent_checksums_sha256"]:
        raise ValueError("parent source checksum manifest drift")
    identifier = bundle_sha(hashes, bindings)
    destination = args.destination_base / identifier / "frozen"
    if destination.exists():
        verify_bundle(destination, identifier=identifier, hashes=hashes, bindings=bindings)
        print(f"source_bundle={destination}")
        print(f"source_bundle_sha256={identifier}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".frozen.", dir=destination.parent))
    try:
        parent_manifest = args.parent / "checksums.sha256"
        parent_hashes = read_checksums(parent_manifest)
        shutil.copyfile(parent_manifest, staging / PARENT_MANIFEST)
        for relative in PARENT_FILES:
            source = args.parent / relative
            if (
                source.is_symlink()
                or not source.is_file()
                or parent_hashes.get(relative) != sha256_file(source)
            ):
                raise ValueError(f"parent source dependency drift: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name, relative in FILES.items():
            source = args.upload / Path(relative).name
            if source.is_symlink() or sha256_file(source) != hashes[name]:
                raise ValueError(f"uploaded source drift: {name}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        receipt = {
            "artifact_kind": "v3339_phase_d_inner_selector_source_bundle_v1",
            "bundle_sha256": identifier,
            "source_hashes": hashes,
            "bindings": bindings,
            "allowlisted_parent_files": list(PARENT_FILES),
            "parent_file_hashes": {
                relative: parent_hashes[relative] for relative in PARENT_FILES
            },
            "parent_source": str(args.parent),
            "parent_source_checksums_sha256": sha256_file(args.parent / "checksums.sha256"),
        }
        (staging / "inner_selector_source_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksums = staging / "checksums.sha256"
        if checksums.exists():
            checksums.unlink()
        lines = []
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            lines.append(f"{sha256_file(path)}  {path.relative_to(staging)}\n")
        checksums.write_text("".join(lines), encoding="utf-8")
        verify_bundle(staging, identifier=identifier, hashes=hashes, bindings=bindings)
        for value in staging.rglob("*"):
            value.chmod(0o500 if value.is_dir() else 0o400)
        staging.chmod(0o500)
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            for value in staging.rglob("*"):
                try:
                    value.chmod(0o700 if value.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(staging)
        raise
    print(f"source_bundle={destination}")
    print(f"source_bundle_sha256={identifier}")


if __name__ == "__main__":
    main()
