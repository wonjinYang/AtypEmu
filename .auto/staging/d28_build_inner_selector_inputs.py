#!/usr/bin/env python3
"""Build a sealed train/dev-only view for the Phase-D inner selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o400)


def checked_copy(source: Path, destination: Path, expected: str) -> None:
    if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
        raise ValueError(f"source drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o400)
    if sha256_file(destination) != expected:
        raise RuntimeError(f"copy drift: {destination}")


def project_checkpoint_manifest(source: Path, destination: Path, expected: str) -> None:
    entries: dict[str, str] = {}
    manifest = source.parent / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or name in entries:
            raise ValueError(f"invalid source checkpoint manifest: {manifest}")
        entries[name] = digest
    if entries.get(source.name) != expected:
        raise ValueError(f"source checkpoint manifest binding drift: {source}")
    destination.write_text(f"{expected}  stage_a_checkpoint.pt\n", encoding="utf-8")
    destination.chmod(0o400)


def rewrite_manifest(
    source: Path,
    expected_sha256: str,
    staging: Path,
    final_root: Path,
    fold: int,
    role: str,
) -> tuple[Path, int]:
    if sha256_file(source) != expected_sha256 or source.is_symlink():
        raise ValueError(f"manifest drift: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not rows or any(row.get("role") != role for row in rows):
        raise ValueError(f"manifest role drift: {source}")
    batch_dir = staging / "data" / f"fold{fold}" / role
    batch_dir.mkdir(parents=True)
    for index, row in enumerate(rows):
        batch_source = Path(row["batch_path"])
        expected = row["batch_sha256"]
        name = f"{index:03d}_{expected[:16]}.pt"
        batch_destination = batch_dir / name
        checked_copy(batch_source, batch_destination, expected)
        row["batch_path"] = str(final_root / "data" / f"fold{fold}" / role / name)
    destination = staging / "data" / f"fold{fold}" / f"{role}_manifest.tsv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    destination.chmod(0o400)
    return destination, len(rows)


def verify_existing(destination: Path, contract_sha256: str) -> dict[str, Any]:
    manifest_path = destination / "input_manifest.json"
    checksums_path = destination / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError("existing destination is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_kind") != "v3339_phase_d_inner_selector_inputs_v1"
        or manifest.get("sealed_root") != str(destination)
        or manifest.get("build_contract_sha256") != contract_sha256
        or manifest.get("outer_held_manifests_or_batches_copied") is not False
    ):
        raise ValueError("existing sealed input contract drift")
    expected = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual = {
        str(path.relative_to(destination)): path
        for path in destination.rglob("*")
        if path.is_file() and path != checksums_path
    }
    if set(actual) != set(expected):
        raise ValueError("existing sealed input inventory drift")
    if (destination.stat().st_mode & 0o777) != 0o500 or any(
        (path.stat().st_mode & 0o777) != 0o500
        for path in destination.rglob("*") if path.is_dir()
    ):
        raise ValueError("existing sealed input directory mode drift")
    for relative, path in actual.items():
        if (
            path.is_symlink()
            or (path.stat().st_mode & 0o777) != 0o400
            or sha256_file(path) != expected[relative]
        ):
            raise ValueError(f"existing sealed input checksum drift: {relative}")
    if (checksums_path.stat().st_mode & 0o777) != 0o400:
        raise ValueError("existing sealed checksum mode drift")
    return manifest


def build(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("artifact_kind")
        != "v3339_phase_d_inner_selector_input_build_contract_v1"
        or contract.get("copied_roles") != ["inner_train", "inner_dev"]
        or contract.get("outer_held_manifests_or_batches_copied") is not False
        or len(contract.get("folds", ())) != 3
    ):
        raise ValueError("input build contract drift")
    destination = Path(contract["destination"])
    if destination.exists():
        return verify_existing(destination, sha256_file(contract_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    final_folds = []
    try:
        batch_root = Path(contract["atom27_batch_root"])
        control_root = Path(contract["control_result_root"])
        q_root = Path(contract["shared_q_root"])
        for row in sorted(contract["folds"], key=lambda value: int(value["fold"])):
            fold = int(row["fold"])
            train, train_count = rewrite_manifest(
                batch_root / f"fold{fold}" / "train_batch_manifest.tsv",
                row["train_manifest_sha256"], staging, destination, fold, "inner_train",
            )
            dev, dev_count = rewrite_manifest(
                batch_root / f"fold{fold}" / "dev_batch_manifest.tsv",
                row["dev_manifest_sha256"], staging, destination, fold, "inner_dev",
            )
            q = staging / "models" / f"fold{fold}" / "q_checkpoint.pt"
            stage_a = staging / "models" / f"fold{fold}" / "stage_a_checkpoint.pt"
            control = staging / "models" / f"fold{fold}" / "control_checkpoint.pt"
            checked_copy(
                q_root / f"fold{fold}" / "checkpoint.pt", q, row["q_checkpoint_sha256"]
            )
            checked_copy(
                Path(contract["stage_a_pattern"].format(fold=fold)),
                stage_a, row["stage_a_checkpoint_sha256"],
            )
            project_checkpoint_manifest(
                Path(contract["stage_a_pattern"].format(fold=fold)),
                stage_a.parent / "checksums.sha256",
                row["stage_a_checkpoint_sha256"],
            )
            checked_copy(
                control_root / "outputs" / "legacy5" / f"fold{fold}" / "checkpoint.pt",
                control, row["control_checkpoint_sha256"],
            )
            final_folds.append({
                "fold": fold,
                "train_manifest": str(destination / train.relative_to(staging)),
                "train_manifest_sha256": sha256_file(train),
                "original_train_manifest_sha256": row["train_manifest_sha256"],
                "train_entity_count": train_count,
                "dev_manifest": str(destination / dev.relative_to(staging)),
                "dev_manifest_sha256": sha256_file(dev),
                "original_dev_manifest_sha256": row["dev_manifest_sha256"],
                "dev_entity_count": dev_count,
                "q_checkpoint": str(destination / q.relative_to(staging)),
                "q_checkpoint_sha256": row["q_checkpoint_sha256"],
                "stage_a_checkpoint": str(destination / stage_a.relative_to(staging)),
                "stage_a_checkpoint_sha256": row["stage_a_checkpoint_sha256"],
                "control_checkpoint": str(destination / control.relative_to(staging)),
                "control_checkpoint_sha256": row["control_checkpoint_sha256"],
            })
        manifest = {
            "artifact_kind": "v3339_phase_d_inner_selector_inputs_v1",
            "sealed_root": str(destination),
            "folds": final_folds,
            "copied_roles": ["inner_train", "inner_dev"],
            "outer_held_manifests_or_batches_copied": False,
            "control_checkpoints_copied_without_summary_history_or_predictions": True,
            "forbidden_paths": [
                contract["control_result_root"],
                contract["atom27_batch_root"],
            ],
            "build_contract_sha256": sha256_file(contract_path),
        }
        write_json(staging / "input_manifest.json", manifest)
        inventory = []
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            if path.is_symlink():
                raise ValueError("sealed inventory contains a symlink")
            inventory.append(f"{sha256_file(path)}  {path.relative_to(staging)}\n")
        checksums = staging / "checksums.sha256"
        checksums.write_text("".join(inventory), encoding="utf-8")
        checksums.chmod(0o400)
        for directory in sorted(
            (value for value in staging.rglob("*") if value.is_dir()), reverse=True
        ):
            directory.chmod(0o500)
        staging.chmod(0o500)
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            for value in staging.rglob("*"):
                try:
                    value.chmod(0o700 if value.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    result = build(args.contract)
    path = Path(result["sealed_root"]) / "input_manifest.json"
    print(f"sealed_input_manifest={path}")
    print(f"sealed_input_manifest_sha256={sha256_file(path)}")
    print("outer_held_manifests_or_batches_copied=false")


if __name__ == "__main__":
    main()
