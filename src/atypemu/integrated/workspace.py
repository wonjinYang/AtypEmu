"""Integrated workspace setup helpers for AtypEmu."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.datasets.registry import (
    DEFAULT_WORKSPACE,
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)
from atypemu.structures import ResidueMapper, StructureLoader
from atypemu.types import NMRTargetBundle


SOURCE_LAYOUT = {
    "AF3": ("AF", "unpacked"),
    "CALVADOS2": ("CALVADOS", "unpacked"),
    "BioEmu": ("BioEmu", None),
}


def setup_integrated_workspace(
    data_root: str | Path,
    bmrb_root: str | Path,
    output_root: str | Path,
    max_per_source: int = 256,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    """Create manifests and split files for integrated AtypEmu workflows.

    Args:
        data_root: Root candidate-structure directory.
        bmrb_root: Root organized BMRB directory.
        output_root: Integrated workspace root to create.
        max_per_source: Default per-source cap written to the config.
        split_ratios: Train, validation, and test ratios.

    Returns:
        Summary dictionary describing the prepared integrated workspace.
    """
    workspace = Path(output_root)
    inventories_dir = workspace / "inventories"
    staging_dir = workspace / "staging"
    pools_dir = workspace / "pools"
    targets_dir = workspace / "targets"
    observables_dir = workspace / "observables"
    teachers_dir = workspace / "teachers"
    splits_dir = workspace / "splits"
    runs_dir = workspace / "runs"
    configs_dir = workspace / "configs"
    datasets_dir = workspace / "datasets"

    for path in [
        inventories_dir,
        staging_dir,
        pools_dir,
        targets_dir,
        observables_dir,
        teachers_dir,
        splits_dir,
        runs_dir,
        configs_dir,
        datasets_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    data_root_path = Path(data_root)
    bmrb_root_path = Path(bmrb_root)
    master_bmrb_ids = _load_master_bmrb_ids(data_root_path)
    source_inventory = _build_source_inventory(data_root_path, master_bmrb_ids)
    accession_rows = _build_accession_inventory(
        data_root_path=data_root_path,
        bmrb_root_path=bmrb_root_path,
        source_inventory=source_inventory,
        master_bmrb_ids=master_bmrb_ids,
    )
    split_payload = _build_splits(accession_rows, split_ratios)

    (inventories_dir / "source_inventory.json").write_text(
        json.dumps(source_inventory, indent=2, sort_keys=True)
    )
    (inventories_dir / "accessions.json").write_text(
        json.dumps(accession_rows, indent=2, sort_keys=True)
    )
    (inventories_dir / "accessions.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in accession_rows)
        + ("\n" if accession_rows else "")
    )
    (inventories_dir / "accessions.txt").write_text(
        "\n".join(row["bmrb_id"] for row in accession_rows)
        + ("\n" if accession_rows else "")
    )
    (splits_dir / "default_split.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True)
    )

    workspace_config = {
        "data_root": str(data_root_path.resolve()),
        "bmrb_root": str(bmrb_root_path.resolve()),
        "workspace_root": str(workspace.resolve()),
        "candidate_sources": {
            source: {
                "physical_dir": physical_dir,
                "structure_root": str(
                    (
                        data_root_path / physical_dir / subdir
                        if subdir is not None
                        else data_root_path / physical_dir
                    ).resolve()
                ),
            }
            for source, (physical_dir, subdir) in SOURCE_LAYOUT.items()
        },
        "defaults": {
            "per_source_cap": int(max_per_source),
            "split_policy": "sequence_hash",
            "split_ratios": {
                "train": split_ratios[0],
                "val": split_ratios[1],
                "test": split_ratios[2],
            },
            "sampling_policy": {
                "within_protein": "source_balanced",
                "across_proteins": "protein_balanced",
            },
        },
        "artifacts": {
            "datasets_dir": str(datasets_dir.resolve()),
            "inventories_dir": str(inventories_dir.resolve()),
            "splits_dir": str(splits_dir.resolve()),
            "staging_dir": str(staging_dir.resolve()),
            "pools_dir": str(pools_dir.resolve()),
            "targets_dir": str(targets_dir.resolve()),
            "observables_dir": str(observables_dir.resolve()),
            "teachers_dir": str(teachers_dir.resolve()),
            "runs_dir": str(runs_dir.resolve()),
        },
    }
    (configs_dir / "training_workspace.json").write_text(
        json.dumps(workspace_config, indent=2, sort_keys=True)
    )
    _write_integrated_registry(
        workspace=workspace,
        data_root_path=data_root_path,
        accession_rows=accession_rows,
        source_inventory=source_inventory,
        split_payload=split_payload,
    )

    _write_integrated_readme(workspace)

    return {
        "accessions": len(accession_rows),
        "train": len(split_payload["train"]),
        "val": len(split_payload["val"]),
        "test": len(split_payload["test"]),
        "workspace_root": str(workspace),
    }


def _build_source_inventory(
    data_root_path: Path,
    master_bmrb_ids: set[str],
) -> dict[str, Any]:
    """Summarize candidate availability per logical source."""
    payload: dict[str, Any] = {}
    for source, (physical_dir, subdir) in SOURCE_LAYOUT.items():
        root = data_root_path / physical_dir
        structure_root = root / subdir if subdir is not None else root
        accession_dirs = (
            [path for path in structure_root.iterdir() if path.is_dir()]
            if structure_root.exists()
            else []
        )
        archived_accessions = _collect_archive_accessions(root, master_bmrb_ids)
        payload[source] = {
            "physical_dir": physical_dir,
            "structure_root": str(structure_root),
            "accessions": len(
                {path.name for path in accession_dirs} | archived_accessions
            ),
            "candidate_files": None,
            "candidate_files_counted": False,
            "archive_chunks": len(list(root.glob("*.zip"))) if root.exists() else 0,
        }
    return payload


def _build_accession_inventory(
    data_root_path: Path,
    bmrb_root_path: Path,
    source_inventory: dict[str, Any],
    master_bmrb_ids: set[str],
) -> list[dict[str, Any]]:
    """Build per-accession metadata for training and fine-tuning."""
    accession_to_sources: dict[str, set[str]] = defaultdict(set)
    accession_to_paths: dict[str, dict[str, str]] = defaultdict(dict)

    for source, config in source_inventory.items():
        structure_root = Path(config["structure_root"])
        physical_dir = str(config["physical_dir"])
        accession_names = _collect_accession_names(
            data_root_path=data_root_path,
            physical_dir=physical_dir,
            structure_root=structure_root,
            master_bmrb_ids=master_bmrb_ids,
        )
        for accession_name in sorted(accession_names):
            accession_to_sources[accession_name].add(source)
            accession_to_paths[accession_name][source] = str(
                structure_root / accession_name
            )

    rows: list[dict[str, Any]] = []
    for bmrb_id in sorted(accession_to_sources):
        bundle_path = bmrb_root_path / "assets" / "bundles" / f"{bmrb_id}.json"
        bundle = (
            NMRTargetBundle.from_json(bundle_path) if bundle_path.exists() else None
        )
        sequence = (
            _infer_sequence_from_bundle(bundle)
            if bundle is not None
            else _infer_sequence(accession_to_paths[bmrb_id])
        )
        sequence_hash = (
            hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else None
        )
        rows.append(
            {
                "bmrb_id": bmrb_id,
                "sources": sorted(accession_to_sources[bmrb_id]),
                "source_paths": accession_to_paths[bmrb_id],
                "bmrb_bundle_path": str(bundle_path) if bundle_path.exists() else None,
                "chemical_shift_count": (
                    0 if bundle is None else len(bundle.chemical_shifts)
                ),
                "j_coupling_count": 0 if bundle is None else len(bundle.j_couplings),
                "noe_count": 0 if bundle is None else len(bundle.noe_restraints),
                "sequence": sequence,
                "sequence_hash": sequence_hash,
                "training_ready": bundle_path.exists(),
            }
        )
    return rows


def _infer_sequence(source_paths: dict[str, str]) -> str:
    """Infer one residue sequence from the first available structure file."""
    loader = StructureLoader()
    for source in ["BioEmu", "AF3", "CALVADOS2"]:
        root = source_paths.get(source)
        if root is None:
            continue
        pdb_paths = sorted(Path(root).glob("*.pdb"))
        if not pdb_paths:
            continue
        structure = loader.load_structure(pdb_paths[0])
        residue_keys = ResidueMapper.extract_residue_keys(structure)
        return "-".join(item.split(":")[-1] for item in residue_keys)
    return ""


def _infer_sequence_from_bundle(bundle: NMRTargetBundle) -> str:
    """Infer one residue sequence from chemical-shift targets when available."""
    residues: dict[tuple[str | None, int], str] = {}
    for target in bundle.chemical_shifts:
        key = (target.chain_id, target.seq_id)
        residues.setdefault(key, target.comp_id)
    ordered = [
        residues[key]
        for key in sorted(residues, key=lambda item: (item[0] or "", item[1]))
    ]
    return "-".join(ordered)


def _collect_accession_names(
    data_root_path: Path,
    physical_dir: str,
    structure_root: Path,
    master_bmrb_ids: set[str],
) -> set[str]:
    """Collect accession names from unpacked trees and remaining zip chunks."""
    accession_names = (
        {path.name for path in structure_root.iterdir() if path.is_dir()}
        if structure_root.exists()
        else set()
    )
    accession_names.update(
        _collect_archive_accessions(data_root_path / physical_dir, master_bmrb_ids)
    )
    return accession_names


def _collect_archive_accessions(
    source_root: Path, master_bmrb_ids: set[str]
) -> set[str]:
    """Collect accession prefixes from zip chunk central directories."""
    if master_bmrb_ids:
        return set(master_bmrb_ids)

    accession_names: set[str] = set()
    for archive_path in sorted(source_root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if "/" not in name:
                    continue
                prefix = name.split("/", 1)[0]
                if prefix.startswith("bmr"):
                    accession_names.add(prefix)
    return accession_names


def _load_master_bmrb_ids(data_root_path: Path) -> set[str]:
    """Load the canonical accession list when available."""
    path = data_root_path / "integrated" / "configs" / "bmrb_ids.txt"
    if not path.exists():
        return set()
    return {
        line.strip().lower() for line in path.read_text().splitlines() if line.strip()
    }


def _build_splits(
    accession_rows: list[dict[str, Any]],
    split_ratios: tuple[float, float, float],
) -> dict[str, list[str]]:
    """Assign accessions to deterministic train, validation, and test splits."""
    train_cutoff = split_ratios[0]
    val_cutoff = split_ratios[0] + split_ratios[1]

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in accession_rows:
        group_key = row["sequence_hash"] or row["bmrb_id"]
        grouped[group_key].append(row["bmrb_id"])

    split_payload = {"train": [], "val": [], "test": []}
    for group_key in sorted(grouped):
        bucket = _hash_to_unit_interval(group_key)
        if bucket < train_cutoff:
            split_name = "train"
        elif bucket < val_cutoff:
            split_name = "val"
        else:
            split_name = "test"
        split_payload[split_name].extend(sorted(grouped[group_key]))

    return split_payload


def _hash_to_unit_interval(value: str) -> float:
    """Map a stable string to the unit interval deterministically."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _write_integrated_readme(workspace: Path) -> None:
    """Write a short README for the prepared integrated workspace."""
    (workspace / "README.md").write_text(
        "# AtypEmu Integrated Workspace\n\n"
        "This directory stores generated manifests and working directories for "
        "AtypEmu offline teacher generation, fine-tuning preparation, and "
        "downstream analysis.\n\n"
        "## Key subdirectories\n\n"
        "- `inventories`: source and accession inventories\n"
        "- `splits`: deterministic train/validation/test assignments\n"
        "- `staging`: per-BMRB staged working directories\n"
        "- `pools`: candidate pool JSONL outputs\n"
        "- `targets`: parsed target bundles used by AtypEmu\n"
        "- `observables`: computed observable matrices\n"
        "- `teachers`: offline teacher outputs and diagnostics\n"
        "- `datasets`: trainer-facing dataframe bundles and manifests\n"
        "- `runs`: run outputs and logs\n"
        "- `configs/training_workspace.json`: canonical workspace configuration\n"
        "- `configs/training_plan.json`: canonical staged training plan\n"
        "- `configs/bmrb_ids.txt`: canonical accession list for workspace prep\n"
        "- `manifest.json`: integrated dataframe-registry contract\n"
    )


def _write_integrated_registry(
    workspace: Path,
    data_root_path: Path,
    accession_rows: list[dict[str, Any]],
    source_inventory: dict[str, Any],
    split_payload: dict[str, list[str]],
) -> None:
    """Write a compact Parquet registry for integrated cross-source artifacts."""
    accessions_frame = pd.DataFrame(accession_rows).replace(
        {r"^\s*$": pd.NA},
        regex=True,
    )
    source_rows = [
        {"logical_source": source, **payload}
        for source, payload in source_inventory.items()
    ]
    source_frame = pd.DataFrame(source_rows).replace({r"^\s*$": pd.NA}, regex=True)
    split_rows = [
        {
            "entity_uid": f"bmrb:{bmrb_id}",
            "source_id": "bmrb",
            "split": split_name,
        }
        for split_name, values in split_payload.items()
        for bmrb_id in values
    ]
    split_frame = pd.DataFrame(split_rows).replace({r"^\s*$": pd.NA}, regex=True)

    tables = {
        "accessions": accessions_frame.convert_dtypes(dtype_backend="pyarrow"),
        "source_inventory": source_frame.convert_dtypes(dtype_backend="pyarrow"),
        "splits": split_frame.convert_dtypes(dtype_backend="pyarrow"),
    }
    metadata: dict[str, DataFrameTableMeta] = {}

    for table_name, dataframe in tables.items():
        if table_name == "splits":
            relative_path = Path("splits") / "default_split.parquet"
        else:
            relative_path = Path("inventories") / f"{table_name}.parquet"
        target_path = workspace / relative_path
        dataframe.to_parquet(target_path, index=False)
        metadata[table_name] = DataFrameTableMeta(
            name=table_name,
            relative_path=str(relative_path),
            columns=list(dataframe.columns),
            dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            row_count=len(dataframe),
            primary_key=(
                ["entity_uid"]
                if table_name == "splits"
                else (
                    ["logical_source"]
                    if table_name == "source_inventory"
                    else ["bmrb_id"]
                )
            ),
            join_keys=(
                ["entity_uid"]
                if table_name == "splits"
                else (
                    ["logical_source"]
                    if table_name == "source_inventory"
                    else ["bmrb_id"]
                )
            ),
            workspace=DEFAULT_WORKSPACE,
            level="entry",
        )

    manifest = DataFrameDatasetManifest(
        dataset_name="integrated",
        dataset_version="1.0.0",
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        root_kind="integrated",
        data_root=str(data_root_path.resolve()),
        dataset_root=str(workspace.resolve()),
        backend="pandas+pyarrow",
        default_workspace=DEFAULT_WORKSPACE,
        workspaces={
            DEFAULT_WORKSPACE: DataFrameWorkspaceMeta(
                workspace=DEFAULT_WORKSPACE,
                backend="pandas+pyarrow",
                tables=metadata,
                wide_views={},
            )
        },
    )
    manifest.to_json(workspace / "manifest.json")
