"""Parquet export helpers for source-owned BMRB dataframe bundles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.bmrb.layout import (
    bmrb_dataset_root,
    bmrb_manifest_root,
)
from atypemu.datasets.registry import (
    DEFAULT_WORKSPACE,
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)
from atypemu.types import NMRTargetBundle


DATASET_NAME = "bmrb"
DATASET_VERSION = "1.0.0"
SOURCE_ID = "bmrb"


def export_bmrb_dataframes(
    bmrb_root: str | Path,
    refresh: bool = False,
    emit_entry_view: bool = True,
) -> dict[str, Any]:
    """Export curated BMRB tables and bundles into Parquet datasets.

    Args:
        bmrb_root: BMRB source root such as ``data/bmrb``.
        refresh: When ``True``, overwrite existing Parquet files.
        emit_entry_view: When ``True``, emit ``entry_view.parquet``.

    Returns:
        Summary dictionary describing the written dataset bundle.
    """
    bmrb_root_path = Path(bmrb_root)
    dataset_root = bmrb_dataset_root(bmrb_root_path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = json.loads(
        (bmrb_manifest_root(bmrb_root_path) / "index.json").read_text()
    )
    entries = _build_entries_frame(manifest_rows)
    tags = _build_tags_frame(entries)
    targets = _build_targets_frame(bmrb_root_path, entries)

    tables = {
        "entries": entries,
        "tags": tags,
        "targets": targets,
    }
    if emit_entry_view:
        tables["entry_view"] = entries.copy().convert_dtypes(dtype_backend="pyarrow")

    metadata: dict[str, DataFrameTableMeta] = {}
    for table_name, dataframe in tables.items():
        relative_path = Path(DEFAULT_WORKSPACE) / f"{table_name}.parquet"
        target_path = dataset_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not target_path.exists():
            dataframe.to_parquet(target_path, index=False)
        metadata[table_name] = _build_table_meta(
            name=table_name,
            dataframe=dataframe,
            relative_path=str(relative_path),
        )

    manifest = DataFrameDatasetManifest(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        created_at_utc=_utc_now(),
        root_kind="source",
        data_root=str(bmrb_root_path.parent.resolve()),
        dataset_root=str(dataset_root.resolve()),
        backend="pandas+pyarrow",
        source_id=SOURCE_ID,
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
    manifest_path = dataset_root / "manifest.json"
    manifest.to_json(manifest_path)
    _write_dataset_readme(dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "tables_written": len(tables),
    }


def _build_entries_frame(manifest_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the canonical BMRB entry table from the manifest rows."""
    rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        bmrb_id = str(row["bmrb_id"])
        entry_uid = f"{SOURCE_ID}:{bmrb_id}"
        bundle_path = row.get("bundle_path")
        if bundle_path:
            bundle_path = _relative_or_original(bundle_path, "bundles")
        nmrstar_path = row.get("nmrstar_path")
        if nmrstar_path:
            nmrstar_path = _relative_or_original(nmrstar_path, "nmrstar")
        noe_path = row.get("noe_path")
        if noe_path:
            noe_path = _relative_or_original(noe_path, "noe")

        rows.append(
            {
                **row,
                "bundle_path": bundle_path,
                "nmrstar_path": nmrstar_path,
                "noe_path": noe_path,
                "source_id": SOURCE_ID,
                "native_id": bmrb_id,
                "entry_uid": entry_uid,
                "asset_path": bundle_path,
            }
        )
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_tags_frame(entries: pd.DataFrame) -> pd.DataFrame:
    """Build long-form availability tags for each BMRB entry."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        tag_pairs = {
            "status": row.get("status"),
            "noe_status": row.get("noe_status"),
            "chemical_shift_positive": (
                "true" if int(row.get("chemical_shift_count") or 0) > 0 else "false"
            ),
            "j_coupling_positive": (
                "true" if int(row.get("j_coupling_count") or 0) > 0 else "false"
            ),
            "noe_positive": "true" if int(row.get("noe_count") or 0) > 0 else "false",
        }
        for namespace, tag in tag_pairs.items():
            if tag is None or tag == "":
                continue
            rows.append(
                {
                    "entry_uid": row["entry_uid"],
                    "source_id": SOURCE_ID,
                    "native_id": row["native_id"],
                    "namespace": namespace,
                    "tag": tag,
                }
            )
    dataframe = pd.DataFrame(rows)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_targets_frame(bmrb_root: Path, entries: pd.DataFrame) -> pd.DataFrame:
    """Build one long-form target table from parsed BMRB bundles."""
    rows: list[dict[str, Any]] = []

    for entry in entries.to_dict(orient="records"):
        bundle_path = entry.get("bundle_path")
        if not bundle_path:
            continue
        bundle_file = Path(bundle_path)
        if not bundle_file.is_absolute():
            bundle_file = bmrb_root / bundle_file
        bundle = NMRTargetBundle.from_json(bundle_file)
        rows.extend(
            _bundle_targets(
                bundle=bundle,
                entry_uid=entry["entry_uid"],
                native_id=entry["native_id"],
            )
        )

    dataframe = pd.DataFrame(rows)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _bundle_targets(
    bundle: NMRTargetBundle,
    entry_uid: str,
    native_id: str,
) -> list[dict[str, Any]]:
    """Expand one target bundle into a long-form measurement table."""
    rows: list[dict[str, Any]] = []
    for target in bundle.chemical_shifts:
        rows.append(
            {
                "entry_uid": entry_uid,
                "source_id": SOURCE_ID,
                "native_id": native_id,
                "target_uid": target.target_id(),
                "measurement_kind": "chemical_shift",
                "seq_id": target.seq_id,
                "comp_id": target.comp_id,
                "atom_id": target.atom_id,
                "value": target.value,
                "uncertainty": target.uncertainty,
            }
        )
    for target in bundle.j_couplings:
        rows.append(
            {
                "entry_uid": entry_uid,
                "source_id": SOURCE_ID,
                "native_id": native_id,
                "target_uid": target.target_id(),
                "measurement_kind": "j_coupling",
                "seq_id": target.seq_id,
                "comp_id": target.comp_id,
                "atom_id": f"{target.atom_id_1}-{target.atom_id_2}",
                "value": target.value,
                "uncertainty": target.uncertainty,
            }
        )
    for target in bundle.noe_restraints:
        rows.append(
            {
                "entry_uid": entry_uid,
                "source_id": SOURCE_ID,
                "native_id": native_id,
                "target_uid": target.target_id(),
                "measurement_kind": "noe_restraint",
                "seq_id": target.seq_id_1,
                "comp_id": target.comp_id_1,
                "atom_id": f"{target.atom_id_1}--{target.atom_id_2}",
                "value": target.target_value,
                "uncertainty": target.uncertainty,
            }
        )
    return rows


def _build_table_meta(
    name: str,
    dataframe: pd.DataFrame,
    relative_path: str,
) -> DataFrameTableMeta:
    """Build metadata for one written BMRB Parquet table."""
    return DataFrameTableMeta(
        name=name,
        relative_path=relative_path,
        columns=list(dataframe.columns),
        dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        row_count=len(dataframe),
        primary_key=_infer_primary_key(name),
        join_keys=_infer_join_keys(name),
        workspace=DEFAULT_WORKSPACE,
        level=_infer_level(name),
    )


def _infer_primary_key(name: str) -> list[str]:
    """Return the canonical primary key for one BMRB table."""
    if name == "targets":
        return ["entry_uid", "target_uid"]
    if name == "tags":
        return ["entry_uid", "namespace", "tag"]
    return ["entry_uid"]


def _infer_join_keys(name: str) -> list[str]:
    """Return the canonical join keys for one BMRB table."""
    if name == "targets":
        return ["entry_uid", "target_uid"]
    return ["entry_uid"]


def _infer_level(name: str) -> str:
    """Return the semantic level for one BMRB table."""
    if name == "targets":
        return "target"
    if name == "tags":
        return "tag"
    return "entry"


def _write_dataset_readme(dataset_root: Path) -> None:
    """Write a short README for the BMRB dataset bundle."""
    readme_path = dataset_root / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "# BMRB Dataset Bundle\n\n"
        "This directory stores source-owned Parquet exports derived from "
        "`data/bmrb/tables` and `data/bmrb/assets`.\n"
    )


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _relative_or_original(path: str, marker: str) -> str:
    """Return a stable source-relative path when one legacy marker is present."""
    path_obj = Path(path)
    if marker in path_obj.parts:
        marker_index = path_obj.parts.index(marker)
        base = "assets" if marker == "bundles" else "downloads"
        return str(Path(base, *path_obj.parts[marker_index:]))
    return path
