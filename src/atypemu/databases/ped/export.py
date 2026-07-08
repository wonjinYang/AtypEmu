"""Parquet export helpers for source-owned PED dataframe bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.ped.layout import (
    PED_WORKSPACES,
    ped_dataset_root,
    ped_models_root,
    ped_tables_root,
)
from atypemu.datasets.registry import (
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)


DATASET_NAME = "ped"
DATASET_VERSION = "2.0.0"
DEFAULT_WORKSPACE = "catalog"
SOURCE_ID = "ped"

TABLE_EXPORT_MAP = {
    "ped_entries": "entries",
    "ped_generation_classes": "generation_classes",
    "ped_bmrb_bridge": "bridges",
    "ped_validation_manifest": "validation_manifest",
    "ped_experimental_tags": "experimental_tags",
    "ped_structural_tags": "structural_tags",
    "ped_models": "models",
}
ENTRY_TABLES = {
    "entries",
    "generation_classes",
    "bridges",
    "validation_manifest",
}
TAG_TABLES = {"experimental_tags", "structural_tags"}
MODEL_MEASURE_COLUMNS = [
    "secondary_structure_entropy",
    "relative_solvent_accessibility",
    "radius_of_gyration",
]


@dataclass(slots=True)
class _WorkspaceExportResult:
    """Summary for one exported PED workspace."""

    workspace_meta: DataFrameWorkspaceMeta
    tables_written: list[str]


def export_ped_dataframes(
    ped_root: str | Path,
    workspace: str = "both",
    refresh: bool = False,
    include_model_assets: bool = True,
    emit_wide_view: bool = True,
) -> dict[str, Any]:
    """Export curated PED TSV tables into source-owned Parquet bundles.

    Args:
        ped_root: PED source root such as ``data/ped``.
        workspace: One workspace or ``both``.
        refresh: When ``True``, overwrite existing Parquet files.
        include_model_assets: When ``False``, skip benchmark model export.
        emit_wide_view: When ``True``, emit ``entry_view.parquet``.

    Returns:
        Summary dictionary describing the exported dataset bundle.
    """
    ped_root_path = Path(ped_root)
    dataset_root = ped_dataset_root(ped_root_path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    workspace_payload: dict[str, DataFrameWorkspaceMeta] = {}
    tables_written = 0

    for workspace_name in _resolve_workspace_selection(workspace):
        result = _export_workspace(
            ped_root=ped_root_path,
            workspace=workspace_name,
            refresh=refresh,
            include_model_assets=include_model_assets,
            emit_wide_view=emit_wide_view,
        )
        workspace_payload[workspace_name] = result.workspace_meta
        tables_written += len(result.tables_written)

    manifest = DataFrameDatasetManifest(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        created_at_utc=_utc_now(),
        root_kind="source",
        data_root=str(ped_root_path.parent.resolve()),
        dataset_root=str(dataset_root.resolve()),
        backend="pandas+pyarrow",
        source_id=SOURCE_ID,
        default_workspace=DEFAULT_WORKSPACE,
        workspaces=workspace_payload,
    )
    manifest_path = dataset_root / "manifest.json"
    manifest.to_json(manifest_path)
    _write_dataset_readme(dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "workspaces": list(workspace_payload),
        "tables_written": tables_written,
    }


def _export_workspace(
    ped_root: Path,
    workspace: str,
    refresh: bool,
    include_model_assets: bool,
    emit_wide_view: bool,
) -> _WorkspaceExportResult:
    """Export one PED workspace into Parquet files."""
    tables_root = ped_tables_root(ped_root, workspace)
    workspace_dataset_root = ped_dataset_root(ped_root) / workspace
    workspace_dataset_root.mkdir(parents=True, exist_ok=True)

    source_tables = [
        "ped_entries",
        "ped_generation_classes",
        "ped_bmrb_bridge",
        "ped_validation_manifest",
        "ped_experimental_tags",
        "ped_structural_tags",
    ]
    if workspace == "benchmark" and include_model_assets:
        source_tables.append("ped_models")

    exported_tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, DataFrameTableMeta] = {}
    written_tables: list[str] = []

    for source_name in source_tables:
        source_path = tables_root / f"{source_name}.tsv"
        if not source_path.exists():
            if source_name == "ped_models" and workspace == "benchmark":
                continue
            raise FileNotFoundError(f"Missing PED source table '{source_path}'.")

        dataframe = _read_public_tsv(source_path, workspace, source_name)
        target_name = TABLE_EXPORT_MAP[source_name]
        target_relative = Path(workspace) / f"{target_name}.parquet"
        target_path = ped_dataset_root(ped_root) / target_relative
        if refresh or not target_path.exists():
            dataframe.to_parquet(target_path, index=False)

        exported_tables[target_name] = dataframe
        metadata[target_name] = _build_table_meta(
            name=target_name,
            dataframe=dataframe,
            relative_path=str(target_relative),
            workspace=workspace,
            level=_infer_level(target_name),
        )
        written_tables.append(target_name)

    wide_views: dict[str, str] = {}
    if emit_wide_view:
        entry_view = _build_entry_view(exported_tables)
        relative_path = Path(workspace) / "entry_view.parquet"
        target_path = ped_dataset_root(ped_root) / relative_path
        if refresh or not target_path.exists():
            entry_view.to_parquet(target_path, index=False)
        metadata["entry_view"] = _build_table_meta(
            name="entry_view",
            dataframe=entry_view,
            relative_path=str(relative_path),
            workspace=workspace,
            level="derived",
        )
        wide_views["entry_view"] = str(relative_path)
        written_tables.append("entry_view")

    workspace_meta = DataFrameWorkspaceMeta(
        workspace=workspace,
        backend="pandas+pyarrow",
        tables=metadata,
        wide_views=wide_views,
    )
    return _WorkspaceExportResult(
        workspace_meta=workspace_meta, tables_written=written_tables
    )


def _resolve_workspace_selection(workspace: str) -> list[str]:
    """Normalize a workspace selector into explicit names."""
    normalized = workspace.strip().lower()
    if normalized == "both":
        return list(PED_WORKSPACES)
    if normalized not in PED_WORKSPACES:
        raise ValueError(
            f"Unsupported PED export workspace '{workspace}'. "
            "Expected 'benchmark', 'catalog', or 'both'."
        )
    return [normalized]


def _read_public_tsv(path: Path, workspace: str, source_name: str) -> pd.DataFrame:
    """Read and normalize one PED public TSV table."""
    dataframe = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=True,
        na_values=["", "None", "null"],
    )
    dataframe = dataframe.replace({r"^\s*$": pd.NA}, regex=True)
    dataframe["source_id"] = SOURCE_ID
    dataframe["workspace"] = workspace
    dataframe["source_table"] = source_name

    if "ped_id" in dataframe.columns:
        dataframe["native_id"] = dataframe["ped_id"]
        dataframe["entry_uid"] = dataframe["ped_id"].map(
            lambda value: f"{SOURCE_ID}:{value}"
        )

    if source_name == "ped_models":
        dataframe["parent_uid"] = dataframe["entry_uid"]
        dataframe["asset_path"] = dataframe.get("model_pdb_path")
        model_root = ped_models_root(path.parents[2], workspace)
        dataframe["asset_root"] = str(model_root)

    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_entry_view(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the canonical PED entry-level trainer view."""
    entries = tables["entries"].copy()
    for table_name in ["generation_classes", "bridges", "validation_manifest"]:
        if table_name not in tables:
            continue
        entries = _merge_entry_frames(entries, tables[table_name], table_name)

    if "models" in tables:
        summary = _summarize_models(tables["models"])
        if not summary.empty:
            entries = _merge_entry_frames(entries, summary, "model_summary")

    entries["source_table"] = "entry_view"
    return entries.convert_dtypes(dtype_backend="pyarrow")


def _merge_entry_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Merge one entry-level payload into the canonical entry view."""
    payload = right.copy()
    drop_columns = [
        column
        for column in ["workspace", "source_table", "source_id", "native_id"]
        if column in payload.columns
    ]
    if drop_columns:
        payload = payload.drop(columns=drop_columns)

    rename_map: dict[str, str] = {}
    for column in payload.columns:
        if column == "entry_uid":
            continue
        if column in left.columns:
            rename_map[column] = f"{source_name}__{column}"
    if rename_map:
        payload = payload.rename(columns=rename_map)
    return left.merge(payload, on="entry_uid", how="left")


def _summarize_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model-level rows to one row per entry."""
    payload = dataframe.copy()
    grouped = payload.groupby("entry_uid", dropna=False)
    summary = grouped.agg(
        ped_model_count=("model_id", "count"),
        ped_ensemble_count=("ensemble_id", "nunique"),
    )
    for column in MODEL_MEASURE_COLUMNS:
        if column in payload.columns:
            summary[f"{column}_mean"] = grouped[column].mean()
    summary = summary.reset_index()
    summary["source_table"] = "model_summary"
    return summary.convert_dtypes(dtype_backend="pyarrow")


def _build_table_meta(
    name: str,
    dataframe: pd.DataFrame,
    relative_path: str,
    workspace: str,
    level: str,
) -> DataFrameTableMeta:
    """Build metadata for one written table."""
    return DataFrameTableMeta(
        name=name,
        relative_path=relative_path,
        columns=list(dataframe.columns),
        dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        row_count=len(dataframe),
        primary_key=_infer_primary_key(name=name),
        join_keys=_infer_join_keys(name=name),
        workspace=workspace,
        level=level,
    )


def _infer_primary_key(name: str) -> list[str]:
    """Infer a stable primary key for one PED table."""
    if name == "models":
        return ["entry_uid", "ensemble_id", "model_id"]
    if name in TAG_TABLES:
        return ["entry_uid", "tag"]
    return ["entry_uid"]


def _infer_join_keys(name: str) -> list[str]:
    """Infer the canonical join keys for one PED table."""
    if name == "models":
        return ["entry_uid", "ensemble_id", "model_id"]
    return ["entry_uid"]


def _infer_level(table_name: str) -> str:
    """Return the semantic level for one PED table."""
    if table_name == "models":
        return "model"
    if table_name in TAG_TABLES:
        return "tag"
    return "entry"


def _write_dataset_readme(dataset_root: Path) -> None:
    """Write the PED dataset README when it is missing."""
    readme_path = dataset_root / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "# PED Dataset Bundle\n\n"
        "This directory stores source-owned Parquet exports derived from "
        "`data/ped/tables`.\n\n"
        "## Layout\n\n"
        "- `manifest.json`: canonical dataset contract\n"
        "- `benchmark/`: asset-backed benchmark workspace, including "
        "`models.parquet`\n"
        "- `catalog/`: metadata-only full PED workspace\n"
    )


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
