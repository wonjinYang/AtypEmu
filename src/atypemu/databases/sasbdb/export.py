"""Parquet export helpers for source-owned SASBDB dataframe bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.sasbdb.config import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_WORKSPACE,
    SOURCE_ID,
)
from atypemu.databases.sasbdb.layout import (
    sasbdb_dataset_root,
    sasbdb_tables_root,
    sasbdb_workspace_dataset_root,
)
from atypemu.databases.sasbdb.normalize import (
    parse_intensity_profile,
    parse_pddf_profile,
)
from atypemu.datasets.registry import (
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)


def export_sasbdb_dataframes(
    sasbdb_root: str | Path,
    refresh: bool = False,
    emit_entry_view: bool = True,
    emit_profiles: bool = True,
) -> dict[str, Any]:
    """Export curated SASBDB source tables into Parquet datasets.

    Args:
        sasbdb_root: Source-owned SASBDB root such as ``data/sasbdb``.
        refresh: Whether to overwrite existing Parquet files.
        emit_entry_view: Whether to emit the trainer-facing entry view.
        emit_profiles: Whether to parse and export intensity/p(r) point tables.

    Returns:
        Summary dictionary describing the written dataset bundle.
    """
    source_root = Path(sasbdb_root)
    dataset_root = sasbdb_dataset_root(source_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    sasbdb_workspace_dataset_root(source_root).mkdir(parents=True, exist_ok=True)

    entries = _read_tsv(source_root, "entries.tsv")
    molecules = _read_tsv(source_root, "molecules.tsv")
    validation = _read_tsv(source_root, "validation_conditions.tsv")
    assets = _read_tsv(source_root, "assets.tsv")

    tables = {
        "entries": entries,
        "molecules": molecules,
        "validation": validation,
        "assets": assets,
    }

    if emit_profiles:
        tables["profiles"] = _build_profiles_frame(source_root, assets)
    if emit_entry_view:
        tables["entry_view"] = _build_entry_view(entries, molecules, validation)

    metadata: dict[str, DataFrameTableMeta] = {}
    for table_name, dataframe in tables.items():
        relative_path = Path(DEFAULT_WORKSPACE) / f"{table_name}.parquet"
        target_path = dataset_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not target_path.exists():
            dataframe.to_parquet(target_path, index=False)
        metadata[table_name] = DataFrameTableMeta(
            name=table_name,
            relative_path=str(relative_path),
            columns=list(dataframe.columns),
            dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            row_count=len(dataframe),
            primary_key=_primary_key(table_name),
            join_keys=_join_keys(table_name),
            workspace=DEFAULT_WORKSPACE,
            level=_level(table_name),
        )

    manifest = DataFrameDatasetManifest(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        created_at_utc=_utc_now(),
        root_kind="source",
        data_root=str(source_root.parent.resolve()),
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


def _read_tsv(source_root: Path, filename: str) -> pd.DataFrame:
    """Read one curated TSV table into a nullable dataframe."""
    table_path = sasbdb_tables_root(source_root) / filename
    dataframe = pd.read_csv(table_path, sep="\t").replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_profiles_frame(source_root: Path, assets: pd.DataFrame) -> pd.DataFrame:
    """Build one long-form profile dataframe from downloaded curve assets."""
    rows: list[dict[str, Any]] = []
    for asset in assets.to_dict(orient="records"):
        if not _is_truthy(asset.get("available")):
            continue
        asset_kind = asset.get("asset_kind")
        asset_path = asset.get("asset_path")
        if asset_kind not in {"intensity_dat", "pddf_out"} or not asset_path:
            continue
        absolute_path = source_root / str(asset_path)
        parser = (
            parse_intensity_profile
            if asset_kind == "intensity_dat"
            else parse_pddf_profile
        )
        profile_kind = "intensity" if asset_kind == "intensity_dat" else "pddf"
        x_unit = "1/angstrom" if profile_kind == "intensity" else "angstrom"
        y_unit = "intensity" if profile_kind == "intensity" else "p_of_r"
        for point in parser(absolute_path):
            rows.append(
                {
                    "entry_uid": asset["entry_uid"],
                    "source_id": SOURCE_ID,
                    "native_id": asset["native_id"],
                    "parent_uid": asset["entry_uid"],
                    "profile_kind": profile_kind,
                    "point_index": point["point_index"],
                    "x_value": point["x_value"],
                    "y_value": point["y_value"],
                    "error_value": point["error_value"],
                    "x_unit": x_unit,
                    "y_unit": y_unit,
                    "asset_path": asset_path,
                }
            )
    dataframe = pd.DataFrame(rows).replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_entry_view(
    entries: pd.DataFrame,
    molecules: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Build the trainer-facing entry view."""
    molecule_summary = _aggregate_molecules(molecules)
    view = entries.merge(
        validation, on=["entry_uid", "source_id", "native_id", "code"], how="left"
    )
    if not molecule_summary.empty:
        view = view.merge(
            molecule_summary,
            on=["entry_uid", "source_id", "native_id", "code"],
            how="left",
        )
    return view.convert_dtypes(dtype_backend="pyarrow")


def _aggregate_molecules(molecules: pd.DataFrame) -> pd.DataFrame:
    """Aggregate molecule-level metadata into entry-level summaries."""
    if molecules.empty:
        return molecules
    grouped = molecules.groupby(
        ["entry_uid", "source_id", "native_id", "code"], dropna=False
    )
    rows = []
    for keys, frame in grouped:
        names = sorted(
            {
                str(value)
                for value in frame["long_name"].dropna().tolist()
                if str(value).strip()
            }
        )
        uniprots = sorted(
            {
                str(value)
                for value in frame["uniprot_code"].dropna().tolist()
                if str(value).strip()
            }
        )
        seq_hashes = sorted(
            {
                str(value)
                for value in frame["sequence_hash"].dropna().tolist()
                if str(value).strip()
            }
        )
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "code": keys[3],
                "molecule_count": len(frame),
                "molecule_names": "|".join(names) if names else pd.NA,
                "uniprot_codes": "|".join(uniprots) if uniprots else pd.NA,
                "sequence_hashes": "|".join(seq_hashes) if seq_hashes else pd.NA,
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _primary_key(table_name: str) -> list[str]:
    """Return the canonical primary key for one SASBDB dataset table."""
    if table_name == "entries":
        return ["entry_uid"]
    if table_name == "molecules":
        return ["molecule_uid"]
    if table_name == "validation":
        return ["entry_uid"]
    if table_name == "assets":
        return ["asset_uid"]
    if table_name == "profiles":
        return ["entry_uid", "profile_kind", "point_index"]
    return ["entry_uid"]


def _join_keys(table_name: str) -> list[str]:
    """Return the canonical join keys for one SASBDB dataset table."""
    if table_name == "molecules":
        return ["entry_uid", "molecule_uid"]
    if table_name == "assets":
        return ["entry_uid", "asset_uid"]
    return ["entry_uid"]


def _level(table_name: str) -> str:
    """Return the semantic level for one SASBDB dataset table."""
    if table_name == "molecules":
        return "model"
    if table_name in {"assets", "profiles"}:
        return "derived"
    return "entry"


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_truthy(value: Any) -> bool:
    """Return whether a dataframe cell should be treated as true."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_dataset_readme(dataset_root: Path) -> None:
    """Write a short dataset README under ``data/sasbdb/datasets``."""
    (dataset_root / "README.md").write_text(
        "# SASBDB Datasets\n\n"
        "This directory stores source-owned SASBDB Parquet exports for "
        "trainer-facing and analysis-facing access.\n\n"
        "The canonical manifest is `manifest.json` and the default workspace "
        "contains `entries.parquet`, `molecules.parquet`, "
        "`validation.parquet`, `assets.parquet`, `profiles.parquet`, and "
        "`entry_view.parquet`.\n"
    )
