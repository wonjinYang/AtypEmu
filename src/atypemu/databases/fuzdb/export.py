"""Parquet export helpers for source-owned FuzDB dataframe bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.fuzdb.config import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_WORKSPACE,
    SOURCE_ID,
)
from atypemu.databases.fuzdb.layout import (
    fuzdb_dataset_root,
    fuzdb_tables_root,
    fuzdb_workspace_dataset_root,
)
from atypemu.datasets.registry import (
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)


def export_fuzdb_dataframes(
    fuzdb_root: str | Path,
    refresh: bool = False,
    emit_entry_view: bool = True,
) -> dict[str, Any]:
    """Export curated FuzDB tables into source-owned Parquet datasets."""
    source_root = Path(fuzdb_root)
    dataset_root = fuzdb_dataset_root(source_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    fuzdb_workspace_dataset_root(source_root).mkdir(parents=True, exist_ok=True)

    tables = {
        "entries": _read_tsv(source_root, "entries.tsv"),
        "fuzzy_regions": _read_tsv(source_root, "fuzzy_regions.tsv"),
        "structure_links": _read_tsv(source_root, "structure_links.tsv"),
        "functional_sites": _read_tsv(source_root, "functional_sites.tsv"),
        "ptm_sites": _read_tsv(source_root, "ptm_sites.tsv"),
        "isoforms": _read_tsv(source_root, "isoforms.tsv"),
        "condensates": _read_tsv(source_root, "condensates.tsv"),
        "references": _read_tsv(source_root, "references.tsv"),
        "crossrefs": _read_tsv(source_root, "crossrefs.tsv"),
        "search_index": _read_tsv(source_root, "search_index.tsv"),
    }
    if emit_entry_view:
        tables["entry_view"] = _build_entry_view(
            entries=tables["entries"],
            fuzzy_regions=tables["fuzzy_regions"],
            functional_sites=tables["functional_sites"],
            references=tables["references"],
            condensates=tables["condensates"],
            crossrefs=tables["crossrefs"],
        )

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
    table_path = fuzdb_tables_root(source_root) / filename
    dataframe = pd.read_csv(table_path, sep="\t", low_memory=False).replace(
        {r"^\s*$": pd.NA},
        regex=True,
    )
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _build_entry_view(
    entries: pd.DataFrame,
    fuzzy_regions: pd.DataFrame,
    functional_sites: pd.DataFrame,
    references: pd.DataFrame,
    condensates: pd.DataFrame,
    crossrefs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact FuzDB entry-level view."""
    view = entries.copy()
    for frame in [
        _aggregate_region_counts(fuzzy_regions),
        _aggregate_functional_site_counts(functional_sites),
        _aggregate_reference_counts(references),
        _aggregate_condensate_counts(condensates),
        _aggregate_crossrefs(crossrefs),
    ]:
        if frame.empty:
            continue
        view = view.merge(
            frame,
            on=["entry_uid", "source_id", "native_id", "fc_id"],
            how="left",
        )
    return view.convert_dtypes(dtype_backend="pyarrow")


def _aggregate_region_counts(fuzzy_regions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fuzzy-region counts into one entry-level summary."""
    if fuzzy_regions.empty:
        return fuzzy_regions
    rows = []
    grouped = fuzzy_regions.groupby(
        ["entry_uid", "source_id", "native_id", "fc_id"],
        dropna=False,
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "fc_id": keys[3],
                "fuzzy_region_count": len(frame),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _aggregate_functional_site_counts(functional_sites: pd.DataFrame) -> pd.DataFrame:
    """Aggregate functional-site counts into one entry-level summary."""
    if functional_sites.empty:
        return functional_sites
    rows = []
    grouped = functional_sites.groupby(
        ["entry_uid", "source_id", "native_id", "fc_id"],
        dropna=False,
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "fc_id": keys[3],
                "functional_site_count": len(frame),
                "functional_site_types": _pipe_join(frame["site_type"]),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _aggregate_reference_counts(references: pd.DataFrame) -> pd.DataFrame:
    """Aggregate citation counts into one entry-level summary."""
    if references.empty:
        return references
    rows = []
    grouped = references.groupby(
        ["entry_uid", "source_id", "native_id", "fc_id"],
        dropna=False,
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "fc_id": keys[3],
                "reference_count": len(frame),
                "reference_scopes": _pipe_join(frame["reference_scope"]),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _aggregate_condensate_counts(condensates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate condensate rows into one entry-level summary."""
    if condensates.empty:
        return condensates
    rows = []
    grouped = condensates.groupby(
        ["entry_uid", "source_id", "native_id", "fc_id"],
        dropna=False,
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "fc_id": keys[3],
                "condensate_count": len(frame),
                "llps_roles": _pipe_join(frame["llps_role"]),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _aggregate_crossrefs(crossrefs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external identifiers into one compact entry-level summary."""
    if crossrefs.empty:
        return crossrefs
    rows = []
    grouped = crossrefs.groupby(
        ["entry_uid", "source_id", "native_id", "fc_id"],
        dropna=False,
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "fc_id": keys[3],
                "xref_namespaces": _pipe_join(frame["xref_namespace"]),
                "xref_count": len(frame),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _pipe_join(values: pd.Series) -> str | Any:
    """Return one pipe-joined string while preserving null when empty."""
    filtered = sorted(
        {str(value).strip() for value in values.dropna().tolist() if str(value).strip()}
    )
    return "|".join(filtered) if filtered else pd.NA


def _primary_key(table_name: str) -> list[str]:
    """Return the canonical primary key for one FuzDB dataset table."""
    if table_name == "entries":
        return ["entry_uid"]
    if table_name == "fuzzy_regions":
        return ["region_uid"]
    if table_name == "structure_links":
        return ["link_uid"]
    if table_name == "functional_sites":
        return ["site_uid"]
    if table_name == "ptm_sites":
        return ["ptm_uid"]
    if table_name == "isoforms":
        return ["isoform_uid"]
    if table_name == "condensates":
        return ["condensate_uid"]
    if table_name == "references":
        return ["reference_uid"]
    if table_name == "crossrefs":
        return ["xref_uid"]
    return ["entry_uid", "search_namespace", "search_value_normalized"]


def _join_keys(table_name: str) -> list[str]:
    """Return the canonical join keys for one FuzDB dataset table."""
    if table_name == "entries":
        return ["entry_uid"]
    return ["entry_uid", "parent_uid"]


def _level(table_name: str) -> str:
    """Return the semantic level for one FuzDB dataset table."""
    if table_name == "entries":
        return "entry"
    return "annotation"


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_dataset_readme(dataset_root: Path) -> None:
    """Write one concise README for the source-owned dataset bundle."""
    (dataset_root / "README.md").write_text(
        "# FuzDB Dataset Bundle\n\n"
        "This directory stores source-owned Parquet exports for FuzDB.\n\n"
        "## Workspace\n\n"
        "- `default/`\n"
        "  - accession-level and annotation-level Parquet tables\n"
        "- `manifest.json`\n"
        "  - source dataset contract consumed by `SourceDataFrameRegistry`\n\n"
        "## Notes\n\n"
        "FuzDB is exported as a fuzzy-complex annotation and benchmark source. "
        "The canonical trainer-facing table is `default/entry_view.parquet`.\n"
    )
