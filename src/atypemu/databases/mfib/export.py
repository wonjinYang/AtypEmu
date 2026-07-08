"""Parquet export helpers for source-owned MFIB dataframe bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.mfib.config import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_WORKSPACE,
    SOURCE_ID,
)
from atypemu.databases.mfib.layout import (
    mfib_dataset_root,
    mfib_tables_root,
    mfib_workspace_dataset_root,
)
from atypemu.datasets.registry import (
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
)


def export_mfib_dataframes(
    mfib_root: str | Path,
    refresh: bool = False,
    emit_entry_view: bool = True,
) -> dict[str, Any]:
    """Export curated MFIB tables into source-owned Parquet datasets."""
    source_root = Path(mfib_root)
    dataset_root = mfib_dataset_root(source_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    mfib_workspace_dataset_root(source_root).mkdir(parents=True, exist_ok=True)

    tables = {
        "entries": _read_tsv(source_root, "entries", "entries.tsv"),
        "chains": _read_tsv(source_root, "chains", "chains.tsv"),
        "regions": _read_tsv(source_root, "regions", "regions.tsv"),
        "evidence": _read_tsv(source_root, "evidence", "evidence.tsv"),
        "go_terms": _read_tsv(source_root, "go_terms", "go_terms.tsv"),
        "related_entries": _read_tsv(
            source_root,
            "related_entries",
            "related_entries.tsv",
        ),
        "crossrefs": _read_tsv(source_root, "crossrefs", "crossrefs.tsv"),
        "search_index": _read_tsv(source_root, "search_index", "search_index.tsv"),
        "assets": _read_tsv(source_root, "assets", "assets.tsv"),
        "cif_entry_features": _read_tsv(
            source_root,
            "cif_entry_features",
            "cif_entry_features.tsv",
        ),
        "cif_chain_features": _read_tsv(
            source_root,
            "cif_chain_features",
            "cif_chain_features.tsv",
        ),
    }
    for table_name in [
        "interface_pairs",
        "interface_residues",
        "geometry_entry_features",
    ]:
        source_path = mfib_tables_root(source_root) / f"{table_name}.tsv"
        if source_path.exists():
            tables[table_name] = _read_tsv(source_root, table_name, f"{table_name}.tsv")
    if emit_entry_view:
        tables["entry_view"] = _build_entry_view(
            entries=tables["entries"],
            chains=tables["chains"],
            evidence=tables["evidence"],
            cif_entry_features=tables["cif_entry_features"],
            geometry_entry_features=tables.get("geometry_entry_features"),
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


def _read_tsv(source_root: Path, table_name: str, filename: str) -> pd.DataFrame:
    """Read one curated MFIB TSV table into a nullable dataframe."""
    table_path = mfib_tables_root(source_root) / filename
    dataframe = pd.read_csv(table_path, sep="\t", low_memory=False).replace(
        {r"^\s*$": pd.NA},
        regex=True,
    )
    for column in _string_columns(table_name, dataframe.columns):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].map(_string_or_na)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _string_columns(table_name: str, columns: pd.Index) -> set[str]:
    """Return one set of identifier-like columns that should remain strings."""
    string_columns = {
        column
        for column in columns
        if column.endswith("_uid") or column.endswith("_path")
    }
    string_columns.update(
        {
            "accession",
            "native_id",
            "pdb_id",
            "chain_id",
            "residue_id",
            "resname",
            "asset_kind",
            "search_namespace",
            "search_value_raw",
            "search_value_normalized",
            "scope",
            "namespace",
            "go_accession",
            "go_name",
            "related_accession",
            "xref_namespace",
            "xref_id",
        }
    )
    if table_name == "entries":
        string_columns.update(
            {
                "publication_pmid",
                "publication_year",
                "publication_volume",
                "publication_issue",
                "publication_pages",
            }
        )
    return string_columns


def _string_or_na(value: Any) -> str | Any:
    """Return one identifier-like value as string while preserving nulls."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _build_entry_view(
    entries: pd.DataFrame,
    chains: pd.DataFrame,
    evidence: pd.DataFrame,
    cif_entry_features: pd.DataFrame,
    geometry_entry_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the compact MFIB entry-level trainer view."""
    view = entries.copy()
    chain_summary = _aggregate_chains(chains)
    evidence_summary = _aggregate_evidence(evidence)
    for frame in [
        chain_summary,
        evidence_summary,
        cif_entry_features,
        geometry_entry_features,
    ]:
        if frame is None:
            continue
        if frame.empty:
            continue
        view = view.merge(
            frame,
            on=["entry_uid", "source_id", "native_id", "accession"],
            how="left",
        )
    return view.convert_dtypes(dtype_backend="pyarrow")


def _aggregate_chains(chains: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chain metadata into one accession-level summary."""
    if chains.empty:
        return chains
    rows = []
    grouped = chains.groupby(
        ["entry_uid", "source_id", "native_id", "accession"], dropna=False
    )
    for keys, frame in grouped:
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "accession": keys[3],
                "chain_count": len(frame),
                "unique_uniprot_ids": _pipe_join(frame["uniprot_id"]),
                "unique_chain_organisms": _pipe_join(frame["source_organism"]),
                "unique_sequence_hashes": _pipe_join(frame["sequence_hash"]),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _aggregate_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evidence coverage into one accession-level summary."""
    if evidence.empty:
        return evidence
    rows = []
    grouped = evidence.groupby(
        ["entry_uid", "source_id", "native_id", "accession"], dropna=False
    )
    for keys, frame in grouped:
        complex_rows = frame.loc[frame["scope"] == "complex"]
        chain_rows = frame.loc[frame["scope"] == "chain"]
        rows.append(
            {
                "entry_uid": keys[0],
                "source_id": keys[1],
                "native_id": keys[2],
                "accession": keys[3],
                "chain_evidence_count": len(chain_rows),
                "complex_evidence_present": not complex_rows.empty,
                "sequence_domain_present": (
                    complex_rows["sequence_domain"].notna().any()
                    if "sequence_domain" in complex_rows
                    else False
                ),
            }
        )
    return pd.DataFrame(rows).convert_dtypes(dtype_backend="pyarrow")


def _pipe_join(series: pd.Series) -> str | Any:
    """Return deterministic pipe-joined non-null values."""
    values = sorted(
        {str(value) for value in series.dropna().tolist() if str(value).strip()}
    )
    return "|".join(values) if values else pd.NA


def _primary_key(table_name: str) -> list[str]:
    """Return the canonical primary key for one MFIB dataset table."""
    if table_name == "entries":
        return ["entry_uid"]
    if table_name == "chains":
        return ["chain_uid"]
    if table_name == "interface_pairs":
        return ["pair_uid"]
    if table_name == "interface_residues":
        return ["pair_uid", "chain_uid", "residue_id"]
    if table_name == "geometry_entry_features":
        return ["entry_uid"]
    if table_name == "regions":
        return ["chain_uid", "region_type", "region_start", "region_end"]
    if table_name == "evidence":
        return ["entry_uid", "scope", "chain_id"]
    if table_name == "assets":
        return ["asset_uid"]
    if table_name == "cif_chain_features":
        return ["chain_uid"]
    return ["entry_uid"]


def _join_keys(table_name: str) -> list[str]:
    """Return the canonical join keys for one MFIB dataset table."""
    if table_name in {
        "chains",
        "regions",
        "evidence",
        "assets",
        "cif_entry_features",
        "geometry_entry_features",
    }:
        return ["entry_uid"]
    if table_name == "interface_pairs":
        return ["entry_uid", "pair_uid"]
    if table_name == "interface_residues":
        return ["entry_uid", "pair_uid", "chain_uid"]
    if table_name == "cif_chain_features":
        return ["entry_uid", "chain_uid"]
    return ["entry_uid"]


def _level(table_name: str) -> str:
    """Return the semantic level for one MFIB dataset table."""
    if table_name in {"chains", "regions", "cif_chain_features", "interface_residues"}:
        return "model"
    if table_name in {"interface_pairs"}:
        return "asset"
    if table_name in {"crossrefs", "search_index", "go_terms"}:
        return "tag"
    if table_name in {"assets"}:
        return "asset"
    return "entry"


def _utc_now() -> str:
    """Return the current UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_dataset_readme(dataset_root: Path) -> None:
    """Write a short README for the MFIB dataset bundle."""
    readme_path = dataset_root / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "# MFIB Dataframes\n\n"
        "This directory stores source-owned Parquet tables derived from the "
        "MFIB download bundles. The `default/` workspace contains curated "
        "entry, chain, evidence, search, asset, CIF feature, and optional "
        "derived geometry tables.\n"
    )
