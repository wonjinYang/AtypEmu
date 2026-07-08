"""Cross-source metadata dataframe export helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atypemu.databases.common.bootstrap import meta_datasets_root
from atypemu.databases.common.catalog import SOURCE_CATALOG
from atypemu.datasets.registry import (
    DEFAULT_WORKSPACE,
    DataFrameDatasetManifest,
    DataFrameTableMeta,
    DataFrameWorkspaceMeta,
    SourceDataFrameRegistry,
)


def export_meta_dataframes(
    data_root: str | Path, refresh: bool = False
) -> dict[str, Any]:
    """Export source-agnostic metadata Parquet tables under ``data/meta/datasets``.

    Args:
        data_root: AtypEmu data root.
        refresh: When ``True``, overwrite existing Parquet files.

    Returns:
        Summary dictionary describing the generated metadata bundle.
    """
    data_root_path = Path(data_root)
    meta_root = data_root_path / "meta"
    dataset_root = meta_datasets_root(data_root_path)
    meta_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    source_catalog = _build_source_catalog_frame(data_root_path)
    bmrb_registry = _load_source_registry(data_root_path, "bmrb")
    ped_catalog_registry = _load_source_registry(
        data_root_path, "ped", workspace="catalog"
    )
    ped_benchmark_registry = _load_source_registry(
        data_root_path,
        "ped",
        workspace="benchmark",
    )
    sasbdb_registry = _load_source_registry(data_root_path, "sasbdb")
    mfib_registry = _load_source_registry(data_root_path, "mfib")
    fuzdb_registry = _load_source_registry(data_root_path, "fuzdb")

    entity_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    crossref_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    if bmrb_registry is not None:
        bmrb_entries = bmrb_registry.load_table("entries")
        bmrb_tags = bmrb_registry.load_table("tags")
        entity_rows.extend(_build_bmrb_entities(bmrb_entries))
        tag_rows.extend(bmrb_tags.to_dict(orient="records"))
        asset_rows.extend(_build_bmrb_assets(bmrb_entries))

    if ped_catalog_registry is not None:
        ped_entries = ped_catalog_registry.load_table("entries")
        ped_bridges = ped_catalog_registry.load_table("bridges")
        ped_experimental_tags = ped_catalog_registry.load_table("experimental_tags")
        ped_structural_tags = ped_catalog_registry.load_table("structural_tags")
        entity_rows.extend(_build_ped_entities(ped_entries))
        crossref_rows.extend(_build_ped_crossrefs(ped_entries))
        tag_rows.extend(
            _namespaced_tags(
                ped_experimental_tags,
                namespace="experimental_procedure",
            )
        )
        tag_rows.extend(
            _namespaced_tags(
                ped_structural_tags,
                namespace="structural_calculation",
            )
        )
        link_rows.extend(_build_ped_bmrb_links(ped_bridges))
        split_rows.extend(_build_ped_split_rows(ped_bridges))

    if ped_benchmark_registry is not None:
        ped_models = ped_benchmark_registry.load_table("models")
        asset_rows.extend(_build_ped_assets(ped_models))

    if sasbdb_registry is not None:
        sasbdb_entries = sasbdb_registry.load_table("entries")
        sasbdb_molecules = sasbdb_registry.load_table("molecules")
        sasbdb_assets = sasbdb_registry.load_table("assets")
        entity_rows.extend(_build_sasbdb_entities(sasbdb_entries, sasbdb_molecules))
        asset_rows.extend(_build_sasbdb_assets(sasbdb_assets))
        crossref_rows.extend(_build_sasbdb_crossrefs(sasbdb_entries, sasbdb_molecules))
        tag_rows.extend(_build_sasbdb_tags(sasbdb_entries, sasbdb_molecules))
        link_rows.extend(
            _build_sasbdb_links(
                molecules=sasbdb_molecules,
                entity_rows=entity_rows,
                crossref_rows=crossref_rows,
            )
        )
        split_rows.extend(_build_sasbdb_split_rows(sasbdb_entries))

    if mfib_registry is not None:
        mfib_entries = mfib_registry.load_table("entries")
        mfib_chains = mfib_registry.load_table("chains")
        mfib_assets = mfib_registry.load_table("assets")
        mfib_crossrefs = mfib_registry.load_table("crossrefs")
        mfib_geometry = (
            mfib_registry.load_table("geometry_entry_features")
            if "geometry_entry_features" in mfib_registry.available_tables()
            else None
        )
        entity_rows.extend(
            _build_mfib_entities(
                mfib_entries,
                mfib_chains,
                geometry_entry_features=mfib_geometry,
            )
        )
        asset_rows.extend(_build_mfib_assets(mfib_assets))
        crossref_rows.extend(_build_mfib_crossrefs(mfib_crossrefs))
        tag_rows.extend(_build_mfib_tags(mfib_entries))
        link_rows.extend(
            _build_mfib_links(
                chains=mfib_chains,
                entity_rows=entity_rows,
                crossref_rows=crossref_rows,
            )
        )
        split_rows.extend(_build_mfib_split_rows(mfib_entries))

    if fuzdb_registry is not None:
        fuzdb_entries = fuzdb_registry.load_table("entries")
        fuzdb_crossrefs = fuzdb_registry.load_table("crossrefs")
        fuzdb_condensates = fuzdb_registry.load_table("condensates")
        entity_rows.extend(_build_fuzdb_entities(fuzdb_entries))
        crossref_rows.extend(_build_fuzdb_crossrefs(fuzdb_crossrefs))
        tag_rows.extend(_build_fuzdb_tags(fuzdb_entries, fuzdb_condensates))
        link_rows.extend(
            _build_fuzdb_links(
                crossrefs=fuzdb_crossrefs,
                entity_rows=entity_rows,
                crossref_rows=crossref_rows,
            )
        )
        split_rows.extend(_build_fuzdb_split_rows(fuzdb_entries))

    integrated_split_path = (
        data_root_path / "integrated" / "splits" / "default_split.json"
    )
    if integrated_split_path.exists():
        split_rows.extend(_build_integrated_split_rows(integrated_split_path))

    tables = {
        "source_catalog": _frame(source_catalog.to_dict(orient="records")),
        "entities": _normalize_entities_frame(_frame(entity_rows)),
        "assets": _frame(asset_rows),
        "crossrefs": _frame(crossref_rows),
        "tags": _frame(tag_rows),
        "links": _frame(link_rows),
        "splits": _frame(split_rows),
    }

    metadata: dict[str, DataFrameTableMeta] = {}
    for table_name, dataframe in tables.items():
        target_path = dataset_root / f"{table_name}.parquet"
        if refresh or not target_path.exists():
            dataframe.to_parquet(target_path, index=False)
        metadata[table_name] = DataFrameTableMeta(
            name=table_name,
            relative_path=target_path.relative_to(dataset_root).as_posix(),
            columns=list(dataframe.columns),
            dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            row_count=len(dataframe),
            primary_key=_primary_key(table_name),
            join_keys=_join_keys(table_name),
            workspace=DEFAULT_WORKSPACE,
            level=_level(table_name),
        )

    manifest = DataFrameDatasetManifest(
        dataset_name="meta",
        dataset_version="1.0.0",
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        root_kind="meta",
        data_root=str(data_root_path.resolve()),
        dataset_root=str(dataset_root.resolve()),
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
    manifest_path = dataset_root / "manifest.json"
    manifest.to_json(manifest_path)
    _write_readme(meta_root)

    return {
        "meta_root": str(meta_root),
        "manifest_path": str(manifest_path),
        "tables_written": len(tables),
    }


def _load_source_registry(
    data_root: Path,
    source_id: str,
    workspace: str | None = None,
) -> SourceDataFrameRegistry | None:
    """Load one source registry when its manifest exists."""
    manifest_path = data_root / source_id / "datasets" / "manifest.json"
    if not manifest_path.exists():
        return None
    return SourceDataFrameRegistry.from_manifest(manifest_path, workspace=workspace)


def _build_source_catalog_frame(data_root: Path) -> pd.DataFrame:
    """Build the source catalog dataframe from the common source registry."""
    rows: list[dict[str, Any]] = []
    for source_name, payload in sorted(
        SOURCE_CATALOG.items(),
        key=lambda item: item[1]["collection_order"],
    ):
        source_id = source_name.lower()
        source_root = data_root / source_id
        rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "source_kind": "database",
                "tier": payload["tier"],
                "collection_order": payload["collection_order"],
                "collection_role": payload["collection_role"],
                "default_use_tiers": "|".join(payload["default_use_tiers"]),
                "default_state_classes": "|".join(payload["default_state_classes"]),
                "root_path": str(source_root),
                "has_downloads": (source_root / "downloads").exists(),
                "has_tables": (source_root / "tables").exists(),
                "has_datasets": (source_root / "datasets").exists(),
                "has_assets": (source_root / "assets").exists(),
                "status": "ready" if source_root.exists() else "missing",
            }
        )
    return _frame(rows)


def _build_bmrb_entities(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-entity rows for BMRB entries."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "bmrb",
                "native_id": row["native_id"],
                "display_name": row["native_id"],
                "sequence": pd.NA,
                "sequence_hash": pd.NA,
                "workspace": DEFAULT_WORKSPACE,
                "entity_kind": "nmr_entry",
                "status": row.get("status"),
            }
        )
    return rows


def _build_bmrb_assets(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-asset rows for BMRB bundle assets."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        asset_path = row.get("asset_path")
        if not asset_path:
            continue
        rows.append(
            {
                "asset_uid": f"bmrb_bundle:{row['native_id']}",
                "entity_uid": row["entry_uid"],
                "source_id": "bmrb",
                "native_id": row["native_id"],
                "parent_uid": row["entry_uid"],
                "asset_kind": "bundle_json",
                "asset_path": asset_path,
                "workspace": DEFAULT_WORKSPACE,
            }
        )
    return rows


def _build_ped_entities(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-entity rows for PED entries."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        sequence = row.get("sequence")
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "ped",
                "native_id": row["native_id"],
                "display_name": row.get("protein_name"),
                "sequence": sequence,
                "sequence_hash": _sequence_hash(sequence),
                "workspace": row.get("workspace"),
                "entity_kind": "ensemble_entry",
                "status": "ready",
            }
        )
    return rows


def _build_ped_crossrefs(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build cross-reference rows for PED entries."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        for accession in _split_pipe(row.get("uniprot_accessions")):
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "ped",
                    "native_id": row["native_id"],
                    "namespace": "UniProt",
                    "xref_value": str(accession),
                    "source_field": "uniprot_accessions",
                }
            )
        for bmrb_id in _split_pipe(row.get("bmrb_ids")):
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "ped",
                    "native_id": row["native_id"],
                    "namespace": "BMRB",
                    "xref_value": str(bmrb_id),
                    "source_field": "bmrb_ids",
                }
            )
    return rows


def _namespaced_tags(dataframe: pd.DataFrame, namespace: str) -> list[dict[str, Any]]:
    """Attach one namespace to a long-form tag dataframe."""
    rows = []
    for row in dataframe.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": row["source_id"],
                "native_id": row["native_id"],
                "namespace": namespace,
                "tag": row["tag"],
            }
        )
    return rows


def _build_ped_assets(models: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-asset rows for PED benchmark models."""
    rows: list[dict[str, Any]] = []
    for row in models.to_dict(orient="records"):
        rows.append(
            {
                "asset_uid": f"ped_model:{row['model_id']}",
                "entity_uid": row["entry_uid"],
                "source_id": "ped",
                "native_id": row["native_id"],
                "parent_uid": row["parent_uid"],
                "asset_kind": "model_pdb",
                "asset_path": row.get("asset_path"),
                "workspace": row.get("workspace"),
            }
        )
    return rows


def _build_ped_bmrb_links(bridges: pd.DataFrame) -> list[dict[str, Any]]:
    """Build curated PED-to-BMRB link rows."""
    rows: list[dict[str, Any]] = []
    for row in bridges.to_dict(orient="records"):
        for bmrb_id in _split_pipe(row.get("overlapping_bmrb_ids")):
            rows.append(
                {
                    "link_uid": f"ped_bmrb:{row['native_id']}:{bmrb_id}",
                    "left_entity_uid": row["entry_uid"],
                    "right_entity_uid": f"bmrb:{bmrb_id}",
                    "left_source_id": "ped",
                    "right_source_id": "bmrb",
                    "link_type": "ped_bmrb_overlap",
                    "evidence": row.get("overlap_sources"),
                    "workspace": row.get("workspace"),
                }
            )
    return rows


def _build_ped_split_rows(bridges: pd.DataFrame) -> list[dict[str, Any]]:
    """Build validation-routing rows for PED entities."""
    rows: list[dict[str, Any]] = []
    for row in bridges.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "ped",
                "split_namespace": "ped_recommended_role",
                "split_value": row.get("recommended_role"),
                "reason": row.get("validation_group"),
            }
        )
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "ped",
                "split_namespace": "ped_overlap_split",
                "split_value": row.get("split_assignment"),
                "reason": row.get("overlapping_bmrb_ids"),
            }
        )
    return rows


def _build_sasbdb_entities(
    entries: pd.DataFrame,
    molecules: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build meta-entity rows for SASBDB accessions."""
    molecule_groups = {
        entry_uid: frame
        for entry_uid, frame in molecules.groupby("entry_uid", dropna=False)
    }
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        molecule_frame = molecule_groups.get(row["entry_uid"])
        sequences = _unique_non_null(
            molecule_frame["sequence"].tolist() if molecule_frame is not None else []
        )
        sequence_hashes = _unique_non_null(
            molecule_frame["sequence_hash"].tolist()
            if molecule_frame is not None
            else []
        )
        names = _unique_non_null(
            molecule_frame["long_name"].tolist() if molecule_frame is not None else []
        )
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "sasbdb",
                "native_id": row["native_id"],
                "display_name": names[0] if names else row["native_id"],
                "sequence": sequences[0] if len(sequences) == 1 else pd.NA,
                "sequence_hash": (
                    sequence_hashes[0] if len(sequence_hashes) == 1 else pd.NA
                ),
                "workspace": DEFAULT_WORKSPACE,
                "entity_kind": "saxs_entry",
                "status": row.get("status"),
            }
        )
    return rows


def _build_sasbdb_assets(assets: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-asset rows for SASBDB source assets."""
    rows: list[dict[str, Any]] = []
    for row in assets.to_dict(orient="records"):
        if not _truthy(row.get("available")):
            continue
        rows.append(
            {
                "asset_uid": row["asset_uid"],
                "entity_uid": row["entry_uid"],
                "source_id": "sasbdb",
                "native_id": row["native_id"],
                "parent_uid": row["entry_uid"],
                "asset_kind": row.get("asset_kind"),
                "asset_path": row.get("asset_path"),
                "workspace": DEFAULT_WORKSPACE,
            }
        )
    return rows


def _build_sasbdb_crossrefs(
    entries: pd.DataFrame,
    molecules: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build cross-reference rows for SASBDB accessions."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        for namespace, field in [
            ("DOI", "publication_doi"),
            ("PMID", "publication_pmid"),
        ]:
            value = row.get(field)
            if value is None or pd.isna(value) or str(value) == "":
                continue
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "sasbdb",
                    "native_id": row["native_id"],
                    "namespace": namespace,
                    "xref_value": str(value),
                    "source_field": field,
                }
            )

    for row in molecules.to_dict(orient="records"):
        value = row.get("uniprot_code")
        if value is None or pd.isna(value) or str(value) == "":
            continue
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "sasbdb",
                "native_id": row["native_id"],
                "namespace": "UniProt",
                "xref_value": str(value),
                "source_field": "uniprot_code",
            }
        )
    return rows


def _build_sasbdb_tags(
    entries: pd.DataFrame,
    molecules: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build namespaced tag rows for SASBDB entries."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        tag = row.get("type_of_curve")
        if tag is None or pd.isna(tag) or str(tag) == "":
            continue
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "sasbdb",
                "native_id": row["native_id"],
                "namespace": "type_of_curve",
                "tag": tag,
            }
        )
    for row in molecules.to_dict(orient="records"):
        for namespace, field in [
            ("molecular_type", "molecular_type"),
            ("oligomerization", "oligomerization"),
        ]:
            tag = row.get(field)
            if tag is None or pd.isna(tag) or str(tag) == "":
                continue
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "sasbdb",
                    "native_id": row["native_id"],
                    "namespace": namespace,
                    "tag": tag,
                }
            )
    return rows


def _build_sasbdb_links(
    molecules: pd.DataFrame,
    entity_rows: list[dict[str, Any]],
    crossref_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build SASBDB links to other sources via sequence hash or UniProt."""
    candidate_entities = [row for row in entity_rows if row["source_id"] != "sasbdb"]
    candidate_crossrefs = [
        row
        for row in crossref_rows
        if row["source_id"] != "sasbdb" and row.get("namespace") == "UniProt"
    ]
    by_sequence = _multi_map(candidate_entities, "sequence_hash", "entity_uid")
    by_uniprot = _multi_map(candidate_crossrefs, "xref_value", "entity_uid")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in molecules.to_dict(orient="records"):
        left_entity_uid = row["entry_uid"]
        sequence_hash = row.get("sequence_hash")
        if sequence_hash is not None and not pd.isna(sequence_hash):
            for right_entity_uid in by_sequence.get(str(sequence_hash), []):
                key = (left_entity_uid, right_entity_uid, "sequence_hash_match")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "link_uid": (
                            f"sasbdb_link:{row['native_id']}:{right_entity_uid}:seq"
                        ),
                        "left_entity_uid": left_entity_uid,
                        "right_entity_uid": right_entity_uid,
                        "left_source_id": "sasbdb",
                        "right_source_id": right_entity_uid.split(":", maxsplit=1)[0],
                        "link_type": "sequence_hash_match",
                        "evidence": sequence_hash,
                        "workspace": DEFAULT_WORKSPACE,
                    }
                )
            if any(key[0] == left_entity_uid for key in seen):
                continue

        uniprot = row.get("uniprot_code")
        if uniprot is None or pd.isna(uniprot):
            continue
        for right_entity_uid in by_uniprot.get(str(uniprot), []):
            key = (left_entity_uid, right_entity_uid, "uniprot_match")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "link_uid": (
                        f"sasbdb_link:{row['native_id']}:{right_entity_uid}:uniprot"
                    ),
                    "left_entity_uid": left_entity_uid,
                    "right_entity_uid": right_entity_uid,
                    "left_source_id": "sasbdb",
                    "right_source_id": right_entity_uid.split(":", maxsplit=1)[0],
                    "link_type": "uniprot_match",
                    "evidence": uniprot,
                    "workspace": DEFAULT_WORKSPACE,
                }
            )
    return rows


def _build_sasbdb_split_rows(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build validation-only split rows for SASBDB."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "sasbdb",
                "split_namespace": "sasbdb_recommended_role",
                "split_value": "validation_only",
                "reason": "saxs_validation",
            }
        )
    return rows


def _build_mfib_entities(
    entries: pd.DataFrame,
    chains: pd.DataFrame,
    geometry_entry_features: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build meta-entity rows for MFIB accessions."""
    chain_groups = {
        entry_uid: frame
        for entry_uid, frame in chains.groupby("entry_uid", dropna=False)
    }
    geometry_by_entry: dict[str, dict[str, Any]] = {}
    if geometry_entry_features is not None and not geometry_entry_features.empty:
        geometry_by_entry = {
            str(row["entry_uid"]): row
            for row in geometry_entry_features.to_dict(orient="records")
        }
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        chain_frame = chain_groups.get(row["entry_uid"])
        geometry_row = geometry_by_entry.get(str(row["entry_uid"]), {})
        sequences = _unique_non_null(
            chain_frame["uniprot_sequence"].tolist() if chain_frame is not None else []
        )
        sequence_hashes = _unique_non_null(
            chain_frame["sequence_hash"].tolist() if chain_frame is not None else []
        )
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "mfib",
                "native_id": row["native_id"],
                "display_name": row.get("name") or row["native_id"],
                "sequence": sequences[0] if len(sequences) == 1 else pd.NA,
                "sequence_hash": (
                    sequence_hashes[0] if len(sequence_hashes) == 1 else pd.NA
                ),
                "workspace": DEFAULT_WORKSPACE,
                "entity_kind": "bound_disorder_complex",
                "geometry_available": geometry_row.get("geometry_available", pd.NA),
                "geometry_pair_count": geometry_row.get("pair_count", pd.NA),
                "geometry_max_interface_area": geometry_row.get(
                    "max_interface_area",
                    pd.NA,
                ),
                "geometry_max_contact_count": geometry_row.get(
                    "max_contact_count",
                    pd.NA,
                ),
                "status": "ready",
            }
        )
    return rows


def _build_mfib_assets(assets: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-asset rows for MFIB CIF assets."""
    rows: list[dict[str, Any]] = []
    for row in assets.to_dict(orient="records"):
        if not _truthy(row.get("available")):
            continue
        rows.append(
            {
                "asset_uid": row["asset_uid"],
                "entity_uid": row["entry_uid"],
                "source_id": "mfib",
                "native_id": row["native_id"],
                "parent_uid": row["parent_uid"],
                "asset_kind": row.get("asset_kind"),
                "asset_path": row.get("asset_path"),
                "workspace": DEFAULT_WORKSPACE,
            }
        )
    return rows


def _build_mfib_crossrefs(crossrefs: pd.DataFrame) -> list[dict[str, Any]]:
    """Build MFIB cross-reference rows for the meta registry."""
    rows: list[dict[str, Any]] = []
    for row in crossrefs.to_dict(orient="records"):
        value = row.get("xref_value")
        if value is None or pd.isna(value) or str(value) == "":
            continue
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "mfib",
                "native_id": row["native_id"],
                "namespace": row.get("namespace"),
                "xref_value": str(value),
                "source_field": row.get("source_field"),
            }
        )
    return rows


def _build_mfib_tags(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build namespaced MFIB tags for class, method, and evidence routing."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        for namespace, field in [
            ("class", "class_name"),
            ("subclass", "subclass_name"),
            ("exp_method", "exp_method"),
            ("assembly", "assembly"),
            ("evidence_level", "evidence_level"),
            ("sequence_domain", "sequence_domain"),
        ]:
            tag = row.get(field)
            if tag is None or pd.isna(tag) or str(tag) == "":
                continue
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "mfib",
                    "native_id": row["native_id"],
                    "namespace": namespace,
                    "tag": tag,
                }
            )
    return rows


def _build_mfib_links(
    chains: pd.DataFrame,
    entity_rows: list[dict[str, Any]],
    crossref_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build MFIB links to other sources via UniProt or sequence hash."""
    candidate_entities = [row for row in entity_rows if row["source_id"] != "mfib"]
    candidate_crossrefs = [
        row
        for row in crossref_rows
        if row["source_id"] != "mfib" and row.get("namespace") == "UniProt"
    ]
    by_sequence = _multi_map(candidate_entities, "sequence_hash", "entity_uid")
    by_uniprot = _multi_map(candidate_crossrefs, "xref_value", "entity_uid")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in chains.to_dict(orient="records"):
        left_entity_uid = row["entry_uid"]
        uniprot = row.get("uniprot_id")
        if uniprot is not None and not pd.isna(uniprot):
            for right_entity_uid in by_uniprot.get(str(uniprot), []):
                key = (left_entity_uid, right_entity_uid, "uniprot_match")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "link_uid": (
                            f"mfib_link:{row['native_id']}:{right_entity_uid}:uniprot"
                        ),
                        "left_entity_uid": left_entity_uid,
                        "right_entity_uid": right_entity_uid,
                        "left_source_id": "mfib",
                        "right_source_id": right_entity_uid.split(":", maxsplit=1)[0],
                        "link_type": "uniprot_match",
                        "evidence": uniprot,
                        "workspace": DEFAULT_WORKSPACE,
                    }
                )

        sequence_hash = row.get("sequence_hash")
        if sequence_hash is None or pd.isna(sequence_hash):
            continue
        for right_entity_uid in by_sequence.get(str(sequence_hash), []):
            key = (left_entity_uid, right_entity_uid, "sequence_hash_match")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "link_uid": (
                        f"mfib_link:{row['native_id']}:{right_entity_uid}:seq"
                    ),
                    "left_entity_uid": left_entity_uid,
                    "right_entity_uid": right_entity_uid,
                    "left_source_id": "mfib",
                    "right_source_id": right_entity_uid.split(":", maxsplit=1)[0],
                    "link_type": "sequence_hash_match",
                    "evidence": sequence_hash,
                    "workspace": DEFAULT_WORKSPACE,
                }
            )
    return rows


def _build_mfib_split_rows(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build benchmark-oriented split rows for MFIB."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "mfib",
                "split_namespace": "mfib_recommended_role",
                "split_value": "benchmark_only",
                "reason": "bound_disorder_complex",
            }
        )
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "mfib",
                "split_namespace": "mfib_auxiliary_role",
                "split_value": "metadata_only",
                "reason": "bound_disorder_complex",
            }
        )
    return rows


def _build_fuzdb_entities(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build meta-entity rows for FuzDB entries."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "fuzdb",
                "native_id": row["native_id"],
                "display_name": row.get("protein_name") or row["native_id"],
                "sequence": row.get("sequence", pd.NA),
                "sequence_hash": row.get("sequence_hash", pd.NA),
                "workspace": DEFAULT_WORKSPACE,
                "entity_kind": "fuzzy_complex",
                "status": "ready",
            }
        )
    return rows


def _build_fuzdb_crossrefs(crossrefs: pd.DataFrame) -> list[dict[str, Any]]:
    """Build FuzDB cross-reference rows for the meta registry."""
    rows: list[dict[str, Any]] = []
    for row in crossrefs.to_dict(orient="records"):
        value = row.get("xref_id")
        if value is None or pd.isna(value) or str(value) == "":
            continue
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "fuzdb",
                "native_id": row["native_id"],
                "namespace": row.get("xref_namespace"),
                "xref_value": str(value),
                "source_field": str(row.get("xref_namespace")).lower(),
            }
        )
    return rows


def _build_fuzdb_tags(
    entries: pd.DataFrame,
    condensates: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build namespaced FuzDB tags for discovery and routing."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        for namespace, field in [
            ("topology_class", "topology_class"),
            ("mechanism_category", "mechanism_category"),
        ]:
            tag = row.get(field)
            if tag is None or pd.isna(tag) or str(tag) == "":
                continue
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "fuzdb",
                    "native_id": row["native_id"],
                    "namespace": namespace,
                    "tag": tag,
                }
            )
        for method in _split_pipe(row.get("detection_methods")):
            rows.append(
                {
                    "entity_uid": row["entry_uid"],
                    "source_id": "fuzdb",
                    "native_id": row["native_id"],
                    "namespace": "detection_method",
                    "tag": method,
                }
            )
    for row in condensates.to_dict(orient="records"):
        tag = row.get("llps_role")
        if tag is None or pd.isna(tag) or str(tag) == "":
            continue
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "fuzdb",
                "native_id": row["native_id"],
                "namespace": "llps_role",
                "tag": tag,
            }
        )
    return rows


def _build_fuzdb_links(
    crossrefs: pd.DataFrame,
    entity_rows: list[dict[str, Any]],
    crossref_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build exact-identifier FuzDB links to other source entities."""
    candidate_entities = [row for row in entity_rows if row["source_id"] != "fuzdb"]
    candidate_crossrefs = [row for row in crossref_rows if row["source_id"] != "fuzdb"]
    by_uniprot = _multi_map(
        [row for row in candidate_crossrefs if row.get("namespace") == "UniProt"],
        "xref_value",
        "entity_uid",
    )
    by_pdb = _multi_map(
        [row for row in candidate_crossrefs if row.get("namespace") == "PDB"],
        "xref_value",
        "entity_uid",
    )
    ped_entities = {
        str(row["native_id"]): str(row["entity_uid"])
        for row in candidate_entities
        if row["source_id"] == "ped"
    }
    bmrb_entities = {
        str(row["native_id"]): str(row["entity_uid"])
        for row in candidate_entities
        if row["source_id"] == "bmrb"
    }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in crossrefs.to_dict(orient="records"):
        namespace = row.get("xref_namespace")
        identifier = row.get("xref_id")
        if identifier is None or pd.isna(identifier) or str(identifier) == "":
            continue
        left_entity_uid = row["entry_uid"]
        identifier = str(identifier)
        candidate_targets: list[tuple[str, str, str]] = []
        if namespace == "UniProt":
            candidate_targets.extend(
                (right_entity_uid, "uniprot_match", identifier)
                for right_entity_uid in by_uniprot.get(identifier, [])
            )
        elif namespace == "PDB":
            candidate_targets.extend(
                (right_entity_uid, "pdb_match", identifier)
                for right_entity_uid in by_pdb.get(identifier, [])
            )
        elif namespace == "PED" and identifier in ped_entities:
            candidate_targets.append(
                (ped_entities[identifier], "ped_match", identifier)
            )
        elif namespace == "BMRB" and identifier in bmrb_entities:
            candidate_targets.append(
                (bmrb_entities[identifier], "bmrb_match", identifier)
            )

        for right_entity_uid, link_type, evidence in candidate_targets:
            key = (left_entity_uid, right_entity_uid, link_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "link_uid": (
                        f"fuzdb_link:{row['native_id']}:{right_entity_uid}:{link_type}"
                    ),
                    "left_entity_uid": left_entity_uid,
                    "right_entity_uid": right_entity_uid,
                    "left_source_id": "fuzdb",
                    "right_source_id": right_entity_uid.split(":", maxsplit=1)[0],
                    "link_type": link_type,
                    "evidence": evidence,
                    "workspace": DEFAULT_WORKSPACE,
                }
            )
    return rows


def _build_fuzdb_split_rows(entries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build benchmark-oriented split rows for FuzDB."""
    rows: list[dict[str, Any]] = []
    for row in entries.to_dict(orient="records"):
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "fuzdb",
                "split_namespace": "fuzdb_recommended_role",
                "split_value": "benchmark_only",
                "reason": "fuzzy_complex",
            }
        )
        rows.append(
            {
                "entity_uid": row["entry_uid"],
                "source_id": "fuzdb",
                "split_namespace": "fuzdb_auxiliary_role",
                "split_value": "metadata_only",
                "reason": "fuzzy_complex",
            }
        )
    return rows


def _build_integrated_split_rows(path: Path) -> list[dict[str, Any]]:
    """Build split rows from the integrated default split JSON."""
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        for bmrb_id in payload.get(split_name, []):
            rows.append(
                {
                    "entity_uid": f"bmrb:{bmrb_id}",
                    "source_id": "bmrb",
                    "split_namespace": "integrated_default",
                    "split_value": split_name,
                    "reason": "sequence_hash",
                }
            )
    return rows


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert rows to a nullable PyArrow-backed dataframe."""
    dataframe = pd.DataFrame(rows).replace({r"^\s*$": pd.NA}, regex=True)
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _normalize_entities_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize meta-entity scalar columns after cross-source row assembly."""
    if "geometry_available" in dataframe.columns:
        dataframe["geometry_available"] = dataframe["geometry_available"].map(
            lambda value: pd.NA if pd.isna(value) else _truthy(value)
        )
    for column in ["geometry_pair_count", "geometry_max_contact_count"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    if "geometry_max_interface_area" in dataframe.columns:
        dataframe["geometry_max_interface_area"] = pd.to_numeric(
            dataframe["geometry_max_interface_area"],
            errors="coerce",
        )
    return dataframe.convert_dtypes(dtype_backend="pyarrow")


def _split_pipe(value: Any) -> list[str]:
    """Split a pipe-delimited string into deterministic values."""
    if value is None or pd.isna(value):
        return []
    return [item for item in str(value).split("|") if item]


def _sequence_hash(sequence: Any) -> str | Any:
    """Return a deterministic hash for one sequence when available."""
    if sequence is None or sequence is pd.NA or str(sequence) == "":
        return pd.NA
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    """Return whether one dataframe-like value should be treated as true."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _unique_non_null(values: list[Any]) -> list[str]:
    """Return deterministic unique non-null strings."""
    unique = {
        str(value)
        for value in values
        if value is not None and not pd.isna(value) and str(value) != ""
    }
    return sorted(unique)


def _multi_map(
    rows: list[dict[str, Any]],
    key_field: str,
    value_field: str,
) -> dict[str, list[str]]:
    """Build a multi-value lookup from dictionaries."""
    mapping: dict[str, list[str]] = {}
    for row in rows:
        key = row.get(key_field)
        value = row.get(value_field)
        if key is None or value is None or pd.isna(key) or pd.isna(value):
            continue
        mapping.setdefault(str(key), [])
        if str(value) not in mapping[str(key)]:
            mapping[str(key)].append(str(value))
    return mapping


def _primary_key(table_name: str) -> list[str]:
    """Return the canonical primary key for one meta table."""
    if table_name == "source_catalog":
        return ["source_id"]
    if table_name == "entities":
        return ["entity_uid"]
    if table_name == "assets":
        return ["asset_uid"]
    if table_name == "crossrefs":
        return ["entity_uid", "namespace", "xref_value"]
    if table_name == "tags":
        return ["entity_uid", "namespace", "tag"]
    if table_name == "links":
        return ["link_uid"]
    return ["entity_uid", "split_namespace", "split_value"]


def _join_keys(table_name: str) -> list[str]:
    """Return the canonical join keys for one meta table."""
    if table_name == "source_catalog":
        return ["source_id"]
    if table_name == "assets":
        return ["entity_uid", "asset_uid"]
    if table_name in {"crossrefs", "tags", "links", "splits"}:
        return ["entity_uid"]
    return ["entity_uid"]


def _level(table_name: str) -> str:
    """Return the semantic level for one meta table."""
    if table_name in {"assets"}:
        return "asset"
    if table_name in {"tags", "crossrefs"}:
        return "tag"
    return "entry"


def _write_readme(meta_root: Path) -> None:
    """Write a short README for the meta dataframe bundle."""
    readme_path = meta_root / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "# AtypEmu Meta Dataframes\n\n"
        "This directory stores source-agnostic metadata tables used for "
        "cross-source joins, split routing, and validation indexing.\n"
    )
