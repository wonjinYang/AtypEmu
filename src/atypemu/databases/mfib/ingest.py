"""MFIB bundle crawl and normalization entrypoints."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.mfib.config import (
    CIF_ARCHIVE_URL,
    COMPLETE_JSON_ZIP_URL,
    COMPLETE_TXT_URL,
    COMPLETE_XML_ZIP_URL,
    DATASET_VERSION,
    DOWNLOADS_PAGE_URL,
    FORMAT_DEFINITION_URL,
    HOME_URL,
    INDIVIDUAL_JSON_URL,
    INDIVIDUAL_XSD_URL,
    SOURCE_ID,
    USER_AGENT,
)
from atypemu.databases.mfib.layout import (
    ensure_mfib_workspace,
    mfib_cif_root,
    mfib_debug_root,
    mfib_download_root,
    mfib_root,
    mfib_tables_root,
)
from atypemu.databases.mfib.normalize import (
    ASSET_COLUMNS,
    CHAIN_COLUMNS,
    CIF_CHAIN_COLUMNS,
    CIF_ENTRY_COLUMNS,
    CROSSREF_COLUMNS,
    ENTRY_COLUMNS,
    EVIDENCE_COLUMNS,
    GO_COLUMNS,
    REGION_COLUMNS,
    RELATED_COLUMNS,
    SEARCH_COLUMNS,
    build_cif_asset_record,
    json_txt_mismatch_counts,
    normalize_entry_bundle,
    parse_mfib_cif_features,
    parse_mfib_json_archive,
    parse_mfib_txt_entries,
)
from atypemu.databases.mfib.tables import write_tsv


def crawl_mfib(
    data_root: str | Path,
    refresh: bool = False,
    extract_cif: bool = True,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Crawl MFIB download bundles into source-owned tables and assets."""
    source_root = mfib_root(data_root)
    ensure_mfib_workspace(source_root)
    download_root = mfib_download_root(source_root)

    download_targets = {
        "downloads_page": (
            DOWNLOADS_PAGE_URL,
            download_root.parent / "_debug" / "pages" / "downloads.html",
        ),
        "complete_json_zip": (
            COMPLETE_JSON_ZIP_URL,
            download_root / "MFIB_complete_json.zip",
        ),
        "complete_xml_zip": (
            COMPLETE_XML_ZIP_URL,
            download_root / "MFIB_complete_xml.zip",
        ),
        "complete_txt": (COMPLETE_TXT_URL, download_root / "MFIB_complete.txt"),
        "format_definition": (
            FORMAT_DEFINITION_URL,
            download_root / "MFIB_format_definition.txt",
        ),
        "individual_json": (
            INDIVIDUAL_JSON_URL,
            download_root / "MFIB_individual.json",
        ),
        "individual_xsd": (INDIVIDUAL_XSD_URL, download_root / "MFIB_individual.xsd"),
    }
    if extract_cif:
        download_targets["cif_archive"] = (
            CIF_ARCHIVE_URL,
            download_root / "MFIB_download_pdb_cif_ALL.zip",
        )

    for url, destination in download_targets.values():
        _download_file(
            url=url,
            destination=destination,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )

    json_entries = parse_mfib_json_archive(download_root / "MFIB_complete_json.zip")
    txt_entries = parse_mfib_txt_entries(download_root / "MFIB_complete.txt")
    accessions = sorted(set(json_entries) | set(txt_entries))

    entry_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    go_rows: list[dict[str, Any]] = []
    related_rows: list[dict[str, Any]] = []
    crossref_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    for accession in accessions:
        (
            entry_row,
            bundle_chain_rows,
            bundle_region_rows,
            bundle_evidence_rows,
            bundle_go_rows,
            bundle_related_rows,
            bundle_crossref_rows,
            bundle_search_rows,
        ) = normalize_entry_bundle(
            accession=accession,
            json_entry=json_entries.get(accession),
            txt_entry=txt_entries.get(accession),
        )
        entry_rows.append(entry_row)
        chain_rows.extend(bundle_chain_rows)
        region_rows.extend(bundle_region_rows)
        evidence_rows.extend(bundle_evidence_rows)
        go_rows.extend(bundle_go_rows)
        related_rows.extend(bundle_related_rows)
        crossref_rows.extend(bundle_crossref_rows)
        search_rows.extend(bundle_search_rows)

    asset_rows: list[dict[str, Any]] = []
    cif_entry_rows: list[dict[str, Any]] = []
    cif_chain_rows: list[dict[str, Any]] = []
    if extract_cif:
        asset_rows, cif_entry_rows, cif_chain_rows = _extract_cif_assets_and_features(
            source_root=source_root,
            entry_rows=entry_rows,
            chain_rows=chain_rows,
            refresh=refresh,
        )

    tables_root = mfib_tables_root(source_root)
    write_tsv(tables_root / "entries.tsv", entry_rows, ENTRY_COLUMNS)
    write_tsv(tables_root / "chains.tsv", chain_rows, CHAIN_COLUMNS)
    write_tsv(tables_root / "regions.tsv", region_rows, REGION_COLUMNS)
    write_tsv(tables_root / "evidence.tsv", evidence_rows, EVIDENCE_COLUMNS)
    write_tsv(tables_root / "go_terms.tsv", go_rows, GO_COLUMNS)
    write_tsv(tables_root / "related_entries.tsv", related_rows, RELATED_COLUMNS)
    write_tsv(tables_root / "crossrefs.tsv", crossref_rows, CROSSREF_COLUMNS)
    write_tsv(tables_root / "search_index.tsv", search_rows, SEARCH_COLUMNS)
    write_tsv(tables_root / "assets.tsv", asset_rows, ASSET_COLUMNS)
    write_tsv(
        tables_root / "cif_entry_features.tsv",
        cif_entry_rows,
        CIF_ENTRY_COLUMNS,
    )
    write_tsv(
        tables_root / "cif_chain_features.tsv",
        cif_chain_rows,
        CIF_CHAIN_COLUMNS,
    )

    mismatch_counts = json_txt_mismatch_counts(json_entries, txt_entries)
    crawl_summary = {
        "source_id": SOURCE_ID,
        "source_version": DATASET_VERSION,
        "source_root": str(source_root),
        "home_url": HOME_URL,
        "json_entries": len(json_entries),
        "txt_entries": len(txt_entries),
        "union_entries": len(accessions),
        "json_only_entries": len(set(json_entries) - set(txt_entries)),
        "txt_only_entries": len(set(txt_entries) - set(json_entries)),
        "entry_rows": len(entry_rows),
        "chain_rows": len(chain_rows),
        "region_rows": len(region_rows),
        "go_rows": len(go_rows),
        "related_rows": len(related_rows),
        "crossref_rows": len(crossref_rows),
        "search_rows": len(search_rows),
        "asset_rows": len(asset_rows),
        "cif_entry_rows": len(cif_entry_rows),
        "cif_chain_rows": len(cif_chain_rows),
        "mismatch_counts": mismatch_counts,
    }
    debug_root = mfib_debug_root(source_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    (debug_root / "mfib_crawl_summary.json").write_text(
        json.dumps(crawl_summary, indent=2, sort_keys=True)
    )
    return crawl_summary


def _download_file(
    url: str,
    destination: Path,
    refresh: bool,
    timeout_seconds: int,
) -> Path:
    """Download one MFIB bundle file when needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        return destination
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _extract_cif_assets_and_features(
    source_root: Path,
    entry_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    refresh: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract relevant CIF assets and build deterministic structure features."""
    archive_path = mfib_download_root(source_root) / "MFIB_download_pdb_cif_ALL.zip"
    if not archive_path.exists():
        return [], [], []

    accessions_by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entry_rows:
        pdb_id = row.get("pdb_id")
        if pdb_id is None:
            continue
        accessions_by_pdb[str(pdb_id).lower()].append(row)

    cif_root = mfib_cif_root(source_root)
    cif_root.mkdir(parents=True, exist_ok=True)
    member_by_pdb = _member_by_pdb_id(archive_path)

    asset_rows: list[dict[str, Any]] = []
    cif_entry_rows: list[dict[str, Any]] = []
    cif_chain_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path) as archive:
        for pdb_id, accession_payloads in sorted(accessions_by_pdb.items()):
            member = member_by_pdb.get(pdb_id)
            target_path = cif_root / f"{pdb_id}.cif"
            available = member is not None
            if available and (refresh or not target_path.exists()):
                target_path.write_bytes(archive.read(member))
            asset_path = (
                str(target_path.relative_to(source_root.parent))
                if available and target_path.exists()
                else None
            )
            shared = len(accession_payloads) > 1

            entry_features: dict[str, Any] = {}
            chain_features: list[dict[str, Any]] = []
            if available and target_path.exists():
                entry_features, chain_features = parse_mfib_cif_features(target_path)

            chain_feature_map = {
                payload["chain_id"]: payload for payload in chain_features
            }
            for row in accession_payloads:
                asset_rows.append(
                    build_cif_asset_record(
                        accession=row["accession"],
                        entry_uid=row["entry_uid"],
                        pdb_id=pdb_id,
                        asset_path=asset_path,
                        available=available and target_path.exists(),
                        shared_across_entries=shared,
                    )
                )
                if entry_features:
                    cif_entry_rows.append(
                        {
                            "accession": row["accession"],
                            "entry_uid": row["entry_uid"],
                            "source_id": SOURCE_ID,
                            "native_id": row["native_id"],
                            "pdb_id": str(pdb_id).upper(),
                            "asset_path": asset_path,
                            **entry_features,
                        }
                    )
                    for chain_id, payload in sorted(chain_feature_map.items()):
                        cif_chain_rows.append(
                            {
                                "accession": row["accession"],
                                "entry_uid": row["entry_uid"],
                                "chain_uid": f"{row['entry_uid']}:{chain_id}",
                                "source_id": SOURCE_ID,
                                "native_id": row["native_id"],
                                "parent_uid": row["entry_uid"],
                                "chain_id": chain_id,
                                **payload,
                            }
                        )
    return asset_rows, cif_entry_rows, cif_chain_rows


def _member_by_pdb_id(archive_path: Path) -> dict[str, str]:
    """Map lowercased PDB IDs to CIF archive members."""
    with zipfile.ZipFile(archive_path) as archive:
        mapping: dict[str, str] = {}
        for member in archive.namelist():
            if not member.lower().endswith(".cif"):
                continue
            mapping.setdefault(Path(member).stem.lower(), member)
        return mapping
