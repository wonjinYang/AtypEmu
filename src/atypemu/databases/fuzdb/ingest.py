"""FuzDB crawl and normalization entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.fuzdb.config import (
    BROWSE_URL,
    ENTRIES_API_URL,
    ENTRIES_FORMAT_URL_TEMPLATE,
    ENTRY_URL_TEMPLATE,
    HOME_URL,
    SOURCE_ID,
    USER_AGENT,
    FuzdbCrawlConfig,
)
from atypemu.databases.fuzdb.layout import (
    ensure_fuzdb_workspace,
    fuzdb_browse_debug_root,
    fuzdb_entries_download_path,
    fuzdb_entries_tsv_download_path,
    fuzdb_entries_txt_download_path,
    fuzdb_entries_xml_download_path,
    fuzdb_entry_debug_root,
    fuzdb_root,
    fuzdb_tables_root,
)
from atypemu.databases.fuzdb.normalize import (
    CONDENSATE_COLUMNS,
    CROSSREF_COLUMNS,
    ENTRY_COLUMNS,
    FUNCTIONAL_SITE_COLUMNS,
    ISOFORM_COLUMNS,
    PTM_COLUMNS,
    REFERENCE_COLUMNS,
    REGION_COLUMNS,
    SEARCH_COLUMNS,
    STRUCTURE_LINK_COLUMNS,
    normalize_fuzdb_entries,
)
from atypemu.databases.fuzdb.tables import write_tsv


def crawl_fuzdb(
    data_root: str | Path,
    refresh: bool = False,
    timeout_seconds: int = 60,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Crawl FuzDB public entry metadata into curated tables.

    The live FuzDB site is a single-page application. We therefore use the
    public ``/api/entries`` JSON feed as the deterministic source-of-truth and
    retain HTML shell pages only for provenance/debugging.
    """
    config = FuzdbCrawlConfig(
        refresh=refresh,
        timeout_seconds=timeout_seconds,
        max_entries=max_entries,
    )
    source_root = fuzdb_root(data_root)
    ensure_fuzdb_workspace(source_root)

    payload = _fetch_entries_payload(
        source_root=source_root,
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    _fetch_entries_export(
        format_name="tsv",
        destination=fuzdb_entries_tsv_download_path(source_root),
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    _fetch_entries_export(
        format_name="xml",
        destination=fuzdb_entries_xml_download_path(source_root),
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    _fetch_entries_export(
        format_name="txt",
        destination=fuzdb_entries_txt_download_path(source_root),
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    raw_entries = list(payload)
    discovered_entries = len(raw_entries)
    if config.max_entries is not None:
        raw_entries = raw_entries[: config.max_entries]

    browse_pages = _capture_browse_pages(
        source_root=source_root,
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    entry_pages = _capture_entry_pages(
        source_root=source_root,
        entry_ids=[
            str(entry["entry_id"]) for entry in raw_entries if entry.get("entry_id")
        ],
        refresh=config.refresh,
        timeout_seconds=config.timeout_seconds,
    )
    normalized = normalize_fuzdb_entries(raw_entries)

    tables_root = fuzdb_tables_root(source_root)
    write_tsv(tables_root / "entries.tsv", normalized["entries"], ENTRY_COLUMNS)
    write_tsv(
        tables_root / "fuzzy_regions.tsv",
        normalized["fuzzy_regions"],
        REGION_COLUMNS,
    )
    write_tsv(
        tables_root / "structure_links.tsv",
        normalized["structure_links"],
        STRUCTURE_LINK_COLUMNS,
    )
    write_tsv(
        tables_root / "functional_sites.tsv",
        normalized["functional_sites"],
        FUNCTIONAL_SITE_COLUMNS,
    )
    write_tsv(tables_root / "ptm_sites.tsv", normalized["ptm_sites"], PTM_COLUMNS)
    write_tsv(tables_root / "isoforms.tsv", normalized["isoforms"], ISOFORM_COLUMNS)
    write_tsv(
        tables_root / "condensates.tsv",
        normalized["condensates"],
        CONDENSATE_COLUMNS,
    )
    write_tsv(
        tables_root / "references.tsv",
        normalized["references"],
        REFERENCE_COLUMNS,
    )
    write_tsv(
        tables_root / "crossrefs.tsv",
        normalized["crossrefs"],
        CROSSREF_COLUMNS,
    )
    write_tsv(
        tables_root / "search_index.tsv",
        normalized["search_index"],
        SEARCH_COLUMNS,
    )

    summary = {
        "source_id": SOURCE_ID,
        "source_root": str(source_root),
        "api_entry_count": discovered_entries,
        "selected_entry_count": len(raw_entries),
        "entry_rows": len(normalized["entries"]),
        "fuzzy_region_rows": len(normalized["fuzzy_regions"]),
        "structure_link_rows": len(normalized["structure_links"]),
        "functional_site_rows": len(normalized["functional_sites"]),
        "ptm_rows": len(normalized["ptm_sites"]),
        "isoform_rows": len(normalized["isoforms"]),
        "condensate_rows": len(normalized["condensates"]),
        "reference_rows": len(normalized["references"]),
        "crossref_rows": len(normalized["crossrefs"]),
        "search_rows": len(normalized["search_index"]),
        "topology_classified_rows": sum(
            1 for row in normalized["entries"] if row.get("topology_class")
        ),
        "mechanism_classified_rows": sum(
            1 for row in normalized["entries"] if row.get("mechanism_category")
        ),
        "browse_page_snapshots": browse_pages,
        "entry_page_snapshots": entry_pages,
    }
    (source_root / "_debug" / "fuzdb_crawl_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _fetch_entries_payload(
    source_root: Path,
    refresh: bool,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch or reuse the public FuzDB entries payload."""
    target_path = fuzdb_entries_download_path(source_root)
    if target_path.exists() and not refresh:
        payload = json.loads(target_path.read_text())
        if not isinstance(payload, list):
            raise ValueError("Cached FuzDB entries payload is not a list.")
        return payload

    response = requests.get(
        ENTRIES_API_URL,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected FuzDB entries payload.")
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _fetch_entries_export(
    format_name: str,
    destination: Path,
    refresh: bool,
    timeout_seconds: int,
) -> Path:
    """Fetch one hidden FuzDB download export for provenance and cross-check."""
    if destination.exists() and not refresh:
        return destination
    response = requests.get(
        ENTRIES_FORMAT_URL_TEMPLATE.format(format_name=format_name),
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    destination.write_text(response.text)
    return destination


def _capture_browse_pages(
    source_root: Path,
    refresh: bool,
    timeout_seconds: int,
) -> int:
    """Capture lightweight SPA shell pages for reproducible debugging."""
    browse_root = fuzdb_browse_debug_root(source_root)
    captures = {
        HOME_URL: browse_root / "home.html",
        BROWSE_URL: browse_root / "browse.html",
    }
    captured = 0
    for url, destination in captures.items():
        if destination.exists() and not refresh:
            captured += 1
            continue
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        destination.write_text(response.text)
        captured += 1
    return captured


def _capture_entry_pages(
    source_root: Path,
    entry_ids: list[str],
    refresh: bool,
    timeout_seconds: int,
) -> int:
    """Capture one SPA shell page per FuzDB entry for provenance."""
    entry_root = fuzdb_entry_debug_root(source_root)
    captured = 0
    for fc_id in entry_ids:
        destination = entry_root / f"{fc_id}.html"
        if destination.exists() and not refresh:
            captured += 1
            continue
        response = requests.get(
            ENTRY_URL_TEMPLATE.format(fc_id=fc_id),
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        destination.write_text(response.text)
        captured += 1
    return captured
