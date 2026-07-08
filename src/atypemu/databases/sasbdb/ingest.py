"""SASBDB crawl and normalization entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.sasbdb.config import (
    ENTRY_HTML_URL_TEMPLATE,
    FASTA_URL_TEMPLATE,
    PROTEIN_CODES_URL,
    SOURCE_ID,
    SUMMARY_BATCH_URL,
    SUMMARY_URL_TEMPLATE,
    USER_AGENT,
    SasbdbCrawlConfig,
)
from atypemu.databases.sasbdb.layout import (
    ensure_sasbdb_workspace,
    sasbdb_fasta_root,
    sasbdb_html_debug_root,
    sasbdb_intensity_root,
    sasbdb_pddf_root,
    sasbdb_protein_codes_path,
    sasbdb_root,
    sasbdb_sascif_root,
    sasbdb_summary_root,
    sasbdb_tables_root,
)
from atypemu.databases.sasbdb.normalize import (
    ASSET_COLUMNS,
    ENTRY_COLUMNS,
    LLM_COLUMNS,
    MOLECULE_COLUMNS,
    VALIDATION_COLUMNS,
    normalize_entry_bundle,
    parse_sascif_payload,
)
from atypemu.databases.sasbdb.tables import write_tsv


def crawl_sasbdb(
    data_root: str | Path,
    protein_only: bool = True,
    refresh: bool = False,
    include_html_fallback: bool = True,
    batch_size: int = 100,
    timeout_seconds: int = 60,
    codes: list[str] | None = None,
) -> dict[str, Any]:
    """Crawl SASBDB accession metadata and source assets.

    Args:
        data_root: Top-level AtypEmu data root or the SASBDB source root.
        protein_only: Whether to restrict the crawl to protein entries.
        refresh: When ``True``, overwrite cached downloads and tables.
        include_html_fallback: Whether to fetch HTML only when structured fields
            remain unresolved.
        batch_size: Number of accessions per summary-list request.
        timeout_seconds: Per-request timeout in seconds.
        codes: Optional explicit accession subset for smoke runs and tests.

    Returns:
        Summary dictionary describing the crawl outputs.
    """
    if not protein_only:
        raise ValueError("SASBDB v1 only supports the protein-only crawl universe.")

    config = SasbdbCrawlConfig(
        protein_only=protein_only,
        refresh=refresh,
        include_html_fallback=include_html_fallback,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
    source_root = sasbdb_root(data_root)
    ensure_sasbdb_workspace(source_root)

    accession_rows = (
        _fetch_protein_codes(timeout_seconds=config.timeout_seconds)
        if codes is None
        else [{"code": code, "status": "Published"} for code in codes]
    )
    sasbdb_protein_codes_path(source_root).write_text(
        json.dumps(accession_rows, indent=2, sort_keys=True)
    )
    published_codes = [
        row["code"]
        for row in accession_rows
        if row.get("status") == "Published" or codes is not None
    ]
    summary_payloads = _fetch_summaries(
        codes=published_codes,
        batch_size=config.batch_size,
        timeout_seconds=config.timeout_seconds,
    )

    entry_rows: list[dict[str, Any]] = []
    molecule_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    html_fetch_count = 0

    for code in published_codes:
        summary = summary_payloads[code]
        summary_path = sasbdb_summary_root(source_root) / f"{code}.json"
        if refresh or not summary_path.exists():
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        sascif_path = _download_optional_file(
            url=summary.get("sascif_data"),
            destination=sasbdb_sascif_root(source_root) / f"{code}.sascif",
            refresh=refresh,
            timeout_seconds=config.timeout_seconds,
        )
        intensity_path = _download_optional_file(
            url=summary.get("intensities_data"),
            destination=sasbdb_intensity_root(source_root) / f"{code}.dat",
            refresh=refresh,
            timeout_seconds=config.timeout_seconds,
        )
        pddf_path = _download_optional_file(
            url=summary.get("pddf_data"),
            destination=sasbdb_pddf_root(source_root) / f"{code}.out",
            refresh=refresh,
            timeout_seconds=config.timeout_seconds,
        )

        entity_ids = _discover_entity_ids(code, summary, sascif_path)
        for entity_id in entity_ids:
            _download_optional_file(
                url=FASTA_URL_TEMPLATE.format(code=code, entity_id=entity_id),
                destination=sasbdb_fasta_root(source_root)
                / code
                / f"{entity_id}.fasta",
                refresh=refresh,
                timeout_seconds=config.timeout_seconds,
            )

        html_text = None
        if include_html_fallback and _needs_html_fallback(summary, sascif_path):
            html_text = _download_html_fallback(
                code=code,
                destination=sasbdb_html_debug_root(source_root) / f"{code}.html",
                refresh=refresh,
                timeout_seconds=config.timeout_seconds,
            )
            html_fetch_count += 1 if html_text else 0

        entry_row, molecules, validation_row, assets, llm = normalize_entry_bundle(
            code=code,
            summary=summary,
            sascif_path=sascif_path,
            intensity_path=intensity_path,
            pddf_path=pddf_path,
            html_text=html_text,
        )
        entry_rows.append(entry_row)
        molecule_rows.extend(molecules)
        validation_rows.append(validation_row)
        asset_rows.extend(assets)
        llm_rows.extend(llm)

    tables_root = sasbdb_tables_root(source_root)
    write_tsv(tables_root / "entries.tsv", entry_rows, ENTRY_COLUMNS)
    write_tsv(tables_root / "molecules.tsv", molecule_rows, MOLECULE_COLUMNS)
    write_tsv(
        tables_root / "validation_conditions.tsv",
        validation_rows,
        VALIDATION_COLUMNS,
    )
    write_tsv(tables_root / "assets.tsv", asset_rows, ASSET_COLUMNS)
    write_tsv(tables_root / "validation_llm.tsv", llm_rows, LLM_COLUMNS)

    crawl_summary = {
        "source_id": SOURCE_ID,
        "source_root": str(source_root),
        "protein_only": protein_only,
        "published_entries": len(published_codes),
        "entry_rows": len(entry_rows),
        "molecule_rows": len(molecule_rows),
        "asset_rows": len(asset_rows),
        "llm_rows": len(llm_rows),
        "html_fetch_count": html_fetch_count,
    }
    (source_root / "_debug" / "sasbdb_crawl_summary.json").write_text(
        json.dumps(crawl_summary, indent=2, sort_keys=True)
    )
    return crawl_summary


def _fetch_protein_codes(timeout_seconds: int) -> list[dict[str, Any]]:
    """Fetch the protein-only accession universe from SASBDB."""
    response = requests.get(
        PROTEIN_CODES_URL,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected SASBDB code-list payload.")
    return payload


def _fetch_summaries(
    codes: list[str],
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    """Fetch summary payloads with one batch-retry and per-code fallback."""
    results: dict[str, dict[str, Any]] = {}
    for chunk_start in range(0, len(codes), batch_size):
        chunk = codes[chunk_start : chunk_start + batch_size]
        payload = _fetch_summary_batch(chunk, timeout_seconds)
        if payload is None:
            payload = _fetch_summary_batch(chunk, timeout_seconds)
        if payload is None:
            for code in chunk:
                results[code] = _fetch_summary_single(code, timeout_seconds)
            continue
        for row in payload:
            results[row["code"]] = row
    return results


def _fetch_summary_batch(
    codes: list[str],
    timeout_seconds: int,
) -> list[dict[str, Any]] | None:
    """Fetch one summary batch with the documented list endpoint."""
    response = requests.post(
        SUMMARY_BATCH_URL,
        json={"codes": ",".join(codes)},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code != 200:
        return None
    payload = response.json()
    if not isinstance(payload, list):
        return None
    return payload


def _fetch_summary_single(code: str, timeout_seconds: int) -> dict[str, Any]:
    """Fetch one accession summary with the per-code endpoint."""
    response = requests.get(
        SUMMARY_URL_TEMPLATE.format(code=code),
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected SASBDB summary payload for {code}.")
    return payload


def _download_optional_file(
    url: str | None,
    destination: Path,
    refresh: bool,
    timeout_seconds: int,
) -> Path | None:
    """Download one file when the URL is present."""
    if not url:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        return destination
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code != 200:
        return None
    destination.write_bytes(response.content)
    return destination


def _discover_entity_ids(
    code: str,
    summary: dict[str, Any],
    sascif_path: Path | None,
) -> list[str]:
    """Resolve the set of molecule entity IDs that should receive FASTA downloads."""
    if sascif_path is not None and sascif_path.exists():
        payload = parse_sascif_payload(sascif_path)
        entity_ids = [
            str(row["entity_id"])
            for row in payload.get("entities", [])
            if row.get("entity_id")
        ]
        if entity_ids:
            return entity_ids

    molecules = (summary.get("experiment") or {}).get("sample", {}).get("molecule", [])
    return [str(index) for index, _ in enumerate(molecules, start=1)]


def _needs_html_fallback(summary: dict[str, Any], sascif_path: Path | None) -> bool:
    """Return whether HTML fallback should be attempted for one accession."""
    experiment = summary.get("experiment") or {}
    buffer_payload = (experiment.get("sample") or {}).get("buffer") or {}
    required = [
        experiment.get("wavelength"),
        experiment.get("cell_temperature"),
        experiment.get("storage_temperature"),
        experiment.get("sample_detector_distance"),
        buffer_payload.get("ph"),
    ]
    if any(value not in {None, ""} for value in required):
        return False
    return sascif_path is None or not sascif_path.exists()


def _download_html_fallback(
    code: str,
    destination: Path,
    refresh: bool,
    timeout_seconds: int,
) -> str | None:
    """Download entry HTML only when structured sources were incomplete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not refresh:
        return destination.read_text()
    response = requests.get(
        ENTRY_HTML_URL_TEMPLATE.format(code=code),
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code != 200:
        return None
    destination.write_text(response.text)
    return response.text
