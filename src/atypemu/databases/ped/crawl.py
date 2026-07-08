"""PED-specific crawl and normalization helpers for AtypEmu.

This module implements a PED-only collection path that mirrors PED entry pages,
captures structured network payloads with Playwright, downloads ensemble model
archives, and writes normalized entry-level and model-level tables.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.common import setup_source_workspace
from atypemu.databases.common.bootstrap import meta_tables_root
from atypemu.databases.ped.layout import (
    ensure_ped_workspace,
    normalize_ped_workspace,
    ped_debug_root,
    ped_models_root,
)


PED_HOME_URL = "https://proteinensemble.org"
PED_BROWSE_URL = f"{PED_HOME_URL}/browse"
PED_API_LANDING_URL = f"{PED_HOME_URL}/api"
PED_DEPOSITION_API_ROOT = "https://deposition.proteinensemble.org/api/v1"
PED_DEPOSITION_ENTRY_LIST_URL = (
    f"{PED_DEPOSITION_API_ROOT}/entries/?offset=0&limit=20&"
    "sort_field=entry_id&sort_order=asc"
)
USER_AGENT = "AtypEmu/0.1 PED collector"
METADATA_ONLY_WORKERS = 8
PED_ENTRY_COLUMNS = [
    "ped_id",
    "entry_url",
    "protein_name",
    "sequence",
    "sequence_length",
    "uniprot_accessions",
    "bmrb_ids",
    "experimental_procedures",
    "experimental_procedures_raw",
    "structural_ensemble_calculation_tags",
    "structural_ensemble_calculation_raw",
    "ensemble_count",
    "model_count",
    "raw_entry_html_path",
    "network_payload_count",
    "crawl_timestamp_utc",
]
PED_MODEL_COLUMNS = [
    "ped_id",
    "ensemble_id",
    "model_id",
    "model_index",
    "model_pdb_url",
    "model_pdb_path",
    "secondary_structure_entropy",
    "relative_solvent_accessibility",
    "radius_of_gyration",
    "measure_source",
    "crawl_timestamp_utc",
]
LEGACY_PED_ID_ALIASES = {
    "PED1AAD": "PED00003",
    "PED9AAA": "PED00001",
    "PED6AAA": "PED00016",
    "PED5AAB": "PED00153",
    "PED2AAA": "PED00004",
    "PED9AAC": "PED00024",
}
SEED_NAME_ALIASES = {
    "beta-synuclein": ["beta synuclein"],
    "Sendai virus phosphoprotein": [
        "sendai virus phosphoprotein",
        "sendai phosphoprotein",
    ],
    "Sic1/Cdc4": ["sic1", "psic1", "cdc4"],
    "p15 PAF": ["p15 paf"],
    "MKK7": ["mkk7"],
    "p27 KID": ["p27 kip1", "p27 kid"],
    "alpha-synuclein": ["alpha synuclein"],
}


@dataclass(frozen=True)
class _PedEntryRow:
    """Normalized PED entry row used for TSV and JSONL exports."""

    ped_id: str
    entry_url: str
    protein_name: str | None
    sequence: str | None
    sequence_length: int | None
    uniprot_accessions: str | None
    bmrb_ids: str | None
    experimental_procedures: str | None
    experimental_procedures_raw: str | None
    structural_ensemble_calculation_tags: str | None
    structural_ensemble_calculation_raw: str | None
    ensemble_count: int
    model_count: int
    raw_entry_html_path: str | None
    network_payload_count: int
    crawl_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the row."""
        return asdict(self)


@dataclass(frozen=True)
class _PedModelRow:
    """Normalized PED model row used for TSV and JSONL exports."""

    ped_id: str
    ensemble_id: str
    model_id: str
    model_index: int
    model_pdb_url: str
    model_pdb_path: str
    secondary_structure_entropy: float | None
    relative_solvent_accessibility: float | None
    radius_of_gyration: float | None
    measure_source: str
    crawl_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the row."""
        return asdict(self)


@dataclass(frozen=True)
class _PedNetworkRecord:
    """Manifest record for one captured PED debug payload."""

    ped_id: str
    response_url: str
    relative_path: str
    content_type: str | None
    status_code: int
    size_bytes: int
    sha256: str
    crawl_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the record."""
        return asdict(self)


@dataclass(frozen=True)
class _PedSeedResolution:
    """Resolution record that maps a curated PED seed to a live PED ID."""

    legacy_id: str
    protein_name: str
    resolved_ped_id: str | None
    resolution_strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for the record."""
        return asdict(self)


def crawl_ped(
    ped_root: str | Path,
    workspace: str = "benchmark",
    seed_only: bool = False,
    ped_ids: list[str] | None = None,
    refresh: bool = False,
    timeout_seconds: int = 60,
    max_entries: int | None = None,
    save_network_payloads: bool = True,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Crawl PED entry metadata, model assets, and PED tabular outputs.

    Args:
        ped_root: Source-owned PED root such as ``data/ped``.
        workspace: PED workspace, either ``benchmark`` or ``catalog``.
        seed_only: Whether to crawl only the curated PED benchmark seeds.
        ped_ids: Optional explicit live PED IDs to crawl.
        refresh: Whether to replace existing crawl outputs.
        timeout_seconds: Per-request or page-load timeout in seconds.
        max_entries: Optional hard limit for the number of entries to crawl.
        save_network_payloads: Whether to persist captured JSON debug payloads.
        metadata_only: Whether to skip model-asset downloads and collect only
            entry-level metadata.

    Returns:
        Summary dictionary describing the crawl result.
    """
    ped_root_path = Path(ped_root)
    workspace_name = normalize_ped_workspace(workspace)
    setup_source_workspace(data_root=ped_root_path.parent)
    manifests_dir, debug_root = ensure_ped_workspace(ped_root_path, workspace_name)
    ped_root_path.mkdir(parents=True, exist_ok=True)
    if workspace_name == "catalog":
        metadata_only = True
    if metadata_only:
        browse_dir = debug_root / "browse"
        browse_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_entries_dir = debug_root / "entries"
        raw_models_dir = ped_models_root(ped_root_path, workspace_name)
        browse_dir = debug_root / "browse"
        for path in [raw_entries_dir, raw_models_dir, browse_dir]:
            path.mkdir(parents=True, exist_ok=True)

    existing_entry_rows = _load_jsonl_by_key(
        debug_root / "ped_entry_records.jsonl",
        key="ped_id",
    )
    existing_model_rows = _load_jsonl_grouped(
        debug_root / "ped_model_records.jsonl",
        key="ped_id",
    )
    existing_network_rows = _load_jsonl_grouped(
        _resolve_debug_manifest(debug_root),
        key="ped_id",
    )

    api_probe = _probe_api_landing(timeout_seconds=timeout_seconds)
    if metadata_only and not seed_only and not ped_ids:
        discovery = {
            "browse_url": PED_BROWSE_URL,
            "listing_url": PED_DEPOSITION_ENTRY_LIST_URL,
            "captured_payloads": 0,
            "strategy": "direct_deposition_api",
        }
    else:
        discovery = _discover_listing_api(
            ped_root=ped_root_path,
            workspace=workspace_name,
            timeout_seconds=timeout_seconds,
            save_network_payloads=save_network_payloads,
        )
    listing_rows = _fetch_listing_rows(
        listing_url=discovery["listing_url"],
        timeout_seconds=timeout_seconds,
    )

    resolution_records: list[_PedSeedResolution] = []
    if ped_ids:
        target_ped_ids = [_normalize_ped_id(item) for item in ped_ids]
    elif seed_only:
        target_ped_ids, resolution_records = _resolve_seed_targets(
            ped_root=ped_root_path,
            listing_rows=listing_rows,
        )
    else:
        target_ped_ids = [row["entry_id"] for row in listing_rows]

    if max_entries is not None:
        target_ped_ids = target_ped_ids[:max_entries]

    entry_rows: list[_PedEntryRow] = []
    model_rows: list[_PedModelRow] = []
    network_rows: list[_PedNetworkRecord] = []
    reused_entries = 0
    crawled_entries = 0

    if metadata_only:
        pending_ped_ids: list[str] = []
        for ped_id in target_ped_ids:
            if not refresh and ped_id in existing_entry_rows:
                entry_rows.append(_entry_row_from_dict(existing_entry_rows[ped_id]))
                model_rows.extend(
                    _model_row_from_dict(row)
                    for row in existing_model_rows.get(ped_id, [])
                )
                network_rows.extend(
                    _network_row_from_dict(row)
                    for row in existing_network_rows.get(ped_id, [])
                )
                reused_entries += 1
                _persist_ped_manifests(
                    manifests_dir=manifests_dir,
                    debug_root=debug_root,
                    entry_rows=entry_rows,
                    model_rows=model_rows,
                    network_rows=network_rows,
                    write_models=workspace_name == "benchmark",
                )
            else:
                pending_ped_ids.append(ped_id)

        with ThreadPoolExecutor(max_workers=METADATA_ONLY_WORKERS) as executor:
            future_to_ped_id = {
                executor.submit(
                    _crawl_one_entry_metadata_only,
                    ped_root=ped_root_path,
                    workspace=workspace_name,
                    ped_id=ped_id,
                    timeout_seconds=timeout_seconds,
                    save_network_payloads=save_network_payloads,
                ): ped_id
                for ped_id in pending_ped_ids
            }
            for future in as_completed(future_to_ped_id):
                entry_row, entry_model_rows, entry_network_rows = future.result()
                entry_rows.append(entry_row)
                model_rows.extend(entry_model_rows)
                network_rows.extend(entry_network_rows)
                crawled_entries += 1
                _persist_ped_manifests(
                    manifests_dir=manifests_dir,
                    debug_root=debug_root,
                    entry_rows=entry_rows,
                    model_rows=model_rows,
                    network_rows=network_rows,
                    write_models=workspace_name == "benchmark",
                )
    else:
        browser = _launch_browser()
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(timeout_seconds * 1000)
            for ped_id in target_ped_ids:
                if not refresh and ped_id in existing_entry_rows:
                    entry_rows.append(_entry_row_from_dict(existing_entry_rows[ped_id]))
                    model_rows.extend(
                        _model_row_from_dict(row)
                        for row in existing_model_rows.get(ped_id, [])
                    )
                    network_rows.extend(
                        _network_row_from_dict(row)
                        for row in existing_network_rows.get(ped_id, [])
                    )
                    reused_entries += 1
                    _persist_ped_manifests(
                        manifests_dir=manifests_dir,
                        debug_root=debug_root,
                        entry_rows=entry_rows,
                        model_rows=model_rows,
                        network_rows=network_rows,
                        write_models=workspace_name == "benchmark",
                    )
                    continue

                entry_row, entry_model_rows, entry_network_rows = _crawl_one_entry(
                    page=page,
                    ped_root=ped_root_path,
                    workspace=workspace_name,
                    ped_id=ped_id,
                    timeout_seconds=timeout_seconds,
                    save_network_payloads=save_network_payloads,
                )
                entry_rows.append(entry_row)
                model_rows.extend(entry_model_rows)
                network_rows.extend(entry_network_rows)
                crawled_entries += 1
                _persist_ped_manifests(
                    manifests_dir=manifests_dir,
                    debug_root=debug_root,
                    entry_rows=entry_rows,
                    model_rows=model_rows,
                    network_rows=network_rows,
                    write_models=workspace_name == "benchmark",
                )
            context.close()
        finally:
            browser.close()

    enrichment_summary = _enrich_benchmark_registry(
        ped_root=ped_root_path,
        workspace=workspace_name,
        entry_rows=entry_rows,
        model_rows=model_rows,
        resolution_records=resolution_records,
    )
    _write_ped_readme(ped_root_path)

    summary = {
        "ped_root": str(ped_root_path),
        "workspace": workspace_name,
        "output_root": str(manifests_dir),
        "seed_only": seed_only,
        "requested_ped_ids": target_ped_ids,
        "entries_requested": len(target_ped_ids),
        "entries_crawled": crawled_entries,
        "entries_reused": reused_entries,
        "entries_total": len(entry_rows),
        "models_total": len(model_rows),
        "network_payloads_total": len(network_rows),
        "save_network_payloads": save_network_payloads,
        "metadata_only": metadata_only,
        "refresh": refresh,
        "timeout_seconds": timeout_seconds,
        "max_entries": max_entries,
        "api_landing_probe": api_probe,
        "listing_discovery": discovery,
        "seed_resolution": [row.to_dict() for row in resolution_records],
        "unresolved_seed_count": sum(
            row.resolved_ped_id is None for row in resolution_records
        ),
        "benchmark_registry_updated": enrichment_summary["updated_records"],
        "crawl_timestamp_utc": _timestamp_utc(),
    }
    (debug_root / "ped_crawl_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _crawl_one_entry(
    page,
    ped_root: Path,
    workspace: str,
    ped_id: str,
    timeout_seconds: int,
    save_network_payloads: bool,
) -> tuple[_PedEntryRow, list[_PedModelRow], list[_PedNetworkRecord]]:
    """Crawl one PED entry page and download its ensemble model assets."""
    entry_dir = ped_debug_root(ped_root, workspace) / "entries" / ped_id
    debug_dir = entry_dir / "debug"
    model_root = ped_models_root(ped_root, workspace) / ped_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    captured_payloads: list[dict[str, Any]] = []

    def handle_response(response) -> None:
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return
        if ped_id not in response.url:
            return
        try:
            payload = response.json()
        except Exception:
            return
        captured_payloads.append(
            {
                "url": response.url,
                "status": response.status,
                "content_type": content_type,
                "payload": payload,
            }
        )

    page.on("response", handle_response)
    page.goto(
        f"{PED_HOME_URL}/entries/{ped_id}",
        wait_until="networkidle",
        timeout=timeout_seconds * 1000,
    )
    html_path = entry_dir / "page.html"
    html_path.write_text(page.content(), encoding="utf-8")
    body_text = page.text_content("body") or ""
    page.remove_listener("response", handle_response)

    network_records: list[_PedNetworkRecord] = []
    if save_network_payloads:
        for index, payload in enumerate(captured_payloads, start=1):
            destination = debug_dir / f"{index:03d}_{_slugify_url(payload['url'])}.json"
            destination.write_text(
                json.dumps(payload["payload"], indent=2, sort_keys=True)
            )
            network_records.append(
                _PedNetworkRecord(
                    ped_id=ped_id,
                    response_url=payload["url"],
                    relative_path=str(destination.relative_to(ped_root)),
                    content_type=payload["content_type"],
                    status_code=int(payload["status"]),
                    size_bytes=destination.stat().st_size,
                    sha256=_sha256_bytes(destination.read_bytes()),
                    crawl_timestamp_utc=_timestamp_utc(),
                )
            )

    entry_payload = _select_entry_payload(captured_payloads)
    if entry_payload is None:
        entry_payload = _fetch_entry_payload_direct(
            ped_id=ped_id,
            timeout_seconds=timeout_seconds,
        )

    entry_row = _build_entry_row(
        ped_id=ped_id,
        entry_payload=entry_payload,
        body_text=body_text,
        ped_root=ped_root,
        workspace=workspace,
        network_payload_count=len(network_records),
        raw_entry_html_path=None,
    )
    model_rows = _download_entry_models(
        ped_id=ped_id,
        entry_payload=entry_payload,
        model_root=model_root,
        timeout_seconds=timeout_seconds,
        ped_root=ped_root,
        workspace=workspace,
    )
    entry_row = _PedEntryRow(
        **{
            **entry_row.to_dict(),
            "ensemble_count": len(entry_payload.get("ensembles", [])),
            "model_count": len(model_rows),
        }
    )
    return entry_row, model_rows, network_records


def _crawl_one_entry_metadata_only(
    ped_root: Path,
    workspace: str,
    ped_id: str,
    timeout_seconds: int,
    save_network_payloads: bool,
) -> tuple[_PedEntryRow, list[_PedModelRow], list[_PedNetworkRecord]]:
    """Crawl one PED entry without downloading model assets."""
    entry_payload = _fetch_entry_payload_direct(
        ped_id=ped_id,
        timeout_seconds=timeout_seconds,
    )

    network_records: list[_PedNetworkRecord] = []
    if save_network_payloads:
        debug_dir = ped_debug_root(ped_root, workspace) / "entries" / ped_id / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        network_path = debug_dir / "001_entry.json"
        network_path.write_text(json.dumps(entry_payload, indent=2, sort_keys=True))
        network_records.append(
            _PedNetworkRecord(
                ped_id=ped_id,
                response_url=f"{PED_DEPOSITION_API_ROOT}/entries/{ped_id}/",
                relative_path=str(network_path.relative_to(ped_root)),
                content_type="application/json",
                status_code=200,
                size_bytes=network_path.stat().st_size,
                sha256=_sha256_bytes(network_path.read_bytes()),
                crawl_timestamp_utc=_timestamp_utc(),
            )
        )

    entry_row = _build_entry_row(
        ped_id=ped_id,
        entry_payload=entry_payload,
        body_text="",
        ped_root=ped_root,
        workspace=workspace,
        network_payload_count=len(network_records),
        raw_entry_html_path=None,
    )
    entry_row = _PedEntryRow(
        **{
            **entry_row.to_dict(),
            "ensemble_count": len(entry_payload.get("ensembles", [])),
            "model_count": _declared_model_count(entry_payload),
        }
    )
    return entry_row, [], network_records


def _build_entry_row(
    ped_id: str,
    entry_payload: dict[str, Any],
    body_text: str,
    ped_root: Path,
    workspace: str,
    network_payload_count: int,
    raw_entry_html_path: str | None = None,
) -> _PedEntryRow:
    """Build one normalized PED entry row."""
    description = entry_payload.get("description") or {}
    ontology_terms = description.get("ontology_terms") or []
    protein_name = description.get("title") or _fallback_protein_name(body_text, ped_id)
    sequence = _collect_sequence(entry_payload.get("construct_chains") or [])
    uniprot_accessions = _pipe_join(
        _extract_uniprot_accessions(entry_payload.get("construct_chains") or [])
    )
    bmrb_ids = _pipe_join(
        _extract_cross_ref_ids(description.get("experimental_cross_reference"), "bmrb")
        + _extract_cross_ref_ids(description.get("entry_cross_reference"), "bmrb")
    )
    experimental_tags = _pipe_join(
        _ontology_tags(ontology_terms, namespace_fragment="Measurement method")
    )
    structural_tags = _pipe_join(
        _structural_tags(
            ontology_terms=ontology_terms,
            structural_text=description.get("structural_ensembles_calculation"),
            md_text=description.get("md_calculation"),
        )
    )
    sequence_length = len(sequence.replace("|", "")) if sequence else None
    return _PedEntryRow(
        ped_id=ped_id,
        entry_url=f"{PED_HOME_URL}/entries/{ped_id}",
        protein_name=protein_name,
        sequence=sequence or None,
        sequence_length=sequence_length,
        uniprot_accessions=uniprot_accessions,
        bmrb_ids=bmrb_ids,
        experimental_procedures=experimental_tags,
        experimental_procedures_raw=_strip_or_none(
            description.get("experimental_procedure")
        ),
        structural_ensemble_calculation_tags=structural_tags,
        structural_ensemble_calculation_raw=_strip_or_none(
            " ".join(
                item
                for item in [
                    description.get("structural_ensembles_calculation"),
                    description.get("md_calculation"),
                ]
                if item
            )
        ),
        ensemble_count=0,
        model_count=0,
        raw_entry_html_path=raw_entry_html_path
        or f"_debug/{workspace}/entries/{ped_id}/page.html",
        network_payload_count=network_payload_count,
        crawl_timestamp_utc=_timestamp_utc(),
    )


def _declared_model_count(entry_payload: dict[str, Any]) -> int:
    """Return the declared total number of models for one PED entry."""
    total = 0
    for ensemble in entry_payload.get("ensembles", []):
        total += int(ensemble.get("models") or 0)
    return total


def _download_entry_models(
    ped_id: str,
    entry_payload: dict[str, Any],
    model_root: Path,
    timeout_seconds: int,
    ped_root: Path,
    workspace: str,
) -> list[_PedModelRow]:
    """Download all ensemble assets for one PED entry and split model PDBs."""
    model_rows: list[_PedModelRow] = []
    for ensemble in entry_payload.get("ensembles", []):
        ensemble_id = ensemble["ensemble_id"]
        archive_url = (
            f"{PED_DEPOSITION_API_ROOT}/entries/{ped_id}/ensembles/{ensemble_id}/"
            "download-all-data/?response_format=json"
        )
        response = requests.get(
            archive_url,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        measures, model_pdbs = _parse_ensemble_archive(response.content)
        ensemble_dir = model_root / ensemble_id
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        for model_index, pdb_bytes in sorted(model_pdbs.items()):
            model_file = f"{ped_id}{ensemble_id}_model_{model_index:05d}.pdb"
            destination = ensemble_dir / model_file
            destination.write_bytes(pdb_bytes)
            metric_row = measures.get(model_index, {})
            model_rows.append(
                _PedModelRow(
                    ped_id=ped_id,
                    ensemble_id=ensemble_id,
                    model_id=f"{ped_id}{ensemble_id}_model_{model_index:05d}",
                    model_index=model_index,
                    model_pdb_url=archive_url,
                    model_pdb_path=str(destination.relative_to(ped_root)),
                    secondary_structure_entropy=_safe_float(
                        metric_row.get("secondary_structure_entropy")
                    ),
                    relative_solvent_accessibility=_safe_float(
                        metric_row.get("relative_solvent_accessibility")
                    ),
                    radius_of_gyration=_safe_float(
                        metric_row.get("radius_of_gyration")
                    ),
                    measure_source=metric_row.get(
                        "measure_source",
                        "ped_archive:gyration_global+dssp_data",
                    ),
                    crawl_timestamp_utc=_timestamp_utc(),
                )
            )
    return model_rows


def _parse_ensemble_archive(
    archive_bytes: bytes,
) -> tuple[dict[int, dict[str, Any]], dict[int, bytes]]:
    """Parse one PED ensemble archive and return measures and split model PDBs."""
    measures: dict[int, dict[str, Any]] = defaultdict(dict)
    pdb_bytes_by_model: dict[int, bytes] = {}

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        dssp_rows = _load_json_member(archive, members, suffix="_dssp_data.json")
        gyration_rows = _load_json_member(
            archive, members, suffix="_gyration_global.json"
        )
        ensemble_member = next(
            (
                member
                for name, member in members.items()
                if name.endswith("ensemble.tar.gz")
            ),
            None,
        )
        if ensemble_member is None:
            raise ValueError("PED ensemble archive is missing ensemble.tar.gz")
        nested_bytes = archive.extractfile(ensemble_member).read()

    for row in gyration_rows:
        model_index = int(row["model"])
        measures[model_index]["radius_of_gyration"] = _safe_float(row.get("gyration"))
        measures[model_index][
            "measure_source"
        ] = "ped_archive:gyration_global+dssp_data"

    rsa_values: dict[int, list[float]] = defaultdict(list)
    ss_states: dict[int, list[str]] = defaultdict(list)
    for row in dssp_rows:
        model_index = int(row["model"])
        rsa_value = _safe_float(row.get("relative_ASA"))
        if rsa_value is not None:
            rsa_values[model_index].append(rsa_value)
        state = str(row.get("secondary_s_dssp") or "").strip()
        if state:
            ss_states[model_index].append(state)

    for model_index, values in rsa_values.items():
        if values:
            measures[model_index]["relative_solvent_accessibility"] = sum(values) / len(
                values
            )
    for model_index, states in ss_states.items():
        if states:
            measures[model_index]["secondary_structure_entropy"] = _shannon_entropy(
                states
            )

    pdb_bytes_by_model = _split_nested_pdb_models(nested_bytes)
    return measures, pdb_bytes_by_model


def _split_nested_pdb_models(nested_archive_bytes: bytes) -> dict[int, bytes]:
    """Split a nested PED ensemble archive into per-model PDB byte payloads."""
    with tarfile.open(fileobj=io.BytesIO(nested_archive_bytes), mode="r:gz") as archive:
        pdb_member = next(
            (member for member in archive.getmembers() if member.name.endswith(".pdb")),
            None,
        )
        if pdb_member is None:
            raise ValueError("PED nested ensemble archive is missing a PDB file")
        pdb_text = archive.extractfile(pdb_member).read().decode("utf-8")
    return _split_multimodel_pdb_text(pdb_text)


def _split_multimodel_pdb_text(pdb_text: str) -> dict[int, bytes]:
    """Split one multi-model PDB string into individual model payloads."""
    lines = pdb_text.splitlines(keepends=True)
    models: dict[int, list[str]] = {}
    current_index: int | None = None
    current_lines: list[str] = []
    model_counter = 0
    for line in lines:
        if line.startswith("MODEL"):
            if current_index is not None and current_lines:
                models[current_index] = list(current_lines)
            model_counter += 1
            current_index = model_counter
            current_lines = [line]
            continue
        if line.startswith("ENDMDL"):
            if current_index is None:
                continue
            current_lines.append(line)
            models[current_index] = list(current_lines)
            current_index = None
            current_lines = []
            continue
        if current_index is not None:
            current_lines.append(line)
    if current_index is not None and current_lines:
        models[current_index] = list(current_lines)
    if not models:
        return {1: pdb_text.encode("utf-8")}
    return {
        index: "".join(model_lines).encode("utf-8")
        for index, model_lines in sorted(models.items())
    }


def _load_json_member(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    suffix: str,
) -> list[dict[str, Any]]:
    """Load one JSON array member from a tar archive by suffix."""
    member = next(
        (item for name, item in members.items() if name.endswith(suffix)), None
    )
    if member is None:
        return []
    return json.loads(archive.extractfile(member).read().decode("utf-8"))


def _probe_api_landing(timeout_seconds: int) -> dict[str, Any]:
    """Probe the public PED API landing URL and classify its response."""
    response = requests.get(
        PED_API_LANDING_URL,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    content_type = response.headers.get("content-type", "")
    text = response.text[:1000]
    looks_like_html = "text/html" in content_type or "<!doctype html" in text.lower()
    usable_json_api = "application/json" in content_type
    return {
        "url": PED_API_LANDING_URL,
        "status_code": response.status_code,
        "content_type": content_type,
        "looks_like_spa_html_shell": looks_like_html,
        "usable_json_api": usable_json_api,
    }


def _discover_listing_api(
    ped_root: Path,
    workspace: str,
    timeout_seconds: int,
    save_network_payloads: bool,
) -> dict[str, Any]:
    """Discover the live PED listing API URL from the rendered browse page."""
    browse_dir = ped_debug_root(ped_root, workspace) / "browse"
    debug_dir = browse_dir / "debug"
    browse_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    listing_url = PED_DEPOSITION_ENTRY_LIST_URL
    captured_payloads: list[dict[str, Any]] = []

    browser = _launch_browser()
    try:
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(timeout_seconds * 1000)

        def handle_response(response) -> None:
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return
            if "/api/v1/entries/" not in response.url:
                return
            try:
                payload = response.json()
            except Exception:
                return
            captured_payloads.append(
                {
                    "url": response.url,
                    "status": response.status,
                    "content_type": content_type,
                    "payload": payload,
                }
            )

        page.on("response", handle_response)
        page.goto(
            PED_BROWSE_URL, wait_until="networkidle", timeout=timeout_seconds * 1000
        )
        (browse_dir / "page.html").write_text(page.content(), encoding="utf-8")
        page.remove_listener("response", handle_response)
        context.close()
    finally:
        browser.close()

    if captured_payloads:
        listing_url = captured_payloads[0]["url"]

    if save_network_payloads:
        for index, payload in enumerate(captured_payloads, start=1):
            destination = debug_dir / f"{index:03d}_{_slugify_url(payload['url'])}.json"
            destination.write_text(
                json.dumps(payload["payload"], indent=2, sort_keys=True)
            )

    return {
        "browse_url": PED_BROWSE_URL,
        "listing_url": listing_url,
        "captured_payloads": len(captured_payloads),
    }


def _fetch_listing_rows(
    listing_url: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Fetch all PED listing rows by paginating the deposition API."""
    parsed_limit = max(_extract_query_int(listing_url, "limit", default=20), 1000)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        paged_url = re.sub(r"offset=\d+", f"offset={offset}", listing_url)
        paged_url = re.sub(r"limit=\d+", f"limit={parsed_limit}", paged_url)
        response = requests.get(
            paged_url,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("result") or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= int(payload.get("count", 0)):
            break
    return rows


def _resolve_seed_targets(
    ped_root: Path,
    listing_rows: list[dict[str, Any]],
) -> tuple[list[str], list[_PedSeedResolution]]:
    """Resolve curated legacy PED seeds to live PED entry IDs."""
    registry = _load_benchmark_registry(ped_root)
    live_titles = {
        row["entry_id"]: _normalize_match_text(
            (row.get("description") or {}).get("title")
        )
        for row in listing_rows
    }
    resolution_rows: list[_PedSeedResolution] = []
    resolved_ids: list[str] = []
    for record in registry:
        if record.get("source_db") != "PED":
            continue
        legacy_id = record["external_id"]
        protein_name = record["protein_name"]
        resolved = None
        strategy = "unresolved"
        alias_id = LEGACY_PED_ID_ALIASES.get(legacy_id)
        if alias_id is not None and alias_id in live_titles:
            resolved = alias_id
            strategy = "curated_alias"
        elif record.get("cross_refs", {}).get("PED") in live_titles:
            resolved = record["cross_refs"]["PED"]
            strategy = "existing_cross_ref"
        else:
            name_candidates = SEED_NAME_ALIASES.get(protein_name, [protein_name])
            for candidate in name_candidates:
                normalized_candidate = _normalize_match_text(candidate)
                matches = [
                    ped_id
                    for ped_id, normalized_title in live_titles.items()
                    if normalized_candidate and normalized_candidate in normalized_title
                ]
                if matches:
                    resolved = matches[0]
                    strategy = "title_match"
                    break
        resolution_rows.append(
            _PedSeedResolution(
                legacy_id=legacy_id,
                protein_name=protein_name,
                resolved_ped_id=resolved,
                resolution_strategy=strategy,
            )
        )
        if resolved and resolved not in resolved_ids:
            resolved_ids.append(resolved)
    return resolved_ids, resolution_rows


def _enrich_benchmark_registry(
    ped_root: Path,
    workspace: str,
    entry_rows: list[_PedEntryRow],
    model_rows: list[_PedModelRow],
    resolution_records: list[_PedSeedResolution],
) -> dict[str, Any]:
    """Update the benchmark registry with PED crawl metadata when available."""
    metadata_tables_root = meta_tables_root(ped_root.parent)
    registry_path = metadata_tables_root / "benchmark_registry.json"
    registry_jsonl_path = metadata_tables_root / "benchmark_registry.jsonl"
    summary_path = metadata_tables_root / "benchmark_summary.json"
    if not registry_path.exists():
        return {"updated_records": 0}

    registry = json.loads(registry_path.read_text())
    entry_by_id = {row.ped_id: row for row in entry_rows}
    model_counts = defaultdict(int)
    for row in model_rows:
        model_counts[row.ped_id] += 1
    resolution_by_legacy = {row.legacy_id: row for row in resolution_records}

    updated_records = 0
    if workspace != "benchmark":
        return {"updated_records": 0}
    for record in registry:
        if record.get("source_db") != "PED":
            continue
        resolution = resolution_by_legacy.get(record["external_id"])
        if resolution is None or resolution.resolved_ped_id is None:
            record["ped_resolution_status"] = "unresolved"
            continue
        entry = entry_by_id.get(resolution.resolved_ped_id)
        if entry is None:
            continue
        record["ped_resolution_status"] = "resolved"
        record["ped_live_id"] = resolution.resolved_ped_id
        record["sequence_hash"] = (
            hashlib.sha256(entry.sequence.encode("utf-8")).hexdigest()
            if entry.sequence
            else None
        )
        first_uniprot = None
        if entry.uniprot_accessions:
            first_uniprot = entry.uniprot_accessions.split("|")[0]
        record["uniprot_accession"] = first_uniprot
        record["cross_refs"]["PED"] = resolution.resolved_ped_id
        record["ped_entry_url"] = entry.entry_url
        record["ped_workspace"] = workspace
        record["ped_ensemble_count"] = entry.ensemble_count
        record["ped_model_count"] = model_counts[resolution.resolved_ped_id]
        record["ped_local_entry_html"] = entry.raw_entry_html_path
        record["ped_local_model_root"] = (
            f"{ped_root.name}/assets/{workspace}/models/{resolution.resolved_ped_id}"
        )
        coordinates = record.setdefault("assets", {}).setdefault("coordinates", [])
        coordinates.append(
            {
                "label": "PED crawled model assets",
                "path": (
                    f"{ped_root.name}/assets/{workspace}/models/"
                    f"{resolution.resolved_ped_id}"
                ),
            }
        )
        updated_records += 1

    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    registry_jsonl_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in registry)
        + ("\n" if registry else "")
    )
    summary = _summarize_registry(registry)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {"updated_records": updated_records}


def _summarize_registry(registry: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the benchmark-summary payload for the external registry."""
    track_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    state_counts: dict[str, int] = defaultdict(int)
    for row in registry:
        track_counts[row["benchmark_track"]] += 1
        source_counts[row["source_db"]] += 1
        state_counts[row["state_class"]] += 1
    return {
        "records": len(registry),
        "track_counts": dict(track_counts),
        "source_counts": dict(source_counts),
        "state_class_counts": dict(state_counts),
    }


def _write_ped_entries_tsv(path: Path, rows: list[_PedEntryRow]) -> None:
    """Write the normalized PED entry table as a TSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PED_ENTRY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.ped_id):
            writer.writerow(row.to_dict())


def _write_ped_models_tsv(path: Path, rows: list[_PedModelRow]) -> None:
    """Write the normalized PED model table as a TSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PED_MODEL_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (item.ped_id, item.ensemble_id, item.model_index)
        ):
            writer.writerow(row.to_dict())


def _persist_ped_manifests(
    manifests_dir: Path,
    debug_root: Path,
    entry_rows: list[_PedEntryRow],
    model_rows: list[_PedModelRow],
    network_rows: list[_PedNetworkRecord],
    write_models: bool,
) -> None:
    """Persist PED TSV and JSONL manifests for the current crawl state."""
    _write_ped_entries_tsv(manifests_dir / "ped_entries.tsv", entry_rows)
    ped_models_path = manifests_dir / "ped_models.tsv"
    if write_models and model_rows:
        _write_ped_models_tsv(ped_models_path, model_rows)
    elif ped_models_path.exists():
        ped_models_path.unlink()
    _write_jsonl(debug_root / "ped_entry_records.jsonl", entry_rows)
    _write_jsonl(debug_root / "ped_model_records.jsonl", model_rows)
    _write_jsonl(debug_root / "ped_debug_manifest.jsonl", network_rows)


def _resolve_debug_manifest(debug_root: Path) -> Path:
    """Resolve the PED debug manifest with backward-compatible fallback."""
    debug_path = debug_root / "ped_debug_manifest.jsonl"
    if debug_path.exists():
        return debug_path
    legacy_path = debug_root / "ped_network_manifest.jsonl"
    return legacy_path


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    """Write dataclass-style rows as JSONL."""
    payload = "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _write_ped_readme(root: Path) -> None:
    """Write the PED-specific crawl README."""
    (root / "README.md").write_text(
        "# PED Layout\n\n"
        "This subtree stores all PED-derived assets, metadata tables, and "
        "debug provenance for AtypEmu. PED is managed under a single root "
        "with source-owned tables, assets, datasets, and one hidden debug "
        "subtree.\n\n"
        "## Source Characteristics\n\n"
        "- PED is the primary external free-state ensemble source used by "
        "AtypEmu for future training augmentation and benchmark assembly.\n"
        "- The public `https://proteinensemble.org/api` endpoint is currently "
        "a single-page-app shell rather than a stable requests-only JSON API.\n"
        "- The collector therefore uses a Playwright hybrid strategy: render "
        "the browse page to discover the live deposition API, capture "
        "structured JSON responses when useful, and then use the deposition "
        "API for scalable entry or asset collection.\n\n"
        "## API Limitations\n\n"
        "- `proteinensemble.org/api` is treated as an unsupported direct JSON "
        "endpoint in v1.\n"
        "- The collector discovers or confirms the live deposition API during "
        "browse rendering and uses the deposition API for entry JSON and "
        "ensemble-archive downloads.\n"
        "- The per-model PDB files are generated by splitting PED multi-model "
        "ensemble PDB assets. They remain source-linked through "
        "`ped_models.tsv`.\n\n"
        "## Layout\n\n"
        "- `tables/benchmark/`: local benchmark and validation tables.\n"
        "- `tables/catalog/`: full live-PED metadata index used for "
        "classification, BMRB overlap analysis, and validation-policy "
        "generation.\n"
        "- `assets/benchmark/models/`: per-model PED PDB assets for the "
        "benchmark workspace.\n"
        "- `datasets/`: source-owned Parquet exports for trainer-facing "
        "loading.\n"
        "- `_debug/benchmark/`: crawl summaries, JSONL sidecars, page "
        "snapshots, and payload captures for the benchmark workspace.\n"
        "- `_debug/catalog/`: metadata-crawl logs and debug sidecars for the "
        "catalog workspace.\n\n"
        "Public files in `tables/benchmark/`:\n\n"
        "- `ped_entries.tsv`\n"
        "- `ped_models.tsv`\n"
        "- `ped_generation_classes.tsv`\n"
        "- `ped_bmrb_bridge.tsv`\n"
        "- `ped_validation_manifest.tsv`\n"
        "- `ped_experimental_tags.tsv`\n"
        "- `ped_structural_tags.tsv`\n"
        "Assets in `assets/benchmark/`:\n\n"
        "- `models/<PED_ID>/<ensemble_id>/*.pdb`\n\n"
        "Public files in `tables/catalog/`:\n\n"
        "- `ped_entries.tsv`\n"
        "- `ped_generation_classes.tsv`\n"
        "- `ped_bmrb_bridge.tsv`\n"
        "- `ped_validation_manifest.tsv`\n"
        "- `ped_experimental_tags.tsv`\n"
        "- `ped_structural_tags.tsv`\n\n"
        "Debug-only files remain under `_debug/` and should not be used as "
        "primary biological tables.\n\n"
        "## Output Schema Notes\n\n"
        "- `experimental_procedures` and "
        "`structural_ensemble_calculation_tags` are pipe-separated normalized "
        "tag lists.\n"
        "- `experimental_procedures_raw` and "
        "`structural_ensemble_calculation_raw` preserve PED-provided text.\n"
        "- `radius_of_gyration` is taken from PED archive JSON directly.\n"
        "- `relative_solvent_accessibility` is the mean of PED-provided "
        "`relative_ASA` values per model from `*_dssp_data.json`.\n"
        "- `secondary_structure_entropy` is computed from PED-provided DSSP "
        "state frequencies per model. The collector does not rerun DSSP or "
        "recompute these measures from local coordinates.\n\n"
        "## Reproducible Commands\n\n"
        "Run these commands from the repository root.\n\n"
        "Install Playwright browser dependencies:\n\n"
        "```bash\n"
        "./scripts/external/install_ped_playwright.sh\n"
        "```\n\n"
        "Crawl PED benchmark assets:\n\n"
        "```bash\n"
        "./scripts/external/crawl_ped_seeds.sh\n"
        "```\n\n"
        "Crawl the full live PED catalog in metadata-only mode:\n\n"
        "```bash\n"
        "./scripts/external/crawl_ped_full.sh\n"
        "```\n\n"
        "Refresh an existing PED crawl:\n\n"
        "```bash\n"
        "./scripts/external/refresh_ped_crawl.sh\n"
        "```\n\n"
        "Classify PED entries by generation strategy:\n\n"
        "```bash\n"
        "./scripts/external/classify_ped.sh\n"
        "```\n\n"
        "Wait for the full catalog crawl to finish and then build the "
        "PED classification and bridge automatically:\n\n"
        "```bash\n"
        "./scripts/external/watch_ped_full_bridge.sh\n"
        "```\n\n"
        "Environment variables:\n\n"
        "- `PED_ROOT`: defaults to `data/ped`.\n"
        "- `PED_WORKSPACE`: `benchmark` or `catalog`.\n"
        "- `PED_TIMEOUT_SECONDS`: defaults to `60`.\n"
        "- `PED_MAX_ENTRIES`: optional hard crawl limit.\n"
        "- `PED_REFRESH=1`: replace cached PED entry outputs.\n"
        "- `PED_SAVE_NETWORK_PAYLOADS=0`: disable JSON payload snapshots.\n\n"
        "Build the PED-to-BMRB bridge after a crawl:\n\n"
        "```bash\n"
        "./scripts/external/prepare_ped_bridge.sh\n"
        "```\n\n"
        "The `catalog` workspace is metadata-only by default so it can scale "
        "to the entire live PED index. The `benchmark` workspace downloads "
        "and splits per-model PDB assets for local validation work.\n\n"
        "## Failure Modes\n\n"
        "- Some historical PED benchmark seeds use legacy IDs that no longer "
        "resolve directly against the current live PED numbering scheme.\n"
        "- In those cases the collector records an unresolved seed rather than "
        "failing the entire crawl.\n"
        "- Missing PED-provided model measures remain empty in TSV and `null` "
        "in JSONL.\n\n"
        "## Rerun and Refresh Policy\n\n"
        "- Re-running without `PED_REFRESH=1` reuses existing entry and model "
        "records when available.\n"
        "- Re-running with refresh replaces the entry page mirror, network "
        "payloads, and split model files for the targeted entries.\n"
    )


def _load_benchmark_registry(ped_root: Path) -> list[dict[str, Any]]:
    """Load the shared source-level benchmark registry."""
    registry_path = meta_tables_root(ped_root.parent) / "benchmark_registry.json"
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text())


def _load_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    """Load JSONL rows into a dictionary keyed by one field."""
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        rows[str(payload[key])] = payload
    return rows


def _load_jsonl_grouped(path: Path, key: str) -> dict[str, list[dict[str, Any]]]:
    """Load JSONL rows grouped by one key field."""
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        grouped[str(payload[key])].append(payload)
    return dict(grouped)


def _entry_row_from_dict(payload: dict[str, Any]) -> _PedEntryRow:
    """Restore an entry dataclass from a manifest dictionary."""
    return _PedEntryRow(**payload)


def _model_row_from_dict(payload: dict[str, Any]) -> _PedModelRow:
    """Restore a model dataclass from a manifest dictionary."""
    return _PedModelRow(**payload)


def _network_row_from_dict(payload: dict[str, Any]) -> _PedNetworkRecord:
    """Restore a network dataclass from a manifest dictionary."""
    return _PedNetworkRecord(**payload)


def _fetch_entry_payload_direct(ped_id: str, timeout_seconds: int) -> dict[str, Any]:
    """Fetch one PED entry JSON directly from the deposition API."""
    response = requests.get(
        f"{PED_DEPOSITION_API_ROOT}/entries/{ped_id}/",
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def _select_entry_payload(
    captured_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the main entry JSON payload from captured network responses."""
    for payload in captured_payloads:
        if re.search(r"/entries/PED\d+/?$", payload["url"]):
            return payload["payload"]
    return None


def _extract_uniprot_accessions(construct_chains: list[dict[str, Any]]) -> list[str]:
    """Extract sorted unique UniProt accessions from construct-chain metadata."""
    values = set()
    for chain in construct_chains:
        for fragment in chain.get("fragments") or []:
            accession = fragment.get("uniprot_acc")
            if accession:
                values.add(str(accession))
    return sorted(values)


def _extract_cross_ref_ids(
    cross_refs: list[dict[str, Any]] | None,
    database_name: str,
) -> list[str]:
    """Extract identifiers for one database from PED cross-reference rows."""
    values = {
        str(item["id"])
        for item in cross_refs or []
        if str(item.get("db", "")).lower() == database_name.lower()
    }
    return sorted(values)


def _collect_sequence(construct_chains: list[dict[str, Any]]) -> str:
    """Collect one pipe-joined construct sequence from PED chain fragments."""
    sequences: list[str] = []
    for chain in construct_chains:
        fragments = sorted(
            chain.get("fragments") or [],
            key=lambda item: (
                item.get("start_position") or 0,
                item.get("end_position") or 0,
            ),
        )
        chain_sequence = "".join(
            str(fragment.get("source_sequence") or "").strip() for fragment in fragments
        )
        if chain_sequence:
            sequences.append(chain_sequence)
    return "|".join(sequences)


def _ontology_tags(
    ontology_terms: list[dict[str, Any]],
    namespace_fragment: str,
) -> list[str]:
    """Extract normalized ontology term names for one namespace fragment."""
    values = []
    for item in ontology_terms:
        namespace = str(item.get("namespace") or "")
        if namespace_fragment.lower() not in namespace.lower():
            continue
        name = _strip_or_none(item.get("name"))
        if name:
            values.append(name)
    return sorted(set(values))


def _structural_tags(
    ontology_terms: list[dict[str, Any]],
    structural_text: str | None,
    md_text: str | None,
) -> list[str]:
    """Extract normalized structural-calculation tags for one PED entry."""
    tags = [
        _strip_or_none(item.get("name"))
        for item in ontology_terms
        if "Measurement method" not in str(item.get("namespace") or "")
    ]
    for chunk in [structural_text, md_text]:
        if not chunk:
            continue
        if "molecular dynamics" in chunk.lower():
            tags.append("Molecular dynamics")
    return sorted({item for item in tags if item})


def _fallback_protein_name(body_text: str, ped_id: str) -> str | None:
    """Extract a best-effort protein title from rendered PED body text."""
    match = re.search(
        rf"{re.escape(ped_id)}\s+(.*?)\s+Data owner",
        " ".join(body_text.split()),
    )
    if match:
        return match.group(1).strip()
    return None


def _normalize_match_text(text: str | None) -> str:
    """Normalize a name-like string for robust seed matching."""
    if text is None:
        return ""
    normalized = text.lower()
    normalized = normalized.replace("β", "beta").replace("α", "alpha")
    normalized = normalized.replace("γ", "gamma")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _normalize_ped_id(value: str) -> str:
    """Normalize a user-provided PED ID string."""
    return value.strip().upper()


def _pipe_join(values: list[str]) -> str | None:
    """Join sorted unique strings with pipe separators."""
    items = sorted({item.strip() for item in values if item and item.strip()})
    return "|".join(items) if items else None


def _shannon_entropy(states: list[str]) -> float:
    """Compute Shannon entropy over a list of categorical states."""
    counts: dict[str, int] = defaultdict(int)
    for state in states:
        counts[state] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(
        -sum((count / total) * math.log(count / total) for count in counts.values())
    )


def _safe_float(value: Any) -> float | None:
    """Convert a numeric-like value to float while preserving missing values."""
    if value in {None, "", "null"}:
        return None
    return float(value)


def _extract_query_int(url: str, key: str, default: int) -> int:
    """Extract an integer query parameter from one URL."""
    match = re.search(rf"[?&]{re.escape(key)}=(\d+)", url)
    return int(match.group(1)) if match else default


def _slugify_url(url: str) -> str:
    """Convert a URL into a filename-safe slug."""
    stripped = re.sub(r"^https?://", "", url)
    return re.sub(r"[^A-Za-z0-9]+", "_", stripped).strip("_").lower()


def _strip_or_none(value: Any) -> str | None:
    """Strip string-like input and return ``None`` when empty."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 checksum for one byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _timestamp_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _launch_browser():
    """Launch a Chromium browser for PED crawl rendering."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for PED crawling. Install it with "
            "`python -m pip install playwright` and run "
            "`python -m playwright install chromium`."
        ) from exc
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)

    class _BrowserWrapper:
        def __init__(self, browser_obj, playwright_obj) -> None:
            self._browser = browser_obj
            self._playwright = playwright_obj

        def new_context(self):
            return self._browser.new_context()

        def close(self) -> None:
            self._browser.close()
            self._playwright.stop()

    return _BrowserWrapper(browser, playwright)
