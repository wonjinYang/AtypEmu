"""Curated source-asset crawl helpers for source-level database roots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.common.bootstrap import meta_tables_root, setup_source_workspace
from atypemu.databases.common.catalog import CURATED_SOURCE_ASSETS, SOURCE_CATALOG


def crawl_source_assets(
    data_root: str | Path,
    sources: list[str] | None = None,
    include_large_assets: bool = False,
    refresh: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Download curated source assets into source-owned data roots.

    Args:
        data_root: AtypEmu ``data/`` root to update.
        sources: Optional subset of source names such as ``["PED", "IDEAL"]``.
        include_large_assets: Whether large optional archives should be
            downloaded.
        refresh: Whether existing downloaded files should be replaced.
        timeout_seconds: Per-request timeout in seconds.

    Returns:
        Summary dictionary describing the crawl outcome.
    """
    root = Path(data_root)
    tables_root = meta_tables_root(root)
    tables_root.mkdir(parents=True, exist_ok=True)
    setup_source_workspace(data_root=root)

    requested_sources = _normalize_requested_sources(sources)
    crawl_summaries: dict[str, dict[str, Any]] = {}
    total_assets = 0
    downloaded_assets = 0
    existing_assets = 0
    error_assets = 0

    for source_db in requested_sources:
        source_root = root / source_db.lower()
        manifest = _crawl_one_source(
            source_db=source_db,
            source_root=source_root,
            include_large_assets=include_large_assets,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )
        crawl_summaries[source_db] = manifest["summary"]
        total_assets += manifest["summary"]["assets_considered"]
        downloaded_assets += manifest["summary"]["downloaded"]
        existing_assets += manifest["summary"]["existing"]
        error_assets += manifest["summary"]["errors"]

    crawl_summary = {
        "data_root": str(root),
        "meta_root": str(root / "meta"),
        "sources_requested": requested_sources,
        "include_large_assets": include_large_assets,
        "refresh": refresh,
        "assets_considered": total_assets,
        "downloaded": downloaded_assets,
        "existing": existing_assets,
        "errors": error_assets,
        "per_source": crawl_summaries,
        "timestamp_utc": _timestamp_utc(),
    }
    (tables_root / "crawl_summary.json").write_text(
        json.dumps(crawl_summary, indent=2, sort_keys=True)
    )
    return crawl_summary


def _normalize_requested_sources(sources: list[str] | None) -> list[str]:
    """Normalize the requested source list.

    Args:
        sources: Optional list of requested source names.

    Returns:
        Canonical source names in crawl order.
    """
    if sources is None:
        return list(SOURCE_CATALOG)

    normalized: list[str] = []
    known = {source.lower(): source for source in SOURCE_CATALOG}
    for item in sources:
        canonical = known.get(item.strip().lower())
        if canonical is None:
            raise ValueError(f"Unsupported external source: {item}")
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _crawl_one_source(
    source_db: str,
    source_root: Path,
    include_large_assets: bool,
    refresh: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Download the curated assets for one source root."""
    asset_records: list[dict[str, Any]] = []
    summary = {
        "assets_considered": 0,
        "downloaded": 0,
        "existing": 0,
        "errors": 0,
        "skipped_large": 0,
    }

    for asset in CURATED_SOURCE_ASSETS.get(source_db, []):
        if asset.get("large_asset") and not include_large_assets:
            summary["skipped_large"] += 1
            continue

        summary["assets_considered"] += 1
        destination = _resolve_asset_destination(source_root, source_db, asset)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = _download_curated_asset(
            destination=destination,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
            **asset,
        )
        asset_records.append(record)
        if record["status"] == "downloaded":
            summary["downloaded"] += 1
        elif record["status"] == "existing":
            summary["existing"] += 1
        else:
            summary["errors"] += 1

    manifest = {
        "source_db": source_db,
        "timestamp_utc": _timestamp_utc(),
        "include_large_assets": include_large_assets,
        "refresh": refresh,
        "summary": summary,
        "assets": asset_records,
    }
    debug_root = _source_debug_root(source_root, source_db)
    debug_root.mkdir(parents=True, exist_ok=True)
    (debug_root / "crawl_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def _resolve_asset_destination(
    source_root: Path,
    source_db: str,
    asset: dict[str, Any],
) -> Path:
    """Resolve the on-disk destination for one curated source asset."""
    relative_path = Path(str(asset["relative_path"]))
    if asset["category"] == "pages":
        return _source_debug_root(source_root, source_db) / relative_path
    if asset["category"] == "downloads":
        return source_root / relative_path
    return source_root / "assets" / relative_path


def _source_debug_root(source_root: Path, source_db: str) -> Path:
    """Return the debug root for one source."""
    if source_db == "PED":
        return source_root / "_debug" / "source"
    return source_root / "_debug"


def _download_curated_asset(
    label: str,
    url: str,
    category: str,
    relative_path: str,
    destination: Path,
    refresh: bool,
    timeout_seconds: int,
    large_asset: bool = False,
) -> dict[str, Any]:
    """Download one curated asset and return its manifest record."""
    record = {
        "label": label,
        "url": url,
        "category": category,
        "relative_path": relative_path,
        "large_asset": large_asset,
        "status": "error",
        "content_type": None,
        "size_bytes": None,
        "sha256": None,
        "error": None,
    }

    if destination.exists() and not refresh:
        record["status"] = "existing"
        record.update(_file_metadata(destination))
        return record

    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "AtypEmu/0.1 external collector"},
        )
        if response.status_code != 200:
            record["error"] = f"http_{response.status_code}"
            return record
        destination.write_bytes(response.content)
        record["status"] = "downloaded"
        record["content_type"] = response.headers.get("content-type")
        record.update(_file_metadata(destination))
        return record
    except Exception as exc:  # pragma: no cover - network-path fallback
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record


def _file_metadata(path: Path) -> dict[str, Any]:
    """Return size and checksum metadata for a local file."""
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _timestamp_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
