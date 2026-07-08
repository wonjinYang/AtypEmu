"""Bootstrap helpers for source-level database roots under ``data/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.common.catalog import (
    _build_benchmark_registry,
    _build_source_catalog,
    _summarize_benchmark_registry,
)


def meta_root(data_root: str | Path) -> Path:
    """Return the cross-source metadata root under ``data/``."""
    return Path(data_root) / "meta"


def meta_tables_root(data_root: str | Path) -> Path:
    """Return the metadata tables directory under ``data/meta``."""
    return meta_root(data_root) / "tables"


def meta_datasets_root(data_root: str | Path) -> Path:
    """Return the metadata dataset directory under ``data/meta``."""
    return meta_root(data_root) / "datasets"


def setup_source_workspace(
    data_root: str | Path,
    capture_source_pages: bool = False,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Create source-level database roots and meta bootstrap manifests.

    Args:
        data_root: AtypEmu ``data/`` root to create or update.
        capture_source_pages: Whether to download lightweight source landing
            pages into each source ``_debug`` subtree.
        timeout_seconds: Per-request timeout used for optional page captures.

    Returns:
        Summary dictionary describing the prepared source workspace.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_root = meta_root(root)
    tables_root = meta_tables_root(root)
    datasets_root = meta_datasets_root(root)
    metadata_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    source_catalog = _build_source_catalog()
    benchmark_registry_path = tables_root / "benchmark_registry.json"
    if benchmark_registry_path.exists():
        benchmark_registry = json.loads(benchmark_registry_path.read_text())
    else:
        benchmark_registry = _build_benchmark_registry()

    snapshot_summary: dict[str, Any] = {}
    for source_db, payload in source_catalog.items():
        source_root = root / source_db.lower()
        debug_source_root = _ensure_source_root(
            source_root=source_root,
            source_db=source_db,
        )
        _write_source_root_readme_if_missing(
            source_root=source_root,
            source_db=source_db,
            source_config=payload,
        )

        source_records = [
            row for row in benchmark_registry if row["source_db"] == source_db
        ]
        manifest_payload = {
            "source": payload,
            "record_count": len(source_records),
            "records": source_records,
        }
        (debug_source_root / "source_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True)
        )
        (debug_source_root / "bootstrap_records.json").write_text(
            json.dumps(source_records, indent=2, sort_keys=True)
        )

        if capture_source_pages:
            snapshot_summary[source_db] = _capture_source_pages(
                source_config=payload,
                debug_source_root=debug_source_root,
                timeout_seconds=timeout_seconds,
            )

    benchmark_summary = _summarize_benchmark_registry(benchmark_registry)
    (tables_root / "source_catalog.json").write_text(
        json.dumps(source_catalog, indent=2, sort_keys=True)
    )
    (tables_root / "benchmark_registry.json").write_text(
        json.dumps(benchmark_registry, indent=2, sort_keys=True)
    )
    (tables_root / "benchmark_registry.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in benchmark_registry)
        + ("\n" if benchmark_registry else "")
    )
    (tables_root / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2, sort_keys=True)
    )
    _write_meta_bootstrap_readme(metadata_root)

    return {
        "data_root": str(root),
        "meta_root": str(metadata_root),
        "sources": len(source_catalog),
        "benchmark_records": len(benchmark_registry),
        "tracks": benchmark_summary["track_counts"],
        "captured_sources": sum(
            1 for row in snapshot_summary.values() if row.get("captured_pages", 0) > 0
        ),
    }


def _ensure_source_root(source_root: Path, source_db: str) -> Path:
    """Create the standard subtree layout for one source root.

    Args:
        source_root: Top-level source-owned directory under ``data/``.
        source_db: Canonical source label such as ``PED`` or ``SASBDB``.

    Returns:
        The debug subtree used for source-wide manifests and page captures.
    """
    if source_db == "PED":
        paths = [
            source_root / "tables" / "benchmark",
            source_root / "tables" / "catalog",
            source_root / "assets" / "benchmark" / "models",
            source_root / "datasets",
            source_root / "_debug" / "benchmark",
            source_root / "_debug" / "catalog",
            source_root / "_debug" / "source" / "pages",
            source_root / "_debug" / "source" / "downloads",
        ]
        debug_source_root = source_root / "_debug" / "source"
    else:
        paths = [
            source_root / "downloads",
            source_root / "tables",
            source_root / "datasets",
            source_root / "assets",
            source_root / "_debug" / "pages",
            source_root / "_debug" / "downloads",
        ]
        debug_source_root = source_root / "_debug"

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return debug_source_root


def _capture_source_pages(
    source_config: dict[str, Any],
    debug_source_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Download lightweight upstream pages for reproducible inspection.

    Args:
        source_config: One source entry from ``SOURCE_CATALOG``.
        debug_source_root: Root under which ``pages`` snapshots are stored.
        timeout_seconds: Per-request timeout in seconds.

    Returns:
        Summary dictionary with capture counts and any errors.
    """
    captured_pages = 0
    errors: list[str] = []
    pages_dir = debug_source_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for label, url in source_config["official_urls"].items():
        try:
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code != 200:
                errors.append(f"{label}:{response.status_code}")
                continue
            suffix = (
                ".json"
                if "json" in response.headers.get("content-type", "")
                else ".html"
            )
            destination = pages_dir / f"{label}{suffix}"
            destination.write_text(response.text)
            captured_pages += 1
        except Exception as exc:  # pragma: no cover - network-path fallback
            errors.append(f"{label}:{type(exc).__name__}")

    return {
        "captured_pages": captured_pages,
        "errors": errors,
    }


def _write_source_root_readme_if_missing(
    source_root: Path,
    source_db: str,
    source_config: dict[str, Any],
) -> None:
    """Write a generic source-root README for scaffolded database sources.

    Args:
        source_root: Top-level on-disk source root under ``data/``.
        source_db: Human-readable database label.
        source_config: Source metadata taken from ``SOURCE_CATALOG``.
    """
    readme_path = source_root / "README.md"
    if readme_path.exists():
        return

    state_classes = ", ".join(
        f"`{value}`" for value in source_config["default_state_classes"]
    )
    use_tiers = ", ".join(f"`{value}`" for value in source_config["default_use_tiers"])
    official_urls = "\n".join(
        f"- `{label}`: {url}"
        for label, url in sorted(source_config["official_urls"].items())
    )

    readme_path.write_text(
        f"# {source_db} Data Root\n\n"
        "## Purpose\n\n"
        f"This directory is the source-owned {source_db} root for AtypEmu.\n\n"
        f"{source_config['notes']}\n\n"
        "## Source policy\n\n"
        f"- collection role: {source_config['collection_role']}\n"
        f"- default state classes: {state_classes}\n"
        f"- default use tiers: {use_tiers}\n\n"
        "## Directory contract\n\n"
        "- `downloads/`\n"
        "  - source-downloaded raw files that remain meaningful inputs\n"
        "- `tables/`\n"
        "  - curated human-readable biological tables\n"
        "- `datasets/`\n"
        "  - Parquet exports and source-level dataset manifests\n"
        "- `assets/`\n"
        "  - heavy reusable source assets when available\n"
        "- `_debug/`\n"
        "  - crawl logs, rendered pages, payload captures, and provenance-only artifacts\n\n"
        "## Public tables\n\n"
        "Curated human-readable outputs belong under `tables/`. "
        "Trainer-facing Parquet bundles and `manifest.json` belong under "
        "`datasets/`. Provenance-only artifacts stay in `_debug/`.\n\n"
        "## Dataset bundle\n\n"
        "When this source gains a dataframe export, the canonical registry "
        "contract is `datasets/manifest.json` plus one or more workspace "
        "directories such as `datasets/default/`.\n\n"
        "## Reproducible commands\n\n"
        "Run these commands from the repository root.\n\n"
        "Bootstrap source roots:\n\n"
        "```bash\n"
        "PYTHONPATH=src python -m atypemu.cli.databases.setup_source_workspace \\\n"
        "  --data-root data\n"
        "```\n\n"
        f"Crawl curated {source_db} assets:\n\n"
        "```bash\n"
        f"./scripts/external/crawl_external_source.sh {source_db}\n"
        "```\n\n"
        "## Official references\n\n"
        f"{official_urls}\n"
    )


def _write_meta_bootstrap_readme(meta_root: Path) -> None:
    """Write the source-bootstrap README under ``data/meta``.

    Args:
        meta_root: Meta root where bootstrap manifests are written.
    """
    (meta_root / "README.md").write_text(
        "# Source Metadata Root\n\n"
        "This subtree stores cross-source metadata for AtypEmu.\n\n"
        "## Directory contract\n\n"
        "- `tables/`\n"
        "  - JSON and JSONL bootstrap metadata such as the source catalog, "
        "benchmark registry, and crawl summaries\n"
        "- `datasets/`\n"
        "  - Parquet outputs and `manifest.json` used by "
        "`MetaDataFrameRegistry`\n\n"
        "## Ownership\n\n"
        "Cross-source bootstrap metadata belongs in `data/meta/tables/`. "
        "Cross-source dataframe outputs belong in `data/meta/datasets/`. "
        "Source-owned biological data stay under `data/<source>/`.\n\n"
        "## Reproducible commands\n\n"
        "Run these commands from the repository root.\n\n"
        "```bash\n"
        "PYTHONPATH=src python -m atypemu.cli.databases.setup_source_workspace \\\n"
        "  --data-root data\n"
        "./scripts/external/crawl_all_external.sh\n"
        "PYTHONPATH=src python -m atypemu.cli.integrated.export_meta_dataframes \\\n"
        "  --data-root data --refresh\n"
        "```\n"
    )
