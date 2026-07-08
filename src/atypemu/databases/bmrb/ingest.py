"""BMRB download, parsing, and manifest preparation helpers."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from atypemu.databases.bmrb.config import MERGED_BASE_URL, RESTRAINTS_LINKS_URL
from atypemu.databases.bmrb.layout import (
    bmrb_bundle_root,
    bmrb_manifest_root,
    bmrb_raw_nmrstar_root,
    bmrb_raw_noe_root,
)
from atypemu.databases.bmrb.nmrstar import PyNMRStarTargetParser


def prepare_bmrb_directory(
    bmrb_ids: list[str],
    output_root: str | Path,
    max_workers: int = 8,
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Download and organize BMRB NMR-STAR files plus parsed bundles.

    Args:
        bmrb_ids: Canonical accession list such as ``["bmr10077"]``.
        output_root: Root directory where the BMRB tree will be created.
        max_workers: Maximum number of concurrent downloads.
        overwrite: Whether to redownload and reparse existing files.
        timeout_seconds: Per-request timeout in seconds.

    Returns:
        Summary dictionary describing the prepared corpus.
    """
    root = Path(output_root)
    raw_star_dir = bmrb_raw_nmrstar_root(root)
    raw_noe_dir = bmrb_raw_noe_root(root)
    parsed_bundle_dir = bmrb_bundle_root(root)
    manifest_dir = bmrb_manifest_root(root)

    for path in [raw_star_dir, raw_noe_dir, parsed_bundle_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    normalized_ids = sorted(
        {bmrb_id.strip().lower() for bmrb_id in bmrb_ids if bmrb_id}
    )
    existing_manifest = _load_existing_manifest(manifest_dir / "index.jsonl")
    download_results = _download_nmrstar_files(
        normalized_ids,
        raw_star_dir,
        overwrite=overwrite,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
    noe_fetch_results = _download_merged_noe_files(
        normalized_ids,
        raw_noe_dir,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )

    parser = PyNMRStarTargetParser()
    manifest_rows: list[dict[str, Any]] = []

    for bmrb_id in normalized_ids:
        star_path = raw_star_dir / f"{bmrb_id}.str"
        bundle_path = parsed_bundle_dir / f"{bmrb_id}.json"
        separate_noe_path = _find_noe_path(raw_noe_dir, bmrb_id)
        previous_row = existing_manifest.get(bmrb_id)

        if not star_path.exists():
            manifest_rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "status": "missing_nmrstar",
                    "nmrstar_path": None,
                    "bundle_path": None,
                    "noe_path": None,
                    "chemical_shift_count": 0,
                    "j_coupling_count": 0,
                    "noe_count": 0,
                }
            )
            continue

        if (
            previous_row is not None
            and bundle_path.exists()
            and not overwrite
            and separate_noe_path is None
        ):
            manifest_rows.append(previous_row)
            continue

        noe_input_path = separate_noe_path or star_path

        try:
            bundle = parser.parse(
                star_path,
                jc_star=star_path,
                noe_star=noe_input_path,
                noe_error_mode="skip",
            )
            bundle.to_json(bundle_path)
            noe_status = str(bundle.metadata.get("noe_status", "not_requested"))
            noe_warning = bundle.metadata.get("noe_parse_error")
            if separate_noe_path is not None:
                noe_source = "separate_restraint_file"
            elif noe_status in {"ok", "empty", "skipped_parse_error"}:
                noe_source = "inline_nmrstar"
            else:
                noe_source = "missing"

            manifest_rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "status": "ok",
                    "nmrstar_path": str(star_path),
                    "bundle_path": str(bundle_path),
                    "noe_path": (
                        None if separate_noe_path is None else str(separate_noe_path)
                    ),
                    "noe_source": noe_source,
                    "noe_status": noe_status,
                    "noe_warning": noe_warning,
                    "chemical_shift_count": len(bundle.chemical_shifts),
                    "j_coupling_count": len(bundle.j_couplings),
                    "noe_count": len(bundle.noe_restraints),
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "status": "parse_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "nmrstar_path": str(star_path),
                    "bundle_path": None,
                    "noe_path": (
                        None if separate_noe_path is None else str(separate_noe_path)
                    ),
                    "noe_status": "parse_error",
                    "chemical_shift_count": 0,
                    "j_coupling_count": 0,
                    "noe_count": 0,
                }
            )

    manifest_jsonl = manifest_dir / "index.jsonl"
    manifest_json = manifest_dir / "index.json"
    summary_json = manifest_dir / "summary.json"
    noe_missing_txt = manifest_dir / "noe_missing.txt"
    manifest_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in manifest_rows)
        + ("\n" if manifest_rows else "")
    )
    manifest_json.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True))

    summary_payload = {
        "entries": len(manifest_rows),
        "ok": sum(row["status"] == "ok" for row in manifest_rows),
        "parse_error": sum(row["status"] == "parse_error" for row in manifest_rows),
        "missing_nmrstar": sum(
            row["status"] == "missing_nmrstar" for row in manifest_rows
        ),
        "chemical_shift_positive": sum(
            row["chemical_shift_count"] > 0
            for row in manifest_rows
            if row["status"] == "ok"
        ),
        "j_coupling_positive": sum(
            row["j_coupling_count"] > 0
            for row in manifest_rows
            if row["status"] == "ok"
        ),
        "noe_positive": sum(
            row["noe_count"] > 0 for row in manifest_rows if row["status"] == "ok"
        ),
        "noe_skipped_parse_error": sum(
            row.get("noe_status") == "skipped_parse_error"
            for row in manifest_rows
            if row["status"] == "ok"
        ),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    noe_missing_txt.write_text(
        "\n".join(
            row["bmrb_id"]
            for row in manifest_rows
            if row["status"] == "ok" and row.get("noe_count", 0) == 0
        )
        + ("\n" if manifest_rows else "")
    )

    _write_bmrb_readme(root)

    ok_count = sum(row["status"] == "ok" for row in manifest_rows)
    return {
        "requested_entries": len(normalized_ids),
        "downloaded_entries": download_results["downloaded"],
        "existing_entries": download_results["existing"],
        "failed_entries": download_results["failed"],
        "noe_downloaded_entries": noe_fetch_results["downloaded"],
        "noe_existing_entries": noe_fetch_results["existing"],
        "noe_available_entries": noe_fetch_results["available"],
        "prepared_entries": ok_count,
        "manifest_path": str(manifest_json),
        "summary_path": str(summary_json),
    }


def _download_nmrstar_files(
    bmrb_ids: list[str],
    raw_star_dir: Path,
    overwrite: bool,
    max_workers: int,
    timeout_seconds: int,
) -> dict[str, int]:
    """Download requested NMR-STAR files in parallel."""
    downloaded = 0
    existing = 0
    failed = 0

    def fetch_one(bmrb_id: str) -> tuple[str, str]:
        path = raw_star_dir / f"{bmrb_id}.str"
        if path.exists() and not overwrite:
            return bmrb_id, "existing"

        numeric_id = bmrb_id.removeprefix("bmr")
        url = f"https://bmrb.io/rest/bmrb/{numeric_id}/nmr-star3"
        try:
            response = requests.get(url, timeout=timeout_seconds)
        except Exception as exc:
            return bmrb_id, f"failed:{type(exc).__name__}"
        if response.status_code != 200:
            return bmrb_id, f"failed:{response.status_code}"
        path.write_text(response.text)
        return bmrb_id, "downloaded"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, bmrb_id): bmrb_id for bmrb_id in bmrb_ids}
        for future in as_completed(futures):
            _, status = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "existing":
                existing += 1
            else:
                failed += 1

    return {"downloaded": downloaded, "existing": existing, "failed": failed}


def _load_existing_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load an existing manifest file when available."""
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["bmrb_id"]: row for row in rows}


def _download_merged_noe_files(
    bmrb_ids: list[str],
    raw_noe_dir: Path,
    overwrite: bool,
    timeout_seconds: int,
) -> dict[str, int]:
    """Download merged restraint STAR files for accessions that have them."""
    merged_links = _fetch_merged_restraint_links(timeout_seconds)
    downloaded = 0
    existing = 0
    available = 0

    for bmrb_id in bmrb_ids:
        relative_path = merged_links.get(bmrb_id)
        if relative_path is None:
            continue

        available += 1
        destination = raw_noe_dir / f"{bmrb_id}.str"
        if destination.exists() and not overwrite:
            existing += 1
            continue

        url = f"{MERGED_BASE_URL}/{relative_path}"
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code != 200:
            continue
        destination.write_text(response.text)
        downloaded += 1

    return {"available": available, "downloaded": downloaded, "existing": existing}


def _fetch_merged_restraint_links(timeout_seconds: int) -> dict[str, str]:
    """Return a mapping from canonical BMRB accession to merged restraint path."""
    html = requests.get(RESTRAINTS_LINKS_URL, timeout=timeout_seconds).text
    return _extract_merged_restraint_links(html)


def _extract_merged_restraint_links(html: str) -> dict[str, str]:
    """Extract merged restraint STAR links from the BMRB link page."""
    mapping: dict[str, str] = {}
    pattern = re.compile(
        r"/ftp/pub/bmrb/nmr_pdb_integrated_data/"
        r"coordinates_restraints_chemshifts/bmrb_plus_pdb/"
        r"(merged_(\d+)_([A-Za-z0-9]+)\.str)"
    )
    for match in pattern.finditer(html):
        relative_path = match.group(1)
        bmrb_id = f"bmr{match.group(2)}"
        mapping.setdefault(bmrb_id, relative_path)
    return mapping


def _find_noe_path(raw_noe_dir: Path, bmrb_id: str) -> Path | None:
    """Return an optional separate NOE restraint file for one accession."""
    candidates = [
        raw_noe_dir / f"{bmrb_id}.str",
        raw_noe_dir / f"{bmrb_id}.star",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_bmrb_readme(root: Path) -> None:
    """Write a small README that documents the organized BMRB layout."""
    readme_path = root / "README.md"
    readme_path.write_text(
        "# BMRB Data Layout\n\n"
        "This directory stores accession-aligned BMRB inputs for AtypEmu.\n\n"
        "## Layout\n\n"
        "- `downloads/nmrstar`: downloaded BMRB NMR-STAR files used for chemical shifts and J-couplings\n"
        "- `downloads/noe`: merged restraint STAR files used as the preferred NOE source when available\n"
        "- `assets/bundles`: parsed `NMRTargetBundle` JSON files\n"
        "- `tables/index.jsonl`: per-accession availability and count summary\n\n"
        "By default, the parser always uses the NMR-STAR file for chemical shifts and J-couplings. "
        "If a merged restraint file exists in `downloads/noe`, it is preferred for NOE extraction. "
        "Otherwise the same NMR-STAR file is checked for inline NOE-style distance-constraint loops.\n"
    )
