"""Merge sharded UCBShift2.0 sidecar summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def merge_ucbshift_sidecar_shards(
    integrated_root: str | Path,
    shard_count: int,
) -> dict[str, Any]:
    """Merge shard summaries into the canonical UCBShift summary.

    Args:
        integrated_root: Integrated workspace root.
        shard_count: Number of expected shard summary files.

    Returns:
        Canonical merged summary payload.
    """

    integrated_root_path = Path(integrated_root)
    ucbshift_root = integrated_root_path / "ucbshift"
    summaries = []
    for shard_index in range(shard_count):
        path = (
            ucbshift_root
            / f"sidecar_generation_summary.shard{shard_index:02d}-of-{shard_count:02d}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing UCBShift shard summary: {path}")
        summaries.append(json.loads(path.read_text()))

    baseline_path = integrated_root_path / "baselines" / "ucbshift2_holdout.csv"
    reference_path = (
        integrated_root_path / "baselines" / "ucbshift2_reference_corpus.csv"
    )
    _merge_csvs(
        [summary.get("baseline_output_path") for summary in summaries],
        baseline_path,
    )
    _merge_csvs(
        [summary.get("reference_corpus_output_path") for summary in summaries],
        reference_path,
    )

    results = []
    for summary in summaries:
        results.extend(summary.get("results", []))

    statuses = {str(summary.get("status")) for summary in summaries}
    if statuses <= {"ok"}:
        status = "ok"
        message = "All UCBShift sidecar shards completed successfully."
    elif "failed_runtime" in statuses or "failed_generation" in statuses:
        status = "failed_generation"
        message = "One or more UCBShift sidecar shards failed."
    else:
        status = "partial"
        message = "UCBShift sidecar shards completed with partial failures."

    payload = {
        "status": status,
        "message": message,
        "shard_count": shard_count,
        "merged_shards": shard_count,
        "total_selected_examples": sum_int(summaries, "total_selected_examples"),
        "requested_examples": sum_int(summaries, "requested_examples"),
        "requested_candidates": sum_int(summaries, "requested_candidates"),
        "written_candidates": sum_int(summaries, "written_candidates"),
        "skipped_existing_candidates": sum_int(
            summaries, "skipped_existing_candidates"
        ),
        "failed_candidates": sum_int(summaries, "failed_candidates"),
        "requested_candidate_examples": sum_int(
            summaries, "requested_candidate_examples"
        ),
        "written_candidate_examples": sum_int(summaries, "written_candidate_examples"),
        "baseline_examples": sum_int(summaries, "baseline_examples"),
        "failed_examples": sum_int(summaries, "failed_examples"),
        "partial_examples": sum_int(summaries, "partial_examples"),
        "baseline_rows": _row_count(baseline_path),
        "reference_examples": _row_count(reference_path),
        "worker_count": max_int(summaries, "worker_count"),
        "ucbshift_worker_count": max_int(summaries, "ucbshift_worker_count"),
        "baseline_output_path": str(baseline_path),
        "reference_corpus_output_path": str(reference_path),
        "dependency_status": {
            "status": status,
            "shard_summaries": [
                str(
                    ucbshift_root
                    / f"sidecar_generation_summary.shard{index:02d}-of-{shard_count:02d}.json"
                )
                for index in range(shard_count)
            ],
        },
        "results": results,
    }
    output_path = ucbshift_root / "sidecar_generation_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"summary_path": str(output_path), **payload}


def sum_int(summaries: list[dict[str, Any]], key: str) -> int:
    """Sum one integer summary key."""

    return int(sum(int(summary.get(key, 0) or 0) for summary in summaries))


def max_int(summaries: list[dict[str, Any]], key: str) -> int:
    """Return the maximum integer summary key."""

    values = [int(summary.get(key, 0) or 0) for summary in summaries]
    return max(values) if values else 0


def _merge_csvs(paths: list[Any], output_path: Path) -> None:
    """Merge optional shard CSV files into one canonical CSV."""

    frames = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    pd.concat(frames, ignore_index=True).drop_duplicates().to_csv(
        output_path,
        index=False,
    )


def _row_count(path: Path) -> int:
    """Return CSV row count when the path exists."""

    if not path.exists() or path.stat().st_size == 0:
        return 0
    return int(len(pd.read_csv(path)))
