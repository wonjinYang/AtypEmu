#!/usr/bin/env python3
"""Summarize two-worker aggregate throughput and independent GPU samples."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("cores", nargs=2, type=Path)
    args = parser.parse_args()
    cores = [json.loads(path.read_text()) for path in args.cores]
    rows = list(csv.reader(args.samples.open(newline="")))
    samples = [
        [float(value.strip()) for value in row]
        for row in rows
        if len(row) == 4 and all(value.strip() for value in row)
    ]
    if len(samples) < 3:
        raise RuntimeError("nvidia-smi produced fewer than three samples")
    util = [row[0] for row in samples]
    memory = [row[2] for row in samples]
    power = [row[3] for row in samples]
    updates = sum(int(core["entity_updates"]) for core in cores)
    metrics = {
        "aggregate_updates_per_s": updates / args.wall_seconds,
        "aggregate_wall_s": args.wall_seconds,
        "gpu_util_mean_pct": statistics.fmean(util),
        "gpu_util_p50_pct": statistics.median(util),
        "gpu_util_p90_pct": percentile(util, 0.90),
        "gpu_memory_peak_mib": max(memory),
        "gpu_power_mean_w": statistics.fmean(power),
        "gpu_sample_count": len(samples),
    }
    payload = {"mode": args.mode, "cores": cores, "metrics": metrics}
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.summary.chmod(0o600)
    for key, value in metrics.items():
        print(f"METRIC {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
