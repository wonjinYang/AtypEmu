#!/usr/bin/env python3
"""Combine trainer timing with independent nvidia-smi samples."""

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
    parser.add_argument("core", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    core = json.loads(args.core.read_text())
    rows = list(csv.reader(args.samples.open(newline="")))
    parsed = [
        [float(value.strip()) for value in row]
        for row in rows
        if len(row) == 4 and all(value.strip() for value in row)
    ]
    if len(parsed) < 3:
        raise RuntimeError("nvidia-smi produced fewer than three independent samples")
    util = [row[0] for row in parsed]
    memory = [row[2] for row in parsed]
    power = [row[3] for row in parsed]
    metrics = {
        "total_s": core["elapsed_seconds"],
        "updates_per_s": core["updates_per_second"],
        "gpu_util_mean_pct": statistics.fmean(util),
        "gpu_util_p50_pct": statistics.median(util),
        "gpu_util_p90_pct": percentile(util, 0.90),
        "gpu_active_fraction": sum(value > 0 for value in util) / len(util),
        "gpu_memory_peak_mib": max(memory),
        "gpu_power_mean_w": statistics.fmean(power),
        "gpu_sample_count": len(util),
    }
    payload = {"core": core, "independent_gpu_metrics": metrics}
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.summary.chmod(0o600)
    for key, value in metrics.items():
        print(f"METRIC {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
