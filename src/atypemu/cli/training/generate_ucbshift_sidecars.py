"""CLI for UCBShift2.0 sidecar generation."""

from __future__ import annotations

import argparse
import json

from atypemu.training.config import UCBShiftGenerationConfig
from atypemu.training.ucbshift import generate_ucbshift_sidecars


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate UCBShift2.0 sidecars and baseline tables for AtypEmu.",
    )
    parser.add_argument("--data-root", required=True, help="Repository data root.")
    parser.add_argument(
        "--integrated-root",
        required=True,
        help="Integrated workspace root such as data/integrated.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional UCBShift2.0 generation config JSON.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to process. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Candidate source to process. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--bmrb-id",
        action="append",
        default=None,
        help="BMRB accession to process. Repeat for multiple accessions.",
    )
    parser.add_argument(
        "--refresh-outputs",
        action="store_true",
        help="Regenerate existing UCBShift sidecars.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of entities to process.",
    )
    parser.add_argument(
        "--max-candidates-per-example",
        type=int,
        default=None,
        help="Optional cap on PDB candidates per entity and source.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["full", "shiftx_only", "shifty_only"],
        default=None,
        help="Optional UCBShift2.0 execution mode override.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Number of candidate-level UCBShift subprocesses to run in parallel.",
    )
    parser.add_argument(
        "--ucbshift-worker-count",
        type=int,
        default=None,
        help="Internal --worker value passed to each CSpred subprocess.",
    )
    parser.add_argument(
        "--subprocess-timeout-seconds",
        type=float,
        default=None,
        help="Wall-time timeout for each CSpred subprocess; <=0 disables it.",
    )
    parser.add_argument(
        "--sidecar-lock-timeout-seconds",
        type=float,
        default=None,
        help="Maximum time to wait for a sidecar lock before marking it failed.",
    )
    parser.add_argument(
        "--sidecar-stale-lock-seconds",
        type=float,
        default=None,
        help="Age after which an orphaned sidecar lock can be reclaimed.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total number of disjoint accession shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index to process.",
    )
    return parser


def main() -> None:
    """Run the UCBShift2.0 sidecar generation CLI."""
    args = build_parser().parse_args()
    config = (
        UCBShiftGenerationConfig.from_json(args.config_json)
        if args.config_json
        else UCBShiftGenerationConfig()
    )
    if args.split:
        config.selected_splits = list(args.split)
    if args.source:
        config.selected_sources = list(args.source)
    if args.bmrb_id:
        config.selected_bmrb_ids = list(args.bmrb_id)
    if args.refresh_outputs:
        config.refresh_outputs = True
    if args.max_examples is not None:
        config.max_examples = args.max_examples
    if args.max_candidates_per_example is not None:
        config.max_candidates_per_example = args.max_candidates_per_example
    if args.prediction_mode is not None:
        config.prediction_mode = args.prediction_mode
    if args.worker_count is not None:
        config.worker_count = args.worker_count
    if args.ucbshift_worker_count is not None:
        config.ucbshift_worker_count = args.ucbshift_worker_count
    if args.subprocess_timeout_seconds is not None:
        config.subprocess_timeout_seconds = args.subprocess_timeout_seconds
    if args.sidecar_lock_timeout_seconds is not None:
        config.sidecar_lock_timeout_seconds = args.sidecar_lock_timeout_seconds
    if args.sidecar_stale_lock_seconds is not None:
        config.sidecar_stale_lock_seconds = args.sidecar_stale_lock_seconds
    if args.shard_count is not None:
        config.shard_count = args.shard_count
    if args.shard_index is not None:
        config.shard_index = args.shard_index

    summary = generate_ucbshift_sidecars(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") in {"failed_runtime", "failed_generation"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
