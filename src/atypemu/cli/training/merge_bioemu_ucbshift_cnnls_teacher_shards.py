"""CLI for merging sharded BioEmu UCBShift2.0/CNNLS teacher artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atypemu.training.bioemu_ucbshift_cnnls_teacher import (
    merge_bioemu_ucbshift_cnnls_teacher_shards,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge shard-local BioEmu UCBShift2.0/CNNLS teacher outputs into "
            "the canonical reports/arrays teacher parquet files."
        ),
    )
    parser.add_argument(
        "--shard-root",
        default=None,
        help="Directory containing shard_XX_of_YY report roots.",
    )
    parser.add_argument(
        "--shard-output-dir",
        action="append",
        default=None,
        help="One shard report root. Repeat for multiple shards.",
    )
    parser.add_argument("--output-dir", required=True, help="Canonical report root.")
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = merge_bioemu_ucbshift_cnnls_teacher_shards(
        shard_output_dirs=[Path(path) for path in args.shard_output_dir or []],
        shard_root=args.shard_root,
        output_dir=args.output_dir,
        require_nonempty=not bool(args.allow_empty),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") == "failed" and not args.allow_empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
