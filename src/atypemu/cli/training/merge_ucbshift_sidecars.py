"""Merge sharded UCBShift2.0 sidecar summaries."""

from __future__ import annotations

import argparse
import json

from atypemu.training.ucbshift_merge import merge_ucbshift_sidecar_shards


def main() -> None:
    """Run the UCBShift shard merge CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-root", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    summary = merge_ucbshift_sidecar_shards(
        integrated_root=args.integrated_root,
        shard_count=args.shard_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") in {"failed_runtime", "failed_generation"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
