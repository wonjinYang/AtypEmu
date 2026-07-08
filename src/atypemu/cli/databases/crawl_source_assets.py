"""CLI for crawling curated source assets into source-owned roots."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.common import crawl_source_assets


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download curated source snapshots and metadata bundles."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Top-level AtypEmu data root to update.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Optional source name to crawl. May be repeated.",
    )
    parser.add_argument(
        "--include-large-assets",
        action="store_true",
        help="Download large optional archives such as structure bundles.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace existing downloaded files.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return parser


def main() -> None:
    """Run the source-asset crawl CLI."""
    args = build_parser().parse_args()
    summary = crawl_source_assets(
        data_root=args.data_root,
        sources=args.source,
        include_large_assets=args.include_large_assets,
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
