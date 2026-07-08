"""CLI for crawling FuzDB API payloads into source-owned tables."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.fuzdb import crawl_fuzdb


def build_parser() -> argparse.ArgumentParser:
    """Build the FuzDB crawl CLI parser."""
    parser = argparse.ArgumentParser(
        description="Crawl FuzDB API payloads into source-owned tables."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Top-level AtypEmu data root containing the fuzdb/ source root.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite cached downloads, HTML snapshots, and curated tables.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional entry cap for smoke runs and debugging.",
    )
    return parser


def main() -> None:
    """Run the FuzDB crawl CLI."""
    args = build_parser().parse_args()
    summary = crawl_fuzdb(
        data_root=args.data_root,
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
        max_entries=args.max_entries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
