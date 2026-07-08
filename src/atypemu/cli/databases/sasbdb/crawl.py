"""CLI for crawling SASBDB accession metadata and assets."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.sasbdb import crawl_sasbdb


def build_parser() -> argparse.ArgumentParser:
    """Build the SASBDB crawl parser."""
    parser = argparse.ArgumentParser(
        description="Crawl SASBDB protein entries into source-owned tables."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Top-level AtypEmu data root containing the sasbdb/ source root.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite cached downloads and tables.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of accessions per summary-list request.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--no-html-fallback",
        action="store_true",
        help="Disable HTML fetches for still-missing validation fields.",
    )
    parser.add_argument(
        "--code",
        action="append",
        default=[],
        help="Optional accession subset for smoke runs. Repeat as needed.",
    )
    return parser


def main() -> None:
    """Run the SASBDB crawl CLI."""
    args = build_parser().parse_args()
    summary = crawl_sasbdb(
        data_root=args.data_root,
        refresh=args.refresh,
        include_html_fallback=not args.no_html_fallback,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        codes=args.code or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
