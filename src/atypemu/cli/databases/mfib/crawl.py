"""CLI for crawling MFIB bundle downloads into source-owned tables."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.mfib import crawl_mfib


def build_parser() -> argparse.ArgumentParser:
    """Build the MFIB crawl CLI parser."""
    parser = argparse.ArgumentParser(
        description="Crawl MFIB download bundles into source-owned tables."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Top-level AtypEmu data root containing the mfib/ source root.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite cached downloads, assets, and curated tables.",
    )
    parser.add_argument(
        "--no-extract-cif",
        action="store_true",
        help="Skip CIF archive download and deterministic structure features.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return parser


def main() -> None:
    """Run the MFIB crawl CLI."""
    args = build_parser().parse_args()
    summary = crawl_mfib(
        data_root=args.data_root,
        refresh=args.refresh,
        extract_cif=not args.no_extract_cif,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
