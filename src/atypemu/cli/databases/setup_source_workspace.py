"""CLI for bootstrapping source-level database roots under ``data/``."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.common import setup_source_workspace


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create source-level database roots and meta bootstrap files."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Top-level AtypEmu data root to create or update.",
    )
    parser.add_argument(
        "--capture-source-pages",
        action="store_true",
        help="Download lightweight source landing pages into source _debug trees.",
    )
    return parser


def main() -> None:
    """Run the source-workspace setup CLI."""
    args = build_parser().parse_args()
    summary = setup_source_workspace(
        data_root=args.data_root,
        capture_source_pages=args.capture_source_pages,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
