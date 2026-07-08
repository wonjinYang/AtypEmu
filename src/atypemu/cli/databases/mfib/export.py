"""CLI for exporting MFIB source-owned Parquet datasets."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.mfib import export_mfib_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the MFIB dataframe export parser."""
    parser = argparse.ArgumentParser(
        description="Export curated MFIB tables into Parquet datasets."
    )
    parser.add_argument(
        "--mfib-root",
        default="data/mfib",
        help="Source-owned MFIB root containing tables/ and datasets/.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet tables.",
    )
    parser.add_argument(
        "--no-entry-view",
        action="store_true",
        help="Skip the compact entry_view parquet.",
    )
    return parser


def main() -> None:
    """Run the MFIB dataframe export CLI."""
    args = build_parser().parse_args()
    summary = export_mfib_dataframes(
        mfib_root=args.mfib_root,
        refresh=args.refresh,
        emit_entry_view=not args.no_entry_view,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
