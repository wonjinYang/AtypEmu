"""CLI for exporting source-owned BMRB dataframe bundles."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.bmrb import export_bmrb_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Export source-owned BMRB Parquet datasets.",
    )
    parser.add_argument(
        "--bmrb-root",
        required=True,
        help="BMRB source root such as data/bmrb.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet files.",
    )
    return parser


def main() -> None:
    """Run the BMRB dataframe export CLI."""
    args = build_parser().parse_args()
    summary = export_bmrb_dataframes(
        bmrb_root=args.bmrb_root,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
