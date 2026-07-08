"""CLI for exporting SASBDB source-owned Parquet datasets."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.sasbdb import export_sasbdb_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the SASBDB export parser."""
    parser = argparse.ArgumentParser(
        description="Export SASBDB source tables into Parquet datasets."
    )
    parser.add_argument(
        "--sasbdb-root",
        default="data/sasbdb",
        help="Source-owned SASBDB root containing tables/, assets/, and datasets/.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet exports.",
    )
    parser.add_argument(
        "--no-entry-view",
        action="store_true",
        help="Skip the derived entry_view parquet.",
    )
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="Skip parsing and exporting curve profiles.",
    )
    return parser


def main() -> None:
    """Run the SASBDB export CLI."""
    args = build_parser().parse_args()
    summary = export_sasbdb_dataframes(
        sasbdb_root=args.sasbdb_root,
        refresh=args.refresh,
        emit_entry_view=not args.no_entry_view,
        emit_profiles=not args.no_profiles,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
