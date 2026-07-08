"""CLI for exporting curated FuzDB tables into Parquet datasets."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.fuzdb import export_fuzdb_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the FuzDB export CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export curated FuzDB tables into source-owned Parquet datasets."
    )
    parser.add_argument(
        "--fuzdb-root",
        default="data/fuzdb",
        help="Source-owned FuzDB root to export.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet outputs and manifest files.",
    )
    parser.add_argument(
        "--no-entry-view",
        action="store_true",
        help="Skip the trainer-facing entry view export.",
    )
    return parser


def main() -> None:
    """Run the FuzDB export CLI."""
    args = build_parser().parse_args()
    summary = export_fuzdb_dataframes(
        fuzdb_root=args.fuzdb_root,
        refresh=args.refresh,
        emit_entry_view=not args.no_entry_view,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
