"""CLI for exporting cross-source metadata Parquet tables."""

from __future__ import annotations

import argparse
import json

from atypemu.integrated import export_meta_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Export source-agnostic metadata Parquet tables.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="AtypEmu data root such as data.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet files.",
    )
    return parser


def main() -> None:
    """Run the metadata export CLI."""
    args = build_parser().parse_args()
    summary = export_meta_dataframes(
        data_root=args.data_root,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
