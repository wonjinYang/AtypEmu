"""CLI for exporting PED trainer-ready Parquet bundles."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.ped.export import export_ped_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the PED dataframe-export CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export curated PED TSV tables into Parquet datasets."
    )
    parser.add_argument(
        "--ped-root",
        default="data/ped",
        help="Source-owned PED root containing tables/, datasets/, and assets/.",
    )
    parser.add_argument(
        "--workspace",
        choices=["benchmark", "catalog", "both"],
        default="both",
        help="PED workspace selection to export.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet tables.",
    )
    parser.add_argument(
        "--no-model-assets",
        action="store_true",
        help="Skip benchmark model-level export.",
    )
    parser.add_argument(
        "--no-wide-view",
        action="store_true",
        help="Skip the derived ped_entry_training_view parquet.",
    )
    return parser


def main() -> None:
    """Run the PED dataframe-export CLI."""
    args = build_parser().parse_args()
    summary = export_ped_dataframes(
        ped_root=args.ped_root,
        workspace=args.workspace,
        refresh=args.refresh,
        include_model_assets=not args.no_model_assets,
        emit_wide_view=not args.no_wide_view,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
