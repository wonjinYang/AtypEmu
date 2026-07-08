"""CLI for exporting trainer-facing integrated dataframes."""

from __future__ import annotations

import argparse
import json

from atypemu.integrated import export_training_dataframes


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Export trainer-facing integrated dataframe bundles.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="AtypEmu data root such as data.",
    )
    parser.add_argument(
        "--integrated-root",
        required=True,
        help="Integrated workspace root such as data/integrated.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing Parquet outputs.",
    )
    return parser


def main() -> None:
    """Run the integrated training-data export CLI."""
    args = build_parser().parse_args()
    summary = export_training_dataframes(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
