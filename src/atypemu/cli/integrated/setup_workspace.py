"""CLI for preparing the local AtypEmu integrated workspace."""

from __future__ import annotations

import argparse
import json

from atypemu.integrated import setup_integrated_workspace


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create manifests and splits for the AtypEmu integrated workspace."
    )
    parser.add_argument(
        "--data-root", required=True, help="Root candidate data directory."
    )
    parser.add_argument("--bmrb-root", required=True, help="Organized BMRB directory.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Integrated workspace root to create or update.",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=256,
        help="Default per-source cap written to the workspace config.",
    )
    return parser


def main() -> None:
    """Run the integrated-workspace setup CLI."""
    args = build_parser().parse_args()
    summary = setup_integrated_workspace(
        data_root=args.data_root,
        bmrb_root=args.bmrb_root,
        output_root=args.output_root,
        max_per_source=args.max_per_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
