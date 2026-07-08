"""CLI for PED generation-method classification."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.ped.classify import classify_ped_entries


def build_parser() -> argparse.ArgumentParser:
    """Build the PED classification CLI parser."""
    parser = argparse.ArgumentParser(
        description="Classify PED entries by ensemble-generation strategy."
    )
    parser.add_argument(
        "--ped-root",
        default="data/ped",
        help="Source-owned PED root containing tables/, assets/, and datasets/.",
    )
    parser.add_argument(
        "--workspace",
        choices=["benchmark", "catalog"],
        default="benchmark",
        help="PED workspace to classify.",
    )
    return parser


def main() -> None:
    """Run the PED classification CLI."""
    args = build_parser().parse_args()
    summary = classify_ped_entries(
        ped_root=args.ped_root,
        workspace=args.workspace,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
