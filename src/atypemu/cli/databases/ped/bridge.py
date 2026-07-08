"""CLI for building PED-to-BMRB bridge manifests."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.ped.bridge import prepare_ped_bridge


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build PED-to-BMRB bridge manifests for training and validation."
    )
    parser.add_argument(
        "--ped-root",
        required=True,
        help="Source-owned PED root such as data/ped.",
    )
    parser.add_argument(
        "--workspace",
        choices=["benchmark", "catalog"],
        default="catalog",
        help="PED workspace used to build the bridge.",
    )
    parser.add_argument(
        "--training-root",
        required=True,
        help="Integrated cross-source workspace root such as data/integrated.",
    )
    return parser


def main() -> None:
    """Run the PED bridge-preparation CLI."""
    args = build_parser().parse_args()
    summary = prepare_ped_bridge(
        ped_root=args.ped_root,
        training_root=args.training_root,
        workspace=args.workspace,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
