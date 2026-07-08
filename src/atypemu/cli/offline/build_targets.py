"""CLI for parsing local NMR targets into one bundle."""

from __future__ import annotations

import argparse

from atypemu.databases.bmrb import PyNMRStarTargetParser


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Build an AtypEmu target bundle.")
    parser.add_argument("--input-star", required=True, help="Primary BMRB STAR file.")
    parser.add_argument("--jc-star", default=None, help="Optional JC STAR file.")
    parser.add_argument("--noe-star", default=None, help="Optional NOE STAR file.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser


def main() -> None:
    """Run the target-bundle builder CLI."""
    args = build_parser().parse_args()
    parser = PyNMRStarTargetParser()
    bundle = parser.parse(args.input_star, jc_star=args.jc_star, noe_star=args.noe_star)
    bundle.to_json(args.output)


if __name__ == "__main__":
    main()
