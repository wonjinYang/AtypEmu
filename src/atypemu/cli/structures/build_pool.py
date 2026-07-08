"""CLI for building candidate-pool manifests."""

from __future__ import annotations

import argparse

from atypemu.structures import build_candidate_pool


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Build an AtypEmu candidate pool.")
    parser.add_argument("--structure-dir", required=True, help="Input PDB directory.")
    parser.add_argument(
        "--source", required=True, help="Source label for all candidates."
    )
    parser.add_argument(
        "--chemical-shift-dir", default=None, help="Optional shift sidecar directory."
    )
    parser.add_argument(
        "--chemical-shift-format",
        default=None,
        help="Optional sidecar format: shiftx2, sparta+, ucbshift, or json.",
    )
    parser.add_argument(
        "--glob-pattern", default="*.pdb", help="Structure glob pattern."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    return parser


def main() -> None:
    """Run the pool builder CLI."""
    args = build_parser().parse_args()
    pool = build_candidate_pool(
        structure_dir=args.structure_dir,
        source=args.source,
        chemical_shift_dir=args.chemical_shift_dir,
        chemical_shift_format=args.chemical_shift_format,
        glob_pattern=args.glob_pattern,
    )
    pool.to_jsonl(args.output)


if __name__ == "__main__":
    main()
