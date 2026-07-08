"""CLI for archive-aware per-BMRB staging from the canonical data layout."""

from __future__ import annotations

import argparse

from atypemu.structures import stage_bmrb_accession


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Stage one BMRB accession from AtypEmu/data."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing AF, CALVADOS, and BioEmu.",
    )
    parser.add_argument(
        "--bmrb-id",
        required=True,
        help="BMRB accession to stage, for example bmr10077.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root where the staged accession directory will be created.",
    )
    parser.add_argument(
        "--bioemu-mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="How to materialize the BioEmu subtree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing staged accession directory.",
    )
    return parser


def main() -> None:
    """Run the staging CLI."""
    args = build_parser().parse_args()
    stage_bmrb_accession(
        data_root=args.data_root,
        bmrb_id=args.bmrb_id,
        output_dir=args.output_dir,
        bioemu_mode=args.bioemu_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
