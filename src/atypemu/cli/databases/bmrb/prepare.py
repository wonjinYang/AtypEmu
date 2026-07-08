"""CLI for creating the organized BMRB directory used by AtypEmu."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atypemu.databases.bmrb import prepare_bmrb_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download and parse BMRB NMR-STAR files for AtypEmu."
    )
    parser.add_argument(
        "--bmrb-id-file",
        required=True,
        help="Text file with one canonical BMRB accession per line.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for the organized BMRB tree.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and reparse existing files.",
    )
    return parser


def main() -> None:
    """Run the BMRB preparation CLI."""
    args = build_parser().parse_args()
    bmrb_ids = [
        line.strip()
        for line in Path(args.bmrb_id_file).read_text().splitlines()
        if line.strip()
    ]
    summary = prepare_bmrb_directory(
        bmrb_ids=bmrb_ids,
        output_root=args.output_root,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
