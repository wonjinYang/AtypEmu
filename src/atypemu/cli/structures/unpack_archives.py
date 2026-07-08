"""CLI for unpacking archived AF and CALVADOS candidate chunks."""

from __future__ import annotations

import argparse
import json

from atypemu.structures import unpack_zip_archives


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Unpack zip archives into a stable working directory."
    )
    parser.add_argument(
        "--source-dir", required=True, help="Directory with zip chunks."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where extracted files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    return parser


def main() -> None:
    """Run the archive unpacking CLI."""
    args = build_parser().parse_args()
    summary = unpack_zip_archives(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
