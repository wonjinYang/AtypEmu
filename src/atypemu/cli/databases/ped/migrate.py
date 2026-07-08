"""CLI for migrating legacy PED data into the source-owned layout."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.ped.layout import migrate_legacy_ped_source_root


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy PED roots into the source-owned PED layout."
    )
    parser.add_argument(
        "--ped-root",
        required=True,
        help="PED root such as data/ped.",
    )
    return parser


def main() -> None:
    """Run the PED layout-migration CLI."""
    args = build_parser().parse_args()
    summary = migrate_legacy_ped_source_root(ped_root=args.ped_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
