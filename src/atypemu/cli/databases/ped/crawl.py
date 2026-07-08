"""CLI for the PED-only crawl pipeline."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.ped.crawl import crawl_ped


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for PED crawling."""
    parser = argparse.ArgumentParser(
        description="Crawl PED entry metadata, model assets, and PED tables."
    )
    parser.add_argument(
        "--ped-root",
        required=True,
        help="Source-owned PED root such as data/ped.",
    )
    parser.add_argument(
        "--workspace",
        choices=["benchmark", "catalog"],
        default="benchmark",
        help="PED workspace to write within the source-owned PED root.",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Crawl only the curated PED benchmark seeds.",
    )
    parser.add_argument(
        "--ped-id",
        action="append",
        default=None,
        help="Optional explicit live PED ID. May be repeated.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace existing PED crawl outputs for the targeted entries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request and page-load timeout in seconds.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional hard limit on the number of PED entries to crawl.",
    )
    parser.add_argument(
        "--save-network-payloads",
        dest="save_network_payloads",
        action="store_true",
        default=True,
        help="Persist captured JSON payloads from rendered PED pages.",
    )
    parser.add_argument(
        "--no-save-network-payloads",
        dest="save_network_payloads",
        action="store_false",
        help="Skip saving captured JSON network payload snapshots.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip model-asset downloads and collect entry metadata only.",
    )
    return parser


def main() -> None:
    """Run the PED-only crawler."""
    args = build_parser().parse_args()
    summary = crawl_ped(
        ped_root=args.ped_root,
        workspace=args.workspace,
        seed_only=args.seed_only,
        ped_ids=args.ped_id,
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
        max_entries=args.max_entries,
        save_network_payloads=args.save_network_payloads,
        metadata_only=args.metadata_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
