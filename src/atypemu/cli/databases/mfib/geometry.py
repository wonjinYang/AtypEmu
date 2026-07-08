"""CLI for computing derived MFIB heavy-geometry features."""

from __future__ import annotations

import argparse
import json

from atypemu.databases.mfib import compute_mfib_geometry_features


def build_parser() -> argparse.ArgumentParser:
    """Build the MFIB geometry CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute derived interface and contact features for MFIB CIF assets."
    )
    parser.add_argument(
        "--mfib-root",
        default="data/mfib",
        help="Source-owned MFIB root containing assets/, tables/, and datasets/.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing geometry tables and heavy assets.",
    )
    parser.add_argument(
        "--contact-level",
        default="residue",
        help="Contact granularity. MFIB v1 supports only 'residue'.",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=5.0,
        help="Residue-contact distance cutoff in angstrom.",
    )
    parser.add_argument(
        "--no-contact-map",
        action="store_true",
        help="Skip contact-map NPZ serialization.",
    )
    parser.add_argument(
        "--compute-interface-area",
        action="store_true",
        help="Enable FreeSASA-based interface and buried surface area calculation.",
    )
    return parser


def main() -> None:
    """Run the MFIB heavy-geometry CLI."""
    args = build_parser().parse_args()
    summary = compute_mfib_geometry_features(
        mfib_root=args.mfib_root,
        refresh=args.refresh,
        contact_level=args.contact_level,
        distance_cutoff=args.distance_cutoff,
        compute_interface_area=args.compute_interface_area,
        compute_contact_map=not args.no_contact_map,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
