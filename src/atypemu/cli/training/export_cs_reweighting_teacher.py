"""CLI for exporting source-aware CS-reweighting teachers."""

from __future__ import annotations

import argparse
import json

from atypemu.training.cs_reweighting_teacher_export import (
    export_cs_reweighting_teacher_for_bioemu,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Export materialized AF/BioEmu/CALVADOS CS-reweighting teachers "
            "as BioEmu latent-training prediction and guard parquet files."
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--integrated-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-namespace", default=None)
    parser.add_argument("--bioemu-only-namespace", default=None)
    parser.add_argument("--teacher-policy", default="multisource_cs_reweighting")
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--min-teacher-ess", type=float, default=20.0)
    parser.add_argument("--non-bioemu-source-mass-threshold", type=float, default=0.90)
    return parser


def main() -> None:
    """Run the export."""

    args = build_parser().parse_args()
    summary = export_cs_reweighting_teacher_for_bioemu(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        output_dir=args.output_dir,
        teacher_namespace=args.teacher_namespace,
        bioemu_only_namespace=args.bioemu_only_namespace,
        selected_splits=args.split,
        teacher_policy=args.teacher_policy,
        min_teacher_ess=args.min_teacher_ess,
        non_bioemu_source_mass_threshold=args.non_bioemu_source_mass_threshold,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
