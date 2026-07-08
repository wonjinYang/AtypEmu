"""CLI for BioEmu UCBShift2.0/CS-reweighting teacher generation."""

from __future__ import annotations

import argparse
import json

from atypemu.training.bioemu_ucbshift_cnnls_teacher import (
    run_bioemu_ucbshift_cnnls_teacher,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BioEmu conformer population teachers from UCBShift2.0 sidecars "
            "using CS-reweighting paper-style offset correction and MaxEnt/BFGS."
        ),
    )
    parser.add_argument("--data-root", required=True, help="Repository data root.")
    parser.add_argument("--run-dir", required=True, help="BioEmu run directory.")
    parser.add_argument(
        "--shift-manifest",
        required=True,
        help=(
            "CSV/TSV/parquet with entity_uid, chemical_shift_path, and optional "
            "candidate_id/sample_index/prior_weight/bioemu_log_prob columns."
        ),
    )
    parser.add_argument(
        "--target-manifest",
        default=None,
        help="Optional entity_uid,target_bundle_path manifest for standalone runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional report directory. Defaults to <run-dir>/reports.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "cs_reweighting",
            "cs-reweighting",
            "csrw",
            "paper_cs_reweighting",
            "paper-cs-reweighting",
            "cs_reweighting_paper",
            "cs-reweighting-paper",
            "csrw_paper",
            "legacy_cs_reweighting",
            "legacy-cs-reweighting",
            "cs_reweighting_legacy",
            "cs-reweighting-legacy",
            "csrw_legacy",
            "student_bme",
            "student-bme",
            "goodbad_bme",
            "goodbad-bme",
            "good_bad_bme",
            "cnnls",
            "nnls",
            "euclidean",
            "maxent",
        ],
        default="cs_reweighting",
        help="Population-weight fitting method.",
    )
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--default-cs-sigma", type=float, default=1.0)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument(
        "--require-all-sidecars",
        action="store_true",
        help="Skip an entity if any conformer sidecar listed in the manifest is missing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_bioemu_ucbshift_cnnls_teacher(
        data_root=args.data_root,
        run_dir=args.run_dir,
        shift_manifest_path=args.shift_manifest,
        output_dir=args.output_dir,
        target_manifest_path=args.target_manifest,
        method=args.method,
        lambda_reg=args.lambda_reg,
        default_cs_sigma=args.default_cs_sigma,
        max_entities=args.max_entities,
        require_all_sidecars=bool(args.require_all_sidecars),
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    if summary.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
