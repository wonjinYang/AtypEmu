"""CLI for UCBShift-parity chemical-shift predictor evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from atypemu.training.cs_prediction_parity import (
    ChemicalShiftParityConfig,
    evaluate_chemical_shift_prediction_parity,
    load_table,
    write_parity_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chemical-shift predictions on the same BMRB rows, split "
            "labels, atom-family mapping, and CCC gate used for UCBShift-parity "
            "benchmarks."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction table with entity_uid and predicted chemical shifts.",
    )
    parser.add_argument(
        "--targets",
        default=Path("data/bmrb/datasets/default/targets.parquet"),
        type=Path,
        help="Target table; defaults to the repository BMRB target parquet.",
    )
    parser.add_argument(
        "--splits",
        default=Path("data/integrated/splits/default_split.parquet"),
        type=Path,
        help="Optional split table with entity_uid and split columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        type=Path,
        help="Optional metric-row table output path.",
    )
    parser.add_argument(
        "--pairs-output",
        default=None,
        type=Path,
        help="Optional joined prediction-target row output path.",
    )
    parser.add_argument("--target-ccc", default=0.95, type=float)
    parser.add_argument("--min-family-count", default=2, type=int)
    parser.add_argument("--gate-split", default="val")
    parser.add_argument(
        "--join-keys",
        default=None,
        help="Comma-separated join keys, e.g. entity_uid,target_id.",
    )
    parser.add_argument(
        "--prediction-value-column",
        default=None,
        help="Override the prediction value column name.",
    )
    parser.add_argument(
        "--target-value-column",
        default=None,
        help="Override the target value column name.",
    )
    parser.add_argument(
        "--prediction-label",
        default="chemical_shift_predictor",
        help="Label stored in the summary JSON.",
    )
    parser.add_argument(
        "--target-label",
        default="bmrb_targets",
        help="Target dataset label stored in the summary JSON.",
    )
    parser.add_argument(
        "--require-uncertainty",
        action="store_true",
        help="Fail unless every gated prediction row has uncertainty metadata.",
    )
    parser.add_argument(
        "--require-ood",
        action="store_true",
        help="Fail unless gated rows include OOD score or OOD flag metadata.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        default=None,
        type=float,
        help="Rows above this prediction uncertainty are treated as high uncertainty.",
    )
    parser.add_argument(
        "--ood-score-threshold",
        default=None,
        type=float,
        help="Rows above this OOD score are treated as generated-support OOD.",
    )
    parser.add_argument(
        "--max-ood-fraction",
        default=0.0,
        type=float,
        help="Maximum allowed OOD row fraction on the gated split.",
    )
    parser.add_argument(
        "--max-high-uncertainty-fraction",
        default=0.0,
        type=float,
        help="Maximum allowed high-uncertainty row fraction on the gated split.",
    )
    parser.add_argument(
        "--positive-reward-threshold",
        default=0.0,
        type=float,
        help="Reward values above this threshold are considered positive reward.",
    )
    parser.add_argument(
        "--max-high-uncertainty-positive-reward-fraction",
        default=0.0,
        type=float,
        help=(
            "Maximum fraction of positive-reward rows that may also be "
            "high-uncertainty."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the parity benchmark CLI."""

    args = build_parser().parse_args(argv)
    predictions = load_table(args.predictions)
    targets = load_table(args.targets)
    splits = load_table(args.splits) if args.splits and args.splits.exists() else None
    join_keys = (
        tuple(item.strip() for item in str(args.join_keys).split(",") if item.strip())
        if args.join_keys
        else None
    )
    config = ChemicalShiftParityConfig(
        target_ccc=float(args.target_ccc),
        min_family_count=int(args.min_family_count),
        gate_split=str(args.gate_split),
        join_keys=join_keys,
        prediction_label=str(args.prediction_label),
        target_label=str(args.target_label),
        require_uncertainty=bool(args.require_uncertainty),
        require_ood=bool(args.require_ood),
        uncertainty_threshold=args.uncertainty_threshold,
        ood_score_threshold=args.ood_score_threshold,
        max_ood_fraction=float(args.max_ood_fraction),
        max_high_uncertainty_fraction=float(args.max_high_uncertainty_fraction),
        positive_reward_threshold=float(args.positive_reward_threshold),
        max_high_uncertainty_positive_reward_fraction=float(
            args.max_high_uncertainty_positive_reward_fraction
        ),
    )
    result = evaluate_chemical_shift_prediction_parity(
        predictions=predictions,
        targets=targets,
        splits=splits,
        config=config,
        prediction_value_column=args.prediction_value_column,
        target_value_column=args.target_value_column,
    )
    write_parity_outputs(
        result=result,
        summary_path=args.output,
        metrics_path=args.metrics_output,
        pairs_path=args.pairs_output,
    )
    print(
        "chemical_shift_prediction_parity "
        f"decision={result['summary']['decision']} "
        f"gate_split={result['summary']['gate_split']} "
        f"joined_pair_count={result['summary']['joined_pair_count']} "
        f"output={args.output}"
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
