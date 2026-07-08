"""CLI for one-shared-q posterior inference over CS predictor sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from atypemu.training.chemical_shift_shared_q import (
    ChemicalShiftSharedQConfig,
    infer_shared_q_posterior_from_paths,
)
from atypemu.training.cs_prediction_parity import ChemicalShiftParityConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--splits", default=None, type=Path)
    parser.add_argument("--output-posterior-predictions", required=True, type=Path)
    parser.add_argument("--output-support-weights", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--pairs-output", default=None, type=Path)
    parser.add_argument("--gate-split", default="val")
    parser.add_argument("--target-ccc", default=0.95, type=float)
    parser.add_argument("--min-family-count", default=2, type=int)
    parser.add_argument("--energy-temperature", default=1.0, type=float)
    parser.add_argument("--energy-scale", default=1.0, type=float)
    parser.add_argument("--prior-log-prob-weight", default=1.0, type=float)
    parser.add_argument("--uncertainty-threshold", default=None, type=float)
    parser.add_argument("--ood-score-threshold", default=None, type=float)
    parser.add_argument("--min-posterior-ess", default=1.0, type=float)
    parser.add_argument("--min-posterior-entropy", default=0.0, type=float)
    parser.add_argument("--max-posterior-top-mass", default=1.0, type=float)
    parser.add_argument("--min-valid-mass", default=1.0, type=float)
    parser.add_argument(
        "--posterior-solver-kind",
        default="energy_softmax_ccc_refined",
        help="Use energy_softmax to disable global CCC refinement.",
    )
    parser.add_argument("--ccc-refinement-steps", default=64, type=int)
    parser.add_argument("--ccc-refinement-learning-rate", default=5.0e-2, type=float)
    parser.add_argument("--ccc-refinement-residual-weight", default=1.0, type=float)
    parser.add_argument("--ccc-refinement-energy-weight", default=5.0e-2, type=float)
    parser.add_argument("--ccc-refinement-kl-weight", default=1.0e-2, type=float)
    parser.add_argument("--ccc-refinement-floor", default=0.95, type=float)
    parser.add_argument("--ccc-refinement-floor-weight", default=2.0, type=float)
    parser.add_argument("--ccc-refinement-macro-floor", default=0.95, type=float)
    parser.add_argument(
        "--ccc-refinement-macro-floor-weight",
        default=2.0,
        type=float,
    )
    parser.add_argument("--ccc-refinement-macro-reward-weight", default=0.25, type=float)
    parser.add_argument(
        "--ccc-refinement-posterior-health-weight",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--ccc-refinement-floor-aggregation",
        default="active_set",
        choices=["active_set", "mean"],
        help="Use active_set to optimize the worst family CCC floor violation.",
    )
    parser.add_argument("--ccc-refinement-min-family-points", default=2, type=int)
    parser.add_argument("--ccc-coordinate-refinement-passes", default=2, type=int)
    parser.add_argument("--ccc-coordinate-candidate-limit", default=16, type=int)
    parser.add_argument(
        "--disable-ccc-coordinate-target-fit-candidates",
        action="store_true",
        help="Disable per-entity target-fit simplex candidates for ablation.",
    )
    parser.add_argument(
        "--allow-missing-support-id",
        action="store_true",
        help="Treat all predictions as one support; diagnostic only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Infer one shared q, write posterior means, and run parity metrics."""

    args = build_parser().parse_args(argv)
    posterior_config = ChemicalShiftSharedQConfig(
        energy_temperature=float(args.energy_temperature),
        energy_scale=float(args.energy_scale),
        prior_log_prob_weight=float(args.prior_log_prob_weight),
        uncertainty_threshold=args.uncertainty_threshold,
        ood_score_threshold=args.ood_score_threshold,
        min_posterior_ess=float(args.min_posterior_ess),
        min_posterior_entropy=float(args.min_posterior_entropy),
        max_posterior_top_mass=float(args.max_posterior_top_mass),
        min_valid_mass=float(args.min_valid_mass),
        require_support_id=not bool(args.allow_missing_support_id),
        posterior_solver_kind=str(args.posterior_solver_kind),
        ccc_refinement_steps=int(args.ccc_refinement_steps),
        ccc_refinement_learning_rate=float(args.ccc_refinement_learning_rate),
        ccc_refinement_residual_weight=float(args.ccc_refinement_residual_weight),
        ccc_refinement_energy_weight=float(args.ccc_refinement_energy_weight),
        ccc_refinement_kl_weight=float(args.ccc_refinement_kl_weight),
        ccc_refinement_floor=float(args.ccc_refinement_floor),
        ccc_refinement_floor_weight=float(args.ccc_refinement_floor_weight),
        ccc_refinement_macro_floor=float(args.ccc_refinement_macro_floor),
        ccc_refinement_macro_floor_weight=float(
            args.ccc_refinement_macro_floor_weight
        ),
        ccc_refinement_macro_reward_weight=float(
            args.ccc_refinement_macro_reward_weight
        ),
        ccc_refinement_posterior_health_weight=float(
            args.ccc_refinement_posterior_health_weight
        ),
        ccc_refinement_floor_aggregation=str(args.ccc_refinement_floor_aggregation),
        ccc_refinement_min_family_points=int(args.ccc_refinement_min_family_points),
        ccc_coordinate_refinement_passes=int(args.ccc_coordinate_refinement_passes),
        ccc_coordinate_candidate_limit=int(args.ccc_coordinate_candidate_limit),
        ccc_coordinate_target_fit_candidates=not bool(
            args.disable_ccc_coordinate_target_fit_candidates
        ),
    )
    parity_config = ChemicalShiftParityConfig(
        target_ccc=float(args.target_ccc),
        min_family_count=int(args.min_family_count),
        gate_split=str(args.gate_split),
        prediction_label=posterior_config.posterior_label,
        require_uncertainty=True,
        require_ood=True,
        uncertainty_threshold=args.uncertainty_threshold,
        ood_score_threshold=args.ood_score_threshold,
        max_high_uncertainty_positive_reward_fraction=0.0,
    )
    summary = infer_shared_q_posterior_from_paths(
        predictions_path=args.predictions,
        targets_path=args.targets,
        splits_path=args.splits,
        posterior_predictions_output=args.output_posterior_predictions,
        support_weights_output=args.output_support_weights,
        summary_output=args.output_summary,
        metrics_output=args.metrics_output,
        pairs_output=args.pairs_output,
        posterior_config=posterior_config,
        parity_config=parity_config,
    )
    print(
        "chemical_shift_shared_q_posterior "
        f"decision={summary['decision']} "
        f"support_count={summary['support_count']} "
        f"posterior={args.output_posterior_predictions}"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
