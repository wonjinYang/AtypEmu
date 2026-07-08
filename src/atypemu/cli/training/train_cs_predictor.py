"""CLI for training a parity-ready tabular chemical-shift predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from atypemu.training.chemical_shift_predictor import (
    ChemicalShiftPredictorConfig,
    train_predictor_from_paths,
)
from atypemu.training.chemical_shift_shared_q import (
    ChemicalShiftSharedQConfig,
    ChemicalShiftSharedQSweepConfig,
)
from atypemu.training.cs_prediction_parity import ChemicalShiftParityConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--splits", default=None, type=Path)
    parser.add_argument("--features", default=None, type=Path)
    parser.add_argument("--support-features", default=None, type=Path)
    parser.add_argument("--output-predictions", required=True, type=Path)
    parser.add_argument("--output-bundle", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--pairs-output", default=None, type=Path)
    parser.add_argument("--output-support-predictions", default=None, type=Path)
    parser.add_argument(
        "--output-shared-q-posterior-predictions",
        default=None,
        type=Path,
    )
    parser.add_argument("--output-shared-q-support-weights", default=None, type=Path)
    parser.add_argument("--output-shared-q-summary", default=None, type=Path)
    parser.add_argument("--shared-q-metrics-output", default=None, type=Path)
    parser.add_argument("--shared-q-pairs-output", default=None, type=Path)
    parser.add_argument("--shared-q-sweep-report-output", default=None, type=Path)
    parser.add_argument("--gate-split", default="val")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--target-ccc", default=0.95, type=float)
    parser.add_argument("--min-family-count", default=2, type=int)
    parser.add_argument("--ridge-alpha", default=1.0e-6, type=float)
    parser.add_argument("--min-family-rows", default=2, type=int)
    parser.add_argument("--uncertainty-floor", default=1.0e-3, type=float)
    parser.add_argument("--ood-score-threshold", default=1.0, type=float)
    parser.add_argument("--reward-uncertainty-scale", default=4.0, type=float)
    parser.add_argument(
        "--disable-pairwise-interactions",
        action="store_true",
        help="Disable bounded numeric interaction features for ablation.",
    )
    parser.add_argument("--max-pairwise-interactions", default=64, type=int)
    parser.add_argument(
        "--family-shrinkage-strength",
        default=4.0,
        type=float,
        help=(
            "Blend low-row family readouts toward the global readout; "
            "use 0 to disable this ablation."
        ),
    )
    parser.add_argument(
        "--disable-family-affine-calibration",
        action="store_true",
        help="Disable family-wise affine prediction calibration for ablation.",
    )
    parser.add_argument(
        "--family-calibration-strength",
        default=4.0,
        type=float,
        help=(
            "Rows/(rows+strength) shrinkage for family affine calibration; "
            "0 uses the fitted affine calibration directly."
        ),
    )
    parser.add_argument("--shared-q-energy-temperature", default=1.0, type=float)
    parser.add_argument("--shared-q-energy-scale", default=1.0, type=float)
    parser.add_argument("--shared-q-prior-log-prob-weight", default=1.0, type=float)
    parser.add_argument("--shared-q-min-posterior-ess", default=1.0, type=float)
    parser.add_argument("--shared-q-min-posterior-entropy", default=0.0, type=float)
    parser.add_argument("--shared-q-max-posterior-top-mass", default=1.0, type=float)
    parser.add_argument("--shared-q-min-valid-mass", default=1.0, type=float)
    parser.add_argument(
        "--shared-q-posterior-solver-kind",
        default="energy_softmax_ccc_refined",
        help="Use energy_softmax to disable global CCC refinement.",
    )
    parser.add_argument("--shared-q-ccc-refinement-steps", default=64, type=int)
    parser.add_argument(
        "--shared-q-ccc-refinement-learning-rate",
        default=5.0e-2,
        type=float,
    )
    parser.add_argument(
        "--shared-q-ccc-refinement-residual-weight",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--shared-q-ccc-refinement-energy-weight",
        default=5.0e-2,
        type=float,
    )
    parser.add_argument("--shared-q-ccc-refinement-kl-weight", default=1.0e-2, type=float)
    parser.add_argument("--shared-q-ccc-refinement-floor", default=0.95, type=float)
    parser.add_argument("--shared-q-ccc-refinement-floor-weight", default=2.0, type=float)
    parser.add_argument("--shared-q-ccc-refinement-macro-floor", default=0.95, type=float)
    parser.add_argument(
        "--shared-q-ccc-refinement-macro-floor-weight",
        default=2.0,
        type=float,
    )
    parser.add_argument(
        "--shared-q-ccc-refinement-macro-reward-weight",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--shared-q-ccc-refinement-posterior-health-weight",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--shared-q-ccc-refinement-floor-aggregation",
        default="active_set",
        choices=["active_set", "mean"],
        help="Use active_set to optimize the worst family CCC floor violation.",
    )
    parser.add_argument("--shared-q-ccc-refinement-min-family-points", default=2, type=int)
    parser.add_argument("--shared-q-ccc-coordinate-refinement-passes", default=2, type=int)
    parser.add_argument("--shared-q-ccc-coordinate-candidate-limit", default=16, type=int)
    parser.add_argument(
        "--disable-shared-q-ccc-coordinate-target-fit-candidates",
        action="store_true",
        help="Disable per-entity target-fit simplex candidates for ablation.",
    )
    parser.add_argument(
        "--shared-q-sweep",
        action="store_true",
        help="Sweep shared-q solver settings and write the best posterior.",
    )
    parser.add_argument(
        "--shared-q-sweep-energy-temperatures",
        default="0.005,0.02,0.05,0.1,0.2,0.5,1.0",
    )
    parser.add_argument("--shared-q-sweep-floor-weights", default="0,2,5,20")
    parser.add_argument("--shared-q-sweep-macro-floor-weights", default="0,2,5,20")
    parser.add_argument("--shared-q-sweep-macro-reward-weights", default="0,0.25,1")
    parser.add_argument("--shared-q-sweep-posterior-health-weights", default="0,1,5")
    parser.add_argument("--shared-q-sweep-energy-weights", default="0,0.05")
    parser.add_argument("--shared-q-sweep-kl-weights", default="0,0.01")
    parser.add_argument("--shared-q-sweep-residual-weights", default="0,1")
    parser.add_argument("--shared-q-sweep-max-configs", default=256, type=int)
    parser.add_argument(
        "--shared-q-sweep-selection-metric",
        default="family_gate_then_macro_ccc",
        choices=[
            "family_gate_then_macro_ccc",
            "family_min_ccc_then_macro",
            "objective_then_macro_ccc",
        ],
        help="Shared-q sweep ranking metric after hard pass/gate checks.",
    )
    parser.add_argument(
        "--prediction-label",
        default="family_conditioned_tabular_cs_predictor",
    )
    parser.add_argument(
        "--join-keys",
        default=None,
        help="Optional comma-separated parity join keys.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Train the predictor and write prediction/parity artifacts."""

    args = build_parser().parse_args(argv)
    if args.support_features is not None:
        required_shared_q_outputs = {
            "--output-support-predictions": args.output_support_predictions,
            "--output-shared-q-posterior-predictions": (
                args.output_shared_q_posterior_predictions
            ),
            "--output-shared-q-support-weights": args.output_shared_q_support_weights,
            "--output-shared-q-summary": args.output_shared_q_summary,
        }
        missing = [
            option for option, value in required_shared_q_outputs.items() if value is None
        ]
        if missing:
            build_parser().error(
                "--support-features requires " + ", ".join(missing)
            )
    train_splits = tuple(
        item.strip() for item in str(args.train_splits).split(",") if item.strip()
    )
    join_keys = (
        tuple(item.strip() for item in str(args.join_keys).split(",") if item.strip())
        if args.join_keys
        else None
    )
    predictor_config = ChemicalShiftPredictorConfig(
        ridge_alpha=float(args.ridge_alpha),
        gate_split=str(args.gate_split),
        train_splits=train_splits or ("train",),
        min_family_rows=int(args.min_family_rows),
        uncertainty_floor=float(args.uncertainty_floor),
        ood_score_threshold=float(args.ood_score_threshold),
        reward_uncertainty_scale=float(args.reward_uncertainty_scale),
        enable_pairwise_interactions=not bool(args.disable_pairwise_interactions),
        max_pairwise_interactions=int(args.max_pairwise_interactions),
        family_shrinkage_strength=float(args.family_shrinkage_strength),
        enable_family_affine_calibration=not bool(
            args.disable_family_affine_calibration
        ),
        family_calibration_strength=float(args.family_calibration_strength),
        prediction_label=str(args.prediction_label),
    )
    parity_config = ChemicalShiftParityConfig(
        target_ccc=float(args.target_ccc),
        min_family_count=int(args.min_family_count),
        gate_split=str(args.gate_split),
        join_keys=join_keys,
        prediction_label=str(args.prediction_label),
        require_uncertainty=True,
        require_ood=True,
        ood_score_threshold=float(args.ood_score_threshold),
        max_high_uncertainty_positive_reward_fraction=0.0,
    )
    shared_q_config = ChemicalShiftSharedQConfig(
        energy_temperature=float(args.shared_q_energy_temperature),
        energy_scale=float(args.shared_q_energy_scale),
        prior_log_prob_weight=float(args.shared_q_prior_log_prob_weight),
        uncertainty_threshold=None,
        ood_score_threshold=float(args.ood_score_threshold),
        min_posterior_ess=float(args.shared_q_min_posterior_ess),
        min_posterior_entropy=float(args.shared_q_min_posterior_entropy),
        max_posterior_top_mass=float(args.shared_q_max_posterior_top_mass),
        min_valid_mass=float(args.shared_q_min_valid_mass),
        posterior_solver_kind=str(args.shared_q_posterior_solver_kind),
        ccc_refinement_steps=int(args.shared_q_ccc_refinement_steps),
        ccc_refinement_learning_rate=float(
            args.shared_q_ccc_refinement_learning_rate
        ),
        ccc_refinement_residual_weight=float(
            args.shared_q_ccc_refinement_residual_weight
        ),
        ccc_refinement_energy_weight=float(args.shared_q_ccc_refinement_energy_weight),
        ccc_refinement_kl_weight=float(args.shared_q_ccc_refinement_kl_weight),
        ccc_refinement_floor=float(args.shared_q_ccc_refinement_floor),
        ccc_refinement_floor_weight=float(args.shared_q_ccc_refinement_floor_weight),
        ccc_refinement_macro_floor=float(args.shared_q_ccc_refinement_macro_floor),
        ccc_refinement_macro_floor_weight=float(
            args.shared_q_ccc_refinement_macro_floor_weight
        ),
        ccc_refinement_macro_reward_weight=float(
            args.shared_q_ccc_refinement_macro_reward_weight
        ),
        ccc_refinement_posterior_health_weight=float(
            args.shared_q_ccc_refinement_posterior_health_weight
        ),
        ccc_refinement_floor_aggregation=str(
            args.shared_q_ccc_refinement_floor_aggregation
        ),
        ccc_refinement_min_family_points=int(
            args.shared_q_ccc_refinement_min_family_points
        ),
        ccc_coordinate_refinement_passes=int(
            args.shared_q_ccc_coordinate_refinement_passes
        ),
        ccc_coordinate_candidate_limit=int(
            args.shared_q_ccc_coordinate_candidate_limit
        ),
        ccc_coordinate_target_fit_candidates=not bool(
            args.disable_shared_q_ccc_coordinate_target_fit_candidates
        ),
    )
    shared_q_sweep_config = None
    if bool(args.shared_q_sweep) or args.shared_q_sweep_report_output is not None:
        shared_q_sweep_config = ChemicalShiftSharedQSweepConfig(
            energy_temperatures=_float_tuple(args.shared_q_sweep_energy_temperatures),
            ccc_refinement_floor_weights=_float_tuple(
                args.shared_q_sweep_floor_weights
            ),
            ccc_refinement_macro_floor_weights=_float_tuple(
                args.shared_q_sweep_macro_floor_weights
            ),
            ccc_refinement_macro_reward_weights=_float_tuple(
                args.shared_q_sweep_macro_reward_weights
            ),
            ccc_refinement_posterior_health_weights=_float_tuple(
                args.shared_q_sweep_posterior_health_weights
            ),
            ccc_refinement_energy_weights=_float_tuple(
                args.shared_q_sweep_energy_weights
            ),
            ccc_refinement_kl_weights=_float_tuple(args.shared_q_sweep_kl_weights),
            ccc_refinement_residual_weights=_float_tuple(
                args.shared_q_sweep_residual_weights
            ),
            max_configs=int(args.shared_q_sweep_max_configs),
            selection_metric=str(args.shared_q_sweep_selection_metric),
        )
    summary = train_predictor_from_paths(
        targets_path=args.targets,
        splits_path=args.splits,
        features_path=args.features,
        predictions_output=args.output_predictions,
        bundle_output=args.output_bundle,
        summary_output=args.output_summary,
        metrics_output=args.metrics_output,
        pairs_output=args.pairs_output,
        support_features_path=args.support_features,
        support_predictions_output=args.output_support_predictions,
        shared_q_posterior_predictions_output=(
            args.output_shared_q_posterior_predictions
        ),
        shared_q_support_weights_output=args.output_shared_q_support_weights,
        shared_q_summary_output=args.output_shared_q_summary,
        shared_q_metrics_output=args.shared_q_metrics_output,
        shared_q_pairs_output=args.shared_q_pairs_output,
        shared_q_sweep_report_output=args.shared_q_sweep_report_output,
        predictor_config=predictor_config,
        parity_config=parity_config,
        shared_q_config=shared_q_config,
        shared_q_sweep_config=shared_q_sweep_config,
    )
    print(
        "chemical_shift_predictor_train "
        f"decision={summary['parity_decision']} "
        f"joined_pair_count={summary['joined_pair_count']} "
        f"bundle={args.output_bundle}"
    )
    print(json.dumps(summary, sort_keys=True))


def _float_tuple(value: str) -> tuple[float, ...]:
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return tuple(values)


if __name__ == "__main__":
    main()
