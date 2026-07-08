"""CLI for sweeping one-shared-q posterior solver settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from atypemu.training.chemical_shift_shared_q import (
    ChemicalShiftSharedQConfig,
    ChemicalShiftSharedQSweepConfig,
    sweep_shared_q_posterior_from_paths,
)
from atypemu.training.cs_prediction_parity import (
    CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES,
    ChemicalShiftParityConfig,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--splits", default=None, type=Path)
    parser.add_argument("--output-posterior-predictions", required=True, type=Path)
    parser.add_argument("--output-support-weights", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--sweep-report-output", default=None, type=Path)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--pairs-output", default=None, type=Path)
    parser.add_argument("--gate-split", default="val")
    parser.add_argument("--target-ccc", default=0.95, type=float)
    parser.add_argument("--min-family-count", default=2, type=int)
    parser.add_argument(
        "--required-atom-families",
        default=None,
        help="Optional comma-separated required atom families for this sweep.",
    )
    parser.add_argument("--energy-scale", default=1.0, type=float)
    parser.add_argument("--prior-log-prob-weight", default=1.0, type=float)
    parser.add_argument("--uncertainty-threshold", default=None, type=float)
    parser.add_argument("--ood-score-threshold", default=None, type=float)
    parser.add_argument("--min-posterior-ess", default=1.0, type=float)
    parser.add_argument("--min-posterior-entropy", default=0.0, type=float)
    parser.add_argument("--max-posterior-top-mass", default=1.0, type=float)
    parser.add_argument("--min-valid-mass", default=1.0, type=float)
    parser.add_argument("--ccc-refinement-steps", default=64, type=int)
    parser.add_argument("--ccc-refinement-floor", default=0.95, type=float)
    parser.add_argument("--ccc-refinement-macro-floor", default=0.95, type=float)
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
        help="Use active_set to select/refine against the worst family floor gap.",
    )
    parser.add_argument("--ccc-coordinate-refinement-passes", default=2, type=int)
    parser.add_argument("--ccc-coordinate-candidate-limit", default=16, type=int)
    parser.add_argument(
        "--disable-ccc-coordinate-target-fit-candidates",
        action="store_true",
        help="Disable per-entity target-fit simplex candidates for ablation.",
    )
    parser.add_argument("--energy-temperatures", default="0.005,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--ccc-refinement-floor-weights", default="0,2,5,20")
    parser.add_argument("--ccc-refinement-macro-floor-weights", default="0,2,5,20")
    parser.add_argument("--ccc-refinement-macro-reward-weights", default="0,0.25,1")
    parser.add_argument("--ccc-refinement-posterior-health-weights", default="0,1,5")
    parser.add_argument("--ccc-refinement-energy-weights", default="0,0.05")
    parser.add_argument("--ccc-refinement-kl-weights", default="0,0.01")
    parser.add_argument("--ccc-refinement-residual-weights", default="0,1")
    parser.add_argument("--max-configs", default=256, type=int)
    parser.add_argument("--selection-split", default=None)
    parser.add_argument(
        "--selection-metric",
        default="family_gate_then_min_macro_ccc",
        choices=[
            "family_gate_then_min_macro_ccc",
            "family_gate_then_macro_ccc",
            "family_min_ccc_then_macro",
            "objective_then_macro_ccc",
        ],
        help="Sweep ranking metric after hard pass/gate checks.",
    )
    parser.add_argument(
        "--allow-missing-support-id",
        action="store_true",
        help="Treat all predictions as one support; diagnostic only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Sweep solver settings and write the best posterior artifacts."""

    args = build_parser().parse_args(argv)
    posterior_config = ChemicalShiftSharedQConfig(
        energy_scale=float(args.energy_scale),
        prior_log_prob_weight=float(args.prior_log_prob_weight),
        uncertainty_threshold=args.uncertainty_threshold,
        ood_score_threshold=args.ood_score_threshold,
        min_posterior_ess=float(args.min_posterior_ess),
        min_posterior_entropy=float(args.min_posterior_entropy),
        max_posterior_top_mass=float(args.max_posterior_top_mass),
        min_valid_mass=float(args.min_valid_mass),
        require_support_id=not bool(args.allow_missing_support_id),
        ccc_refinement_steps=int(args.ccc_refinement_steps),
        ccc_refinement_floor=float(args.ccc_refinement_floor),
        ccc_refinement_macro_floor=float(args.ccc_refinement_macro_floor),
        ccc_refinement_macro_reward_weight=float(
            args.ccc_refinement_macro_reward_weight
        ),
        ccc_refinement_posterior_health_weight=float(
            args.ccc_refinement_posterior_health_weight
        ),
        ccc_refinement_floor_aggregation=str(args.ccc_refinement_floor_aggregation),
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
        required_atom_families=(
            _str_tuple(args.required_atom_families)
            if args.required_atom_families
            else CANONICAL_CHEMICAL_SHIFT_ATOM_FAMILIES
        ),
        prediction_label=posterior_config.posterior_label,
        require_uncertainty=True,
        require_ood=True,
        uncertainty_threshold=args.uncertainty_threshold,
        ood_score_threshold=args.ood_score_threshold,
        max_high_uncertainty_positive_reward_fraction=0.0,
    )
    sweep_config = ChemicalShiftSharedQSweepConfig(
        energy_temperatures=_float_tuple(args.energy_temperatures),
        ccc_refinement_floor_weights=_float_tuple(args.ccc_refinement_floor_weights),
        ccc_refinement_macro_floor_weights=_float_tuple(
            args.ccc_refinement_macro_floor_weights
        ),
        ccc_refinement_macro_reward_weights=_float_tuple(
            args.ccc_refinement_macro_reward_weights
        ),
        ccc_refinement_posterior_health_weights=_float_tuple(
            args.ccc_refinement_posterior_health_weights
        ),
        ccc_refinement_energy_weights=_float_tuple(args.ccc_refinement_energy_weights),
        ccc_refinement_kl_weights=_float_tuple(args.ccc_refinement_kl_weights),
        ccc_refinement_residual_weights=_float_tuple(
            args.ccc_refinement_residual_weights
        ),
        max_configs=int(args.max_configs),
        selection_split=args.selection_split,
        selection_metric=str(args.selection_metric),
    )
    summary = sweep_shared_q_posterior_from_paths(
        predictions_path=args.predictions,
        targets_path=args.targets,
        splits_path=args.splits,
        posterior_predictions_output=args.output_posterior_predictions,
        support_weights_output=args.output_support_weights,
        summary_output=args.output_summary,
        sweep_report_output=args.sweep_report_output,
        metrics_output=args.metrics_output,
        pairs_output=args.pairs_output,
        posterior_config=posterior_config,
        parity_config=parity_config,
        sweep_config=sweep_config,
    )
    sweep = summary["sweep"]
    print(
        "chemical_shift_shared_q_sweep "
        f"decision={summary['decision']} "
        f"best_index={sweep['best_sweep_index']} "
        f"evaluated={sweep['evaluated_config_count']} "
        f"posterior={args.output_posterior_predictions}"
    )
    print(json.dumps(summary, sort_keys=True))


def _float_tuple(value: str) -> tuple[float, ...]:
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return tuple(values)


def _str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


if __name__ == "__main__":
    main()
