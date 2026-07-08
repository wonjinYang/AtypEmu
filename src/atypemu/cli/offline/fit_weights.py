"""CLI for fitting simplex weights from observable matrices."""

from __future__ import annotations

import argparse

from atypemu.cli.common import load_bundle, load_observables, save_weights_csv
from atypemu.energy import MultiObservablePosteriorEnergy
from atypemu.reweighting import EuclideanSimplexReweighter, MaxEntReweighter


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Fit AtypEmu simplex weights.")
    parser.add_argument("--targets", required=True, help="Input target bundle JSON.")
    parser.add_argument("--observables", required=True, help="Input observable NPZ.")
    parser.add_argument(
        "--method",
        default="maxent",
        choices=["maxent", "euclidean"],
        help="Reweighting method.",
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.01, help="Regularization strength."
    )
    parser.add_argument(
        "--beta-cs", type=float, default=1.0, help="Chemical-shift channel weight."
    )
    parser.add_argument(
        "--beta-j", type=float, default=1.0, help="J-coupling channel weight."
    )
    parser.add_argument(
        "--beta-noe", type=float, default=1.0, help="NOE channel weight."
    )
    parser.add_argument("--output", required=True, help="Output fit-result JSON.")
    parser.add_argument(
        "--weights-csv", default=None, help="Optional CSV export for fitted weights."
    )
    return parser


def main() -> None:
    """Run the weight-fitting CLI."""
    args = build_parser().parse_args()
    bundle = load_bundle(args.targets)
    observables = load_observables(args.observables)

    if args.method == "maxent":
        reweighter = MaxEntReweighter()
        regularizer = "maxent"
    else:
        reweighter = EuclideanSimplexReweighter()
        regularizer = "euclidean"

    energy_model = MultiObservablePosteriorEnergy(
        reweighter=reweighter,
        beta_cs=args.beta_cs,
        beta_j=args.beta_j,
        beta_noe=args.beta_noe,
        lambda_reg=args.lambda_reg,
        regularizer=regularizer,
    )
    result = energy_model.score(bundle, observables)
    result.to_json(args.output)

    if args.weights_csv:
        save_weights_csv(args.weights_csv, observables.candidate_ids(), result.weights)


if __name__ == "__main__":
    main()
