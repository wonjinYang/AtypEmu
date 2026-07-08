"""CLI for evaluating fitted AtypEmu weights."""

from __future__ import annotations

import argparse

from atypemu.cli.common import load_breakdown_as_solution, load_bundle, load_observables
from atypemu.energy import MultiObservablePosteriorEnergy
from atypemu.evaluation import Evaluator
from atypemu.reweighting import EuclideanSimplexReweighter


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate an AtypEmu fit result.")
    parser.add_argument("--targets", required=True, help="Input target bundle JSON.")
    parser.add_argument("--observables", required=True, help="Input observable NPZ.")
    parser.add_argument("--fit-result", required=True, help="Input fit-result JSON.")
    parser.add_argument("--output", required=True, help="Output evaluation JSON.")
    return parser


def main() -> None:
    """Run the evaluation CLI."""
    args = build_parser().parse_args()
    bundle = load_bundle(args.targets)
    observables = load_observables(args.observables)
    solution = load_breakdown_as_solution(args.fit_result)
    energy_model = MultiObservablePosteriorEnergy(
        reweighter=EuclideanSimplexReweighter(),
        regularizer="euclidean",
    )
    report = Evaluator(energy_model).evaluate(bundle, observables, solution)
    report.to_json(args.output)


if __name__ == "__main__":
    main()
