"""CLI for building observable matrices from a pool and targets."""

from __future__ import annotations

import argparse

from atypemu.cli.common import load_bundle, load_pool
from atypemu.observables import ChemicalShiftMatrixBuilder, JCouplingHead, NOEHead
from atypemu.types import ObservableBundle


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Compute AtypEmu observable matrices.")
    parser.add_argument("--targets", required=True, help="Input target bundle JSON.")
    parser.add_argument("--pool", required=True, help="Input candidate pool JSONL.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    return parser


def main() -> None:
    """Run the observable computation CLI."""
    args = build_parser().parse_args()
    bundle = load_bundle(args.targets)
    pool = load_pool(args.pool)

    observable_bundle = ObservableBundle(
        chemical_shifts=(
            ChemicalShiftMatrixBuilder().build(pool, bundle)
            if bundle.chemical_shifts
            else None
        ),
        j_couplings=JCouplingHead().build(pool, bundle) if bundle.j_couplings else None,
        noe_restraints=NOEHead().build(pool, bundle) if bundle.noe_restraints else None,
    )
    observable_bundle.to_npz(args.output)


if __name__ == "__main__":
    main()
