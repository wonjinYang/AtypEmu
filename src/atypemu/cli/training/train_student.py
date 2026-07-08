"""CLI for first-stage density-student training."""

from __future__ import annotations

import argparse
import json

from atypemu.training.config import StudentTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train the first AtypEmu density student from materialized teachers.",
    )
    parser.add_argument("--data-root", required=True, help="Repository data root.")
    parser.add_argument(
        "--integrated-root",
        required=True,
        help="Integrated workspace root such as data/integrated.",
    )
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional student-training config JSON.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional learning-rate override.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override such as cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random-seed override.",
    )
    parser.add_argument(
        "--weight-kl-weight",
        type=float,
        default=None,
        help="Optional teacher-weight KL coefficient override.",
    )
    parser.add_argument(
        "--chemical-shift-loss-weight",
        type=float,
        default=None,
        help="Optional chemical-shift supervision coefficient override.",
    )
    return parser


def main() -> None:
    """Run the student-training CLI."""
    args = build_parser().parse_args()
    try:
        from atypemu.training.trainer import run_student_training
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise SystemExit(
            "PyTorch is required for `atypemu-train-student`. "
            "Run this command inside the training container or another torch-enabled environment."
        ) from exc

    config = (
        StudentTrainingConfig.from_json(args.config_json)
        if args.config_json
        else StudentTrainingConfig()
    )
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.weight_kl_weight is not None:
        config.weight_kl_weight = args.weight_kl_weight
    if args.chemical_shift_loss_weight is not None:
        config.chemical_shift_loss_weight = args.chemical_shift_loss_weight

    summary = run_student_training(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
