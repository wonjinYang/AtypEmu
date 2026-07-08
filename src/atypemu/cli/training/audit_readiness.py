"""CLI for production training-readiness audits."""

from __future__ import annotations

import argparse
import json

from atypemu.training import TrainingReadinessConfig, audit_training_readiness


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Audit whether an integrated workspace is ready for full student training.",
    )
    parser.add_argument("--data-root", required=True, help="Repository data root.")
    parser.add_argument(
        "--integrated-root",
        required=True,
        help="Integrated workspace root such as data/integrated.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional readiness-audit config JSON.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit with status 1 when the readiness gate is not satisfied.",
    )
    return parser


def main() -> None:
    """Run the readiness-audit CLI."""
    args = build_parser().parse_args()
    config = (
        TrainingReadinessConfig.from_json(args.config_json)
        if args.config_json
        else TrainingReadinessConfig()
    )
    summary = audit_training_readiness(
        data_root=args.data_root,
        integrated_root=args.integrated_root,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_ready and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
