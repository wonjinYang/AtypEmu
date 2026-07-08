"""CLI for BioEmu probe UCBShift2.0/CS-reweighting teacher generation."""

from __future__ import annotations

import argparse
import json

from atypemu.training.bioemu_probe_ucbshift_teacher import (
    run_bioemu_probe_ucbshift_cnnls_teacher,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export detached BioEmu probe conformers, run UCBShift2.0 sidecars, "
            "and fit CS-reweighting population teachers for latent NMR training."
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--probe-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--pdb-root",
        default=None,
        help=(
            "Optional shared root for exported BioEmu sample_*.pdb files. "
            "Use this to let sharded sidecar jobs reuse one PDB cache."
        ),
    )
    parser.add_argument(
        "--sidecar-root",
        default=None,
        help=(
            "Optional shared root for UCBShift sample_*.csv sidecars. "
            "Use this to let sharded jobs share locked/reusable sidecars."
        ),
    )
    parser.add_argument("--ucbshift-config-json", default=None)
    parser.add_argument(
        "--method",
        choices=[
            "cs_reweighting",
            "cs-reweighting",
            "csrw",
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
    )
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--default-cs-sigma", type=float, default=1.0)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--max-conformers-per-entity", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--refresh-outputs", action="store_true")
    parser.add_argument("--require-all-sidecars", action="store_true")
    parser.add_argument(
        "--ready-probe-outputs-only",
        action="store_true",
        help=(
            "Process only probe entries whose detached BioEmu NPZ outputs already "
            "exist. Useful for sidecar warmstarts while sampling is still running."
        ),
    )
    parser.add_argument(
        "--auto-fill-missing-sidecars",
        action="store_true",
        help=(
            "Before fitting the teacher, copy the nearest usable same-entity "
            "sidecar into failed BioEmu sample slots and write a replacement report."
        ),
    )
    parser.add_argument(
        "--sidecar-only",
        action="store_true",
        help="Generate/reuse UCBShift sidecars but skip CS-reweighting teacher fitting.",
    )
    parser.add_argument(
        "--skip-pdb-export",
        action="store_true",
        help="Reuse already exported sample_*.pdb files under the teacher structure dir.",
    )
    parser.add_argument(
        "--allow-topology-fallback",
        action="store_true",
        help="Use topology.pdb as a single fallback sample when NPZ export is unavailable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_bioemu_probe_ucbshift_cnnls_teacher(
        data_root=args.data_root,
        run_dir=args.run_dir,
        probe_name=args.probe_name,
        output_dir=args.output_dir,
        pdb_root=args.pdb_root,
        sidecar_root=args.sidecar_root,
        ucbshift_config_path=args.ucbshift_config_json,
        method=args.method,
        lambda_reg=args.lambda_reg,
        default_cs_sigma=args.default_cs_sigma,
        max_entities=args.max_entities,
        max_conformers_per_entity=args.max_conformers_per_entity,
        refresh_outputs=bool(args.refresh_outputs),
        worker_count=args.worker_count,
        require_all_sidecars=bool(args.require_all_sidecars),
        skip_pdb_export=bool(args.skip_pdb_export),
        allow_topology_fallback=bool(args.allow_topology_fallback),
        ready_probe_outputs_only=bool(args.ready_probe_outputs_only),
        auto_fill_missing_sidecars=bool(args.auto_fill_missing_sidecars),
        sidecar_only=bool(args.sidecar_only),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    if summary.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
