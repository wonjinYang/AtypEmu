#!/usr/bin/env python3
"""Run the current all-label shared-q autoresearch candidate."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from candidates.torsion_assimilator import (
    ACTUATOR_SPECS,
    GEOMETRY_COLUMNS,
    SUPPORT_COUNT,
    actuator_delta,
    attach_ucbshift_anchor,
    calibrate_reference_bounds,
    coordinate_audit,
    crossfit_anchor_selection,
    fold_copy,
    load_fold,
    normalization,
    optimize_assimilation,
    predict_surface,
    q_frame,
    sha256_file,
    surface_frame,
    train_observer,
)

SEQUENCE_ANCHOR_SHA256 = {
    "A": "165f34ced53aabb4b908a43716ac8f4b76e4b3990507994b4fea0aa7d4fabd58",
    "B": "97346891f5fa7e7daa41a81a73cf5c90f9bbb577f293f6d816e76ea6576553ee",
}
SEQUENCE_ANCHOR_SUMMARY_SHA256 = {
    "A": "3f64bba6c197b7318b23f1587c4f218de027e89372aec754a8d21cb876642cef",
    "B": "2419aee1fc84353749565c201275694fcc8f864d972456f276017433a27e7f38",
}
SEQUENCE_ANCHOR_RECEIPT_SHA256 = (
    "2869e7b28d08580de1046c453a6acafe04f77a45a0b1eecd6aba86ec2bd6149f"
)
GEOMETRY_OBSERVER_PLAN_SHA256 = (
    "1d3bf7035d46fa257a8e6167e9b2f7bfdc5e60037f2aa9eb64d268b491d46d3d"
)


def candidate_source_hashes(inventory_path: Path, data_root: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    inventory_path = inventory_path.resolve()
    data_root = data_root.resolve()
    if inventory_path != (root / ".auto/frozen/all_label_inventory.json").resolve():
        raise ValueError("candidate requires the frozen all-label inventory")
    paths = (
        root / ".auto/measure.sh",
        root / ".auto/checks.sh",
        inventory_path,
        root / ".auto/preunblind/run85_geometry_observer_plan.json",
        data_root / "commitment.json",
        data_root / "feature_receipt.json",
        Path(__file__).resolve(),
        root / "gpuopt/candidates/torsion_assimilator.py",
    )
    hashes: dict[str, str] = {}
    for path in paths:
        try:
            identity = str(path.relative_to(root))
        except ValueError:
            identity = str(path)
        hashes[identity] = sha256_file(path)
    return hashes


def freeze_candidate_sources(
    output_root: Path, inventory_path: Path, data_root: Path
) -> dict[str, str]:
    hashes = candidate_source_hashes(inventory_path, data_root)
    if (
        hashes[".auto/preunblind/run85_geometry_observer_plan.json"]
        != GEOMETRY_OBSERVER_PLAN_SHA256
    ):
        raise ValueError("geometry observer plan hash mismatch")
    path = output_root / "geometry_observer_source_commitment.json"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = freeze_candidate_sources(
        args.output_root, args.inventory, args.data_root
    )
    plan = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / ".auto/preunblind/run85_geometry_observer_plan.json"
        ).read_text()
    )
    if tuple(plan["geometry_columns"]) != GEOMETRY_COLUMNS:
        raise ValueError("geometry observer columns differ from frozen plan")
    commitment = json.loads((args.data_root / "commitment.json").read_text())
    inventory = json.loads(args.inventory.read_text())
    atom_index = {"<UNK>": 0}
    atom_index.update(
        {
            atom_id: index
            for index, atom_id in enumerate(inventory["eligible_atom_ids"], start=1)
        }
    )
    fold_entities = {
        fold: sorted(
            (
                entity
                for entity in commitment["entities"]
                if entity.get("split") == "train"
                and entity.get("observer_fold") == fold
            ),
            key=lambda entity: str(entity["entity_uid"]),
        )
        for fold in ("A", "B")
    }
    device = torch.device("cuda")
    embedding_cache = args.data_root.parent / "all_atom_shared_q_e2e_v0" / "esm2_cache"
    structure_root = args.data_root.parent / "BioEmu"
    validity_directions = {}
    output_names = []
    cache_root = args.output_root.parents[1] / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "torsion_geometry_observer_folds_v1.pkl"
    cache_key = {
        "version": 1,
        "commitment_sha256": sha256_file(args.data_root / "commitment.json"),
        "feature_receipt_sha256": sha256_file(args.data_root / "feature_receipt.json"),
        "inventory_sha256": sha256_file(args.inventory),
    }
    raw_folds = None
    if cache_path.exists():
        print(f"loading cache {cache_path}", flush=True)
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)  # noqa: S301 - local hash-bound cache
        if cached.get("key") == cache_key:
            raw_folds = cached["folds"]
    if raw_folds is None:
        print("materializing target/fold cache", flush=True)
        raw_folds = {
            fold: load_fold(
                args.data_root,
                fold_entities[fold],
                atom_index=atom_index,
                embedding_cache=embedding_cache,
            )
            for fold in ("A", "B")
        }
        temporary = cache_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"key": cache_key, "folds": raw_folds}, handle, protocol=5)
        temporary.replace(cache_path)
        print(f"wrote cache {cache_path}", flush=True)

    sequence_anchor_hashes = {}
    for fold in ("A", "B"):
        anchor_root = (
            Path(__file__).resolve().parent
            / "assets"
            / "sequence_final_atom_macro_ensemble3_v0"
            / fold
        )
        anchor_path = anchor_root / "predictions.parquet"
        summary_path = anchor_root / "summary.json"
        actual_hash = sha256_file(anchor_path)
        if actual_hash != SEQUENCE_ANCHOR_SHA256[fold]:
            raise ValueError(f"sequence anchor hash mismatch for fold {fold}")
        if sha256_file(summary_path) != SEQUENCE_ANCHOR_SUMMARY_SHA256[fold]:
            raise ValueError(f"sequence anchor summary hash mismatch for fold {fold}")
        anchor_summary = json.loads(summary_path.read_text())
        if (
            anchor_summary.get("checkpoint_selection")
            != "final_epoch_target_unread"
            or len(anchor_summary.get("history", ())) != 8
        ):
            raise ValueError(f"sequence anchor is not fixed-final for fold {fold}")
        receipt_path = anchor_root.parent / "receipt.json"
        if sha256_file(receipt_path) != SEQUENCE_ANCHOR_RECEIPT_SHA256:
            raise ValueError("sequence anchor provenance receipt hash mismatch")
        # Column projection is part of the independence contract: the colocated
        # target_value column is never materialized in this candidate process.
        anchor = pd.read_parquet(anchor_path, columns=("target_id", "prediction"))
        if anchor["target_id"].duplicated().any():
            raise ValueError(f"duplicate sequence anchor targets in fold {fold}")
        aligned = raw_folds[fold]["frame"]["target_id"].map(
            anchor.set_index("target_id")["prediction"]
        )
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy(float)).all():
            raise ValueError(f"incomplete sequence anchor coverage in fold {fold}")
        raw_folds[fold]["anchor"] = aligned.to_numpy(dtype=np.float32)
        sequence_anchor_hashes[fold] = actual_hash

    ucb_anchor_root = (
        args.data_root.parent
        / "all_atom_shared_q_e2e_v0"
        / "ucbshift_x_anchor_v0"
    )
    ucb_anchor_stats = {
        fold: attach_ucbshift_anchor(raw_folds[fold], ucb_anchor_root)
        for fold in ("A", "B")
    }

    for direction_number, (train_fold, eval_fold, direction) in enumerate(
        (("A", "B", "A_to_B"), ("B", "A", "B_to_A"))
    ):
        train = fold_copy(raw_folds[train_fold])
        evaluation = fold_copy(raw_folds[eval_fold])
        anchor_selection = crossfit_anchor_selection(train, evaluation)
        train, evaluation = normalization(train, evaluation)
        reference_bounds = calibrate_reference_bounds(train)
        print(f"{direction}: training observer", flush=True)
        model = train_observer(
            train,
            atom_levels=len(atom_index),
            device=device,
            seed=20260829 + direction_number,
        )
        print(f"{direction}: assimilating D_e", flush=True)
        delta, q, reference_offset, gradient_norm = optimize_assimilation(
            model,
            evaluation,
            device=device,
            reference_bounds_ppm=reference_bounds,
        )
        conditioned = predict_surface(
            model,
            evaluation,
            device=device,
            delta_raw=delta,
            reference_offset=reference_offset,
        )
        no_coordinate = predict_surface(
            model,
            evaluation,
            device=device,
            delta_raw=None,
            reference_offset=reference_offset,
        )
        uniform_q = np.full(
            (len(evaluation["entities"]), SUPPORT_COUNT),
            1.0 / SUPPORT_COUNT,
            dtype=np.float64,
        )
        outputs = {
            f"surface_{direction}.parquet": surface_frame(evaluation, conditioned),
            f"surface_no_coordinate_{direction}.parquet": surface_frame(
                evaluation, no_coordinate
            ),
            f"q_{direction}.parquet": q_frame(evaluation, q.cpu().numpy()),
            f"q_uniform_{direction}.parquet": q_frame(evaluation, uniform_q),
        }
        for name, frame in outputs.items():
            frame.to_parquet(args.output_root / name, index=False)
            output_names.append(name)
        coordinate_name = f"coordinate_audit_{direction}.npz"
        coordinate_audit(
            evaluation,
            delta,
            structure_root=structure_root,
            output=args.output_root / coordinate_name,
        )
        output_names.append(coordinate_name)
        validity_directions[direction] = {
            "train_entity_uids": [
                str(entity["entity_uid"]) for entity in fold_entities[train_fold]
            ],
            "eval_entity_uids": [
                str(entity["entity_uid"]) for entity in fold_entities[eval_fold]
            ],
            "coordinate_generator_cs_gradient_norm": gradient_norm,
            "frozen_coordinate_observer_target_delta": 0.0,
            "reference_offset": {
                element: {
                    "mean_abs_ppm": float(np.abs(reference_offset[:, index]).mean()),
                    "max_abs_ppm": float(np.abs(reference_offset[:, index]).max()),
                }
                for index, element in enumerate(("H", "C", "N"))
            },
            "source_calibrated_reference_bounds_ppm": {
                element: float(reference_bounds[index])
                for index, element in enumerate(("H", "C", "N"))
            },
            "actuator": {
                name: {
                    "mean_abs_radians": float(
                        actuator_delta(delta)[..., dimension].abs().mean().item()
                    ),
                    "max_abs_radians": float(
                        actuator_delta(delta)[..., dimension].abs().max().item()
                    ),
                    "raw_saturation_fraction": float(
                        (delta[..., dimension].tanh().abs() > 0.95)
                        .float()
                        .mean()
                        .item()
                    ),
                }
                for dimension, (name, _bound) in enumerate(ACTUATOR_SPECS)
            },
            "q_mean_entropy": float(
                (-q * q.clamp_min(1.0e-30).log()).sum(dim=1).mean().item()
            ),
            "crossfit_anchor_selection": anchor_selection,
            "target_free_complete_coordinate_geometry_features": list(
                GEOMETRY_COLUMNS
            ),
        }
        del model, train, evaluation, delta, q
        torch.cuda.empty_cache()
        print(f"{direction}: outputs complete", flush=True)

    validity = {
        "contract": "atypemu_all_label_e2e_candidate_validity_v1",
        "assigned_D_e_consumed_by_q": True,
        "assigned_D_e_consumed_by_coordinate_generator": True,
        "one_q_per_entity": True,
        "observer_reads_assigned_target_values_directly": False,
        "support_predictions_derived_from_complete_coordinates": True,
        "outer_sealed_entities_read": False,
        "target_free_complete_coordinate_geometry_observer": True,
        "geometry_observer_source_commitment": source_hashes,
        "frozen_sequence_anchor_sha256": sequence_anchor_hashes,
        "sequence_anchor_columns_read": ["target_id", "prediction"],
        "frozen_ucbshift_x_anchor": ucb_anchor_stats,
        "ucbshift_x_arrays_read": ["target_ids", "support_ids", "prediction_ppm"],
        "directions": validity_directions,
        "output_sha256": {
            name: sha256_file(args.output_root / name) for name in output_names
        },
    }
    if source_hashes != candidate_source_hashes(args.inventory, args.data_root):
        raise ValueError("candidate sources changed during geometry-observer execution")
    (args.output_root / "validity.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
