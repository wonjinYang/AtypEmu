#!/usr/bin/env python3
"""Backpressure checks for all-label candidate provenance and coordinate causality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DIRECTIONS = {"A_to_B": ("A", "B"), "B_to_A": ("B", "A")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    validity_path = args.run_root / "validity.json"
    validity = json.loads(validity_path.read_text())
    required_truths = {
        "assigned_D_e_consumed_by_q": True,
        "assigned_D_e_consumed_by_coordinate_generator": True,
        "one_q_per_entity": True,
        "observer_reads_assigned_target_values_directly": False,
        "support_predictions_derived_from_complete_coordinates": True,
        "outer_sealed_entities_read": False,
    }
    if validity.get("contract") != "atypemu_all_label_e2e_candidate_validity_v1":
        raise ValueError("candidate validity contract mismatch")
    for field, expected in required_truths.items():
        if validity.get(field) is not expected:
            raise ValueError(f"invalid candidate field: {field}")

    commitment = json.loads((args.data_root / "commitment.json").read_text())
    fold_entities = {
        fold: sorted(
            str(entity["entity_uid"])
            for entity in commitment["entities"]
            if entity.get("split") == "train" and entity.get("observer_fold") == fold
        )
        for fold in ("A", "B")
    }
    audits = validity.get("directions", {})
    if set(audits) != set(DIRECTIONS):
        raise ValueError("both crossfit directions are required")
    for direction, (train_fold, eval_fold) in DIRECTIONS.items():
        audit = audits[direction]
        if audit.get("train_entity_uids") != fold_entities[train_fold]:
            raise ValueError(f"wrong training entities: {direction}")
        if audit.get("eval_entity_uids") != fold_entities[eval_fold]:
            raise ValueError(f"wrong evaluation entities: {direction}")
        if set(audit["train_entity_uids"]) & set(audit["eval_entity_uids"]):
            raise ValueError(f"crossfit entity overlap: {direction}")
        gradient = float(audit.get("coordinate_generator_cs_gradient_norm", 0.0))
        if not math.isfinite(gradient) or gradient <= 0.0:
            raise ValueError(
                f"no CS-loss gradient to coordinate generator: {direction}"
            )
        perturbation = float(
            audit.get("frozen_coordinate_observer_target_delta", math.inf)
        )
        if perturbation != 0.0:
            raise ValueError(
                f"observer surface directly responds to targets: {direction}"
            )

        coordinate_path = args.run_root / f"coordinate_audit_{direction}.npz"
        arrays = np.load(coordinate_path)
        conditioned = np.asarray(arrays["conditioned_coordinates"], dtype=np.float64)
        no_evidence = np.asarray(arrays["no_evidence_coordinates"], dtype=np.float64)
        atom_mask = np.asarray(arrays["atom_mask"], dtype=bool)
        if conditioned.shape != no_evidence.shape or conditioned.shape[-1] != 3:
            raise ValueError(f"coordinate audit shape mismatch: {direction}")
        if atom_mask.shape != conditioned.shape[:-1] or not atom_mask.any():
            raise ValueError(f"coordinate atom mask mismatch: {direction}")
        if (
            not np.isfinite(conditioned[atom_mask]).all()
            or not np.isfinite(no_evidence[atom_mask]).all()
        ):
            raise ValueError(f"nonfinite complete-coordinate audit: {direction}")
        delta = conditioned[atom_mask] - no_evidence[atom_mask]
        rmsd = float(np.sqrt(np.mean(np.square(delta))))
        if not math.isfinite(rmsd) or rmsd <= 1.0e-6:
            raise ValueError(f"NMR evidence did not move coordinates: {direction}")

    expected_outputs = [
        *(f"surface_{direction}.parquet" for direction in DIRECTIONS),
        *(f"surface_no_coordinate_{direction}.parquet" for direction in DIRECTIONS),
        *(f"q_{direction}.parquet" for direction in DIRECTIONS),
        *(f"q_uniform_{direction}.parquet" for direction in DIRECTIONS),
        *(f"coordinate_audit_{direction}.npz" for direction in DIRECTIONS),
    ]
    hashes = validity.get("output_sha256", {})
    if set(hashes) != set(expected_outputs):
        raise ValueError("candidate output hash inventory mismatch")
    for name in expected_outputs:
        if hashes[name] != sha256_file(args.run_root / name):
            raise ValueError(f"candidate output hash mismatch: {name}")

    primary = json.loads((args.run_root / "score.json").read_text())[
        "all_label_macro_one_shared_q_ccc"
    ]
    no_coordinate = json.loads(
        (args.run_root / "no_coordinate_score.json").read_text()
    )["all_label_macro_one_shared_q_ccc"]
    uniform_q = json.loads((args.run_root / "uniform_q_score.json").read_text())[
        "all_label_macro_one_shared_q_ccc"
    ]
    if not float(primary) > float(no_coordinate):
        raise ValueError("coordinate actuator does not improve the primary metric")
    if not float(primary) > float(uniform_q):
        raise ValueError("learned shared q does not improve the primary metric")
    print("all-label E2E candidate validity checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
