#!/usr/bin/env python3
"""Independent arithmetic verifier for a frozen all-label shared-q score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("receipt_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual:
        raise ValueError(f"receipt mismatch: {path}")
    payload["receipt_sha256"] = claimed
    return payload


def read_targets(data_root: Path) -> pd.DataFrame:
    commitment = json.loads((data_root / "commitment.json").read_text())
    entities = sorted(
        (
            entity
            for entity in commitment["entities"]
            if entity.get("split") == "train"
            and entity.get("observer_fold") in {"A", "B"}
        ),
        key=lambda entity: str(entity["entity_uid"]),
    )
    frames = []
    for entity in entities:
        entity_uid = str(entity["entity_uid"])
        bmrb_id = str(entity["bmrb_id"])
        if Path(bmrb_id).name != bmrb_id or "\\" in bmrb_id:
            raise ValueError("unsafe bmrb_id")
        targets = pd.read_parquet(data_root / "targets" / f"{bmrb_id}.parquet")
        features = pd.read_parquet(data_root / "features" / f"{bmrb_id}.parquet")
        targets = targets[targets["entity_uid"].astype(str).eq(entity_uid)].copy()
        features = features[features["entity_uid"].astype(str).eq(entity_uid)].copy()
        targets["target_id"] = targets["target_id"].astype(str)
        features["target_id"] = features["target_id"].astype(str)
        features["atom_id"] = features["atom_id"].astype(str).str.strip().str.upper()
        identities = features[["target_id", "atom_id"]].drop_duplicates()
        if identities["target_id"].duplicated().any():
            raise ValueError("ambiguous Atom_ID")
        if "atom_id" in targets:
            targets = targets.rename(columns={"atom_id": "target_atom_id"})
        joined = targets.merge(
            identities, on="target_id", how="left", validate="one_to_one"
        )
        if "target_atom_id" in joined:
            target_atom_id = (
                joined["target_atom_id"].astype(str).str.strip().str.upper()
            )
            if not target_atom_id.eq(joined["atom_id"]).all():
                raise ValueError("target/feature Atom_ID mismatch")
        joined["entity_uid"] = entity_uid
        frames.append(joined[["entity_uid", "target_id", "atom_id", "target_value"]])
    rows = pd.concat(frames, ignore_index=True)
    rows["target_value"] = pd.to_numeric(rows["target_value"], errors="coerce")
    return rows[rows["target_value"].map(math.isfinite)].copy()


def ccc(target: list[float], prediction: list[float]) -> float:
    if len(target) != len(prediction) or len(target) < 2:
        raise ValueError("invalid CCC arrays")
    n = len(target)
    target_mean = math.fsum(target) / n
    prediction_mean = math.fsum(prediction) / n
    target_variance = math.fsum((value - target_mean) ** 2 for value in target) / n
    prediction_variance = (
        math.fsum((value - prediction_mean) ** 2 for value in prediction) / n
    )
    covariance = (
        math.fsum(
            (left - target_mean) * (right - prediction_mean)
            for left, right in zip(target, prediction, strict=True)
        )
        / n
    )
    denominator = (
        target_variance + prediction_variance + (target_mean - prediction_mean) ** 2
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("undefined CCC")
    return 2.0 * covariance / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--surface", type=Path, action="append", required=True)
    parser.add_argument("--q", type=Path, action="append", required=True)
    args = parser.parse_args()

    inventory = load_receipt(args.inventory)
    recorded = load_receipt(args.score)
    if recorded["surface_file_sha256"] != [file_hash(path) for path in args.surface]:
        raise ValueError("surface hash mismatch")
    if recorded["q_file_sha256"] != [file_hash(path) for path in args.q]:
        raise ValueError("q hash mismatch")
    if recorded["inventory_file_sha256"] != file_hash(args.inventory):
        raise ValueError("inventory hash mismatch")

    targets = read_targets(args.data_root)
    surface = pd.concat(
        [pd.read_parquet(path) for path in args.surface], ignore_index=True
    )
    posterior = pd.concat([pd.read_parquet(path) for path in args.q], ignore_index=True)
    for frame in (surface, posterior):
        frame["entity_uid"] = frame["entity_uid"].astype(str)
        frame["support_id"] = frame["support_id"].astype(str)
    surface["target_id"] = surface["target_id"].astype(str)
    posterior["posterior_weight"] = posterior["posterior_weight"].astype(float)
    joined = surface.merge(
        posterior,
        on=["entity_uid", "support_id"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["entity_uid", "target_id", "support_id"], kind="stable")
    joined["term"] = (
        joined["support_prediction"].astype(float) * joined["posterior_weight"]
    )
    means = (
        joined.groupby(["entity_uid", "target_id"], sort=True)["term"]
        .agg(lambda values: math.fsum(float(value) for value in values))
        .rename("prediction")
        .reset_index()
    )
    scored = targets.merge(means, on=["entity_uid", "target_id"], validate="one_to_one")
    if len(scored) != len(targets):
        raise ValueError("target coverage mismatch")

    expected_labels = list(inventory["eligible_atom_ids"])
    recorded_by_label = {
        row["atom_id"]: float(row["ccc"]) for row in recorded["per_label"]
    }
    independently_computed = []
    for atom_id in expected_labels:
        part = scored[scored["atom_id"].eq(atom_id)]
        value = ccc(
            [float(item) for item in part["target_value"]],
            [float(item) for item in part["prediction"]],
        )
        if not math.isclose(
            value, recorded_by_label[atom_id], abs_tol=1e-12, rel_tol=0.0
        ):
            raise ValueError(f"per-label CCC mismatch: {atom_id}")
        independently_computed.append(value)
    macro = math.fsum(independently_computed) / len(independently_computed)
    if not math.isclose(
        macro,
        float(recorded["all_label_macro_one_shared_q_ccc"]),
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise ValueError("macro CCC mismatch")
    print(f"independent_verifier=PASS macro={macro:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
