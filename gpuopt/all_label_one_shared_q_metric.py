#!/usr/bin/env python3
"""Freeze and score the all-label macro one-shared-q development metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CONTRACT = "atypemu_all_label_macro_one_shared_q_ccc_v1"
INVENTORY_KIND = "atypemu_all_label_inventory_v1"
COMMITMENT_KIND = "atypemu_all_label_evaluator_commitment_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_receipted(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["receipt_sha256"] = receipt(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_receipted(path: Path, *, kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = payload.pop("receipt_sha256", None)
    if expected != receipt(payload):
        raise ValueError(f"receipt mismatch: {path}")
    if payload.get("artifact_kind") != kind:
        raise ValueError(f"artifact kind mismatch: {path}")
    payload["receipt_sha256"] = expected
    return payload


def canonical_atom_id(value: object) -> str:
    atom_id = str(value).strip().upper()
    if not atom_id or atom_id in {"NAN", "NONE"}:
        raise ValueError("empty canonical BMRB Atom_ID")
    return atom_id


def safe_file_stem(value: object, *, field: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or Path(text).name != text or "\\" in text:
        raise ValueError(f"unsafe {field}: {text!r}")
    return text


def selected_entities(commitment: dict[str, Any]) -> list[dict[str, Any]]:
    entities = [
        entity
        for entity in commitment["entities"]
        if entity.get("split") == "train" and entity.get("observer_fold") in {"A", "B"}
    ]
    if not entities:
        raise ValueError("no A/B development entities in commitment")
    entity_uids = [str(entity["entity_uid"]) for entity in entities]
    if len(entity_uids) != len(set(entity_uids)):
        raise ValueError("duplicate selected entity_uid in commitment")
    return sorted(entities, key=lambda row: str(row["entity_uid"]))


def canonical_target_rows(data_root: Path) -> pd.DataFrame:
    commitment_path = data_root / "commitment.json"
    commitment = json.loads(commitment_path.read_text())
    frames: list[pd.DataFrame] = []
    for entity in selected_entities(commitment):
        bmrb_id = safe_file_stem(entity["bmrb_id"], field="bmrb_id")
        entity_uid = str(entity["entity_uid"])
        targets = pd.read_parquet(
            data_root / "targets" / f"{bmrb_id}.parquet",
            columns=["entity_uid", "target_id", "target_value"],
        )
        targets = targets[targets["entity_uid"].astype(str).eq(entity_uid)].copy()
        if targets[["target_id", "target_value"]].isna().any().any():
            raise ValueError(f"missing target identity/value: {entity_uid}")
        targets["target_id"] = targets["target_id"].astype(str)
        features = pd.read_parquet(
            data_root / "features" / f"{bmrb_id}.parquet",
            columns=["entity_uid", "target_id", "atom_id"],
        )
        features = features[features["entity_uid"].astype(str).eq(entity_uid)].copy()
        features["target_id"] = features["target_id"].astype(str)
        features["atom_id"] = features["atom_id"].map(canonical_atom_id)
        identity = features[["target_id", "atom_id"]].drop_duplicates()
        if identity["target_id"].astype(str).duplicated().any():
            raise ValueError(f"target maps to multiple Atom_ID values: {entity_uid}")
        joined = targets.merge(
            identity, on="target_id", how="left", validate="one_to_one"
        )
        if joined["atom_id"].isna().any() or len(joined) != len(targets):
            raise ValueError(f"target/Atom_ID coverage mismatch: {entity_uid}")
        joined["entity_uid"] = entity_uid
        joined["target_id"] = joined["target_id"].astype(str)
        frames.append(joined[["entity_uid", "target_id", "atom_id", "target_value"]])
    rows = pd.concat(frames, ignore_index=True)
    if rows[["entity_uid", "target_id"]].duplicated().any():
        raise ValueError("duplicate entity-qualified target identity")
    return rows.sort_values(["entity_uid", "target_id"], kind="stable").reset_index(
        drop=True
    )


def row_identity_sha256(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in rows.itertuples(index=False):
        value = float(row.target_value)
        encoded = json.dumps(
            [str(row.entity_uid), str(row.target_id), str(row.atom_id), value.hex()],
            separators=(",", ":"),
        )
        digest.update(encoded.encode() + b"\n")
    return digest.hexdigest()


def inventory_payload(data_root: Path) -> dict[str, Any]:
    rows = canonical_target_rows(data_root)
    finite = rows[
        np.isfinite(pd.to_numeric(rows["target_value"], errors="coerce"))
    ].copy()
    finite["target_value"] = finite["target_value"].astype(float)
    labels: list[dict[str, Any]] = []
    eligible: list[str] = []
    for atom_id, assigned_part in rows.groupby("atom_id", sort=True):
        part = finite[finite["atom_id"].eq(atom_id)]
        values = part["target_value"].to_numpy(dtype=np.float64)
        variance = (
            float(np.mean((values - values.mean()) ** 2)) if len(values) else None
        )
        reasons: list[str] = []
        if len(values) == 0:
            reasons.append("no_finite_targets")
        else:
            if len(values) < 2:
                reasons.append("fewer_than_two_finite_targets")
            if not math.isfinite(variance) or variance <= 0.0:
                reasons.append("zero_target_variance")
        is_eligible = not reasons
        if is_eligible:
            eligible.append(str(atom_id))
        labels.append(
            {
                "atom_id": str(atom_id),
                "assigned_row_count": int(len(assigned_part)),
                "eligible": is_eligible,
                "entity_count": int(part["entity_uid"].nunique()),
                "finite_row_count": int(len(part)),
                "ineligible_reasons": reasons,
                "target_variance": variance,
            }
        )
    commitment_path = data_root / "commitment.json"
    feature_receipt_path = data_root / "feature_receipt.json"
    entities = selected_entities(json.loads(commitment_path.read_text()))
    if not eligible:
        raise ValueError("frozen primary inventory has no CCC-eligible Atom_ID labels")
    return {
        "artifact_kind": INVENTORY_KIND,
        "contract": CONTRACT,
        "cohort": "observer-fold A/B crossfit development entries",
        "eligibility_rule": {
            "group": "canonical BMRB Atom_ID only",
            "prediction_fields_read": [],
            "required_finite_target_count": 2,
            "required_positive_target_variance": True,
        },
        "eligible_atom_ids": eligible,
        "labels": labels,
        "assigned_row_count": int(len(rows)),
        "finite_assigned_row_count": int(len(finite)),
        "nonfinite_assigned_row_count": int(len(rows) - len(finite)),
        "entity_count": len(entities),
        "entity_uid_sha256": hashlib.sha256(
            "\n".join(str(entity["entity_uid"]) for entity in entities).encode()
        ).hexdigest(),
        "finite_row_identity_sha256": row_identity_sha256(finite),
        "source_bindings": {
            "commitment_sha256": sha256_file(commitment_path),
            "feature_receipt_sha256": sha256_file(feature_receipt_path),
        },
    }


def concordance_correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.size < 2 or target.size != prediction.size:
        raise ValueError("CCC requires equal arrays with at least two values")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("CCC inputs must be finite; labels may not be masked")
    target_delta = target - target.mean()
    prediction_delta = prediction - prediction.mean()
    denominator = float(
        np.mean(np.square(target_delta))
        + np.mean(np.square(prediction_delta))
        + (target.mean() - prediction.mean()) ** 2
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("CCC denominator is undefined")
    return float(2.0 * np.mean(target_delta * prediction_delta) / denominator)


def read_tables(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise ValueError("at least one table is required")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def verify_commitment(commitment_path: Path, inventory_path: Path) -> dict[str, Any]:
    commitment = load_receipted(commitment_path, kind=COMMITMENT_KIND)
    inventory = load_receipted(inventory_path, kind=INVENTORY_KIND)
    if commitment.get("contract") != CONTRACT or inventory.get("contract") != CONTRACT:
        raise ValueError("metric contract mismatch")
    if commitment["inventory_file_sha256"] != sha256_file(inventory_path):
        raise ValueError("inventory bytes differ from evaluator commitment")
    if commitment["evaluator_source_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("evaluator source differs from frozen commitment")
    sibling = Path(__file__).resolve().parent
    for field, name in (
        ("evaluator_test_sha256", "test_all_label_one_shared_q_metric.py"),
        ("independent_verifier_sha256", "verify_all_label_one_shared_q_score.py"),
    ):
        if commitment[field] != sha256_file(sibling / name):
            raise ValueError(f"{name} differs from frozen commitment")
    return inventory


def score(
    *,
    data_root: Path,
    inventory_path: Path,
    commitment_path: Path,
    surface_paths: list[Path],
    q_paths: list[Path],
) -> dict[str, Any]:
    inventory = verify_commitment(commitment_path, inventory_path)
    rows = canonical_target_rows(data_root)
    finite = rows[
        np.isfinite(pd.to_numeric(rows["target_value"], errors="coerce"))
    ].copy()
    finite["target_value"] = finite["target_value"].astype(float)
    if row_identity_sha256(finite) != inventory["finite_row_identity_sha256"]:
        raise ValueError("canonical scoring targets differ from frozen inventory")
    expected_sources = inventory["source_bindings"]
    if expected_sources["commitment_sha256"] != sha256_file(
        data_root / "commitment.json"
    ):
        raise ValueError("cohort commitment differs from frozen inventory")
    if expected_sources["feature_receipt_sha256"] != sha256_file(
        data_root / "feature_receipt.json"
    ):
        raise ValueError("feature receipt differs from frozen inventory")

    surface = read_tables(surface_paths)
    q = read_tables(q_paths)
    surface_required = {"entity_uid", "target_id", "support_id", "support_prediction"}
    q_required = {"entity_uid", "support_id", "posterior_weight"}
    if not surface_required.issubset(surface.columns):
        raise ValueError(
            f"surface missing columns: {sorted(surface_required - set(surface.columns))}"
        )
    if set(q.columns) != q_required:
        raise ValueError(
            "q table must contain only entity_uid, support_id, posterior_weight"
        )
    for frame in (surface, q):
        if frame[["entity_uid", "support_id"]].isna().any().any():
            raise ValueError("missing entity_uid/support_id")
        frame["entity_uid"] = frame["entity_uid"].astype(str)
        frame["support_id"] = frame["support_id"].astype(str)
        if frame["support_id"].isin({"", "nan", "None"}).any():
            raise ValueError("empty support_id")
    if surface["target_id"].isna().any():
        raise ValueError("missing target_id")
    surface["target_id"] = surface["target_id"].astype(str)
    if surface[["entity_uid", "target_id", "support_id"]].duplicated().any():
        raise ValueError("duplicate target-by-support prediction")
    if q[["entity_uid", "support_id"]].duplicated().any():
        raise ValueError("duplicate entity-level q weight")
    if not np.isfinite(
        pd.to_numeric(surface["support_prediction"], errors="coerce")
    ).all():
        raise ValueError("nonfinite support prediction invalidates the candidate")
    q["posterior_weight"] = pd.to_numeric(q["posterior_weight"], errors="coerce")
    if (
        not np.isfinite(q["posterior_weight"]).all()
        or (q["posterior_weight"] < 0).any()
    ):
        raise ValueError("q weights must be finite and nonnegative")
    sums = q.groupby("entity_uid", sort=False)["posterior_weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("each entity-level q must sum to one")

    expected_keys = finite[["entity_uid", "target_id"]]
    actual_keys = surface[["entity_uid", "target_id"]].drop_duplicates()
    coverage = expected_keys.merge(actual_keys, how="outer", indicator=True)
    if not coverage["_merge"].eq("both").all():
        raise ValueError(
            "candidate surface does not cover exactly every finite assigned target"
        )
    q_keys = q[["entity_uid", "support_id"]]
    surface_q = surface.merge(
        q_keys, on=["entity_uid", "support_id"], how="outer", indicator=True
    )
    if not surface_q["_merge"].eq("both").all():
        raise ValueError(
            "surface support axes do not match the one entity-level q table"
        )
    support_counts = surface.groupby(["entity_uid", "target_id"])[
        "support_id"
    ].nunique()
    entity_support_counts = q.groupby("entity_uid")["support_id"].nunique()
    expected_counts = support_counts.index.get_level_values("entity_uid").map(
        entity_support_counts
    )
    if not np.array_equal(support_counts.to_numpy(), expected_counts.to_numpy()):
        raise ValueError("some targets omit support members from their entity-level q")

    weighted = surface.merge(q, on=["entity_uid", "support_id"], validate="many_to_one")
    weighted["weighted_prediction"] = (
        weighted["support_prediction"].astype(float) * weighted["posterior_weight"]
    )
    prediction = (
        weighted.sort_values(["entity_uid", "target_id", "support_id"], kind="stable")
        .groupby(["entity_uid", "target_id"], sort=True)["weighted_prediction"]
        .agg(lambda values: math.fsum(float(value) for value in values))
        .rename("prediction")
        .reset_index()
    )
    scored = finite.merge(
        prediction, on=["entity_uid", "target_id"], validate="one_to_one"
    )
    if len(scored) != len(finite) or not np.isfinite(scored["prediction"]).all():
        raise ValueError(
            "missing or nonfinite posterior means invalidate the candidate"
        )

    eligible = list(inventory["eligible_atom_ids"])
    if not eligible:
        raise ValueError("frozen primary inventory has no CCC-eligible Atom_ID labels")
    per_label: list[dict[str, Any]] = []
    for atom_id in eligible:
        part = scored[scored["atom_id"].eq(atom_id)]
        value = concordance_correlation(
            part["target_value"].to_numpy(), part["prediction"].to_numpy()
        )
        per_label.append(
            {
                "atom_id": atom_id,
                "ccc": value,
                "entity_count": int(part["entity_uid"].nunique()),
                "row_count": int(len(part)),
            }
        )
    if [row["atom_id"] for row in per_label] != eligible:
        raise AssertionError("eligible inventory order drift")
    macro = float(np.mean([row["ccc"] for row in per_label]))
    return {
        "artifact_kind": "atypemu_all_label_macro_one_shared_q_score_v1",
        "contract": CONTRACT,
        "scientific_scope": "crossfit development model selection only",
        "final_sealed_claim_allowed": False,
        "primary_metric": "all_label_macro_one_shared_q_ccc",
        "all_label_macro_one_shared_q_ccc": macro,
        "goal": 0.95,
        "goal_achieved": macro >= 0.95,
        "eligible_atom_label_count": len(eligible),
        "finite_assigned_row_count": len(scored),
        "all_finite_assigned_rows_covered": True,
        "one_entity_level_q_recomputed_by_evaluator": True,
        "per_label": per_label,
        "diagnostic_ineligible_labels": [
            row for row in inventory["labels"] if not row["eligible"]
        ],
        "inventory_file_sha256": sha256_file(inventory_path),
        "evaluator_commitment_file_sha256": sha256_file(commitment_path),
        "surface_file_sha256": [sha256_file(path) for path in surface_paths],
        "q_file_sha256": [sha256_file(path) for path in q_paths],
    }


def evaluator_commitment_payload(inventory_path: Path) -> dict[str, Any]:
    inventory = load_receipted(inventory_path, kind=INVENTORY_KIND)
    sibling = Path(__file__).resolve().parent
    return {
        "artifact_kind": COMMITMENT_KIND,
        "contract": CONTRACT,
        "primary_formula": (
            "mean over frozen eligible canonical BMRB Atom_ID labels of population-moment "
            "CCC(target, sum_k q_e[k] * support_prediction[k]); one q_e per entry"
        ),
        "goal": 0.95,
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "inventory_file_sha256": sha256_file(inventory_path),
        "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_test_sha256": sha256_file(
            sibling / "test_all_label_one_shared_q_metric.py"
        ),
        "independent_verifier_sha256": sha256_file(
            sibling / "verify_all_label_one_shared_q_score.py"
        ),
        "semantics": {
            "task": "unseen-entry NMR evidence co-satisfaction/assimilation",
            "assigned_D_e_may_condition_q_e_and_generator": True,
            "same_D_e_is_scored": True,
            "eligibility_reads_predictions": False,
            "constant_prediction_exclusion_forbidden": True,
            "nonfinite_or_missing_prediction_invalidates_candidate": True,
            "exact_comp_id_atom_id_cells_are_primary": False,
            "masked_label_metrics_are_primary": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-inventory")
    freeze.add_argument("--data-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze_evaluator = subparsers.add_parser("freeze-evaluator")
    freeze_evaluator.add_argument("--inventory", type=Path, required=True)
    freeze_evaluator.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("score")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--commitment", type=Path, required=True)
    evaluate.add_argument("--surface", type=Path, action="append", required=True)
    evaluate.add_argument("--q", type=Path, action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze-inventory":
        write_receipted(args.output, inventory_payload(args.data_root))
        print(f"inventory={args.output} sha256={sha256_file(args.output)}")
        return 0
    if args.command == "freeze-evaluator":
        write_receipted(args.output, evaluator_commitment_payload(args.inventory))
        print(f"evaluator_commitment={args.output} sha256={sha256_file(args.output)}")
        return 0
    result = score(
        data_root=args.data_root,
        inventory_path=args.inventory,
        commitment_path=args.commitment,
        surface_paths=args.surface,
        q_paths=args.q,
    )
    write_receipted(args.output, result)
    print(
        f"METRIC all_label_macro_one_shared_q_ccc={result['all_label_macro_one_shared_q_ccc']:.12f}"
    )
    print(f"goal_achieved={int(result['goal_achieved'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
