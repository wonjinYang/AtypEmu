#!/usr/bin/env python3
"""Small dependency-light checks for the frozen primary metric."""

from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import all_label_one_shared_q_metric as metric


def check_ccc_and_constant_prediction() -> None:
    target = np.asarray([1.0, 2.0, 4.0])
    assert math.isclose(metric.concordance_correlation(target, target), 1.0)
    assert metric.concordance_correlation(target, np.ones(3)) == 0.0
    try:
        metric.concordance_correlation(target, np.asarray([1.0, np.nan, 4.0]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite prediction was masked")


def check_one_shared_q_recomputation() -> None:
    surface = pd.DataFrame(
        {
            "entity_uid": ["e", "e", "e", "e"],
            "target_id": ["t1", "t1", "t2", "t2"],
            "support_id": ["s1", "s2", "s1", "s2"],
            "support_prediction": [0.0, 2.0, 10.0, 14.0],
        }
    )
    q = pd.DataFrame(
        {
            "entity_uid": ["e", "e"],
            "support_id": ["s1", "s2"],
            "posterior_weight": [0.75, 0.25],
        }
    )
    weighted = surface.merge(q, on=["entity_uid", "support_id"], validate="many_to_one")
    means = (
        (weighted["support_prediction"] * weighted["posterior_weight"])
        .groupby(weighted["target_id"])
        .sum()
    )
    assert means.to_dict() == {"t1": 0.5, "t2": 11.0}
    assert set(q.columns) == {"entity_uid", "support_id", "posterior_weight"}


def check_receipts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "inventory.json"
        metric.write_receipted(
            path,
            {"artifact_kind": metric.INVENTORY_KIND, "contract": metric.CONTRACT},
        )
        loaded = metric.load_receipted(path, kind=metric.INVENTORY_KIND)
        assert loaded["contract"] == metric.CONTRACT
        payload = path.read_text().replace(metric.CONTRACT, "wrong", 1)
        path.write_text(payload)
        try:
            metric.load_receipted(path, kind=metric.INVENTORY_KIND)
        except ValueError as error:
            assert "receipt" in str(error)
        else:
            raise AssertionError("tampered receipt was accepted")


def synthetic_data_root(root: Path) -> Path:
    data_root = root / "data"
    (data_root / "targets").mkdir(parents=True)
    (data_root / "features").mkdir()
    entities = []
    for index, ca, ne2 in ((1, 1.0, 5.0), (2, 2.0, 7.0)):
        entity_uid = f"entity-{index}"
        bmrb_id = f"bmr{index}"
        entities.append(
            {
                "entity_uid": entity_uid,
                "bmrb_id": bmrb_id,
                "split": "train",
                "observer_fold": "A" if index == 1 else "B",
            }
        )
        targets = pd.DataFrame(
            {
                "entity_uid": [entity_uid] * 3,
                "target_id": ["ca", "hb2", "ne2"],
                "target_value": [ca, 3.0, ne2],
            }
        )
        targets.to_parquet(data_root / "targets" / f"{bmrb_id}.parquet", index=False)
        features = pd.DataFrame(
            {
                "entity_uid": [entity_uid] * 6,
                "target_id": ["ca", "ca", "hb2", "hb2", "ne2", "ne2"],
                "atom_id": ["CA", "CA", "HB2", "HB2", "NE2", "NE2"],
            }
        )
        features.to_parquet(data_root / "features" / f"{bmrb_id}.parquet", index=False)
    (data_root / "commitment.json").write_text(
        json.dumps({"entities": entities}) + "\n"
    )
    (data_root / "feature_receipt.json").write_text("{}\n")
    return data_root


def check_end_to_end_scoring_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_root = synthetic_data_root(root)
        inventory_path = root / "inventory.json"
        metric.write_receipted(inventory_path, metric.inventory_payload(data_root))
        inventory = metric.load_receipted(inventory_path, kind=metric.INVENTORY_KIND)
        assert inventory["eligible_atom_ids"] == ["CA", "NE2"]
        assert next(row for row in inventory["labels"] if row["atom_id"] == "HB2")[
            "ineligible_reasons"
        ] == ["zero_target_variance"]
        commitment_path = root / "evaluator_commitment.json"
        metric.write_receipted(
            commitment_path,
            metric.evaluator_commitment_payload(inventory_path),
        )
        surface_rows = []
        q_rows = []
        targets = metric.canonical_target_rows(data_root)
        for entity_uid, part in targets.groupby("entity_uid", sort=True):
            q_rows.extend(
                [
                    {
                        "entity_uid": entity_uid,
                        "support_id": "s1",
                        "posterior_weight": 0.75,
                    },
                    {
                        "entity_uid": entity_uid,
                        "support_id": "s2",
                        "posterior_weight": 0.25,
                    },
                ]
            )
            for row in part.itertuples(index=False):
                surface_rows.extend(
                    [
                        {
                            "entity_uid": entity_uid,
                            "target_id": row.target_id,
                            "support_id": "s1",
                            "support_prediction": row.target_value,
                        },
                        {
                            "entity_uid": entity_uid,
                            "target_id": row.target_id,
                            "support_id": "s2",
                            "support_prediction": row.target_value,
                        },
                    ]
                )
        surface_path = root / "surface.parquet"
        q_path = root / "q.parquet"
        pd.DataFrame(surface_rows).to_parquet(surface_path, index=False)
        pd.DataFrame(q_rows).to_parquet(q_path, index=False)
        result = metric.score(
            data_root=data_root,
            inventory_path=inventory_path,
            commitment_path=commitment_path,
            surface_paths=[surface_path],
            q_paths=[q_path],
        )
        assert result["all_label_macro_one_shared_q_ccc"] == 1.0
        assert result["eligible_atom_label_count"] == 2
        score_path = root / "score.json"
        metric.write_receipted(score_path, result)
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(metric.__file__).resolve().parent
                    / "verify_all_label_one_shared_q_score.py"
                ),
                "--data-root",
                str(data_root),
                "--inventory",
                str(inventory_path),
                "--score",
                str(score_path),
                "--surface",
                str(surface_path),
                "--q",
                str(q_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        broken = pd.DataFrame(surface_rows[:-1])
        broken.to_parquet(surface_path, index=False)
        try:
            metric.score(
                data_root=data_root,
                inventory_path=inventory_path,
                commitment_path=commitment_path,
                surface_paths=[surface_path],
                q_paths=[q_path],
            )
        except ValueError as error:
            assert "support" in str(error) or "cover" in str(error)
        else:
            raise AssertionError("missing assigned target/support row was accepted")


if __name__ == "__main__":
    check_ccc_and_constant_prediction()
    check_one_shared_q_recomputation()
    check_receipts()
    check_end_to_end_scoring_contract()
    print("all-label macro one-shared-q metric checks: PASS")
