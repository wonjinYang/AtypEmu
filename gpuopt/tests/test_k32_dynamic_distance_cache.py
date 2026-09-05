from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from gpuopt import materialize_k32_dynamic_distance_cache as cache


ROOT = Path(__file__).resolve().parents[2]


def _records() -> dict[str, object]:
    # Two SER residues.  Their CA->CB axes are parallel, so rotating either OG
    # gives a simple, signed finite-difference check.
    return {
        "name": np.asarray(["CA", "CB", "OG", "CA", "CB", "OG"]),
        "resname": np.asarray(["SER"] * 6),
        "chain": np.asarray(["A"] * 6),
        "seq_id": np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        "element": np.asarray(["C", "C", "O", "C", "C", "O"]),
        "coordinate": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [1.5, 1.5, 0.0],
                [0.0, 0.0, 4.0],
                [1.5, 0.0, 4.0],
                [1.5, 1.5, 4.0],
            ],
            dtype=np.float64,
        ),
        "residue_key": ((0, "A", 1, ""),) * 3 + ((0, "A", 2, ""),) * 3,
    }


def _rotate(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    relative = point - origin
    return (
        relative * np.cos(angle)
        + np.cross(axis, relative) * np.sin(angle)
        + axis * np.dot(axis, relative) * (1.0 - np.cos(angle))
        + origin
    )


class DynamicDistanceCacheTests(unittest.TestCase):
    def test_support_roster_matches_independent_checker_literal(self) -> None:
        module = ast.parse(
            (ROOT / "gpuopt/check_k32_complete_coordinate_supports.py").read_text()
        )
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "EXPECTED_SUPPORTS"
                for target in node.targets
            )
        )
        self.assertEqual(cache.EXPECTED_SUPPORTS, tuple(ast.literal_eval(assignment.value)))

    def test_signed_self_and_neighbor_jacobians_match_finite_difference(self) -> None:
        records = _records()
        targets = pd.DataFrame(
            {"target_id": ["t1"], "seq_id": [1], "comp_id": ["SER"], "atom_id": ["OG"]}
        )
        observed = cache._support_dynamic_geometry(records, targets)
        carbon = cache.ELEMENTS.index("C")
        hydrogen = cache.ELEMENTS.index("H")
        nitrogen = cache.ELEMENTS.index("N")
        oxygen = cache.ELEMENTS.index("O")
        sulfur = cache.ELEMENTS.index("S")
        self.assertAlmostEqual(float(observed["distances"][0, oxygen]), 4.0)
        self.assertAlmostEqual(
            float(observed["distances"][0, carbon]), float(np.hypot(1.5, 4.0)), places=6
        )
        self.assertTrue(observed["distance_available"][0, [carbon, oxygen]].all())
        self.assertFalse(
            observed["distance_available"][0, [hydrogen, nitrogen, sulfur]].any()
        )
        self.assertTrue(
            np.array_equal(
                observed["distances"][0, [hydrogen, nitrogen, sulfur]],
                np.full(3, cache.DISTANCE_FILL, dtype=np.float32),
            )
        )
        self.assertEqual(int(observed["distance_neighbor_seq_ids"][0, carbon]), 2)
        self.assertEqual(int(observed["distance_neighbor_seq_ids"][0, oxygen]), 2)
        epsilon = 1.0e-7
        point_a = records["coordinate"][2]
        point_b = records["coordinate"][5]
        axis_a = records["coordinate"][1] - records["coordinate"][0]
        axis_b = records["coordinate"][4] - records["coordinate"][3]
        plus_a = _rotate(point_a, records["coordinate"][0], axis_a, epsilon)
        minus_a = _rotate(point_a, records["coordinate"][0], axis_a, -epsilon)

        def distance_a(point: np.ndarray) -> float:
            return float(np.linalg.norm(point - point_b))

        fd_self = (distance_a(plus_a) - distance_a(minus_a)) / (2.0 * epsilon)
        plus_b = _rotate(point_b, records["coordinate"][3], axis_b, epsilon)
        minus_b = _rotate(point_b, records["coordinate"][3], axis_b, -epsilon)

        def distance_b(point: np.ndarray) -> float:
            return float(np.linalg.norm(point_a - point))

        fd_neighbor = (distance_b(plus_b) - distance_b(minus_b)) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(observed["distance_self_jacobian"][0, oxygen, 0]), fd_self, places=6
        )
        self.assertAlmostEqual(
            float(observed["distance_neighbor_jacobian"][0, oxygen, 0]),
            fd_neighbor,
            places=6,
        )
        self.assertLess(float(observed["distance_self_jacobian"][0, oxygen, 0]), 0.0)
        self.assertGreater(float(observed["distance_neighbor_jacobian"][0, oxygen, 0]), 0.0)

    def test_lysine_chi2_self_jacobian_matches_finite_difference(self) -> None:
        records = {
            "name": np.asarray(["N", "CA", "CB", "CG", "CD", "O"]),
            "resname": np.asarray(["LYS"] * 5 + ["ALA"]),
            "chain": np.asarray(["A"] * 6),
            "seq_id": np.asarray([1, 1, 1, 1, 1, 2], dtype=np.int64),
            "element": np.asarray(["N", "C", "C", "C", "C", "O"]),
            "coordinate": np.asarray(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [1.5, 1.5, 0.0],
                    [2.5, 1.5, 1.0],
                    [3.0, 3.0, 2.0],
                ],
                dtype=np.float64,
            ),
            "residue_key": ((0, "A", 1, ""),) * 5 + ((0, "A", 2, ""),),
        }
        targets = pd.DataFrame(
            {"target_id": ["t"], "seq_id": [1], "comp_id": ["LYS"], "atom_id": ["CD"]}
        )
        observed = cache._support_dynamic_geometry(records, targets)
        oxygen = cache.ELEMENTS.index("O")
        epsilon = 1.0e-7
        point = records["coordinate"][4]
        origin = records["coordinate"][2]
        axis = records["coordinate"][3] - origin
        neighbor = records["coordinate"][5]
        plus = _rotate(point, origin, axis, epsilon)
        minus = _rotate(point, origin, axis, -epsilon)
        expected = (
            np.linalg.norm(plus - neighbor) - np.linalg.norm(minus - neighbor)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(observed["distance_self_jacobian"][0, oxygen, 1]),
            float(expected),
            places=6,
        )

    def test_cysteine_sulfur_hydrogen_is_a_chi1_descendant(self) -> None:
        records = {
            "name": np.asarray(["CA", "CB", "SG", "HG", "O"]),
            "resname": np.asarray(["CYS"] * 4 + ["ALA"]),
            "chain": np.asarray(["A"] * 5),
            "seq_id": np.asarray([1, 1, 1, 1, 2], dtype=np.int64),
            "element": np.asarray(["C", "C", "S", "H", "O"]),
            "coordinate": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [1.5, 1.8, 0.0],
                    [1.5, 1.8, 1.34],
                    [3.0, 3.0, 2.0],
                ],
                dtype=np.float64,
            ),
            "residue_key": ((0, "A", 1, ""),) * 4 + ((0, "A", 2, ""),),
        }
        targets = pd.DataFrame(
            {"target_id": ["t"], "seq_id": [1], "comp_id": ["CYS"], "atom_id": ["HG"]}
        )
        observed = cache._support_dynamic_geometry(records, targets)
        oxygen = cache.ELEMENTS.index("O")
        self.assertNotEqual(
            float(observed["distance_self_jacobian"][0, oxygen, 0]), 0.0
        )

    def test_missing_atom_is_available_false_and_filled_without_derivatives(self) -> None:
        records = _records()
        targets = pd.DataFrame(
            {"target_id": ["missing"], "seq_id": [1], "comp_id": ["SER"], "atom_id": ["HD1"]}
        )
        observed = cache._support_dynamic_geometry(records, targets)
        self.assertFalse(bool(observed["target_atom_available"][0]))
        self.assertFalse(observed["distance_available"][0].any())
        self.assertTrue(np.array_equal(observed["distances"][0], np.full(5, 10.0, np.float32)))
        self.assertTrue(np.array_equal(observed["distance_neighbor_seq_ids"][0], np.full(5, -1)))
        self.assertTrue(np.all(observed["distance_self_jacobian"] == 0.0))
        self.assertTrue(np.all(observed["distance_neighbor_jacobian"] == 0.0))

    def test_source_feature_read_is_column_limited_and_roster_is_exact(self) -> None:
        supports = list(cache.EXPECTED_K8_SUPPORT_IDS)
        frame = pd.DataFrame(
            {
                "entity_uid": ["e"] * 16 + ["other"] * 8,
                "target_id": ["t2"] * 8 + ["t1"] * 8 + ["held"] * 8,
                "seq_id": [2] * 8 + [1] * 8 + [999] * 8,
                "comp_id": ["ALA"] * 24,
                "atom_id": ["CA"] * 24,
                "support_id": supports * 3,
                "target_value": [99.0] * 24,
            }
        )
        with mock.patch.object(pd, "read_parquet", return_value=frame) as reader:
            result = cache._read_source_targets(Path("features.parquet"), "e")
        self.assertEqual(result["target_id"].tolist(), ["t1", "t2"])
        self.assertNotIn("target_value", result.columns)
        self.assertNotIn("target_value", reader.call_args.kwargs["columns"])
        self.assertEqual(
            tuple(reader.call_args.kwargs["columns"]),
            (
                "entity_uid",
                "target_id",
                "seq_id",
                "comp_id",
                "atom_id",
                "support_id",
            ),
        )

    def test_json_artifact_is_o_excl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            cache.write_json_new(path, {"one": 1})
            with self.assertRaises(FileExistsError):
                cache.write_json_new(path, {"one": 2})
            self.assertEqual(json.loads(path.read_text())["one"], 1)

    @mock.patch.object(cache, "EXPECTED_ENTITY_COUNT", 1)
    def test_feature_roster_rejects_target_bearing_parquet_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / cache.FEATURE_ROOT_RELATIVE / "bmr1.parquet"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"not opened after schema rejection")
            schema = mock.Mock(
                names=[
                    "entity_uid",
                    "split",
                    "observer_fold",
                    "support_id",
                    "target_value",
                ]
            )
            with mock.patch.object(cache.pq, "read_schema", return_value=schema):
                with self.assertRaisesRegex(ValueError, "target-bearing"):
                    cache._feature_roster(
                        root, [{"entity_uid": "e", "bmrb_id": "bmr1"}]
                    )

    @mock.patch.object(cache, "EXPECTED_ENTITY_COUNT", 1)
    @mock.patch.object(cache, "EXPECTED_FEATURE_ROW_COUNT", 1)
    @mock.patch.object(cache, "EXPECTED_TARGET_SUPPORT_ROWS", 32)
    @mock.patch.object(cache, "EXPECTED_TARGET_AVAILABLE_ROWS", 32)
    @mock.patch.object(cache, "EXPECTED_TARGET_MISSING_ROWS", 0)
    def test_freeze_binds_source_roster_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Bind the exact implementation bytes under the root-relative path.
            script = root / cache.SCRIPT_RELATIVE
            script.parent.mkdir(parents=True)
            shutil.copyfile(Path(cache.__file__), script)
            parent_path = root / cache.PARENT_RELATIVE
            parent_path.parent.mkdir(parents=True)
            parent_path.write_text(
                json.dumps(
                    {
                        "entities": [
                            {"entity_uid": "e", "bmrb_id": "bmr1", "split": "train", "observer_fold": "A"},
                            {"entity_uid": "held", "bmrb_id": "bmr2", "split": "test", "observer_fold": "A"},
                        ]
                    }
                )
            )
            feature = root / cache.FEATURE_ROOT_RELATIVE / "bmr1.parquet"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"source feature bytes")
            receipt_path = root / cache.PARENT_RECEIPT_RELATIVE
            receipt_path.parent.mkdir(parents=True)
            parent_source_path = root / cache.PARENT_SOURCE_COMMITMENT_RELATIVE
            parent_source_path.parent.mkdir(parents=True, exist_ok=True)
            parent_source_path.write_text(
                json.dumps(
                    {
                        "contract": "k32_support_asset_source_commitment_v1",
                        "target_values_read": False,
                        "source_gate_authorized": False,
                        "formal_evaluation_authorized": False,
                        "support_indices": list(cache.EXPECTED_SUPPORTS),
                        "support_count": 32,
                        "pair_count": 5184,
                    }
                )
            )
            outputs = []
            for support in cache.EXPECTED_SUPPORTS:
                relative = (
                    f"{cache.PARENT_ASSET_RELATIVE}/bmr1/bmr1_BioEmu_{support}.pdb"
                )
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"pdb-{support}".encode())
                outputs.append(
                    {
                        "entity_uid": "e",
                        "bmrb_id": "bmr1",
                        "support_id": f"BioEmu_{support}",
                        "support_index": support,
                        "output_relative_path": relative,
                        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                )
            receipt_path.write_text(
                json.dumps(
                    {
                        "contract": cache.PARENT_RECEIPT_CONTRACT,
                        "target_values_read": False,
                        "source_gate_authorized": False,
                        "formal_evaluation_authorized": False,
                        "source_commitment_sha256": cache.sha256_file(parent_source_path),
                        "outputs": outputs,
                    }
                )
            )
            for relative in (
                cache.CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE,
                cache.CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE,
                cache.CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE,
            ):
                bound = root / relative
                bound.parent.mkdir(parents=True, exist_ok=True)
                bound.write_text(json.dumps({"relative": relative}))
            k8_source = root / cache.CORRECTED_K8_SOURCE_COMMITMENT_RELATIVE
            k8_consumed = root / cache.CORRECTED_K8_CONSUMED_AUTHORIZATION_RELATIVE
            k8_claim = root / cache.CORRECTED_K8_EXTERNAL_CLAIM_RELATIVE
            decision_path = root / cache.CORRECTED_K8_DECISION_RELATIVE
            decision_path.write_text(
                json.dumps(
                    {
                        "contract": "corrected_k8_dynamic_coordinate_source_oof_check_v1",
                        "selected_for_k32_followup": True,
                        "source_commitment_sha256": cache.sha256_file(k8_source),
                        "consumed_authorization_sha256": cache.sha256_file(k8_consumed),
                        "external_claim_sha256": cache.sha256_file(k8_claim),
                    }
                )
            )
            backpressure_path = root / cache.CORRECTED_K8_BACKPRESSURE_RELATIVE
            backpressure_path.write_text(
                json.dumps(
                    {
                        "contract": "corrected_k8_job135711_backpressure_recovery_receipt_v1",
                        "backpressure_checks_passed": True,
                        "source_science_rerun": False,
                        "formal_metrics_opened": False,
                        "decision_sha256": cache.sha256_file(decision_path),
                        "source_commitment_sha256": cache.sha256_file(k8_source),
                    }
                )
            )
            plan_path = root / cache.PLAN_RELATIVE
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(
                    {
                        "contract": "atypemu-k32-dynamic-distance-source-cache-plan-v1",
                        "scope": {
                            "entity_count": 1,
                            "target_identity_count": cache.EXPECTED_FEATURE_ROW_COUNT,
                            "target_support_count": cache.EXPECTED_TARGET_SUPPORT_ROWS,
                            "support_count": 32,
                        },
                        "provenance": {
                            "parent_k32_receipt_sha256": cache.sha256_file(receipt_path),
                            "corrected_k8_decision_sha256": cache.sha256_file(decision_path),
                            "corrected_k8_backpressure_recovery_receipt_sha256": cache.sha256_file(
                                backpressure_path
                            ),
                        },
                        "acceptance": {
                            "target_values_read": False,
                            "source_gate_authorized": False,
                            "formal_evaluation_authorized": False,
                        },
                    }
                )
            )
            commitment_path = root / cache.SOURCE_COMMITMENT_RELATIVE
            args = argparse.Namespace(
                root=root,
                output=commitment_path,
                parent_commitment=Path(cache.PARENT_RELATIVE),
                parent_receipt=Path(cache.PARENT_RECEIPT_RELATIVE),
            )
            feature_record = {
                "bmrb_id": "bmr1",
                "entity_uid": "e",
                "relative_path": str(feature.relative_to(root)),
                "sha256": cache.sha256_file(feature),
                "row_count": 8,
            }
            with mock.patch.object(cache, "_feature_roster", return_value=[feature_record]):
                self.assertEqual(cache.freeze(args), 0)
                commitment = json.loads(commitment_path.read_text())
                self.assertEqual(commitment["entity_count"], 1)
                self.assertEqual(tuple(commitment["support_indices"]), cache.EXPECTED_SUPPORTS)
                self.assertEqual(len(commitment["pdb_files"]), 32)
                self.assertEqual(commitment["entities"][0]["entity_uid"], "e")
                self.assertFalse(
                    any("targets" in row["relative_path"] for row in commitment["features"])
                )
                with self.assertRaises(FileExistsError):
                    cache.freeze(args)
                tampered = dict(commitment)
                tampered["entity_count"] = 2
                tampered_path = root / "tampered.json"
                tampered_path.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, "exact count|canonical hash"):
                    cache.verify_commitment(root, tampered_path)


if __name__ == "__main__":
    unittest.main()
