from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from gpuopt import check_k32_dynamic_distance_cache as checker


ROOT = Path(__file__).resolve().parents[2]


def _pdb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  SER A   1       0.000   0.000   0.000  1.00  0.00           C  ",
                "ATOM      2  CB  SER A   1       1.500   0.000   0.000  1.00  0.00           C  ",
                "ATOM      3  OG  SER A   1       1.500   1.500   0.000  1.00  0.00           O  ",
                "ATOM      4  CA  SER A   2       0.000   0.000   4.000  1.00  0.00           C  ",
                "ATOM      5  CB  SER A   2       1.500   0.000   4.000  1.00  0.00           C  ",
                "ATOM      6  OG  SER A   2       1.500   1.500   4.000  1.00  0.00           O  ",
                "TER",
                "END",
            ]
        )
        + "\n"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(root: Path) -> tuple[argparse.Namespace, pd.DataFrame, Path]:
    entity = {
        "entity_uid": "entity-1",
        "bmrb_id": "bmr1",
        "split": "train",
        "observer_fold": "A",
    }
    supports = checker.EXPECTED_SUPPORTS
    feature_frame = pd.DataFrame(
        {
            "entity_uid": [entity["entity_uid"]] * 8,
            "target_id": ["target-1"] * 8,
            "seq_id": [1] * 8,
            "comp_id": ["SER"] * 8,
            "atom_id": ["OG"] * 8,
            "support_id": list(checker.SOURCE_SUPPORT_IDS),
        }
    )
    feature_path = root / "data/all_atom_observer_v1/features/bmr1.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_bytes(b"synthetic source feature parquet placeholder")
    script_path = root / checker.SCRIPT_RELATIVE
    script_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "gpuopt/materialize_k32_dynamic_distance_cache.py",
        script_path,
    )
    parent = root / checker.PARENT_RELATIVE
    _write_json(parent, {"entities": [entity]})

    parent_source = root / checker.PARENT_SOURCE_RELATIVE
    _write_json(
        parent_source,
        {
            "contract": checker.PARENT_SOURCE_CONTRACT,
            "target_values_read": False,
            "source_gate_authorized": False,
            "formal_evaluation_authorized": False,
            "support_indices": list(supports),
            "support_count": 32,
            "pair_count": 5184,
        },
    )
    parent_receipt = root / checker.PARENT_RECEIPT_RELATIVE
    parent_outputs = []
    pdb_manifest = []
    for support in supports:
        relative = (
            f"data/k32_complete_coordinate_supports_v4/bmr1/"
            f"bmr1_BioEmu_{support}.pdb"
        )
        path = root / relative
        _pdb(path)
        digest = _sha(path)
        parent_outputs.append(
            {
                "entity_uid": entity["entity_uid"],
                "bmrb_id": entity["bmrb_id"],
                "support_id": f"BioEmu_{support}",
                "support_index": support,
                "output_relative_path": relative,
                "output_sha256": digest,
            }
        )
        pdb_manifest.append(
            {
                "entity_uid": entity["entity_uid"],
                "bmrb_id": entity["bmrb_id"],
                "support_id": f"BioEmu_{support}",
                "support_index": support,
                "relative_path": relative,
                "sha256": digest,
            }
        )
    _write_json(
        parent_receipt,
        {
            "contract": checker.PARENT_RECEIPT_CONTRACT,
            "target_values_read": False,
            "source_gate_authorized": False,
            "formal_evaluation_authorized": False,
            "source_commitment_sha256": _sha(parent_source),
            "outputs": parent_outputs,
        },
    )

    # All bound prerequisite files are tiny synthetic receipts.  The checker
    # validates their hashes through the source commitment and validates the
    # plan's parent-receipt binding independently.
    prereq_paths = {}
    for relative in checker.BOUND_PREREQUISITES:
        path = root / relative
        prereq_paths[relative] = path
        if relative == checker.PLAN_RELATIVE or path.exists():
            continue
        _write_json(path, {"synthetic": relative})
    plan = root / checker.PLAN_RELATIVE
    _write_json(
        plan,
        {
            "contract": checker.PLAN_CONTRACT,
            "scope": {
                "entity_count": 1,
                "support_count": 32,
                "target_identity_count": 1,
                "target_support_count": 32,
            },
            "provenance": {
                "parent_k32_receipt_sha256": _sha(parent_receipt),
                "corrected_k8_decision_sha256": _sha(
                    root / checker.CORRECTED_K8_DECISION_RELATIVE
                ),
                "corrected_k8_backpressure_recovery_receipt_sha256": _sha(
                    root / checker.CORRECTED_K8_BACKPRESSURE_RELATIVE
                ),
            },
            "acceptance": {
                "target_values_read": False,
                "source_gate_authorized": False,
                "formal_evaluation_authorized": False,
            },
        },
    )

    source = {
        "contract": checker.SOURCE_CONTRACT,
        "target_values_read": False,
        "source_gate_authorized": False,
        "formal_evaluation_authorized": False,
        "support_indices": list(supports),
        "support_ids": list(checker.EXPECTED_SUPPORT_IDS),
        "entity_count": 1,
        "expected_feature_row_count": 1,
        "expected_target_support_rows": 32,
        "expected_target_available_rows": 32,
        "expected_target_missing_rows": 0,
        "feature_file_count": 1,
        "pdb_file_count": 32,
        "script_relative_path": checker.SCRIPT_RELATIVE,
        "script_sha256": _sha(root / checker.SCRIPT_RELATIVE),
        "parent_commitment_relative_path": checker.PARENT_RELATIVE,
        "parent_commitment_sha256": _sha(parent),
        "parent_receipt_relative_path": checker.PARENT_RECEIPT_RELATIVE,
        "parent_receipt_sha256": _sha(parent_receipt),
        "bound_files": [
            {"relative_path": relative, "sha256": _sha(path)}
            for relative, path in prereq_paths.items()
        ],
        "entities": [entity],
        "features": [
            {
                "bmrb_id": "bmr1",
                "entity_uid": "entity-1",
                "relative_path": "data/all_atom_observer_v1/features/bmr1.parquet",
                "sha256": _sha(feature_path),
                "row_count": 8,
            }
        ],
        "pdb_files": pdb_manifest,
    }
    source["commitment_sha256"] = checker.canonical_sha256(source)
    source_path = root / checker.SOURCE_RELATIVE
    _write_json(source_path, source)

    targets = checker._target_rows(feature_frame, entity["entity_uid"])
    output_root = root / checker.OUTPUT_ROOT_RELATIVE
    output_root.mkdir(parents=True, exist_ok=True)
    one = checker._replay_geometry(
        checker._pdb_records(root / pdb_manifest[0]["relative_path"]), targets
    )
    arrays = {
        key: np.stack(
            [
                checker._replay_geometry(
                    checker._pdb_records(root / row["relative_path"]), targets
                )[key]
                for row in pdb_manifest
            ],
            axis=1,
        )
        for key in one
    }
    payload = {
        "entity_uid": np.asarray(["entity-1"], dtype=str),
        "bmrb_id": np.asarray(["bmr1"], dtype=str),
        "target_ids": np.asarray(["target-1"], dtype=str),
        "support_ids": np.asarray(checker.EXPECTED_SUPPORT_IDS, dtype=str),
        "support_indices": np.asarray(supports, dtype=np.int16),
        "seq_ids": np.asarray([1], dtype=np.int32),
        "comp_ids": np.asarray(["SER"], dtype=str),
        "atom_ids": np.asarray(["OG"], dtype=str),
        "element_order": np.asarray(checker.ELEMENTS, dtype=str),
        **arrays,
    }
    output = output_root / "bmr1.npz"
    np.savez_compressed(output, **payload)
    output_record = {
        "entity_uid": "entity-1",
        "bmrb_id": "bmr1",
        "output_relative_path": f"{checker.OUTPUT_ROOT_RELATIVE}/bmr1.npz",
        "output_sha256": _sha(output),
        "target_count": 1,
        "target_support_row_count": 32,
        "target_available_count": 32,
        "target_missing_count": 0,
        "target_ids_sha256": checker._stable_ids_hash(np.asarray(["target-1"])),
        "support_ids": list(checker.EXPECTED_SUPPORT_IDS),
        "array_shapes": {
            "nearest_interresidue_distances_angstrom": [1, 32, 5],
            "distance_self_jacobian": [1, 32, 5, 4],
            "distance_neighbor_jacobian": [1, 32, 5, 4],
            "distance_neighbor_seq_ids": [1, 32, 5],
            "target_atom_available": [1, 32],
            "distance_available": [1, 32, 5],
        },
    }
    receipt = root / checker.RECEIPT_RELATIVE
    _write_json(
        receipt,
        {
            "contract": checker.RECEIPT_CONTRACT,
            "target_values_read": False,
            "source_gate_authorized": False,
            "formal_evaluation_authorized": False,
            "source_commitment_sha256": _sha(source_path),
            "parent_receipt_sha256": _sha(parent_receipt),
            "support_indices": list(supports),
            "support_ids": list(checker.EXPECTED_SUPPORT_IDS),
            "entity_count": 1,
            "output_count": 1,
            "target_count": 1,
            "target_support_row_count": 32,
            "target_available_count": 32,
            "target_missing_count": 0,
            "outputs": [output_record],
        },
    )
    args = argparse.Namespace(
        root=root,
        source_commitment=source_path,
        receipt=receipt,
    )
    return args, feature_frame, output


class DynamicDistanceCacheCheckerTests(unittest.TestCase):
    def test_source_scope_must_equal_parent_entity_fold(self) -> None:
        entity = {"entity_uid": "e", "observer_fold": "A"}
        frame = pd.DataFrame(
            {
                "entity_uid": ["e"] * 8,
                "split": ["train"] * 8,
                "observer_fold": ["A"] * 8,
                "support_id": list(checker.SOURCE_SUPPORT_IDS),
            }
        )
        with mock.patch.object(checker.pd, "read_parquet", return_value=frame):
            checker.verify_source_scope(Path("synthetic.parquet"), entity)
        frame["observer_fold"] = "B"
        with mock.patch.object(checker.pd, "read_parquet", return_value=frame):
            with self.assertRaisesRegex(ValueError, "fold scope mismatch"):
                checker.verify_source_scope(Path("synthetic.parquet"), entity)

    def test_source_read_requests_only_the_six_identity_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "entity_uid": ["e"] * 8,
                "target_id": ["t"] * 8,
                "seq_id": [1] * 8,
                "comp_id": ["SER"] * 8,
                "atom_id": ["OG"] * 8,
                "support_id": list(checker.SOURCE_SUPPORT_IDS),
            }
        )
        with mock.patch.object(checker.pd, "read_parquet", return_value=frame) as reader:
            checker.read_source_targets(Path("synthetic.parquet"), "e")
        self.assertEqual(tuple(reader.call_args.kwargs["columns"]), checker.IDENTITY_COLUMNS)
        self.assertNotIn("target_value", reader.call_args.kwargs["columns"])

    def test_tiny_artifacts_pass_and_hash_count_path_and_npz_tampering_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, frame, output = _fixture(root)
            patches = {
                "EXPECTED_ENTITY_COUNT": 1,
                "EXPECTED_FEATURE_ROW_COUNT": 1,
                "EXPECTED_TARGET_SUPPORT_ROWS": 32,
                "EXPECTED_TARGET_AVAILABLE_ROWS": 32,
                "EXPECTED_TARGET_MISSING_ROWS": 0,
            }
            parquet_frame = frame.assign(split="train", observer_fold="A")

            def read_parquet(_path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
                return parquet_frame.loc[:, list(columns)].copy()

            with mock.patch.multiple(checker, **patches), mock.patch.object(
                checker.pd, "read_parquet", side_effect=read_parquet
            ):
                summary = checker.check(args)
                self.assertTrue(summary["passed"])
                self.assertEqual(summary["verified_target_support_rows"], 32)

                receipt = Path(args.receipt)
                original = json.loads(receipt.read_text())
                count_tampered = {**original, "target_count": 2}
                receipt.write_text(json.dumps(count_tampered))
                with self.assertRaisesRegex(ValueError, "receipt exact count mismatch"):
                    checker.check(args)

                path_tampered = json.loads(receipt.read_text())
                path_tampered["target_count"] = 1
                path_tampered["outputs"][0]["output_relative_path"] = (
                    f"{checker.OUTPUT_ROOT_RELATIVE}/wrong.npz"
                )
                receipt.write_text(json.dumps(path_tampered))
                with self.assertRaisesRegex(ValueError, "noncanonical cache output path"):
                    checker.check(args)
                restored = json.loads(receipt.read_text())
                restored["outputs"][0]["output_relative_path"] = (
                    f"{checker.OUTPUT_ROOT_RELATIVE}/bmr1.npz"
                )
                receipt.write_text(json.dumps(restored))

                # The receipt still contains the original hash, so even a
                # valid-looking NPZ byte mutation is rejected before replay.
                output.write_bytes(output.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "cache output hash mismatch"):
                    checker.check(args)

    def test_npz_dtype_and_unavailable_invariants_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _args, frame, output = _fixture(root)
            entity = {"entity_uid": "entity-1", "bmrb_id": "bmr1"}
            targets = checker._target_rows(frame, "entity-1")
            with mock.patch.object(checker.pd, "read_parquet", return_value=frame):
                checker.verify_npz(output, entity, targets)
            arrays = dict(np.load(output, allow_pickle=False))
            arrays["support_indices"] = arrays["support_indices"].astype(np.int32)
            np.savez_compressed(output, **arrays)
            with mock.patch.object(checker.pd, "read_parquet", return_value=frame):
                with self.assertRaisesRegex(ValueError, "NPZ dtype mismatch"):
                    checker.verify_npz(output, entity, targets)


if __name__ == "__main__":
    unittest.main()
