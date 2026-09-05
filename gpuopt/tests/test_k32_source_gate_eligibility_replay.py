"""Synthetic adversarial checks for K32 source-cell eligibility receipts."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from gpuopt import check_k32_nested_k8_source_gate as checker
from gpuopt.candidates import k32_nested_k8_adapter as adapter
from gpuopt.source_gate_eligibility import build_eligibility_receipt


class EligibilityReplayTests(unittest.TestCase):
    def frame(self, entity: str, values: tuple[float, float, float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entity_uid": [entity] * 3,
                "target_id": [f"{entity}:target:{name}" for name in ("a", "b", "c")],
                "seq_id": [1, 2, 3],
                "comp_id": ["ALA", "GLY", "SER"],
                "atom_id": ["A", "B", "C"],
                "target_value": values,
                "split": ["train"] * 3,
                "observer_fold": ["A"] * 3,
            }
        )

    def half(
        self,
        frame: pd.DataFrame,
        *,
        role: str,
        normalization: adapter.SourceNormalization | None = None,
    ) -> adapter.SourceHalfInputs:
        frozen = ("A", "B", "C")
        selected = frame.loc[frame["atom_id"].isin(("A", "B"))].reset_index(
            drop=True
        )
        receipt = build_eligibility_receipt(
            frame,
            selected,
            frozen_atom_ids=frozen,
            fold="A",
            held_half=0,
            role=role,
        )
        surface = adapter.Surface(
            selected["target_id"].to_numpy(), ("support",), (), None
        )
        targets = adapter.source_targets(
            {"frame": selected, "source_eligibility_receipt": receipt},
            surface,
            np.zeros((len(selected), 1), dtype=np.float32),
            normalization=normalization,
        )
        return adapter.SourceHalfInputs(
            frame=selected,
            surface=surface,
            anchor=np.zeros((len(selected), 1), dtype=np.float32),
            context=(np.zeros(len(selected), dtype=np.int64),) * 3,
            targets=targets,
            eligibility_receipt=receipt,
        )

    def test_replay_rejects_omission_substitution_and_source_tampering(self) -> None:
        train_entities = {"bmrb:1:entity:1", "bmrb:2:entity:1"}
        evaluation_entities = {"bmrb:3:entity:1", "bmrb:4:entity:1"}
        frames = {
            "bmrb:1:entity:1": self.frame("bmrb:1:entity:1", (0.0, 1.0, 8.0)),
            "bmrb:2:entity:1": self.frame("bmrb:2:entity:1", (1.0, 3.0, 8.0)),
            "bmrb:3:entity:1": self.frame("bmrb:3:entity:1", (2.0, 4.0, 7.0)),
            "bmrb:4:entity:1": self.frame("bmrb:4:entity:1", (4.0, 8.0, 7.0)),
        }
        train = self.half(
            pd.concat([frames[key] for key in sorted(train_entities)], ignore_index=True),
            role="train",
        )
        evaluation = self.half(
            pd.concat(
                [frames[key] for key in sorted(evaluation_entities)], ignore_index=True
            ),
            role="evaluation",
            normalization=train.targets.normalization,
        )
        sealed_train, sealed_evaluation = adapter.seal_cell_eligibility_receipts(
            train, evaluation
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files: dict[str, str] = {}
            path_frames: dict[Path, pd.DataFrame] = {}
            for entity, frame in frames.items():
                relative = (
                    "data/all_atom_observer_v1/targets/"
                    f"bmr{entity.split(':')[1]}.parquet"
                )
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entity.encode())
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                path_frames[path.resolve()] = frame

            def read_target(path: object, *, columns: tuple[str, ...]) -> pd.DataFrame:
                self.assertEqual(columns, checker.SOURCE_TARGET_COLUMNS)
                return path_frames[Path(path).resolve()].copy()

            kwargs = {
                "fold": "A",
                "held_half": 0,
                "train_entities": train_entities,
                "evaluation_entities": evaluation_entities,
                "frozen_atom_ids": ("A", "B", "C"),
                "sealed_train_receipt": sealed_train,
                "sealed_evaluation_receipt": sealed_evaluation,
            }
            with mock.patch.object(checker.pd, "read_parquet", side_effect=read_target):
                replayed = checker.replay_committed_cell_eligibility(
                    root, {"files": files}, **kwargs
                )
            self.assertEqual(tuple(map(len, replayed)), (4, 4))

            omitted = dict(sealed_train)
            omitted["input_row_count"] -= 1
            with (
                mock.patch.object(checker.pd, "read_parquet", side_effect=read_target),
                self.assertRaisesRegex(ValueError, "receipt replay mismatch"),
            ):
                checker.replay_committed_cell_eligibility(
                    root, {"files": files},
                    **(kwargs | {"sealed_train_receipt": omitted}),
                )

            substituted = dict(sealed_evaluation)
            substituted["normalization_provenance"] = dict(
                sealed_evaluation["normalization_provenance"]
            )
            substituted["normalization_provenance"]["global_scale"] = 99.0
            with (
                mock.patch.object(checker.pd, "read_parquet", side_effect=read_target),
                self.assertRaisesRegex(ValueError, "receipt replay mismatch"),
            ):
                checker.replay_committed_cell_eligibility(
                    root, {"files": files},
                    **(kwargs | {"sealed_evaluation_receipt": substituted}),
                )

            tampered = root / "data/all_atom_observer_v1/targets/bmr1.parquet"
            tampered.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "target binding mismatch"):
                checker.committed_source_target_frame(
                    root, {"files": files}, entity_uid="bmrb:1:entity:1", fold="A"
                )

    def test_checker_has_no_candidate_adapter_import(self) -> None:
        self.assertNotIn("k32_nested_k8_adapter", inspect.getsource(checker))


if __name__ == "__main__":
    unittest.main()
