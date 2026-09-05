#!/usr/bin/env python3
"""Focused checks for the CCC-free complete-coordinate Jacobian loss path."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from gpuopt.candidates import corrected_k8_dynamic_coordinate as candidate
from gpuopt import check_corrected_k8_dynamic_coordinate_source_gate as source_checker
from gpuopt import freeze_corrected_k8_dynamic_coordinate_source_gate as source_freezer
from gpuopt import run_corrected_k8_dynamic_coordinate_source_gate as source_gate
from gpuopt.run_corrected_k8_dynamic_coordinate_source_gate import (
    consume_authorization,
)
from gpuopt.source_gate_eligibility import build_eligibility_receipt


def external_authorization_repo(
    root: Path, authorization: Path, authorization_ref: str
) -> tuple[Path, str]:
    repository = root / "external-authorizations.git"
    subprocess.run(["git", "init", "--bare", "-q", repository], check=True)
    blob = subprocess.run(
        ["git", f"--git-dir={repository}", "hash-object", "-w", authorization],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            f"--git-dir={repository}",
            "update-ref",
            authorization_ref,
            blob,
            "0" * 40,
        ],
        check=True,
    )
    return repository, blob


class GeometryProbe(nn.Module):
    def forward(self, esm, categorical, numeric, geometry, torsion):
        del esm, categorical, numeric, torsion
        return geometry[:, candidate.DYNAMIC_DISTANCE_INDICES[0]]


class LossGradientPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_gate.load_science_api()

    def test_source_coordinate_audit_binds_committed_pdb_and_state_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb_dir = root / "data/BioEmu/bmr1"
            pdb_dir.mkdir(parents=True)
            (pdb_dir / "bmr1_BioEmu_1.pdb").write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000\n"
            )
            base = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
            output = root / "coordinates.npz"

            def write(atom_state, emitted_base=base):
                np.savez(
                    output,
                    conditioned_coordinates=base
                    + np.asarray([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0]], np.float32),
                    no_evidence_coordinates=emitted_base,
                    atom_mask=np.ones(2, dtype=bool),
                    atom_state_index=np.asarray(atom_state, np.int32),
                    atom_name=np.asarray(["N", "CA"]),
                    atom_element=np.asarray(["N", "C"]),
                    atom_seq_id=np.asarray([1, 1], np.int32),
                    state_entity_uid=np.asarray(["uid"]),
                    state_support_id=np.asarray(["BioEmu_1"]),
                )

            kwargs = {
                "root": root,
                "expected_entities": {"uid"},
                "support_ids": {"uid": ["BioEmu_1"]},
                "bmrb_ids": {"uid": "bmr1"},
            }
            write([0, 0])
            self.assertEqual(
                source_checker.audit_coordinate_output(output, **kwargs)["state_count"],
                1,
            )
            write([0, 1])
            with self.assertRaisesRegex(ValueError, "atom-state coverage"):
                source_checker.audit_coordinate_output(output, **kwargs)
            write([0, 0], emitted_base=base + 0.1)
            with self.assertRaisesRegex(ValueError, "differs from committed PDB"):
                source_checker.audit_coordinate_output(output, **kwargs)

    def test_source_gate_support_roster_is_exact_k8(self):
        self.assertEqual(
            source_gate.EXPECTED_SUPPORT_IDS,
            tuple(
                f"BioEmu_{index}"
                for index in (1, 126, 251, 376, 501, 626, 751, 876)
            ),
        )

    def test_all_model_row_arrays_share_the_eligibility_subset(self):
        self.assertEqual(candidate.SOURCE_ROW_ARRAYS, source_gate.ROW_ARRAYS)

    def test_source_manifest_inventory_matches_freezer_runner_and_checker(self):
        self.assertEqual(
            set(source_freezer.STATIC_FILES), source_gate.STATIC_COMMITTED_FILES
        )
        self.assertEqual(
            source_gate.STATIC_COMMITTED_FILES,
            source_checker.STATIC_COMMITTED_FILES,
        )
        root = Path(__file__).resolve().parents[2]
        data_commitment = json.loads(
            (root / "data/all_atom_observer_v1/commitment.json").read_text()
        )
        freezer_paths, _entities = source_freezer.collect_source_paths(root)
        freezer_manifest = {str(path.relative_to(root)) for path in freezer_paths}
        self.assertEqual(
            freezer_manifest,
            source_gate.expected_source_manifest(root, data_commitment),
        )
        self.assertEqual(
            freezer_manifest,
            source_checker.independent_expected_source_manifest(
                root, data_commitment
            ),
        )

    def test_source_halves_keep_sequence_clusters_intact(self):
        entities = [
            {"sequence_cluster_id": cluster}
            for cluster in ("shared", "solo_a", "shared", "solo_b")
        ]
        assignment = source_gate.sequence_cluster_halves(entities)
        self.assertEqual(assignment[0], assignment[2])
        self.assertEqual(set(assignment.values()), {0, 1})

    def test_source_label_eligibility_is_prediction_independent(self):
        frame = pd.DataFrame(
            {
                "atom_id": ["A", "A", "B", "B"],
                "target_value": [1.0, 1.0, 0.0, 1.0],
            }
        )
        for prediction in (
            np.asarray([1.0, 1.0, 0.0, 1.0]),
            np.asarray([0.0, 2.0, 0.5, 0.5]),
        ):
            _, per_label = source_gate.macro_atom_id_ccc(
                frame, prediction, ["A", "B"]
            )
            self.assertEqual(set(per_label), {"B"})

    def test_atom_family_diagnostic_is_unweighted_over_atom_ids(self):
        diagnostics = source_gate.atom_family_macro_diagnostics(
            {"C": 1.0, "CA": 0.0, "H": 0.25, "N": -0.5}
        )
        self.assertEqual(diagnostics, {"C": 0.5, "H": 0.25, "N": -0.5})

    def test_uniform_q_factorial_is_computed_but_not_used_for_gate(self):
        source_checker.verify_control_factorial(Path(source_gate.__file__))
        source = inspect.getsource(source_gate.main)
        self.assertIn("fixed_uniform_q=True", source)
        self.assertIn('part["uniform_q"]', source)
        gate_block = source.split('result["gate"] =', maxsplit=1)[1].split(
            'result["gate"]["pass"]', maxsplit=1
        )[0]
        self.assertNotIn("uniform_q", gate_block)

    def test_eligibility_filters_rows_before_model_phase_and_replays_independently(self):
        frame = pd.DataFrame(
            {
                "entity_uid": ["e1", "e2", "e1", "e2", "e1"],
                "target_id": ["a1", "a2", "b1", "b2", "c1"],
                "atom_id": ["A", "A", "B", "B", "C"],
                "target_value": [0.0, 1.0, 2.0, 2.0, 3.0],
            }
        )
        values = {
            "frame": frame,
            "entity_index": np.asarray([0, 1, 0, 1, 0], dtype=np.int64),
            "entities": [{"entity_uid": "e1"}, {"entity_uid": "e2"}],
            "residue_index": np.arange(len(frame), dtype=np.int64),
            "residue_keys": [("row", index) for index in range(len(frame))],
            "distance_neighbor_residue": np.full(
                (len(frame), candidate.SUPPORT_COUNT, 5), -1, dtype=np.int64
            ),
        }
        for key in set(source_gate.ROW_ARRAYS) - {
            "distance_neighbor_residue",
            "entity_index",
            "residue_index",
        }:
            values[key] = np.arange(len(frame), dtype=np.float32)[:, None]
        selected = source_gate.restrict_source_subset_eligibility(
            values,
            frozen_atom_ids=["A", "B", "C"],
            fold="A",
            held_half=0,
            role="observer_training",
        )
        self.assertEqual(selected["frame"]["target_id"].tolist(), ["a1", "a2"])
        for key in source_gate.ROW_ARRAYS:
            self.assertEqual(len(selected[key]), 2, key)
        candidate.require_source_eligibility(selected)
        independently_selected, independent_receipt = (
            source_checker.independent_eligibility_receipt(
                frame,
                frozen_atom_ids=["A", "B", "C"],
                fold="A",
                held_half=0,
                role="observer_training",
            )
        )
        self.assertEqual(independently_selected["target_id"].tolist(), ["a1", "a2"])
        self.assertEqual(
            selected["source_eligibility_receipt"], independent_receipt
        )
        misaligned = dict(selected)
        misaligned["geometry"] = selected["geometry"][:1]
        with self.assertRaisesRegex(ValueError, "row-alignment mismatch"):
            candidate.require_source_eligibility(misaligned)
        selected["frame"].loc[0, "target_value"] = 99.0
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            candidate.require_source_eligibility(selected)

    def test_every_model_phase_has_a_bound_eligibility_guard(self):
        source_checker.verify_model_phase_eligibility_guards(
            Path(candidate.__file__)
        )

    def test_actual_entrypoint_consumes_synthetic_authorization_before_science(self):
        class ScienceReadAttempt(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / ".auto/frozen/all_label_inventory.json"
            data_commitment = root / "data/all_atom_observer_v1/commitment.json"
            audit = root / (
                "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/"
                "post_handoff_audit.json"
            )
            inventory.parent.mkdir(parents=True)
            data_commitment.parent.mkdir(parents=True)
            audit.parent.mkdir(parents=True)
            inventory.write_text(json.dumps({"eligible_atom_ids": ["A"]}))
            data_commitment.write_text(json.dumps({"entities": []}))
            canonical_audit = Path(__file__).resolve().parents[2] / (
                "reports/experiments/job134710_k8_dynamic_coordinate_source_gate/"
                "post_handoff_audit.json"
            )
            audit.write_bytes(canonical_audit.read_bytes())
            source_commitment = root / "source_commitment.json"
            source_commitment.write_text(
                json.dumps(
                    {
                        "contract": source_gate.COMMITMENT_CONTRACT,
                        "invalidated_parent_job": "134710",
                        "invalidated_parent_disposition": "INVALIDATED",
                        "canonical_invalidation_sha256": (
                            source_gate.CANONICAL_INVALIDATION_SHA256
                        ),
                        "formal_evaluation_authorized": False,
                        "target_values_opened": False,
                        "source_entity_count": 0,
                        "support_ids": list(source_gate.EXPECTED_SUPPORT_IDS),
                        "files": {
                            str(inventory.relative_to(root)): source_gate.sha256_file(
                                inventory
                            ),
                            str(data_commitment.relative_to(root)): (
                                source_gate.sha256_file(data_commitment)
                            ),
                            str(audit.relative_to(root)): source_gate.sha256_file(
                                audit
                            ),
                        },
                    }
                )
            )
            source_commitment_sha256 = source_gate.sha256_file(source_commitment)
            authorization = root / "authorization.json"
            authorization_ref = (
                "refs/atypemu-authorizations/corrected-k8/job-12345"
            )
            authorization.write_text(
                json.dumps(
                    {
                        "contract": source_gate.AUTHORIZATION_CONTRACT,
                        "source_commitment_sha256": source_commitment_sha256,
                        "epochs": source_gate.EXPECTED_EPOCHS,
                        "steps": source_gate.EXPECTED_STEPS,
                        "authorized": True,
                        "container_image_sha256": "b" * 64,
                        "slurm_job_id": "12345",
                        "authorization_ref": authorization_ref,
                    }
                )
            )
            authorization_bytes = authorization.read_bytes()
            authorization_blob = hashlib.sha1(  # noqa: S324
                f"blob {len(authorization_bytes)}\0".encode() + authorization_bytes
            ).hexdigest()
            authorization_repo, stored_blob = external_authorization_repo(
                root, authorization, authorization_ref
            )
            self.assertEqual(stored_blob, authorization_blob)
            consumed = root / "consumed.json"
            external_claim = root / "external_claim.json"
            args = Namespace(
                root=root,
                output=root / "result.json",
                source_commitment=source_commitment,
                authorization=authorization,
                consumed_authorization=consumed,
                external_authorization_claim=external_claim,
                authorization_git_blob=authorization_blob,
                authorization_ref=authorization_ref,
                container_image_sha256="b" * 64,
                epochs=source_gate.EXPECTED_EPOCHS,
                steps=source_gate.EXPECTED_STEPS,
            )
            with (
                mock.patch.object(source_gate, "parse_args", return_value=args),
                mock.patch.object(
                    source_gate,
                    "STATIC_COMMITTED_FILES",
                    {
                        str(inventory.relative_to(root)),
                        str(data_commitment.relative_to(root)),
                        str(audit.relative_to(root)),
                    },
                ),
                mock.patch.object(
                    source_gate,
                    "EXTERNAL_AUTHORIZATION_GIT_DIR",
                    authorization_repo,
                ),
                mock.patch.object(source_gate, "load_science_api"),
                mock.patch.object(
                    source_gate,
                    "verify_support_structure_receipt",
                    return_value={"preflight": True},
                ),
                mock.patch.object(
                    source_gate,
                    "load_fold",
                    side_effect=ScienceReadAttempt("science read sentinel"),
                ),
                self.assertRaisesRegex(ScienceReadAttempt, "science read sentinel"),
                mock.patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}),
            ):
                source_gate.main()
            self.assertTrue(consumed.is_file())
            self.assertTrue(external_claim.is_file())
            self.assertEqual(
                json.loads(consumed.read_text())["contract"],
                source_gate.CONSUMED_CONTRACT,
            )

    def test_source_authorization_precedes_source_file_opening(self):
        source = inspect.getsource(source_gate.main)
        self.assertLess(
            source.index("consume_authorization("),
            source.index("verify_source_commitment("),
        )
        self.assertLess(
            source.index("verify_source_commitment("),
            source.index("load_science_api()"),
        )
        self.assertLess(
            source.index("load_science_api()"),
            source.index("inventory = json.loads("),
        )
        self.assertLess(
            source.index("verify_source_manifest_complete("),
            source.index("values = load_fold("),
        )

    def test_freezer_never_deserializes_source_targets(self):
        source = inspect.getsource(source_freezer)
        self.assertNotIn("read_parquet", source)
        self.assertNotIn("pyarrow", source)
        self.assertNotIn("import pandas", source)

    def test_source_authorization_is_consumed_once_before_science(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / "authorization.json"
            consumed = root / "consumed.json"
            payload = {
                "contract": source_gate.AUTHORIZATION_CONTRACT,
                "source_commitment_sha256": "a" * 64,
                "epochs": 1024,
                "steps": 100,
                "authorized": True,
                "container_image_sha256": "b" * 64,
                "slurm_job_id": "12345",
                "authorization_ref": (
                    "refs/atypemu-authorizations/corrected-k8/job-12345"
                ),
            }
            authorization.write_text(json.dumps(payload))
            expected_authorization = hashlib.sha256(
                authorization.read_bytes()
            ).hexdigest()
            authorization_git_blob = hashlib.sha1(  # noqa: S324
                f"blob {len(authorization.read_bytes())}\0".encode()
                + authorization.read_bytes()
            ).hexdigest()
            authorization_ref = (
                "refs/atypemu-authorizations/corrected-k8/job-12345"
            )
            authorization_repo, stored_blob = external_authorization_repo(
                root, authorization, authorization_ref
            )
            self.assertEqual(stored_blob, authorization_git_blob)
            external_claim = root / "external_claim.json"
            (
                actual_authorization,
                consumed_hash,
                external_claim_sha256,
                external_claim_git_blob,
            ) = consume_authorization(
                authorization,
                consumed,
                external_claim,
                source_commitment_sha256="a" * 64,
                epochs=1024,
                steps=100,
                authorization_git_blob=authorization_git_blob,
                authorization_ref=authorization_ref,
                container_image_sha256="b" * 64,
                slurm_job_id="12345",
                authorization_git_dir=authorization_repo,
            )
            self.assertEqual(actual_authorization, expected_authorization)
            self.assertEqual(
                external_claim_sha256,
                hashlib.sha256(external_claim.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                subprocess.run(
                    [
                        "git",
                        f"--git-dir={authorization_repo}",
                        "rev-parse",
                        f"{authorization_ref}^{{blob}}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                external_claim_git_blob,
            )
            self.assertEqual(
                consumed_hash, hashlib.sha256(consumed.read_bytes()).hexdigest()
            )
            with self.assertRaisesRegex(ValueError, "ref is not fresh"):
                consume_authorization(
                    authorization,
                    consumed,
                    external_claim,
                    source_commitment_sha256="a" * 64,
                    epochs=1024,
                    steps=100,
                    authorization_git_blob=authorization_git_blob,
                    authorization_ref=(
                        "refs/atypemu-authorizations/corrected-k8/job-12345"
                    ),
                    container_image_sha256="b" * 64,
                    slurm_job_id="12345",
                    authorization_git_dir=authorization_repo,
                )
            with self.assertRaisesRegex(ValueError, "authorization mismatch: slurm_job_id"):
                consume_authorization(
                    authorization,
                    root / "different-job-consumed.json",
                    root / "different-job-claim.json",
                    source_commitment_sha256="a" * 64,
                    epochs=1024,
                    steps=100,
                    authorization_git_blob=authorization_git_blob,
                    authorization_ref=(
                        "refs/atypemu-authorizations/corrected-k8/job-12346"
                    ),
                    container_image_sha256="b" * 64,
                    slurm_job_id="12346",
                    authorization_git_dir=authorization_repo,
                )

    def test_runner_process_imports_no_project_science_modules(self):
        source_checker.verify_project_import_guard(Path(source_gate.__file__))
        script = """
import sys
import gpuopt.run_corrected_k8_dynamic_coordinate_source_gate
blocked = {
    'gpuopt.candidates.corrected_k8_dynamic_coordinate',
    'gpuopt.source_gate_eligibility',
}
loaded = blocked.intersection(sys.modules)
if loaded:
    raise SystemExit(f'project science modules imported before authorization: {loaded}')
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=source_gate.ROOT,
            check=True,
        )

    def test_slurm_authorization_source_is_external_and_job_bound(self):
        source = (
            source_gate.ROOT
            / "gpuopt/slurm/run_corrected_k8_dynamic_coordinate_source_gate_l40s.sbatch"
        ).read_text()
        self.assertIn(
            'AUTH_GIT_DIR="/scratch/wkyu514/yang07/atypemu_external_authorizations.git"',
            source,
        )
        self.assertNotIn("${AUTH_GIT_DIR:-", source)
        self.assertNotIn("${AUTHORIZATION_GIT_BLOB:", source)
        self.assertIn("job-$SLURM_JOB_ID", source)
        self.assertIn('set -o noclobber', source)
        runner_source = inspect.getsource(source_gate.consume_authorization)
        self.assertIn('"update-ref"', runner_source)
        self.assertIn("authorization_git_blob,", runner_source)

    def test_measure_body_is_source_bound_and_one_shot(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / ".auto/measure_body.sh").read_text()
        self.assertIn('test ! -e "$ROOT/.auto/runs/current"', source)
        self.assertIn("MODE=gate sbatch --hold --parsable", source)
        self.assertIn('with path.open("x")', source)
        self.assertIn(
            "AUTH_REPO=/scratch/wkyu514/yang07/atypemu_external_authorizations.git",
            source,
        )
        self.assertIn(
            "SOURCE_REF=refs/atypemu-source-commitments/corrected-k8/final-v7",
            source,
        )
        self.assertIn("EXPECTED_SOURCE_COMMITMENT_SHA256", source)
        self.assertIn("EXPECTED_SOURCE_COMMITMENT_GIT_BLOB", source)
        self.assertIn('git rev-parse "$SOURCE_REF^{blob}"', source)
        self.assertIn('test "$REMOTE_PINNED_SOURCE_SHA" = "$SOURCE_SHA"', source)
        self.assertIn('git update-ref "$AUTH_REF" "$AUTH_BLOB" "$ZERO"', source)
        self.assertIn("scontrol release '$JOB_ID'", source)
        self.assertIn("failed_corrected_k8_job", source)
        self.assertIn("RESULT_RSYNC_RC=", source)
        self.assertIn("transfer_status.txt", source)
        self.assertNotIn("REMOTE_IMAGE_SHA", source)
        self.assertNotIn("rm -rf", source)
        self.assertNotIn("run_all_label_candidate", source)
        self.assertNotIn("all_label_one_shared_q_metric", source)
        self.assertIn(".auto/measure_body.sh", source_gate.STATIC_COMMITTED_FILES)
        self.assertIn(".auto/measure_body.sh", source_checker.STATIC_COMMITTED_FILES)
        self.assertIn(".auto/measure_body.sh", set(source_freezer.STATIC_FILES))
        self.assertNotIn(".auto/measure.sh", source_gate.STATIC_COMMITTED_FILES)
        slurm_source = (
            root
            / "gpuopt/slurm/run_corrected_k8_dynamic_coordinate_source_gate_l40s.sbatch"
        ).read_text()
        image_sha256 = (
            "67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531"
        )
        self.assertIn(f"IMAGE_SHA={image_sha256}", source)
        self.assertIn(f"EXPECTED_IMAGE_SHA256={image_sha256}", slurm_source)
        for name in (
            "unchecked_result.json",
            "decision.json",
            "authorization.json",
            "consumed_authorization.json",
            "external_claim.json",
        ):
            self.assertIn(f"results/{name}", slurm_source)
            self.assertIn(name, source)
        checks_source = (Path(__file__).resolve().parents[2] / ".auto" / "checks.sh").read_text()
        self.assertIn("check_mode.json", source)
        self.assertIn("corrected_k8_source_commitment.json", checks_source)
        self.assertIn("unchecked_result.json", checks_source)
        self.assertNotIn('current/source_commitment.json', checks_source)
        self.assertNotIn('current/result.json', checks_source)
        self.assertLess(
            slurm_source.index("ACTUAL_IMAGE_SHA256=$(sha256sum"),
            slurm_source.index('if [[ "$MODE" == "test" ]]'),
        )

    def test_stale_output_is_rejected_before_authorization_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            stale = output.parent / "source_coordinate_A_half0.npz"
            stale.write_bytes(b"stale")
            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                source_gate.verify_output_paths_fresh(output)

    def test_unscored_backbone_actuators_are_disabled(self):
        raw = torch.full(
            (1, candidate.SUPPORT_COUNT, len(candidate.ACTUATOR_SPECS)), 99.0
        )
        physical = candidate.actuator_delta(raw)
        self.assertTrue(torch.equal(physical[..., :2], torch.zeros_like(physical[..., :2])))

    def test_analytic_chi_distance_jacobian_matches_central_difference(self):
        records = {
            "name": np.asarray(["CA", "CB", "OG", "HG", "C"], dtype=str),
            "resname": np.asarray(["SER", "SER", "SER", "SER", "ALA"], dtype=str),
            "chain": np.asarray(["A"] * 5, dtype=str),
            "seq_id": np.asarray([1, 1, 1, 1, 2], dtype=np.int64),
            "element": np.asarray(["C", "C", "O", "H", "C"], dtype=str),
            "coordinate": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [2.5, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 2.0, 1.0],
                ],
                dtype=np.float64,
            ),
        }
        velocity = candidate._chi_coordinate_velocities(records)
        direction = (
            records["coordinate"][2] - records["coordinate"][4]
        )
        direction /= np.linalg.norm(direction)
        analytic = float(velocity[2, 0] @ direction)
        epsilon = 1.0e-6
        plus = candidate.rotate_about_axis(
            records["coordinate"][[2]],
            records["coordinate"][0],
            records["coordinate"][1] - records["coordinate"][0],
            epsilon,
        )[0]
        minus = candidate.rotate_about_axis(
            records["coordinate"][[2]],
            records["coordinate"][0],
            records["coordinate"][1] - records["coordinate"][0],
            -epsilon,
        )[0]
        finite_difference = (
            np.linalg.norm(plus - records["coordinate"][4])
            - np.linalg.norm(minus - records["coordinate"][4])
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, finite_difference, places=7)

    def test_data_loss_reaches_delta_only_through_dynamic_geometry(self):
        batch = 1
        geometry = torch.zeros(
            batch, candidate.SUPPORT_COUNT, len(candidate.OBSERVER_GEOMETRY_COLUMNS)
        )
        self_jacobian = torch.zeros(batch, candidate.SUPPORT_COUNT, 5, 4)
        self_jacobian[:, :, 0, 0] = torch.linspace(
            0.25, 1.0, candidate.SUPPORT_COUNT
        )
        neighbor_jacobian = torch.zeros_like(self_jacobian)
        neighbor_residue = torch.full(
            (batch, candidate.SUPPORT_COUNT, 5), -1, dtype=torch.long
        )
        residue = torch.zeros(batch, dtype=torch.long)
        delta_raw = nn.Parameter(
            torch.zeros(1, candidate.SUPPORT_COUNT, len(candidate.ACTUATOR_SPECS))
        )
        active = candidate.actuate_coordinate_geometry(
            geometry,
            residue,
            delta_raw,
            self_jacobian,
            neighbor_jacobian,
            neighbor_residue,
        )
        zeroed = candidate.actuate_coordinate_geometry(
            geometry,
            residue,
            delta_raw,
            self_jacobian,
            neighbor_jacobian,
            neighbor_residue,
            jacobian_scale=0.0,
        )
        esm = torch.zeros(batch, candidate.ESM_DIM)
        categorical = torch.zeros(batch, 4, dtype=torch.long)
        numeric = torch.zeros(batch, 10)
        torsion = torch.zeros(
            batch, candidate.SUPPORT_COUNT, len(candidate.TORSION_COLUMNS)
        )
        response = candidate.coordinate_response(
            GeometryProbe(), esm, categorical, numeric, geometry, active, torsion
        )
        loss = torch.square(response.mean() - 1.0)
        gradient = torch.autograd.grad(loss, delta_raw)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)
        self.assertTrue(torch.equal(zeroed, geometry))
        zero_gradient = torch.autograd.grad(zeroed.sum(), delta_raw)[0]
        self.assertTrue(torch.equal(zero_gradient, torch.zeros_like(zero_gradient)))

    def test_unavailable_torsion_does_not_move_emitted_coordinates(self):
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C  \n"
            "ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00  0.00           O  \n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structure = root / "bmr1"
            structure.mkdir()
            for support_id in range(candidate.SUPPORT_COUNT):
                (structure / f"bmr1_support{support_id}.pdb").write_text(pdb)
            frame = pd.DataFrame(
                {
                    "entity_uid": ["entity", "entity"],
                    "target_id": ["target:1", "target:2"],
                    "atom_id": ["A", "A"],
                    "target_value": [0.0, 1.0],
                }
            )
            values = {
                "frame": frame,
                "residue_index": np.asarray([0, 0], dtype=np.int64),
                "residue_keys": [("entity", 1)],
                "torsion": np.zeros(
                    (2, candidate.SUPPORT_COUNT, len(candidate.TORSION_COLUMNS)),
                    dtype=np.float32,
                ),
                "entities": [{"entity_uid": "entity", "bmrb_id": "bmr1"}],
                "support_ids": [
                    f"support{index}" for index in range(candidate.SUPPORT_COUNT)
                ],
            }
            for key in candidate.SOURCE_ROW_ARRAYS:
                if key not in values:
                    values[key] = np.zeros((len(frame), 1), dtype=np.float32)
            values["source_eligibility_receipt"] = build_eligibility_receipt(
                frame,
                frame,
                frozen_atom_ids=["A"],
                fold="A",
                held_half=0,
                role="evidence_assimilation",
            )
            delta = torch.full(
                (1, candidate.SUPPORT_COUNT, len(candidate.ACTUATOR_SPECS)), 10.0
            )
            output = root / "audit.npz"
            candidate.coordinate_audit(
                values, delta, structure_root=root, output=output
            )
            with np.load(output) as arrays:
                self.assertTrue(
                    np.array_equal(
                        arrays["conditioned_coordinates"],
                        arrays["no_evidence_coordinates"],
                    )
                )
                self.assertEqual(len(arrays["state_entity_uid"]), candidate.SUPPORT_COUNT)
                self.assertEqual(
                    set(arrays["state_support_id"]),
                    {f"support{index}" for index in range(candidate.SUPPORT_COUNT)},
                )
                self.assertEqual(
                    set(arrays["atom_state_index"]), set(range(candidate.SUPPORT_COUNT))
                )

    def test_ccc_is_not_a_loss_dependency(self):
        for function in (candidate.train_observer, candidate.optimize_assimilation):
            source = inspect.getsource(function).lower()
            self.assertNotIn("ccc", source)
        self.assertNotIn("ccc", inspect.getsource(candidate.coordinate_response).lower())


if __name__ == "__main__":
    unittest.main()
