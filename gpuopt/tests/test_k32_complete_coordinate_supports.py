from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from gpuopt import check_k32_complete_coordinate_supports as checker
from gpuopt import materialize_k32_complete_coordinate_supports as materializer


ROOT = Path(__file__).resolve().parents[2]


def _write_alanine(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C  ",
        "ATOM      3  C   ALA A   1       2.900   0.000   0.000  1.00  0.00           C  ",
        "ATOM      4  O   ALA A   1       4.100   0.000   0.000  1.00  0.00           O  ",
        "ATOM      5  CB  ALA A   1       1.450   1.500   0.000  1.00  0.00           C  ",
        "ATOM      6  H   ALA A   1      -0.960   0.000   0.000  1.00  0.00           H  ",
        "TER",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n")


class K32CompleteCoordinateSupportTests(unittest.TestCase):
    def test_topology_audits_do_not_cache_across_pdb_files(self) -> None:
        self.assertFalse(hasattr(materializer, "_TOPOLOGY_CACHE"))
        self.assertFalse(hasattr(checker, "_TOPOLOGY_CACHE"))

    def test_physicality_rejects_broken_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.pdb"
            _write_alanine(path)
            text = path.read_text().replace("   1.450   0.000", "   2.050   0.000", 1)
            path.write_text(text)
            observed = materializer.physicality_audit(materializer.pdb_records(path))
            self.assertFalse(observed["pass"])
            self.assertGreater(observed["backbone_violation_count"], 0)

    def test_ter_separates_consecutive_residues_from_peptide_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two_segments.pdb"
            one = Path(temporary) / "one.pdb"
            _write_alanine(one)
            first = [line for line in one.read_text().splitlines() if line.startswith("ATOM")]
            second = []
            for line in first:
                shifted = list(line)
                shifted[6:11] = f"{int(line[6:11]) + 10:5d}"
                shifted[22:26] = f"{2:4d}"
                shifted[30:38] = f"{float(line[30:38]) + 10.0:8.3f}"
                second.append("".join(shifted))
            path.write_text("\n".join([*first, "TER", *second, "TER", "END"]) + "\n")
            records = materializer.pdb_records(path)
            self.assertEqual(records["sequence"], "AA")
            self.assertEqual({key[0] for key in records["residue_order"]}, {0, 1})
            self.assertTrue(materializer.physicality_audit(records)["pass"])
            checked = checker.read_coordinates(path)
            self.assertEqual({key[0] for key in checked["order"]}, {0, 1})
            self.assertTrue(checker.physicality(checked)[0])

    def test_freeze_authorize_copy_and_independently_check_full_roster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                materializer.MATERIALIZER_RELATIVE,
                materializer.CHECKER_RELATIVE,
                materializer.PLAN_RELATIVE,
                materializer.TOPOLOGY_AUDIT_RELATIVE,
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            template = root / "template.pdb"
            _write_alanine(template)
            entities = []
            plan = json.loads((root / materializer.PLAN_RELATIVE).read_text())
            supports = [int(value) for value in plan["support_indices"]]
            primary_root = root / "data" / "BioEmu"
            for entity_index in range(162):
                bmrb_id = f"bmr{10000 + entity_index}"
                entities.append(
                    {
                        "bmrb_id": bmrb_id,
                        "entity_uid": f"{bmrb_id}:1",
                        "sequence": "A",
                    }
                )
                for support_index in supports:
                    destination = (
                        primary_root
                        / bmrb_id
                        / f"{bmrb_id}_BioEmu_{support_index}.pdb"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(template, destination)
            broken = primary_root / "bmr10000" / "bmr10000_BioEmu_32.pdb"
            broken.unlink()
            _write_alanine(broken)
            broken.write_text(
                broken.read_text().replace("   1.450   0.000", "   2.050   0.000", 1)
            )
            replacement = primary_root / "bmr10000" / "bmr10000_BioEmu_31.pdb"
            os.link(template, replacement)
            broken_k8 = primary_root / "bmr10000" / "bmr10000_BioEmu_126.pdb"
            broken_k8.unlink()
            _write_alanine(broken_k8)
            broken_k8.write_text(
                broken_k8.read_text().replace("   1.450   0.000", "   2.050   0.000", 1)
            )
            os.link(
                template,
                primary_root / "bmr10000" / "bmr10000_BioEmu_125.pdb",
            )
            parent = root / materializer.PARENT_RELATIVE
            parent.parent.mkdir(parents=True, exist_ok=True)
            parent.write_text(json.dumps({"entities": entities}) + "\n")
            source_commitment = root / "source_commitment.json"
            freeze_args = argparse.Namespace(
                root=root,
                plan=Path(materializer.PLAN_RELATIVE),
                parent_commitment=Path(materializer.PARENT_RELATIVE),
                primary_root=Path("data/BioEmu"),
                output=source_commitment,
            )
            self.assertEqual(materializer.freeze(freeze_args), 0)
            frozen = json.loads(source_commitment.read_text())
            self.assertEqual(frozen["pair_count"], 162 * 32)
            self.assertEqual(frozen["replacement_count"], 2)
            self.assertEqual(frozen["legacy_k8_replacement_count"], 1)
            checker.validate_source_inventory(frozen["pairs"], entities)
            malformed_rows = [dict(row) for row in frozen["pairs"]]
            malformed_rows[0]["support_index"] = 999
            with self.assertRaisesRegex(ValueError, "source support identity"):
                checker.validate_source_inventory(malformed_rows, entities)
            with self.assertRaisesRegex(ValueError, "parent entity roster"):
                checker.validate_source_inventory(frozen["pairs"], [*entities, entities[0]])
            injected = dict(frozen)
            injected["files"] = {
                **frozen["files"],
                "data/forbidden_target.parquet": "0" * 64,
            }
            injected_path = root / "injected.json"
            injected_path.write_text(json.dumps(injected) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                materializer.verify_commitment(root, injected_path)

            preflight_receipt = root / "preflight.json"
            self.assertEqual(
                materializer.preflight(
                    argparse.Namespace(
                        root=root,
                        source_commitment=source_commitment,
                        primary_root=Path("data/BioEmu"),
                        receipt=preflight_receipt,
                    )
                ),
                0,
            )
            authorization_git_dir = root / "authorizations.git"
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(authorization_git_dir)],
                check=True,
            )
            authorization_ref = "refs/atypemu-authorizations/k32-support/test-job"
            output_root = root / "data/k32_complete_coordinate_supports_v4"
            source_sha256 = materializer.sha256_file(source_commitment)
            authorization = {
                "contract": materializer.AUTHORIZATION_CONTRACT,
                "authorized": True,
                "authorization_ref": authorization_ref,
                "output_relative_path": "data/k32_complete_coordinate_supports_v4",
                "preflight_receipt_sha256": materializer.sha256_file(
                    preflight_receipt
                ),
                "slurm_job_id": "unit-test-job",
                "source_commitment_sha256": source_sha256,
            }
            authorization_path = root / "authorization.json"
            authorization_path.write_text(
                json.dumps(authorization, sort_keys=True, separators=(",", ":")) + "\n"
            )
            authorization_blob = subprocess.run(
                [
                    "git",
                    f"--git-dir={authorization_git_dir}",
                    "hash-object",
                    "-w",
                    str(authorization_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "git",
                    f"--git-dir={authorization_git_dir}",
                    "update-ref",
                    authorization_ref,
                    authorization_blob,
                ],
                check=True,
            )
            receipt = root / "receipt.json"
            consumed = root / "consumed.json"
            claim = root / "claim.json"
            materialize_args = argparse.Namespace(
                root=root,
                source_commitment=source_commitment,
                preflight_receipt=preflight_receipt,
                output_root=output_root,
                receipt=receipt,
                authorization_git_dir=authorization_git_dir,
                authorization_ref=authorization_ref,
                authorization_git_blob=authorization_blob,
                consumed_authorization=consumed,
                external_claim=claim,
                slurm_job_id="unit-test-job",
            )
            self.assertEqual(materializer.materialize(materialize_args), 0)
            summary = checker.check(
                argparse.Namespace(
                    root=root,
                    source_commitment=source_commitment,
                    preflight_receipt=preflight_receipt,
                    receipt=receipt,
                    consumed_authorization=consumed,
                    external_claim=claim,
                    authorization_git_dir=authorization_git_dir,
                )
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["pair_count"], 162 * 32)
            self.assertEqual(summary["replacement_count"], 2)
            self.assertEqual(summary["legacy_k8_replacement_count"], 1)

            consumed_payload = json.loads(consumed.read_text())
            claim_payload = json.loads(claim.read_text())
            consumed_payload["authorization_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "authorization content hash"):
                checker.verify_external_consumption(
                    authorization_git_dir,
                    source_sha256,
                    materializer.sha256_file(preflight_receipt),
                    consumed_payload,
                    claim_payload,
                )

            preflight_payload = json.loads(preflight_receipt.read_text())
            preflight_payload["source_gate_authorized"] = True
            preflight_receipt.write_text(
                json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "source-gate authorization"):
                checker.check(
                    argparse.Namespace(
                        root=root,
                        source_commitment=source_commitment,
                        preflight_receipt=preflight_receipt,
                        receipt=receipt,
                        consumed_authorization=consumed,
                        external_claim=claim,
                        authorization_git_dir=authorization_git_dir,
                    )
                )
            preflight_payload["source_gate_authorized"] = False
            preflight_receipt.write_text(
                json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "not fresh"):
                materializer.consume_authorization(
                    argparse.Namespace(
                        root=root,
                        output_root=output_root,
                        authorization_git_dir=authorization_git_dir,
                        authorization_ref=authorization_ref,
                        authorization_git_blob=authorization_blob,
                        consumed_authorization=root / "consumed_reuse.json",
                        external_claim=root / "claim_reuse.json",
                        slurm_job_id="unit-test-job",
                    ),
                    source_commitment_sha256=source_sha256,
                    preflight_receipt_sha256=materializer.sha256_file(
                        preflight_receipt
                    ),
                )

            consumed_payload = json.loads(consumed.read_text())
            claim_blob = consumed_payload["external_claim_git_blob"]
            subprocess.run(
                [
                    "git",
                    f"--git-dir={authorization_git_dir}",
                    "update-ref",
                    authorization_ref,
                    authorization_blob,
                    claim_blob,
                ],
                check=True,
            )
            with self.assertRaises(subprocess.CalledProcessError):
                materializer.consume_authorization(
                    argparse.Namespace(
                        root=root,
                        output_root=output_root,
                        authorization_git_dir=authorization_git_dir,
                        authorization_ref=authorization_ref,
                        authorization_git_blob=authorization_blob,
                        consumed_authorization=root / "consumed_aba.json",
                        external_claim=root / "claim_aba.json",
                        slurm_job_id="unit-test-job",
                    ),
                    source_commitment_sha256=source_sha256,
                    preflight_receipt_sha256=materializer.sha256_file(
                        preflight_receipt
                    ),
                )
            subprocess.run(
                [
                    "git",
                    f"--git-dir={authorization_git_dir}",
                    "update-ref",
                    authorization_ref,
                    claim_blob,
                    authorization_blob,
                ],
                check=True,
            )

            receipt_payload = json.loads(receipt.read_text())
            receipt_payload["outputs"][0]["selected_source_index"] += 1
            receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "output metadata mismatch"):
                checker.check(
                    argparse.Namespace(
                        root=root,
                        source_commitment=source_commitment,
                        preflight_receipt=preflight_receipt,
                        receipt=receipt,
                        consumed_authorization=consumed,
                        external_claim=claim,
                        authorization_git_dir=authorization_git_dir,
                    )
                )
            receipt_payload["outputs"][0]["selected_source_index"] -= 1
            receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")

            receipt_payload = json.loads(receipt.read_text())
            original_output_path = receipt_payload["outputs"][0]["output_relative_path"]
            receipt_payload["outputs"][0]["output_relative_path"] = (
                "data/k32_complete_coordinate_supports_v4/bmr10000/wrong.pdb"
            )
            receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "noncanonical output coordinate path"):
                checker.check(
                    argparse.Namespace(
                        root=root,
                        source_commitment=source_commitment,
                        preflight_receipt=preflight_receipt,
                        receipt=receipt,
                        consumed_authorization=consumed,
                        external_claim=claim,
                        authorization_git_dir=authorization_git_dir,
                    )
                )
            receipt_payload["outputs"][0]["output_relative_path"] = original_output_path
            receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")

            first_output = root / json.loads(receipt.read_text())["outputs"][0][
                "output_relative_path"
            ]
            first_output.write_bytes(first_output.read_bytes() + b"REMARK tampered\n")
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                checker.check(
                    argparse.Namespace(
                        root=root,
                        source_commitment=source_commitment,
                        preflight_receipt=preflight_receipt,
                        receipt=receipt,
                        consumed_authorization=consumed,
                        external_claim=claim,
                        authorization_git_dir=authorization_git_dir,
                    )
                )


if __name__ == "__main__":
    unittest.main()
