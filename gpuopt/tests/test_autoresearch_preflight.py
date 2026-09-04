#!/usr/bin/env python3
"""Validation tests for corrected all-label autoresearch setup."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from gpuopt.preflight_all_label_autoresearch import (
    collect_errors,
    eligibility_precedes_fit,
    resolve_archive_member,
)
from gpuopt.source_gate_eligibility import (
    eligible_source_atom_ids,
    source_eligible_row_mask,
)

ROOT = Path(__file__).resolve().parents[2]


class AutoresearchPreflightTests(unittest.TestCase):
    def test_source_eligibility_excludes_sparse_constant_and_nonfinite_labels(self):
        atom_ids = ("CA", "CA", "CA", "CB", "CB", "N", "N", "H")
        targets = (1.0, 2.0, float("nan"), 3.0, 3.0, float("nan"), 4.0, 8.0)
        inventory = ("N", "CA", "CB", "H")
        self.assertEqual(
            eligible_source_atom_ids(atom_ids, targets, inventory), ("CA",)
        )
        self.assertEqual(
            source_eligible_row_mask(atom_ids, targets, inventory),
            (True, True, False, False, False, False, False, False),
        )

    def test_quarantined_runner_fails_eligibility_order_guard(self):
        source = subprocess.run(
            (
                "git",
                "show",
                "95a8906:gpuopt/run_dynamic_coordinate_source_gate.py",
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        self.assertFalse(eligibility_precedes_fit(source))
        self.assertTrue(
            eligibility_precedes_fit(
                "def run():\n eligible_source_atom_ids()\n normalization()\n train_observer()\n optimize_assimilation()"
            )
        )
        self.assertFalse(
            eligibility_precedes_fit(
                "# eligible_source_atom_ids()\ndef run():\n normalization()\n train_observer()\n optimize_assimilation()"
            )
        )

    def test_archive_member_must_remain_below_archive_root(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive"
            archive.mkdir()
            self.assertEqual(
                resolve_archive_member(archive, "records/result.json"),
                archive / "records/result.json",
            )
            with self.assertRaisesRegex(ValueError, "escapes root"):
                resolve_archive_member(archive, "../outside.json")

    def test_current_workspace_passes_no_science_preflight(self):
        self.assertEqual(
            collect_errors(
                ROOT, expect_retained_baseline=True, corrected_runner=None
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
