from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from gpuopt.candidates import check_condition_uncertainty_inputs as checker


ROOT = Path(__file__).resolve().parents[2]


class ConditionUncertaintyInputCheckerTests(unittest.TestCase):
    def test_target_free_self_test_and_no_freezer_import(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / checker.CHECKER_RELATIVE), "--self-test"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("STATUS HOLD_TARGET_UNREAD_INPUT_SELF_TEST_PASS", result.stdout)
        source = (ROOT / checker.CHECKER_RELATIVE).read_text()
        self.assertNotIn("import freeze_condition_uncertainty_inputs", source)

    def test_duplicate_json_and_receipt_clobber_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            checker._json_object(b'{"x": 1, "x": 2}', "duplicate")
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            checker._write_once(receipt, {"fixed": True})
            with self.assertRaises(FileExistsError):
                checker._write_once(receipt, {"fixed": False})


if __name__ == "__main__":
    unittest.main()
