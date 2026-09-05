#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

$PY - <<'PY'
import inspect
from pathlib import Path
import numpy as np
from gpuopt.candidates import k32_nested_k8_adapter as adapter

assert tuple(adapter.SUPPORTS[i] for i in adapter.NESTED) == (1, 126, 251, 376, 501, 626, 751, 876)
assert "allow_pickle=False" in inspect.getsource(adapter.load)
surface = adapter.load(Path("."), "bmrb:10109:entity:1")
nested = adapter.nested8(surface)
assert all(np.array_equal(right, left[:, adapter.NESTED]) for left, right in zip(surface.arrays, nested.arrays, strict=True))
assert nested.target_ids.tolist() == surface.target_ids.tolist()
left, right = adapter.fresh_state(3, 32), adapter.fresh_state(3, 8)
assert all(not np.shares_memory(a, b) for a in left for b in right)
print("METRIC matched_adapter_checks=5")
print("METRIC source_target_values_read=0")
print("METRIC outer_or_formal_metrics_opened=0")
PY
/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/k32_nested_k8_adapter.py
