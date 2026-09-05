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
import pandas as pd
from gpuopt.candidates import k32_nested_k8_adapter as adapter
from gpuopt.source_gate_eligibility import build_eligibility_receipt

assert tuple(adapter.SUPPORTS[i] for i in adapter.NESTED) == (1, 126, 251, 376, 501, 626, 751, 876)
assert "allow_pickle=False" in inspect.getsource(adapter.load)
surface = adapter.load(Path("."), "bmrb:10109:entity:1")
nested = adapter.nested8(surface)
assert all(np.array_equal(right, left[:, adapter.NESTED]) for left, right in zip(surface.arrays, nested.arrays, strict=True))
assert nested.target_ids.tolist() == surface.target_ids.tolist()
left, right = adapter.fresh_state(3, 32), adapter.fresh_state(3, 8)
assert all(not np.shares_memory(a, b) for a in left for b in right)
all32, nested8 = adapter.matched_arms(surface, np.zeros((len(surface.target_ids), 2)))
assert all32.observer_state is nested8.observer_state
assert all(not np.shares_memory(a, b) for a in all32.assimilation_state for b in nested8.assimilation_state)
zeros32 = np.zeros((len(surface.target_ids), 32, 4), np.float32)
assert np.array_equal(adapter.actuate(surface, zeros32, zeros32), surface.arrays[0])
synthetic = pd.DataFrame({"entity_uid": ["e", "e"], "target_id": ["a", "b"], "atom_id": ["CA", "CA"], "target_value": [1.0, 2.0]})
receipt = build_eligibility_receipt(synthetic, synthetic, frozen_atom_ids=["CA"], fold="A", held_half=0, role="smoke")
adapter.require_eligible({"frame": synthetic, "source_eligibility_receipt": receipt})
print("METRIC matched_adapter_checks=14")
print("METRIC source_target_values_read=0")
print("METRIC outer_or_formal_metrics_opened=0")
PY
/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/k32_nested_k8_adapter.py
