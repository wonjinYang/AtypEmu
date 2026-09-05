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
import torch
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
rows = len(surface.target_ids)
observer = np.zeros((rows, 2), np.float32)
inputs = adapter.bind_model_inputs(surface, support_ids=adapter.SUPPORT_IDS, observer_state=observer, torsion=np.zeros((rows, 32, 1), np.float32), geometry=np.zeros((rows, 32, 1), np.float32), dynamic_distance=surface.arrays[0], support_anchor=np.zeros((rows, 32), np.float32))
small_inputs = adapter.nested_model_inputs(inputs)
assert small_inputs.observer_state is inputs.observer_state
assert np.array_equal(small_inputs.support_anchor, inputs.support_anchor[:, adapter.NESTED])
try:
    adapter.bind_model_inputs(surface, support_ids=small_inputs.surface.support_ids, observer_state=observer, torsion=inputs.torsion, geometry=inputs.geometry, dynamic_distance=surface.arrays[0], support_anchor=inputs.support_anchor)
except ValueError as error:
    assert "support roster" in str(error)
else:
    raise AssertionError("incomplete K8 base surface was accepted as K32")
probe = adapter.Surface(surface.target_ids[:16], surface.support_ids, tuple(value[:16] for value in surface.arrays))
self_delta = torch.zeros((16, 32, 4), requires_grad=True)
neighbor_delta = torch.zeros((16, 32, 4), requires_grad=True)
active_distance, available = adapter.differentiable_actuate(probe, self_delta, neighbor_delta)
assert np.array_equal(active_distance.detach().numpy(), probe.arrays[0])
torch.manual_seed(20260905)
inventory = tuple(__import__("json").load(open(".auto/frozen/all_label_inventory.json"))["eligible_atom_ids"])
atom, residue, position = adapter.context_indices(surface, inventory)
observer_model = adapter.DynamicDistanceObserver(len(inventory) + 1)
observer_model(active_distance, available, torch.from_numpy(atom[:16]), torch.from_numpy(residue[:16]), torch.from_numpy(position[:16])).sum().backward()
assert self_delta.grad is not None and torch.isfinite(self_delta.grad).all() and self_delta.grad.norm() > 0
small_distance = active_distance[:, adapter.NESTED]
small_available = available[:, adapter.NESTED]
assert observer_model(small_distance, small_available, torch.from_numpy(atom[:16]), torch.from_numpy(residue[:16]), torch.from_numpy(position[:16])).shape == (16, 8)
anchor = adapter.sequence_anchor(Path("."), "B", surface)
assert anchor.shape == (len(surface.target_ids), 32) and np.array_equal(anchor[:, 0], anchor[:, -1])
assert np.all(anchor[:, adapter.NESTED] == anchor[:, :1])
assert surface.row_context is adapter.nested8(surface).row_context
training_surface = adapter.Surface(
    surface.target_ids[:16], surface.support_ids,
    tuple(value[:16] for value in surface.arrays),
    (
        np.arange(1, 17, dtype=np.int32),
        np.asarray(["ALA"] * 16),
        np.asarray(["CA", "CB"] * 8),
    ),
)
seq_ids, comp_ids, atom_ids = training_surface.row_context
training_frame = pd.DataFrame({
    "entity_uid": "synthetic",
    "target_id": training_surface.target_ids,
    "seq_id": seq_ids,
    "comp_id": comp_ids,
    "atom_id": atom_ids,
    "target_value": np.arange(16, dtype=np.float32) + np.asarray([0.0, 0.25] * 8),
})
training_receipt = build_eligibility_receipt(training_frame, training_frame, frozen_atom_ids=sorted(set(atom_ids)), fold="A", held_half=0, role="smoke")
synthetic_values = {"frame": training_frame, "source_eligibility_receipt": training_receipt}
synthetic_anchor = np.zeros((16, 32), np.float32)
targets = adapter.source_targets(synthetic_values, training_surface, synthetic_anchor)
for name in set(atom_ids):
    assert np.isclose(targets.weight[atom_ids == name].sum(), targets.weight.sum() / len(set(atom_ids)))
torch.manual_seed(20260905)
fit_model = adapter.DynamicDistanceObserver(len(inventory) + 1)
fit_context = adapter.context_indices(training_surface, inventory)
initial_loss, final_loss = adapter.fit_source_observer(fit_model, training_surface, targets, fit_context, epochs=96)
assert final_loss < initial_loss * 0.5
assert "evaluation" not in inspect.signature(adapter.fit_source_observer).parameters
print("METRIC matched_adapter_checks=29")
print("METRIC source_target_values_read=0")
print("METRIC outer_or_formal_metrics_opened=0")
PY
/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/k32_nested_k8_adapter.py
