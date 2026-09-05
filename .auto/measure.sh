#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

$PY - <<'PY'
import inspect
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import numpy as np
import pandas as pd
import torch
from gpuopt.candidates import k32_nested_k8_adapter as adapter
from gpuopt import k32_source_gate_authorization as gate_authorization
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
neighbor_zeros32 = np.zeros((*surface.arrays[0].shape, 4), np.float32)
assert np.array_equal(adapter.actuate(surface, zeros32, neighbor_zeros32), surface.arrays[0])
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
probe = adapter.Surface(
    surface.target_ids[:16], surface.support_ids,
    tuple(value[:16] for value in surface.arrays),
    tuple(value[:16] for value in surface.row_context),
)
self_delta = torch.zeros((16, 32, 4), requires_grad=True)
neighbor_delta = torch.zeros((16, 32, 5, 4), requires_grad=True)
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
synthetic_values, training_surface = adapter.eligible_source_subset(
    training_frame, training_surface,
    frozen_atom_ids=tuple(sorted(set(atom_ids))), fold="A", held_half=0, role="smoke",
)
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
residue_ids = adapter.residue_inventory(probe)
residue_delta = torch.zeros((len(residue_ids), 32, 4), requires_grad=True)
row_delta, neighbor_row_delta = adapter.gather_residue_deltas(probe, residue_ids, residue_delta)
assert row_delta.shape == (16, 32, 4) and neighbor_row_delta.shape == (16, 32, 5, 4)
for seq_id in set(probe.row_context[0]):
    rows_for_residue = np.flatnonzero(probe.row_context[0] == seq_id)
    assert torch.equal(row_delta[rows_for_residue], row_delta[rows_for_residue[:1]].expand(len(rows_for_residue), -1, -1))
mapped_distance, mapped_available = adapter.differentiable_actuate(probe, row_delta, neighbor_row_delta)
fit_model(mapped_distance, mapped_available, torch.from_numpy(atom[:16]), torch.from_numpy(residue[:16]), torch.from_numpy(position[:16])).sum().backward()
assert residue_delta.grad is not None and torch.isfinite(residue_delta.grad).all() and residue_delta.grad.norm() > 0
results = {
    mode: adapter.optimize_assimilation(
        fit_model, training_surface, targets, synthetic_anchor, fit_context,
        mode=mode, steps=24,
    )
    for mode in ("full", "no_coordinate", "uniform_q", "anchor_only")
}
assert all(result.final_data_loss <= result.initial_data_loss for result in results.values())
assert np.isclose(results["full"].q.sum(), 1.0) and results["full"].q.shape == (32,)
assert np.array_equal(results["uniform_q"].q, np.full(32, 1 / 32, np.float32))
assert not np.any(results["no_coordinate"].residue_delta) and not np.any(results["anchor_only"].residue_delta)
assert results["full"].coordinate_gradient_norm > 0 and results["uniform_q"].coordinate_gradient_norm > 0
assert all(not np.shares_memory(left.q, right.q) for left in results.values() for right in results.values() if left is not right)
plan = adapter.source_crossfit_plan(Path("."))
assert len(plan) == 4 and {(part.fold, part.held_half) for part in plan} == {("A", 0), ("A", 1), ("B", 0), ("B", 1)}
assert all(not (set(part.train_clusters) & set(part.evaluation_clusters)) for part in plan)
assert all(not (set(part.train_entities) & set(part.evaluation_entities)) for part in plan)
assert len(set(entity for part in plan for entity in part.evaluation_entities)) == 135
second = adapter.load(Path("."), "bmrb:10142:entity:1")
combined = adapter.concatenate_surfaces((surface, second))
assert len(combined.target_ids) == len(surface.target_ids) + len(second.target_ids)
combined_nested = adapter.nested8(combined)
separate_nested = adapter.concatenate_surfaces((adapter.nested8(surface), adapter.nested8(second)))
assert np.array_equal(combined_nested.target_ids, separate_nested.target_ids)
assert all(np.array_equal(left, right) for left, right in zip(combined_nested.arrays, separate_nested.arrays, strict=True))
held_frame = training_frame.assign(target_value=training_frame["target_value"] * 10)
held_values, held_surface = adapter.eligible_source_subset(
    held_frame, training_surface,
    frozen_atom_ids=("CA", "CB"), fold="A", held_half=1, role="assimilation_smoke",
)
held_targets = adapter.source_targets(
    held_values, held_surface, synthetic_anchor, normalization=targets.normalization,
)
self_normalized_held = adapter.source_targets(held_values, held_surface, synthetic_anchor)
assert np.array_equal(held_targets.scale, targets.scale)
assert not np.array_equal(self_normalized_held.scale, targets.scale)
ineligible_frame = training_frame.copy()
ineligible_frame.loc[15, "atom_id"] = "ZZ"
ineligible_context = list(training_surface.row_context)
ineligible_context[2] = ineligible_frame["atom_id"].to_numpy(str)
ineligible_surface = adapter.Surface(
    training_surface.target_ids, training_surface.support_ids,
    training_surface.arrays, tuple(ineligible_context),
)
eligible_values, eligible_surface = adapter.eligible_source_subset(
    ineligible_frame, ineligible_surface,
    frozen_atom_ids=("CA", "CB", "ZZ"), fold="A", held_half=0, role="exclusion_smoke",
)
assert len(eligible_values["frame"]) == 15 and len(eligible_surface.target_ids) == 15
assert "ZZ" not in set(eligible_values["frame"]["atom_id"])
k32_result, k8_result = adapter.matched_assimilation(
    fit_model, training_surface, targets, synthetic_anchor, fit_context,
    mode="full", steps=12,
)
assert k32_result.prediction.shape == k8_result.prediction.shape == (16,)
assert np.isfinite(k32_result.prediction).all() and np.isfinite(k8_result.prediction).all()
assert not np.shares_memory(k32_result.q, k8_result.q)
k32_score, k32_labels = adapter.macro_atom_id_ccc(
    training_frame, k32_result.prediction, ("CA", "CB"),
)
k8_score, k8_labels = adapter.macro_atom_id_ccc(
    training_frame, k8_result.prediction, ("CA", "CB"),
)
assert np.isfinite([k32_score, k8_score]).all()
assert set(k32_labels) == set(k8_labels) == {"CA", "CB"}
assert next(fit_model.parameters()).device.type == "cpu"
assert all(not parameter.requires_grad for parameter in fit_model.parameters())
assert "device" in inspect.signature(adapter.fit_source_observer).parameters
assert "batch_size" in inspect.signature(adapter.fit_source_observer).parameters
assert "device" in inspect.signature(adapter.matched_assimilation).parameters
groups = adapter.source_entity_slices(training_surface)
entity_uid = str(training_surface.target_ids[0]).split(":target:", 1)[0]
assert list(groups) == [entity_uid]
assert np.array_equal(groups[entity_uid], np.arange(16))
target_subset = adapter.subset_source_targets(targets, groups[entity_uid][::2])
assert len(target_subset.normalized) == 8
assert target_subset.normalization is targets.normalization
target_read_called = False

def forbidden_target_read(*args, **kwargs):
    global target_read_called
    target_read_called = True
    raise AssertionError("target reader reached before authorization")

with mock.patch.object(adapter.pd, "read_parquet", side_effect=forbidden_target_read):
    try:
        adapter.load_source_entity_targets(
            Path("."), training_surface, entity_uid=entity_uid, fold="A",
            expected_sha256="0" * 64, access=None,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("preauthorization target read did not fail closed")
assert not target_read_called
annotation = inspect.signature(adapter.load_source_entity_targets).parameters["access"].annotation
assert "ConsumedAuthorization" in annotation
with TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    source_commitment = temporary / "source.json"
    source_commitment.write_text("{}\n")
    source_hash = adapter.sha256(source_commitment)
    job = "135999"
    ref = f"refs/atypemu-authorizations/k32-source/job-{job}"
    authorization = {
        "contract": gate_authorization.AUTHORIZATION_CONTRACT,
        "source_commitment_sha256": source_hash,
        "epochs": 3,
        "steps": 5,
        "authorized": True,
        "container_image_sha256": "1" * 64,
        "slurm_job_id": job,
        "authorization_ref": ref,
    }
    authorization_path = temporary / "authorization.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    authorization_blob = gate_authorization._git_blob(authorization_path.read_bytes())
    spec = gate_authorization.AuthorizationSpec(
        source_hash, 3, 5, "1" * 64, job, ref, authorization_blob,
    )
    claim_blob = None
    def fake_git(command, **kwargs):
        global claim_blob
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, authorization_blob + "\n", "")
        if "hash-object" in command:
            claim_blob = gate_authorization._git_blob(Path(command[-1]).read_bytes())
            return subprocess.CompletedProcess(command, 0, claim_blob + "\n", "")
        if "update-ref" in command:
            assert command[-2] == claim_blob and command[-1] == authorization_blob
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)
    consumed_path = temporary / "consumed.json"
    claim_path = temporary / "claim.json"
    with mock.patch.object(gate_authorization.subprocess, "run", side_effect=fake_git):
        capability = gate_authorization.consume_authorization(
            authorization_path, source_commitment, consumed_path, claim_path,
            spec=spec, authorization_git_dir=temporary / "external.git",
        )
    assert capability.source_commitment_sha256 == source_hash
    assert consumed_path.is_file() and claim_path.is_file()
    try:
        gate_authorization.consume_authorization(
            authorization_path, source_commitment, consumed_path, claim_path,
            spec=spec, authorization_git_dir=temporary / "external.git",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("consumed authorization was reusable")
residue_ids = adapter.residue_inventory(surface)
zero_delta = np.zeros((len(residue_ids), 4), np.float32)
zero_state = adapter.emit_coordinate_state(
    Path("."), entity_uid="bmrb:10109:entity:1", support_id="BioEmu_1",
    residue_ids=residue_ids, residue_delta=zero_delta,
)
assert np.array_equal(zero_state.base, zero_state.conditioned)
movable = next(
    int(seq_id) for seq_id, residue in zip(zero_state.atom_seq_id, zero_state.residue_name)
    if residue in adapter.SIDECHAIN_CHI_BONDS and int(seq_id) in residue_ids
)
delta = zero_delta.copy()
delta[residue_ids.index(movable), 0] = 0.01
moved_state = adapter.emit_coordinate_state(
    Path("."), entity_uid="bmrb:10109:entity:1", support_id="BioEmu_1",
    residue_ids=residue_ids, residue_delta=delta,
)
maximum_displacement, minimum_distance = adapter.audit_coordinate_state(
    moved_state, require_motion=True,
)
assert 0 < maximum_displacement <= 1 and minimum_distance >= 0.5
assert np.array_equal(moved_state.base, zero_state.base)
base_replay, _ = adapter.recompute_fixed_neighbor_distances(
    surface, zero_state, support_number=0,
)
available = surface.arrays[5][:, 0]
assert np.max(np.abs(base_replay[available] - surface.arrays[0][:, 0][available])) < 2e-5
jacobian_magnitude = np.abs(surface.arrays[1][:, 0]) + np.abs(surface.arrays[2][:, 0])
row, element, chi = np.unravel_index(
    np.argmax(jacobian_magnitude), jacobian_magnitude.shape,
)
self_part = abs(surface.arrays[1][row, 0, element, chi])
neighbor_part = abs(surface.arrays[2][row, 0, element, chi])
actuated_residue = (
    int(surface.row_context[0][row])
    if self_part >= neighbor_part
    else int(surface.arrays[3][row, 0, element])
)
linear_delta = np.zeros_like(zero_delta)
linear_delta[residue_ids.index(actuated_residue), chi] = 1e-4
linear_state = adapter.emit_coordinate_state(
    Path("."), entity_uid="bmrb:10109:entity:1", support_id="BioEmu_1",
    residue_ids=residue_ids, residue_delta=linear_delta,
)
linear_base, exact_moved = adapter.recompute_fixed_neighbor_distances(
    surface, linear_state, support_number=0,
)
residue_lookup = {seq_id: index for index, seq_id in enumerate(residue_ids)}
self_delta = linear_delta[[residue_lookup[int(value)] for value in surface.row_context[0]]]
neighbor_delta = np.zeros((*surface.arrays[3][:, 0].shape, 4), np.float32)
for target_row in range(len(surface.target_ids)):
    for descriptor in range(5):
        neighbor_seq_id = int(surface.arrays[3][target_row, 0, descriptor])
        if neighbor_seq_id in residue_lookup:
            neighbor_delta[target_row, descriptor] = linear_delta[residue_lookup[neighbor_seq_id]]
linearized = (
    linear_base
    + np.einsum("nec,nc->ne", surface.arrays[1][:, 0], self_delta)
    + np.einsum("nec,nec->ne", surface.arrays[2][:, 0], neighbor_delta)
)
assert np.max(np.abs(exact_moved[available] - linearized[available])) < 2e-5
assert np.max(np.abs(exact_moved[available] - linear_base[available])) > 1e-7
print("METRIC matched_adapter_checks=84")
print("METRIC source_target_values_read=0")
print("METRIC outer_or_formal_metrics_opened=0")
PY
draft="$(mktemp)"
rm -f "$draft"
$PY gpuopt/freeze_k32_nested_k8_source_gate_draft.py --root . --output "$draft"
$PY - "$draft" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text())
assert receipt["draft_only"] is True and receipt["authorization_allowed"] is False
assert receipt["runner_and_checker_pending"] is True
assert receipt["source_entity_count"] == receipt["target_file_count"] == receipt["cache_file_count"] == 135
from gpuopt.freeze_k32_nested_k8_source_gate_draft import STATIC_FILES
assert len(receipt["files"]) == len(STATIC_FILES) + 270
for relative, expected in receipt["files"].items():
    assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected
try:
    from unittest import mock
    from gpuopt.freeze_k32_nested_k8_source_gate_draft import main
    with mock.patch("sys.argv", ["freeze", "--root", ".", "--output", sys.argv[1]]):
        main()
except FileExistsError:
    pass
else:
    raise AssertionError("draft source commitment was clobbered")
PY
preflight="$(mktemp)"
rm -f "$preflight"
$PY gpuopt/run_k32_nested_k8_source_gate.py \
  --root . --draft-commitment "$draft" --preflight-output "$preflight" --preflight-only
$PY - "$preflight" <<'PY'
import ast
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text())
assert receipt["production_ready"] is False
assert receipt["authorization_consumed"] is False
assert receipt["source_target_values_read"] is False
assert receipt["formal_or_outer_access"] is False
assert receipt["crossfit_cells"] == 4 and receipt["support_count"] == 32
assert receipt["source_target_reader_calls"] == 0 and receipt["anchor_rows"] > 0
tree = ast.parse(Path("gpuopt/run_k32_nested_k8_source_gate.py").read_text())
top_imports = set()
for node in tree.body:
    if isinstance(node, ast.Import):
        top_imports.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        top_imports.add(str(node.module).split(".")[0])
assert top_imports <= {"__future__", "argparse", "hashlib", "json", "pathlib", "sys", "typing"}
print("METRIC matched_adapter_checks=98")
PY
rm -f "$preflight"
rm -f "$draft"
/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/k32_nested_k8_adapter.py \
  gpuopt/k32_source_gate_authorization.py \
  gpuopt/freeze_k32_nested_k8_source_gate_draft.py \
  gpuopt/run_k32_nested_k8_source_gate.py
