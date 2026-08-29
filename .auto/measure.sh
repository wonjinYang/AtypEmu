#!/usr/bin/env bash
set -euo pipefail

PY=/home/yang07/anaconda3/envs/atypemu/bin/python
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT="$ROOT/.auto/runs/current"

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

if ! "$PY" "$ROOT/gpuopt/run_all_label_candidate.py" \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --inventory "$ROOT/.auto/frozen/all_label_inventory.json" \
  --output-root "$RUN_ROOT" \
  >"$RUN_ROOT/candidate.log" 2>&1; then
  cat "$RUN_ROOT/candidate.log" >&2
  exit 1
fi

"$PY" "$ROOT/gpuopt/all_label_one_shared_q_metric.py" score \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --inventory "$ROOT/.auto/frozen/all_label_inventory.json" \
  --commitment "$ROOT/.auto/frozen/evaluator_commitment.json" \
  --surface "$RUN_ROOT/surface_no_coordinate_A_to_B.parquet" \
  --surface "$RUN_ROOT/surface_no_coordinate_B_to_A.parquet" \
  --q "$RUN_ROOT/q_A_to_B.parquet" \
  --q "$RUN_ROOT/q_B_to_A.parquet" \
  --output "$RUN_ROOT/no_coordinate_score.json" \
  >"$RUN_ROOT/no_coordinate_score.log"

"$PY" "$ROOT/gpuopt/all_label_one_shared_q_metric.py" score \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --inventory "$ROOT/.auto/frozen/all_label_inventory.json" \
  --commitment "$ROOT/.auto/frozen/evaluator_commitment.json" \
  --surface "$RUN_ROOT/surface_A_to_B.parquet" \
  --surface "$RUN_ROOT/surface_B_to_A.parquet" \
  --q "$RUN_ROOT/q_uniform_A_to_B.parquet" \
  --q "$RUN_ROOT/q_uniform_B_to_A.parquet" \
  --output "$RUN_ROOT/uniform_q_score.json" \
  >"$RUN_ROOT/uniform_q_score.log"

"$PY" "$ROOT/gpuopt/all_label_one_shared_q_metric.py" score \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --inventory "$ROOT/.auto/frozen/all_label_inventory.json" \
  --commitment "$ROOT/.auto/frozen/evaluator_commitment.json" \
  --surface "$RUN_ROOT/surface_A_to_B.parquet" \
  --surface "$RUN_ROOT/surface_B_to_A.parquet" \
  --q "$RUN_ROOT/q_A_to_B.parquet" \
  --q "$RUN_ROOT/q_B_to_A.parquet" \
  --output "$RUN_ROOT/score.json"

"$PY" "$ROOT/gpuopt/verify_all_label_one_shared_q_score.py" \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --inventory "$ROOT/.auto/frozen/all_label_inventory.json" \
  --score "$RUN_ROOT/score.json" \
  --surface "$RUN_ROOT/surface_A_to_B.parquet" \
  --surface "$RUN_ROOT/surface_B_to_A.parquet" \
  --q "$RUN_ROOT/q_A_to_B.parquet" \
  --q "$RUN_ROOT/q_B_to_A.parquet"
