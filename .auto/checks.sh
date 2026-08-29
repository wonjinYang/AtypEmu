#!/usr/bin/env bash
set -euo pipefail

PY=/home/yang07/anaconda3/envs/atypemu/bin/python
ROOT=$(cd "$(dirname "$0")/.." && pwd)

"$PY" "$ROOT/gpuopt/test_all_label_one_shared_q_metric.py"
"$PY" "$ROOT/gpuopt/check_all_label_candidate_validity.py" \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --run-root "$ROOT/.auto/runs/current"
ruff check "$ROOT/gpuopt/run_all_label_candidate.py" "$ROOT/gpuopt/candidates"
