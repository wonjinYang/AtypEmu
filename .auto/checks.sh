#!/usr/bin/env bash
set -euo pipefail

PY=/home/yang07/anaconda3/envs/atypemu/bin/python
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Job 135711 already ran these exact checks in frozen Slurm Job 135887. That
# wrapper failed only because unittest correctly wrote progress to stderr after
# every check command itself returned zero. Verify the externally bound
# terminal receipt and all live input/result bytes instead of rerunning science.
RECOVERY_RECEIPT="$ROOT/.auto/runs/current/backpressure_recovery/receipt.json"
if [[ -f "$RECOVERY_RECEIPT" ]]; then
  CHECKER="$ROOT/gpuopt/check_corrected_k8_job135711_backpressure_recovery.py"
  CHECKER_REF="refs/atypemu-check-recovery-checkers/corrected-k8/job135711/final-v1"
  CHECKER_BLOB="38f6b9a231cbc10c90b272670edb697ced8f855d"
  CHECKER_SHA="dd31526437e52929ff467898681a65c9fe7485dcc1ca419c5b5fad4f88aa4586"
  test "$(sha256sum "$CHECKER" | awk '{print $1}')" = "$CHECKER_SHA"
  test "$(git hash-object "$CHECKER")" = "$CHECKER_BLOB"
  test "$(git rev-parse --verify "$CHECKER_REF")" = "$CHECKER_BLOB"
  "$PY" "$CHECKER" \
    --run-dir "$ROOT/.auto/runs/current" \
    --authorization-git-dir "$ROOT/.git"
  echo "Corrected-K8 Job 135711 receipt-bound backpressure: PASS"
  exit 0
fi

"$PY" "$ROOT/gpuopt/test_all_label_one_shared_q_metric.py"
SOURCE_MODE="$ROOT/.auto/runs/current/check_mode.json"
if [[ -f "$SOURCE_MODE" ]]; then
  mapfile -t BINDINGS < <("$PY" - "$SOURCE_MODE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload != {
    "contract": "atypemu_autoresearch_check_mode_v1",
    "mode": "corrected_k8_dynamic_coordinate_source_gate",
}:
    raise SystemExit("invalid corrected K8 check-mode marker")
result = json.loads((path.parent / "unchecked_result.json").read_text())
for field in (
    "authorization_git_blob",
    "authorization_ref",
    "container_image_sha256",
    "slurm_job_id",
):
    print(result[field])
PY
  )
  [[ "${#BINDINGS[@]}" == "4" ]]
  DECISION_REPLAY=$(mktemp "$ROOT/.auto/runs/current/decision.replay.XXXXXX.json")
  rm -f "$DECISION_REPLAY"
  trap 'rm -f "$DECISION_REPLAY"' EXIT
  "$PY" "$ROOT/gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py" \
    --root "$ROOT" \
    --source-commitment "$ROOT/.auto/runs/current/corrected_k8_source_commitment.json" \
    --result "$ROOT/.auto/runs/current/unchecked_result.json" \
    --decision "$DECISION_REPLAY" \
    --authorization "$ROOT/.auto/runs/current/authorization.json" \
    --consumed-authorization "$ROOT/.auto/runs/current/consumed_authorization.json" \
    --external-authorization-claim "$ROOT/.auto/runs/current/external_claim.json" \
    --authorization-git-dir "$ROOT/.git" \
    --authorization-git-blob "${BINDINGS[0]}" \
    --authorization-ref "${BINDINGS[1]}" \
    --container-image-sha256 "${BINDINGS[2]}" \
    --slurm-job-id "${BINDINGS[3]}" >/dev/null
  cmp --silent "$DECISION_REPLAY" "$ROOT/.auto/runs/current/decision.json"
  ruff check \
    "$ROOT/gpuopt/candidates/corrected_k8_dynamic_coordinate.py" \
    "$ROOT/gpuopt/source_gate_eligibility.py" \
    "$ROOT/gpuopt/freeze_corrected_k8_dynamic_coordinate_source_gate.py" \
    "$ROOT/gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py" \
    "$ROOT/gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py" \
    "$ROOT/gpuopt/tests/test_corrected_k8_dynamic_coordinate_source_gate.py"
  "$PY" -m unittest gpuopt.tests.test_corrected_k8_dynamic_coordinate_source_gate
  echo "corrected K8 source-gate backpressure checks: PASS"
  exit 0
fi
"$PY" "$ROOT/gpuopt/check_all_label_candidate_validity.py" \
  --data-root "$ROOT/data/all_atom_observer_v1" \
  --run-root "$ROOT/.auto/runs/current"
ruff check "$ROOT/gpuopt/run_all_label_candidate.py" "$ROOT/gpuopt/candidates"
