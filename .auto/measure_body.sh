#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_HOST=iremb6-server
STAGE=/scratch/wkyu514/yang07/atypemu_corrected_k8_gate_20260905
AUTH_REPO=/scratch/wkyu514/yang07/atypemu_external_authorizations.git
SOURCE_REF=refs/atypemu-source-commitments/corrected-k8/final-v7
SOURCE_COMMITMENT="$ROOT/.auto/staging/corrected_k8_source_commitment.json"
IMAGE_SHA=67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531
SOURCE_SHA=$(sha256sum "$SOURCE_COMMITMENT" | awk '{print $1}')
SOURCE_BLOB=$(git hash-object "$SOURCE_COMMITMENT")
: "${EXPECTED_SOURCE_COMMITMENT_SHA256:?must be pinned by the experiment command}"
: "${EXPECTED_SOURCE_COMMITMENT_GIT_BLOB:?must be pinned by the experiment command}"
test "$SOURCE_SHA" = "$EXPECTED_SOURCE_COMMITMENT_SHA256"
test "$SOURCE_BLOB" = "$EXPECTED_SOURCE_COMMITMENT_GIT_BLOB"
JOB_ID=""
RELEASED=0

cleanup_held_job() {
  if [[ -n "$JOB_ID" && "$RELEASED" == 0 ]]; then
    ssh -o BatchMode=yes "$REMOTE_HOST" "scancel '$JOB_ID'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_held_job EXIT

test ! -e "$ROOT/.auto/runs/current"
REMOTE_SOURCE_SHA=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "sha256sum '$STAGE/results/corrected_k8_source_commitment.json'" |
    awk '{print $1}'
)
LOCAL_PINNED_SOURCE_BLOB=$(git rev-parse "$SOURCE_REF^{blob}")
REMOTE_PINNED_SOURCE_BLOB=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "git --git-dir='$AUTH_REPO' rev-parse '$SOURCE_REF^{blob}'"
)
REMOTE_PINNED_SOURCE_SHA=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "git --git-dir='$AUTH_REPO' cat-file blob '$REMOTE_PINNED_SOURCE_BLOB'" |
    sha256sum | awk '{print $1}'
)
REMOTE_REFS=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "git --git-dir='$AUTH_REPO' for-each-ref --format='%(refname)' refs/atypemu-authorizations/corrected-k8/"
)
test "$REMOTE_SOURCE_SHA" = "$SOURCE_SHA"
test "$LOCAL_PINNED_SOURCE_BLOB" = "$SOURCE_BLOB"
test "$REMOTE_PINNED_SOURCE_BLOB" = "$SOURCE_BLOB"
test "$REMOTE_PINNED_SOURCE_SHA" = "$SOURCE_SHA"
test -z "$REMOTE_REFS"

JOB_ID=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$STAGE' && MODE=gate sbatch --hold --parsable gpuopt/slurm/run_corrected_k8_dynamic_coordinate_source_gate_l40s.sbatch"
)
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
AUTH_REF="refs/atypemu-authorizations/corrected-k8/job-$JOB_ID"
AUTH_LOCAL="$ROOT/.auto/staging/corrected_k8_source_authorization_job${JOB_ID}.json"
export JOB_ID AUTH_REF AUTH_LOCAL SOURCE_SHA IMAGE_SHA
python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "authorization_ref": os.environ["AUTH_REF"],
    "authorized": True,
    "container_image_sha256": os.environ["IMAGE_SHA"],
    "contract": "corrected_k8_dynamic_coordinate_source_gate_authorization_v1",
    "epochs": 1024,
    "formal_evaluation_authorized": False,
    "slurm_job_id": os.environ["JOB_ID"],
    "source_commitment_sha256": os.environ["SOURCE_SHA"],
    "source_only": True,
    "steps": 100,
}
path = Path(os.environ["AUTH_LOCAL"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

AUTH_BLOB=$(git hash-object -w "$AUTH_LOCAL")
REMOTE_BLOB=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "git --git-dir='$AUTH_REPO' hash-object -w --stdin" <"$AUTH_LOCAL"
)
test "$REMOTE_BLOB" = "$AUTH_BLOB"
ZERO=0000000000000000000000000000000000000000
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "git --git-dir='$AUTH_REPO' update-ref '$AUTH_REF' '$AUTH_BLOB' '$ZERO'; test \$(git --git-dir='$AUTH_REPO' rev-parse '$AUTH_REF^{blob}') = '$AUTH_BLOB'"
git update-ref "$AUTH_REF" "$AUTH_BLOB" "$ZERO"
test "$(git rev-parse "$AUTH_REF^{blob}")" = "$AUTH_BLOB"

ssh -o BatchMode=yes "$REMOTE_HOST" "scontrol release '$JOB_ID'"
RELEASED=1
trap - EXIT
echo "CORRECTED_K8_SLURM_JOB_ID=$JOB_ID"

while ssh -o BatchMode=yes "$REMOTE_HOST" "squeue -h -j '$JOB_ID'" | grep -q .; do
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "squeue -h -j '$JOB_ID' -o '%i %T %M %R'" || true
  sleep 30
done
STATE=$(
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "sacct -X -n -j '$JOB_ID' --format=State | head -1 | xargs"
)

TMP_RESULT="$ROOT/.auto/runs/corrected_k8_job${JOB_ID}.tmp"
test ! -e "$TMP_RESULT"
mkdir -p "$TMP_RESULT"
RESULT_RSYNC_RC=0
RSYNC_RSH='ssh -o BatchMode=yes' rsync -az \
  "$REMOTE_HOST:$STAGE/results/" "$TMP_RESULT/" || RESULT_RSYNC_RC=$?
RSYNC_RSH='ssh -o BatchMode=yes' rsync -az \
  "$REMOTE_HOST:$STAGE/corrected-k8-gate.$JOB_ID.out" \
  "$REMOTE_HOST:$STAGE/corrected-k8-gate.$JOB_ID.err" \
  "$TMP_RESULT/" || true
if [[ "$RESULT_RSYNC_RC" != 0 ]]; then
  for name in authorization.json consumed_authorization.json external_claim.json; do
    if [[ ! -s "$TMP_RESULT/$name" ]]; then
      RSYNC_RSH='ssh -o BatchMode=yes' rsync -az \
        "$REMOTE_HOST:$STAGE/results/$name" "$TMP_RESULT/$name" || true
    fi
  done
fi

# Mirror the irreversible external claim into the local Git repository even
# when post-consumption execution fails, so failure evidence remains bound.
if [[ -s "$TMP_RESULT/external_claim.json" ]]; then
  CLAIM_BLOB=$(git hash-object -w "$TMP_RESULT/external_claim.json")
  REMOTE_TIP=$(
    ssh -o BatchMode=yes "$REMOTE_HOST" \
      "git --git-dir='$AUTH_REPO' rev-parse '$AUTH_REF^{blob}'"
  )
  test "$REMOTE_TIP" = "$CLAIM_BLOB"
  git update-ref "$AUTH_REF" "$CLAIM_BLOB" "$AUTH_BLOB"
  test "$(git rev-parse "$AUTH_REF^{blob}")" = "$CLAIM_BLOB"
fi

if [[ "$STATE" != COMPLETED* || "$RESULT_RSYNC_RC" != 0 ]]; then
  (
    set -o noclobber
    printf 'slurm_job_id=%s\nstate=%s\nresult_rsync_rc=%s\nremote_stage=%s\nauthorization_ref=%s\nsource_commitment_sha256=%s\n' \
      "$JOB_ID" "$STATE" "$RESULT_RSYNC_RC" "$STAGE" "$AUTH_REF" "$SOURCE_SHA" \
      >"$TMP_RESULT/transfer_status.txt"
  )
  FAILED_RESULT="$ROOT/.auto/runs/failed_corrected_k8_job${JOB_ID}"
  test ! -e "$FAILED_RESULT"
  mv "$TMP_RESULT" "$FAILED_RESULT"
  tail -160 "$FAILED_RESULT/corrected-k8-gate.$JOB_ID.out" 2>/dev/null || true
  tail -160 "$FAILED_RESULT/corrected-k8-gate.$JOB_ID.err" 2>/dev/null || true
  echo "SLURM_JOB_ID=$JOB_ID STATE=$STATE"
  exit 1
fi

for name in \
  corrected_k8_source_commitment.json \
  authorization.json \
  consumed_authorization.json \
  external_claim.json \
  unchecked_result.json \
  decision.json \
  source_oof_A.parquet \
  source_oof_B.parquet \
  source_coordinate_A_half0.npz \
  source_coordinate_A_half1.npz \
  source_coordinate_B_half0.npz \
  source_coordinate_B_half1.npz; do
  test -s "$TMP_RESULT/$name"
done
mv "$TMP_RESULT" "$ROOT/.auto/runs/current"
printf '%s\n' '{"contract":"atypemu_autoresearch_check_mode_v1","mode":"corrected_k8_dynamic_coordinate_source_gate"}' \
  >"$ROOT/.auto/runs/current/check_mode.json"

export DECISION="$ROOT/.auto/runs/current/decision.json"
python3 - <<'PY'
import json
import os

with open(os.environ["DECISION"]) as handle:
    decision = json.load(handle)
gains = [float(decision["folds"][fold]["gain"]) for fold in ("A", "B")]
gains.append(float(decision["combined"]["gain"]))
print(f"SLURM_JOB_ID={decision['slurm_job_id']}")
print(f"METRIC min_gain_over_0.0005={min(gains) - 0.0005:.12f}")
print(f"METRIC source_A_gain={gains[0]:.12f}")
print(f"METRIC source_B_gain={gains[1]:.12f}")
print(f"METRIC source_combined_gain={gains[2]:.12f}")
print(
    "METRIC selected_for_k32="
    f"{int(bool(decision['selected_for_k32_followup']))}"
)
PY
