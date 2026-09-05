#!/bin/bash
#SBATCH --job-name=corrected-k8-gate
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=corrected-k8-gate.%j.out
#SBATCH --error=corrected-k8-gate.%j.err

set -eo pipefail
source /etc/profile.d/lmod.sh
module load Singularity/4.3.4
set -u

STAGE="${STAGE:-/scratch/wkyu514/yang07/atypemu_corrected_k8_gate_20260905}"
IMAGE="${IMAGE:-/scratch/wkyu514/yang07/AtypEmu/atypemu.sif}"
EXPECTED_IMAGE_SHA256=67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531
MODE="${MODE:-gate}"
OUTPUT="${OUTPUT:-$STAGE/results/unchecked_result.json}"
DECISION="${DECISION:-$STAGE/results/decision.json}"
SOURCE_COMMITMENT="${SOURCE_COMMITMENT:-$STAGE/results/corrected_k8_source_commitment.json}"
AUTHORIZATION="${AUTHORIZATION:-$STAGE/results/authorization.json}"
CONSUMED_AUTHORIZATION="${CONSUMED_AUTHORIZATION:-$STAGE/results/consumed_authorization.json}"
EXTERNAL_CLAIM="${EXTERNAL_CLAIM:-$STAGE/results/external_claim.json}"
AUTH_GIT_DIR="/scratch/wkyu514/yang07/atypemu_external_authorizations.git"
EPOCHS="${EPOCHS:-1024}"
STEPS="${STEPS:-100}"
mkdir -p "$STAGE/results" "$STAGE/cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
SINGULARITY_BINDS=(--bind "$STAGE:/workspace")
[[ -f "$IMAGE" ]]
ACTUAL_IMAGE_SHA256=$(sha256sum "$IMAGE" | cut -d' ' -f1)
[[ "$ACTUAL_IMAGE_SHA256" == "$EXPECTED_IMAGE_SHA256" ]]

if [[ "$MODE" == "test" ]]; then
  COMMAND=(python -m unittest -v gpuopt.tests.test_corrected_k8_dynamic_coordinate_source_gate)
elif [[ "$MODE" == "gate" ]]; then
  [[ "$EPOCHS" == "1024" && "$STEPS" == "100" ]]
  for PATH_VALUE in "$OUTPUT" "$DECISION" "$SOURCE_COMMITMENT" "$AUTHORIZATION" "$CONSUMED_AUTHORIZATION" "$EXTERNAL_CLAIM"; do
    case "$PATH_VALUE" in
      "$STAGE"/*) ;;
      *) echo "gate path escapes stage: $PATH_VALUE" >&2; exit 65 ;;
    esac
  done
  [[ -f "$SOURCE_COMMITMENT" && -f "$IMAGE" && -d "$AUTH_GIT_DIR" ]]
  SINGULARITY_BINDS+=(--bind "$AUTH_GIT_DIR:$AUTH_GIT_DIR")
  OUTPUT_DIR=$(dirname "$OUTPUT")
  for GENERATED in \
    "$OUTPUT" "$DECISION" "$AUTHORIZATION" "$CONSUMED_AUTHORIZATION" "$EXTERNAL_CLAIM" \
    "$OUTPUT_DIR/source_oof_A.parquet" "$OUTPUT_DIR/source_oof_B.parquet" \
    "$OUTPUT_DIR/source_coordinate_A_half0.npz" \
    "$OUTPUT_DIR/source_coordinate_A_half1.npz" \
    "$OUTPUT_DIR/source_coordinate_B_half0.npz" \
    "$OUTPUT_DIR/source_coordinate_B_half1.npz"; do
    [[ ! -e "$GENERATED" ]] || {
      echo "refusing stale corrected K8 output: $GENERATED" >&2
      exit 66
    }
  done
  AUTHORIZATION_REF="refs/atypemu-authorizations/corrected-k8/job-$SLURM_JOB_ID"
  AUTHORIZATION_GIT_BLOB=$(git --git-dir="$AUTH_GIT_DIR" rev-parse "$AUTHORIZATION_REF^{blob}")
  git --git-dir="$AUTH_GIT_DIR" cat-file -e "$AUTHORIZATION_GIT_BLOB^{blob}"
  set -o noclobber
  git --git-dir="$AUTH_GIT_DIR" cat-file blob "$AUTHORIZATION_GIT_BLOB" > "$AUTHORIZATION"
  set +o noclobber
  AUTHORIZED_IMAGE_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["container_image_sha256"])' "$AUTHORIZATION")
  [[ "$AUTHORIZED_IMAGE_SHA256" == "$EXPECTED_IMAGE_SHA256" ]]
  COMMAND=(python gpuopt/run_corrected_k8_dynamic_coordinate_source_gate.py
    --root /workspace
    --output "/workspace/${OUTPUT#"$STAGE"/}"
    --source-commitment "/workspace/${SOURCE_COMMITMENT#"$STAGE"/}"
    --authorization "/workspace/${AUTHORIZATION#"$STAGE"/}"
    --consumed-authorization "/workspace/${CONSUMED_AUTHORIZATION#"$STAGE"/}"
    --external-authorization-claim "/workspace/${EXTERNAL_CLAIM#"$STAGE"/}"
    --authorization-git-blob "$AUTHORIZATION_GIT_BLOB"
    --authorization-ref "$AUTHORIZATION_REF"
    --container-image-sha256 "$ACTUAL_IMAGE_SHA256"
    --epochs "$EPOCHS"
    --steps "$STEPS")
else
  echo "unknown MODE=$MODE" >&2
  exit 64
fi

singularity exec --nv \
  "${SINGULARITY_BINDS[@]}" \
  --pwd /workspace \
  "$IMAGE" \
  env PYTHONPATH=/workspace "${COMMAND[@]}"

if [[ "$MODE" == "gate" ]]; then
  singularity exec --nv \
    --bind "$STAGE:/workspace" \
    --bind "$AUTH_GIT_DIR:$AUTH_GIT_DIR:ro" \
    --pwd /workspace \
    "$IMAGE" \
    env PYTHONPATH=/workspace \
    python gpuopt/check_corrected_k8_dynamic_coordinate_source_gate.py \
      --root /workspace \
      --result "/workspace/${OUTPUT#"$STAGE"/}" \
      --decision "/workspace/${DECISION#"$STAGE"/}" \
      --source-commitment "/workspace/${SOURCE_COMMITMENT#"$STAGE"/}" \
      --authorization "/workspace/${AUTHORIZATION#"$STAGE"/}" \
      --consumed-authorization "/workspace/${CONSUMED_AUTHORIZATION#"$STAGE"/}" \
      --external-authorization-claim "/workspace/${EXTERNAL_CLAIM#"$STAGE"/}" \
      --authorization-git-dir "$AUTH_GIT_DIR" \
      --authorization-git-blob "$AUTHORIZATION_GIT_BLOB" \
      --authorization-ref "$AUTHORIZATION_REF" \
      --container-image-sha256 "$ACTUAL_IMAGE_SHA256" \
      --slurm-job-id "$SLURM_JOB_ID"
fi
