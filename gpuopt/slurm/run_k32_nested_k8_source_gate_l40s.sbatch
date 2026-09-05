#!/bin/bash
#SBATCH --job-name=k32-nested-k8-gate
#SBATCH --partition=l40sq
#SBATCH --nodelist=iREMB-C-08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=k32-nested-k8-gate.%j.out
#SBATCH --error=k32-nested-k8-gate.%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load Singularity/4.3.4
STAGE="${STAGE:-/scratch/wkyu514/yang07/atypemu_k32_nested_k8_gate_20260905}"
IMAGE="${IMAGE:-/scratch/wkyu514/yang07/AtypEmu/atypemu.sif}"
IMAGE_SHA256=67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531
COMMITMENT="${SOURCE_COMMITMENT:-$STAGE/results/source_commitment.json}"
AUTH="${AUTHORIZATION:-$STAGE/results/authorization.json}"
CONSUMED="${CONSUMED:-$STAGE/results/consumed_authorization.json}"
CLAIM="${EXTERNAL_CLAIM:-$STAGE/results/external_claim.json}"
OUTPUT="${OUTPUT_DIR:-$STAGE/results/run}"
AUTH_GIT=/scratch/wkyu514/yang07/atypemu_external_authorizations.git
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && -f "$IMAGE" && -f "$COMMITMENT" && -d "$AUTH_GIT" ]]
for path in "$AUTH" "$CONSUMED" "$CLAIM" "$OUTPUT"; do
  case "$path" in "$STAGE"/*) ;; *) exit 65;; esac
  [[ ! -e "$path" ]] || exit 66
done
ACTUAL_IMAGE_SHA256=$(sha256sum "$IMAGE" | cut -d' ' -f1)
[[ "$ACTUAL_IMAGE_SHA256" == "$IMAGE_SHA256" ]]
AUTH_REF="refs/atypemu-authorizations/k32-source/job-$SLURM_JOB_ID"
AUTH_BLOB=$(git --git-dir="$AUTH_GIT" rev-parse "$AUTH_REF^{blob}")
set -o noclobber
git --git-dir="$AUTH_GIT" cat-file blob "$AUTH_BLOB" > "$AUTH"
set +o noclobber
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
singularity exec --nv --bind "$STAGE:/workspace" --bind "$AUTH_GIT:$AUTH_GIT" \
  --pwd /workspace "$IMAGE" env PYTHONPATH=/workspace \
  python gpuopt/run_k32_nested_k8_source_gate.py --root /workspace \
  --source-commitment "/workspace/${COMMITMENT#"$STAGE"/}" \
  --authorization "/workspace/${AUTH#"$STAGE"/}" \
  --consumed "/workspace/${CONSUMED#"$STAGE"/}" \
  --external-claim "/workspace/${CLAIM#"$STAGE"/}" \
  --authorization-git-dir "$AUTH_GIT" --authorization-ref "$AUTH_REF" \
  --authorization-git-blob "$AUTH_BLOB" \
  --container-image-sha256 "$ACTUAL_IMAGE_SHA256" --slurm-job-id "$SLURM_JOB_ID" \
  --epochs 1024 --steps 100 --device cuda \
  --output-dir "/workspace/${OUTPUT#"$STAGE"/}"
