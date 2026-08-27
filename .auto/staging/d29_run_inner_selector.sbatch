#!/bin/bash
#SBATCH --partition=h200q
#SBATCH --nodelist=iREMB-C-06
#SBATCH --gres=gpu:h200_01:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append

set -euo pipefail
umask 077
for control_name in ${!SINGULARITY@} ${!APPTAINER@} ${!SBATCH@}; do unset "$control_name"; done
export PATH=/usr/bin:/bin
: "${SOURCE:?}" "${SOURCE_BUNDLE_SHA256:?}" "${INPUT_ROOT:?}" "${INPUT_SHA256:?}" \
  "${TRAINER_SHA256:?}" "${OUTPUT_ROOT:?}" "${IMAGE_SHA256:?}"
ROOT=/scratch/wkyu514/yang07/AtypEmu
IMAGE=$ROOT/atypemu.sif
CONTAINER=/home/APP/Software/Singularity/4.3.4/bin/singularity
RESULT=$OUTPUT_ROOT/result.json
[[ -n ${CUDA_VISIBLE_DEVICES:-} && ! -e $RESULT ]]
(cd "$SOURCE" && sha256sum -c checksums.sha256 >/dev/null)
(cd "$INPUT_ROOT" && sha256sum -c checksums.sha256 >/dev/null)
python3 - "$SOURCE/inner_selector_source_receipt.json" "$SOURCE_BUNDLE_SHA256" \
  "$INPUT_SHA256" "$IMAGE_SHA256" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['bundle_sha256']==sys.argv[2]
assert p['bindings']['input_manifest_sha256']==sys.argv[3]
assert p['bindings']['container_image_sha256']==sys.argv[4]
PY
[[ $(sha256sum "$INPUT_ROOT/input_manifest.json" | cut -d' ' -f1) == "$INPUT_SHA256" ]]
[[ $(sha256sum "$SOURCE/analysis/v3339_train_atom27_phase_d.py" | cut -d' ' -f1) == "$TRAINER_SHA256" ]]
[[ $(sha256sum "$IMAGE" | cut -d' ' -f1) == "$IMAGE_SHA256" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p') == 'NVIDIA H200' ]]
mkdir -p "$OUTPUT_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 PYTHONHASHSEED=3339
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
container=("$CONTAINER" exec --nv --cleanenv --containall --no-home --net --network none
  --env CUDA_VISIBLE_DEVICES=0 --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env OMP_NUM_THREADS=8 --env PYTHONHASHSEED=3339
  --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 --env WANDB_MODE=disabled
  --bind "$SOURCE:$SOURCE:ro" --bind "$INPUT_ROOT:$INPUT_ROOT:ro"
  --bind "$OUTPUT_ROOT:$OUTPUT_ROOT:rw")
baseline_args=()
if [[ -n ${BASELINE_RECEIPT:-} ]]; then
  [[ -n ${BASELINE_RECEIPT_SHA256:-} ]]
  [[ $(sha256sum "$BASELINE_RECEIPT" | cut -d' ' -f1) == "$BASELINE_RECEIPT_SHA256" ]]
  container+=(--bind "$BASELINE_RECEIPT:$BASELINE_RECEIPT:ro")
  baseline_args+=(--baseline-receipt "$BASELINE_RECEIPT"
    --baseline-receipt-sha256 "$BASELINE_RECEIPT_SHA256")
fi
container+=("$IMAGE")
if [[ ${ATYP_IMPORT_SMOKE_ONLY:-0} == 1 ]]; then
  "${container[@]}" /opt/conda/bin/python -I -c \
    "import importlib.util; p='$SOURCE/analysis/v3339_train_atom27_phase_d.py'; s=importlib.util.spec_from_file_location('v3339_import_smoke',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('IMPORT_SMOKE_OK',m.D2_SOURCE_SHA256,m.V6_TRAINER_SHA256)"
  exit 0
fi
"${container[@]}" /opt/conda/bin/python -I "$SOURCE/analysis/v3339_inner_chart_selector.py" \
  --trainer "$SOURCE/analysis/v3339_train_atom27_phase_d.py" \
  --trainer-sha256 "$TRAINER_SHA256" \
  --input-manifest "$INPUT_ROOT/input_manifest.json" \
  --input-manifest-sha256 "$INPUT_SHA256" \
  --menu "$SOURCE/clean_menu_builder_output.json" \
  --stage-a-source "$SOURCE/analysis/v3339_train_all_bmrb_stage_a.py" \
  --q-transfer-receipt "$SOURCE/q_initialization_transfer_receipt_d4v2.json" \
  --fixed-support-source "$SOURCE/deps/fixed_support_measure.py" \
  --source-bundle-sha256 "$SOURCE_BUNDLE_SHA256" \
  --container-image-sha256 "$IMAGE_SHA256" \
  "${baseline_args[@]}" --output "$RESULT" --device cuda
[[ -s $RESULT ]]
echo inner_selector_complete
