#!/bin/bash
set -euo pipefail
umask 077
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CACHE=/home/yang07/.cache/atypemu_gpuopt
IMAGE=$CACHE/image/atypemu.sif
BASE=$CACHE/benchmark
[[ -f $IMAGE ]]
[[ -f $CACHE/data/fold1/train_batch_manifest.tsv ]]
if [[ -f $BASE/baseline/core.json ]]; then mode=candidate; else mode=baseline; fi
source_sha=$(cat \
  "$ROOT/gpuopt/v3339_train_atom27_phase_d.py" \
  "$ROOT/gpuopt/v3339_atom27_explicit_h.py" | sha256sum | cut -d' ' -f1)
run=$BASE/runs/$(date +%Y%m%d_%H%M%S)_${mode}_${source_sha:0:12}
mkdir -p "$run/result"
cp -a "$CACHE/code" "$run/code"
chmod -R u+w "$run/code"
cp "$ROOT/gpuopt/v3339_train_atom27_phase_d.py" \
  "$run/code/analysis/v3339_train_atom27_phase_d.py"
cp "$ROOT/gpuopt/v3339_atom27_explicit_h.py" \
  "$run/code/analysis/v3339_atom27_explicit_h.py"
chmod -R a-w "$run/code"

cuda_libs=/usr/lib/wsl/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/torch/lib
container=(singularity exec --cleanenv --containall --no-home
  --env CUDA_VISIBLE_DEVICES=0 --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env OMP_NUM_THREADS=8 --env PYTHONHASHSEED=3339 --env LD_LIBRARY_PATH="$cuda_libs"
  --bind /dev/dxg:/dev/dxg --bind /usr/lib/wsl:/usr/lib/wsl:ro
  --bind "$CACHE:$CACHE:rw" --bind "$ROOT/gpuopt:$ROOT/gpuopt:ro" "$IMAGE")
rm -f "$run/result/timing.ready" "$run/result/timing.done"
"${container[@]}" /opt/conda/bin/python -I "$ROOT/gpuopt/benchmark_phase_d.py" \
  --trainer "$run/code/analysis/v3339_train_atom27_phase_d.py" \
  --train-manifest "$CACHE/data/fold1/train_batch_manifest.tsv" \
  --q-checkpoint "$CACHE/checkpoints/q_legacy5_fold1/checkpoint.pt" \
  --q-transfer-receipt "$run/code/q_initialization_transfer_receipt_d4v2.json" \
  --fixed-support-source "$run/code/deps/fixed_support_measure.py" \
  --stage-a-source "$run/code/analysis/v3339_train_all_bmrb_stage_a.py" \
  --stage-a-checkpoint "$CACHE/checkpoints/stage_a_fold1/checkpoint.pt" \
  --resume-state "$CACHE/checkpoints/resume/legacy5_fold1.pt" \
  --result "$run/result/core.json" \
  --timing-ready "$run/result/timing.ready" \
  --timing-done "$run/result/timing.done" &
trainer_pid=$!
for _ in $(seq 1 600); do
  [[ -e $run/result/timing.ready ]] && break
  kill -0 "$trainer_pid" 2>/dev/null || { wait "$trainer_pid"; exit $?; }
  sleep 0.1
done
[[ -e $run/result/timing.ready ]]
nvidia-smi -i 0 \
  --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 200 > "$run/result/gpu_samples.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; kill "$trainer_pid" 2>/dev/null || true' EXIT
wait "$trainer_pid"
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT

"${container[@]}" /opt/conda/bin/python -I "$ROOT/gpuopt/summarize_benchmark.py" \
  "$run/result/core.json" "$run/result/gpu_samples.csv" "$run/result/summary.json" \
  > "$run/result/metrics.txt"
if [[ $mode == baseline ]]; then
  rm -rf "$BASE/baseline.new"
  mkdir -p "$BASE/baseline.new"
  cp "$run/result/core.json" "$BASE/baseline.new/core.json"
  cp "$run/result/summary.json" "$BASE/baseline.new/summary.json"
  chmod -R a-w "$BASE/baseline.new"
  rm -rf "$BASE/baseline"
  mv "$BASE/baseline.new" "$BASE/baseline"
else
  python3 - "$BASE/baseline/core.json" "$run/result/core.json" <<'PY'
import json,sys
baseline,candidate=(json.load(open(path)) for path in sys.argv[1:])
for key in ('start_epoch','epochs','entity_updates','cohort_size','cohort_sha256',
            'model_state_sha256','optimizer_state_sha256','history_sha256'):
    if baseline[key] != candidate[key]:
        raise SystemExit(f'exactness mismatch: {key}')
print('exact_model_optimizer_history_match=1')
PY
fi
echo "benchmark_mode=$mode source_sha256=$source_sha local_gpu=RTX_3060_Ti"
cat "$run/result/metrics.txt"
