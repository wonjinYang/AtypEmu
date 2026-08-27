#!/bin/bash
set -euo pipefail
umask 077
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CACHE=/home/yang07/.cache/atypemu_gpuopt
IMAGE=$CACHE/image/atypemu.sif
BASE=$CACHE/concurrency
workers=4
[[ -f $IMAGE ]]
[[ -f $CACHE/data/fold1/train_batch_manifest.tsv ]]
if [[ -f $BASE/baseline/worker0.json ]]; then mode=candidate; else mode=baseline; fi
source_sha=$(cat \
  "$ROOT/gpuopt/v3339_train_atom27_phase_d.py" \
  "$ROOT/gpuopt/v3339_atom27_explicit_h.py" | sha256sum | cut -d' ' -f1)
run=$BASE/runs/$(date +%Y%m%d_%H%M%S)_${mode}_${source_sha:0:12}_${workers}w_$$
for worker in $(seq 0 $((workers - 1))); do
  mkdir -p "$run/result/worker$worker"
done
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

pids=()
for worker in $(seq 0 $((workers - 1))); do
  out=$run/result/worker$worker
  "${container[@]}" /opt/conda/bin/python -I "$ROOT/gpuopt/benchmark_phase_d.py" \
    --trainer "$run/code/analysis/v3339_train_atom27_phase_d.py" \
    --train-manifest "$CACHE/data/fold1/train_batch_manifest.tsv" \
    --q-checkpoint "$CACHE/checkpoints/q_legacy5_fold1/checkpoint.pt" \
    --q-transfer-receipt "$run/code/q_initialization_transfer_receipt_d4v2.json" \
    --fixed-support-source "$run/code/deps/fixed_support_measure.py" \
    --stage-a-source "$run/code/analysis/v3339_train_all_bmrb_stage_a.py" \
    --stage-a-checkpoint "$CACHE/checkpoints/stage_a_fold1/checkpoint.pt" \
    --resume-state "$CACHE/checkpoints/resume/legacy5_fold1.pt" \
    --result "$out/core.json" --timing-ready "$out/timing.ready" \
    --timing-done "$out/timing.done" --start-signal "$out/start.signal" &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

wait_for_file() {
  local path=$1 pid=$2
  for _ in $(seq 1 3600); do
    [[ -e $path ]] && return 0
    kill -0 "$pid" 2>/dev/null || { wait "$pid"; return $?; }
    sleep 0.1
  done
  echo "timed out waiting for $path" >&2
  return 1
}
for worker in $(seq 0 $((workers - 1))); do
  wait_for_file "$run/result/worker$worker/timing.ready" "${pids[$worker]}"
done
nvidia-smi -i 0 \
  --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 200 > "$run/result/gpu_samples.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; kill "${pids[@]}" 2>/dev/null || true' EXIT
start_ns=$(date +%s%N)
if [[ $mode == baseline ]]; then
  for worker in $(seq 0 $((workers - 1))); do
    touch "$run/result/worker$worker/start.signal"
    wait_for_file "$run/result/worker$worker/timing.done" "${pids[$worker]}"
  done
else
  for worker in $(seq 0 $((workers - 1))); do
    touch "$run/result/worker$worker/start.signal"
  done
  for worker in $(seq 0 $((workers - 1))); do
    wait_for_file "$run/result/worker$worker/timing.done" "${pids[$worker]}"
  done
fi
end_ns=$(date +%s%N)
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
for pid in "${pids[@]}"; do wait "$pid"; done
trap - EXIT
wall_seconds=$(python3 -c "print(($end_ns-$start_ns)/1e9)")

if [[ $mode == baseline ]]; then
  rm -rf "$BASE/baseline.new"
  mkdir -p "$BASE/baseline.new"
  for worker in $(seq 0 $((workers - 1))); do
    cp "$run/result/worker$worker/core.json" "$BASE/baseline.new/worker$worker.json"
  done
  chmod -R a-w "$BASE/baseline.new"
  rm -rf "$BASE/baseline"
  mv "$BASE/baseline.new" "$BASE/baseline"
else
  python3 - "$BASE/baseline" "$run/result" <<'PY'
import json,sys
from pathlib import Path
baseline,result=map(Path,sys.argv[1:])
keys=('start_epoch','epochs','entity_updates','cohort_size','cohort_sha256',
      'model_state_sha256','optimizer_state_sha256','history_sha256')
workers=sorted(result.glob('worker*/core.json'))
for worker_path in workers:
    worker=int(worker_path.parent.name.removeprefix('worker'))
    expected_path=baseline/f'worker{worker}.json'
    if not expected_path.exists():
        expected_path=baseline/'worker0.json'
    expected=json.loads(expected_path.read_text())
    actual=json.loads(worker_path.read_text())
    for key in keys:
        if expected[key] != actual[key]:
            raise SystemExit(f'worker{worker} exactness mismatch: {key}')
print(f'all_{len(workers)}_workers_exact_model_optimizer_history_match=1')
PY
fi
core_args=()
for worker in $(seq 0 $((workers - 1))); do
  core_args+=("$run/result/worker$worker/core.json")
done
python3 "$ROOT/gpuopt/summarize_concurrency.py" \
  --mode "$mode" --wall-seconds "$wall_seconds" \
  --samples "$run/result/gpu_samples.csv" --summary "$run/result/summary.json" \
  "${core_args[@]}" \
  > "$run/result/metrics.txt"
echo "benchmark_mode=$mode workers=$workers source_sha256=$source_sha local_gpu=RTX_3060_Ti"
cat "$run/result/metrics.txt"
