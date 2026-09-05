#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/yang07/anaconda3/envs/atypemu/bin/python \
  "$ROOT/gpuopt/check_corrected_k8_job135711_backpressure_recovery.py" \
  --run-dir "$ROOT/.auto/runs/current" \
  --authorization-git-dir "$ROOT/.git" \
  --emit-metrics
