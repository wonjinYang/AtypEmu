#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

$PY gpuopt/check_k32_dynamic_distance_cache.py \
  --root "$ROOT" \
  --source-commitment "$ROOT/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json" \
  --receipt "$ROOT/.auto/runs/k32_dynamic_distance_cache_receipt_v1.json" \
  --output-root "$ROOT/data/k32_dynamic_distance_cache_v1" \
  --summary "$ROOT/.auto/runs/k32_dynamic_distance_cache_independent_check_v1r1.json"
