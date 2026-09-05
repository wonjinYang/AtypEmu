#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

SOURCE=.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json
OUTPUT=data/k32_dynamic_distance_cache_v1
RECEIPT=.auto/runs/k32_dynamic_distance_cache_receipt_v1.json
CHECK=.auto/runs/k32_dynamic_distance_cache_independent_check_v1.json

"$PY" gpuopt/materialize_k32_dynamic_distance_cache.py freeze \
  --root "$ROOT" \
  --output "$SOURCE"
"$PY" gpuopt/materialize_k32_dynamic_distance_cache.py materialize \
  --root "$ROOT" \
  --source-commitment "$SOURCE" \
  --output-root "$OUTPUT" \
  --receipt "$RECEIPT"
"$PY" gpuopt/check_k32_dynamic_distance_cache.py \
  --root "$ROOT" \
  --source-commitment "$SOURCE" \
  --receipt "$RECEIPT" \
  --output-root "$OUTPUT" \
  --summary "$CHECK"
