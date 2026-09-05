#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python
OUTPUT="$($PY -m unittest gpuopt.tests.test_k32_dynamic_distance_cache -v 2>&1)"
printf '%s\n' "$OUTPUT"
COUNT="$(printf '%s\n' "$OUTPUT" | sed -n 's/^Ran \([0-9][0-9]*\) tests.*/\1/p')"
test "$COUNT" = 9
echo "METRIC passed_smokes=$COUNT"
