#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

$PY gpuopt/candidates/nested_support_count_plan.py \
  --root . --self-test --acknowledge-hold-only
$PY gpuopt/candidates/nested_support_catalog.py self-test
$PY gpuopt/candidates/check_nested_support_catalog.py \
  --summary .auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json \
  --shard-archive .auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz \
  --self-test
$PY - <<'PY'
import hashlib
import json
from pathlib import Path

receipt = json.loads(Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_catalog_feasibility_receipt.json"
).read_text())
assert receipt["status"] == "COUNT_FEASIBILITY_ONLY_HOLD"
assert receipt["catalog"]["entity_count"] == 135
assert receipt["catalog"]["heavy_topology_compatible_unique_coordinate_count"] == 134850
assert receipt["level_count_feasibility"]["768"]["all_entities_count_feasible"] is True
assert receipt["level_count_feasibility"]["1536"] == {
    "all_entities_count_feasible": False,
    "entity_count_feasible": 0,
    "maximum_entity_shortfall": 545,
    "total_shortfall": 72510,
}
assert receipt["catalog_summary"]["sha256"] == hashlib.sha256(Path(
    receipt["catalog_summary"]["external_relative_path"]
).read_bytes()).hexdigest()
assert all(receipt[name] is False for name in (
    "authorization_consumed", "outer_or_formal_metrics_opened",
    "science_executed", "source_scores_read", "target_values_read",
))
print("METRIC support_catalog_receipt_checks=11")
PY

/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/nested_support_count_plan.py \
  gpuopt/candidates/nested_support_catalog.py \
  gpuopt/candidates/check_nested_support_catalog.py
