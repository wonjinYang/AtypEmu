#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/yang07/anaconda3/envs/atypemu/bin/python

$PY gpuopt/candidates/nested_support_count_plan.py \
  --self-test --acknowledge-hold-only
if $PY gpuopt/candidates/nested_support_count_plan.py \
  --root . --self-test --acknowledge-hold-only >/dev/null 2>&1; then
  echo "legacy arbitrary plan-root interface was accepted" >&2
  exit 1
fi
$PY gpuopt/candidates/nested_support_catalog.py self-test
$PY gpuopt/candidates/check_nested_support_catalog.py \
  --check-bound-evidence --self-test

if $PY gpuopt/candidates/check_nested_support_catalog.py \
  --summary .auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json \
  --self-test >/dev/null 2>&1; then
  echo "legacy arbitrary evidence-path interface was accepted" >&2
  exit 1
fi

TMP=$(mktemp -d "$ROOT/.auto/catalog-binding-negative.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p \
  "$TMP/gpuopt/candidates" \
  "$TMP/.auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3"
ln "$ROOT/gpuopt/candidates/check_nested_support_catalog.py" \
  "$TMP/gpuopt/candidates/check_nested_support_catalog.py"
ln "$ROOT/gpuopt/candidates/nested_support_catalog.py" \
  "$TMP/gpuopt/candidates/nested_support_catalog.py"
ln "$ROOT/.auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json" \
  "$TMP/.auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_summary.json"
ln "$ROOT/.auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz" \
  "$TMP/.auto/staging/atypemu_nested_support_count_v1_catalog_yulab_v3/catalog_v3_shards.tar.gz"
ln "$ROOT/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json" \
  "$TMP/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
printf '{}\n' > "$TMP/.auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
if $PY "$TMP/gpuopt/candidates/check_nested_support_catalog.py" \
  --check-bound-evidence >/dev/null 2>&1; then
  echo "empty roster was accepted" >&2
  exit 1
fi
rm "$TMP/.auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
ln "$ROOT/.auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json" \
  "$TMP/.auto/staging/atypemu_nested_support_count_v1_entity_roster_v3.json"
rm "$TMP/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
cp "$ROOT/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json" \
  "$TMP/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
printf '\n' >> "$TMP/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
if $PY "$TMP/gpuopt/candidates/check_nested_support_catalog.py" \
  --check-bound-evidence >/dev/null 2>&1; then
  echo "appended source commitment was accepted" >&2
  exit 1
fi
printf '{}\n' > "$TMP/.auto/staging/k32_dynamic_distance_cache_source_commitment_v1.json"
if $PY "$TMP/gpuopt/candidates/check_nested_support_catalog.py" \
  --check-bound-evidence >/dev/null 2>&1; then
  echo "empty source commitment was accepted" >&2
  exit 1
fi
rm -rf "$TMP"
trap - EXIT
$PY - <<'PY'
import hashlib
import json
from pathlib import Path

receipt = json.loads(Path(
    "gpuopt/preunblind/atypemu_nested_support_count_v1_catalog_feasibility_receipt.json"
).read_text())
assert receipt["status"] == "HEAVY_TOPOLOGY_COUNT_FEASIBILITY_ONLY_ALL_ATOM_UNCLAIMED_HOLD"
assert receipt["catalog"]["entity_count"] == 135
assert receipt["catalog"]["heavy_topology_compatible_unique_coordinate_count"] == 134850
assert receipt["heavy_topology_level_count_feasibility"]["768"]["all_entities_count_feasible"] is True
assert receipt["heavy_topology_level_count_feasibility"]["1536"] == {
    "all_entities_count_feasible": False,
    "entity_count_feasible": 0,
    "maximum_entity_shortfall": 545,
    "total_shortfall": 72510,
}
for binding in receipt["evidence"].values():
    assert binding["raw_sha256"] == hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
assert receipt["claims"]["heavy_topology_count_feasible_levels"] == [32, 128, 768]
assert receipt["claims"]["all_atom_count_feasible_levels"] == []
assert receipt["claims"]["all_atom_recount_required"] is True
assert receipt["catalog"]["all_atom_topology_count_diagnostics"]["reference_topology_entity_count_ge_k"] == {"32": 130, "128": 121, "768": 108, "1536": 0}
assert all(receipt[name] is False for name in (
    "authorization_consumed", "outer_or_formal_metrics_opened",
    "science_executed", "source_scores_read", "target_values_read",
))
print("METRIC support_catalog_receipt_checks=21")
PY

/home/yang07/anaconda3/bin/ruff check \
  gpuopt/candidates/nested_support_count_plan.py \
  gpuopt/candidates/nested_support_catalog.py \
  gpuopt/candidates/check_nested_support_catalog.py \
  gpuopt/candidates/nested_support_all_atom_recount.py \
  gpuopt/candidates/check_nested_support_all_atom_recount.py \
  gpuopt/candidates/check_nested_support_all_atom_recount_raw.py \
  gpuopt/candidates/check_nested_support_all_atom_raw_evidence.py
