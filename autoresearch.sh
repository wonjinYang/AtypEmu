#!/bin/bash
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "$0")" && pwd)
REMOTE=iremb6-server
REMOTE_ROOT=/scratch/wkyu514/yang07/AtypEmu
PARENT=$REMOTE_ROOT/.scratch/v3339_phase_d13_gpu_optimized_source_frozen
INPUT_ROOT=$REMOTE_ROOT/.scratch/v3339_phase_d_inner_selector_inputs_v2
INPUT_SHA256=5094dc56e0a8b9d8cdc5a9e3c72bd83c84d5510576d3773e8c1968def750c070
PARENT_CHECKSUMS_SHA256=16ec1e063198013e670b87c0efc48c614e0a1f3d4bcb0dd038c14583a03048c5
CONTAINER_IMAGE_SHA256=67418b92c18eb82988f41b4dd902b36a4ee5e6d6dae4c1a263df382eb73f7531
SOURCE_BASE=$REMOTE_ROOT/.scratch/v3339_phase_d_inner_selector_sources
RESULT_BASE=$REMOTE_ROOT/results/v3339_ar_inner_selector

atom27_sha=$(sha256sum "$ROOT/gpuopt/v3339_atom27_explicit_h.py" | cut -d' ' -f1)
trainer_sha=$(sha256sum "$ROOT/gpuopt/v3339_train_atom27_phase_d.py" | cut -d' ' -f1)
selector_sha=$(sha256sum "$ROOT/gpuopt/v3339_inner_chart_selector.py" | cut -d' ' -f1)
runner_sha=$(sha256sum "$ROOT/.auto/staging/d29_run_inner_selector.sbatch" | cut -d' ' -f1)
menu_sha=$(sha256sum "$ROOT/.auto/preunblind/clean_menu_builder_output.json" | cut -d' ' -f1)
stage_script_sha=$(sha256sum "$ROOT/.auto/staging/d29_stage_inner_selector_source.py" | cut -d' ' -f1)
[[ $menu_sha == 3d120deccf219f517938467bd0e18fb4f2fffe1d9444b09250c6f908e4cac42f ]]
grep -q "D2_SOURCE_SHA256 = \"$atom27_sha\"" "$ROOT/gpuopt/v3339_train_atom27_phase_d.py"

hashes_json=$(python3 - "$atom27_sha" "$trainer_sha" "$selector_sha" "$runner_sha" "$menu_sha" <<'PY'
import json,sys
print(json.dumps(dict(zip(('atom27','trainer','selector','runner','menu'),sys.argv[1:])),sort_keys=True,separators=(',',':')))
PY
)
bindings_json=$(python3 - "$CONTAINER_IMAGE_SHA256" "$INPUT_SHA256" \
  "$PARENT_CHECKSUMS_SHA256" "$stage_script_sha" <<'PY'
import json,sys
print(json.dumps(dict(zip(('container_image_sha256','input_manifest_sha256','parent_checksums_sha256','stage_script_sha256'),sys.argv[1:])),sort_keys=True,separators=(',',':')))
PY
)
bundle=$(python3 - "$hashes_json" "$bindings_json" <<'PY'
import hashlib,json,sys
payload={'source_hashes':json.loads(sys.argv[1]),'bindings':json.loads(sys.argv[2])}
print(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
PY
)
upload=$REMOTE_ROOT/.scratch/v3339_ar_inner_selector_uploads/$bundle
source=$SOURCE_BASE/$bundle/frozen
output=$RESULT_BASE/$bundle

ssh "$REMOTE" "mkdir -p '$upload'; chmod 700 '$upload'"
scp -q "$ROOT/gpuopt/v3339_atom27_explicit_h.py" \
  "$REMOTE:$upload/v3339_atom27_explicit_h.py"
scp -q "$ROOT/gpuopt/v3339_train_atom27_phase_d.py" \
  "$REMOTE:$upload/v3339_train_atom27_phase_d.py"
scp -q "$ROOT/gpuopt/v3339_inner_chart_selector.py" \
  "$REMOTE:$upload/v3339_inner_chart_selector.py"
scp -q "$ROOT/.auto/staging/d29_run_inner_selector.sbatch" \
  "$REMOTE:$upload/run_inner_selector.sbatch"
scp -q "$ROOT/.auto/preunblind/clean_menu_builder_output.json" \
  "$REMOTE:$upload/clean_menu_builder_output.json"
scp -q "$ROOT/.auto/staging/d29_stage_inner_selector_source.py" \
  "$REMOTE:$upload/stage_source.py"
printf '%s\n' "$hashes_json" | ssh "$REMOTE" "cat > '$upload/hashes.json'"
printf '%s\n' "$bindings_json" | ssh "$REMOTE" "cat > '$upload/bindings.json'"
ssh "$REMOTE" "python3 '$upload/stage_source.py' \
  --parent '$PARENT' --upload '$upload' --destination-base '$SOURCE_BASE' \
  --hashes-json \"\$(cat '$upload/hashes.json')\" \
  --bindings-json \"\$(cat '$upload/bindings.json')\"" >/dev/null
if [[ ${ATYP_STAGE_ONLY:-0} == 1 ]]; then
  echo "source_bundle=$source"
  exit 0
fi

candidate_active=$(python3 - "$ROOT/gpuopt/v3339_atom27_explicit_h.py" <<'PY'
import ast,sys
tree=ast.parse(open(sys.argv[1],encoding='utf-8').read())
value=None
for node in tree.body:
    if isinstance(node,(ast.Assign,ast.AnnAssign)):
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        if any(isinstance(target,ast.Name) and target.id=='PHYSICS_CHART_CANDIDATE' for target in targets):
            value=ast.literal_eval(node.value)
print(int(value is not None))
PY
)
baseline_receipt=
baseline_receipt_sha=
if [[ $candidate_active == 1 ]]; then
  baseline_binding=$ROOT/.auto/baseline/inner_selector.json
  [[ -s $baseline_binding ]]
  readarray -t baseline_fields < <(python3 - "$baseline_binding" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p['remote_result_path'])
print(p['result_file_sha256'])
PY
)
  baseline_receipt=${baseline_fields[0]}
  baseline_receipt_sha=${baseline_fields[1]}
  [[ $(ssh "$REMOTE" "sha256sum '$baseline_receipt' | cut -d' ' -f1") == "$baseline_receipt_sha" ]]
fi

if ! ssh "$REMOTE" "test -s '$output/result.json'"; then
  ssh "$REMOTE" "mkdir -p '$output'; chmod 700 '$output'"
  if ssh "$REMOTE" "test -s '$output/submission.json'"; then
    job_id=$(ssh "$REMOTE" "python3 -c 'import json; print(json.load(open(\"$output/submission.json\"))[\"job_id\"])'")
  else
    job_id=$(ssh "$REMOTE" "sbatch --parsable \
      --job-name='atyp-d29-${bundle:0:8}' \
      --output='$output/slurm-%j.out' --error='$output/slurm-%j.out' \
      --export=ALL,SOURCE='$source',SOURCE_BUNDLE_SHA256='$bundle',INPUT_ROOT='$INPUT_ROOT',INPUT_SHA256='$INPUT_SHA256',TRAINER_SHA256='$trainer_sha',OUTPUT_ROOT='$output',IMAGE_SHA256='$CONTAINER_IMAGE_SHA256',BASELINE_RECEIPT='$baseline_receipt',BASELINE_RECEIPT_SHA256='$baseline_receipt_sha' \
      '$source/run_inner_selector.sbatch'")
    ssh "$REMOTE" "python3 - <<'PY'
import json
from pathlib import Path
p=Path('$output/submission.json')
p.write_text(json.dumps({'job_id':'$job_id','bundle_sha256':'$bundle'},sort_keys=True)+'\\n')
p.chmod(0o400)
PY"
  fi
  for _ in $(seq 1 360); do
    if ssh "$REMOTE" "test -s '$output/result.json'"; then break; fi
    state=$(ssh "$REMOTE" "sacct -j '$job_id' --format=JobIDRaw,State -n -P | awk -F'|' -v id='$job_id' '\$1==id {print \$2; exit}'" || true)
    case "$state" in
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
        ssh "$REMOTE" "cat '$output/slurm-$job_id.out' 2>/dev/null || true"
        exit 1
        ;;
    esac
    sleep 15
  done
fi
ssh "$REMOTE" "test -s '$output/result.json'"
mkdir -p "$ROOT/.auto/selector/results"
local_result=$ROOT/.auto/selector/results/$bundle.json
scp -q "$REMOTE:$output/result.json" "$local_result"
python3 "$ROOT/gpuopt/summarize_inner_selector.py" "$local_result" \
  --trainer-sha256 "$trainer_sha" \
  --input-manifest-sha256 "$INPUT_SHA256" \
  --source-bundle-sha256 "$bundle" \
  --container-image-sha256 "$CONTAINER_IMAGE_SHA256"
echo "experiment_source_bundle=$bundle local_result=$local_result"
