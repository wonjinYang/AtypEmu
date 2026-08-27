#!/bin/bash
set -euo pipefail
python3 -m py_compile gpuopt/benchmark_phase_d.py gpuopt/summarize_benchmark.py \
  gpuopt/summarize_concurrency.py gpuopt/summarize_inner_selector.py \
  gpuopt/v3339_train_atom27_phase_d.py gpuopt/v3339_atom27_explicit_h.py \
  gpuopt/v3339_inner_chart_selector.py \
  .auto/staging/d28_build_inner_selector_inputs.py \
  .auto/staging/d29_stage_inner_selector_source.py
bash -n gpuopt/run_local_benchmark.sh gpuopt/run_local_concurrency_benchmark.sh \
  autoresearch.sh .auto/staging/d29_run_inner_selector.sbatch
python3 -m json.tool .auto/preunblind/clean_menu_builder_output.json >/dev/null
test "$(sha256sum .auto/preunblind/clean_menu_builder_output.json | cut -d' ' -f1)" = \
  3d120deccf219f517938467bd0e18fb4f2fffe1d9444b09250c6f908e4cac42f
python3 -m json.tool .auto/development/d30_ser_hn_acceptor_menu.json >/dev/null
test "$(sha256sum .auto/development/d30_ser_hn_acceptor_menu.json | cut -d' ' -f1)" = \
  0085c13f82f1c8e0b240bc3aa81c8759791552f6a317f5e3efec9babec9a00fd
python3 -m json.tool .auto/development/d30_ser_hn_acceptor_commitment.json >/dev/null
atom27_sha=$(sha256sum gpuopt/v3339_atom27_explicit_h.py | cut -d' ' -f1)
grep -q "D2_SOURCE_SHA256 = \"$atom27_sha\"" gpuopt/v3339_train_atom27_phase_d.py
git diff --check
