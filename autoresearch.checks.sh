#!/bin/bash
set -euo pipefail
python3 -m py_compile gpuopt/benchmark_phase_d.py gpuopt/summarize_benchmark.py \
  gpuopt/v3339_train_atom27_phase_d.py gpuopt/v3339_atom27_explicit_h.py
bash -n gpuopt/run_local_benchmark.sh autoresearch.sh
git diff --check
