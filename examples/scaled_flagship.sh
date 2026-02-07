#!/usr/bin/env bash
set -euo pipefail

python -m bnsyn.tools.run_scaled_sleep_stack \
  --out artifacts/local_runs/scaled_sleep_stack_n2000 \
  --seed 123 \
  --n 2000 \
  --steps-wake 2400 \
  --steps-sleep 1800

python -m bnsyn.tools.benchmark_sleep_stack_scale
