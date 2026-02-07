#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python scripts/run_scaled_sleep_stack.py \
  --out artifacts/local_runs/scaled_sleep_stack_n2000 \
  --seed 123 \
  --n 2000 \
  --steps-wake 2400 \
  --steps-sleep 1800

PYTHONPATH=src python scripts/benchmark_sleep_stack_scale.py
