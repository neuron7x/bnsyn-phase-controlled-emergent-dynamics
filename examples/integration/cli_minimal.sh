#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="${1:-results/integration_cli_minimal.json}"
mkdir -p "$(dirname "$OUT_FILE")"
python -m bnsyn.cli demo --steps 120 --dt-ms 0.1 --seed 123 --N 32 > "$OUT_FILE"
echo "wrote: $OUT_FILE"
