#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
VENV_BNSYN="${ROOT_DIR}/.venv/bin/bnsyn"
OUT_DIR="${ROOT_DIR}/artifacts/canonical_run"
SMOKE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUT_DIR="$2"
      shift 2
      ;;
    --smoke)
      SMOKE_ONLY=1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Usage: ./scripts/run_canonical_local.sh [--output <dir>] [--smoke]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: .venv missing. Run ./scripts/bootstrap_local_linux.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${VENV_PYTHON}" -m scripts.local_doctor

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
"${VENV_BNSYN}" run --profile canonical --plot --export-proof --output "${OUT_DIR}"

if [[ "${SMOKE_ONLY}" -eq 0 ]]; then
  "${VENV_BNSYN}" proof-validate-bundle "${OUT_DIR}"
  "${VENV_BNSYN}" proof-evaluate "${OUT_DIR}"
  "${VENV_BNSYN}" proof-check-determinism "${OUT_DIR}"
  "${VENV_BNSYN}" proof-check-envelope "${OUT_DIR}"
fi

OUT_DIR_PATH="${OUT_DIR}" "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_PATH"])
proof_path = out_dir / "proof_report.json"
if not proof_path.exists():
    raise SystemExit(f"ERROR: missing {proof_path}")
proof = json.loads(proof_path.read_text(encoding="utf-8"))
verdict = str(proof.get("verdict", "UNKNOWN"))
if verdict.upper() != "PASS":
    raise SystemExit(f"ERROR: proof_report verdict is {verdict}, expected PASS")
print(f"Canonical proof verdict: {verdict}")
print(f"Canonical artifact bundle: {out_dir}")
PY
