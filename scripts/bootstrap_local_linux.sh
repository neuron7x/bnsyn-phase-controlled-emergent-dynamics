#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Unsupported OS $(uname -s). This bootstrap script supports Linux." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Install Python 3.11+ first." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python 3.11+ required. Install Python 3.11+ then re-run bootstrap.")
print(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Reusing existing virtualenv at ${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e ".[plot]"
"${VENV_DIR}/bin/python" -m scripts.local_doctor

echo "Bootstrap complete."
echo "Run canonical proof with: ./scripts/run_canonical_local.sh"
