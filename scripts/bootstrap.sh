#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=".venv"
READY_FILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRAS="dev,test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --ready-file)
      READY_FILE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install -U pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -e ".[${EXTRAS}]"
"${VENV_PATH}/bin/python" -m pip check

if [[ -n "${READY_FILE}" ]]; then
  mkdir -p "$(dirname "${READY_FILE}")"
  : > "${READY_FILE}"
fi

echo "bootstrap complete: venv=${VENV_PATH} extras=${EXTRAS}"
