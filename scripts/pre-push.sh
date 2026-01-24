#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running pre-push checks..."
echo ""

echo "1️⃣  Format check..."
ruff format --check .

echo "2️⃣  Linting..."
ruff check .

echo "3️⃣  Type checking..."
mypy src --strict

echo "4️⃣  Smoke tests..."
pytest -m "not validation" -q

echo "5️⃣  Coverage..."
pytest --cov=src/bnsyn --cov-fail-under=85 -q

echo "6️⃣  SSOT gates..."
python scripts/validate_bibliography.py
python scripts/validate_claims.py
python scripts/scan_normative_tags.py

echo "7️⃣  Security..."
pip-audit --desc

echo ""
echo "✅ All pre-push checks passed! Ready for PR."
