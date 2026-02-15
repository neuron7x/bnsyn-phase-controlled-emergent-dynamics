#!/bin/bash
set -e

echo "🔍 Running pre-push checks..."
echo ""

echo "1️⃣  Code formatting..."
ruff format --check . || { echo "❌ Format check failed"; exit 1; }

echo "2️⃣  Linting..."
ruff check . || { echo "❌ Lint check failed"; exit 1; }

echo "3️⃣  Type checking (strict)..."
mypy src --strict || { echo "❌ Type check failed"; exit 1; }

echo "4️⃣  Smoke tests..."
pytest -m "not validation" -q --tb=short || { echo "❌ Tests failed"; exit 1; }

echo "5️⃣  Coverage (≥85%)..."
pytest --cov=src/bnsyn --cov-fail-under=85 -q || { echo "❌ Coverage below 85%"; exit 1; }

echo "6️⃣  SSOT gates..."
python -m scripts.validate_bibliography || { echo "❌ Bibliography validation failed"; exit 1; }
python -m scripts.validate_claims || { echo "❌ Claims validation failed"; exit 1; }
python -m scripts.scan_normative_tags || { echo "❌ Normative tag scan failed"; exit 1; }
python -m scripts.validate_pr_gates || { echo "❌ PR gates validation failed"; exit 1; }
python -m scripts.validate_required_status_contexts || { echo "❌ Required status contexts validation failed"; exit 1; }
python -m scripts.sync_required_status_contexts --check || { echo "❌ Required status contexts are not synchronized"; exit 1; }

echo "7️⃣  Security audit..."
gitleaks detect --redact --source=. || { echo "❌ Gitleaks failed"; exit 1; }
mkdir -p artifacts
pip-audit --desc --format json --output artifacts/pip-audit.json || { echo "❌ Pip audit failed"; exit 1; }
bandit -r src/ -ll || { echo "❌ Bandit security check failed"; exit 1; }

echo ""
echo "✅ All pre-push checks passed! Ready for PR."
exit 0
