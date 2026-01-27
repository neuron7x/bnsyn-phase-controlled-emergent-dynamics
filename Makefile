.PHONY: dev-setup check test test-determinism test-validation coverage quality format fix lint mypy ssot security clean docs validate-claims-coverage docs-evidence

dev-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	pre-commit install
	pre-commit autoupdate

test:
	pytest -m "not validation" -v --tb=short

test-determinism:
	pytest tests/test_determinism.py tests/test_properties_determinism.py -v

test-validation:
	pytest -m validation -v

coverage:
	pytest -m "not validation" --cov=src/bnsyn --cov-report=html --cov-report=term-missing --cov-fail-under=85
	@echo "Coverage report: htmlcov/index.html"

quality: format lint mypy ssot security
	@echo "✅ All quality checks passed"

format:
	ruff format .
	@echo "Formatted code"

fix:
	ruff check . --fix
	@echo "Fixed lint issues"

lint:
	ruff check .
	pylint src/bnsyn

mypy:
	mypy src --strict --config-file pyproject.toml

ssot:
	python scripts/validate_bibliography.py
	python scripts/validate_claims.py
	python scripts/scan_normative_tags.py

validate-claims-coverage:
	python scripts/validate_claims_coverage.py --format markdown

docs-evidence:
	python scripts/generate_evidence_coverage.py

security:
	gitleaks detect --redact --verbose --source=.
	pip-audit --desc
	bandit -r src/ -ll

check: format lint mypy coverage ssot security
	@echo "✅ All checks passed"

docs:
	sphinx-build docs docs/_build
	@echo "Docs built at docs/_build"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name .coverage -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaned temporary files"
