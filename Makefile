.PHONY: dev-setup dev-env-offline wheelhouse-build wheelhouse-validate wheelhouse-report wheelhouse-clean check test test-determinism test-validation coverage coverage-baseline coverage-gate quality format fix lint mypy ssot security clean docs validate-claims-coverage docs-evidence mutation mutation-ci mutation-baseline mutation-check mutation-check-strict release-readiness

LOCK_FILE ?= requirements-lock.txt
WHEELHOUSE_DIR ?= wheelhouse
PYTHON_VERSION ?= 3.11
WHEELHOUSE_REPORT ?= artifacts/wheelhouse_report.json

dev-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,test]"
	pre-commit install
	pre-commit autoupdate


wheelhouse-build:
	python -m scripts.build_wheelhouse build --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION)

wheelhouse-validate:
	python -m scripts.build_wheelhouse validate --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION) --report $(WHEELHOUSE_REPORT)

dev-env-offline: wheelhouse-validate
	pip install --no-index --find-links $(WHEELHOUSE_DIR) -r $(LOCK_FILE)
	pip install --no-index --find-links $(WHEELHOUSE_DIR) --no-deps -e .
	pre-commit install

wheelhouse-clean:
	rm -rf $(WHEELHOUSE_DIR) $(WHEELHOUSE_REPORT)

wheelhouse-report: wheelhouse-validate
	@echo "Wheelhouse report: $(WHEELHOUSE_REPORT)"

test:
	python -m pytest -m "not validation" -q

test-determinism:
	python -m pytest tests/test_determinism.py tests/properties/test_properties_determinism.py -q

test-validation:
	python -m pytest -m validation -q

coverage:
	python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml -q

coverage-baseline: coverage
	python -m scripts.generate_coverage_baseline --coverage-xml coverage.xml --output quality/coverage_gate.json --minimum-percent 99.0

coverage-gate: coverage
	python -m scripts.check_coverage_gate --coverage-xml coverage.xml --baseline quality/coverage_gate.json


mutation:
	@echo "🧬 Running mutation profile (reproducible local workflow step)..."
	@python -m pip install -e ".[test]" -q
	@python -m pip install mutmut==2.4.5 -q
	@python -m scripts.run_mutation_pipeline

mutation-ci:
	@echo "🧬 Emitting mutation CI artifacts to local files..."
	@baseline_file=quality/mutation_baseline.json; \
	output_file=.mutation_ci_output; \
	summary_file=.mutation_ci_summary.md; \
	: > $$output_file; \
	: > $$summary_file; \
	GITHUB_OUTPUT=$$output_file GITHUB_STEP_SUMMARY=$$summary_file python -m scripts.mutation_ci_summary --baseline $$baseline_file --write-output --write-summary

mutation-baseline:
	@echo "🧬 Running mutation testing to establish baseline..."
	@python -m pip install -e ".[test]" -q
	@python -m pip install mutmut==2.4.5 -q
	@python -m scripts.generate_mutation_baseline

mutation-check:
	@echo "🧬 Running mutation testing against baseline..."
	@python -m pip install -e ".[test]" -q
	@python -m pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@python -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: ±{baseline['tolerance_delta']}%)\")"
	@python -m scripts.validate_mutation_baseline
	@python -m scripts.run_mutation_pipeline
	@python -m scripts.check_mutation_score --advisory

mutation-check-strict:
	@echo "🧬 Running mutation testing against baseline (STRICT MODE)..."
	@python -m pip install -e ".[test]" -q
	@python -m pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@python -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: ±{baseline['tolerance_delta']}%)\")"
	@python -m scripts.validate_mutation_baseline
	@python -m scripts.run_mutation_pipeline
	@python -m scripts.check_mutation_score --strict

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
	python -m scripts.validate_bibliography
	python -m scripts.validate_claims
	python -m scripts.scan_normative_tags

validate-claims-coverage:
	python -m scripts.validate_claims_coverage --format markdown

docs-evidence:
	python -m scripts.generate_evidence_coverage

security:
	gitleaks detect --redact --verbose --source=.
	pip-audit --desc
	bandit -r src/ -ll

check: format lint mypy coverage ssot security
	@echo "✅ All checks passed"

docs:
	sphinx-build docs docs/_build
	@echo "Docs built at docs/_build"

release-readiness:
	python -m scripts.release_readiness

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name .coverage -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f .mutmut-cache
	rm -rf $(WHEELHOUSE_DIR) $(WHEELHOUSE_REPORT)
	@echo "Cleaned temporary files"
