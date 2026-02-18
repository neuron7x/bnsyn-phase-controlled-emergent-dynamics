.PHONY: install setup demo dev-setup quickstart-smoke dev-env-offline wheelhouse-build wheelhouse-validate wheelhouse-report wheelhouse-clean check test test-gate test-determinism test-validation test-property coverage coverage-fast coverage-baseline coverage-gate quality format fix lint mypy ssot security sbom cleanroom clean docs build release validate-claims-coverage docs-evidence mutation mutation-ci mutation-baseline mutation-check mutation-check-strict release-readiness manifest manifest-validate manifest-check inventory inventory-check

LOCK_FILE ?= requirements-lock.txt
WHEELHOUSE_DIR ?= wheelhouse
PYTHON_VERSION ?= 3.11
WHEELHOUSE_REPORT ?= artifacts/wheelhouse_report.json
SETUP_CMD ?= python -m pip install -e ".[test]"
TEST_CMD ?= python -m pytest -m "not (validation or property)" -q

setup:
	python -V
	python -m pip --version
	$(SETUP_CMD)
	python -m pip check

install: setup
	@echo "✅ install completed via setup"

demo:
	@python -m scripts.run_quickstart_demo

dev-setup:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev,test]"
	pre-commit install
	pre-commit autoupdate

quickstart-smoke:
	python -m scripts.check_quickstart_consistency
	python -m pip install -e .
	python -m pip show bnsyn
	bnsyn --help
	bnsyn sleep-stack --help
	bnsyn demo --steps 120 --dt-ms 0.1 --seed 123 --N 32 | python -c "import json,sys; data=json.load(sys.stdin); assert 'demo' in data, 'missing demo key'; print('quickstart demo output validated')"


wheelhouse-build:
	python -m scripts.build_wheelhouse build --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION)

wheelhouse-validate:
	python -m scripts.build_wheelhouse validate --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION) --report $(WHEELHOUSE_REPORT)

dev-env-offline: wheelhouse-validate
	python -m pip install --no-index --find-links $(WHEELHOUSE_DIR) -r $(LOCK_FILE)
	python -m pip install --no-index --find-links $(WHEELHOUSE_DIR) --no-deps -e .
	pre-commit install

wheelhouse-clean:
	rm -rf $(WHEELHOUSE_DIR) $(WHEELHOUSE_REPORT)

wheelhouse-report: wheelhouse-validate
	@echo "Wheelhouse report: $(WHEELHOUSE_REPORT)"

test:
	$(MAKE) test-gate

test-gate:
	$(TEST_CMD)

test-determinism:
	python -m pytest tests/test_determinism.py tests/test_properties_determinism.py -q

test-validation:
	python -m pytest -m "validation" -q

test-property:
	python -m pytest -m "property" -q

coverage:
	python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml -q

coverage-fast:
	python -m pytest -m "not (validation or property)" --cov=bnsyn --cov-report=term-missing --cov-report=xml:coverage.xml -q

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

api-contract:
	python -m scripts.check_api_contract --baseline quality/api_contract_baseline.json

validate-api-maturity:
	python -m scripts.validate_api_maturity

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
	pylint --fail-under=9.5 src/bnsyn

mypy:
	mypy src --strict --config-file pyproject.toml

ssot:
	python -m scripts.validate_bibliography
	python -m scripts.validate_claims
	python -m scripts.scan_normative_tags
	python -m scripts.validate_pr_gates
	python -m scripts.validate_required_status_contexts
	python -m scripts.sync_required_status_contexts --check
	$(MAKE) inventory-check
	python -m scripts.validate_api_maturity
	$(MAKE) api-contract
	$(MAKE) manifest-check
	$(MAKE) traceability-check

validate-claims-coverage:
	python -m scripts.validate_claims_coverage --format markdown

docs-evidence:
	python -m scripts.generate_evidence_coverage

SECURITY_REPORT ?= artifacts/pip-audit.json

security:
	python -m pip install --upgrade pip==26.0.1
	python -m pip install -e ".[dev]"
	mkdir -p artifacts
	python -m scripts.ensure_gitleaks -- detect --redact --verbose --source=.
	python -m pip_audit --desc --format json --output $(SECURITY_REPORT)
	python -m bandit -r src/ -ll

SBOM_REPORT ?= artifacts/prod_ready/reports/sbom.cdx.json

sbom:
	python -m pip install cyclonedx-bom==7.1.0
	mkdir -p $(dir $(SBOM_REPORT))
	cyclonedx-py environment --output-format JSON --output-file $(SBOM_REPORT)

cleanroom:
	$(MAKE) clean
	$(MAKE) install
	$(MAKE) build
	$(MAKE) test
	bnsyn --help

check: format lint mypy coverage ssot security
	@echo "✅ All checks passed"

docs:
	python -m pip install -e ".[docs]"
	python -m sphinx -b html docs docs/_build/html
	@echo "Docs built at docs/_build/html"

build:
	python -m pip install -e . build
	python -m build

release: build release-readiness
	@echo "✅ release artifacts and readiness report generated"

release-readiness:
	python -m scripts.release_readiness

manifest:
	python -m tools.manifest generate

manifest-validate:
	python -m tools.manifest validate

manifest-check: manifest manifest-validate
	git diff --exit-code -- .github/REPO_MANIFEST.md manifest/repo_manifest.computed.json

inventory:
	python tools/generate_inventory.py

inventory-check:
	python tools/generate_inventory.py --check

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


traceability-check:
	python -m scripts.validate_traceability

public-surfaces:
	python -m scripts.discover_public_surfaces
