.PHONY: install setup demo reproduce dev-setup quickstart-smoke dev-env-offline wheelhouse-build wheelhouse-validate wheelhouse-report wheelhouse-clean check test test-all test-gate test-diagnostics test-determinism test-validation test-integration test-e2e test-property entropy-gate coverage coverage-fast coverage-baseline coverage-gate quality format fix lint mypy typecheck ssot security sbom profile cleanroom clean docs build release validate-claims-coverage docs-evidence mutation mutation-ci mutation-baseline mutation-check mutation-check-strict release-readiness manifest manifest-validate manifest-check inventory inventory-check perfection-gate launch-gate smlrs-gate dsio-gate ci-artifacts flake-report

PYTHON := ./.venv/bin/python
ENV_READY := .venv/.ready-dev
BOOTSTRAP_SCRIPT := scripts/bootstrap.sh

LOCK_FILE ?= requirements-lock.txt
WHEELHOUSE_DIR ?= wheelhouse
PYTHON_VERSION ?= 3.11
WHEELHOUSE_REPORT ?= artifacts/wheelhouse_report.json
TEST_CMD ?= $(PYTHON) -m pytest -m "not (validation or property)" -q
JUNIT_DIR ?= artifacts/tests
JUNIT_FAST ?= $(JUNIT_DIR)/junit-fast.xml
JUNIT_ALL ?= $(JUNIT_DIR)/junit-all.xml
AFFECTED ?= 0
PKG ?=
SVC ?=

ensure-venv:
	@test -x $(PYTHON) || python3 -m venv .venv

$(ENV_READY): pyproject.toml $(BOOTSTRAP_SCRIPT)
	bash $(BOOTSTRAP_SCRIPT) --venv .venv --ready-file $(ENV_READY) --extras dev,test

setup: $(ENV_READY)
	$(PYTHON) -V
	$(PYTHON) -m pip --version
	$(PYTHON) -m pip check

install: setup
	@echo "install completed via setup"

demo:
	@$(PYTHON) -m scripts.run_quickstart_demo

reproduce:
	$(PYTHON) -m scripts.run_quickstart_demo
	$(PYTHON) -m scripts.write_reproduce_manifest
	$(PYTHON) -m scripts.verify_reproducible_artifacts --spec docs/proof/repro_spec.json --runs 2 --report artifacts/reproducibility_report.json


dev-setup:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[dev,test]"
	$(PYTHON) -m pre_commit install
	$(PYTHON) -m pre_commit autoupdate

quickstart-smoke:
	$(PYTHON) -m scripts.check_quickstart_consistency
	$(PYTHON) -m pip show bnsyn
	$(PYTHON) -m bnsyn --help
	$(PYTHON) -m bnsyn run --help
	$(PYTHON) -m bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='ok' and d['artifact_dir'].endswith('canonical_run') and d['artifacts']==['emergence_plot.png','summary_metrics.json','criticality_report.json','avalanche_report.json','phase_space_report.json','population_rate_trace.npy','sigma_trace.npy','coherence_trace.npy','phase_space_rate_sigma.png','phase_space_rate_coherence.png','phase_space_activity_map.png','avalanche_fit_report.json','robustness_report.json','envelope_report.json','run_manifest.json','proof_report.json'], f'smoke failed: {d}'; print('quickstart canonical run output validated')"


wheelhouse-build:
	$(PYTHON) -m scripts.build_wheelhouse build --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION)

wheelhouse-validate:
	$(PYTHON) -m scripts.build_wheelhouse validate --lock-file $(LOCK_FILE) --wheelhouse $(WHEELHOUSE_DIR) --python-version $(PYTHON_VERSION) --report $(WHEELHOUSE_REPORT)

dev-env-offline: wheelhouse-validate
	$(PYTHON) -m pip install --no-index --find-links $(WHEELHOUSE_DIR) -r $(LOCK_FILE)
	$(PYTHON) -m pip install --no-index --find-links $(WHEELHOUSE_DIR) --no-deps -e .
	$(PYTHON) -m pre_commit install

wheelhouse-clean:
	rm -rf $(WHEELHOUSE_DIR) $(WHEELHOUSE_REPORT)

wheelhouse-report: wheelhouse-validate
	@echo "Wheelhouse report: $(WHEELHOUSE_REPORT)"

PYTHON_READY_TARGETS := setup install demo reproduce dev-setup quickstart-smoke wheelhouse-build wheelhouse-validate dev-env-offline test test-all test-gate test-diagnostics test-determinism test-validation test-property test-integration test-e2e entropy-gate coverage coverage-fast coverage-baseline coverage-gate mutation mutation-ci mutation-baseline mutation-check mutation-check-strict api-contract validate-api-maturity format fix lint mypy ssot validate-claims-coverage docs-evidence security sbom profile ci-artifacts flake-report cleanroom docs build release release-readiness manifest manifest-validate inventory inventory-check perfection-gate launch-gate smlrs-gate dsio-gate traceability-check public-surfaces
$(PYTHON_READY_TARGETS): $(ENV_READY)

test:
	$(MAKE) test-gate

test-all:
	mkdir -p $(JUNIT_DIR)
	$(PYTHON) -m pytest -q --junitxml=$(JUNIT_ALL)

test-integration:
	$(PYTHON) -m pytest -m "integration" -q

test-e2e:
	$(PYTHON) -m pytest -m "e2e" -q

test-gate:
	$(TEST_CMD)

test-diagnostics:
	$(PYTHON) -m scripts.run_pytest_with_diagnostics \
		--markers "not (validation or property)" \
		--junit $(JUNIT_FAST) \
		--log $(JUNIT_DIR)/pytest-fast.log \
		--output-json $(JUNIT_DIR)/failure-diagnostics.json \
		--output-md $(JUNIT_DIR)/failure-diagnostics.md

test-determinism:
	$(PYTHON) -m pytest tests/test_determinism.py tests/test_properties_determinism.py -q

test-validation:
	$(PYTHON) -m pytest -m "validation" -q

test-property:
	$(PYTHON) -m pytest -m "property" -q

entropy-gate:
	$(PYTHON) -m tools.entropy_gate --mode current

coverage:
	$(PYTHON) -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml -q

coverage-fast:
	$(PYTHON) -m pytest -m "not (validation or property)" --cov=bnsyn --cov-report=term-missing --cov-report=xml:coverage.xml -q

coverage-baseline: coverage
	$(PYTHON) -m scripts.generate_coverage_baseline --coverage-xml coverage.xml --output quality/coverage_gate.json --minimum-percent 99.0

coverage-gate: coverage
	$(PYTHON) -m scripts.check_coverage_gate --coverage-xml coverage.xml --baseline quality/coverage_gate.json


mutation:
	@echo "Running mutation profile (reproducible local workflow step)..."
	@$(PYTHON) -m pip install -e ".[test]" -q
	@$(PYTHON) -m pip install mutmut==2.4.5 -q
	@$(PYTHON) -m scripts.run_mutation_pipeline

mutation-ci:
	@echo "Emitting mutation CI artifacts to local files..."
	@baseline_file=quality/mutation_baseline.json; \
	output_file=.mutation_ci_output; \
	summary_file=.mutation_ci_summary.md; \
	: > $$output_file; \
	: > $$summary_file; \
	GITHUB_OUTPUT=$$output_file GITHUB_STEP_SUMMARY=$$summary_file $(PYTHON) -m scripts.mutation_ci_summary --baseline $$baseline_file --write-output --write-summary

mutation-baseline:
	@echo "Running mutation testing to establish baseline..."
	@$(PYTHON) -m pip install -e ".[test]" -q
	@$(PYTHON) -m pip install mutmut==2.4.5 -q
	@$(PYTHON) -m scripts.generate_mutation_baseline

mutation-check:
	@echo "Running mutation testing against baseline..."
	@$(PYTHON) -m pip install -e ".[test]" -q
	@$(PYTHON) -m pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@$(PYTHON) -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: +/-{baseline['tolerance_delta']}%)\")"
	@$(PYTHON) -m scripts.validate_mutation_baseline
	@$(PYTHON) -m scripts.run_mutation_pipeline
	@$(PYTHON) -m scripts.check_mutation_score --advisory

mutation-check-strict:
	@echo "Running mutation testing against baseline (STRICT MODE)..."
	@$(PYTHON) -m pip install -e ".[test]" -q
	@$(PYTHON) -m pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@$(PYTHON) -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: +/-{baseline['tolerance_delta']}%)\")"
	@$(PYTHON) -m scripts.validate_mutation_baseline
	@$(PYTHON) -m scripts.run_mutation_pipeline
	@$(PYTHON) -m scripts.check_mutation_score --strict

api-contract:
	$(PYTHON) -m scripts.check_api_contract --baseline quality/api_contract_baseline.json

validate-api-maturity:
	$(PYTHON) -m scripts.validate_api_maturity

quality: format lint mypy ssot security
	@echo "All quality checks passed"

format:
	$(PYTHON) -m ruff format .
	@echo "Formatted code"

fix:
	$(PYTHON) -m ruff check . --fix
	@echo "Fixed lint issues"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pylint --fail-under=9.5 src/bnsyn

mypy:
	$(PYTHON) -m mypy src --strict --config-file pyproject.toml

typecheck: mypy

ssot:
	$(PYTHON) -m scripts.validate_bibliography
	$(PYTHON) -m scripts.validate_claims
	$(PYTHON) -m scripts.scan_normative_tags
	$(PYTHON) -m scripts.validate_pr_gates
	$(PYTHON) -m scripts.validate_required_status_contexts
	$(PYTHON) -m scripts.sync_required_status_contexts --check
	$(MAKE) inventory-check
	$(PYTHON) -m scripts.validate_api_maturity
	$(MAKE) api-contract
	$(MAKE) manifest-check
	$(MAKE) traceability-check

validate-claims-coverage:
	$(PYTHON) -m scripts.validate_claims_coverage --format markdown

docs-evidence:
	$(PYTHON) -m scripts.generate_evidence_coverage

SECURITY_ARTIFACT_DIR ?= artifacts/security
SECURITY_GITLEAKS_REPORT ?= $(SECURITY_ARTIFACT_DIR)/gitleaks-report.json
SECURITY_REPORT ?= $(SECURITY_ARTIFACT_DIR)/pip-audit.json
SECURITY_SAST_REPORT ?= $(SECURITY_ARTIFACT_DIR)/bandit.json

security:
	$(PYTHON) --version
	$(PYTHON) -m pip --version
	$(PYTHON) -m pip install --upgrade pip==26.0.1
	$(PYTHON) -m pip install --require-hashes -r requirements-lock.txt
	$(PYTHON) -m pip install --no-deps -e .
	mkdir -p $(SECURITY_ARTIFACT_DIR)
	$(PYTHON) -m scripts.ensure_gitleaks -- detect --redact --verbose --source=. --config=.gitleaks.toml --report-format=json --report-path=$(SECURITY_GITLEAKS_REPORT)
	$(PYTHON) -m pip_audit --desc --format json --requirement requirements-lock.txt --output $(SECURITY_REPORT)
	$(PYTHON) -m bandit -r src/ -ll -f json -o $(SECURITY_SAST_REPORT)

SBOM_REPORT ?= artifacts/sbom/sbom.cdx.json
SBOM_LOCK_FILE ?= requirements-sbom-lock.txt

sbom:
	$(PYTHON) --version
	$(PYTHON) -m pip --version
	$(PYTHON) -m pip install --require-hashes --no-deps -r $(SBOM_LOCK_FILE)
	mkdir -p $(dir $(SBOM_REPORT))
	./.venv/bin/cyclonedx-py environment --output-format JSON --output-file $(SBOM_REPORT)

profile:
	$(PYTHON) -m scripts.profile_kernels --help

ci-artifacts: test-all flake-report
	$(PYTHON) -m scripts.write_evidence_manifest

flake-report:
	$(PYTHON) -m scripts.compute_flake_report --junit $(JUNIT_ALL) --protocol .repo-governor/protocol.yml --output artifacts/flake-report.json --summary artifacts/flake-report.md

cleanroom:
	$(MAKE) clean
	$(MAKE) install
	$(MAKE) build
	$(MAKE) test
	$(PYTHON) -m bnsyn --help

check: format lint mypy coverage ssot security
	@echo "All checks passed"

docs:
	$(PYTHON) -m pip install -e ".[docs]"
	$(PYTHON) -m sphinx -b html docs docs/_build/html
	@echo "Docs built at docs/_build/html"

build:
	$(PYTHON) -m pip install -e . build
	$(PYTHON) -m build

release: build release-readiness
	@echo "release artifacts and readiness report generated"

release-readiness:
	$(PYTHON) -m scripts.release_readiness

manifest:
	$(PYTHON) -m tools.manifest generate

manifest-validate:
	$(PYTHON) -m tools.manifest validate

manifest-check: manifest manifest-validate
	git diff --exit-code -- .github/REPO_MANIFEST.md manifest/repo_manifest.computed.json

inventory:
	$(PYTHON) tools/generate_inventory.py

inventory-check:
	$(PYTHON) tools/generate_inventory.py --check

perfection-gate:
	$(PYTHON) -m scripts.perfection_gate

launch-gate:
	$(PYTHON) -m scripts.launch_gate

smlrs-gate:
	$(PYTHON) -m scripts.smlrs_gate

dsio-gate:
	$(PYTHON) -m scripts.dsio_gate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name .coverage -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f .mutmut-cache
	rm -rf $(WHEELHOUSE_DIR) $(WHEELHOUSE_REPORT)
	rm -f artifacts/demo.json artifacts/demo.sha256 artifacts/reproduce_manifest.json artifacts/reproducibility_report.json
	@echo "Cleaned temporary files"


traceability-check:
	$(PYTHON) -m scripts.validate_traceability

public-surfaces:
	$(PYTHON) -m scripts.discover_public_surfaces
