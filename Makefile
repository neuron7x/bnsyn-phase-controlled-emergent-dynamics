.PHONY: dev-setup check test test-determinism test-validation coverage coverage-gate quality format fix lint mypy ssot security clean docs validate-claims-coverage docs-evidence mutation-baseline mutation-check mutation-check-strict release-readiness

dev-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,test]"
	pre-commit install
	pre-commit autoupdate

test:
	python -m pytest -m "not validation" -q

test-determinism:
	pytest tests/test_determinism.py tests/test_properties_determinism.py -v

test-validation:
	pytest -m validation -v

coverage:
	python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml -q

coverage-gate: coverage
	python scripts/check_coverage_gate.py --coverage-xml coverage.xml --baseline quality/coverage_gate.json

mutation-baseline:
	@echo "🧬 Running mutation testing to establish baseline..."
	@pip install -e ".[test]" -q
	@pip install mutmut==2.4.5 -q
	@python scripts/generate_mutation_baseline.py

mutation-check:
	@echo "🧬 Running mutation testing against baseline..."
	@pip install -e ".[test]" -q
	@pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@python -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: ±{baseline['tolerance_delta']}%)\")"
	@mutmut run --paths-to-mutate="src/bnsyn/neuron/adex.py,src/bnsyn/plasticity/stdp.py,src/bnsyn/plasticity/three_factor.py,src/bnsyn/temperature/schedule.py" --tests-dir=tests --runner="pytest -x -q -m 'not validation and not property and not benchmark'"
	@mutmut results
	@python scripts/check_mutation_score.py --advisory

mutation-check-strict:
	@echo "🧬 Running mutation testing against baseline (STRICT MODE)..."
	@pip install -e ".[test]" -q
	@pip install mutmut==2.4.5 -q
	@rm -rf .mutmut-cache
	@python -c "import json; baseline=json.load(open('quality/mutation_baseline.json')); print(f\"Baseline: {baseline['baseline_score']}% (tolerance: ±{baseline['tolerance_delta']}%)\")"
	@mutmut run --paths-to-mutate="src/bnsyn/neuron/adex.py,src/bnsyn/plasticity/stdp.py,src/bnsyn/plasticity/three_factor.py,src/bnsyn/temperature/schedule.py" --tests-dir=tests --runner="pytest -x -q -m 'not validation and not property and not benchmark'"
	@mutmut results
	@python scripts/check_mutation_score.py --strict

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

release-readiness:
	python scripts/release_readiness.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name .coverage -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f .mutmut-cache
	@echo "Cleaned temporary files"
