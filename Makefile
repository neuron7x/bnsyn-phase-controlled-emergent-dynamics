.PHONY: validate-claims validate-bibliography validate-normative ssot test-smoke test-validation ci-local ci-full bench bench-micro bench-sweep bench-report docs docs-clean docs-linkcheck

# SSOT validators
validate-claims:
	python scripts/validate_claims.py

validate-bibliography:
	python scripts/validate_bibliography.py

validate-normative:
	python scripts/scan_normative_tags.py

# Alias: run all SSOT validators
ssot: validate-bibliography validate-claims validate-normative

# Test targets
test-smoke:
	pytest -m "not validation"

test-validation:
	pytest -m validation

# Local CI check (SSOT + smoke)
ci-local: ssot test-smoke

# Full CI check (format, lint, audit, smoke, validation)
ci-full:
	ruff format --check .
	ruff check .
	python scripts/audit_spec_implementation.py
	$(MAKE) ssot
	$(MAKE) test-smoke
	$(MAKE) test-validation

# Benchmark targets
bench:
	python scripts/run_benchmarks.py --suite full --json-out benchmarks/results/full.json

bench-micro:
	python scripts/run_benchmarks.py --suite micro --json-out benchmarks/results/micro.json

bench-sweep:
	python benchmarks/run_benchmarks.py --scenario full --repeats 5 --out results/bench_full.csv --json results/bench_full.json

bench-report:
	python benchmarks/report.py --input results/bench.csv --output docs/benchmarks/README.md

# Docs targets
docs:
	sphinx-build -b html -W -a -v docs/api docs/_build/html

docs-clean:
	rm -rf docs/_build

docs-linkcheck:
	sphinx-build -b linkcheck docs/api docs/_build/linkcheck
