.PHONY: validate-claims validate-bibliography validate-normative ssot test-smoke test-validation ci-local ci-full bench bench-micro bench-sweep bench-report

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
bench-micro:
	python scripts/run_benchmarks.py --suite micro --json-out benchmarks/results/bench_micro.json

bench:
	python scripts/run_benchmarks.py --suite full --json-out benchmarks/results/bench_full.json

bench-sweep: bench

bench-report:
	@echo "Benchmark reports are emitted to JSON under benchmarks/results/"
