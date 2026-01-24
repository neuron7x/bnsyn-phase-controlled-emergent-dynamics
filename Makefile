.PHONY: validate-claims validate-bibliography validate-normative ssot test-smoke test-validation ci-local bench bench-sweep bench-report

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

# Benchmark targets
bench:
	python benchmarks/run_benchmarks.py --scenario quick --repeats 3 --out results/bench.csv --json results/bench.json

bench-sweep:
	python benchmarks/run_benchmarks.py --scenario full --repeats 5 --out results/bench_full.csv --json results/bench_full.json

bench-report:
	python benchmarks/report.py --input results/bench.csv --output docs/benchmarks/README.md
