.PHONY: validate-claims validate-bibliography validate-normative ssot test-smoke test-validation ci-local docs-api docs-api-clean docstrings-check

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

# Docs targets
docs-api:
	sphinx-build -W -b html docs/sphinx docs/sphinx/_build/html

docs-api-clean:
	rm -rf docs/sphinx/_build

docstrings-check:
	python scripts/check_docstrings.py
