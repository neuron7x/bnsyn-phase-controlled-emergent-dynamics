.PHONY: validate-claims validate-bibliography validate-normative doc-links ssot lint typecheck test-smoke test-validation ci-local

# SSOT validators
validate-claims:
	python scripts/validate_claims.py

validate-bibliography:
	python scripts/validate_bibliography.py

validate-normative:
	python scripts/scan_normative_tags.py

doc-links:
	python scripts/check_doc_links.py

# Alias: run all SSOT validators
ssot: validate-bibliography validate-claims validate-normative doc-links

# Quality targets
lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy src

# Test targets
test-smoke:
	pytest -m "not validation"

test-validation:
	pytest -m validation

# Local CI check (SSOT + smoke)
ci-local: ssot test-smoke
