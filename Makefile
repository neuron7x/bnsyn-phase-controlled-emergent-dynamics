.PHONY: validate-claims validate-bibliography validate-all


validate-claims:
	python scripts/validate_claims.py

validate-bibliography:
	python scripts/validate_bibliography.py

validate-all: validate-claims validate-bibliography
