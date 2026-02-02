# CI Gates

## Overview

The CI configuration enforces deterministic verification and documentation correctness. Gates are designed to be reproducible and run with repository-defined tooling.

## Core gates (PR)

| Gate | Purpose | Local command |
| --- | --- | --- |
| SSOT validation | Validate bibliographic and governance artifacts. | `python scripts/validate_bibliography.py`<br>`python scripts/validate_claims.py`<br>`python scripts/scan_governed_docs.py`<br>`python scripts/scan_normative_tags.py` |
| Safety schema validation | Validate STPA safety artifacts against schemas. | `make validate-safety` |
| Docs link check | Ensure internal docs links and anchors are valid. | `python tools/check_docs_links.py docs` |
| Smoke tests | Deterministic smoke test suite. | `pytest -m "not (validation or property)"` |

## Full validation gates

| Gate | Purpose | Local command |
| --- | --- | --- |
| Full test suite | Complete test coverage. | `pytest -q` |
| Validation tests | Long-running validation suite. | `pytest -m validation` |

## Related references

- [CI gate policy](../CI_GATES.md)
- [Documentation index](../INDEX.md)
