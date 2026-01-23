# Audit Findings

## Baseline Error Ledger

### FND-0001
- **Symptom:** `scripts/validate_bibliography.py` failed with `ModuleNotFoundError: No module named 'yaml'`.
- **Root cause:** `PyYAML` dependency not installed in the execution environment.
- **Anchors:** `scripts/validate_bibliography.py:17` (import `yaml`).
- **Fix strategy:** Ensure validator runs without missing dependency by adding `PyYAML` to runtime requirements or documenting setup in reproducibility docs; keep validator logic intact.
- **Acceptance criteria:** `python scripts/validate_bibliography.py` runs to completion without dependency errors.

### FND-0002
- **Symptom:** `pytest -m "not validation"` fails during test collection with `ModuleNotFoundError: No module named 'bnsyn'` across multiple tests.
- **Root cause:** Python package not installed or import path not configured for tests.
- **Anchors:** `tests/test_*` imports, e.g., `tests/test_adex_smoke.py:2`.
- **Fix strategy:** Configure tests to import the local package via editable install, `PYTHONPATH`, or test configuration (`conftest.py`).
- **Acceptance criteria:** `pytest -m "not validation"` collects and runs tests without import errors.

### FND-0003
- **Symptom:** `pytest -m validation` fails during test collection with same `ModuleNotFoundError: No module named 'bnsyn'`.
- **Root cause:** Same as FND-0002.
- **Anchors:** `tests/test_validation_largeN.py:2` and other test imports.
- **Fix strategy:** Same as FND-0002.
- **Acceptance criteria:** `pytest -m validation` collects and runs tests without import errors.
