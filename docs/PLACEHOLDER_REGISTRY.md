# Placeholder Registry

Canonical registry for placeholder scan findings produced by `python -m scripts.scan_placeholders`.

## Findings

- ID: PH-0001
- Type: code
- Path: `src/bnsyn/emergence/crystallizer.py:283-290`
- Symptom: `pass_in_except` in PCA failure fallback.
- Impact: silent exception branch reduced explicitness of deterministic fallback.
- Priority: P1
- Status: RESOLVED
- Acceptance Criteria:
  - LinAlgError in `_update_pca` preserves previous `_pca_components` and `_pca_mean`.
  - Warning is emitted for fallback branch.
- Tests:
  - `tests/test_crystallizer_edge_cases.py::test_crystallizer_pca_failure_retains_previous`
- Evidence:
  - `artifacts/ci_logs/pytest_targeted.txt`
  - `artifacts/ci_logs/pytest_full.txt`

- ID: PH-0002
- Type: test
- Path: `tests/test_coverage_gate.py:24-30`
- Symptom: `pass_in_except` in expected FileNotFoundError assertion.
- Impact: weaker assertion style for failure branch.
- Priority: P1
- Status: RESOLVED
- Acceptance Criteria:
  - Missing coverage file path raises `FileNotFoundError` with strict assertion style.
- Tests:
  - `tests/test_coverage_gate.py::test_read_coverage_percent_missing_file_fails`
- Evidence:
  - `artifacts/ci_logs/pytest_targeted.txt`
  - `artifacts/ci_logs/pytest_full.txt`

- ID: PH-0003
- Type: test
- Path: `tests/validation/test_chaos_integration.py:253-260`
- Symptom: `pass_in_except` in ValueError branch for extreme AdEx state.
- Impact: non-explicit control flow in chaos validation test.
- Priority: P1
- Status: RESOLVED
- Acceptance Criteria:
  - Test explicitly accepts ValueError while asserting finite outputs on non-error path.
- Tests:
  - `tests/validation/test_chaos_integration.py::test_adex_handles_extreme_values_robustly`
- Evidence:
  - `artifacts/ci_logs/pytest_targeted.txt`
  - `artifacts/ci_logs/pytest_full.txt`

## Scan Evidence

- `artifacts/ci_logs/scan_placeholders.txt`
- `artifacts/ci_logs/scan_placeholders.json`
