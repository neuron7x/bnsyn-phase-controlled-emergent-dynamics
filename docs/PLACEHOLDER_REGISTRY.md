# Placeholder Registry

Canonical registry for placeholder scan findings produced by `python -m scripts.scan_placeholders`.

## Findings

- ID: PH-0001
- Type: code
- Path: `src/bnsyn/emergence/crystallizer.py:290`
- Symptom: `pass_in_except`
- Impact: exception branch keeps previous PCA state without explicit guard-level assertion.
- Priority: P1
- Status: OPEN
- Exit criteria: replace `pass` branch with explicit fallback semantics and add deterministic tests for SVD failure behavior.

- ID: PH-0002
- Type: test
- Path: `tests/test_coverage_gate.py:31`
- Symptom: `pass_in_except`
- Impact: expected-exception branch relies on `pass` instead of strict assertion helper.
- Priority: P1
- Status: OPEN
- Exit criteria: rewrite using pytest exception assertions without `pass` and preserve failure signal quality.

- ID: PH-0003
- Type: test
- Path: `tests/validation/test_chaos_integration.py:261`
- Symptom: `pass_in_except`
- Impact: chaos validation exception branch uses `pass`, reducing explicitness of contract checks.
- Priority: P1
- Status: OPEN
- Exit criteria: rewrite with explicit assertion strategy for accepted ValueError branch.

## Scan Evidence

- `artifacts/ci_logs/scan_placeholders.txt`
- `artifacts/ci_logs/scan_placeholders.json`
