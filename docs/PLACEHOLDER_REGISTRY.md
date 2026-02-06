# Placeholder Registry

This registry tracks placeholder/stub scan findings and their disposition.

## Findings

- ID: PH-0001
  - Type: code
  - Path: `src/bnsyn/emergence/crystallizer.py:283-290`
  - Symptom: `except np.linalg.LinAlgError` branch ends with `pass`.
  - Impact: None; branch preserves the previous PCA basis intentionally after logging warning.
  - Owner: agent
  - Priority: P1
  - Acceptance Criteria (AC): on SVD failure, previous PCA components remain unchanged and warning is emitted.
  - Test Plan (TP): covered by existing deterministic PCA-failure path tests in `tests/test_crystallizer.py` (warning branch and no-crash behavior).
  - Status: DONE
  - Exit criteria: behavior verified as intentional exception fallback; no placeholder logic remains.

- ID: PH-0002
  - Type: code
  - Path: `src/bnsyn/provenance/manifest_builder.py:69-73`
  - Symptom: `except metadata.PackageNotFoundError: pass`.
  - Impact: None; explicit fallback path reads `pyproject.toml` and returns stable version.
  - Owner: agent
  - Priority: P1
  - Acceptance Criteria (AC): if package metadata is unavailable, version is resolved from `pyproject.toml` or defaults to `0.0.0`.
  - Test Plan (TP): existing tests in `tests/test_manifest_builder.py` validate fallback git/version behavior.
  - Status: DONE
  - Exit criteria: fallback path confirmed deterministic and covered.

- ID: PH-0003
  - Type: docs
  - Path: `GOVERNANCE_VERIFICATION_REPORT.md:1-45`
  - Symptom: contains explicit template markers.
  - Impact: None; file is intentionally non-executable and policy-gated to prevent stale certification claims.
  - Owner: agent
  - Priority: P2
  - Status: DONE
  - Exit criteria: retained by design; documented as policy template, not implementation placeholder.

- ID: PH-0004
  - Type: test
  - Path: `tests/test_coverage_gate.py:27-31`, `tests/validation/test_chaos_integration.py:254-261`
  - Symptom: `try/except` assertions using `pass` in expected exception branches.
  - Impact: None; test intent is explicit and behaviorally complete.
  - Owner: agent
  - Priority: P1
  - Acceptance Criteria (AC): tests fail on unexpected success and accept documented exception outcomes.
  - Test Plan (TP): covered by current baseline pytest run.
  - Status: DONE
  - Exit criteria: no missing assertions or unexercised placeholder paths.

## Scan Evidence

- `artifacts/ci_logs/scan_rg_keywords.txt`
- `artifacts/ci_logs/scan_py_stubs.txt`
- `artifacts/ci_logs/scan_docs_stubs.txt`
