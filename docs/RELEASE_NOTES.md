# Release Notes

## Release Candidate

This release candidate focuses on verifiable readiness for a public demo with
deterministic behavior, build/install validation, and audit-grade evidence.

### Verified in this RC

- **Quality gates**: `make check` (format, lint, mypy, coverage, SSOT, security).
- **Test gates**: `make test` (non-validation suite).
- **Coverage**: ≥95% line coverage for `src/bnsyn` with JSON report.
- **Determinism**: three consecutive non-validation test runs.
- **Packaging**: `pip install -e ".[dev]"` and `python -m build`.
- **Security evidence**: gitleaks, pip-audit, and bandit logs captured.

See `artifacts/release_rc/` for the command logs and reports used as evidence.
