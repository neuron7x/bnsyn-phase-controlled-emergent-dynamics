# Contributing Guide

## Before Creating a PR

1. Install dev environment:
   ```bash
   make dev-setup
   ```

2. Make your changes

3. Run all checks:
   ```bash
   make check
   ```

4. Verify coverage:
   ```bash
   make coverage
   ```

5. Test locally with Docker:
   ```bash
   docker build -t bnsyn-dev .
   docker run bnsyn-dev
   ```

## PR Checklist

See `.github/pull_request_template.md` — all items are REQUIRED.

## CI/CD Pipeline

All PRs must pass:
- **determinism**: 3 identical runs with same seed
- **quality**: ruff, mypy, pylint
- **build**: python -m build + import
- **tests-smoke**: pytest -m "not validation" with ≥85% coverage
- **ssot**: bibliography, claims, normative tags validation
- **security**: gitleaks, pip-audit, bandit
- **finalize**: all jobs must pass

See [CI_GATES.md](docs/CI_GATES.md) for exact commands.

## CI/CD Quality Gates

### Coverage Requirements
- **Minimum:** 85% line coverage (enforced by `--cov-fail-under=85`)
- **Reporting:** Codecov (authenticated upload with fallback artifacts)
- **Failure Mode:** If Codecov is unavailable, the local coverage threshold still enforces the gate

### Known CI Failure Modes
1. **Codecov Rate Limit (HTTP 429)**
   - **Root Cause:** Anonymous upload without token
   - **Resolution:** Ensure `CODECOV_TOKEN` secret is configured
   - **Fallback:** Coverage artifacts still uploaded to GitHub Actions

2. **Determinism Test Flakiness**
   - **Detection:** Multiple runs with different results
   - **Mitigation:** `PYTHONHASHSEED=0` and RNG isolation tests
