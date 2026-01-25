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
- **Primary Gate:** Self-contained `scripts/validate_coverage.py` (no external dependencies)
- **Observability:** Codecov upload for trend tracking (non-blocking)
- **Artifacts:** Coverage reports always preserved in GitHub Actions artifacts

### Coverage Validation Architecture
1. **PRIMARY GATE (blocking):** `python scripts/validate_coverage.py`
   - Validates coverage.json against 85% threshold
   - Self-contained, no external dependencies
   - Generates markdown report with coverage hotspots
   - Exit code 0 on pass, 1 on fail

2. **OBSERVABILITY LAYER (non-blocking):** Codecov upload
   - `fail_ci_if_error: false` - Codecov failures do NOT block CI
   - `continue-on-error: true` - Workflow continues on Codecov errors
   - Provides historical trend tracking when available

3. **ARTIFACTS (always preserved):** Coverage reports uploaded with `if: always()`

### Known CI Failure Modes
1. **Codecov Rate Limit (HTTP 429) - NON-BLOCKING**
   - **Behavior:** Codecov upload fails, but CI passes if coverage threshold met
   - **Quality Gate:** Primary validator still enforces 85% threshold
   - **Observability:** Degraded (no Codecov trend), coverage artifacts still available
   - **Action Required:** None - this is expected fail-safe behavior

2. **Coverage Below 85% Threshold - BLOCKING**
   - **Detection:** `validate_coverage.py` exits with code 1
   - **Report:** Markdown output shows coverage hotspots (lowest 5 files)
   - **Resolution:** Add tests to increase coverage above 85%

3. **Determinism Test Flakiness**
   - **Detection:** Multiple runs with different results
   - **Mitigation:** `PYTHONHASHSEED=0` and RNG isolation tests
