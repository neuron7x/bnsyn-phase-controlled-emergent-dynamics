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
