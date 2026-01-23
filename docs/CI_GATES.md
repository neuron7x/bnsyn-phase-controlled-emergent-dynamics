# CI Gates

This repository enforces audit-grade CI gates. The commands listed below are the
exact commands executed by CI.

## PR-required checks

**Workflow: `ci-pr` (.github/workflows/ci-pr.yml)**

- **Job: `ssot`**
  - `python scripts/validate_bibliography.py`
  - `python scripts/validate_claims.py`
  - `python scripts/scan_normative_tags.py`
- **Job: `quality`**
  - `ruff format --check .`
  - `ruff check .`
  - `mypy src`
- **Job: `build`**
  - `python -m build`
  - `python -c "import bnsyn"`
- **Job: `tests-smoke`**
  - `pytest -m "not validation"`
- **Job: `gitleaks`**
  - `gitleaks detect --redact --verbose --source=.`
  - `gitleaks detect --redact --verbose --log-opts=<base..head>`
- **Job: `pip-audit`**
  - `pip-audit`

**Workflow: `codeql` (.github/workflows/codeql.yml)**

- **Job: `analyze`**
  - `github/codeql-action` (Python)

## Scheduled/manual checks (non-PR blocking)

**Workflow: `ci-validation` (.github/workflows/ci-validation.yml)**

- **Job: `ssot`**
  - `python scripts/validate_bibliography.py`
  - `python scripts/validate_claims.py`
  - `python scripts/scan_normative_tags.py`
- **Job: `tests-validation`**
  - `pytest -m validation`

## Required checks list (branch protection)

Configure branch protection for the `main` branch to require the following
checks to pass:

- `ci-pr / ssot`
- `ci-pr / quality`
- `ci-pr / build`
- `ci-pr / tests-smoke`
- `ci-pr / gitleaks`
- `ci-pr / pip-audit`
- `codeql / analyze`

## How to enable branch protection

1. Open repository **Settings → Branches**.
2. Add or edit the protection rule for `main`.
3. Enable **Require status checks to pass before merging**.
4. Select the required checks listed above (exact names).
5. Save the rule.
