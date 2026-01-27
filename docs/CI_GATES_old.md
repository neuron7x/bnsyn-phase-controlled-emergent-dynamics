# CI Gates

This repository enforces audit-grade CI gates. The commands listed below are the
exact commands executed by CI.

## PR checks

**Workflow: `ci-pr` (.github/workflows/ci-pr.yml)**

- **Job: `ssot`**
  - `python scripts/validate_bibliography.py`
  - `python scripts/validate_claims.py`
  - `python scripts/scan_governed_docs.py`
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

**Workflow: `ci-smoke` (.github/workflows/ci-smoke.yml)**

- **Job: `ssot`**
  - `python scripts/validate_bibliography.py`
  - `python scripts/validate_claims.py`
  - `python scripts/scan_governed_docs.py`
  - `python scripts/scan_normative_tags.py`
- **Job: `tests-smoke`**
  - `pytest -m "not validation"`

**Workflow: `codeql` (.github/workflows/codeql.yml)**

- **Job: `analyze`**
  - `github/codeql-action` (Python)

## Scheduled/manual checks (non-PR blocking)

**Workflow: `ci-validation` (.github/workflows/ci-validation.yml)**

- **Job: `ssot`**
  - `python scripts/validate_bibliography.py`
  - `python scripts/validate_claims.py`
  - `python scripts/scan_governed_docs.py`
  - `python scripts/scan_normative_tags.py`
- **Job: `tests-validation`**
  - `pytest -m validation`

## Branch protection check list

Configure branch protection for the `main` branch to include the following
checks:

- `ci-pr / ssot`
- `ci-pr / quality`
- `ci-pr / build`
- `ci-pr / tests-smoke`
- `ci-pr / gitleaks`
- `ci-pr / pip-audit`
- `ci-smoke / ssot`
- `ci-smoke / tests-smoke`
- `codeql / analyze`

## How to enable branch protection

1. Open repository **Settings → Branches**.
2. Add or edit the protection rule for `main`.
3. Enable **Require status checks to pass before merging**.
4. Select the checks listed above (exact names).
5. Save the rule.
