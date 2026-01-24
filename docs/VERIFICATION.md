# Verification

## Local commands

```bash
make ssot
make lint
make typecheck
make test-smoke
make test-validation
make ci-local
```

## CI mapping

- **ci-pr / Evidence SSOT (claims+bib+rules)**: `make ssot`
- **ci-pr / Static quality (ruff+mypy)**: `ruff format --check .`, `ruff check .`, `mypy src`
- **ci-pr / Build (package installability)**: `python -m build`, `python -m pip install dist/*.whl`, `python -c "import bnsyn; import bnsyn.cli; print('OK')"`
- **ci-pr / Tests: smoke (fast)**: `pytest -m "not validation"`
- **ci-pr / Secrets scan (gitleaks)**: `gitleaks detect --redact --verbose --source=.`
- **ci-pr / Dependency vulns (pip-audit)**: `pip-audit`
- **codeql / analyze**: GitHub CodeQL Python analysis

## Branch protection required checks

- `ci-pr / Evidence SSOT (claims+bib+rules)`
- `ci-pr / Static quality (ruff+mypy)`
- `ci-pr / Build (package installability)`
- `ci-pr / Tests: smoke (fast)`
- `ci-pr / Secrets scan (gitleaks)`
- `ci-pr / Dependency vulns (pip-audit)`
- `codeql / analyze`
