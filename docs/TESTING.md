# Testing & Coverage (Canonical)

This document is the **single source of truth** for running tests and coverage in this repository.

## Install test dependencies

```bash
python -m pip install -e ".[test]"
```

Expected output pattern:
- `Successfully installed bnsyn-...`
- No import errors for `pytest`, `pytest-cov`, `hypothesis`.

## Run default test suite

```bash
make test
```

Equivalent explicit command:

```bash
python -m pytest -m "not validation" -q
```

Expected output pattern:
- Dots for passing tests.
- Final summary with `passed` and optional `skipped`.

## Run smoke marker tests

```bash
python -m pytest -m smoke -q
```

Expected output pattern:
- Only smoke-marked tests.

## Generate coverage artifacts

```bash
make coverage
```

Equivalent explicit command:

```bash
python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml -q
```

Artifacts:
- Terminal report with missing lines by module.
- `coverage.xml` at repository root.

## Generate / refresh coverage baseline

```bash
make coverage-baseline
```

Equivalent explicit command:

```bash
python scripts/generate_coverage_baseline.py --coverage-xml coverage.xml --output quality/coverage_gate.json --minimum-percent 99.0
```

This baseline uses the same metric enforced by the gate: `coverage.xml line-rate`.

## Enforce coverage gate

```bash
make coverage-gate
```

Gate behavior:
- Fails if current coverage drops below baseline in `quality/coverage_gate.json`.
- Fails if current coverage drops below minimum floor in `quality/coverage_gate.json`.

## CI parity checks (local)

Use the same checks enforced in PR CI:

```bash
python -m pytest -q
python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml -q
ruff check .
```

If a tool is unavailable locally, install via:

```bash
python -m pip install -e ".[test]"
```

Deferred gate note:
- `mypy` is configured in CI quality workflow; run locally only after full `.[dev]` install.


## Offline dependency workflow (Python 3.11)

Build the local wheelhouse from pinned lock dependencies:

```bash
make wheelhouse-build
```

Validate that every pinned dependency in `requirements-lock.txt` has a matching wheel in `wheelhouse/`:

```bash
make wheelhouse-validate
```

Install the development environment fully offline from the local wheelhouse:

```bash
make dev-env-offline
```

Equivalent install command:

```bash
pip install --no-index --find-links wheelhouse -r requirements-lock.txt
```

## Updating lock and wheelhouse

1. Refresh lock file after dependency changes:

```bash
pip-compile --extra=dev --generate-hashes --output-file=requirements-lock.txt pyproject.toml
```

2. Rebuild wheels for Python 3.11 and re-run validation:

```bash
make wheelhouse-build
make wheelhouse-validate
```
