# Contributing

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
pytest -m "not validation"
```

## Pre-commit hooks
- Commit-time hooks run formatting and type checks.
- Push-time hooks run smoke tests, coverage, and SSOT gates.

```bash
pre-commit run --all-files
```

## Test tiers
- smoke: fast deterministic CI
- validation: slow / statistical (run via workflow_dispatch or schedule)
