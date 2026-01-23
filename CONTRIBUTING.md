# Contributing

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -m "not validation"
```

## Test tiers
- smoke: fast deterministic CI
- validation: slow / statistical (run via workflow_dispatch or schedule)
