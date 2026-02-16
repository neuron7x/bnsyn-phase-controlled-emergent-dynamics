# Noise Cleanup Report

## Expected Commands
- Tests: `python -m pytest -m "not validation" -q`
- Lint: `ruff check .`
- Lint: `pylint src/bnsyn`
- Typecheck: `mypy src --strict --config-file pyproject.toml`
- Build: `python -m build`

## Actions
- Enumerated cache/noise candidates using `find` and git status/ignore checks.
- Classified each candidate against T1–T4 with per-path evidence logs.
- Deleted only approved cache/byproduct directories.
- Re-ran quality gates and recorded environment limitations.

## Deleted Items
- `./experiments/__pycache__`
- `./.pytest_cache`
- `./entropy/__pycache__`
- `./benchmarks/__pycache__`
- `./tools/__pycache__`
- `./.ruff_cache`
- `./scripts/__pycache__`
- `./.mypy_cache`
- `./tests/__pycache__`

## Kept Items
- No additional candidates were kept after rubric evaluation.

## Validation Outcome
- `ruff check .`: pass.
- `mypy src --strict --config-file pyproject.toml`: pass.
- `pytest` collection fails due missing dependencies (`psutil`, `hypothesis`, `yaml`) in environment.
- `pylint` unavailable in environment.
- `python -m build` unavailable (`build` module missing).
