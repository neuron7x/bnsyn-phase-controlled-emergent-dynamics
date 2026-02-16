# Practical Usability Blockers Report

## Intended use inference
- **Inferred intended use**: Deterministic Bio-AI research/validation workflow with CLI execution, reproducible experiments, and CI-like verification gates.
- **Evidence sources (>=2 required, met with 3)**:
  1. `README.md` canonical onboarding commands.
  2. `docs/usage_workflows.md` golden-path workflow steps.
  3. `.github/workflows/ci-pr-atomic.yml` verification/build gates.

## Go/No-Go
- **Recommendation: NO-GO** until P0 blockers BLK-001 and BLK-002 are resolved.

## Top blockers (P0/P1 first)

### BLK-001 (P0) — Canonical test command fails after setup
- **Symptom**: test suite fails with `TypeError: string indices must be integers, not 'str'`.
- **Repro commands**:
  1. `python -m pip install -e '.[dev]'`
  2. `python -m pytest tests -q`
- **Evidence**:
  - `cmd:python -m pytest tests -q` → `proof_bundle/logs/250_20260216T120237Z.log` (exit 1)
  - `cmd:python -m pytest -m 'not validation' -q` → `proof_bundle/logs/253_20260216T120524Z.log` (exit 1)
  - `cmd:sed -n '1,220p' proof_bundle/index.json` → `proof_bundle/logs/264_20260216T120811Z.log` (shape is object with `artifacts`, not list)

### BLK-002 (P0) — Build gate command unavailable after documented install
- **Symptom**: `python -m build` fails with missing module.
- **Repro commands**:
  1. `python -m pip install -e '.[dev]'`
  2. `python -m build`
- **Evidence**:
  - `cmd:python -m build` → `proof_bundle/logs/257_20260216T120748Z.log` (exit 1)
  - `cmd:python -m pip install -e '.[dev]'` → `proof_bundle/logs/248_20260216T120209Z.log` (exit 0)
  - CI build gate requires the same command (`python -m build`) in `.github/workflows/ci-pr-atomic.yml` via `proof_bundle/logs/270_20260216T120835Z.log`

## Fastest unblock sequence
1. **BLK-002**: make build prerequisite deterministic and aligned between docs/CI (install `build` where required).
2. **BLK-001**: reconcile `proof_bundle/index.json` contract between producer and tests.

## Additional observations (non-blocking)
- `ruff check .`, `pylint src/bnsyn`, and `mypy src --strict --config-file pyproject.toml` pass after dev install (`proof_bundle/logs/254_20260216T120711Z.log`, `255_...`, `256_...`).
- Primary CLI/discovery workflows run (`python -m bnsyn.cli --help`, `python -m scripts.check_quickstart_consistency`, `make docs`).
