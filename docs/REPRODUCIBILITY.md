# Reproducibility & Determinism Protocol

## Environment setup

Run from repository root:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Expected output includes installation of `bnsyn` and development dependencies.

## SSOT gates

```bash
python scripts/validate_bibliography.py
python scripts/validate_claims.py
python scripts/scan_normative_tags.py
```

Expected outputs:
- `OK: bibliography SSOT validated.`
- `[claims-gate] OK: ... claims validated; ... normative.`
- `[normative-scan] OK: governed docs have no orphan normative statements.`

## Smoke tests

```bash
pytest -m "not validation"
```

Expected output reports passing smoke tests with the validation suite deselected.

## Validation tests

```bash
pytest -m validation
```

Expected output reports passing validation tests.

## Determinism rules (repo contract)

1) All randomness flows through `numpy.random.Generator` created by `bnsyn.rng.seed_all(seed)`.
2) No hidden global RNGs inside modules.
3) All stochastic updates use √dt scaling for noise terms (when present).
4) Δt-invariance checks compare dt vs dt/2 against reference tolerances.
