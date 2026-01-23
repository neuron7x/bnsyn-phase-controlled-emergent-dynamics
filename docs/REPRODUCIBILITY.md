# Reproducibility & Determinism Protocol

## Determinism rules (repo contract)

1) All randomness must flow through `numpy.random.Generator` created by `bnsyn.rng.seed_all(seed)`.
2) No hidden global RNGs inside modules.
3) All stochastic updates must use **√dt** scaling for noise terms (when present).
4) Δt-invariance checks must compare dt vs dt/2 against reference tolerances.

## CI test segregation

- Smoke tests: `pytest -m "not validation"`
- Validation tests (slow): `pytest -m validation`
 - SSOT gates: `python scripts/validate_bibliography.py` and `python scripts/validate_claims.py`

This repo keeps CI fast and deterministic while allowing heavy statistical validation to run
manually or on schedule.

## Packaging

- `pyproject.toml` pins minimal runtime deps.
- `LICENSE` + `CITATION.cff` included.
