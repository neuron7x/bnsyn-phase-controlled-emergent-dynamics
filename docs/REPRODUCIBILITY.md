# Reproducibility & Determinism Protocol

**Navigation**: [INDEX.md](INDEX.md) | [SPEC.md](SPEC.md) | [GOVERNANCE.md](GOVERNANCE.md)

## Environment setup

Run from repository root:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,test]"
```

Expected output includes installation of `bnsyn` and development dependencies.

## SSOT gates

```bash
python scripts/validate_bibliography.py
python scripts/validate_claims.py
python scripts/scan_normative_tags.py
python scripts/scan_governed_docs.py
```

Expected outputs:
- `OK: bibliography SSOT validated.` (with counts for bibkeys, mapping, claims, lock)
- `[claims-gate] OK: N claims validated; M normative.`
- `OK: normative tag scan passed.`
- `[governed-docs] OK: governed docs have no orphan normative statements.`

## Generate evidence coverage

```bash
python scripts/generate_evidence_coverage.py
```

Expected output:
- `[evidence-coverage] Generated docs/EVIDENCE_COVERAGE.md`
- `[evidence-coverage] Claims: N`
- `[evidence-coverage] Tier-A: M`
- `[evidence-coverage] Normative: K`

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

## Rebuild sources lock (manual)

```bash
python scripts/rebuild_sources_lock.py
```

This regenerates `bibliography/sources.lock` deterministically. It is NOT run in CI
automatically; use it when updating bibliography entries.

## Determinism rules (repo contract)

1) All randomness flows through `numpy.random.Generator` created by `bnsyn.rng.seed_all(seed)`.
2) No hidden global RNGs inside modules.
3) All stochastic updates use √dt scaling for noise terms (when present).
4) Δt-invariance checks compare dt vs dt/2 against reference tolerances.
