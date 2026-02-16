# RUNBOOK: determinism regression

## Detection signals
- CI job: `determinism` in `.github/workflows/ci-pr-atomic.yml`.
- Artifact/log source: pytest output from `tests/test_determinism.py` and `tests/properties/test_properties_determinism.py`.

## Triage checklist
1. Confirm failing seed/test pair from CI logs.
2. Check recent changes touching `src/bnsyn/rng.py`, `src/bnsyn/sim/network.py`, or CLI defaults.
3. Verify no global RNG usage in test output.
4. Compare run 1/2/3 outputs in determinism job summary.

## Reproduction commands
```bash
python -m pytest tests/test_determinism.py tests/properties/test_properties_determinism.py -q
python -m pytest tests/test_determinism.py::test_no_global_numpy_rng_usage -q
```

## Rollback/mitigation procedure
1. Revert the latest merge that changed RNG plumbing or state update order.
2. Pin deterministic seed path in affected call sites.
3. Re-run determinism tests locally and in PR CI.

## Known failure modes
| Failure mode | Signal | Mitigation |
|---|---|---|
| Global NumPy RNG used | `test_no_global_numpy_rng_usage` fails | route randomness through `seed_all()` pack |
| Non-deterministic iteration order | run1/run2 mismatch only in aggregate metrics | enforce sorted/stable iteration |
| Hidden time-based seed | repeated local run mismatch | remove wall-clock seed defaults |
