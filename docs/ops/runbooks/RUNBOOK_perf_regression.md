# RUNBOOK: performance regression

## Detection signals
- CI job: `performance-budgets` (added in PR workflow).
- Artifact/log source: `quality/perf_results.json` compared against `quality/perf_baseline.json`.

## Triage checklist
1. Identify metric exceeding budget (`duration_seconds` by profile).
2. Diff recent changes in `src/bnsyn/sim/` and `benchmarks/`.
3. Validate deterministic profile inputs (seed/N/steps unchanged).
4. Check environment skew (Python version, CPU class) in CI logs.

## Reproduction commands
```bash
python scripts/run_perf_smoke.py --output quality/perf_results.json
python scripts/check_perf_budget.py --baseline quality/perf_baseline.json --budgets quality/perf_budgets.yml --results quality/perf_results.json
```

## Rollback/mitigation procedure
1. Revert performance-impacting merge when over-budget > tolerated threshold.
2. If intentional, update baseline only with maintainer approval and rationale in PR.
3. Land optimized fix and re-enable gate.

## Known failure modes
| Failure mode | Signal | Mitigation |
|---|---|---|
| Algorithmic complexity increase | step-time rises with same N/steps | restore O(N) path / vectorization |
| Debug logging in hot loop | CPU spikes, noisy logs | remove per-step logging |
| Changed benchmark fixture | profile mismatch warning | restore canonical profile |
