# AOC v1.0 Evidence Contract

Each run directory must contain:

- `zeropoint.json`
- `final_artifact.json`
- `run_summary.json`
- `sigma_trace.json`
- `delta_trace.json`
- `audit_trace.json`
- `auditor_reliability_trace.json`
- `modulation_trace.json`
- `state_trace.json`
- `termination_verdict.json`
- `evidence_bundle/` (contains copies of all the files above)

## `termination_verdict.json`

Required keys:

- `status` in `{PASS, FAIL, MAX_ITER, INCONCLUSIVE}`
- `stop_reason` in `{productive_emergence, critical_failure, max_iterations, drift_exceeded, insufficient_progress, inconclusive_audit, other}`
- `iteration`
- `delta`
- `sigma_distance`
- `audit_passed`
- `band.min_delta`
- `band.max_delta`

All traces must persist for pass and non-pass paths.
