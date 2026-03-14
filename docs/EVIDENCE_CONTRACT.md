# AOC v1.0 Evidence Contract

Required files per run:

- `final_artifact.md`
- `zeropoint.json`
- `run_summary.json`
- `sigma_trace.json`
- `delta_trace.json`
- `audit_trace.json`
- `auditor_reliability_trace.json`
- `termination_verdict.json`
- `evidence_bundle/`

`termination_verdict.json` fields:
- `status`
- `stop_reason`
- `iteration`
- `delta`
- `sigma_distance`
- `audit_passed`
- `band.min_delta`
- `band.max_delta`
