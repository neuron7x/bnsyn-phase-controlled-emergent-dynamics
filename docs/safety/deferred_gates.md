# Safety Gates Status

## Enforced Gates

| gate_id | tool | workflow/job | status | reproduction command | owner | follow-up link |
| --- | --- | --- | --- | --- | --- | --- |
| SAFETY-ARTIFACTS | `tools/safety/check_safety_artifacts.py` | `.github/workflows/workflow-integrity.yml` / `validate-workflows` | enforced | `python tools/safety/check_safety_artifacts.py` | Safety Engineering | N/A |
| WORKFLOW-ACTIONLINT | `actionlint` + `shellcheck` | `.github/workflows/workflow-integrity.yml` / `validate-workflows` (PR: changed workflows only; schedule: advisory full scan) | enforced | `actionlint -verbose .github/workflows/*.yml` | Release Engineering | N/A |

## Deferred Gates

| gate_id | tool | workflow/job | status | reproduction command | owner | follow-up link |
| --- | --- | --- | --- | --- | --- | --- |
| DETERMINISTIC-THREADS | Thread/BLAS determinism controls | _Not yet implemented_ (planned gate for runtime/CI env controls) | deferred | `rg -n "OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS|NUMEXPR" src docs` | Safety Engineering | docs/safety/followups.md#FUP-001 |
| NUMERIC-HEALTH-RUNTIME | Runtime NaN/Inf checks | _Not yet implemented_ (planned gate for runtime health checks) | deferred | `rg -n "validate_numeric_health" src/bnsyn` | Safety Engineering | docs/safety/followups.md#FUP-002 |
