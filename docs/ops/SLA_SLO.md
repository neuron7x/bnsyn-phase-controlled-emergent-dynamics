---
service: bnsyn-cli-and-simulation
owner: bnsyn-maintainers
reviewed_on: 2026-02-16
slo_window: 28d
---

# SLA/SLO

## Targets

- **determinism_pass_rate**: ">=99.0%" (CI `determinism` job green over rolling 28 days).
- **contract_health**: ">=99.5%" (CLI + schema contract tests green in PR CI).
- **perf_smoke_p95_seconds**: "<=2.50" (fixed benchmark case in `scripts/check_perf_budget.py`).
- **security_scan_freshness_hours**: "<=168" (`dependency-watch` workflow must run at least weekly).
- **docs_build_health**: ">=99.0%" (`docs-pr` job green).

## Core latency profile

- Dataset/profile: deterministic sleep-stack smoke (`N=64`, `steps_wake=80`, `steps_sleep=60`, `seed=7`, backend `reference`).
- Latency SLO:
  - P50 <= 1.80 s
  - P95 <= 2.50 s
  - P99 <= 3.20 s

## Error budget policy

- Error budget: 1.0% failed SLO checks per rolling 28d window.
- Freeze conditions:
  1. determinism pass rate < 99.0%
  2. contract health < 99.5%
  3. perf_smoke_p95_seconds budget breach on default branch
  4. security scan freshness exceeds 168 hours
- During freeze: only reliability/security fixes merge; feature changes blocked until two consecutive green runs of violated gates.

## Evidence commands

```bash
python scripts/validate_ops_slo.py docs/ops/SLA_SLO.md
python scripts/check_perf_budget.py --baseline quality/perf_baseline.json --budgets quality/perf_budgets.yml
```
