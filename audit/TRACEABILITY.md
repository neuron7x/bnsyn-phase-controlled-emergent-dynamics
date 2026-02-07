# Audit Traceability Matrix

| Metric | JSONPath | Source Script | Artifact Path |
|---|---|---|---|
| Inventory total files | `$.inventory.total_files` | `scripts/compute_repo_metrics.py` | `artifacts/math_audit/computed_metrics.json` |
| Category LOC totals | `$.categories.<CAT>.loc_code` | `scripts/compute_repo_metrics.py` | `artifacts/math_audit/computed_metrics.json` |
| Repo crosscheck inventory | `$.inventory_total_files` | `scripts/crosscheck_repo_metrics.py` | `artifacts/math_audit/crosscheck_metrics.json` |
| Workflow count | `$.workflow_count` | `scripts/compute_ci_cd_metrics.py` | `artifacts/math_audit/ci_cd_metrics.json` |
| Workflow crosscheck | `$.workflow_count` | `scripts/crosscheck_ci_cd_metrics.py` | `artifacts/math_audit/crosscheck_ci_cd_metrics.json` |
| Reconciliation booleans | `$.*` | `scripts/run_audit.sh` | `artifacts/math_audit/reconciliation_report.json` |
| ODT claim hashes | `$.sources[*].sha256` | `scripts/extract_odt_claims.py` | `audit_inputs/odt_sources/*.sha256` |
| Replacement model costs | `$.replacement_cost_model.cost_usd` | `scripts/valuation_model.py` | `artifacts/math_audit/valuation_results.json` |
| Market-rate verification status | `$.market_rate_model.status` | `scripts/valuation_model.py` | `artifacts/math_audit/valuation_results.json` |
| Final gate status | `(text)` | `scripts/run_audit.sh` | `artifacts/math_audit/final_status.txt` |
