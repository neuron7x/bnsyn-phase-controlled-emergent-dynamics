# Repository Work Assessment & Valuation Report

## STATUS
**NOT VERIFIED ❌**

## 1. Repository Overview
- HEAD: 715aa014afad29da67f40a4f19fb43d4f32d1a1c
- Branch: work
- Total commits: 587
- Active days (UTC commit dates): 16

## 2. Artifact Inventory
| Category | Files | LOC (code) | LOC (comments) | LOC (blank) |
|---|---:|---:|---:|---:|
| CORE_LOGIC | 68 | 7159 | 985 | 1852 |
| TESTS | 148 | 11625 | 555 | 3140 |
| CI_CD | 29 | 2607 | 100 | 301 |
| INFRASTRUCTURE | 3 | 148 | 0 | 47 |
| DOCUMENTATION | 94 | 6459 | 2676 | 3099 |
| SCRIPTS_TOOLING | 123 | 12464 | 477 | 2359 |
| DATA_SCHEMAS | 4 | 315 | 6 | 84 |
| CONFIGURATION | 42 | 3364 | 1867 | 656 |
| STATIC_ASSETS | 8 | 25414 | 316 | 2074 |
| GENERATED | 114 | 111402 | 113 | 309 |
| **TOTAL COUNTABLE** | **519** | **69555** | **6982** | **13612** |

Reconciliation: partition=True, disjoint=True, loc_sum=True, no_hidden=True.

## 3. Quality Assessment
- Complexity: grade A (3.5330188679245285).
- Tests: files=148, functions=639, negative_asserts=216, property_markers=83.
- CI/CD workflows: 28 files, 51 jobs, 269 steps.

## 4. Effort Estimation
- Replacement hours model: low=958.26, base=1197.82, high=1497.28.
- Market-rate model: UNKNOWN_NEEDS_EVIDENCE.

## 5. Monetary Valuation
- Replacement cost (assumption): low=$57495.6, base=$107803.8, high=$209619.2.
- External contributions: UNKNOWN (evidence-gated).

## 6. Metric Traceability
| Metric | JSONPath | Source | Artifact |
|---|---|---|---|
| inventory_total_files | $.inventory.total_files | scripts/compute_repo_metrics.py | artifacts/math_audit/computed_metrics.json |
| ci_cd.workflow_count | $.workflow_count | scripts/ci_cd_analyzer.py | artifacts/math_audit/ci_cd_metrics.json |
| countable_loc_code | $.totals.countable_loc_code | scripts/compute_repo_metrics.py | artifacts/math_audit/computed_metrics.json |
| reconciliation.I-G | $.I-G CI/CD nonzero-if-present | scripts/run_audit.sh | artifacts/math_audit/reconciliation_report.json |
| valuation.base_cost | $.replacement_cost_usd_assumption.base | scripts/valuation_model.py | artifacts/math_audit/valuation_results.json |

## 7. Reproduce
`bash scripts/run_audit.sh && .venv_audit/bin/python -m pytest -q tests/test_audit_rules.py && .venv_audit/bin/python - <<'PY'
import json;from jsonschema import validate;from pathlib import Path;validate(json.loads(Path('audit_report_data.json').read_text()),json.loads(Path('schemas/audit_report_data.schema.json').read_text()));print('SCHEMA_OK')
PY`

## 8. Evidence Index
- `artifacts/math_audit/toolchain_versions.json`
- `artifacts/math_audit/environment_facts.json`
- `artifacts/math_audit/inputs_manifest.json`
- `artifacts/math_audit/claims.json`
- `artifacts/math_audit/computed_metrics.json`
- `artifacts/math_audit/crosscheck_metrics.json`
- `artifacts/math_audit/diff_report.json`
- `artifacts/math_audit/reconciliation_report.json`
- `artifacts/math_audit/valuation_inputs.json`
- `artifacts/math_audit/valuation_results.json`
- `artifacts/math_audit/ci_cd_metrics.json`

## 9. NEEDS_EVIDENCE
- Missing `/mnt/data/codex pr.odt`; provide file to verify ODT claims extraction.
- Market-rate sources snapshot not present; provide auditable rate sources + timestamp/FX snapshot.
