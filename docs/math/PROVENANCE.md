# Data Provenance — BN-Syn Math Audit

## Source-of-Truth Policy
- Canonical audit metrics: `artifacts/math_audit/computed_metrics.json`
- Independent cross-check metrics: `artifacts/math_audit/crosscheck_metrics.json`
- Invariant gate result: `artifacts/math_audit/reconciliation_report.json`
- CI/CD workflow analytics: `artifacts/math_audit/ci_cd_metrics.json`
- Valuation model I/O: `artifacts/math_audit/valuation_inputs.json`, `artifacts/math_audit/valuation_results.json`

## Reproduction Command (single source)
- `bash scripts/run_audit.sh`

## Artifacts Folder Provenance
| Artifact | Source of Truth | Reproduction Command |
|---|---|---|
| `artifacts/math_audit/computed_metrics.json` | `scripts/compute_repo_metrics.py` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/crosscheck_metrics.json` | `scripts/crosscheck_metrics.py` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/reconciliation_report.json` | invariant block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/ci_cd_metrics.json` | `scripts/ci_cd_analyzer.py` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/claims.json` | ODT parser block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/valuation_inputs.json` | `scripts/valuation_model.py` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/valuation_results.json` | `scripts/valuation_model.py` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/core_tests_sha256.json` | checksum block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/inputs_manifest.json` | manifest block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/toolchain_versions.json` | toolchain capture block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/environment_facts.json` | environment capture block in `scripts/run_audit.sh` | `bash scripts/run_audit.sh` |
| `artifacts/math_audit/raw/*` | command logs emitted by scripts/commands | `bash scripts/run_audit.sh` |

## Legacy files
Legacy artifacts in `artifacts/math_audit/` not listed above are non-canonical historical outputs and are not used by the final work-volume model.
