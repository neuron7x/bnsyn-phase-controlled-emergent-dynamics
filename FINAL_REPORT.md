# FINAL_REPORT

## CI_EXECUTABILITY_STATUS
NEEDS_EVIDENCE

- Historical workflow evidence exists in:
  - `artifacts/audit/workflow_226681253_runs.tsv`
  - `artifacts/audit/workflow_229502046_runs.tsv`
- Current HEAD CI evidence query:
  - `artifacts/audit/runs_for_head.json` (result: `total_count = 0`)
- Required human action to close:
  1. Push current branch to GitHub.
  2. Wait for successful runs of `ci-pr-atomic` and `workflow-integrity`.
  3. Record immutable run URLs below.

### REQUIRED RUN URLS (current PR head)
- ci-pr-atomic: NEEDS_EVIDENCE
- workflow-integrity: NEEDS_EVIDENCE

## BATTLE_USAGE_STATUS
FORMALIZED_NON_USAGE

- Repository declares pre-production status in `docs/STATUS.md`.
- PR-gate workflow enforces anti-overclaim check via `scripts.validate_status_claims`.

## READYNESS_PERCENT
75

Rationale (fail-closed):
- Start 100
- -25: missing immutable CI run proof for current PR head

## Local Evidence
- `artifacts/ci_local/pip_install.log`
- `artifacts/ci_local/ruff_format.log`
- `artifacts/ci_local/ruff_check.log`
- `artifacts/ci_local/mypy_strict.log`
- `artifacts/ci_local/pytest_q.log`
- `artifacts/ci_local/validate_status_claims.log`
- `artifacts/ci_local/manifest_generate.log`
- `artifacts/ci_local/manifest_validate.log`
- `artifacts/ci_local/summary.tsv` (all exit codes == 0)
