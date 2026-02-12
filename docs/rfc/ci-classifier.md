# RFC: CI Change Classifier Hardening

## Status
Accepted

## Scope
`ci-pr-atomic` changed-file classification and merge-gate dependency mapping.

## Invariants
- Classifier is fail-closed: unknown paths set `unknown_changed=true` and force non-docs routing.
- PR context source of truth is GitHub PR files API.
- Non-PR context uses deterministic `git diff --name-only HEAD~1...HEAD`.
- Classifier emits deterministic evidence lines:
  - `pr_number`, `base_sha`, `head_sha` for PR events
  - `changed_files_count`, `changed_files_sha256` for all events

## Security Controls
- API timeout and bounded retries.
- Retryable HTTP statuses: `429`, `500`, `502`, `503`, `504`.
- Explicit User-Agent.
- Authorization only via `GITHUB_TOKEN`; token is never printed.

## Merge-Gate Contract
- `finalize` is always-run, fail-closed, and validates required upstream job results.
- docs-only path is permitted only when no sensitive flags are asserted.

## Validation Guide
- `python -m scripts.validate_pr_gates`
- `python -m scripts.validate_required_status_contexts`
- `python -m scripts.validate_workflow_contracts`
- `python -m pytest tests/test_validate_pr_gates.py tests/test_schema_contracts.py tests/test_classify_changes.py -q`

## Risk Notes
- GitHub API outage can fail classification (intentional fail-closed behavior).
- Non-PR fallback depends on `HEAD~1` availability in checkout history.
