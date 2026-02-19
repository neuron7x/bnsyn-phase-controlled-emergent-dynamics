# Branch Protection Governance Baseline (`main`)

## Mandatory repository settings

1. [ ] Enable Branch protection for `main`.
2. [ ] Disable direct push to `main` (PR-only merge path).
3. [ ] Enable **Require a pull request before merging**.
4. [ ] Enable **Require status checks to pass before merging**.
5. [ ] Add required check: `ci-pr-atomic`.
6. [ ] Add required check: `math-quality-gate`.
7. [ ] Add required check: `workflow-integrity`.
8. [ ] Add required check: `dependency-review`.
9. [ ] Add required check: `scientific-product-gate`.
10. [ ] Add required check: `pr-gate`.
11. [ ] Enable (or explicitly document) **Require branches to be up to date before merging**.
12. [ ] Enable (or explicitly document) **Dismiss stale approvals**.

## Required status contexts (job-level + matrix-level)

All contexts below MUST be green before merge:

- `ci-pr-atomic / gate-profile`
- `ci-pr-atomic / determinism`
- `ci-pr-atomic / quality`
- `ci-pr-atomic / build`
- `ci-pr-atomic / smoke-wheel-matrix (py3.11)`
- `ci-pr-atomic / smoke-wheel-matrix (py3.12)`
- `ci-pr-atomic / tests-smoke`
- `ci-pr-atomic / ssot`
- `ci-pr-atomic / security`
- `ci-pr-atomic / finalize`
- `Workflow Integrity / Validate workflow files`
- `Math Quality Gate / math-gate`
- `dependency-review / dependency-review`
- `scientific-product-gate / scientific-product`
- `pr-gate / pr-gate`

## Governance-proof evidence (blocking semantics)

| Requirement | Evidence URL (Checks tab / settings) | Timestamp (UTC) | Owner |
| --- | --- | --- | --- |
| Required checks block merge | PENDING | PENDING | PENDING |
| Matrix contexts enforced separately | PENDING | PENDING | PENDING |
| Branch protection has PR-only merge | PENDING | PENDING | PENDING |
| Dismiss stale approvals policy | PENDING | PENDING | PENDING |
| Up-to-date branch policy decision | PENDING | PENDING | PENDING |

## Control PR protocol (negative testing)

- [ ] **Control PR #1:** break property invariant → merge blocked.
- [ ] **Control PR #2:** break validation invariant → merge blocked.
- [ ] **Control PR #3:** break docs build → merge blocked.
- [ ] **Control PR #4:** introduce risky dependency diff → dependency-review blocks merge.
- [ ] **Control PR #5:** break install-from-wheel smoke → merge blocked.

### Control PR evidence log

| Control PR | PR URL | Expected failing check(s) | Observed failing check(s) | Merge blocked? | Evidence URL |
| --- | --- | --- | --- | --- | --- |
| #1 property invariant | PENDING | `ci-pr-atomic / property-tests-pr` | PENDING | PENDING | PENDING |
| #2 validation invariant | PENDING | `ci-pr-atomic / validation-tests-pr` | PENDING | PENDING | PENDING |
| #3 docs build | PENDING | `ci-pr-atomic / docs-pr` | PENDING | PENDING | PENDING |
| #4 risky dependency | PENDING | `dependency-review / dependency-review` | PENDING | PENDING | PENDING |
| #5 wheel smoke | PENDING | `ci-pr-atomic / smoke-wheel-matrix (py3.11)`, `ci-pr-atomic / smoke-wheel-matrix (py3.12)` | PENDING | PENDING | PENDING |
