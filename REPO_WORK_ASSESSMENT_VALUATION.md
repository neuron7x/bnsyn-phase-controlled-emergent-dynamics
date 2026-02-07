# Repository Work Assessment & Valuation Report

## STATUS
**NOT VERIFIED ❌**

## Executive Summary
- Inventory files: **616**.
- CI/CD: **28 workflows**, **51 jobs**, **269 steps**.
- Reconciliation checks passing: **9/9**.
- Market-rate model: **NOT VERIFIED**.

## Bounded Assumptions Table (Replacement Model)
| Parameter | Min | Base | Max |
|---|---:|---:|---:|
| Hours per LOC | 0.011765 | 0.016667 | 0.025000 |
| USD hourly rate | 60 | 90 | 140 |

## Valuation Outputs
- Replacement hours (min/base/max): 828.72 / 1174.02 / 1761.03.
- Replacement cost USD (min/base/max): $49723.20 / $105661.80 / $246544.20.
- Market-rate model status: NOT VERIFIED.
- Market-rate reason: One or more configured sources missing extracted values.

## Deterministic Reproduce
`./scripts/run_audit.sh`

## NEEDS_EVIDENCE
- Add repo-local ODT files under `audit_inputs/` to produce non-empty `audit_inputs/claims.json`.
- Ensure all `audit_inputs/sources.yml` extractions return non-null values for market-rate verification.
