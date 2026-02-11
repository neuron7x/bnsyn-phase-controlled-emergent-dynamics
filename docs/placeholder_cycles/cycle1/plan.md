# CYCLE 1-6 — PH Closure Execution Plan

## Scope Compression (minimum viable scope)

Execution scope is constrained to PH files and direct validation dependencies:

1. PH runtime/test targets:
   - `src/bnsyn/emergence/crystallizer.py` (PH-0001)
   - `tests/test_coverage_gate.py` (PH-0002)
   - `tests/validation/test_chaos_integration.py` (PH-0003)
2. PH registry/meta-validation dependencies:
   - `docs/PLACEHOLDER_REGISTRY.md`
   - `tests/test_scan_placeholders.py`
   - `scripts/scan_placeholders.py`

No adjacent modules are modified.

## CYCLE 2 — Architecture (Interfaces + Seams)

Dependency seams and contracts:

- Numeric backend seam: `np.linalg.svd` failure path in crystallizer (`LinAlgError`) must fail-closed by retaining previous deterministic PCA state.
- File I/O seam: coverage XML read path must raise explicit `FileNotFoundError` for missing artifacts.
- Validation seam: AdEx chaos test accepts only explicit safe rejection (`ValueError`) or finite-valued result states.
- Registry seam: placeholder registry must remain machine-parseable and validate schema/uniqueness/status contracts.

Registry validation rules:

- PH schema fields: `ID`, `Path`, `Signature`, `Risk`, `Owner`, `Fix Strategy`, `Test Strategy`, `Verification Test`, `Status`, optional `Evidence Ref`.
- ID uniqueness: all PH IDs globally unique.
- Status contract:
  - `OPEN` → requires non-empty `Fix Strategy` and `Test Strategy`.
  - `CLOSED` → requires non-empty `Evidence Ref`.

## PH_BATCHES

### PH_BATCH_01 — Runtime-critical emergence path
- PH: `PH-0001`
- fix_strategy: `guard_fail_closed`
- test_strategy: `regression`

### PH_BATCH_02 — Test harness placeholder guards
- PH: `PH-0002`, `PH-0003`
- fix_strategy: `implement_minimal` / `guard_fail_closed`
- test_strategy: `regression`

## CYCLE 3-6 Execution Gates

- CYCLE 3 (Implement): remove/replace placeholder behavior minimally, keep deterministic contracts.
- CYCLE 4 (Tests): run per-PH regression tests + registry/meta-scan tests.
- CYCLE 5 (Reviewer): block merge on OPEN statuses, missing PH test, missing evidence, extra scope.
- CYCLE 6 (QA): enforce CI-green, zero placeholder findings, valid registry, evidence bundle linkage.

## Exit Criteria

- Every PH entry is `CLOSED` with `Evidence Ref`.
- Placeholder scan reports zero findings.
- Registry schema + uniqueness + status contract validations pass.
- CI-required checks are green.
