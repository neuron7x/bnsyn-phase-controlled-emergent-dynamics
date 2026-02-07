# CI + Battle Usage Verification Audit

Date (as-of): 2026-02-07 Europe/Zaporozhye  
Target: https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics

## FACTS

1. PR-gate workflows detected from `.github/workflows/*.yml`:
   - `.github/workflows/ci-pr-atomic.yml`
   - `.github/workflows/workflow-integrity.yml`
2. GitHub Actions run evidence (main branch, last 10 sampled) is exported in:
   - `artifacts/audit/workflow_226681253_runs.tsv`
   - `artifacts/audit/workflow_229502046_runs.tsv`
   Both sampled windows show `completed/success` entries only.
3. Local gate reproduction artifacts are in `artifacts/ci_local/`:
   - `ruff_format.log` => pass (`312 files already formatted`)
   - `mypy_strict.log` => pass (`Success: no issues found in 69 source files`)
   - determinism pytest and SSOT/security command logs saved with exit codes in `*.exit`.
4. Requested snapshot `/mnt/data/bnsyn-phase-controlled-emergent-dynamics-main.zip` is absent in runtime; verification executed against `/workspace/bnsyn-phase-controlled-emergent-dynamics`.
5. Formal non-usage declaration now exists at `docs/STATUS.md` with explicit statement:  
   `This project is research-grade / pre-production. No battle usage claimed.`
6. CI anti-overclaim guard added via `python -m scripts.validate_status_claims` in PR-gate workflow.

## INFERENCES

1. Local quality-gate divergence recorded in the previous audit (`ruff format`, `mypy --strict`) is closed in current working state.
2. Battle-usage ambiguity is closed through formalized non-usage policy (D2 path), not through external adoption proof.
3. PR-branch convergence to GitHub CI cannot be proven from this runtime because no PR-triggered run URL for the current head can be created here.

## GAP STATUS

See `GAP_TABLE.md`.
- Closed: GAP-001, GAP-002, GAP-004.
- Still blocking without external action: GAP-003, GAP-005.

## FINAL STATUS (fail-closed)

- `CI_EXECUTABILITY_STATUS = PARTIAL` (historical main runs proven green; current PR-head run URL missing).
- `BATTLE_USAGE_STATUS = WEAK` (formal non-usage declared and enforced; no external adoption evidence claimed).
- `READYNESS_PERCENT = 55` using strict fail-closed scoring:
  - Start 100
  - CI status != PROVEN_GREEN: -25
  - No snapshot-zip local reproduction proof: -20
  - Battle usage not PROVEN: -0 (formal non-usage path selected)

