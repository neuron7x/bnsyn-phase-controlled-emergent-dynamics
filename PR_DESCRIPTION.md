**What changed**

* Consolidated validation orchestration into `.github/workflows/ci-validation.yml` with `mode` routing (standard/elite/chaos/property), added 02:30 UTC property schedule, and removed the redundant `ci-property-tests.yml` schedule owner.
* Enforced deterministic pip pinning via the composite action `.github/actions/pin-pip` using a single `PIP_VERSION` source, plus post-pin logging in workflows that install dependencies.
* Made artifact uploads best-effort with `if-no-files-found: ignore` where artifacts are optional.
* Inventory (workflows + actions; touched = yes/no):
  * `.github/workflows/_reusable_benchmarks.yml` — reusable benchmark suites (touched: yes)
  * `.github/workflows/_reusable_chaos_tests.yml` — chaos test suite (touched: yes)
  * `.github/workflows/_reusable_property_tests.yml` — property test suite (touched: yes)
  * `.github/workflows/_reusable_pytest.yml` — reusable pytest suite (touched: yes)
  * `.github/workflows/_reusable_quality.yml` — reusable lint/type checks (touched: yes)
  * `.github/workflows/_reusable_ssot.yml` — SSOT gates (touched: yes)
  * `.github/workflows/_reusable_validation_tests.yml` — validation tests (touched: yes)
  * `.github/workflows/benchmarks.yml` — benchmarks shim (touched: no)
  * `.github/workflows/ci-benchmarks-elite.yml` — elite benchmarks shim (touched: no)
  * `.github/workflows/ci-benchmarks.yml` — benchmarks entrypoint (touched: no)
  * `.github/workflows/ci-pr-atomic.yml` — PR gate orchestration (touched: yes)
  * `.github/workflows/ci-pr.yml` — PR gate wrapper (touched: no)
  * `.github/workflows/ci-smoke.yml` — smoke tests (touched: yes)
  * `.github/workflows/ci-validation.yml` — validation orchestrator (touched: yes)
  * `.github/workflows/codecov-health.yml` — Codecov health check (touched: no)
  * `.github/workflows/codeql.yml` — CodeQL scan (touched: no)
  * `.github/workflows/dependency-watch.yml` — dependency watch (touched: yes)
  * `.github/workflows/docs.yml` — docs build (touched: yes)
  * `.github/workflows/formal-coq.yml` — Coq proofs (touched: yes)
  * `.github/workflows/formal-tla.yml` — TLA+ checks (touched: yes)
  * `.github/workflows/physics-equivalence.yml` — physics equivalence (touched: yes)
  * `.github/workflows/quality-mutation.yml` — mutation testing (touched: yes)
  * `.github/workflows/science.yml` — flagship experiment (touched: yes)
  * `.github/workflows/workflow-integrity.yml` — workflow integrity checks (touched: yes)
  * `.github/actions/pin-pip/action.yml` — pip pin composite action (touched: yes)

**Why**

* Ensure a single authoritative validation orchestration entrypoint without duplicate schedules.
* Enforce deterministic pip toolchain behavior with a single pinned version and verifiable logging.
* Prevent optional artifact uploads from failing CI.

**Evidence**

* Run URL: BLOCKED — no GitHub workflow dispatch executed in this environment (merge must be blocked until run URLs are provided).
* Logs: BLOCKED — no PR-branch run logs available; pip pin log must be captured from a PR-branch run.
* Lint proof:
  * `actionlint` (pass): executed locally.
  * `yamllint .github/workflows` (fail): existing repository YAML style violations (line-length, document-start, trailing-spaces) require repository-wide lint policy decision.

**Compatibility note**

* Python >= 3.9 → pip 26.0 pinned via `.github/actions/pin-pip` (single source of truth).

---

## Test hardening upgrade (determinism + entropy)

**Upgraded**
- Added deterministic test topology inventory generator: `scripts/generate_tests_inventory.py` producing `tests_inventory.json`.
- Added principal-level invariant tests in `tests/test_principal_test_hardening.py`:
  - `TestEntropyMonotonicity`
  - `TestDeterministicExecutionChain`
  - `TestWorkflowPinningIntegrity`
  - `TestArtifactProvenanceInvariant`
- Hardened CI by adding explicit `timeout-minutes` to control-plane jobs in `.github/workflows/ci-validation.yml`.

**Removed**
- No test deletions in this change set.

**Invariants added**
- Determinism offender counters cannot regress versus entropy baseline.
- Inventory generation hash remains stable across repeated runs.
- External workflow actions must be full SHA-pinned.
- Manifest provenance digest must remain SHA-addressed with non-empty invariant set.

**Entropy reduction rationale**
- Explicit timeouts reduce hanging workflow entropy and enforce fail-closed CI behavior.
- Deterministic inventory generation and pinning checks reduce nondeterministic drift in quality signals.
