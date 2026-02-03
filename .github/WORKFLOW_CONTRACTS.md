# CI/CD Workflow Contracts

**Version:** 1.1
**Date (UTC):** 2026-02-03
**Repository:** neuron7x/bnsyn-phase-controlled-emergent-dynamics
**Total workflows:** 23
**Breakdown:** 20 primary + 3 reusable

## Axiom Dictionary

* **A1 — Determinism & Toolchain Pinning**
* **A2 — CI Correctness & Regression Safety**
* **A3 — Workflow Integrity & Provenance (branch/sha correctness, tamper resistance)**
* **A4 — Performance & Benchmark Fidelity**
* **A5 — Chaos / Robustness / Fault Injection**
* **A6 — Quality / Mutation / Adversarial Testing**
* **A7 — Formal Verification – Proof Assistants (e.g., Coq)**
* **A8 — Formal Verification – Temporal Logic / Model Checking (e.g., TLA)**
* **A9 — Security & Permissions Hygiene (least privilege, untrusted code boundaries)**

## Migration Notes (v1.1)

* Expanded axiom dictionary from prior legend to include A1–A9 with explicit names; all uses in this document map to the dictionary above.

## Global Notes

* Workflow-level timeouts are not supported in GitHub Actions; only job-level `timeout-minutes` are recorded.
* Timeout notation: use **Not set** when `timeout-minutes` is absent in the workflow YAML.

## Workflow Inventory Index

* Count: 23 workflows
* Files (lexicographic):
  * _reusable_benchmarks.yml
  * _reusable_pytest.yml
  * _reusable_quality.yml
  * benchmarks.yml
  * chaos-validation.yml
  * ci-benchmarks-elite.yml
  * ci-benchmarks.yml
  * ci-pr-atomic.yml
  * ci-pr.yml
  * ci-property-tests.yml
  * ci-smoke.yml
  * ci-validation-elite.yml
  * ci-validation.yml
  * codecov-health.yml
  * codeql.yml
  * dependency-watch.yml
  * docs.yml
  * formal-coq.yml
  * formal-tla.yml
  * physics-equivalence.yml
  * quality-mutation.yml
  * science.yml
  * workflow-integrity.yml

---

## _reusable_pytest.yml

**Path:** `.github/workflows/_reusable_pytest.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide a reusable pytest job with coverage enforcement and optional Codecov uploads for caller workflows.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `markers`, `coverage-threshold`, `timeout-minutes`, `upload-codecov`) and optional secret `CODECOV_TOKEN`.

**Timeout(s):**

* `pytest`: `${{ inputs.timeout-minutes }}`

**Jobs:**

* `pytest` — Runs pytest with coverage thresholds, generates summaries, and uploads artifacts/Codecov if configured.

**Evidence:**

* `./workflows/_reusable_pytest.yml`

---

## _reusable_quality.yml

**Path:** `.github/workflows/_reusable_quality.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide reusable lint/type/quality checks (ruff, mypy, pylint) for CI callers.

**Axiom focus:**

* A6 — Quality / Mutation / Adversarial Testing
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `mypy-strict`, `pylint-threshold`).

**Timeout(s):**

* `ruff`: Not set
* `mypy`: Not set
* `pylint`: Not set

**Jobs:**

* `ruff` — Format and lint checks.
* `mypy` — Type checking (strict optional).
* `pylint` — Enforces pylint score threshold.

**Evidence:**

* `./workflows/_reusable_quality.yml`

---

## _reusable_benchmarks.yml

**Path:** `.github/workflows/_reusable_benchmarks.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute benchmark suites based on tier/profile inputs and upload performance artifacts for regression tracking.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`tier`, `profile`, `scenario`) and optional secrets (`BENCHMARK_GPG_PASSPHRASE`, `SLACK_WEBHOOK_URL`).

**Timeout(s):**

* `micro-benchmarks`: 10
* `elite-benchmarks`: 30

**Jobs:**

* `micro-benchmarks` — PR/dispatch micro benchmarks with regression checks and artifacts.
* `nightly-benchmarks` — Baseline suite with dependency audit and artifact publication.
* `scenario-benchmarks` — Scenario runner with report generation.
* `elite-benchmarks` — Full benchmark suite with baseline comparison.

**Evidence:**

* `./workflows/_reusable_benchmarks.yml`

---

## benchmarks.yml

**Path:** `.github/workflows/benchmarks.yml`
**Status:** Deprecated shim (delegates to `ci-benchmarks.yml`)

**Intent (1–2 sentences):**

* Preserve legacy entrypoints and schedules while delegating execution to `ci-benchmarks.yml`.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `workflow_dispatch` with tier/profile/scenario inputs.
* `pull_request`.
* `schedule` (daily standard baseline, weekly scenario, weekly elite).

**Timeout(s):**

* `benchmarks-pr`: Not set
* `benchmarks-standard`: Not set
* `benchmarks-weekly-scenario`: Not set
* `benchmarks-elite`: Not set

**Jobs:**

* `benchmarks-pr` — Delegates micro benchmarks to `ci-benchmarks.yml`.
* `benchmarks-standard` — Delegates daily baseline to `ci-benchmarks.yml`.
* `benchmarks-weekly-scenario` — Delegates weekly scenario to `ci-benchmarks.yml`.
* `benchmarks-elite` — Delegates elite baseline to `ci-benchmarks.yml`.

**Evidence:**

* `./workflows/benchmarks.yml`

---

## chaos-validation.yml

**Path:** `.github/workflows/chaos-validation.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute chaos fault-injection tests and extended property tests to validate resilience and invariants under stress.

**Axiom focus:**

* A5 — Chaos / Robustness / Fault Injection
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (nightly 04:00 UTC).
* `workflow_dispatch` with `test_subset` input.

**Timeout(s):**

* `chaos-tests`: 60
* `property-tests`: 45
* `summary`: Not set

**Jobs:**

* `chaos-tests` — Matrixed chaos tests per fault type.
* `property-tests` — Hypothesis property tests with statistics.
* `summary` — Aggregates outcomes and fails on suite failure.

**Evidence:**

* `./workflows/chaos-validation.yml`

---

## ci-benchmarks-elite.yml

**Path:** `.github/workflows/ci-benchmarks-elite.yml`
**Status:** Deprecated shim (delegates to `ci-benchmarks.yml`)

**Intent (1–2 sentences):**

* Preserve the legacy elite benchmark schedule while delegating execution to `ci-benchmarks.yml`.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `schedule` (weekly Sunday 03:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `benchmarks`: Not set (delegated to reusable workflow)

**Jobs:**

* `benchmarks` — Delegates elite baseline to `ci-benchmarks.yml`.

**Evidence:**

* `./workflows/ci-benchmarks-elite.yml`

---

## ci-benchmarks.yml

**Path:** `.github/workflows/ci-benchmarks.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Canonical benchmark entrypoint that routes tier/profile inputs to the reusable benchmark executor.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `pull_request`.
* `schedule` (daily 02:00 UTC, weekly Sunday 03:00 UTC).
* `workflow_dispatch`.
* `workflow_call` with secrets `BENCHMARK_GPG_PASSPHRASE`, `SLACK_WEBHOOK_URL`.

**Timeout(s):**

* `benchmarks-pr`: Not set
* `benchmarks-standard`: Not set
* `benchmarks-elite`: Not set

**Jobs:**

* `benchmarks-pr` — Delegates micro benchmarks to `_reusable_benchmarks.yml`.
* `benchmarks-dispatch` — Delegates user-selected tier/profile to `_reusable_benchmarks.yml`.
* `benchmarks-call` — Delegates workflow callers to `_reusable_benchmarks.yml`.
* `benchmarks-standard` — Delegates daily baseline to `_reusable_benchmarks.yml`.
* `benchmarks-elite` — Delegates elite baseline to `_reusable_benchmarks.yml`.

**Evidence:**

* `./workflows/ci-benchmarks.yml`

---

## ci-pr-atomic.yml

**Path:** `.github/workflows/ci-pr-atomic.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide an atomic PR gate that enforces determinism, quality, build integrity, tests, SSOT, and security checks.

**Axiom focus:**

* A1 — Determinism & Toolchain Pinning
* A2 — CI Correctness & Regression Safety
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `pull_request`.
* `push` (branches: `main`).
* `workflow_dispatch`.
* `workflow_call`.

**Timeout(s):**

* `determinism`: Not set
* `quality`: Not set
* `build`: Not set
* `tests-smoke`: 10
* `ssot`: Not set
* `security`: Not set
* `finalize`: Not set

**Jobs:**

* `determinism` — Runs determinism tests 3x and checks RNG isolation.
* `quality` — Reuses quality workflow for lint/type checks.
* `build` — Builds package and verifies import.
* `tests-smoke` — Runs smoke tests via reusable pytest workflow.
* `ssot` — Validates bibliography/claims/governed docs/normative tags.
* `security` — Runs gitleaks, pip-audit, bandit and uploads report.
* `finalize` — Emits pass gate message when all checks succeed.

**Evidence:**

* `./workflows/ci-pr-atomic.yml`

---

## ci-pr.yml

**Path:** `.github/workflows/ci-pr.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide a thin wrapper entry point that delegates to `ci-pr-atomic.yml`.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `workflow_dispatch`.
* `workflow_call`.

**Jobs:**

* `ci-pr-atomic` — Delegates to the canonical `ci-pr-atomic.yml` workflow.

**Evidence:**

* `./workflows/ci-pr.yml`

---

## ci-property-tests.yml

**Path:** `.github/workflows/ci-property-tests.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run nightly property-based tests using Hypothesis for invariant checking.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (nightly 02:30 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `property-tests`: 15

**Jobs:**

* `property-tests` — Runs Hypothesis property tests and uploads artifacts.

**Evidence:**

* `./workflows/ci-property-tests.yml`

---

## ci-smoke.yml

**Path:** `.github/workflows/ci-smoke.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Provide quick SSOT and smoke-test feedback for PRs and main branch changes.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `push` (branches: `main`).
* `pull_request`.

**Timeout(s):**

* `ssot`: Not set
* `tests-smoke`: Not set

**Jobs:**

* `ssot` — Validates bibliography/claims/governed docs/normative tags.
* `tests-smoke` — Runs smoke tests without validation/property markers.

**Evidence:**

* `./workflows/ci-smoke.yml`

---

## ci-validation-elite.yml

**Path:** `.github/workflows/ci-validation-elite.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute daily scientific validation and property tests with artifacted summaries.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (daily 02:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `validation`: 30
* `property-tests`: 30
* `summary`: Not set

**Jobs:**

* `validation` — Runs validation test suite with logs and artifacts.
* `property-tests` — Runs Hypothesis property tests with logs and artifacts.
* `summary` — Generates aggregate summary and notes non-blocking status.

**Evidence:**

* `./workflows/ci-validation-elite.yml`

---

## ci-validation.yml

**Path:** `.github/workflows/ci-validation.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run weekly SSOT checks and validation tests on schedule or manual dispatch.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_dispatch`.
* `schedule` (weekly Sunday 03:00 UTC).

**Timeout(s):**

* `ssot`: Not set
* `tests-validation`: Not set

**Jobs:**

* `ssot` — Validates bibliography/claims/governed docs/normative tags.
* `tests-validation` — Runs validation tests.

**Evidence:**

* `./workflows/ci-validation.yml`

---

## codecov-health.yml

**Path:** `.github/workflows/codecov-health.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Monitor Codecov availability and token configuration to keep coverage reporting functional.

**Axiom focus:**

* A3 — Workflow Integrity & Provenance

**Trigger(s):**

* `schedule` (every 6 hours).
* `workflow_dispatch`.

**Timeout(s):**

* `health`: Not set

**Jobs:**

* `health` — Checks Codecov API responsiveness and secret configuration.

**Evidence:**

* `./workflows/codecov-health.yml`

---

## codeql.yml

**Path:** `.github/workflows/codeql.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Perform CodeQL security analysis on main branch and weekly schedule.

**Axiom focus:**

* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `push` (branches: `main`).
* `schedule` (weekly Sunday 04:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `analyze`: 15

**Jobs:**

* `analyze` — Runs CodeQL init/analyze for Python.

**Evidence:**

* `./workflows/codeql.yml`

---

## dependency-watch.yml

**Path:** `.github/workflows/dependency-watch.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Detect outdated dependencies and security advisories on a weekly cadence.

**Axiom focus:**

* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `schedule` (weekly Monday 08:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `check-outdated`: Not set

**Jobs:**

* `check-outdated` — Compares dependency lock output and runs pip-audit advisories.

**Evidence:**

* `./workflows/dependency-watch.yml`

---

## docs.yml

**Path:** `.github/workflows/docs.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Build documentation for PR verification and manual checks.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `pull_request`.
* `workflow_dispatch`.

**Timeout(s):**

* `build-docs`: Not set

**Jobs:**

* `build-docs` — Builds docs and uploads HTML artifacts.

**Evidence:**

* `./workflows/docs.yml`

---

## formal-coq.yml

**Path:** `.github/workflows/formal-coq.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Compile and verify Coq proofs for formal specifications.

**Axiom focus:**

* A7 — Formal Verification – Proof Assistants (e.g., Coq)

**Trigger(s):**

* `schedule` (nightly 01:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `coq-proof-check`: 20

**Jobs:**

* `coq-proof-check` — Installs Coq and compiles proof files with artifacted outputs.

**Evidence:**

* `./workflows/formal-coq.yml`

---

## formal-tla.yml

**Path:** `.github/workflows/formal-tla.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Run nightly TLA+ model checking with configurable MaxSteps and artifacted reports.

**Axiom focus:**

* A8 — Formal Verification – Temporal Logic / Model Checking (e.g., TLA)

**Trigger(s):**

* `schedule` (nightly 02:00 UTC).
* `workflow_dispatch` with `max_steps` input.

**Timeout(s):**

* `tla-model-check`: 30

**Jobs:**

* `tla-model-check` — Downloads TLA+ tools, generates config, runs TLC, and uploads reports.

**Evidence:**

* `./workflows/formal-tla.yml`

---

## physics-equivalence.yml

**Path:** `.github/workflows/physics-equivalence.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Validate physics-preserving transformations between reference and accelerated backends.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `pull_request` with path filters.
* `push` (branches: `main`).
* `workflow_dispatch`.

**Timeout(s):**

* `validate-physics`: 15

**Jobs:**

* `validate-physics` — Runs reference/accelerated benchmarks, verifies equivalence, and uploads artifacts.

**Evidence:**

* `./workflows/physics-equivalence.yml`

---

## quality-mutation.yml

**Path:** `.github/workflows/quality-mutation.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Run mutation testing against critical modules and enforce mutation score baseline.

**Axiom focus:**

* A6 — Quality / Mutation / Adversarial Testing

**Trigger(s):**

* `schedule` (nightly 03:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `mutation-testing`: 120

**Jobs:**

* `mutation-testing` — Runs mutmut, computes score, enforces baseline, uploads artifacts.

**Evidence:**

* `./workflows/quality-mutation.yml`

---

## science.yml

**Path:** `.github/workflows/science.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute a flagship experiment workflow to validate scientific hypotheses with artifacts.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_dispatch`.
* `schedule` (weekly Sunday 00:00 UTC).

**Timeout(s):**

* `flagship-experiment`: 30

**Jobs:**

* `flagship-experiment` — Runs experiment, visualizes results, verifies hypothesis, uploads artifacts.

**Evidence:**

* `./workflows/science.yml`

---

## workflow-integrity.yml

**Path:** `.github/workflows/workflow-integrity.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Lint and validate workflow files for integrity, encoding safety, and safety artifacts.

**Axiom focus:**

* A3 — Workflow Integrity & Provenance
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `pull_request` (branches: `main`).
* `push` (branches: `main`).

**Timeout(s):**

* `validate-workflows`: 5

**Jobs:**

* `validate-workflows` — Runs actionlint, scans for encoding violations, and validates safety artifacts.

**Evidence:**

* `./workflows/workflow-integrity.yml`

---

## Redundancy & Consolidation

1. **ci-smoke.yml vs ci-pr-atomic.yml** (Candidate for consolidation)
   * Rationale: `ci-smoke.yml` and `ci-pr-atomic.yml` share `push` (main) and `pull_request` triggers, and `ci-smoke` job set (`ssot`, `tests-smoke`) overlaps with `ci-pr-atomic` which already runs SSOT and smoke tests via reusable workflow. `ci-pr.yml` is now a wrapper and no longer a target for consolidation.
   * Safe target: `ci-pr-atomic.yml` as SSOT.
   * Removal criteria:
    * [ ] Branch protection updated to remove `ci-smoke.yml` requirements.
    * [ ] `ci-pr-atomic.yml` smoke + SSOT checks verified equivalent for at least 2 weeks.
    * [ ] No downstream integrations rely on `ci-smoke.yml` status checks.
   * Risks: Branch protection or external status check dependencies could block merges; mitigate by updating protection rules and notifying integrators.
   * Evidence: `./workflows/ci-smoke.yml`, `./workflows/ci-pr-atomic.yml`

2. **ci-validation.yml vs ci-validation-elite.yml** (Candidate for consolidation)
   * Rationale: Both use `schedule` + `workflow_dispatch` and run validation suites; `ci-validation-elite` includes validation + property tests, while `ci-validation` runs SSOT + validation.
   * Safe target: `ci-validation-elite.yml` as SSOT after adding SSOT coverage or calling a reusable SSOT job.
   * Removal criteria:
     * [ ] SSOT checks added to `ci-validation-elite.yml` or provided via reusable job.
     * [ ] Parity validation run confirms equivalent validation outputs.
     * [ ] Weekly cadence requirements documented or rescheduled in `ci-validation-elite.yml`.
   * Risks: Losing weekly-only reporting or SSOT visibility; mitigate by maintaining schedule parity or adding reporting artifacts.
   * Evidence: `./workflows/ci-validation.yml`, `./workflows/ci-validation-elite.yml`

3. **ci-property-tests.yml vs chaos-validation.yml** (Candidate for consolidation)
   * Rationale: Both run `schedule` + `workflow_dispatch` and execute property tests; `chaos-validation` already includes a `property-tests` job.
   * Safe target: `chaos-validation.yml` as SSOT for property tests.
   * Removal criteria:
     * [ ] Hypothesis profile parity confirmed (`ci` vs `thorough`).
     * [ ] Runtime/cadence impact documented and accepted.
     * [ ] Branch protection or required checks updated if needed.
   * Risks: Profile differences may change coverage or runtime; mitigate by aligning profiles before removal.
   * Evidence: `./workflows/ci-property-tests.yml`, `./workflows/chaos-validation.yml`

4. **benchmarks.yml vs ci-benchmarks-elite.yml vs ci-benchmarks.yml** (Consolidation in progress)
   * Rationale: Benchmark execution is centralized in `_reusable_benchmarks.yml`, with `ci-benchmarks.yml` as the canonical entrypoint.
   * Current state: `benchmarks.yml` and `ci-benchmarks-elite.yml` are compatibility shims delegating to `ci-benchmarks.yml`.
   * Removal criteria:
     * [ ] Branch protection and downstream dependencies updated.
     * [ ] Legacy schedules retired or migrated without duplicate runs.
   * Risks: Duplicate benchmark runs while shims remain; mitigate by removing shims after protection verification.
   * Evidence: `./workflows/_reusable_benchmarks.yml`, `./workflows/benchmarks.yml`, `./workflows/ci-benchmarks-elite.yml`, `./workflows/ci-benchmarks.yml`

---

## Completeness & Consistency

* Inventory count: 23
* Contract blocks count: 23
* Missing: []
* Mandatory named workflows present: YES (chaos-validation.yml, ci-validation-elite.yml, ci-benchmarks-elite.yml, workflow-integrity.yml, quality-mutation.yml, formal-coq.yml, formal-tla.yml)
