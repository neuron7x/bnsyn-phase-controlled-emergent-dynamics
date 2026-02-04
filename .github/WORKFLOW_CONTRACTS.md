# CI/CD Workflow Contracts

**Version:** 1.1
**Date (UTC):** 2026-02-03
**Repository:** neuron7x/bnsyn-phase-controlled-emergent-dynamics
**Total workflows:** 25
**Breakdown:** 17 primary + 8 reusable

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
* Branch protection rules are defined in **PR-Gate Definition (Authoritative)**.
* This contract is normative even if current branch protection temporarily deviates.

## PR-Gate Definition (Authoritative)

* Required PR-gates MUST be: `ci-pr-atomic.yml`, `workflow-integrity.yml`.
* Required checks MUST NOT include any workflow other than the PR-gates listed above.
* If any other section conflicts with this, this section wins.

## Gate Class Policy (Normative)

* PR-gate workflows MAY be required checks.
* Long-running workflows MUST NOT be required checks and MUST NOT block merge.
* Manual-only workflows MUST NOT be required checks.
* If a workflow is not explicitly labeled PR-gate, it MUST NOT be required in branch protection.

## Workflow Inventory Index

* Count: 25 workflows
* Files (lexicographic):
  * _reusable_benchmarks.yml
  * _reusable_chaos_tests.yml
  * _reusable_formal_science.yml
  * _reusable_property_tests.yml
  * _reusable_pytest.yml
  * _reusable_quality.yml
  * _reusable_ssot.yml
  * _reusable_validation_tests.yml
  * benchmarks.yml
  * ci-benchmarks-elite.yml
  * ci-benchmarks.yml
  * ci-pr-atomic.yml
  * ci-pr.yml
  * ci-smoke.yml
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

**Gate Class:** Manual-only

**Gate Rationale:**

* Reusable workflow invoked by callers; gating is defined by the parent workflow, so it is not a standalone PR gate despite enforcing A2.

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

**Gate Class:** Manual-only

**Gate Rationale:**

* Reusable lint/type checks are enforced by caller workflows; this template supports A6/A2 but is not a direct PR gate.

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

## _reusable_chaos_tests.yml

**Path:** `.github/workflows/_reusable_chaos_tests.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* Chaos suite is invoked by scheduled or dispatch workflows; it supports A5/A2 without being a direct PR gate.

**Intent (1–2 sentences):**

* Provide a reusable chaos test matrix job with fault-type fan-out and artifact capture.

**Axiom focus:**

* A5 — Chaos / Robustness / Fault Injection
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `test-subset`, `timeout-minutes`, `upload-artifacts`).

**Timeout(s):**

* `chaos-tests`: `${{ inputs['timeout-minutes'] }}`

**Jobs:**

* `chaos-tests` — Runs chaos tests per fault type with summaries and optional artifacts.

**Evidence:**

* `./workflows/_reusable_chaos_tests.yml`

---

## _reusable_property_tests.yml

**Path:** `.github/workflows/_reusable_property_tests.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* Property-based tests are executed by caller workflows, so this reusable job is not a standalone PR gate even though it enforces A2.

**Intent (1–2 sentences):**

* Provide a reusable property-based test job with optional Hypothesis profiles and artifact capture.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `markers`, `extra-args`, `hypothesis-profile`, `log-file`, `junit-file`, `junit-enabled`, `summary-title`, `upload-artifacts`, `upload-hypothesis-cache`, `artifact-name`, `timeout-minutes`).

**Timeout(s):**

* `property-tests`: `${{ inputs['timeout-minutes'] }}`

**Jobs:**

* `property-tests` — Runs property tests, writes summaries, and uploads artifacts if configured.

**Evidence:**

* `./workflows/_reusable_property_tests.yml`

---

## _reusable_ssot.yml

**Path:** `.github/workflows/_reusable_ssot.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* SSOT checks are consumed by caller workflows; this reusable job supports A2 but has no independent PR-gate status.

**Intent (1–2 sentences):**

* Provide reusable SSOT validation gates for bibliography, claims, and governance scans.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`).

**Timeout(s):**

* `ssot`: Not set

**Jobs:**

* `ssot` — Runs SSOT validation scripts.

**Evidence:**

* `./workflows/_reusable_ssot.yml`

---

## _reusable_validation_tests.yml

**Path:** `.github/workflows/_reusable_validation_tests.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* Validation tests are driven by parent workflows; this reusable job enforces A2 but is not a PR gate on its own.

**Intent (1–2 sentences):**

* Provide a reusable validation test job with optional artifacts and summaries.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `markers`, `extra-args`, `log-file`, `junit-file`, `junit-enabled`, `summary-title`, `upload-artifacts`, `artifact-name`, `timeout-minutes`).

**Timeout(s):**

* `validation-tests`: `${{ inputs['timeout-minutes'] }}`

**Jobs:**

* `validation-tests` — Runs validation tests, writes summaries, and uploads artifacts if configured.

**Evidence:**

* `./workflows/_reusable_validation_tests.yml`

---

## _reusable_benchmarks.yml

**Path:** `.github/workflows/_reusable_benchmarks.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* Benchmark execution is orchestrated by caller workflows; this reusable job supports A4/A2 but does not create a standalone PR gate.

**Intent (1–2 sentences):**

* Execute benchmark suites based on tier/profile inputs and upload performance artifacts for regression tracking.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`tier`, `profile`, `scenario`, `publish_baseline`) and optional secrets (`BENCHMARK_GPG_PASSPHRASE`, `SLACK_WEBHOOK_URL`).

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

## _reusable_formal_science.yml

**Path:** `.github/workflows/_reusable_formal_science.yml`
**Status:** Active

**Gate Class:** Manual-only

**Gate Rationale:**

* Formal/science tasks are invoked by scheduled or dispatch workflows; this template supports A7/A8/A2 without being a direct PR gate.

**Intent (1–2 sentences):**

* Provide a reusable formal/science job template that executes Coq proofs, TLA+ model checks, or flagship experiments based on explicit inputs.

**Axiom focus:**

* A7 — Formal Verification – Proof Assistants (e.g., Coq)
* A8 — Formal Verification – Temporal Logic / Model Checking (e.g., TLA)
* A2 — CI Correctness & Regression Safety

**Owner:**

* @neuron7x

**Trigger(s):**

* `workflow_call` with inputs (`proof-suite`, `model-set`, `time-budget`, `max-steps`).

**Timeout(s):**

* `formal-science`: `${{ inputs.time-budget }}`

**Jobs:**

* `formal-science` — Runs Coq, TLA+, or science experiment workflows based on the suite input.

**Evidence:**

* `./workflows/_reusable_formal_science.yml`

---

## benchmarks.yml

**Path:** `.github/workflows/benchmarks.yml`
**Status:** Deprecated shim (delegates to `ci-benchmarks.yml`)

**Gate Class:** Long-running

**Gate Rationale:**

* Non-blocking benchmark dispatch preserves A4 performance tracking without gating PR merges.

**Intent (1–2 sentences):**

* Preserve legacy entrypoints and schedules while delegating execution to `ci-benchmarks.yml`.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `workflow_dispatch` with tier/profile/scenario inputs.

**Timeout(s):**

* `benchmarks-dispatch`: Not set

**Jobs:**

* `benchmarks-dispatch` — Delegates workflow_dispatch inputs to `ci-benchmarks.yml`.

**Evidence:**

* `./workflows/benchmarks.yml`

---

## ci-benchmarks-elite.yml

**Path:** `.github/workflows/ci-benchmarks-elite.yml`
**Status:** Deprecated shim (delegates to `ci-benchmarks.yml`)

**Gate Class:** Long-running

**Gate Rationale:**

* Elite benchmark schedule preserves A4 fidelity tracking while remaining non-blocking for PR merges.

**Intent (1–2 sentences):**

* Preserve the legacy elite benchmark schedule while delegating execution to `ci-benchmarks.yml`.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

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

**Gate Class:** Long-running

**Gate Rationale:**

* Benchmark routing provides A4 performance regression visibility as a non-blocking long-running check.

**Intent (1–2 sentences):**

* Canonical benchmark entrypoint that routes tier/profile inputs to the reusable benchmark executor.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `pull_request`.
* `schedule` (daily 02:00 UTC, weekly Sunday 03:00 UTC).
* `workflow_dispatch`.
* `workflow_call` with inputs (`tier`, `profile`, `scenario`, `publish_baseline`) and secrets `BENCHMARK_GPG_PASSPHRASE`, `SLACK_WEBHOOK_URL`.

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

**Gate Class:** PR-gate

**Gate Rationale:**

* Required PR gate enforcing A1/A2/A9 determinism, correctness, and security before merge.

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

**Gate Class:** Manual-only

**Gate Rationale:**

* Thin wrapper for dispatch/call usage; gating is handled by `ci-pr-atomic.yml`, so this is not a direct PR gate.

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

## ci-smoke.yml

**Path:** `.github/workflows/ci-smoke.yml`
**Status:** Candidate for consolidation

**Gate Class:** Long-running

**Gate Rationale:**

* Provides optional A2 smoke/SSOT signal but remains non-blocking while `ci-pr-atomic.yml` serves as the required PR gate.

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

## ci-validation.yml

**Path:** `.github/workflows/ci-validation.yml`
**Status:** Active

**Gate Class:** Long-running

**Gate Rationale:**

* Scheduled validation and chaos suites provide non-blocking A2 regression coverage without gating PR merges.

**Intent (1–2 sentences):**

* Orchestrate scheduled and manual validation modes (standard/elite/chaos) with SSOT, validation, property, and chaos suites.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_dispatch` with `mode` and `chaos_subset` inputs.
* `schedule` (weekly Sunday 03:00 UTC, daily 02:00 UTC, daily 02:30 UTC, daily 04:00 UTC).

**Timeout(s):**

* `ssot`: Not set
* `validation-tests`: 30
* `property-tests`: 30
* `chaos-tests`: 60

**Jobs:**

* `determine` — Resolves scheduled/manual mode inputs.
* `ssot` — Runs SSOT checks (standard mode).
* `validation-tests` — Runs validation tests (standard/elite).
* `property-tests` — Runs property tests (elite/chaos).
* `chaos-tests` — Runs chaos matrix tests (chaos).
* `elite-summary` — Aggregates validation + property results (elite).
* `chaos-summary` — Aggregates chaos + property results and fails on suite failure (chaos).

**Evidence:**

* `./workflows/ci-validation.yml`

---

## codecov-health.yml

**Path:** `.github/workflows/codecov-health.yml`
**Status:** Active

**Gate Class:** Long-running

**Gate Rationale:**

* Periodic Codecov health checks support A3 provenance without blocking PR merges.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Scheduled/push CodeQL analysis supports A9 security hygiene as a non-blocking long-running check.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Weekly dependency monitoring supports A9 security posture without gating PR merges.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Documentation builds provide A2 correctness feedback but remain non-blocking to avoid gating PR merges.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Nightly formal proof checks uphold A7 assurance as non-blocking long-running validation.

**Intent (1–2 sentences):**

* Compile and verify Coq proofs for formal specifications via the reusable formal/science template.

**Axiom focus:**

* A7 — Formal Verification – Proof Assistants (e.g., Coq)

**Owner:**

* @neuron7x

**Trigger(s):**

* `schedule` (nightly 01:00 UTC).
* `workflow_dispatch`.

**Timeout(s):**

* `coq-proof-check`: 20

**Jobs:**

* `coq-proof-check` — Installs Coq and compiles proof files with artifacted outputs.

**Delegation & provenance:**

* Delegates execution to `_reusable_formal_science.yml` with `proof-suite=coq`, `model-set=specs/coq`, `time-budget=20`.
* Critical outputs: `coq-proof-verification-${{ github.sha }}` artifact (`coq_output_*.txt`, `coq_summary.md`) and the job summary lines for compilation status.

**Evidence:**

* `./workflows/formal-coq.yml`

---

## formal-tla.yml

**Path:** `.github/workflows/formal-tla.yml`
**Status:** Active

**Gate Class:** Long-running

**Gate Rationale:**

* Nightly TLA+ model checks uphold A8 assurance as non-blocking long-running validation.

**Intent (1–2 sentences):**

* Run nightly TLA+ model checking with configurable MaxSteps via the reusable formal/science template.

**Axiom focus:**

* A8 — Formal Verification – Temporal Logic / Model Checking (e.g., TLA)

**Owner:**

* @neuron7x

**Trigger(s):**

* `schedule` (nightly 02:00 UTC).
* `workflow_dispatch` with `max-steps` input.

**Timeout(s):**

* `tla-model-check`: 30

**Jobs:**

* `tla-model-check` — Downloads TLA+ tools, generates config, runs TLC, and uploads reports.

**Delegation & provenance:**

* Delegates execution to `_reusable_formal_science.yml` with `proof-suite=tla`, `model-set=specs/tla`, `time-budget=30`, and `max-steps` defaulting to `100` on schedules.
* Critical outputs: `tla-model-check-report-${{ github.sha }}` artifact (`tlc_output.txt`, `tla_summary.md`, `BNsyn_runtime.cfg`) and the job summary lines containing model checking status.

**Evidence:**

* `./workflows/formal-tla.yml`

---

## physics-equivalence.yml

**Path:** `.github/workflows/physics-equivalence.yml`
**Status:** Active

**Gate Class:** Long-running

**Gate Rationale:**

* Physics equivalence checks enforce A2 correctness but remain non-blocking to keep PR gates focused on `ci-pr-atomic.yml`.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Mutation testing supports A6 quality posture as a non-blocking long-running check.

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

**Gate Class:** Long-running

**Gate Rationale:**

* Scheduled flagship experiments support A2 scientific validation without gating PR merges.

**Intent (1–2 sentences):**

* Execute a flagship experiment workflow to validate scientific hypotheses with artifacts via the reusable formal/science template.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Owner:**

* @neuron7x

**Trigger(s):**

* `workflow_dispatch`.
* `schedule` (weekly Sunday 00:00 UTC).

**Timeout(s):**

* `flagship-experiment`: 30

**Jobs:**

* `flagship-experiment` — Runs experiment, visualizes results, verifies hypothesis, uploads artifacts.

**Delegation & provenance:**

* Delegates execution to `_reusable_formal_science.yml` with `proof-suite=science`, `model-set=temp_ablation_v2`, `time-budget=30`.
* Critical outputs: `experiment-results` and `experiment-figures` artifacts plus summary manifest in the job summary.

**Evidence:**

* `./workflows/science.yml`

---

## workflow-integrity.yml

**Path:** `.github/workflows/workflow-integrity.yml`
**Status:** Active

**Gate Class:** PR-gate

**Gate Rationale:**

* Required PR gate enforcing A3/A9 workflow integrity and permissions hygiene before merge.

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
   * Removal criteria: REQUIRES branch protection to require only PR-gate workflows (`ci-pr-atomic.yml`, `workflow-integrity.yml`) and remove `ci-smoke.yml`; REQUIRES `ci-pr-atomic.yml` smoke + SSOT checks to be equivalent for at least 2 weeks; REQUIRES no downstream integrations to rely on `ci-smoke.yml` status checks.
   * Risks: Branch protection or external status check dependencies could block merges; mitigate by updating protection rules and notifying integrators.
   * Evidence: `./workflows/ci-smoke.yml`, `./workflows/ci-pr-atomic.yml`

2. **ci-validation.yml (mode orchestration)** (Consolidation completed)
   * Rationale: Validation, property, and chaos suites are orchestrated via a single workflow with mode routing.
   * Current state: `ci-validation.yml` routes standard/elite/chaos schedules and dispatch inputs to reusable jobs.
   * Follow-up criteria: MUST ensure branch protection requires only PR-gate workflows (`ci-pr-atomic.yml`, `workflow-integrity.yml`) and not `ci-validation`; SHOULD ensure scheduled cadence matches prior weekly/daily runs.
   * Evidence: `./workflows/ci-validation.yml`, `./workflows/_reusable_ssot.yml`, `./workflows/_reusable_validation_tests.yml`, `./workflows/_reusable_property_tests.yml`, `./workflows/_reusable_chaos_tests.yml`

3. **ci-validation.yml (property mode)** (Consolidation completed)
   * Rationale: Property tests are now scheduled and dispatched via `ci-validation.yml` with a dedicated `property` mode.
   * Current state: `ci-validation.yml` owns the 02:30 UTC property schedule and `workflow_dispatch` supports `mode=property`.
   * Follow-up criteria: SHOULD ensure property mode cadence and profile alignment with `ci` expectations.
   * Evidence: `./workflows/ci-validation.yml`, `./workflows/_reusable_property_tests.yml`

4. **benchmarks.yml vs ci-benchmarks-elite.yml vs ci-benchmarks.yml** (Consolidation in progress)
   * Rationale: Benchmark execution is centralized in `_reusable_benchmarks.yml`, with `ci-benchmarks.yml` as the canonical entrypoint.
   * Current state: `benchmarks.yml` and `ci-benchmarks-elite.yml` are compatibility shims delegating to `ci-benchmarks.yml`.
   * Removal criteria: REQUIRES branch protection to require only PR-gate workflows (`ci-pr-atomic.yml`, `workflow-integrity.yml`) and downstream dependencies updated; REQUIRES legacy schedules to be retired or migrated without duplicate runs.
   * Risks: Duplicate benchmark runs while shims remain; mitigate by removing shims after protection verification.
   * Evidence: `./workflows/_reusable_benchmarks.yml`, `./workflows/benchmarks.yml`, `./workflows/ci-benchmarks-elite.yml`, `./workflows/ci-benchmarks.yml`

---

## Completeness & Consistency

* Inventory count: 25
* Contract blocks count: 25
* Missing: []
* Mandatory named workflows present: YES (ci-validation.yml, ci-benchmarks-elite.yml, workflow-integrity.yml, quality-mutation.yml, formal-coq.yml, formal-tla.yml)
