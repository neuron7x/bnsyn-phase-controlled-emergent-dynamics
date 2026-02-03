# CI/CD Workflow Contracts

## Workflow Inventory Index

* Count: 22 workflows
* Files (lexicographic):
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

## `_reusable_pytest.yml`

**Path:** `.github/workflows/_reusable_pytest.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide a reusable pytest job with coverage enforcement and optional Codecov uploads for caller workflows.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `markers`, `coverage-threshold`, `timeout-minutes`, `upload-codecov`) and optional secret `CODECOV_TOKEN`. `file:.github/workflows/_reusable_pytest.yml:L3-L29`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `pytest`: `${{ inputs.timeout-minutes }}`. `file:.github/workflows/_reusable_pytest.yml:L31-L36`

**Jobs:**

* `pytest` — Runs pytest with coverage thresholds, generates summaries, and uploads artifacts/Codecov if configured. `file:.github/workflows/_reusable_pytest.yml:L31-L198`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `pytest` with coverage, summary generation, `codecov/codecov-action@v5`, `actions/upload-artifact@v4`. `file:.github/workflows/_reusable_pytest.yml:L37-L191`
  * Notes: timeout is controlled by input; coverage artifacts uploaded on success/failure. `file:.github/workflows/_reusable_pytest.yml:L35-L195`

---

## `_reusable_quality.yml`

**Path:** `.github/workflows/_reusable_quality.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide reusable lint/type/quality checks (ruff, mypy, pylint) for CI callers.

**Axiom focus:**

* A6 — Quality / Mutation / Adversarial Testing
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_call` with inputs (`python-version`, `mypy-strict`, `pylint-threshold`). `file:.github/workflows/_reusable_quality.yml:L3-L17`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `ruff`: UNSET. `file:.github/workflows/_reusable_quality.yml:L19-L92`
  * `mypy`: UNSET. `file:.github/workflows/_reusable_quality.yml:L94-L161`
  * `pylint`: UNSET. `file:.github/workflows/_reusable_quality.yml:L163-L220`

**Jobs:**

* `ruff` — Enforces formatting and lint rules with summary and artifacts on failure. `file:.github/workflows/_reusable_quality.yml:L19-L92`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `ruff format`, `ruff check`, `actions/upload-artifact@v4`. `file:.github/workflows/_reusable_quality.yml:L24-L89`
* `mypy` — Runs mypy in strict or non-strict mode with summary and logs. `file:.github/workflows/_reusable_quality.yml:L94-L161`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `mypy`, `actions/upload-artifact@v4`. `file:.github/workflows/_reusable_quality.yml:L98-L157`
* `pylint` — Enforces pylint score threshold with summary and logs. `file:.github/workflows/_reusable_quality.yml:L163-L220`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `pylint`, `actions/upload-artifact@v4`. `file:.github/workflows/_reusable_quality.yml:L167-L217`

---

## `benchmarks.yml`

**Path:** `.github/workflows/benchmarks.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run benchmark scenarios on a weekly schedule or on-demand with scenario selection.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `workflow_dispatch` with `scenario` input (choice). `file:.github/workflows/benchmarks.yml:L3-L18`
* `schedule` (weekly, Monday 00:00 UTC). `file:.github/workflows/benchmarks.yml:L18-L20`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `benchmark`: UNSET. `file:.github/workflows/benchmarks.yml:L25-L82`

**Jobs:**

* `benchmark` — Runs selected benchmark scenario, generates report, uploads artifacts. `file:.github/workflows/benchmarks.yml:L25-L82`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `benchmarks/run_benchmarks.py`, `benchmarks/report.py`, `actions/upload-artifact@v4`. `file:.github/workflows/benchmarks.yml:L30-L69`

---

## `chaos-validation.yml`

**Path:** `.github/workflows/chaos-validation.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute chaos fault-injection tests and extended property tests to validate resilience and invariants under stress.

**Axiom focus:**

* A5 — Chaos / Robustness / Fault Injection
* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (nightly 04:00 UTC). `file:.github/workflows/chaos-validation.yml:L3-L6`
* `workflow_dispatch` with `test_subset` input. `file:.github/workflows/chaos-validation.yml:L7-L18`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `chaos-tests`: 60. `file:.github/workflows/chaos-validation.yml:L27-L35`
  * `property-tests`: 45. `file:.github/workflows/chaos-validation.yml:L82-L85`
  * `summary`: UNSET. `file:.github/workflows/chaos-validation.yml:L142-L177`

**Jobs:**

* `chaos-tests` — Runs matrixed chaos tests for fault types with summary and artifacts. `file:.github/workflows/chaos-validation.yml:L28-L81`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `pytest tests/validation/test_chaos_*`, `actions/upload-artifact@v4`. `file:.github/workflows/chaos-validation.yml:L37-L80`
  * Notes: matrix over fault types; fail-fast false. `file:.github/workflows/chaos-validation.yml:L32-L35`
* `property-tests` — Runs Hypothesis property tests with statistics and artifacts. `file:.github/workflows/chaos-validation.yml:L82-L140`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `pytest -m property`, `actions/upload-artifact@v4`. `file:.github/workflows/chaos-validation.yml:L86-L139`
* `summary` — Aggregates chaos/property outcomes and fails if either suite failed. `file:.github/workflows/chaos-validation.yml:L142-L184`
  * Key steps: summary generation, status check. `file:.github/workflows/chaos-validation.yml:L151-L183`

---

## `ci-benchmarks-elite.yml`

**Path:** `.github/workflows/ci-benchmarks-elite.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run a weekly benchmark suite with baseline comparison for performance regression awareness.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `schedule` (weekly Sunday 03:00 UTC). `file:.github/workflows/ci-benchmarks-elite.yml:L3-L7`
* `workflow_dispatch`. `file:.github/workflows/ci-benchmarks-elite.yml:L7-L9`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `benchmarks`: 30. `file:.github/workflows/ci-benchmarks-elite.yml:L17-L21`

**Jobs:**

* `benchmarks` — Runs multiple benchmark scripts, compares against baseline, uploads reports. `file:.github/workflows/ci-benchmarks-elite.yml:L17-L97`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `bench_*.py`, `scripts/compare_benchmarks.py`, `actions/upload-artifact@v4`. `file:.github/workflows/ci-benchmarks-elite.yml:L23-L82`
  * Notes: concurrency configured to allow completion. `file:.github/workflows/ci-benchmarks-elite.yml:L13-L16`

---

## `ci-benchmarks.yml`

**Path:** `.github/workflows/ci-benchmarks.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Run scheduled benchmark baselines and on-demand micro-benchmarks for regression monitoring.

**Axiom focus:**

* A4 — Performance & Benchmark Fidelity

**Trigger(s):**

* `pull_request`. `file:.github/workflows/ci-benchmarks.yml:L3-L5`
* `schedule` (daily 02:00 UTC). `file:.github/workflows/ci-benchmarks.yml:L5-L7`
* `workflow_dispatch`. `file:.github/workflows/ci-benchmarks.yml:L7-L8`
* `workflow_call` with secrets `BENCHMARK_GPG_PASSPHRASE`, `SLACK_WEBHOOK_URL`. `file:.github/workflows/ci-benchmarks.yml:L8-L13`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `nightly-benchmarks`: UNSET. `file:.github/workflows/ci-benchmarks.yml:L26-L112`
  * `micro-benchmarks`: 10. `file:.github/workflows/ci-benchmarks.yml:L113-L166`

**Jobs:**

* `nightly-benchmarks` — Scheduled baseline benchmarks with artifact publication and optional Slack notification. `file:.github/workflows/ci-benchmarks.yml:L26-L112`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `benchmarks/run_benchmarks.py`, `gh release upload`, `actions/upload-artifact@v4`, `slackapi/slack-github-action@v2.1.1`. `file:.github/workflows/ci-benchmarks.yml:L41-L112`
  * Notes: uses secrets for GPG/Slack; permissions include contents write. `file:.github/workflows/ci-benchmarks.yml:L15-L23`
* `micro-benchmarks` — PR/dispatch benchmarks with regression checks and artifact upload. `file:.github/workflows/ci-benchmarks.yml:L113-L166`
  * Key steps: `scripts/run_benchmarks.py`, `scripts/check_benchmark_regressions.py`, `actions/upload-artifact@v4`. `file:.github/workflows/ci-benchmarks.yml:L139-L165`

**Notes / Risk flags (optional but recommended):**

* Elevated permissions (contents: write, security-events: write) and release publishing in scheduled job. `file:.github/workflows/ci-benchmarks.yml:L15-L23`

---

## `ci-pr-atomic.yml`

**Path:** `.github/workflows/ci-pr-atomic.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide an atomic PR gate that enforces determinism, quality, build integrity, tests, SSOT, and security checks.

**Axiom focus:**

* A1 — Determinism & Toolchain Pinning
* A2 — CI Correctness & Regression Safety
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `pull_request`. `file:.github/workflows/ci-pr-atomic.yml:L3-L5`
* `push` (branches: `main`). `file:.github/workflows/ci-pr-atomic.yml:L5-L6`
* `workflow_dispatch`. `file:.github/workflows/ci-pr-atomic.yml:L6-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `determinism`: UNSET. `file:.github/workflows/ci-pr-atomic.yml:L22-L87`
  * `quality`: UNSET (delegated to reusable workflow). `file:.github/workflows/ci-pr-atomic.yml:L88-L93`
  * `build`: UNSET. `file:.github/workflows/ci-pr-atomic.yml:L95-L111`
  * `tests-smoke`: 10 (via reusable input + reusable job timeout). `file:.github/workflows/ci-pr-atomic.yml:L112-L120` and `file:.github/workflows/_reusable_pytest.yml:L31-L36`
  * `ssot`: UNSET. `file:.github/workflows/ci-pr-atomic.yml:L123-L140`
  * `security`: UNSET. `file:.github/workflows/ci-pr-atomic.yml:L142-L175`
  * `finalize`: UNSET. `file:.github/workflows/ci-pr-atomic.yml:L176-L191`

**Jobs:**

* `determinism` — Runs determinism tests 3x and checks RNG isolation. `file:.github/workflows/ci-pr-atomic.yml:L22-L87`
  * Key steps: `pytest tests/test_determinism.py`, summary generation. `file:.github/workflows/ci-pr-atomic.yml:L36-L86`
* `quality` — Reuses quality workflow for lint/type checks. `file:.github/workflows/ci-pr-atomic.yml:L88-L93`
  * Key steps: `.github/workflows/_reusable_quality.yml`. `file:.github/workflows/ci-pr-atomic.yml:L88-L93`
* `build` — Builds package and verifies import. `file:.github/workflows/ci-pr-atomic.yml:L95-L111`
  * Key steps: `python -m build`, import check. `file:.github/workflows/ci-pr-atomic.yml:L108-L111`
* `tests-smoke` — Runs smoke tests via reusable pytest workflow with Codecov upload. `file:.github/workflows/ci-pr-atomic.yml:L112-L121`
  * Key steps: `.github/workflows/_reusable_pytest.yml` with coverage threshold 95. `file:.github/workflows/ci-pr-atomic.yml:L112-L121`
* `ssot` — Validates bibliography/claims/governed docs/normative tags. `file:.github/workflows/ci-pr-atomic.yml:L123-L140`
* `security` — Runs gitleaks, pip-audit, bandit and uploads report. `file:.github/workflows/ci-pr-atomic.yml:L142-L174`
* `finalize` — Emits pass gate message when all checks succeed. `file:.github/workflows/ci-pr-atomic.yml:L176-L191`

---

## `ci-pr.yml`

**Path:** `.github/workflows/ci-pr.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Provide comprehensive PR validation including SSOT governance, build, tests, benchmarks, and security scans.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `push` (branches: `main`). `file:.github/workflows/ci-pr.yml:L3-L6`
* `pull_request`. `file:.github/workflows/ci-pr.yml:L6-L7`
* `workflow_dispatch`. `file:.github/workflows/ci-pr.yml:L7-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `ssot`: UNSET. `file:.github/workflows/ci-pr.yml:L17-L146`
  * `dependency-consistency`: UNSET. `file:.github/workflows/ci-pr.yml:L147-L182`
  * `quality`: UNSET (delegated to reusable workflow). `file:.github/workflows/ci-pr.yml:L184-L189`
  * `manifest-verification`: UNSET. `file:.github/workflows/ci-pr.yml:L191-L211`
  * `build`: UNSET. `file:.github/workflows/ci-pr.yml:L212-L254`
  * `docs-build`: UNSET. `file:.github/workflows/ci-pr.yml:L255-L270`
  * `tests-smoke`: 10 (via reusable input + reusable job timeout). `file:.github/workflows/ci-pr.yml:L272-L279` and `file:.github/workflows/_reusable_pytest.yml:L31-L36`
  * `tests-core-only`: UNSET. `file:.github/workflows/ci-pr.yml:L281-L296`
  * `ci-benchmarks`: UNSET. `file:.github/workflows/ci-pr.yml:L298-L318`
  * `gitleaks`: UNSET. `file:.github/workflows/ci-pr.yml:L319-L334`
  * `pip-audit`: UNSET. `file:.github/workflows/ci-pr.yml:L335-L360`
  * `bandit`: UNSET. `file:.github/workflows/ci-pr.yml:L362-L377`

**Jobs:**

* `ssot` — Enforces SSOT, governance gates, and claims coverage checks. `file:.github/workflows/ci-pr.yml:L17-L146`
  * Key steps: `validate_bibliography.py`, `validate_claims.py`, `verify_formal_constants.py`, `lint_ci_truthfulness.py`, `validate_claims_coverage.py`. `file:.github/workflows/ci-pr.yml:L33-L115`
* `dependency-consistency` — Validates dependency SSOT and audits. `file:.github/workflows/ci-pr.yml:L147-L182`
  * Key steps: `validate-pyproject`, dry-run resolution, `pip-audit`. `file:.github/workflows/ci-pr.yml:L165-L176`
* `quality` — Reuses quality workflow for lint/type checks. `file:.github/workflows/ci-pr.yml:L184-L189`
* `manifest-verification` — Ensures API manifest inventories are consistent. `file:.github/workflows/ci-pr.yml:L191-L211`
* `build` — Builds package, verifies import, and summarizes. `file:.github/workflows/ci-pr.yml:L212-L254`
* `docs-build` — Builds Sphinx docs. `file:.github/workflows/ci-pr.yml:L255-L270`
* `tests-smoke` — Runs smoke tests via reusable pytest workflow. `file:.github/workflows/ci-pr.yml:L272-L279`
* `tests-core-only` — Runs core tests without visualization dependencies. `file:.github/workflows/ci-pr.yml:L281-L296`
* `ci-benchmarks` — Runs core benchmarks for determinism/scaling/criticality. `file:.github/workflows/ci-pr.yml:L298-L318`
* `gitleaks` — Scans for secrets. `file:.github/workflows/ci-pr.yml:L319-L334`
* `pip-audit` — Scans dependencies for vulnerabilities. `file:.github/workflows/ci-pr.yml:L335-L360`
* `bandit` — Runs Python security linter. `file:.github/workflows/ci-pr.yml:L362-L377`

---

## `ci-property-tests.yml`

**Path:** `.github/workflows/ci-property-tests.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run nightly property-based tests using Hypothesis for invariant checking.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (nightly 02:30 UTC). `file:.github/workflows/ci-property-tests.yml:L3-L7`
* `workflow_dispatch`. `file:.github/workflows/ci-property-tests.yml:L7-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `property-tests`: 15. `file:.github/workflows/ci-property-tests.yml:L9-L12`

**Jobs:**

* `property-tests` — Runs Hypothesis property tests and uploads artifacts. `file:.github/workflows/ci-property-tests.yml:L10-L43`
  * Key steps: `actions/checkout@v4`, `actions/setup-python@v5`, `pytest -m property`, `actions/upload-artifact@v4`. `file:.github/workflows/ci-property-tests.yml:L17-L43`

---

## `ci-smoke.yml`

**Path:** `.github/workflows/ci-smoke.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Provide quick SSOT and smoke-test feedback for PRs and main branch changes.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `push` (branches: `main`). `file:.github/workflows/ci-smoke.yml:L3-L6`
* `pull_request`. `file:.github/workflows/ci-smoke.yml:L6-L7`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `ssot`: UNSET. `file:.github/workflows/ci-smoke.yml:L11-L32`
  * `tests-smoke`: UNSET. `file:.github/workflows/ci-smoke.yml:L33-L48`

**Jobs:**

* `ssot` — Validates bibliography/claims/governed docs/normative tags. `file:.github/workflows/ci-smoke.yml:L12-L32`
* `tests-smoke` — Runs smoke tests without validation/property markers. `file:.github/workflows/ci-smoke.yml:L33-L48`

---

## `ci-validation-elite.yml`

**Path:** `.github/workflows/ci-validation-elite.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute daily scientific validation and property tests with artifacted summaries.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `schedule` (daily 02:00 UTC). `file:.github/workflows/ci-validation-elite.yml:L3-L7`
* `workflow_dispatch`. `file:.github/workflows/ci-validation-elite.yml:L7-L9`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `validation`: 30. `file:.github/workflows/ci-validation-elite.yml:L18-L64`
  * `property-tests`: 30. `file:.github/workflows/ci-validation-elite.yml:L65-L112`
  * `summary`: UNSET. `file:.github/workflows/ci-validation-elite.yml:L114-L133`

**Jobs:**

* `validation` — Runs validation test suite with logs and artifacts. `file:.github/workflows/ci-validation-elite.yml:L18-L64`
* `property-tests` — Runs Hypothesis property tests with logs and artifacts. `file:.github/workflows/ci-validation-elite.yml:L65-L112`
* `summary` — Generates aggregate summary and notes non-blocking status. `file:.github/workflows/ci-validation-elite.yml:L114-L133`

---

## `ci-validation.yml`

**Path:** `.github/workflows/ci-validation.yml`
**Status:** Candidate for consolidation

**Intent (1–2 sentences):**

* Run weekly SSOT checks and validation tests on schedule or manual dispatch.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_dispatch`. `file:.github/workflows/ci-validation.yml:L3-L6`
* `schedule` (weekly Sunday 03:00 UTC). `file:.github/workflows/ci-validation.yml:L5-L6`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `ssot`: UNSET. `file:.github/workflows/ci-validation.yml:L11-L32`
  * `tests-validation`: UNSET. `file:.github/workflows/ci-validation.yml:L33-L48`

**Jobs:**

* `ssot` — Validates bibliography/claims/governed docs/normative tags. `file:.github/workflows/ci-validation.yml:L12-L32`
* `tests-validation` — Runs validation tests. `file:.github/workflows/ci-validation.yml:L33-L48`

---

## `codecov-health.yml`

**Path:** `.github/workflows/codecov-health.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Monitor Codecov availability and token configuration to keep coverage reporting functional.

**Axiom focus:**

* A3 — Workflow Integrity & Provenance

**Trigger(s):**

* `schedule` (every 6 hours). `file:.github/workflows/codecov-health.yml:L3-L6`
* `workflow_dispatch`. `file:.github/workflows/codecov-health.yml:L5-L7`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `health`: UNSET. `file:.github/workflows/codecov-health.yml:L11-L31`

**Jobs:**

* `health` — Checks Codecov API responsiveness and secret configuration. `file:.github/workflows/codecov-health.yml:L11-L31`
  * Key steps: `curl` API check, token check. `file:.github/workflows/codecov-health.yml:L15-L30`

---

## `codeql.yml`

**Path:** `.github/workflows/codeql.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Perform CodeQL security analysis on main branch and weekly schedule.

**Axiom focus:**

* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `push` (branches: `main`). `file:.github/workflows/codeql.yml:L3-L6`
* `schedule` (weekly Sunday 04:00 UTC). `file:.github/workflows/codeql.yml:L6-L7`
* `workflow_dispatch`. `file:.github/workflows/codeql.yml:L7-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `analyze`: 15. `file:.github/workflows/codeql.yml:L20-L24`

**Jobs:**

* `analyze` — Runs CodeQL init/analyze for Python. `file:.github/workflows/codeql.yml:L20-L34`
  * Key steps: `github/codeql-action/init@v3`, `github/codeql-action/analyze@v3`. `file:.github/workflows/codeql.yml:L28-L34`

---

## `dependency-watch.yml`

**Path:** `.github/workflows/dependency-watch.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Detect outdated dependencies and security advisories on a weekly cadence.

**Axiom focus:**

* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `schedule` (weekly Monday 08:00 UTC). `file:.github/workflows/dependency-watch.yml:L3-L6`
* `workflow_dispatch`. `file:.github/workflows/dependency-watch.yml:L6-L7`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `check-outdated`: UNSET. `file:.github/workflows/dependency-watch.yml:L11-L42`

**Jobs:**

* `check-outdated` — Compares dependency lock output and runs pip-audit advisories. `file:.github/workflows/dependency-watch.yml:L11-L42`
  * Key steps: `pip-compile --upgrade`, `pip-audit`. `file:.github/workflows/dependency-watch.yml:L25-L42`

---

## `docs.yml`

**Path:** `.github/workflows/docs.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Build documentation for PR verification and manual checks.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `pull_request`. `file:.github/workflows/docs.yml:L3-L5`
* `workflow_dispatch`. `file:.github/workflows/docs.yml:L5-L6`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `build-docs`: UNSET. `file:.github/workflows/docs.yml:L10-L40`

**Jobs:**

* `build-docs` — Builds docs and uploads HTML artifacts. `file:.github/workflows/docs.yml:L11-L40`
  * Key steps: `make docs`, `actions/upload-artifact@v4`. `file:.github/workflows/docs.yml:L32-L39`

---

## `formal-coq.yml`

**Path:** `.github/workflows/formal-coq.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Compile and verify Coq proofs for formal specifications.

**Axiom focus:**

* A7 — Formal Verification – Proof Assistants (e.g., Coq)

**Trigger(s):**

* `schedule` (nightly 01:00 UTC). `file:.github/workflows/formal-coq.yml:L3-L7`
* `workflow_dispatch`. `file:.github/workflows/formal-coq.yml:L7-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `coq-proof-check`: 20. `file:.github/workflows/formal-coq.yml:L16-L20`

**Jobs:**

* `coq-proof-check` — Installs Coq and compiles proof files with artifacted outputs. `file:.github/workflows/formal-coq.yml:L17-L118`
  * Key steps: `ocaml/setup-ocaml@v3`, `opam install coq`, `coqc *.v`, `actions/upload-artifact@v4`. `file:.github/workflows/formal-coq.yml:L25-L110`

---

## `formal-tla.yml`

**Path:** `.github/workflows/formal-tla.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Run nightly TLA+ model checking with configurable MaxSteps and artifacted reports.

**Axiom focus:**

* A8 — Formal Verification – Temporal Logic / Model Checking (e.g., TLA)

**Trigger(s):**

* `schedule` (nightly 02:00 UTC). `file:.github/workflows/formal-tla.yml:L3-L7`
* `workflow_dispatch` with `max_steps` input. `file:.github/workflows/formal-tla.yml:L7-L13`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `tla-model-check`: 30. `file:.github/workflows/formal-tla.yml:L21-L24`

**Jobs:**

* `tla-model-check` — Downloads TLA+ tools, generates config, runs TLC, and uploads reports. `file:.github/workflows/formal-tla.yml:L21-L229`
  * Key steps: `actions/setup-java@v5`, download/sha verify, `tla2sany.SANY`, TLC run, `actions/upload-artifact@v4`. `file:.github/workflows/formal-tla.yml:L34-L208`

---

## `physics-equivalence.yml`

**Path:** `.github/workflows/physics-equivalence.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Validate physics-preserving transformations between reference and accelerated backends.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `pull_request` with path filters. `file:.github/workflows/physics-equivalence.yml:L3-L10`
* `push` (branches: `main`). `file:.github/workflows/physics-equivalence.yml:L10-L12`
* `workflow_dispatch`. `file:.github/workflows/physics-equivalence.yml:L12-L13`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `validate-physics`: 15. `file:.github/workflows/physics-equivalence.yml:L19-L24`

**Jobs:**

* `validate-physics` — Runs reference/accelerated benchmarks, verifies equivalence, and uploads artifacts. `file:.github/workflows/physics-equivalence.yml:L20-L101`
  * Key steps: `benchmark_physics.py`, `verify_equivalence.py`, `calculate_throughput_gain.py`, `actions/upload-artifact@v4`. `file:.github/workflows/physics-equivalence.yml:L41-L86`
  * Notes: PR comment on failure. `file:.github/workflows/physics-equivalence.yml:L87-L101`

---

## `quality-mutation.yml`

**Path:** `.github/workflows/quality-mutation.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Run mutation testing against critical modules and enforce mutation score baseline.

**Axiom focus:**

* A6 — Quality / Mutation / Adversarial Testing

**Trigger(s):**

* `schedule` (nightly 03:00 UTC). `file:.github/workflows/quality-mutation.yml:L3-L7`
* `workflow_dispatch`. `file:.github/workflows/quality-mutation.yml:L6-L8`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `mutation-testing`: 120. `file:.github/workflows/quality-mutation.yml:L16-L20`

**Jobs:**

* `mutation-testing` — Runs mutmut, computes score, enforces baseline, uploads artifacts. `file:.github/workflows/quality-mutation.yml:L17-L132`
  * Key steps: `mutmut run`, `check_mutation_score.py`, `actions/upload-artifact@v4`. `file:.github/workflows/quality-mutation.yml:L55-L117`

---

## `science.yml`

**Path:** `.github/workflows/science.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Execute a flagship experiment workflow to validate scientific hypotheses with artifacts.

**Axiom focus:**

* A2 — CI Correctness & Regression Safety

**Trigger(s):**

* `workflow_dispatch`. `file:.github/workflows/science.yml:L3-L5`
* `schedule` (weekly Sunday 00:00 UTC). `file:.github/workflows/science.yml:L5-L7`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `flagship-experiment`: 30. `file:.github/workflows/science.yml:L12-L16`

**Jobs:**

* `flagship-experiment` — Runs experiment, visualizes results, verifies hypothesis, uploads artifacts. `file:.github/workflows/science.yml:L13-L73`
  * Key steps: `experiments.runner`, `visualize_experiment.py`, `verify_hypothesis`, `actions/upload-artifact@v4`. `file:.github/workflows/science.yml:L32-L56`

---

## `workflow-integrity.yml`

**Path:** `.github/workflows/workflow-integrity.yml`
**Status:** Active

**Intent (1–2 sentences):**

* Lint and validate workflow files for integrity, encoding safety, and safety artifacts.

**Axiom focus:**

* A3 — Workflow Integrity & Provenance
* A9 — Security & Permissions Hygiene

**Trigger(s):**

* `pull_request` (branches: `main`). `file:.github/workflows/workflow-integrity.yml:L3-L6`
* `push` (branches: `main`). `file:.github/workflows/workflow-integrity.yml:L6-L7`

**Timeout(s):**

* Workflow-level: N/A (GitHub has no workflow-level timeout key)
* Job-level:
  * `validate-workflows`: 5. `file:.github/workflows/workflow-integrity.yml:L12-L16`

**Jobs:**

* `validate-workflows` — Runs actionlint, scans for encoding violations, and validates safety artifacts. `file:.github/workflows/workflow-integrity.yml:L12-L70`
  * Key steps: `rhysd/actionlint@v1.7.7`, Python integrity scan, `tools/safety/check_safety_artifacts.py`. `file:.github/workflows/workflow-integrity.yml:L24-L70`

---

## Redundancy & Consolidation

1. **ci-smoke.yml vs ci-pr.yml** (Candidate for consolidation)
   * Rationale (R1, R2, R3): `ci-smoke.yml` and `ci-pr.yml` share identical trigger types for `push` (main) and `pull_request`, and `ci-smoke` job set (`ssot`, `tests-smoke`) overlaps with `ci-pr` which already runs SSOT and smoke tests via reusable workflow. Axiom focus aligns on CI correctness without differentiating constraints. `file:.github/workflows/ci-smoke.yml:L3-L48`, `file:.github/workflows/ci-pr.yml:L3-L279`
   * Proposed consolidation target: Merge/replace `ci-smoke.yml` into `ci-pr.yml` by making smoke checks a required subset, then remove `ci-smoke.yml` after confirming branch protection rules. 
   * Why safe: `ci-pr.yml` already includes SSOT and smoke tests; removing duplicate should not reduce coverage. `file:.github/workflows/ci-pr.yml:L17-L279`
   * What could break: Any external branch protection explicitly requiring `ci-smoke.yml` might need updates.

2. **ci-validation.yml vs ci-validation-elite.yml** (Candidate for consolidation)
   * Rationale (R1, R2, R3): Both use schedule + workflow_dispatch triggers and run validation suites; `ci-validation-elite` includes validation and property tests and a summary job, while `ci-validation` runs SSOT + validation. Overlapping validation domain with similar triggers and axiom focus. `file:.github/workflows/ci-validation.yml:L3-L48`, `file:.github/workflows/ci-validation-elite.yml:L3-L133`
   * Proposed consolidation target: Fold SSOT checks into `ci-validation-elite.yml` or make `ci-validation.yml` call a reusable SSOT job, then remove `ci-validation.yml` after parity. 
   * Why safe: `ci-validation-elite` already exercises validation and property tests daily; adding SSOT would cover the remaining job. `file:.github/workflows/ci-validation-elite.yml:L18-L112`
   * What could break: Weekly cadence expectations or SSOT-only reporting would move to daily unless explicitly scheduled.

3. **ci-property-tests.yml vs chaos-validation.yml** (Candidate for consolidation)
   * Rationale (R1, R2): Both run on schedule + workflow_dispatch; `chaos-validation` includes a `property-tests` job running Hypothesis tests, overlapping purpose with `ci-property-tests`. `file:.github/workflows/ci-property-tests.yml:L3-L43`, `file:.github/workflows/chaos-validation.yml:L3-L140`
   * Proposed consolidation target: Keep `chaos-validation.yml` as the nightly property-test source of truth and remove `ci-property-tests.yml` after confirming profiles align. 
   * Why safe: Property tests already run in `chaos-validation` with artifacts and summaries. `file:.github/workflows/chaos-validation.yml:L82-L140`
   * What could break: Profile differences (`ci` vs `thorough`) may change coverage/time; align profiles before removal.

4. **benchmarks.yml vs ci-benchmarks-elite.yml vs ci-benchmarks.yml** (Candidate for consolidation)
   * Rationale (R1, R2, R3): All three workflows run benchmark suites via schedule + workflow_dispatch with overlapping performance intent. `ci-benchmarks.yml` already supports schedule, manual, PR, and workflow_call with benchmark execution. `file:.github/workflows/benchmarks.yml:L3-L82`, `file:.github/workflows/ci-benchmarks-elite.yml:L3-L97`, `file:.github/workflows/ci-benchmarks.yml:L3-L166`
   * Proposed consolidation target: Use `ci-benchmarks.yml` as the single benchmark workflow, moving `benchmarks.yml` scenarios and elite baseline comparisons into it, then remove the redundant workflows. 
   * Why safe: `ci-benchmarks.yml` already covers scheduled runs and PR-triggered micro-benchmarks; consolidating reduces duplication. `file:.github/workflows/ci-benchmarks.yml:L3-L166`
   * What could break: Scenario selection inputs and baseline comparison steps must be preserved to avoid losing reports.

---

## Completeness Check

* Inventory count: 22
* Contract blocks count: 22
* Missing: []
* Mandatory named workflows present: YES (chaos-validation.yml, ci-validation-elite.yml, ci-benchmarks-elite.yml, workflow-integrity.yml, quality-mutation.yml, formal-coq.yml, formal-tla.yml)
