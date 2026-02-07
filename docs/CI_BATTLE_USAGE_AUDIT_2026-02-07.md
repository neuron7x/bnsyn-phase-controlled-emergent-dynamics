# CI + Battle Usage Verification Audit

Date (as-of): 2026-02-07 Europe/Zaporozhye
Target: https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics

## 1) FACTS (with hard evidence)

### A1. Repository metadata (GitHub API/UI)
- Repository metadata endpoint: `GET /repos/neuron7x/bnsyn-phase-controlled-emergent-dynamics` saved at `artifacts/audit/repo.json`.
- License: `MIT`; primary language: `Python`; open issues count: `0`; pushed_at: `2026-02-07T11:29:24Z`. Evidence: `artifacts/audit/repo.json`.
- Languages breakdown endpoint saved at `artifacts/audit/languages.json`: Python, TeX, Makefile, TLA, Rocq Prover, Shell, Dockerfile.
- Contributors API snapshot saved at `artifacts/audit/contributors.json`; count in retrieved list: `3` (`neuron7x`, `Copilot`, `dependabot[bot]`).
- Releases API snapshot saved at `artifacts/audit/releases.json`; count: `0`.
- Tags API snapshot saved at `artifacts/audit/tags.json`; count: `0`.
- Commit count (main): from `GET /commits?sha=main&per_page=1` header `rel="last" ... page=571` saved at `artifacts/audit/commits_headers.txt` => `571` commits.
- Actions visibility proof: workflows endpoint `GET /actions/workflows` returned active workflows (saved at `artifacts/audit/workflows_api.json` and flattened `artifacts/audit/workflows_list.tsv`).

### A2. Workflow tree evidence (.github/workflows)
- Local workflow files count: `27` (`.github/workflows/*.yml`).
- `pull_request` trigger found in exactly:
  - `.github/workflows/ci-pr-atomic.yml` (`on:`, `pull_request:`)
  - `.github/workflows/workflow-integrity.yml` (`on:`, `pull_request:`)
- Trigger scan command artifact: `rg -n "^on:|pull_request|schedule|workflow_dispatch" .github/workflows/*.yml` output captured in terminal log.

### A3. README + pyproject sources
- README source path: `README.md`.
- pyproject source path: `pyproject.toml`.

### B1. Workflows enumerated from API
- Workflow inventory from `GET /actions/workflows` saved in `artifacts/audit/workflows_api.json`.
- Flattened list (name, id, path, state) in `artifacts/audit/workflows_list.tsv`.

### B2. PR-gate workflow run evidence (branch=main)
- PR-gate workflow IDs mapped from API paths:
  - `ci-pr-atomic` -> `226681253`
  - `Workflow Integrity` -> `229502046`
- Runs queried with:
  - `GET /actions/workflows/226681253/runs?branch=main&per_page=20`
  - `GET /actions/workflows/229502046/runs?branch=main&per_page=20`
- Last 10 runs exported to:
  - `artifacts/audit/workflow_226681253_runs.tsv`
  - `artifacts/audit/workflow_229502046_runs.tsv`
- Observed in both last-10 snapshots: all `status=completed`, all `conclusion=success`, latest `created_at=2026-02-07T11:29:25Z`.

### B4. Flakiness check facts
- Failure count among last 10 runs:
  - workflow `226681253`: 0 failures.
  - workflow `229502046`: 0 failures.
- No failure URLs to list from sampled window.

### B5. Secrets/permissions gating facts
- `ci-pr-atomic.yml` uses:
  - `permissions: contents: read`
  - optional `CODECOV_TOKEN` (`workflow_call.secrets.CODECOV_TOKEN.required: false`)
  - `GITHUB_TOKEN` for gitleaks action step.
- Despite secret usage, sampled recent `main` runs are successful for the workflow.

### C1/C2. Local reproducibility execution
- Requested snapshot zip `/mnt/data/bnsyn-phase-controlled-emergent-dynamics-main.zip` was not present in this environment (lookup command returned “No such file or directory”).
- Local reproducibility executed against current checkout at `/workspace/bnsyn-phase-controlled-emergent-dynamics`.
- Environment and install logs:
  - `artifacts/ci_local/tool_versions.txt`
  - `artifacts/ci_local/pip_upgrade.log`
  - `artifacts/ci_local/pip_install.log`
- PR-gate-equivalent command outputs/logs saved under `artifacts/ci_local/*.log` with exit codes in `artifacts/ci_local/*.exit`; consolidated in `artifacts/ci_local/summary.tsv`.
- Local result highlights from `summary.tsv`:
  - failed: `ruff_format` (exit 1), `mypy_strict` (exit 1)
  - passed: determinism tests (3 runs), RNG isolation test, pylint, SSOT validation commands, manifest checks, `pip-audit`, `bandit`.

### D1. Public battle-usage signals
- Releases count: `0` (`artifacts/audit/releases.json`).
- Tags count: `0` (`artifacts/audit/tags.json`).
- Stars/forks from repo metadata: `0/0` (`artifacts/audit/repo.json`).
- Recent issues/PR sample (`artifacts/audit/issues_all.tsv`, `artifacts/audit/pulls_all.tsv`) shows activity dominated by owner account (`neuron7x`).
- Contributors API includes bot/service accounts (`Copilot`, `dependabot[bot]`) in addition to owner.

### D2. In-repo usage/deployment evidence
- Operational/deployment-style docs exist (examples):
  - `docs/RELEASE_PIPELINE.md`
  - `docs/CONFERENCE_RUNBOOK.md`
  - `docs/appendix/PRODUCTION_AUDIT.md`
  - `docs/RELEASE_READINESS.md`
(Existence verified from repository tree listing in terminal.)

## 2) INFERENCES (strictly derived)
- CI executability in GitHub Actions is **proven runnable** for PR-gate workflows because both PR-gates have recent successful `main` runs in the last 30 days and all sampled last-10 runs succeeded.
- Local reproducibility is **partially divergent** from the sampled remote CI outcomes because local strict quality checks (`ruff format`, `mypy --strict`) failed while most other PR-gate-equivalent checks passed.
- Flakiness is **not indicated** in sampled windows (0/10 failures for each PR-gate workflow).
- Battle-usage external corroboration is weak: no releases/tags/stars/forks and no clear non-owner human usage signal from sampled issue/PR data.

## 3) HYPOTHESES / NEEDS_EVIDENCE
- NEEDS_EVIDENCE: provided zip path `/mnt/data/bnsyn-phase-controlled-emergent-dynamics-main.zip` is unavailable in this runtime; to fully satisfy strict snapshot requirement, provide that file or an alternate mounted path.
- NEEDS_EVIDENCE: exact GitHub UI-visible contributor/commit widgets as rendered in browser were not screenshot-captured; API evidence was used instead.
- Verification steps if needed:
  1. Re-run local phase against actual mounted snapshot zip.
  2. Capture GitHub UI screenshots for top-bar counts (commits/contributors/releases/tags/issues).

## 4) Phase E Verdict

### E1. PR-gate workflow status table
| PR-gate workflow | Last run date (UTC) | Conclusion | Run URL |
|---|---:|---|---|
| ci-pr-atomic (226681253) | 2026-02-07T11:29:25Z | success | https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/runs/21779363591 |
| Workflow Integrity (229502046) | 2026-02-07T11:29:25Z | success | https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/runs/21779363597 |

### E1. Local reproduction table
| Local reproduction item | Result | Evidence |
|---|---|---|
| Dependency install in isolated venv | PASS | `artifacts/ci_local/pip_install.log` |
| PR-gate-equivalent checks (aggregate) | MIXED | `artifacts/ci_local/summary.tsv` |
| Strict format/type gates | FAIL locally | `artifacts/ci_local/ruff_format.log`, `artifacts/ci_local/mypy_strict.log` |

### E1. Battle usage signals table
| Signal | Status | Evidence |
|---|---|---|
| Releases/tags | none | `artifacts/audit/releases.json`, `artifacts/audit/tags.json` |
| External community activity | not demonstrated | `artifacts/audit/issues_all.tsv`, `artifacts/audit/pulls_all.tsv`, `artifacts/audit/contributors.json` |
| Internal production/deploy docs | present | repository docs paths listed above |

### E2. Final outputs
- `CI_EXECUTABILITY_STATUS = PROVEN_GREEN`
- `BATTLE_USAGE_STATUS = NOT_PROVEN`
- `READYNESS_PERCENT = 80`

Readiness calculation:
- Start 100
- CI is PROVEN_GREEN: -0
- Local reproducibility proof exists: -0
- Battle usage NOT_PROVEN: -20
- Flakiness detected: -0
Final score per required formula: **80%**.
