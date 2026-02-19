# Repo Readiness Orchestration — 2026-02-19

## Step 1: Inventory (JSON)

```json
{
  "python_version_targets": {
    "runtime": ">=3.11",
    "classifiers": ["3.11", "3.12"],
    "observed_runner": "3.12.12"
  },
  "dependency_files": [
    "pyproject.toml",
    "requirements-lock.txt",
    "bibliography/sources.lock"
  ],
  "make_targets": {
    "core": ["install", "test", "test-gate", "lint", "mypy", "build", "demo", "release"],
    "reproducibility": ["test-determinism", "wheelhouse-build", "wheelhouse-validate", "manifest", "inventory"],
    "source": "Makefile"
  },
  "test_entrypoints": {
    "primary": "python -m pytest -m \"not (validation or property)\" -q",
    "markers": ["validation", "property"],
    "additional_targets": ["make test", "make test-validation", "make test-property", "make test-gate"]
  },
  "ci_workflows": [
    {"file": ".github/workflows/ci-pr-atomic.yml", "name": "CI PR Atomic", "purpose": "fast PR gate"},
    {"file": ".github/workflows/ci-pr.yml", "name": "CI PR", "purpose": "pull request CI umbrella"},
    {"file": ".github/workflows/ci-validation.yml", "name": "CI Validation", "purpose": "validation-marker jobs"},
    {"file": ".github/workflows/quality-mutation.yml", "name": "Quality Mutation", "purpose": "mutation quality gates"},
    {"file": ".github/workflows/release-pipeline.yml", "name": "Release Pipeline", "purpose": "release automation"},
    {"file": ".github/workflows/codeql.yml", "name": "CodeQL", "purpose": "SAST scanning"}
  ],
  "docs_entrypoints": [
    "README.md",
    "docs/START_HERE.md",
    "docs/QUICKSTART.md",
    "docs/REPRODUCIBILITY.md"
  ],
  "reproducibility_hooks": {
    "demo": "make demo",
    "reproduce": "UNKNOWN",
    "canonical_artifact_manifest": "manifest/repo_manifest.computed.json"
  },
  "unknowns": [
    "No explicit `make reproduce` target discovered in Makefile.",
    "Checksum tolerance contract for reproducibility is not codified in one command entrypoint."
  ]
}
```

## Step 2: Risk Triage (ranked)

1. **RISK:** No single `make reproduce` entrypoint for canonical artifact+manifest generation.
   - **SCORE:** `0.80 * 9 * 7 = 50.4`
   - **MITIGATION:** Gate G5 via a focused PR that adds `make reproduce` and writes deterministic manifest/checksums.
2. **RISK:** Determinism policy is spread across docs/targets, not one explicit SSOT command path.
   - **SCORE:** `0.70 * 8 * 6 = 33.6`
   - **MITIGATION:** Gates G0/G1 by codifying a canonical install+lock+hash command and wiring CI reusable workflow to it.
3. **RISK:** Workflow count is high; role overlap can drift and reduce CI signal quality.
   - **SCORE:** `0.60 * 7 * 6 = 25.2`
   - **MITIGATION:** Gate G7 via workflow purpose registry + validation script against drift.
4. **RISK:** Optional local `pylint` gate may fail on unprepared environments.
   - **SCORE:** `0.50 * 5 * 8 = 20.0`
   - **MITIGATION:** Gate G3 by documenting bootstrap (`python -m pip install -e ".[test]"`) and keeping lint commands in `make lint`.
5. **RISK:** Security evidence is fragmented across docs and workflow outputs.
   - **SCORE:** `0.40 * 8 * 5 = 16.0`
   - **MITIGATION:** Gate G4/G10 by centralizing generated evidence pointers under `proof_bundle/logs/` + release artifacts.

## Step 3: PR Series Plan (max 7 PRs)

### PR-1 — Deterministic install SSOT (G0, G1)
- **Files:** `Makefile`, `.github/workflows/_reusable_pytest.yml`, `docs/DETERMINISM.md`
- **Acceptance criteria:** one canonical install command from lockfile with hashes, pip version printed before/after pin in CI.
- **Evidence commands:** `python --version`; `python -m pip --version`; `make install`; `python -m pip check`.

### PR-2 — Test gate hardening (G2, G3)
- **Files:** `Makefile`, `pyproject.toml`, `.github/workflows/ci-pr-atomic.yml`, `docs/TESTING.md`
- **Acceptance criteria:** `make test-gate` stable; marker split remains deterministic; lint/type targets align with docs.
- **Evidence commands:** `make test-gate`; `ruff check .`; `mypy src --strict --config-file pyproject.toml`.

### PR-3 — Security gate normalization (G4)
- **Files:** `.github/workflows/codeql.yml`, `.github/workflows/dependency-review.yml`, `docs/SECURITY_GITLEAKS.md`
- **Acceptance criteria:** gitleaks + dependency review + code scanning mapped to one security matrix.
- **Evidence commands:** `make security`; workflow dispatch logs.

### PR-4 — Reproducibility command surface (G5, G10)
- **Files:** `Makefile`, `scripts/release_pipeline.py`, `docs/REPRODUCIBILITY.md`
- **Acceptance criteria:** `make reproduce` exists, emits artifact manifest with checksums + git SHA and fails on drift.
- **Evidence commands:** `make reproduce`; `python -m scripts.verify_reproducible_artifacts --help`.

### PR-5 — 5-minute onboarding path (G6)
- **Files:** `docs/START_HERE.md`, `README.md`, `docs/QUICKSTART.md`
- **Acceptance criteria:** prerequisites/install/demo/reproduce/tests each as single command with expected outputs.
- **Evidence commands:** `make demo`; `make quickstart-smoke`.

### PR-6 — Interface/ADR/version discipline (G8)
- **Files:** `docs/API_STABILITY.md`, `docs/VERSIONING.md`, `docs/adr/*`
- **Acceptance criteria:** public interface contract documented and ADR link added for any breaking surface.
- **Evidence commands:** `make api-contract`; `make public-surfaces`.

### PR-7 — Release readiness bundle (G9, G10)
- **Files:** `docs/RELEASE_READINESS.md`, `docs/RELEASE_NOTES.md`, `proof_bundle/logs/*`
- **Acceptance criteria:** tag-ready checklist + artifact map + evidence logs cross-linked.
- **Evidence commands:** `python -m build`; `make release-readiness`.
