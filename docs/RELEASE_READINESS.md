# Release Readiness Protocol

This document defines the **blocking release gates** and the deterministic checklist
for declaring BN-Syn ready for public production/demo use.

## Single-Command Gate (Blocking)

Generate the release readiness report (JSON + Markdown):

```bash
python -m scripts.release_readiness
```

If any blocking checks fail, the command exits with non-zero status and the
release is **BLOCKED**. Reports are written to:

- `artifacts/release_readiness.json`
- `artifacts/release_readiness.md`

## Blocking Release Requirements

The release readiness gate enforces these requirements:

1. **Core repository files present**: `README.md`, `LICENSE`, `SECURITY.md`,
   `CITATION.cff`, `requirements-lock.txt`.
2. **Governance evidence present**: `VERIFICATION_REPORT.md`,
   `GOVERNANCE_VERIFICATION_REPORT.md`, `HARDENING_SUMMARY.md`,
   `README_CLAIMS_GATE.md`.
3. **Quality documentation present**: `docs/QUALITY_INFRASTRUCTURE.md`,
   `docs/CI_GATES.md`, `docs/TESTING_MUTATION.md`.
4. **Quality scripts present**: governance and formal-verification gates
   (`scripts/lint_ci_truthfulness.py`, `scripts/verify_formal_constants.py`)
   and mutation scripts.
5. **Project version defined** in `pyproject.toml`.
6. **Mutation baseline non-trivial and factual**: `quality/mutation_baseline.json`
   must have `status="active"`, `metrics.total_mutants > 0`, and `metrics.killed_mutants > 0`.
7. **Entropy gate consistency**: current repository entropy metrics must satisfy
   comparators in `entropy/policy.json` against `entropy/baseline.json` (no regressions).

## Mutation Baseline (Required for Release)

The mutation baseline is a **blocking** requirement. Generate it with:

```bash
make mutation-baseline
```

This installs test dependencies and runs mutmut against the critical modules.
If the test suite does not run cleanly, the baseline generation fails and the
release remains **BLOCKED** until resolved.

## Optional (Advisory) Mode

To generate the report without failing the command:

```bash
python -m scripts.release_readiness --advisory
```


## Release Pipeline Automation

For changelog/version/build/publish dry-run automation, see [`docs/RELEASE_PIPELINE.md`](RELEASE_PIPELINE.md).
