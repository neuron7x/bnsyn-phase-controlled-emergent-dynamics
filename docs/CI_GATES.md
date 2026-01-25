# CI Gates

## Merge-Blocking Checks for PRs

All jobs in `.github/workflows/ci-pr.yml` block pull request merge if they fail.

| Gate | Workflow Job | When It Runs | Local Command | Blocks Merge |
| --- | --- | --- | --- | --- |
| SSOT governance | `ssot-governance` | PR/push/dispatch | `python scripts/validate_bibliography.py`<br>`python scripts/validate_claims.py`<br>`python scripts/scan_normative_tags.py`<br>`python scripts/scan_governed_docs.py` | Yes |
| Tests (non-validation) | `tests` | PR/push/dispatch | `pytest -m "not validation"` | Yes |
| Code quality | `code-quality` | PR/push/dispatch | `ruff check .`<br>`mypy src` | Yes |
| Dependency review | `dependency-security` | PR only | GitHub Action: `actions/dependency-review-action` | Yes |
| OSV scan | `dependency-security` | PR/push/dispatch | `osv-scanner --severity=HIGH --format sarif --output osv.sarif .` | Yes |
| Secrets scan | `secrets` | PR/push/dispatch | `gitleaks detect --source . --report-path gitleaks-repo.sarif --report-format sarif`<br>`gitleaks detect --source . --report-path gitleaks-pr.sarif --report-format sarif --log-opts "BASE..HEAD"` | Yes |
| SAST (CodeQL) | `sast` | PR/push/dispatch | CodeQL CLI (optional), or GitHub Action `github/codeql-action` | Yes |
| Micro-benchmarks | `benchmarks-pr` | PR/push/dispatch | `python scripts/run_benchmarks.py --suite ci --output benchmarks/pr-results.json --summary benchmarks/pr-summary.json` | Yes |
| Physics invariants + benchmarks | `physics-benchmarks` | PR + weekly | `pytest tests/validation/test_physical_invariants.py`<br>`pytest --benchmark-json=bench.json`<br>`python scripts/compare_physics_baseline.py --bench bench.json --metrics artifacts/physics_metrics.json --output diffs.json` | Yes |

## Scheduled Validation & Performance

These workflows run weekly and on manual dispatch. They do not block PRs directly but provide continuous validation artifacts.

| Workflow | Job | When It Runs | Local Command | Artifact Output |
| --- | --- | --- | --- | --- |
| `ci-validation.yml` | `validation` | weekly/dispatch | `python scripts/validate_bibliography.py`<br>`python scripts/validate_claims.py`<br>`python scripts/scan_normative_tags.py`<br>`python scripts/scan_governed_docs.py`<br>`pytest -m validation` | `validation-logs-<run_id>` |
| `ci-validation.yml` | `benchmarks-weekly` | weekly/dispatch | `python scripts/run_benchmarks.py --suite full --output benchmarks/weekly-results.json --summary benchmarks/weekly-summary.json` | `benchmarks-weekly-<run_id>` |
| `sbom.yml` | `generate-sbom` | weekly/dispatch | `syft packages dir:. -o syft-json=sbom.syft.json` | `sbom-<run_id>` |
| `ci-bench.yml` | `physics-benchmarks` | weekly | `pytest tests/validation/test_physical_invariants.py`<br>`pytest --benchmark-json=bench.json`<br>`python scripts/compare_physics_baseline.py --bench bench.json --metrics artifacts/physics_metrics.json --output diffs.json` | `physics-benchmarks-<run_id>` |

## Artifacts

Artifacts are uploaded for:

- Gitleaks reports (`gitleaks-repo-<run_id>`, `gitleaks-pr-<run_id>`)
- Validation logs (`validation-logs-<run_id>`)
- Benchmarks (`benchmarks-pr-<run_id>`, `benchmarks-weekly-<run_id>`)
- SBOM (`sbom-<run_id>`)
