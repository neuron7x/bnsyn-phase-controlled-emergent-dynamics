# Repository Manifest (Generated)

- Manifest version: `1.0`
- Generated marker: `deterministic`
- Repository fingerprint: `899ade497e523f5dec89f99390b46e3f0e0703cac7b0d660f9e951467a01968e`
- Required PR gates source: `.github/PR_GATES.yml`
- Required PR gates SHA-256: `f270bd5b9ebd7ead1d5f566ce9cd4c06049ea26193237d6dc72354be8be858ad`

## Metrics

- Workflow files (`.github/workflows/*.yml`): **40**
- Reusable workflow files (`_reusable_*.yml`): **9**
- Workflows declaring `workflow_call`: **11**
- Required PR gates (`.github/PR_GATES.yml`): **7**
- Coverage minimum percent (`quality/coverage_gate.json`): **98.0**
- Coverage baseline percent (`quality/coverage_gate.json`): **98.32**
- Mutation baseline score (`quality/mutation_baseline.json`): **51.61**
- Mutation total mutants (`quality/mutation_baseline.json`): **31**
- `ci_manifest.json` exists: **False**
- `ci_manifest.json` references in scoped scan: **0**
- `ci_manifest.json` scan scope:
  - `.github/workflows`
  - `scripts`
  - `docs`
  - `Makefile`
  - `README.md`

## Invariants

| ID | Statement | Enforcement | Evidence kind |
|---|---|---|---|
| INV-001 | Repository manifest markdown and computed snapshot are generated artifacts. | `python -m tools.manifest generate && git diff --exit-code -- .github/REPO_MANIFEST.md manifest/repo_manifest.computed.json` | artifact |
| INV-002 | PR gate workflows required by .github/PR_GATES.yml exist in .github/workflows/. | `python -m tools.manifest validate` | path:line-span |
| INV-003 | Coverage gate baseline is defined and parseable from quality/coverage_gate.json. | `python -m tools.manifest validate` | path:line-span |
| INV-004 | Mutation baseline is defined and parseable from quality/mutation_baseline.json. | `python -m tools.manifest validate` | path:line-span |
| INV-005 | ci_manifest.json is not used by automation and must remain absent. | `python -m tools.manifest validate` | artifact |

## Evidence Rules

Accepted pointer formats:
- `path:line-span`
- `artifact`
