# Repository Manifest (Generated)

- Manifest version: `1.0`
- Generated marker: `deterministic`
- Repository fingerprint: `19758bd3093820883206b5b965d24994557f56c996ad2fc7ca7d41d0eed665b0`
- Required PR gates source: `.github/PR_GATES.yml`
- Required PR gates SHA-256: `cf889e9662eae1fff328239d931b364c6465ee5aae07a1ad0b52d16fc9ab0bc9`

## Metrics

- Workflow files (`.github/workflows/*.yml`): **29**
- Reusable workflow files (`_reusable_*.yml`): **9**
- Workflows declaring `workflow_call`: **11**
- Required PR gates (`.github/PR_GATES.yml`): **4**
- Coverage minimum percent (`quality/coverage_gate.json`): **99.0**
- Coverage baseline percent (`quality/coverage_gate.json`): **99.57**
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
