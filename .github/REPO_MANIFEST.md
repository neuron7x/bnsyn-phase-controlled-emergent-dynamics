# Repository Manifest (Generated)

- Manifest version: `1.0`
- Generated marker: `deterministic`
- Repository fingerprint: `099436a63fd2dd978074ed93fe4a71c1968091561fd9bf1ddd95e0ccbe520d7f`
- Required PR gates source: `.github/PR_GATES.yml`
- Required PR gates SHA-256: `e1ba5284084ab99fa941e2d0f20480e55054fa06cc54f8f1a4d37b176edbd3df`

## Metrics

- Workflow files (`.github/workflows/*.yml`): **28**
- Reusable workflow files (`_reusable_*.yml`): **9**
- Workflows declaring `workflow_call`: **11**
- Required PR gates (`.github/PR_GATES.yml`): **3**
- Coverage minimum percent (`quality/coverage_gate.json`): **99.0**
- Coverage baseline percent (`quality/coverage_gate.json`): **99.57**
- Mutation baseline score (`quality/mutation_baseline.json`): **0.0**
- Mutation total mutants (`quality/mutation_baseline.json`): **103**
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
