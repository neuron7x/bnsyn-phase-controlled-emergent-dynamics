# RUNBOOK: dependency vulnerability

## Detection signals
- CI jobs: `dependency-watch`, `dependency-review`, and `codeql`.
- Artifact/log source: vulnerability report in workflow summary.

## Triage checklist
1. Record package, version, CVE id, and severity.
2. Determine exploitability in this repository context.
3. Verify fixed version availability.
4. Check lockfile and build impact.

## Reproduction commands
```bash
python -m pip install -e ".[dev]"
pip-audit
```

## Rollback/mitigation procedure
1. Patch dependency to fixed version in `pyproject.toml` + lockfile.
2. If no patch exists, pin to non-vulnerable alternative or disable affected optional path with test coverage.
3. Backport security patch to supported release branches.

## Known failure modes
| Failure mode | Signal | Mitigation |
|---|---|---|
| False positive advisory | advisory not affecting used code path | document justification and suppress with expiry |
| Transitive vuln only | direct deps clean, transitive flagged | tighten pinned transitive version |
| Fix breaks ABI | tests fail after bump | apply compatible patch release or vendor fix |
