# RUNBOOK: release failure

## Detection signals
- CI jobs: `release-pipeline` and `release-readiness`.
- Artifact/log source: failed build/upload or failed readiness report artifact.

## Triage checklist
1. Determine failing stage (build, tests, publish, post-release checks).
2. Confirm tag/version consistency (`pyproject.toml` vs package metadata).
3. Inspect wheel install smoke output.
4. Validate required checks status contexts.

## Reproduction commands
```bash
python -m build
python -m pytest -m "not validation" -q
python scripts/release_readiness_report.py --output quality/release_readiness_report.md
```

## Rollback/mitigation procedure
1. Stop publication and revoke incomplete release artifacts.
2. Retag only after green readiness report and full CI pass.
3. Publish patch release with corrected metadata/process issue.

## Known failure modes
| Failure mode | Signal | Mitigation |
|---|---|---|
| Version mismatch | readiness report shows mismatch | update version source and regenerate artifacts |
| Build reproducibility break | wheel smoke install fails | lock dependencies, rebuild |
| Missing required status context | release gate script fails | sync required checks configuration |
