# Change Management

This document summarizes how release-oriented changes are verified and staged using existing repository automation.

## Primary references

- [docs/RELEASE_PIPELINE.md](RELEASE_PIPELINE.md)
- [scripts/release_pipeline.py](../scripts/release_pipeline.py)
- [Makefile](../Makefile)

## Release verification flow

1. Read current version from `pyproject.toml`.
2. Optionally compute a bump target (`patch`, `minor`, `major`).
3. Validate that `CHANGELOG.md` contains a matching `## [X.Y.Z]` section.
4. In verify-only mode, stop after checks.
5. In full mode, build artifacts and run publish dry-run validation.

## Local commands

```bash
python -m scripts.release_pipeline --verify-only
python -m scripts.release_pipeline
python -m scripts.release_pipeline --bump patch --apply-version-bump --verify-only
```

## CI/workflow alignment

- Release pipeline workflow: `.github/workflows/release-pipeline.yml`
- Manual dispatch supports `bump` and `apply-version-bump` inputs as documented in `docs/RELEASE_PIPELINE.md`.

## Documentation and inventory maintenance

For documentation updates, run inventory regeneration so `INVENTORY.json` reflects repository state:

```bash
python tools/generate_inventory.py
git diff --exit-code -- INVENTORY.json
```
