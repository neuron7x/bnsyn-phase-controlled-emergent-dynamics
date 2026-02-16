# Hygiene: Remove Proven Digital Noise

## Scope
Deletions only for deterministic local byproducts/noise; no source logic/config behavior changes.

## Summary
- Deleted 9 untracked deterministic cache/byproduct directories after T1–T4 checks.
- Added audit artifacts: manifest, report, proof logs, proof index, and hashes.

## Evidence Pointers
- Candidate discovery: `proof_bundle/logs/014_find_cache_dirs.log`, `proof_bundle/logs/025_check_top_noise.log`
- Classification: `proof_bundle/logs/026`–`061` logs
- Dry-run safety checks: `proof_bundle/logs/016_dryrun_clean_ignored.log`, `proof_bundle/logs/017_dryrun_clean_all.log`
- Deletions: `proof_bundle/logs/062`–`079`, `proof_bundle/logs/087_final_delete_noise.log`
- Final absence check: `proof_bundle/logs/088_final_verify_noise_absent.log`

## Reproduction Commands
```bash
git ls-files -o -i --exclude-standard
find . -maxdepth 2 \( -type d -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \)
ruff check .
mypy src --strict --config-file pyproject.toml
```
