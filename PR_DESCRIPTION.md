# SleepCycle Validation + Documentation Link Repairs

## Problem
- SleepCycle.wake did not enforce a deterministic, message-stable guard for non-positive record_interval values, leaving error messaging inconsistent with test expectations and masking the regression target for record_interval=0.
- Multiple docstring/test references pointed to missing docs/features/*.md paths.
- The PR description typo (12,683× vs 24,693×) is still present on the GitHub PR description, not in-repo.

## Fix
- Added a deterministic guard at the start of SleepCycle.wake: record_interval <= 0 now raises ValueError("record_interval must be > 0").
- Updated test expectations and added regression tests for record_interval <= 0.
- Replaced all docs/features/*.md references with existing canonical docs:
  - docs/features/provenance_manifest.md → docs/REPRODUCIBILITY.md
  - docs/features/emergence_crystallizer.md → docs/emergence_tracking.md
  - docs/features/viz_dashboard.md → docs/LEGENDARY_QUICKSTART.md
  - docs/features/memory.md → docs/sleep_stack.md
- PR description edit required (not in repo): replace 12,683× with 24,693×.

## Before/After
- Before: wake(record_interval=0 or -1) raised ValueError("record_interval must be a positive integer, got <value>") only after record_memories gating; no stable message requirement.
- After: wake(record_interval<=0) raises ValueError("record_interval must be > 0") deterministically at method entry.

Doc link fix list:
- docs/features/provenance_manifest.md → docs/REPRODUCIBILITY.md
- docs/features/emergence_crystallizer.md → docs/emergence_tracking.md
- docs/features/viz_dashboard.md → docs/LEGENDARY_QUICKSTART.md
- docs/features/memory.md → docs/sleep_stack.md

## Verification
```
pytest -q tests/test_sleep_cycle.py
.........                                                                [100%]
```
```
make test
========== 419 passed, 4 skipped, 98 deselected, 4 warnings in 56.37s ==========
```
```
make quality
✅ All quality checks passed
```
```
make docs
build succeeded, 54 warnings.
```
```
make ssot
OK: bibliography SSOT validated.
[claims-gate] OK: 26 claims validated; 22 normative.
OK: normative tag scan passed.
```

## Risks
- Low: changes are validation-only, doc reference updates, and tests. No public API/CLI changes.

## Rollback
- Revert the commit; no data migrations or schema changes required.
