# M0 Bootstrap Foundation (PR-00 direction correction)

M0 establishes a machine-readable control plane that is explicitly aligned to the canonical proof objective:

Clone → Run → See → Verify emergent dynamics.

## Foundation controls

- `ci/milestone_state.json`: lifecycle state for milestones `M0` through `M11`.
- `ci/statistical_power_config.json`: planned canonical avalanche-admission thresholds (stored now, enforced later).
- `ci/validation_gates.json`: canonical gate registry using `G1`..`G8` IDs and explicit `wired` vs `planned` statuses.
- `schemas/proof-report.schema.json`: bootstrap proof-report contract for verdict + gate-level machine validation.
- `configs/canonical_profile.yaml`: deterministic canonical profile scaffold used by `bnsyn run --profile canonical`.

## Canonical command shape and honesty

Bootstrap CLI compatibility remains:

```bash
bnsyn run --profile canonical --plot --export-proof
```

At M0 this means:
- profile routing is deterministic and active;
- the interface is reserved for canonical run/proof flow;
- `--plot` and `--export-proof` are accepted but not yet fully wired.

This keeps the interface stable without claiming proof-export completeness before M1/M2 implementation.
