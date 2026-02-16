# API stability and SemVer policy

## Public API surface
Public API is limited to these import paths:
- `bnsyn` (`__version__`)
- `bnsyn.cli` (`main`)
- `bnsyn.sim.network` (`Network`, `NetworkParams`, `run_simulation`)
- `bnsyn.neuron.adex` (`AdExState`, `adex_step`, `adex_step_adaptive`, `adex_step_with_error_tracking`)
- `bnsyn.synapse.conductance` (`ConductanceState`, `ConductanceSynapses`, `nmda_mg_block`)

Everything else is internal and may change without notice.

## Deprecation policy
- Add deprecation note in docs and changelog for at least one minor release before removal.
- Keep backward-compatible shims during deprecation window.

## SemVer rules
- PATCH: bugfixes only, no signature removals/renames.
- MINOR: backward-compatible additions.
- MAJOR: breaking signature/module changes.

## Compatibility gate
```bash
python scripts/check_public_api_compat.py --baseline quality/public_api_snapshot.json
```
