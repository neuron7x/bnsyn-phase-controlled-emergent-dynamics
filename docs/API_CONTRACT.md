# BN-Syn API Contract

This document defines the public API surface of BN-Syn and identifies stable vs internal interfaces.

## Stable Public API (Supported)

The following modules are considered stable for external use and are covered by documentation and CI:

- `bnsyn.config`
- `bnsyn.rng`
- `bnsyn.cli`
- `bnsyn.sim.network`
- `bnsyn.neuron.adex`
- `bnsyn.synapse.conductance`
- `bnsyn.plasticity.three_factor`
- `bnsyn.criticality.branching`
- `bnsyn.temperature.schedule`
- `bnsyn.connectivity.sparse`

Recommended import style:

```python
from bnsyn import config, rng
from bnsyn.sim.network import Network, NetworkParams
```

## Internal / Private Modules (Not Stable)

Modules not listed above are considered internal and may change without notice. Internal modules may be used
in tests or experiments but are not part of the public API contract.

## References

- [SPEC](SPEC.md)
- [SSOT](SSOT.md)
- [REPRODUCIBILITY](REPRODUCIBILITY.md)
