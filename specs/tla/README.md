# TLA+ Formal Specifications for BNsyn

This directory contains TLA+ specifications for formally verifying critical invariants in the BNsyn thermostated bio-AI system.

## Overview

The BNsyn system implements a temperature-gated plasticity mechanism with criticality control. These formal specifications ensure that key safety properties hold across all possible execution paths.

## Files

- **BNsyn.tla**: Main TLA+ specification module defining the system model and invariants
- **BNsyn.cfg**: Configuration file specifying constants, invariants, and properties to check
- **README.md**: This documentation file

## Verified Invariants

### INV-1: TempMonotone
**Description**: Temperature never increases during the cooling phase (unless already at minimum floor).

**Importance**: Ensures the temperature schedule is monotonically decreasing, which is critical for the consolidation phase to work correctly.

**Formal Statement**: `[]((phase = "active" /\ temperature > Tmin) => (temperature' <= temperature))`

### INV-2: GateSoundness
**Description**: Plasticity gate value correctly reflects the temperature state relative to the critical temperature Tc.

**Importance**: The plasticity gate controls when synaptic changes are allowed. Gate soundness ensures that plasticity is correctly regulated by temperature.

**Formal Statement**: 
```tla
[](((temperature < Tc) => (gate >= 0.5)) /\
   ((temperature > Tc) => (gate <= 0.5)))
```

### INV-3: SigmaClamp
**Description**: The criticality parameter (sigma) always stays within defined bounds [SigmaMin, SigmaMax].

**Importance**: Sigma controls the branching ratio and criticality of the network. Values outside bounds could lead to pathological dynamics (sub-critical silence or super-critical explosions).

**Formal Statement**: `[](sigma >= SigmaMin /\ sigma <= SigmaMax)`

### INV-4: PhaseConsistency
**Description**: Phase transitions follow the valid state machine: active → consolidating → cooled. Cannot skip states or transition backwards.

**Importance**: Ensures the system progresses through phases in the correct order, preventing invalid state combinations.

**Formal Statement**: `[]((phase = "active" /\ phase' = "cooled") => FALSE)`

### INV-5: GateTemperatureCorrelation
**Description**: During the consolidating phase, the plasticity gate must be open (gate > 0).

**Importance**: Consolidation requires plasticity to be active. This invariant ensures the gate is open when consolidation occurs.

**Formal Statement**: `[]((phase = "consolidating") => (gate > 0.0))`

## Running the Model Checker

### Prerequisites

You need the TLA+ Toolbox or the TLC command-line model checker:

- **TLA+ Toolbox**: https://lamport.azurewebsites.net/tla/toolbox.html
- **TLC CLI**: Part of the TLA+ distribution

### Using TLC Command Line

```bash
# Download TLC if not already available
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Run the model checker
java -jar tla2tools.jar -config BNsyn.cfg BNsyn.tla
```

### Expected Output

A successful run will show:
```
TLC2 Version ...
...
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
  calculated (optimistic):  val = ...
```

If an invariant is violated, TLC will provide:
1. The violated invariant
2. A counterexample trace showing the sequence of states leading to the violation

## Configuration Parameters

The configuration file (`BNsyn.cfg`) defines the following constants matching the Python implementation:

| Constant | Value | Description |
|----------|-------|-------------|
| T0 | 1.0 | Initial temperature |
| Tmin | 0.001 | Minimum temperature floor |
| Alpha | 0.95 | Cooling factor (geometric decay) |
| Tc | 0.1 | Critical temperature for gate activation |
| GateTau | 0.02 | Sigmoid sharpness parameter |
| SigmaMin | 0.8 | Minimum allowed sigma value |
| SigmaMax | 1.2 | Maximum allowed sigma value |
| MaxSteps | 100 | Maximum simulation steps for model checking |

## Extending the Specification

To add new invariants:

1. Define the invariant in `BNsyn.tla` using TLA+ temporal logic
2. Add the invariant name to the `INVARIANTS` section in `BNsyn.cfg`
3. Run TLC to verify the new property

Example invariant structure:
```tla
MyNewInvariant ==
    [](<condition that should always hold>)
```

## Integration with CI/CD

The `.github/workflows/formal-tla.yml` workflow automatically runs TLC on every push to verify all invariants. The workflow:

1. Downloads the TLA+ tools
2. Runs TLC with the specified configuration
3. Reports any invariant violations
4. Uploads the full model checking report as an artifact

## References

- **TLA+ Homepage**: https://lamport.azurewebsites.net/tla/tla.html
- **TLA+ Language Manual**: https://lamport.azurewebsites.net/tla/summary.pdf
- **Specifying Systems Book**: https://lamport.azurewebsites.net/tla/book.html
- **BNsyn SPEC**: `docs/SPEC.md` in the repository root

## Limitations

- The TLA+ model is a simplified abstraction of the full Python implementation
- Real values are approximated (gate sigmoid is simplified)
- Numerical precision issues are not modeled
- The model checks a bounded state space (MaxSteps = 100)

For complete verification, property-based testing and chaos engineering complement formal verification.
