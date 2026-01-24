# Cold Emergence in BN-Syn

## Overview

**Cold emergence** refers to deterministic, information-driven system organization that occurs without affective or motivational modulation. This is in contrast to "hot" emergence driven by reward signals and exploratory behavior.

## Theoretical Foundation

### Definition

BN-Syn implements cold emergence through its multi-scale architecture:

```
Micro (AdEx + conductance) → Meso (three-factor plasticity) → 
Macro (criticality control: σ tracking) → Meta (temperature schedule)
```

This architecture exhibits cold emergence because:

1. **Micro-level**: Deterministic neurons + synaptic conductances (NO explicit motivational modulation)
2. **Meso-level**: Three-factor plasticity is deterministic, not probabilistic reward-based
3. **Macro-level**: **Criticality σ (sigma) at phase transition boundary** = thermodynamic cold emergence
4. **Meta-level**: Temperature schedule controls phase transitions

### Key Mechanisms

#### 1. Attractor Stabilization through Phase Control

Cold emergence manifests through stable attractors formed via phase (thermodynamic) control:

- **Lyapunov function** measures attractor stability
- **Basin of attraction** mapping in state space
- **Cold reset mechanism**: When system enters cold phase, attractors are sharply defined without noise

**Implementation**: `bnsyn.cold_emergence.controller.ColdPhaseAttractorController`

```python
from bnsyn.cold_emergence import ColdPhaseAttractorController

controller = ColdPhaseAttractorController(target_sigma=1.0)

# Compute stability
lyapunov = controller.compute_lyapunov_exponent(state, perturbed_state, dt=0.1)

# Guide to attractor
force, temp_reduction = controller.stabilize_cold_attractor(
    current_state, target_state, current_sigma
)

# Check if in cold phase
is_cold = controller.is_in_cold_phase(sigma=1.0, lyapunov=-0.2)
```

**Properties**:
- Lyapunov exp < 0 → cold, deterministic phase
- Lyapunov exp ≈ 0 → critical phase (maximal information integration)
- Lyapunov exp > 0 → hot, exploratory phase

#### 2. Information Integration (Φ, Integrated Information Theory)

Cold emergence maximizes integration without exploration:

**Implementation**: `bnsyn.cold_emergence.information_metrics.IntegratedInformationMetric`

```python
from bnsyn.cold_emergence import IntegratedInformationMetric

iit = IntegratedInformationMetric()

# Compute integrated information
phi = iit.compute_phi(state, partition_size=5)

# Compute synergy vs redundancy
synergy = iit.compute_synergy(state)
```

**Properties**:
- Φ measures how much the system is integrated (irreducible to parts)
- Cold emergence: high Φ + low entropy
- Synergy > 0.5 indicates emergent information processing

#### 3. Functional System Organization (Anokhin-style)

Integration of Anokhin's functional systems **without motivational tension**:

**Implementation**: `bnsyn.cold_emergence.functional_systems.ColdFunctionalSystem`

```python
from bnsyn.cold_emergence import ColdFunctionalSystem
import numpy as np

# Initialize with goal representation
goal = np.zeros(10)
system = ColdFunctionalSystem(goal_representation=goal, error_threshold=0.1)

# Afferent synthesis (cold, logical integration)
integrated = system.afferent_synthesis(sensory_input, memory_context)

# Acceptor of result (informational error, not dopamine)
error = system.acceptor_of_result(predicted_state)

# Execute cold program
success, correction = system.execute_cold_program(action, sensory, memory)
```

**Key differences from motivational systems**:
- NO dopamine/reward signals
- Organization is INFORMATIONAL, not driven by affect
- Error signals are for correction, not reinforcement

#### 4. Formal Verification

Mathematical validation that emergence is genuinely cold:

**Implementation**: `bnsyn.cold_emergence.validator.ColdEmergenceValidator`

```python
from bnsyn.cold_emergence import ColdEmergenceValidator

validator = ColdEmergenceValidator(
    lyapunov_threshold=-0.1,
    phi_threshold=0.3,
    synergy_threshold=0.5
)

# Comprehensive validation
result = validator.validate_cold_emergence(state, lyapunov_exp, temperature)

assert result["is_truly_cold_emergent"]
assert result["is_deterministic"]
assert result["is_integrated"]
assert result["is_synergistic"]
```

**Axioms of cold emergence**:
1. **Determinism**: Lyapunov exp < 0 (exponential stability)
2. **Integration**: Φ > threshold (non-reducible)
3. **Organization without rewards**: synergy > 0.5 (information is emergent, not redundant)

## Integration with BN-Syn Core

Cold emergence components integrate with existing BN-Syn architecture:

### Temperature Control
- Temperature schedule (`bnsyn.temperature.schedule`) controls transition to cold phase
- Low temperature → deterministic dynamics → cold emergence
- Temperature acts as mesoscopic control of cold emergence

### Criticality Control
- Sigma (σ) tracking (`bnsyn.criticality.branching`) identifies critical phase
- σ ≈ 1.0 is the critical point where cold emergence is maximal
- Homeostatic gain control maintains system near criticality

### Network Dynamics
- AdEx neurons + conductances provide deterministic substrate
- Three-factor plasticity enables structured learning without explicit reward
- Cold emergence naturally arises from the interaction of these components

## Expected Results

After cold emergence integration, BN-Syn achieves:

✅ **Explicit cold emergence mechanism** — system can be directed into cold phase operating without motivational drive

✅ **Formal Anokhin integration** — afferent synthesis + acceptor-of-result without dopamine

✅ **IIT validation** — proof that Φ > critical value in cold phase

✅ **Thermodynamic control** — temperature as mesoscopic control of cold emergence

✅ **Deterministic attractors** — attractors with negative Lyapunov exponent = fully cold

## References

- **Integrated Information Theory (IIT)**: Tononi et al. - measures system integration
- **Anokhin Functional Systems**: P.K. Anokhin - functional organization principles
- **Critical Dynamics**: Beggs & Plenz - criticality in neural systems
- **Phase Transitions**: Thermodynamic control of emergent behavior
- **Lyapunov Exponents**: Dynamical system stability analysis

## Usage Examples

### Example 1: Validate Cold Emergence in Network

```python
import numpy as np
from bnsyn.sim.network import Network, NetworkParams
from bnsyn.config import AdExParams, SynapseParams, CriticalityParams
from bnsyn.cold_emergence import ColdEmergenceValidator

# Create network
rng = np.random.default_rng(42)
net = Network(
    NetworkParams(N=100),
    AdExParams(),
    SynapseParams(),
    CriticalityParams(),
    dt_ms=0.1,
    rng=rng
)

# Run simulation
trajectory = []
for _ in range(100):
    net.step()
    trajectory.append(net.state.V_mV.copy())

# Validate cold emergence
validator = ColdEmergenceValidator()
state = trajectory[-1]
lyap = validator.measure_lyapunov(trajectory, dt=0.1)

result = validator.validate_cold_emergence(
    state, 
    lyapunov_exponent=lyap,
    temperature=0.05
)

print(f"Is cold emergent: {result['is_truly_cold_emergent']}")
print(f"Φ = {result['integrated_information_phi']:.3f}")
print(f"Synergy = {result['synergy_vs_redundancy']:.3f}")
```

### Example 2: Attractor Control

```python
from bnsyn.cold_emergence import ColdPhaseAttractorController
import numpy as np

# Initialize controller
controller = ColdPhaseAttractorController(target_sigma=1.0)

# Guide system to cold attractor
current_state = np.random.normal(size=100)
target_state = np.zeros(100)

force, temp_factor = controller.stabilize_cold_attractor(
    current_state,
    target_state,
    current_sigma=1.05
)

# Apply force to system dynamics
# ... (integrate with network step)
```

### Example 3: Functional System

```python
from bnsyn.cold_emergence import ColdFunctionalSystem
import numpy as np

# Define goal (internal model)
goal = np.array([0.0] * 10)
system = ColdFunctionalSystem(goal_representation=goal)

# Simulate sensory-motor loop
for step in range(100):
    sensory = np.random.normal(size=5)
    memory = np.random.normal(size=5)
    action = np.random.normal(size=5)
    
    success, error = system.execute_cold_program(action, sensory, memory)
    
    if success:
        print(f"Step {step}: Goal achieved (error={error:.3f})")
    else:
        print(f"Step {step}: Correction needed (error={error:.3f})")
```

## Testing

Cold emergence components are tested in `tests/test_cold_emergence_smoke.py`:

```bash
pytest tests/test_cold_emergence_smoke.py -v
```

## See Also

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) - System architecture
- [`docs/SPEC.md`](SPEC.md) - Formal specification
- [`src/bnsyn/cold_emergence/`](../src/bnsyn/cold_emergence/) - Implementation
- [`claims/claims.yml`](../claims/claims.yml) - Evidence and claims
