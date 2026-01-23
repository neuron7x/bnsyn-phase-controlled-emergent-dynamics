# Architecture

## Control loops

- **Micro**: AdEx + conductance synapses update per timestep
- **Meso**: Plasticity updates (three-factor)
- **Macro**: Criticality controller adjusts global gain to keep σ near target
- **Meta**: Temperature scheduler gates plasticity / consolidation phases

## State layout (reference simulator)

- neuron state: (V, w, spiked)
- synapse state: (g_ampa, g_nmda, g_gabaa)
- criticality: (σ estimator, gain controller)


## Optional governance extension: Verified Contribution Gating (VCG)

VCG is an **engineering-contract** module that enforces *result-based reciprocity* at the system boundary: agents that repeatedly match a social/interaction pattern without measurable contribution are **deprioritized** by withdrawing future routing priority and resource budget. VCG is specified as a non-core extension (does not change the 12 core neurodynamics components) and is documented in `docs/VCG.md`.
