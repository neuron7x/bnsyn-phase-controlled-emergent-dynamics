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
