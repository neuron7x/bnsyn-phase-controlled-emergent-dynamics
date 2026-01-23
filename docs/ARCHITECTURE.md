# Architecture

This document binds the BN-Syn architecture to the formal specification (`docs/SPEC.md`)
and SSOT evidence registry (`bibliography/*`, `claims/claims.yml`).

## Control loops

- **Micro**: AdEx + conductance synapses update per timestep.
- **Meso**: Plasticity updates (three-factor learning).
- **Macro**: Criticality controller adjusts global gain to keep σ near target.
- **Meta**: Temperature scheduler gates plasticity / consolidation phases.

## Evidence crosswalk (core components)

| Component | SPEC section | Claim IDs |
| --- | --- | --- |
| AdEx neuron dynamics | P0-1 | CLM-0001, CLM-0002 |
| Conductance synapses + NMDA Mg²⁺ block | P0-2 | CLM-0003 |
| Three-factor learning | P0-3 | CLM-0004 |
| Neuromodulated STDP + eligibility traces | P0-3 | CLM-0005 |
| Criticality σ tracking | P0-4 | CLM-0006, CLM-0007 |
| Subsampling-corrected σ estimation | P0-4 | CLM-0008 |
| Power-law validation for avalanches | P0-4 | CLM-0009 |
| Temperature schedule + gating | P1-5 | CLM-0019 |
| Dual-weight consolidation | P1-6 | CLM-0010 |
| Governance SSOT policy | P2-8..12 | CLM-0011 |
| Reproducibility process anchors | P2-8..12 | CLM-0012, CLM-0013, CLM-0014 |

## State layout (reference simulator)

- neuron state: (V, w, spiked)
- synapse state: (g_ampa, g_nmda, g_gabaa)
- criticality: (σ estimator, gain controller)

## Optional governance extension: Verified Contribution Gating (VCG)

VCG is an **engineering-contract** module that enforces result-based reciprocity at the system boundary:
agents that repeatedly match a social/interaction pattern without measurable contribution are
deprioritized by withdrawing future routing priority and resource budget. VCG is specified
as a non-core extension (does not change the 12 core neurodynamics components) and is documented
in `docs/VCG.md`.
