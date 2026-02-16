# Architecture

This document binds the BN-Syn architecture to the formal specification (`docs/SPEC.md`)
and the SSOT evidence registry (`bibliography/*`, `claims/claims.yml`). It is a
traceability index, not an independent source of truth. The authoritative definition
of behavior remains the specification, while evidence provenance is governed by SSOT.

**Navigation**: [INDEX.md](INDEX.md) | [GOVERNANCE.md](GOVERNANCE.md) | [SPEC.md](SPEC.md) | [MODULE_RESPONSIBILITY_MATRIX.md](MODULE_RESPONSIBILITY_MATRIX.md)

## Scope and traceability

- **Spec authority**: equations, invariants, and parameters are defined in `docs/SPEC.md`.
- **Evidence authority**: claim IDs and bibliographic provenance are defined in
  `claims/claims.yml` and `bibliography/*`.
- **Purpose**: expose the mapping between architecture components and their formal
  references without introducing new semantics.
- **Audit artifacts**: component mappings live in `docs/spec_to_code.yml` and the
  verification table is tracked in `docs/COMPONENT_AUDIT.md`.

## Control loops (timescale-segregated)

- **Micro (per timestep)**: AdEx + conductance synapses update neuron and synapse state.
- **Meso (per learning event)**: three-factor plasticity accumulates eligibility and
  applies neuromodulated updates.
- **Macro (slow control)**: criticality controller adjusts global gain to keep σ near target.
- **Meta (schedule)**: temperature gating switches exploration ↔ consolidation phases.

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
| Governance SSOT policy | P2-8..P2-12 | CLM-0011 |
| Reproducibility process anchors | P2-8..P2-12 | CLM-0012, CLM-0013, CLM-0014 |

The table is intentionally minimal: it enumerates the canonical components and
the authoritative claim IDs that substantiate them. For full equations and
parameterization, consult the cited SPEC sections.

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

## Related Work / Bio-Inspired & Neuromorphic Context (Tier-S)

**⚠️ NON-NORMATIVE SECTION**: The following sources provide contextual inspiration and design vocabulary
for bio-inspired, neuromorphic, and embodied AI research directions. These Tier-S materials are NOT used
to justify equations, parameters, or normative claims in the BN-Syn system specification.

The 2026 landscape of bio-inspired and neuromorphic computing offers diverse perspectives on energy
efficiency, physical embodiment, and biomimetic design principles:

- **SPIE BIMPS 2026** provides context on biologically inspired materials and processes, motivating
  exploration of how material properties can inform computational architectures.
- **3Bs Materials Tech 2026** informs design vocabulary for biomimetic approaches, offering insights
  into how biological structures inspire synthetic systems.
- **BioINSP 2026** explores bioinspired hierarchical structures and biomineralisation-inspired systems,
  providing context for multi-scale organizational principles.
- **ACM Neuromorphic Computing 2026** offers a broader neuromorphic computing landscape, informing
  perspectives on brain-inspired computation beyond the specific neuron models used in BN-Syn.
- **CEC GMU Energy-Efficient AI** provides context on how brainpower inspires energy-efficient AI,
  motivating but not prescribing efficiency considerations.
- **MDPI Bio-Inspired AI Special Issue** explores intersections of generative AI and biomimicry,
  offering perspectives on bio-inspired learning mechanisms.
- **Frontiers Embodied AI** informs thinking about physical and embodied artificial intelligence,
  providing context for embodied computation paradigms.

These sources contribute to the intellectual context and motivation for exploring bio-inspired approaches
but do NOT constitute evidence for any specific parameter choice, equation, or algorithmic decision in
the BN-Syn architecture. For normative references that substantiate the system specification, see the
evidence crosswalk table above and consult `claims/claims.yml` for Tier-A claims.
