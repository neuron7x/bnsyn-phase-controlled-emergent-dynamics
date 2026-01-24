# Architecture

This document binds the BN-Syn architecture to the formal specification (`docs/SPEC.md`)
and the SSOT evidence registry (`bibliography/*`, `claims/claims.yml`). It is a
traceability index, not an independent source of truth. The authoritative definition
of behavior remains the specification, while evidence provenance is governed by SSOT.

**Navigation**: [INDEX.md](INDEX.md) | [GOVERNANCE.md](GOVERNANCE.md) | [SPEC.md](SPEC.md)

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

## Cold Emergence Layer

BN-Syn implements **cold emergence**: deterministic, information-driven organization
without affective/motivational modulation. Cold emergence naturally arises from the
multi-scale architecture and is controlled through thermodynamic phase transitions:

- **Attractor control**: Phase control via temperature and criticality (σ) stabilizes
  deterministic attractors (Lyapunov exp < 0).
- **Information integration**: System exhibits high integrated information (Φ) when
  in cold phase, indicating non-reducible emergent organization.
- **Functional systems**: Anokhin-style afferent synthesis + acceptor-of-result operate
  without dopamine/reward, using purely informational error signals.
- **Validation**: Formal verification confirms cold emergence through determinism,
  integration, and synergistic (non-redundant) information processing.

See [`docs/COLD_EMERGENCE.md`](COLD_EMERGENCE.md) for detailed theoretical foundation
and implementation details.

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
| Cold emergence mechanisms | (extension) | CLM-0020, CLM-0021 |
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
