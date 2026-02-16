     1	# Architecture
     2	
     3	This document binds the BN-Syn architecture to the formal specification (`docs/SPEC.md`)
     4	and the SSOT evidence registry (`bibliography/*`, `claims/claims.yml`). It is a
     5	traceability index, not an independent source of truth. The authoritative definition
     6	of behavior remains the specification, while evidence provenance is governed by SSOT.
     7	
     8	**Navigation**: [INDEX.md](INDEX.md) | [GOVERNANCE.md](GOVERNANCE.md) | [SPEC.md](SPEC.md)
     9	
    10	## Scope and traceability
    11	
    12	- **Spec authority**: equations, invariants, and parameters are defined in `docs/SPEC.md`.
    13	- **Evidence authority**: claim IDs and bibliographic provenance are defined in
    14	  `claims/claims.yml` and `bibliography/*`.
    15	- **Purpose**: expose the mapping between architecture components and their formal
    16	  references without introducing new semantics.
    17	- **Audit artifacts**: component mappings live in `docs/spec_to_code.yml` and the
    18	  verification table is tracked in `docs/COMPONENT_AUDIT.md`.
    19	
    20	## Control loops (timescale-segregated)
    21	
    22	- **Micro (per timestep)**: AdEx + conductance synapses update neuron and synapse state.
    23	- **Meso (per learning event)**: three-factor plasticity accumulates eligibility and
    24	  applies neuromodulated updates.
    25	- **Macro (slow control)**: criticality controller adjusts global gain to keep σ near target.
    26	- **Meta (schedule)**: temperature gating switches exploration ↔ consolidation phases.
    27	
    28	## Evidence crosswalk (core components)
    29	
    30	| Component | SPEC section | Claim IDs |
    31	| --- | --- | --- |
    32	| AdEx neuron dynamics | P0-1 | CLM-0001, CLM-0002 |
    33	| Conductance synapses + NMDA Mg²⁺ block | P0-2 | CLM-0003 |
    34	| Three-factor learning | P0-3 | CLM-0004 |
    35	| Neuromodulated STDP + eligibility traces | P0-3 | CLM-0005 |
    36	| Criticality σ tracking | P0-4 | CLM-0006, CLM-0007 |
    37	| Subsampling-corrected σ estimation | P0-4 | CLM-0008 |
    38	| Power-law validation for avalanches | P0-4 | CLM-0009 |
    39	| Temperature schedule + gating | P1-5 | CLM-0019 |
    40	| Dual-weight consolidation | P1-6 | CLM-0010 |
    41	| Governance SSOT policy | P2-8..P2-12 | CLM-0011 |
    42	| Reproducibility process anchors | P2-8..P2-12 | CLM-0012, CLM-0013, CLM-0014 |
    43	
    44	The table is intentionally minimal: it enumerates the canonical components and
    45	the authoritative claim IDs that substantiate them. For full equations and
    46	parameterization, consult the cited SPEC sections.
    47	
    48	## State layout (reference simulator)
    49	
    50	- neuron state: (V, w, spiked)
    51	- synapse state: (g_ampa, g_nmda, g_gabaa)
    52	- criticality: (σ estimator, gain controller)
    53	
    54	## Optional governance extension: Verified Contribution Gating (VCG)
    55	
    56	VCG is an **engineering-contract** module that enforces result-based reciprocity at the system boundary:
    57	agents that repeatedly match a social/interaction pattern without measurable contribution are
    58	deprioritized by withdrawing future routing priority and resource budget. VCG is specified
    59	as a non-core extension (does not change the 12 core neurodynamics components) and is documented
    60	in `docs/VCG.md`.
    61	
    62	## Related Work / Bio-Inspired & Neuromorphic Context (Tier-S)
    63	
    64	**⚠️ NON-NORMATIVE SECTION**: The following sources provide contextual inspiration and design vocabulary
    65	for bio-inspired, neuromorphic, and embodied AI research directions. These Tier-S materials are NOT used
    66	to justify equations, parameters, or normative claims in the BN-Syn system specification.
    67	
    68	The 2026 landscape of bio-inspired and neuromorphic computing offers diverse perspectives on energy
    69	efficiency, physical embodiment, and biomimetic design principles:
    70	
    71	- **SPIE BIMPS 2026** provides context on biologically inspired materials and processes, motivating
    72	  exploration of how material properties can inform computational architectures.
    73	- **3Bs Materials Tech 2026** informs design vocabulary for biomimetic approaches, offering insights
    74	  into how biological structures inspire synthetic systems.
    75	- **BioINSP 2026** explores bioinspired hierarchical structures and biomineralisation-inspired systems,
    76	  providing context for multi-scale organizational principles.
    77	- **ACM Neuromorphic Computing 2026** offers a broader neuromorphic computing landscape, informing
    78	  perspectives on brain-inspired computation beyond the specific neuron models used in BN-Syn.
    79	- **CEC GMU Energy-Efficient AI** provides context on how brainpower inspires energy-efficient AI,
    80	  motivating but not prescribing efficiency considerations.
    81	- **MDPI Bio-Inspired AI Special Issue** explores intersections of generative AI and biomimicry,
    82	  offering perspectives on bio-inspired learning mechanisms.
    83	- **Frontiers Embodied AI** informs thinking about physical and embodied artificial intelligence,
    84	  providing context for embodied computation paradigms.
    85	
    86	These sources contribute to the intellectual context and motivation for exploring bio-inspired approaches
    87	but do NOT constitute evidence for any specific parameter choice, equation, or algorithmic decision in
    88	the BN-Syn architecture. For normative references that substantiate the system specification, see the
    89	evidence crosswalk table above and consult `claims/claims.yml` for Tier-A claims.
