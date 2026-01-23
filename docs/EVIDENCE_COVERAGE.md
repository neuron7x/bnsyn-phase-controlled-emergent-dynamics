# Evidence Coverage

| Claim ID | Tier | Normative | Bibkey | DOI | Spec Section | Implementation Paths | Verification Paths |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLM-0001 | Tier-A | true | brette2005adaptive | 10.1152/jn.00686.2005 | P0-1 AdEx neuron model | `src/bnsyn/neuron/adex.py`, `src/bnsyn/sim/network.py` | `tests/test_adex_smoke.py`, `tests/test_network_smoke.py` |
| CLM-0002 | Tier-A | true | brette2005adaptive | 10.1152/jn.00686.2005 | P0-1 AdEx model | `src/bnsyn/neuron/adex.py`, `src/bnsyn/sim/network.py` | `tests/test_adex_smoke.py`, `tests/test_network_smoke.py` |
| CLM-0003 | Tier-A | true | jahr1990voltage | 10.1523/JNEUROSCI.10-09-03178.1990 | P0-1 NMDA block | `src/bnsyn/synapse/conductance.py` | `tests/test_synapse_smoke.py` |
| CLM-0004 | Tier-A | true | fremaux2016neuromodulated | 10.3389/fncir.2015.00085 | P0-2 Three-factor learning | `src/bnsyn/plasticity/three_factor.py` | `tests/test_plasticity_smoke.py` |
| CLM-0005 | Tier-A | true | izhikevich2007solving | 10.1093/cercor/bhl152 | P0-2 Neuromodulated STDP | `src/bnsyn/plasticity/three_factor.py` | `tests/test_plasticity_smoke.py` |
| CLM-0006 | Tier-A | true | beggs2003neuronal | 10.1523/JNEUROSCI.23-35-11167.2003 | P0-3 Avalanche exponents | `src/bnsyn/criticality/branching.py` | `tests/test_criticality_smoke.py` |
| CLM-0007 | Tier-A | true | beggs2003neuronal | 10.1523/JNEUROSCI.23-35-11167.2003 | P0-3 Branching parameter σ | `src/bnsyn/criticality/branching.py` | `tests/test_criticality_smoke.py` |
| CLM-0008 | Tier-A | true | wilting2018inferring | 10.1038/s41467-018-04725-4 | P0-3 MR estimator | `src/bnsyn/criticality/analysis.py` | `tests/validation/test_criticality_validation.py` |
| CLM-0009 | Tier-A | true | clauset2009power | 10.1137/070710111 | P0-3 Power-law fitting | `src/bnsyn/criticality/analysis.py` | `tests/validation/test_criticality_validation.py` |
| CLM-0010 | Tier-A | true | frey1997synaptic | 10.1038/385533a0 | P1-5 Synaptic tagging | `src/bnsyn/consolidation/dual_weight.py` | `tests/test_consolidation_smoke.py` |
| CLM-0019 | Tier-A | true | kirkpatrick1983annealing | 10.1126/science.220.4598.671 | P1-5 Temperature schedule | `src/bnsyn/temperature/schedule.py` | `tests/test_temperature_smoke.py` |
| CLM-0011 | Tier-A | true | wilkinson2016fair | 10.1038/sdata.2016.18 | P2-8..12 FAIR principles | `scripts/validate_bibliography.py`, `scripts/validate_claims.py`, `scripts/scan_normative_tags.py` | `scripts/validate_bibliography.py`, `scripts/validate_claims.py`, `scripts/scan_normative_tags.py` |
| CLM-0012 | Tier-S | false | neurips2026checklist | NODOI | P2-8..12 Reproducibility checklist | `scripts/validate_claims.py` | `scripts/validate_claims.py` |
| CLM-0013 | Tier-S | false | acm2020badges | NODOI | P2-8..12 Artifact badges | `scripts/validate_claims.py` | `scripts/validate_claims.py` |
| CLM-0014 | Tier-S | false | pytorch2026randomness | NODOI | P2-8..12 Determinism docs | `scripts/validate_claims.py` | `scripts/validate_claims.py` |
| CLM-0015 | Tier-A | true | trivers1971reciprocal | 10.1086/406755 | GOV-1 Result-based reciprocity (foundation) | `src/bnsyn/vcg.py` | `tests/test_vcg_smoke.py` |
| CLM-0016 | Tier-A | true | axelrod1981cooperation | 10.1126/science.7466396 | GOV-1 Symmetric reciprocity (tit-for-tat) | `src/bnsyn/vcg.py` | `tests/test_vcg_smoke.py` |
| CLM-0017 | Tier-A | true | nowak1998imagescoring | 10.1038/31225 | GOV-1 Reputation / indirect reciprocity | `src/bnsyn/vcg.py` | `tests/test_vcg_smoke.py` |
| CLM-0018 | Tier-A | true | fehr2002punishment | 10.1038/415137a | GOV-1 Costly sanctioning / defector suppression | `src/bnsyn/vcg.py` | `tests/test_vcg_smoke.py` |
