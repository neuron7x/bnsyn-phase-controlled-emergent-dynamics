{
  "schema": "schemas/document_contract.schema.json",
  "document": "docs/SPEC.md",
  "definition": {
    "terms": []
  },
  "formal_contract": {
    "inputs": [],
    "outputs": [],
    "invariants": []
  },
  "implementation_links": [],
  "tests": [],
  "verification_commands": [],
  "mapping_table": [],
  "drift_guards": [
    "python scripts/validate_document_contracts.py"
  ],
  "data": {
    "components": [
      {
        "id": "P0-1",
        "name": "AdEx neuron (Brette & Gerstner, 2005)"
      },
      {
        "id": "P0-2",
        "name": "Conductance synapses + NMDA Mg\u00b2\u207a block (Jahr\u2013Stevens, 1990)"
      },
      {
        "id": "P0-3",
        "name": "Three-factor learning (Fr\u00e9maux & Gerstner, 2016)"
      },
      {
        "id": "P0-4",
        "name": "Criticality \u03c3 and gain control"
      },
      {
        "id": "P1-5",
        "name": "Temperature schedule + gating"
      },
      {
        "id": "P1-6",
        "name": "Dual-weight consolidation (STC-inspired)"
      },
      {
        "id": "P1-7",
        "name": "Energy regularization objective terms"
      },
      {
        "id": "P2-8",
        "name": "Numerical methods (Euler/RK2/exp decay)"
      },
      {
        "id": "P2-9",
        "name": "Determinism protocol (seed + explicit RNG)"
      },
      {
        "id": "P2-10",
        "name": "Calibration utilities (f\u2013I fit)"
      },
      {
        "id": "P2-11",
        "name": "Reference network simulator (small-N)"
      },
      {
        "id": "P2-12",
        "name": "Bench harness contract (CLI + metrics)"
      }
    ]
  }
}
