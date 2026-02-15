# BN-Syn Documentation Index

This is the **single navigation hub** for all documentation in the BN-Syn repository.
Use this index to find any document, specification, or governance artifact.

---

## Quick Links

| Purpose | Document |
|---------|----------|
| **What is BN-Syn** | <a href="../README.md">README.md</a> |
| **Formal Specification** | [SPEC.md](SPEC.md) |
| **Architecture** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Testing & Coverage (Canonical)** | [TESTING.md](TESTING.md) |
| **Architecture & Invariants** | [ARCHITECTURE_INVARIANTS.md](ARCHITECTURE_INVARIANTS.md) |
| **Traceability Map** | [TRACEABILITY_MAP.md](TRACEABILITY_MAP.md) |
| **Reproducibility** | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| **Governance** | [GOVERNANCE.md](GOVERNANCE.md) |
| **SSOT Policy** | [SSOT.md](SSOT.md) |
| **Documentation Formalization** | [DOCUMENTATION_FORMALIZATION.md](DOCUMENTATION_FORMALIZATION.md) |
| **Inventory** | [INVENTORY.md](INVENTORY.md) |
| **CLI Reference** | [CLI_REFERENCE.md](CLI_REFERENCE.md) |
| **Configuration Reference** | [CONFIGURATION.md](CONFIGURATION.md) |
| **Change Management** | [CHANGE_MANAGEMENT.md](CHANGE_MANAGEMENT.md) |
| **Decision Log (ADR Index)** | [DECISIONS.md](DECISIONS.md) |
| **Release Readiness** | [RELEASE_READINESS.md](RELEASE_READINESS.md) |
| **Demo Runbook** | [DEMO.md](DEMO.md) |
| **Release Notes** | [RELEASE_NOTES.md](RELEASE_NOTES.md) |
| **Experimental Hypothesis** | [HYPOTHESIS.md](HYPOTHESIS.md) (NON-NORMATIVE) |
| **Troubleshooting** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| **Contributing** | <a href="../CONTRIBUTING.md">CONTRIBUTING.md</a> |

---

## Core Specification & Architecture

| Document | Purpose |
|----------|---------|
| [SPEC.md](SPEC.md) | 12-component formal specification: equations, calibration, failure envelopes |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture ↔ evidence crosswalk (component → claim mapping) |
| [ARCHITECTURE_INVARIANTS.md](ARCHITECTURE_INVARIANTS.md) | Runtime invariants, boundaries, and failure modes |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Determinism protocol, environment setup, test commands |
| [TESTING.md](TESTING.md) | Canonical test + coverage commands and artifacts |
| [TRACEABILITY_MAP.md](TRACEABILITY_MAP.md) | Requirement/constraint to code, tests, and CI gates |

---

## Governance & Traceability

| Document | Purpose |
|----------|---------|
| [GOVERNANCE.md](GOVERNANCE.md) | One-page governance entry linking all policies |
| [SSOT.md](SSOT.md) | Single-source-of-truth policy summary |
| [SSOT_RULES.md](SSOT_RULES.md) | Authoritative machine-readable rule registry |
| [NORMATIVE_LABELING.md](NORMATIVE_LABELING.md) | Normative vs non-normative labeling policy |
| [CONSTITUTIONAL_AUDIT.md](CONSTITUTIONAL_AUDIT.md) | Constitutional constraints for claims |
| [INVENTORY.md](INVENTORY.md) | Governed path inventory |

---

## Project and Community Docs

| Document | Purpose |
|----------|---------|
| <a href="../CODE_OF_CONDUCT.md">CODE_OF_CONDUCT.md</a> | Collaboration behavior baseline |
| <a href="../MAINTAINERS.md">MAINTAINERS.md</a> | Maintainer role and operations links |
| <a href="../SUPPORT.md">SUPPORT.md</a> | Support and reporting entry points |
| <a href="../ROADMAP.md">ROADMAP.md</a> | Summary of currently tracked roadmap items |

---

## Safety Engineering

| Document | Purpose |
|----------|---------|
| [safety/stpa.md](safety/stpa.md) | System-theoretic process analysis (losses, hazards, constraints) |
| [safety/hazard_log.yml](safety/hazard_log.yml) | Machine-readable hazard log |
| [safety/traceability.yml](safety/traceability.yml) | Requirement → hazard → constraint → test traceability |
| [safety/followups.md](safety/followups.md) | Safety follow-up actions |

---

## Evidence & Bibliography

| Artifact | Purpose |
|----------|---------|
| [bibliography/bnsyn.bib](../bibliography/bnsyn.bib) | BibTeX entries for all cited sources |
| [bibliography/mapping.yml](../bibliography/mapping.yml) | Claim ID → bibkey mappings |
| [bibliography/sources.lock](../bibliography/sources.lock) | Deterministic hash lock for sources |
| [claims/claims.yml](../claims/claims.yml) | Evidence ledger with claim definitions |
| [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) | Human-readable bibliography summary |
| [EVIDENCE_COVERAGE.md](EVIDENCE_COVERAGE.md) | Evidence coverage table |

---

## Extension Modules

| Document | Purpose |
|----------|---------|
| [VCG.md](VCG.md) | Verified Contribution Gating — optional governance extension |
| [CRITICALITY_CONTROL_VS_MEASUREMENT.md](CRITICALITY_CONTROL_VS_MEASUREMENT.md) | Runtime proxy vs offline measurement |

---

## Audit & Findings

| Document | Purpose |
|----------|---------|
| [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) | Audit findings and error ledger |
| [appendix/EXECUTIVE_SUMMARY.md](appendix/EXECUTIVE_SUMMARY.md) | Imported executive summary (NON-NORMATIVE) |
| [appendix/PRODUCTION_AUDIT.md](appendix/PRODUCTION_AUDIT.md) | Imported production audit (NON-NORMATIVE) |
| [appendix/PRODUCTION_ROADMAP.md](appendix/PRODUCTION_ROADMAP.md) | Imported roadmap (NON-NORMATIVE) |

---

## Implementation & Tests

| Path | Purpose |
|------|---------|
| `src/bnsyn/` | Reference implementation |
| `tests/` | Smoke tests (`pytest -m "not validation"`) |
| `tests/validation/` | Validation tests (`pytest -m validation`) |
| `scripts/` | SSOT validators and maintenance scripts |

---

## CI & Workflows

| Workflow | Purpose |
|----------|---------|
| [ci-smoke.yml](../.github/workflows/ci-smoke.yml) | SSOT gates + smoke tests (every PR) |
| [ci-validation.yml](../.github/workflows/ci-validation.yml) | Validation tests (weekly + manual) |

---

## Navigation Principles

1. **Single source**: Each concept has exactly one authoritative document.
2. **Links over duplication**: Documents link to sources rather than duplicating content.
3. **Claim traceability**: Normative statements bind to identifiers such as `CLM-0001` in `claims/claims.yml`.
4. **Governed vs non-governed**: See [INVENTORY.md](INVENTORY.md) for path classification.
