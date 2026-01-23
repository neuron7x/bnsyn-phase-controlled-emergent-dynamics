# Repository Structure Contract

**Navigation**: [INDEX.md](INDEX.md) | [SSOT.md](SSOT.md) | [GOVERNANCE.md](GOVERNANCE.md)

This document defines the authoritative structure of the BN-Syn repository. Validators and CI
pipelines reference this document to enforce structural invariants.

## Top-Level Directories

| Directory       | Purpose                                                       | Stability       |
| --------------- | ------------------------------------------------------------- | --------------- |
| `src/`          | Production source code for BN-Syn library                     | Stable paths    |
| `tests/`        | Automated tests (smoke and validation)                        | Stable paths    |
| `scripts/`      | SSOT validators, audit scripts, and utilities                 | Stable paths    |
| `docs/`         | Governed documentation, specifications, and audit artifacts   | Stable paths    |
| `bibliography/` | BibTeX, mapping, and sources lock files                       | Stable paths    |
| `claims/`       | Claims registry (claims.yml) and related metadata             | Stable paths    |
| `.github/`      | GitHub workflows and CI configuration                         | Stable paths    |

## Governed Paths (Authoritative List)

The following paths are **governed** and subject to SSOT enforcement. Scripts like
`scripts/scan_governed_docs.py` read this list to enforce normative tagging rules.

```yaml
governed_paths:
  docs:
    - docs/SPEC.md
    - docs/ARCHITECTURE.md
    - docs/GOVERNANCE.md
    - docs/SSOT.md
    - docs/SSOT_RULES.md
    - docs/BIBLIOGRAPHY.md
    - docs/REPRODUCIBILITY.md
    - docs/EVIDENCE_COVERAGE.md
    - docs/VCG.md
  readme:
    - README.md
  claims:
    - claims/claims.yml
  bibliography:
    - bibliography/mapping.yml
    - bibliography/bnsyn.bib
    - bibliography/sources.lock
```

## What Goes Where Rules

1. **Source code** (`src/bnsyn/**`): All production Python modules implementing BN-Syn
   neuron models, synapses, plasticity, criticality, temperature scheduling, and VCG.

2. **Tests** (`tests/**`):
   - `tests/*.py`: Smoke tests (fast, critical-path, no `@pytest.mark.validation`)
   - `tests/validation/*.py`: Validation tests (slow, statistical; **must** use `@pytest.mark.validation`)

3. **Scripts** (`scripts/`): SSOT validators, audit generators, and utility scripts.
   - `validate_bibliography.py`: Enforces bibliography SSOT rules
   - `validate_claims.py`: Enforces claim traceability rules
   - `scan_normative_tags.py`: Scans for orphan normative tags
   - `scan_governed_docs.py`: Scans governed docs for normative compliance
   - `rebuild_sources_lock.py`: Regenerates sources.lock deterministically
   - `generate_evidence_coverage.py`: Generates EVIDENCE_COVERAGE.md

4. **Documentation** (`docs/`):
   - Governed docs subject to normative tagging rules
   - Appendix materials (`docs/appendix/`) are informational, not governed

5. **Bibliography** (`bibliography/`):
   - `bnsyn.bib`: BibTeX entries for all cited sources
   - `mapping.yml`: Claim-to-source mapping with bibkey, tier, and section
   - `sources.lock`: Deterministic SHA256-locked source references

6. **Claims** (`claims/`):
   - `claims.yml`: Authoritative claim registry with traceability fields

## Stability Rules

1. **No path renaming** without updating all references in validators, claims, and docs.
2. **No file deletion** in governed paths without explicit deprecation process.
3. **Tier enum values** are immutable: `Tier-A`, `Tier-S`, `Tier-B`, `Tier-C`.
4. **Claim IDs** follow format `CLM-####` and are never reused after deletion.
5. **Test directories** maintain strict separation between smoke (`tests/*.py`) and
   validation (`tests/validation/*.py`) suites.
