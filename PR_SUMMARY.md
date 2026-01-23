# PR: Governance extension + Tier-A bibliography expansion (VCG)

## What
Adds an optional, contract-defined governance extension **Verified Contribution Gating (VCG)** and expands the Tier-A bibliography with authoritative reciprocity/cooperation sources.

## Why
- VCG formalizes *result-based reciprocity* as deterministic resource allocation, reducing parasitic pattern-only behavior.
- Bibliography remains SSOT-driven: all normative claims map to Tier-A DOI sources; standards remain Tier-S.

## Files
- NEW: `docs/VCG.md` (module spec + acceptance criteria)
- UPDATED: `bibliography/bnsyn.bib` (+4 Tier-A entries)
- UPDATED: `bibliography/sources.lock` (+4 locked entries; SHA256 over lock string)
- UPDATED: `bibliography/mapping.yml` (+CLM-0015..0018)
- UPDATED: `claims/claims.yml` (adds proven reciprocity claims; resolves prior CLM-0017 collision by renaming unproven benchmark claim to CLM-0099)
- UPDATED: `docs/BIBLIOGRAPHY.md` (lists new Tier-A sources)
- UPDATED: `docs/ARCHITECTURE.md` (mentions optional VCG extension)
- NEW: `.github/workflows/bibliography-gate.yml` (runs `scripts/validate_bibliography.py`)
- UPDATED: `Makefile` (adds `validate-bibliography` + `validate-all`)

## Proof commands
```bash
make validate-all
python scripts/validate_bibliography.py
python scripts/validate_claims.py
```

## Notes
- VCG is explicitly **non-core** and must not affect neuron/synapse dynamics (invariant I4 in `docs/VCG.md`).
