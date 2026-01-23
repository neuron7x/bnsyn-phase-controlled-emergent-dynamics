# Normative vs Non-Normative Labeling Policy

## Purpose
Define labeling rules for requirements and claims tied to SSOT evidence.

## Labeling tags
- **[NORMATIVE]**: Required for correctness, reproducibility, or safety gates.
- **[NON-NORMATIVE]**: Illustrative guidance, examples, or optional material.

## Tier rules (SSOT-aligned)
- Tier-A: peer-reviewed sources with DOI; required for all [NORMATIVE] scientific claims.
- Tier-S: standards/docs; non-normative only.
- Tier-B/Tier-C: non-normative only.

## Claim binding
- Any [NORMATIVE] quantitative statement MUST include a `CLM-XXXX` identifier.
- Claim IDs are authoritative in `claims/claims.yml` and must map in `bibliography/mapping.yml`.

## Authority
`docs/SSOT.md` is the single-source-of-truth for tier definitions and validator behavior.
