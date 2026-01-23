# /docs/SSOT.md
## Single-Source-of-Truth Policy for BN-Syn

Rule 1: Every normative scientific claim references a Tier-A bibkey from `bibliography/bnsyn.bib` (peer-reviewed + DOI).

Rule 2: Process/compliance rules may cite Tier-S (standards/docs, no DOI required).

Rule 3: Tier-C (preprints/arXiv/blogs) are non-normative only and MUST NOT appear in normative sections.

Rule 4: `bibliography/sources.lock` is immutable and MUST contain valid 64-hex SHA256 computed over the declared LOCK_STRING.

Rule 5: `bibliography/mapping.yml` covers all CLM-IDs. Any normative-scientific CLM must map to Tier-A.

Rule 6: PR fails if:
- any mapped bibkey is missing from `bnsyn.bib`, or
- any Tier-A mapping lacks DOI in `bnsyn.bib`, or
- any sources.lock line has invalid SHA256, or
- any CLM entry is malformed.

References: NeurIPS checklist (Tier-S), ACM badges (Tier-S), FAIR (Tier-A `wilkinson2016fair`).
