# /docs/SSOT.md
## Single-Source-of-Truth Policy for BN-Syn

Rule 1: Tier enum SSOT: `Tier-A`, `Tier-S`, `Tier-B`, `Tier-C` (case-sensitive, no aliases).

Rule 2: Tier-A is peer-reviewed and MUST include DOI.

Rule 3: Tier-S is standards/docs; DOI optional; canonical URL required.

Rule 4: Tier-B/Tier-C are non-normative only and MUST NOT be referenced by [NORMATIVE] claims.

Rule 5: Every normative scientific claim references a Tier-A bibkey from `bibliography/bnsyn.bib`.

Rule 6: `bibliography/sources.lock` is immutable and MUST contain valid 64-hex SHA256 computed over the declared LOCK_STRING.

Rule 7: `bibliography/mapping.yml` covers all CLM-IDs. Any normative-scientific CLM must map to Tier-A.

Rule 8: PR fails if:
- any mapped bibkey is missing from `bnsyn.bib`, or
- any Tier-A mapping lacks DOI in `bnsyn.bib`, or
- any Tier-S mapping lacks canonical URL in `sources.lock`, or
- any sources.lock line has invalid SHA256, or
- any CLM entry is malformed.

References: NeurIPS checklist (Tier-S), ACM badges (Tier-S), FAIR (Tier-A `wilkinson2016fair`).
