# SSOT Rules (Authoritative)

This document is the machine-readable SSOT rule registry. Validators read this list
and compare against their built-in rule inventory to prevent drift.

```yaml
rules:
  - id: SSR-001
    statement: Tier enum values are limited to Tier-A, Tier-S, Tier-B, Tier-C.
    enforcement: scripts/validate_bibliography.py, scripts/validate_claims.py
    failure_code: SSOT-001
  - id: SSR-002
    statement: Tier-A claims carry normative=true; Tier-S/B/C claims carry normative=false.
    enforcement: scripts/validate_claims.py, scripts/validate_bibliography.py
    failure_code: SSOT-002
  - id: SSR-003
    statement: Mapping entries reference existing bibkeys in bibliography/bnsyn.bib.
    enforcement: scripts/validate_bibliography.py
    failure_code: SSOT-003
  - id: SSR-004
    statement: Claims and mapping form a closed set with aligned tier, bibkey, and spec_section.
    enforcement: scripts/validate_bibliography.py, scripts/validate_claims.py
    failure_code: SSOT-004
  - id: SSR-005
    statement: Tier-A bibkeys include DOI values in bnsyn.bib and appear in sources.lock.
    enforcement: scripts/validate_bibliography.py
    failure_code: SSOT-005
  - id: SSR-006
    statement: Tier-S lock entries use NODOI and include canonical_url and retrieved_date.
    enforcement: scripts/validate_bibliography.py
    failure_code: SSOT-006
  - id: SSR-007
    statement: sources.lock SHA256 matches the computed lock string.
    enforcement: scripts/validate_bibliography.py
    failure_code: SSOT-007
  - id: SSR-008
    statement: Claims include bibkey, spec_section, implementation_paths, and verification_paths fields.
    enforcement: scripts/validate_claims.py
    failure_code: SSOT-008
  - id: SSR-009
    statement: Traceability paths exist and live under src/ or scripts/ (implementation) and tests/ or scripts/ (verification).
    enforcement: scripts/validate_claims.py
    failure_code: SSOT-009
  - id: SSR-010
    statement: Governed docs lines with normative signals include CLM identifiers.
    enforcement: scripts/scan_normative_tags.py
    failure_code: SSOT-010
```
