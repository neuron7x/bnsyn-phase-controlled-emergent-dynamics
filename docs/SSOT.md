# /docs/SSOT.md
## Single-Source-of-Truth Policy for BN-Syn

This document is a human summary of SSOT governance. The authoritative, machine-readable
rule registry lives in `docs/SSOT_RULES.md`.

### Summary highlights
- Tier enums use the canonical labels: `Tier-A`, `Tier-S`, `Tier-B`, `Tier-C`.
- Tier-A claims are normative; Tier-S/B/C claims are non-normative.
- Bibliography mappings align claims to bibkeys and spec sections.
- `sources.lock` carries deterministic hashes for all bibkeys.
- Validators enforce SSOT rules defined in `docs/SSOT_RULES.md`.

Reference examples (Tier-S and Tier-A):
- NeurIPS checklist (Tier-S), ACM badges (Tier-S), FAIR (Tier-A `wilkinson2016fair`).
