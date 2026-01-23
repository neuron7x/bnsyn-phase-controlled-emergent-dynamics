# /docs/SSOT.md
## Single-Source-of-Truth Policy for BN-Syn

This document is a human summary of SSOT governance. The authoritative, machine-readable
rule registry lives in `docs/SSOT_RULES.md` and is enforced by the validators in `scripts/`.

**Navigation**: [INDEX.md](INDEX.md) | [GOVERNANCE.md](GOVERNANCE.md) | [SSOT_RULES.md](SSOT_RULES.md)

## Authority chain

1) **Rules**: `docs/SSOT_RULES.md` defines the normative constraints.
2) **Data**: `bibliography/` and `claims/` instantiate those constraints.
3) **Validators**: `scripts/validate_bibliography.py`, `scripts/validate_claims.py`, and
   `scripts/scan_normative_tags.py` enforce rule adherence.
4) **Documentation**: this document provides a concise narrative overview.

## Summary highlights (non-exhaustive)

- Tier enums use canonical labels: `Tier-A`, `Tier-S`, `Tier-B`, `Tier-C`.
- Tier-A claims are normative; Tier-S/B/C claims are non-normative.
- Bibliography mappings align claims to bibkeys and spec sections.
- `sources.lock` carries deterministic hashes for all bibkeys.
- Validators enforce SSOT rules defined in `docs/SSOT_RULES.md`.

## Reference examples (Tier-S and Tier-A)

- NeurIPS checklist (Tier-S), ACM badges (Tier-S), FAIR (Tier-A `wilkinson2016fair`).
