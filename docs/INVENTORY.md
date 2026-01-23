# Repository Inventory (SSOT Scope)

This inventory defines governed paths and their SSOT alignment. Governed documents are
scanned for normative signals by `scripts/scan_normative_tags.py`. It is intentionally
explicit to keep audit surfaces bounded and to make rule coverage verifiable.

## Governed documents
- `docs/ARCHITECTURE.md`
- `docs/BIBLIOGRAPHY.md`
- `docs/CONSTITUTIONAL_AUDIT.md`
- `docs/CRITICALITY_CONTROL_VS_MEASUREMENT.md`
- `docs/EVIDENCE_COVERAGE.md`
- `docs/INVENTORY.md`
- `docs/NORMATIVE_LABELING.md`
- `docs/REPRODUCIBILITY.md`
- `docs/SPEC.md`
- `docs/SSOT.md`
- `docs/SSOT_RULES.md`
- `docs/VCG.md`

## Repository sections

| Path | Purpose | Governed by |
| --- | --- | --- |
| `docs/` | Formal specification, architecture, reproducibility, and governance docs. | `docs/SSOT_RULES.md`, `docs/SSOT.md` |
| `bibliography/` | SSOT bibliography (`bnsyn.bib`), mappings, lockfile. | `docs/SSOT_RULES.md` |
| `claims/` | Evidence ledger for claims (`claims.yml`). | `docs/SSOT_RULES.md`, `docs/NORMATIVE_LABELING.md` |
| `scripts/` | SSOT validators (`validate_bibliography.py`, `validate_claims.py`, `scan_normative_tags.py`). | `docs/SSOT_RULES.md` |
| `.github/workflows/` | CI gates for SSOT + tests. | `docs/SSOT_RULES.md`, `docs/REPRODUCIBILITY.md` |
| `src/` | Reference implementation (`bnsyn`). | `docs/SPEC.md` |
| `tests/` | Smoke + validation tests. | `docs/SPEC.md`, `docs/REPRODUCIBILITY.md` |

## Maintenance expectations

- Keep the governed documents list aligned with additions/removals in `docs/`.
- Ensure any new SSOT-relevant directory is reflected in the repository sections table.
