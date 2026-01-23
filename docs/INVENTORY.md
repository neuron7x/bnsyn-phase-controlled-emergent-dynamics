# Repository Inventory (Phase 1)

| Path | Purpose | Governed by |
| --- | --- | --- |
| `docs/` | Formal specification, architecture, reproducibility, and governance docs. | `docs/SSOT.md`, `docs/CONSTITUTIONAL_AUDIT.md` |
| `bibliography/` | SSOT bibliography (`bnsyn.bib`), mappings, lockfile. | `docs/SSOT.md` |
| `claims/` | Evidence ledger for normative claims (`claims.yml`). | `docs/SSOT.md`, `docs/NORMATIVE_LABELING.md` |
| `scripts/` | SSOT validators (`validate_bibliography.py`, `validate_claims.py`). | `docs/SSOT.md` |
| `.github/workflows/` | CI gates for SSOT + tests. | `docs/SSOT.md` |
| `src/` | Reference implementation (`bnsyn`). | `docs/SPEC.md` |
| `tests/` | Smoke + validation tests. | `docs/SPEC.md`, `docs/REPRODUCIBILITY.md` |
