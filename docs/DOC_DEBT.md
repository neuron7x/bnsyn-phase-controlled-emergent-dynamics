# Documentation Debt

- GAP: No unified statement for stable/experimental/internal compatibility before this update.
  - ACTION: maintain `docs/API_STABILITY.md` as SSOT-facing summary.
- GAP: No deterministic generator for `docs/PROJECT_SURFACES.md` before this update.
  - ACTION: run `python -m scripts.discover_public_surfaces` when public surfaces change.
- GAP: No machine validator for traceability table before this update.
  - ACTION: enforce `python -m scripts.validate_traceability` in local and CI gates.
