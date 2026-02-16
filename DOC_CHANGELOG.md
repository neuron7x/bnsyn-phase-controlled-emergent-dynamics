# Documentation Change Log

**Statement:** Documentation-only; no logic changes.

## Changes Delivered
- Added a repository routing map with directory purposes, stability labels, and "where to find X" lookup table (`docs/repo_map.md`).
- Added command-precise operational workflows for setup, validation, running flows, artifact generation, and docs builds (`docs/usage_workflows.md`).
- Reworked onboarding block in `README.md` to include mission/scope/non-goals and direct links to new navigation pages.
- Added full scripts formalization:
  - `scripts/README.md` inventory table for every file under `scripts/`.
  - `docs/scripts/index.md` plus one per-script page in `docs/scripts/*.md`.
- Added contracts conceptual documentation in `docs/contracts/index.md`.
- Expanded contracts API exposure by adding `contracts` modules to `docs/modules.rst` and cross-links from `docs/api/index.md`.
- Added/strengthened docstrings for public contract assertions in `src/contracts/math_contracts.py` (doc-only).
- Added `proof_bundle/` artifacts: repo state, command log, doc inventory, and SHA256 hash manifest.

## Meta-Audit Self-Validation

### Coverage completeness
- Scripts documented: **64 / 64** (`scripts/*` files covered by both `scripts/README.md` and `docs/scripts/*.md`).
- Contracts API documentation: `contracts` and `contracts.math_contracts` now included in Sphinx module autosummary.

### Evidence density
- All navigation and workflow docs reference concrete repository paths and executable commands.
- Build/test commands are recorded in `proof_bundle/commands.log` with exit codes.

### Contradiction scan
- Existing docs referenced `docs/SCRIPTS/index.md` (uppercase path), while new formalized deliverable required `docs/scripts/index.md`.
  - Resolution: README and root docs index now point to lowercase `docs/scripts/index.md` while preserving existing content elsewhere.

### Blind spots (UNKNOWN/TBD)
- Some scripts still lack module docstrings and expose limited static output-path evidence; those pages are marked `UNKNOWN/TBD` and recommend direct source inspection.
- No behavioral assertions were added beyond existing code semantics; this run did not execute every individual script workload.

## Build/Verification Notes
- `make docs` executed successfully in this run (see `proof_bundle/commands.log`).
- Any future script interface drift should be synchronized by regenerating script docs from source.
