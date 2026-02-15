## What changed
- Added missing documentation artifacts for project/community, references, and process docs:
  - `CODE_OF_CONDUCT.md`, `MAINTAINERS.md`, `SUPPORT.md`, `ROADMAP.md`
  - `docs/CLI_REFERENCE.md`, `docs/CONFIGURATION.md`
  - `docs/CHANGE_MANAGEMENT.md`, `docs/DECISIONS.md`, `docs/decisions/ADR-000-template.md`
  - `docs/DOCUMENTATION_GAP_LEDGER.md`
- Updated docs navigation:
  - `docs/INDEX.md` links to new docs
  - `docs/index.rst` adds a minimal Project Docs toctree
- Regenerated and committed `INVENTORY.json` after docs additions.

## Why
- CI policy requires PR descriptions to include required sections.
- Previous PR body was a placeholder and failed the `actions/github-script` policy check.
- Inventory gate also required refreshed `INVENTORY.json` after docs file additions.

## Risk
- Low risk.
- Documentation-only changes plus inventory metadata refresh.
- No behavior/API/runtime logic changes.

## Evidence
- Required scans and docs build were executed successfully:
  - `python tools/generate_inventory.py`
  - `git diff --exit-code -- INVENTORY.json`
  - `python -m scripts.scan_placeholders`
  - `python -m scripts.scan_governed_docs`
  - `python -m scripts.scan_normative_tags`
  - `make docs`
- Policy-compliant section headings are present exactly as required.

## How to test
```bash
python tools/generate_inventory.py
git diff --exit-code -- INVENTORY.json
python -m scripts.scan_placeholders
python -m scripts.scan_governed_docs
python -m scripts.scan_normative_tags
make docs
```
