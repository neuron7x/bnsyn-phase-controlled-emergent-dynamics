# Documentation Gap Ledger

This ledger records missing documentation identified from repository state and the implementation acceptance criteria.

| Path | Purpose | Source of truth | Acceptance |
|---|---|---|---|
| `CODE_OF_CONDUCT.md` | Collaboration behavior baseline for repository interactions. | Existing repository community/security files (`CONTRIBUTING.md`, `SECURITY.md`). | File exists with behavior and reporting sections, linked to repository channels. |
| `MAINTAINERS.md` | Maintainer role and operational references. | Governance and contribution docs (`docs/GOVERNANCE.md`, `CONTRIBUTING.md`, `docs/RELEASE_PIPELINE.md`). | File exists and references current governance/release docs. |
| `SUPPORT.md` | User support entry for issues/discussions/security paths. | Existing user-facing docs (`README.md`, `docs/TROUBLESHOOTING.md`, `SECURITY.md`). | File exists and includes help channels and issue diagnostics guidance. |
| `ROADMAP.md` | Project roadmap summary bound to tracked planning artifacts. | `worklist.json`, `plan.md`. | File exists and cites active priority groups/items from those files. |
| `docs/CLI_REFERENCE.md` | Command and flag reference for the `bnsyn` CLI. | `src/bnsyn/cli.py`, usage examples in `README.md` and `docs/LEGENDARY_QUICKSTART.md`. | File exists with subcommands, argument defaults, and example invocations present in repo docs. |
| `docs/CONFIGURATION.md` | Reference for stable parameter models and defaults. | `src/bnsyn/config.py`, component mapping in `docs/SPEC.md`. | File exists with model fields/defaults and links to SPEC sections. |
| `docs/CHANGE_MANAGEMENT.md` | Documented release/change flow from existing automation. | `docs/RELEASE_PIPELINE.md`, `scripts/release_pipeline.py`, `Makefile`. | File exists describing verify/build/publish-dry-run steps and related commands. |
| `docs/DECISIONS.md` | Decision log index and ADR workflow starter. | Existing governance structure (`docs/GOVERNANCE.md`) and repository docs layout. | File exists with ADR index and link to ADR template only (`ADR-000`). |
| `docs/decisions/ADR-000-template.md` | Reusable ADR template for future decisions. | `docs/DECISIONS.md` and repository documentation conventions. | Template exists with status/date/context/decision/consequences/evidence sections. |
