# GAP TABLE — CI_BATTLE_USAGE_AUDIT_2026-02-07

| gap_id | description | risk | blocking |
|---|---|---|---|
| GAP-001 | Local gate failure: `ruff format --check .` was non-zero in audit evidence. | CI/local determinism mismatch; merge risk. | yes |
| GAP-002 | Local gate failure: `mypy src --strict --config-file pyproject.toml` was non-zero in audit evidence. | Type-safety gate not reproducible locally. | yes |
| GAP-003 | Snapshot source mismatch: requested `/mnt/data/bnsyn-phase-controlled-emergent-dynamics-main.zip` absent; run executed from workspace checkout. | Reproducibility provenance ambiguity. | yes |
| GAP-004 | Battle usage verdict was `NOT_PROVEN` without permanent anti-overclaim guard. | External readiness ambiguity; governance risk. | yes |
| GAP-005 | No PR-specific fresh CI run URL captured for current branch revision. | Cannot prove branch-level CI convergence from immutable run artifact. | yes |
