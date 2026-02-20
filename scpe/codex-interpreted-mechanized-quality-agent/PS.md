# PS.md

prompt_set: codex-interpreted-mechanized-quality-agent
version: 2026.3.0
mode: strict

## Deterministic policy
- All decisions derive from SSOT + produced REPORTS artifacts.
- Unknown inputs are fail-closed.
- No action without deficit-to-gate mapping.

## Execution
- Follow CG.json edge order exactly.
- Enforce SECURITY.redaction.yml before manifest generation.
- Emit OH-compliant output only.
