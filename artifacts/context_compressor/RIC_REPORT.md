# Recursive Integrity Check (PASS 0)

- Scope: commands truth, policy truth, code truth, state truth.
- Commit: `35bd4e8aa77e343e7c37470fdbd5a54bb38f6461`.

## Contradictions
- Total: 1

### §RIS:contradiction:test-marker-contract#0cb50848744f45fb
- Severity: P0
- Type: command_mismatch
- Summary: AGENTS.md test command excludes only validation, while Makefile/docs exclude both validation and property.
- Evidence:
  - `file:AGENTS.md:L30-L33`
  - `file:Makefile:L8-L8`
  - `file:docs/TESTING.md:L21-L25`
- Sync Patch Plan: Compressor artifacts treat Makefile+docs/TESTING as runtime truth; AGENTS mismatch recorded for human sync patch.

## Determinism
- Stable IDs computed from `TYPE:STABLE_KEY` using sha256 first 64 bits.
- Node/edge ordering sorted lexicographically by ID.

## PASS 6 Proofs
- KG.json sha256: `eb69220432b9741d5b3cb69a816a7d67d9bba8791908e691c3940abe9e9ac123`
- Missing evidence: `0`
- Broken refs: `0`
- Contradictions: `1`
- Verdict: `PASS`
