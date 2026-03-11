# Pytest Failure Diagnostics

## Source of truth
Pytest remains the only pass/fail source of truth. The diagnostics subsystem is post-run and additive.

## Artifact contract
The diagnostics flow writes:
- `artifacts/tests/failure-diagnostics.json`
- `artifacts/tests/failure-diagnostics.md`

The JSON payload is validated against:
- `schemas/pytest-failure-diagnostics.schema.json`

## Local flow
`make test-diagnostics` runs `scripts/run_pytest_with_diagnostics.py`, which:
1. runs pytest with JUnit output
2. tees pytest output to a log file
3. always runs diagnostics generation
4. exits with the original pytest exit code

## CI flow
Reusable pytest workflow executes diagnostics with `if: always()` and uploads diagnostics artifacts on both success and failure.

## Redaction policy
Redaction applies only to published excerpts (`message`, traceback/log excerpts, stdout/stderr snippets). Raw JUnit and logs are not mutated.
Patterns include:
- `ghp_...`
- `github_pat_...`
- `Bearer ...`
- long hex/key-like blobs

This is bounded, deterministic masking, not perfect secret detection.

## Limitations
- JUnit is primary; log parsing is secondary enrichment.
- Only top-N annotations are emitted when enabled.
- No timestamps or host metadata are added, for deterministic outputs.

## LLM diagnosis workflow
Copy `artifacts/tests/failure-diagnostics.md` into an LLM prompt when triaging failures. It includes normalized failure reasons and per-test reproduce commands.
