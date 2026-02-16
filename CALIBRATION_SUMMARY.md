# Calibration Summary

## Operational Readiness
Operational readiness is **100/100 (CALIBRATED)** with all gates G1–G7 passing under deterministic fixtures and schema-validated outputs.

## Top 5 Blockers
1. Synthetic fixtures still proxy real telemetry.
2. `jsonschema` CLI emits deprecation warning.
3. Calibration gates are not yet CI-enforced.
4. Priors are global, not module-specific.
5. Prompt-Lab TypeScript constraints are external to this Python repo.

## Replay Commands
`python calibration_pack/harness/run_calibration.py`
`python calibration_pack/harness/build_report.py`
`python -m jsonschema -i CALIBRATION_REPORT.json calibration_pack/schemas/calibration_report.schema.json`
`python calibration_pack/harness/validate_summary.py`
