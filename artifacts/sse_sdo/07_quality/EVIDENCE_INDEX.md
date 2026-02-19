# EVIDENCE_INDEX

## G1 POLICY_SCHEMA_STRICT
- cmd: `python scripts/sse_policy_load.py`
- §REF:blob:artifacts/sse_sdo/logs/g1.log#a12b7cb43c9d9134b5bb1b35e9096b66775d9e92e7611d1cc92b02edd6782a87

## G2 LAW_POLICE_PRESENT
- cmd: `python -m pytest -q tests/test_policy_schema_contract.py tests/test_policy_to_execution_contract.py tests/test_required_checks_contract.py tests/test_ssot_alignment_contract.py tests/test_workflow_integrity_contract.py`
- §REF:blob:artifacts/sse_sdo/logs/g2.log#83a1593c5b5783693fe3006d1f6597bcb4f3a5968f23b59481ee09a0477f910a

## G3 SSOT_ALIGNMENT
- cmd: `python scripts/sse_drift_check.py`
- §REF:blob:artifacts/sse_sdo/logs/g3.log#a12b7cb43c9d9134b5bb1b35e9096b66775d9e92e7611d1cc92b02edd6782a87

## G4 WORKFLOW_INTEGRITY
- cmd: `python -m pytest -q tests/test_workflow_integrity_contract.py`
- §REF:blob:artifacts/sse_sdo/logs/g4.log#423b1d0e014eb1eab96f4420f7b344c2615be505dd574b756ac884826ca74f2d
