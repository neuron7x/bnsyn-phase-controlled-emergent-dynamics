# Data Provenance — BN-Syn

## artifacts/math_audit/manifest.json
- Generator: `python scripts/generate_math_data.py`
- Seed: N/A (deterministic filesystem walk)
- Dependencies: repository tracked files in `results/`, `benchmarks/`, `docs/`, `src/`, root config files
- Reproduction: `python scripts/generate_math_data.py`
- Checksum (SHA256): `23f7c40a3252dde276d38584b4ee723540ee6b1cbca82f2c3b40fab328f1d209`

## artifacts/math_audit/validator_report.json
- Generator: `python scripts/math_validate.py`
- Seed: N/A (deterministic contract execution)
- Dependencies: `artifacts/math_audit/manifest.json`, contract set in `src/contracts/math_contracts.py`
- Reproduction: `python scripts/math_validate.py`
- Checksum (SHA256): `eef6021d202b7ca6f2940d859af7925fd3edc444ed579d5b52d3a8a518c5f329`

## artifacts/math_audit/validator_report.md
- Generator: `python scripts/math_validate.py`
- Seed: N/A
- Dependencies: validator execution results
- Reproduction: `python scripts/math_validate.py`
- Checksum (SHA256): `c0a9eaaf814cc40181ffadc17719653950d017ea596ed8ac619d27ccc770b1ae`

## artifacts/math_audit/baseline_env.txt
- Generator: phase-0 environment baseline command block
- Seed: N/A
- Dependencies: local Python/pip/runtime platform and installed packages
- Reproduction:
  - `python --version`
  - `python -m pip --version`
  - `python -c "import platform; print(platform.platform())"`
  - `python -m pip freeze`
- Checksum (SHA256): `a719e4b5d759d2017c565db33a99cb2113b228cd3f5efbc9259f0e468ed55105`

## artifacts/math_audit/phase_a_audit.txt
- Generator: Phase A audit command bundle
- Seed: N/A
- Dependencies: repository file inventory and command outputs
- Reproduction: rerun Phase A command block from task specification
- Checksum (SHA256): `cafca403fc233004f4b6a3d0ceadfbd763d55d46c17b0cbcb255a945c86f0dd6`

## artifacts/math_audit/hardened_run.log
- Generator: `python scripts/math_validate.py 2>&1 | tee artifacts/math_audit/hardened_run.log`
- Seed: N/A
- Dependencies: validator runtime output stream
- Reproduction: command above
- Checksum (SHA256): `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
