# Battle Readiness Report

Status: PASS

## Versions
- Python: 3.12.12 (`proof_bundle/logs/step7_versions.log`)
- pip: 26.0.1 (`proof_bundle/logs/step7_versions.log`)
- OS: Linux 6.12.47 x86_64 (`proof_bundle/logs/step7_versions.log`)
- Git SHA: e4f6be9ab7a3f5f476034d4feb2a538fb43959d4 (`proof_bundle/logs/step7_versions.log`)

## Scorecard (A–G)
- A) Installability — PASS  
  Evidence: editable install baseline and extras install (`proof_bundle/logs/step6_install_editable_runtime.log`, `proof_bundle/logs/step6_install_test_docs_security.log`, `proof_bundle/logs/step7_hard_acceptance_docs_cmd.log`).
- B) Runtime UX — PASS  
  Evidence: CLI help + deterministic demo output + schema check (`proof_bundle/logs/step6_bnsyn_help.log`, `proof_bundle/logs/step6_bnsyn_demo.log`, `proof_bundle/logs/step1_demo_schema_validate.log`, outputs `proof_bundle/demo_clean.json`).
- C) Test Integrity — PASS  
  Evidence: collect-only + test gate (`proof_bundle/logs/step6_pytest_collect.log`, `proof_bundle/logs/step6_make_test_gate.log`).
- D) Packaging — PASS  
  Evidence: build succeeds and wheel imports in clean venv (`proof_bundle/logs/step6_python_build_fixed.log`, `proof_bundle/logs/step6_wheel_install.log`, `proof_bundle/logs/step6_wheel_import.log`).
- E) Docs Truthfulness — PASS  
  Evidence: hard acceptance command succeeds in clean env (`proof_bundle/logs/step7_hard_acceptance_docs_cmd.log`).
- F) Security Hygiene — PASS  
  Evidence: reproducible security gate with pinned gitleaks bootstrap + pip-audit + bandit (`proof_bundle/logs/step7_hard_acceptance_security_cmd.log`, `proof_bundle/logs/step0_ensure_gitleaks.log`).
- G) CI Truthfulness — PASS  
  Evidence: CI workflow uses `make docs` and `make security` with aligned extras (`proof_bundle/logs/step0_ci_atomic_scan.log`, `.github/workflows/ci-pr-atomic.yml`).

## Blockers
| Priority | Blocker | Status | Minimal fix direction | Proof |
|---|---|---|---|---|
| P0 | `make docs` depended on undeclared local pre-install state | Resolved | Bootstrap docs extra from `make docs`; keep docs extra pinned | `proof_bundle/logs/step1_make_docs_baseline.log`, `proof_bundle/logs/step7_hard_acceptance_docs_cmd.log` |
| P0 | No explicit `security` extra for reproducible local/CI security tooling | Resolved | Add `[security]` extra and align local/CI installs to it | `proof_bundle/logs/step6_install_test_docs_security.log`, `proof_bundle/logs/fix_ci_security_install.log` |
| P1 | `python -m build` absent from clean extra path | Resolved | Pin `build==1.3.0` in `test`/`dev` extras | `proof_bundle/logs/step6_python_build.log`, `proof_bundle/logs/step6_python_build_fixed.log` |

## Canonical Commands
```bash
python -m pip install -e .
python -m bnsyn --help
bnsyn demo --steps 20 --dt-ms 0.1 --seed 123 --N 16
python -m pytest --collect-only -q
make test-gate
python -m pip install -e ".[test,docs]"
make docs
python -m pip install -e ".[security]"
make security
python -m build
```

## Diff Summary
- Added `security` optional dependency group and pinned `build` in `test`/`dev` extras.
- Made `make docs` self-bootstrapping via `.[docs]`; made `make security` install `.[security]`.
- Updated CI security job to install `.[security]` and run module-invoked tools.
- Updated README/SECURITY instructions to include exact reproducible docs/security commands.
- Captured proof bundle logs and snapshots under `proof_bundle/` and mirrored logs under `artifacts/audit/logs/`.

## Proof Bundle
- Logs: `proof_bundle/logs/*.log`
- Mirror logs: `artifacts/audit/logs/*.log`
- Demo outputs: `proof_bundle/demo_step1.json`, `proof_bundle/demo_clean.json`
- Freeze snapshots: `proof_bundle/pre_install_freeze.txt`, `proof_bundle/post_install_freeze.txt`
