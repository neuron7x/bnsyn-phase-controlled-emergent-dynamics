# 100% VOLUME & EVIDENCE REPORT (BN-SYN · 28-DAY WINDOW)
**Execution Date:** 2026-02-20
**Global Readiness:** 75.00%
**Status:** BLOCKED
**Contradictions Detected:** 0

## 1. GITHUB TOOLING EVIDENCE (DOMAIN A)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| CI Workflows (Last 5) | `gh` installed, but unauthenticated (`gh auth status` not logged in) | FAIL | `proof_bundle/logs/127_gh_auth_status_stderr.log`, `proof_bundle/logs/129_gh_run_list_stderr.log` |
| Dependabot Alerts | API call blocked by missing auth token/session | FAIL | `proof_bundle/logs/80_generate_security_scan.log`, `artifacts/security/security_scan.json` |
| CodeQL/SAST Alerts | API call blocked by missing auth token/session | FAIL | `proof_bundle/logs/80_generate_security_scan.log`, `artifacts/security/security_scan.json` |
| SSOT Branch Prot. | branch protection not readable without authenticated GitHub API access | FAIL | `proof_bundle/logs/128_gh_repo_view_stderr.log` |

## 2. TEST & CODEBASE EVIDENCE (DOMAIN B)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| Total Tests Run | 1025 | INFO | `proof_bundle/logs/82_coverage_and_junit_summary.log` |
| Tests Passed/Failed | Passed: 1019, Failed: 0, Skipped: 6 | PASS | `proof_bundle/logs/81_pytest_protocol_full.log`, `proof_bundle/logs/82_coverage_and_junit_summary.log` |
| Code Coverage (%) | 99% | PASS | `proof_bundle/logs/81_pytest_protocol_full.log`, `proof_bundle/logs/82_coverage_and_junit_summary.log` |
| Mypy Strict Errors | 0 (`--strict`, `--strict --ignore-missing-imports`) | PASS | `proof_bundle/logs/113_mypy_strict_final4.log`, `proof_bundle/logs/112_mypy_ignore_final3.log` |
| Ruff Violations | 0 | PASS | `proof_bundle/logs/119_ruff_required_final.log` |

## 3. FORMAL & META-PROTOCOL EVIDENCE (DOMAIN C)
| Protocol / Spec | File / Path | Status | sha256 Signature |
|-----------------|-------------|--------|------------------|
| GHTPO-2026.02 | `GHTPO-2026.02.md` | VERIFIED | `manifest/repo_manifest.json` |
| ELPEG-2026.04 | `ELPEG-2026.04.md` | VERIFIED | `manifest/repo_manifest.json` |
| BIO-DIGITAL-S12 | `BIO-DIGITAL-S12.yaml` | VERIFIED | `manifest/repo_manifest.json` |
| TLA+ / Coq Proofs| `specs/tla/BNsyn.tla`, `specs/coq/BNsyn_Sigma.v` | VERIFIED | `manifest/repo_manifest.json` |

## 4. PRODUCT UX & BLOCKERS (DOMAIN D)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| CLI Error Path | invalid input returns clean deterministic error, exit code 2, no traceback | PASS | `proof_bundle/logs/94_cli_invalid_gate.log`, `proof_bundle/logs/92_new_cli_exception_test.log` |
| Evidence Bundles | security + governance evidence bundles generated | PASS | `artifacts/security/security_scan.json`, `manifest/repo_manifest.json` |

## 5. EXECUTIVE VOLUME SUMMARY (28 DAYS)
- **Total Commits Analyzed:** 911
- **Total Synthetic Objects Integrated:** 1456
- **Lines of Code / Tests Written:** Added 402008, Deleted 50811, Total Changed 452819
- **P0 BLOCKERS:**
  1. GitHub API telemetry inaccessible without authenticated GitHub session/token.

**END OF REPORT.**
