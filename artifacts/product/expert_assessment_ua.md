# 100% VOLUME & EVIDENCE REPORT (BN-SYN · 28-DAY WINDOW)
**Execution Date:** 2026-02-20
**Global Readiness:** 33.75%
**Status:** BLOCKED
**Contradictions Detected:** 0

## 1. GITHUB TOOLING EVIDENCE (DOMAIN A)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| CI Workflows (Last 5) | gh CLI unavailable; runs not extracted | FAIL | `proof_bundle/logs/63_gh_run_list_stderr.log` |
| Dependabot Alerts | gh CLI unavailable; alerts not extracted | FAIL | `proof_bundle/logs/65_gh_dependabot_stderr.log` |
| CodeQL/SAST Alerts | gh CLI unavailable; alerts not extracted | FAIL | `proof_bundle/logs/64_gh_code_scanning_stderr.log` |
| SSOT Branch Prot. | branch protection state not extracted | FAIL | `proof_bundle/logs/32_git_remote.log` |

## 2. TEST & CODEBASE EVIDENCE (DOMAIN B)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| Total Tests Run | 753 | INFO | `proof_bundle/logs/46_junit_summary.log` |
| Tests Passed/Failed | Passed: 746, Failed: 0, Skipped: 7 | PASS | `proof_bundle/logs/46_junit_summary.log` |
| Code Coverage (%) | coverage.json missing (pytest-cov args failed) | FAIL | `proof_bundle/logs/36_pytest_junit_cov.log`, `proof_bundle/logs/38_reports_presence.log` |
| Mypy Strict Errors | 2 | FAIL | `proof_bundle/logs/39_mypy_protocol.log` |
| Ruff Violations | 0 | PASS | `proof_bundle/logs/40_ruff_protocol.log` |

## 3. FORMAL & META-PROTOCOL EVIDENCE (DOMAIN C)
| Protocol / Spec | File / Path | Status | sha256 Signature |
|-----------------|-------------|--------|------------------|
| GHTPO-2026.02 | not found in repository scan | STUB | N/A |
| ELPEG-2026.04 | not found in repository scan | STUB | N/A |
| BIO-DIGITAL-S12 | not found in repository scan | STUB | N/A |
| TLA+ / Coq Proofs| `specs/tla/BNsyn.tla`, `specs/coq/BNsyn_Sigma.v` | VERIFIED | `b94d832c2a9a8cb4508cb04171b5a24440cd1b0953e758de291059277a3c69c6`, `6c182c5461874d22799c10e4d000425f06460e118bbdcb6c9e28c1b2d5daad4a` |

## 4. PRODUCT UX & BLOCKERS (DOMAIN D)
| Metric | Extracted Data | Status | Anchor / Ref |
|--------|----------------|--------|--------------|
| CLI Error Path | Traceback on invalid input (`bnsyn demo --steps -1 ...`) | BLOCKED | `artifacts/product/evidence/logs/reliability_checks.log`, `proof_bundle/logs/48_reliability_log_extract.log` |
| Evidence Bundles | 6 files in `artifacts/product/evidence` | PASS | `proof_bundle/logs/47_evidence_bundle_count.log` |

## 5. EXECUTIVE VOLUME SUMMARY (28 DAYS)
- **Total Commits Analyzed:** 911
- **Total Synthetic Objects Integrated:** 1456
- **Lines of Code / Tests Written:** Added 402008, Deleted 50811, Total Changed 452819
- **P0 BLOCKERS:**
  1. Unhandled CLI traceback on invalid input path.
  2. GitHub CI/security telemetry not extracted (`gh` CLI unavailable), Domain A unresolved.

**END OF REPORT.**
