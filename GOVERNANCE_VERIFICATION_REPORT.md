# BN-SYN GOVERNANCE VERIFICATION REPORT (TEMPLATE)

This file is a **template** only. Do **not** commit filled results or PASS/FAIL claims.
Generate verification evidence in CI artifacts or in the PR description instead.

---

## Metadata (fill in when generating a report)

- **Date**:
- **PR / Commit**:
- **Executor**:

## Verification Commands

Record the exact commands executed and their outputs (attach logs/artifacts):

```bash
# Example (replace with actual commands run)
pytest -q tests/test_tla_invariants_guard.py tests/test_vcg_invariants_guard.py
make test
```

## Evidence Links

- CI run URL / artifact links:
- Logs (paths or URLs):

## Notes / Limitations

- Known limitations observed during the run:
- Deviations from expected behavior:

---

**Policy**: This repository must not contain evergreen certification claims.
