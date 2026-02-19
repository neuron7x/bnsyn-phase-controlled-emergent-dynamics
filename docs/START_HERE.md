# START_HERE

This is the **only canonical onboarding funnel**.

## A) Prerequisites

- OS: Linux/macOS shell.
- Python: `>=3.11`.
- Verify toolchain:

```bash
python --version
python -m pip --version
```

## B) Install (single command)

```bash
make setup
```

Expected success lines:
- `python -m pip --version`
- `Successfully installed ...` (or requirements already satisfied)
- `python -m pip check` with no dependency conflicts.

## C) Demo (single command)

```bash
make demo
```

Expected success lines:
- `Demo artifact written: artifacts/demo.json`
- `Demo output validated against expected snapshot`

Visible output:
- `artifacts/demo.json`

## D) Tests (single command)

```bash
make test
```

Expected success lines:
- pytest progress dots and final `[100%]`
- no `FAILED` lines.

## E) Reproduce (single command)

```bash
make reproduce
```

Expected success lines:
- `WROTE artifacts/reproduce_manifest.json`
- `WROTE artifacts/demo.sha256`
- `PASS reproducibility`

Expected artifacts:
- `artifacts/demo.json`
- `artifacts/demo.sha256`
- `artifacts/reproduce_manifest.json`
- `artifacts/reproducibility_report.json`

## F) Troubleshooting (top 10)

| Symptom substring | Likely cause | Fix |
|---|---|---|
| `No module named pip` | Python bootstrap missing | `python -m ensurepip --upgrade` |
| `Could not find a version that satisfies` | stale pip/setuptools | `python -m pip install --upgrade pip setuptools wheel` |
| `ModuleNotFoundError: hypothesis` | missing test extras | `python -m pip install -e ".[test]"` |
| `command not found: pylint` | optional lint tool not installed | `python -m pip install -e ".[dev,test]"` |
| `INVENTORY.json is out of date` | generated file drift | `python tools/generate_inventory.py` |
| `tests/test_generate_inventory.py` assertion mismatch | repo file-count changed | `python tools/generate_inventory.py && make test` |
| `PASS reproducibility` missing | nondeterministic or missing artifact | run `make reproduce` again, inspect `artifacts/reproducibility_report.json` |
| `Failed to capture git SHA` warning | detached/non-git execution context | run inside git checkout; warning is non-fatal in tests |
| `Permission denied` writing `artifacts/` | filesystem permissions | `mkdir -p artifacts && chmod u+w artifacts` |
| `make: *** [test] Error` | failing tests in fast gate | rerun `make test`, then inspect failing nodeids in output |

## G) Cleanup

```bash
make clean
```

Removes caches and onboarding artifacts (`artifacts/demo.json`, checksum, reproduce manifest, reproducibility report).

## Advanced paths

- Architecture depth: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- Testing depth: [docs/TESTING.md](TESTING.md)
- Reproducibility depth: [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)


## H) Security & SBOM (canonical)

```bash
make security
make sbom
```

`make security` installs scanner/audit/SAST dependencies from `requirements-lock.txt` with hashes.
`make sbom` installs `cyclonedx-bom` from `requirements-sbom-lock.txt` with hashes.

Expected security artifacts:
- `artifacts/security/gitleaks-report.json`
- `artifacts/security/pip-audit.json`
- `artifacts/security/bandit.json`
- `artifacts/sbom/sbom.cdx.json`
