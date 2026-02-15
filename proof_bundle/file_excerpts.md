# File Excerpts

- pyproject toolchain pins: `requires-python = ">=3.11"`, pinned deps and build-system pin.
- requirements-lock generated with hashes via pip-compile.
- CI lockfile freshness check in `.github/workflows/ci-pr-atomic.yml`.
- Security hooks in Makefile and `.gitleaks.toml`.
- API contract is Python module API, not web marketplace API.
