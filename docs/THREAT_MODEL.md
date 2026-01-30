# Threat Model (CI/CD, Dependencies, Secrets, Container)

## Scope

Focused on repository security controls for CI/CD, dependency integrity, secrets hygiene,
and container usage. This is not a production deployment model.

## Assets

- GitHub Actions workflows and reusable workflow contracts
- Dependency manifests and lock files (`pyproject.toml`, `requirements-lock.txt`)
- Security tooling configurations (`.gitleaks.toml`, bandit rules via `pyproject.toml`)
- Container build context and Dockerfile
- Evidence logs and governance reports

## Trust Boundaries

- GitHub-hosted runners executing workflows
- Dependency registries (PyPI) and artifact downloads
- Local developer environments running `make` targets

## Threats (STRIDE-lite)

| Category | Example Risk | Primary Mitigations |
| --- | --- | --- |
| Spoofing | Malicious action from untrusted source | Action allowlist enforcement, dependency-review workflow |
| Tampering | Modified dependency introduces vulnerable code | Locked dependencies with hashes, pip-audit gating |
| Repudiation | Missing evidence of security checks | PR evidence logs in `PR_DESCRIPTION_EVIDENCE.md` |
| Information Disclosure | Secrets committed to repo | gitleaks scanning + secret hygiene guidance |
| Denial of Service | CI overload from heavy suites | Heavy suites remain scheduled/manual only |
| Elevation of Privilege | Container runs as root | Non-root container user |

## Security Controls

- `make security` runs gitleaks, pip-audit, and bandit.
- PR workflows enforce security checks and dependency review.
- Action usage is restricted to a vetted allowlist.
- Docker image uses a non-root user and hash-locked dependencies.

## Verification

```bash
make security
python scripts/verify_actions_supply_chain.py
```
