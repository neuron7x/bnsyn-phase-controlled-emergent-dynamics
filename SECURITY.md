# Security

This repository is a research-grade simulator. Do not deploy it as a security boundary.

## Supported Versions

Security fixes are provided only for the `main` branch. Releases are research snapshots and
are not supported for security backports.

## Reporting a Vulnerability

If you discover a vulnerability (e.g., supply-chain risk in workflows), open a private report
via the repository's security advisory feature (GitHub Security Advisories). Include:

- A concise description of the issue and affected files.
- Reproduction steps in a local environment.
- Any suggested mitigations.

## Scope

In scope:
- CI/CD workflows and supply-chain integrity
- Dependency security and vulnerability management
- Secrets hygiene (accidental exposure)
- Container hardening for local/CI use

Out of scope:
- Offensive exploitation guidance
- Production hardening for unrelated deployments

## Incident Response Runbook

1. Triage severity and impacted versions (main branch only).
2. Reproduce locally with deterministic commands.
3. Apply a minimal fix; add or strengthen tests/checks when relevant.
4. Update documentation and evidence logs.
5. Rotate exposed credentials if applicable and invalidate leaked tokens.

## Verification Commands

Run the security gates locally before requesting review:

```bash
make security
```
