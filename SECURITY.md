# Security

This repository is a research-grade simulator. Do not deploy it as a security boundary.

If you discover a vulnerability (e.g., supply-chain risk in workflows), open a private report
via the repository's security advisory feature (GitHub Security Advisories).


## Local Secret Scan

Run the reproducible security gate with pinned Python dependencies and pinned gitleaks bootstrap:

```bash
python -m pip install -e ".[security]"
make security
```
