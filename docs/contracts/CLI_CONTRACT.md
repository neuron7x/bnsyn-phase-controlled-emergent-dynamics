# CLI contract v1

Contract source: `contracts/cli_contract.v1.json`.

## Scope
Public CLI entrypoint is `bnsyn` (`python -m bnsyn.cli`).

## Stability guarantees
- Top-level help includes `usage:`, `positional arguments:`, and `options:` sections.
- Stable subcommands in v1: `demo`, `run`, `dtcheck`, `sleep-stack`.
- Invalid command exits with non-zero status and includes `invalid choice` in stderr.

## Verification
```bash
python -m pytest tests/contracts/test_cli_contract.py -q
```
