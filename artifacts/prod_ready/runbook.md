# Runtime Runbook

## Start
- Activate env: `source .venv_prr/bin/activate`
- Demo run: `bnsyn demo --steps 20 --dt-ms 0.1 --seed 123 --N 16`

## Fault injection / shutdown
- Timeout test: `timeout -s TERM 2s bnsyn demo --steps 100000 --dt-ms 0.1 --seed 123 --N 64`
- Expected exit code: `124`

## Release rollback
- Reinstall previous known-good wheel from artifact store or prior `dist/` manifest entry.
