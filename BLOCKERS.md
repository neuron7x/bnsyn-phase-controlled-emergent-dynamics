# BLOCKERS

## Verification blockers
1. Missing ODT source file for claim reconciliation.
   - Expected paths:
     - `/mnt/data/кодекс чек.odt`
     - `/mnt/data/codex pr.odt` (fallback)
   - Required command after file is provided:
     - `bash scripts/run_audit.sh`

2. Missing external market-rate source snapshots for sourced model-2 valuation.
   - Required evidence:
     - auditable URL snapshots
     - access timestamp
     - FX source (if non-USD normalization used)
   - Required command after evidence is provided:
     - `bash scripts/run_audit.sh`
