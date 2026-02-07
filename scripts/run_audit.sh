#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

OUT="artifacts/math_audit"
RAW="$OUT/raw"
mkdir -p "$RAW" "$OUT/sources" audit_inputs/odt_sources

# phase 0: deterministic environment capture
{ git rev-parse HEAD; } > "$RAW/p0_git_rev_parse_head.txt"
{ git status --porcelain=v1; } > "$RAW/p0_git_status_porcelain.txt"
{ git branch --show-current; } > "$RAW/p0_git_branch_show_current.txt"
{ git log -1 --date=iso-strict; } > "$RAW/p0_git_log_1.txt"
{ python --version; } > "$RAW/p0_python_version.txt"
{ python -m pip --version; } > "$RAW/p0_pip_version.txt"
{ uname -a; } > "$RAW/p0_uname_a.txt"

python - <<'PY'
import json
from pathlib import Path
raw = Path('artifacts/math_audit/raw')
obj = {
    'python': raw.joinpath('p0_python_version.txt').read_text(encoding='utf-8').strip(),
    'pip': raw.joinpath('p0_pip_version.txt').read_text(encoding='utf-8').strip(),
    'uname': raw.joinpath('p0_uname_a.txt').read_text(encoding='utf-8').strip(),
}
Path('artifacts/math_audit/toolchain_versions.json').write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

# phase 1: ODT claims extraction from repo-local optional inputs
odt_inputs=()
while IFS= read -r -d '' f; do
  odt_inputs+=("$f")
done < <(find audit_inputs -maxdepth 2 -type f -name '*.odt' -print0 | sort -z)
python scripts/extract_odt_claims.py --inputs "${odt_inputs[@]}" --output audit_inputs/claims.json --hash-dir audit_inputs/odt_sources

# phase 2: canonical metrics
python scripts/compute_repo_metrics.py
python scripts/compute_ci_cd_metrics.py

# phase 3: independent crosschecks + gate reports
python scripts/crosscheck_repo_metrics.py
python scripts/crosscheck_ci_cd_metrics.py

python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path
from jsonschema import validate

out = Path('artifacts/math_audit')
computed = json.loads((out / 'computed_metrics.json').read_text(encoding='utf-8'))
crosscheck = json.loads((out / 'crosscheck_metrics.json').read_text(encoding='utf-8'))
ci = json.loads((out / 'ci_cd_metrics.json').read_text(encoding='utf-8'))
ci_cross = json.loads((out / 'crosscheck_ci_cd_metrics.json').read_text(encoding='utf-8'))

orig = (out / 'computed_metrics.json').read_bytes()
orig_hash = hashlib.sha256(orig).hexdigest()
subprocess.run(['python', 'scripts/compute_repo_metrics.py'], check=True)
recomputed = (out / 'computed_metrics.json').read_bytes()
recomputed_hash = hashlib.sha256(recomputed).hexdigest()
computed = json.loads((out / 'computed_metrics.json').read_text(encoding='utf-8'))

schema = json.loads(Path('schemas/audit_report_data.schema.json').read_text(encoding='utf-8'))
report = json.loads(Path('audit_report_data.json').read_text(encoding='utf-8'))
validate(report, schema)
newline_ok = Path('audit_report_data.json').read_bytes().endswith(b'\n')

cat_match = True
for category, values in computed['categories'].items():
    other = crosscheck['categories'][category]
    cat_match = cat_match and values['files_count'] == other['files_count']
    cat_match = cat_match and values['loc_code'] == other['loc_code']
    cat_match = cat_match and values['loc_comments'] == other['loc_comments']
    cat_match = cat_match and values['loc_blank'] == other['loc_blank']

workflows = ci['workflow_count']
ci_nonzero_ok = workflows == 0 or (ci['job_count'] > 0 and ci['step_count'] > 0)

reconciliation = {
    'partition_complete': computed['inventory']['total_files'] == sum(v['files_count'] for v in computed['categories'].values()),
    'countable_totals_match': computed['totals']['countable_loc_code'] == sum(v['loc_code'] for k, v in computed['categories'].items() if k != 'GENERATED'),
    'inventory_match': computed['inventory']['total_files'] == crosscheck['inventory_total_files'],
    'category_loc_match': cat_match,
    'ci_counts_match': ci['workflow_count'] == ci_cross['workflow_count'] and ci['job_count'] == ci_cross['job_count'] and ci['step_count'] == ci_cross['step_count'],
    'ci_nonzero_if_present': ci_nonzero_ok,
    'schema_valid': True,
    'newline_terminated_report': newline_ok,
    'deterministic_compute_repo_metrics': orig_hash == recomputed_hash,
}

(out / 'reconciliation_report.json').write_text(json.dumps(reconciliation, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(out / 'diff_report.json').write_text(json.dumps({
    'repo_metrics_match': reconciliation['inventory_match'] and reconciliation['category_loc_match'],
    'ci_metrics_match': reconciliation['ci_counts_match'],
    'deterministic_compute_repo_metrics': reconciliation['deterministic_compute_repo_metrics'],
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

# phase 4: source snapshots + valuation
python scripts/snapshot_sources.py
python scripts/valuation_model.py

# phase 5: proof bundle for CI artifacts
sha=$(git rev-parse --short HEAD)
bundle="proof_bundle_${sha}.tar.gz"
tar -czf "$bundle" \
  "$RAW" \
  "$OUT/toolchain_versions.json" \
  "$OUT/computed_metrics.json" \
  "$OUT/crosscheck_metrics.json" \
  "$OUT/ci_cd_metrics.json" \
  "$OUT/crosscheck_ci_cd_metrics.json" \
  "$OUT/reconciliation_report.json" \
  "$OUT/diff_report.json" \
  "$OUT/sources" \
  "$OUT/sources_extracted.json" \
  "$OUT/valuation_inputs.json" \
  "$OUT/valuation_results.json" \
  audit_inputs/claims.json \
  audit_inputs/odt_sources
mv "$bundle" "$OUT/$bundle"

# final fail-closed status
python - <<'PY'
import json
from pathlib import Path

claims = json.loads(Path('audit_inputs/claims.json').read_text(encoding='utf-8'))
recon = json.loads(Path('artifacts/math_audit/reconciliation_report.json').read_text(encoding='utf-8'))
valuation = json.loads(Path('artifacts/math_audit/valuation_results.json').read_text(encoding='utf-8'))

status = 'VERIFIED'
if not claims.get('claims'):
    status = 'NOT VERIFIED'
if not all(bool(v) for v in recon.values()):
    status = 'NOT VERIFIED'
if valuation['market_rate_model']['status'] != 'VERIFIED':
    status = 'NOT VERIFIED'

Path('artifacts/math_audit/final_status.txt').write_text(status + '\n', encoding='utf-8')
if status != 'VERIFIED':
    raise SystemExit(2)
PY
