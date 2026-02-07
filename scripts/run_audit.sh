#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p artifacts/math_audit/raw artifacts/math_audit/sources audit_inputs/odt_sources

# phase 0
{ git rev-parse HEAD; } > artifacts/math_audit/raw/p0_git_rev_parse_head.txt
{ git status --porcelain=v1; } > artifacts/math_audit/raw/p0_git_status_porcelain.txt
{ git branch --show-current; } > artifacts/math_audit/raw/p0_git_branch_show_current.txt
{ git log -1 --date=iso-strict; } > artifacts/math_audit/raw/p0_git_log_1.txt
{ git remote -v || true; } > artifacts/math_audit/raw/p0_git_remote_v.txt
{ python --version; } > artifacts/math_audit/raw/p0_python_version.txt
{ python -m pip --version; } > artifacts/math_audit/raw/p0_pip_version.txt
{ uname -a; } > artifacts/math_audit/raw/p0_uname_a.txt

python - <<'PY'
import json
from pathlib import Path
raw=Path('artifacts/math_audit/raw')
obj={
 'python': raw.joinpath('p0_python_version.txt').read_text().strip(),
 'pip': raw.joinpath('p0_pip_version.txt').read_text().strip(),
 'uname': raw.joinpath('p0_uname_a.txt').read_text().strip(),
}
Path('artifacts/math_audit/toolchain_versions.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
PY

# phase 1 claims
python scripts/extract_odt_claims.py --inputs \
  "/mnt/data/codex pr.odt" \
  "/mnt/data/кодекс чек.odt" \
  "/mnt/data/кодекс останній.odt" \
  --output audit_inputs/claims.json --hash-dir audit_inputs/odt_sources

# phase 2
python scripts/compute_repo_metrics.py
python scripts/compute_ci_cd_metrics.py

# phase 3
python scripts/crosscheck_repo_metrics.py
python scripts/crosscheck_ci_cd_metrics.py

python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from jsonschema import validate

out=Path('artifacts/math_audit')
cm=json.loads((out/'computed_metrics.json').read_text())
cr=json.loads((out/'crosscheck_metrics.json').read_text())
ci=json.loads((out/'ci_cd_metrics.json').read_text())
cc=json.loads((out/'crosscheck_ci_cd_metrics.json').read_text())

# determinism
orig=(out/'computed_metrics.json').read_bytes()
orig_h=hashlib.sha256(orig).hexdigest()
subprocess.run(['python','scripts/compute_repo_metrics.py'],check=True)
again=(out/'computed_metrics.json').read_bytes()
again_h=hashlib.sha256(again).hexdigest()
cm=json.loads((out/'computed_metrics.json').read_text())

# schema
schema=json.loads(Path('schemas/audit_report_data.schema.json').read_text())
report=json.loads(Path('audit_report_data.json').read_text())
validate(report,schema)
newline_ok=Path('audit_report_data.json').read_bytes().endswith(b'\n')

workflows=ci['workflow_count']
rec={
 'I-A': cm['inventory']['total_files']==sum(v['files_count'] for v in cm['categories'].values()),
 'I-B': cm['totals']['countable_loc_code']==sum(v['loc_code'] for k,v in cm['categories'].items() if k!='GENERATED'),
 'I-C': cm['inventory']['total_files']==cr['inventory_total_files'],
 'I-D': newline_ok,
 'I-E': orig_h==again_h,
 'I-F': ((workflows==0) or (ci['workflow_count']>=1 and ci['job_count']>0 and ci['step_count']>0 and cc['workflow_count']==ci['workflow_count'])),
}
(out/'reconciliation_report.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
(out/'diff_report.json').write_text(json.dumps({'repo_metrics_match': cm['inventory']['total_files']==cr['inventory_total_files'], 'ci_workflow_match': cc['workflow_count']==ci['workflow_count']},indent=2,sort_keys=True)+'\n')
PY

# phase 4
python scripts/snapshot_sources.py
python scripts/valuation_model.py

python - <<'PY'
import hashlib, json
from pathlib import Path
cm=json.loads(Path('artifacts/math_audit/computed_metrics.json').read_text())
out=[]
for cat in ['CORE_LOGIC','TESTS']:
    for rel in cm['categories'][cat]['files']:
        p=Path(rel)
        out.append({'category':cat,'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
Path('artifacts/math_audit/core_tests_sha256.json').write_text(json.dumps(sorted(out,key=lambda x:(x['category'],x['path'])),indent=2,sort_keys=True)+'\n')
PY

# final status gate
python - <<'PY'
import json
from pathlib import Path
claims=json.loads(Path('audit_inputs/claims.json').read_text())
rec=json.loads(Path('artifacts/math_audit/reconciliation_report.json').read_text())
val=json.loads(Path('artifacts/math_audit/valuation_results.json').read_text())
verified = bool(claims.get('claims')) and all(rec.values()) and val.get('market_rate_model_status')=='VERIFIED'
Path('artifacts/math_audit/final_status.txt').write_text('VERIFIED' if verified else 'NOT VERIFIED')
if not verified:
    raise SystemExit(2)
PY
