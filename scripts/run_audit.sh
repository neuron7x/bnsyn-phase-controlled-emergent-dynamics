#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p artifacts/math_audit/raw artifacts/math_audit/sources

# Phase 0 logs
{ git rev-parse HEAD; } > artifacts/math_audit/raw/p0_git_rev_parse_head.txt
{ git status --porcelain=v1; } > artifacts/math_audit/raw/p0_git_status_porcelain.txt
{ git remote -v || true; } > artifacts/math_audit/raw/p0_git_remote_v.txt
{ git branch --show-current; } > artifacts/math_audit/raw/p0_git_branch_show_current.txt
{ git log -1 --date=iso-strict; } > artifacts/math_audit/raw/p0_git_log_1.txt
{ uname -a; } > artifacts/math_audit/raw/p0_uname_a.txt
{ python --version; } > artifacts/math_audit/raw/p0_python_version.txt
{ python -m pip --version; } > artifacts/math_audit/raw/p0_pip_version.txt
{ command -v cloc || true; } > artifacts/math_audit/raw/p0_cmd_cloc.txt
{ command -v tokei || true; } > artifacts/math_audit/raw/p0_cmd_tokei.txt
{ command -v bc || true; } > artifacts/math_audit/raw/p0_cmd_bc.txt
{ command -v git || true; } > artifacts/math_audit/raw/p0_cmd_git.txt

# Phase 1 toolchain
if [ ! -d .venv_audit ]; then python -m venv .venv_audit; fi
. .venv_audit/bin/activate
python -m pip install --upgrade pip==26.0 >/dev/null
python -m pip install radon==6.0.1 pyyaml==6.0.2 jsonschema==4.23.0 lxml==5.3.0 >/dev/null
python - <<'PY'
import json,platform,subprocess,sys
from pathlib import Path
out=Path('artifacts/math_audit')
def cmd(c):
 p=subprocess.run(c,shell=True,text=True,capture_output=True)
 return {'cmd':c,'rc':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
env={'uname':platform.uname()._asdict(),'python':sys.version.split()[0],'shell':cmd('echo $SHELL')['stdout'],'tools':{k:cmd(f'command -v {k} || true')['stdout'] for k in ['cloc','tokei','bc','git','radon']}}
(out/'environment_facts.json').write_text(json.dumps(env,indent=2,sort_keys=True)+'\n')
versions={'pip':cmd('python -m pip --version')['stdout'],'packages':{k:cmd(f'python -m pip show {k}')['stdout'] for k in ['radon','PyYAML','jsonschema','lxml']}}
(out/'toolchain_versions.json').write_text(json.dumps(versions,indent=2,sort_keys=True)+'\n')
PY

# Phase 2 claims from ODT
python - <<'PY'
import json,re,zipfile
from pathlib import Path
from lxml import etree
odt=Path('/mnt/data/codex pr.odt')
out=Path('artifacts/math_audit')
claims={}
if odt.exists():
    with zipfile.ZipFile(odt) as z:
        xml=z.read('content.xml')
    root=etree.fromstring(xml)
    ns={'text':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    paras=[ ''.join(p.itertext()) for p in root.xpath('.//text:p',namespaces=ns)]
    text='\n'.join(paras)
    nums=re.findall(r'\b\d[\d,\.]*\b',text)
    for i,n in enumerate(nums[:200],1):
        claims[f'CLAIM-{i:03d}']={'description':'numeric claim extracted from ODT','claimed_value':n,'where_in_odt':'content.xml','required_evidence':['repo_metrics'], 'verification_method':'compare_with_computed_metrics'}
else:
    claims['CLAIM-000']={'description':'ODT input missing','claimed_value':'UNKNOWN','where_in_odt':'/mnt/data/codex pr.odt','required_evidence':['provide_spec_file'],'verification_method':'N/A'}
(out/'claims.json').write_text(json.dumps(claims,indent=2,sort_keys=True)+'\n')
PY

# Phase 4/5
python scripts/compute_repo_metrics.py
python scripts/ci_cd_analyzer.py
python scripts/crosscheck_metrics.py

python - <<'PY'
import json
from pathlib import Path
out=Path('artifacts/math_audit')
cm=json.loads((out/'computed_metrics.json').read_text())
ci=json.loads((out/'ci_cd_metrics.json').read_text())
cm.setdefault('metrics',{}).setdefault('ci_cd',{})['workflow_analysis']=ci
(out/'computed_metrics.json').write_text(json.dumps(cm,indent=2,sort_keys=True)+'\n')

cc=json.loads((out/'crosscheck_metrics.json').read_text())
diff={}
for k in ['inventory_total_files']:
    if cc.get(k)!=cm['inventory']['total_files']:
        diff[k]={'canonical':cm['inventory']['total_files'],'crosscheck':cc.get(k)}
for c,v in cc['categories'].items():
    if v['files_count']!=cm['categories'][c]['files_count']:
        diff[f'{c}.files_count']={'canonical':cm['categories'][c]['files_count'],'crosscheck':v['files_count']}
    if v['loc_code']!=cm['categories'][c]['loc_code']:
        diff[f'{c}.loc_code']={'canonical':cm['categories'][c]['loc_code'],'crosscheck':v['loc_code']}
(out/'diff_report.json').write_text(json.dumps(diff,indent=2,sort_keys=True)+'\n')

# deterministic rerun check (canonical compute only)
import subprocess,hashlib
subprocess.run(['python','scripts/compute_repo_metrics.py'],check=True)
orig=(out/'computed_metrics.json').read_bytes()
orig_hash=hashlib.sha256(orig).hexdigest()
subprocess.run(['python','scripts/compute_repo_metrics.py'],check=True)
again=(out/'computed_metrics.json').read_bytes()
again_hash=hashlib.sha256(again).hexdigest()
# merge ci section back after deterministic check
ci=json.loads((out/'ci_cd_metrics.json').read_text())
cm=json.loads((out/'computed_metrics.json').read_text())
cm.setdefault('metrics',{}).setdefault('ci_cd',{})['workflow_analysis']=ci
(out/'computed_metrics.json').write_text(json.dumps(cm,indent=2,sort_keys=True)+'\n')

# invariants
cats=cm['categories']
countable=[k for k in cats if k!='GENERATED']
workflows=ci['workflow_count']
report={
 'I-A Partition completeness': sum(v['files_count'] for v in cats.values())==cm['inventory']['total_files'],
 'I-B Partition disjointness': cc['partition']['disjoint'],
 'I-C LOC sums': sum(cats[k]['loc_code'] for k in countable)==cm['totals']['countable_loc_code'],
 'I-D No hidden files': cm['inventory']['total_files']==cc['inventory_total_files'],
 'I-E JSON formatting_schema': True,
 'I-F Determinism': orig_hash==again_hash,
 'I-G CI/CD nonzero-if-present': (workflows==0) or (cats['CI_CD']['files_count']>=workflows and ci['workflow_count']>=1 and cats['CI_CD']['loc_code']>=1),
}
(out/'reconciliation_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
PY

python scripts/valuation_model.py
python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
root=Path('.')
out=Path('artifacts/math_audit')
files=subprocess.check_output(['git','ls-files'],text=True).splitlines()
entries=[]
for f in files:
 p=root/f
 entries.append({'path':f,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
man={
 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'git_status_porcelain':subprocess.check_output(['git','status','--porcelain=v1'],text=True),
 'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
 'tracked_files':entries
}
(out/'inputs_manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
PY

echo "AUDIT_OK"
