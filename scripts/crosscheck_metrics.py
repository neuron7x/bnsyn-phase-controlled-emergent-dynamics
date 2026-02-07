#!/usr/bin/env python3
from __future__ import annotations
import fnmatch, json, subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def git_files() -> list[str]:
    p = subprocess.run(['git','ls-files'], cwd=ROOT, text=True, capture_output=True, check=True)
    return [x for x in p.stdout.splitlines() if x]


def load_rules():
    return yaml.safe_load((ROOT/'audit'/'classification_rules.yml').read_text())


def excluded(path: str, rules: dict) -> bool:
    return any(fnmatch.fnmatch(path, e['pattern']) for e in rules['excludes'])


def category(path: str, rules: dict) -> str:
    order=['GENERATED','CI_CD','INFRASTRUCTURE','TESTS','DATA_SCHEMAS','CORE_LOGIC','SCRIPTS_TOOLING','DOCUMENTATION','STATIC_ASSETS','CONFIGURATION']
    for c in order:
        if any(fnmatch.fnmatch(path,p) for p in rules['categories'].get(c,[])):
            return c
    return 'CONFIGURATION'


def loc(path: Path) -> tuple[int,int,int]:
    code=comment=blank=0
    try:
        lines=path.read_text(encoding='utf-8',errors='ignore').splitlines()
    except Exception:
        return 0,0,0
    for line in lines:
        s=line.strip()
        if not s:
            blank+=1
        elif s.startswith(('#','//','/*','*','--')):
            comment+=1
        else:
            code+=1
    return code,comment,blank


def main() -> None:
    rules=load_rules()
    files=[f for f in git_files() if not excluded(f,rules)]
    cats={k:{'files_count':0,'loc_code':0} for k in ['CORE_LOGIC','TESTS','CI_CD','INFRASTRUCTURE','DOCUMENTATION','SCRIPTS_TOOLING','DATA_SCHEMAS','CONFIGURATION','STATIC_ASSETS','GENERATED']}
    file_sets={k:set() for k in cats}
    for f in files:
        c=category(f,rules)
        file_sets[c].add(f)
        cats[c]['files_count']+=1
        lc=loc(ROOT/f)[0]
        cats[c]['loc_code']+=lc

    union=set().union(*file_sets.values())
    disjoint=True
    keys=list(file_sets)
    overlaps=[]
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            inter=file_sets[a]&file_sets[b]
            if inter:
                disjoint=False
                overlaps.append({'a':a,'b':b,'count':len(inter)})

    workflows=[f for f in files if fnmatch.fnmatch(f,'.github/workflows/*.yml') or fnmatch.fnmatch(f,'.github/workflows/*.yaml') or fnmatch.fnmatch(f,'.github/workflows/**/*.yml') or fnmatch.fnmatch(f,'.github/workflows/**/*.yaml')]
    out={
        'inventory_total_files':len(files),
        'categories':cats,
        'partition':{
            'union_count':len(union),
            'equals_inventory':len(union)==len(files),
            'disjoint':disjoint,
            'overlaps':overlaps,
        },
        'ci_cd_inclusion':{
            'workflow_files_present':len(workflows),
            'ci_cd_files':cats['CI_CD']['files_count'],
            'invariant_pass': (len(workflows)==0) or (cats['CI_CD']['files_count']>=len(workflows) and cats['CI_CD']['loc_code']>=1)
        }
    }
    (OUT/'crosscheck_metrics.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    main()
