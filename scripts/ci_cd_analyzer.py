#!/usr/bin/env python3
from __future__ import annotations
import fnmatch, json, re, subprocess
from collections import Counter
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'/'math_audit'


def git_files()->list[str]:
    p=subprocess.run(['git','ls-files'],cwd=ROOT,text=True,capture_output=True,check=True)
    return p.stdout.splitlines()


def main()->None:
    files=[f for f in git_files() if fnmatch.fnmatch(f,'.github/workflows/*.yml') or fnmatch.fnmatch(f,'.github/workflows/*.yaml') or fnmatch.fnmatch(f,'.github/workflows/**/*.yml') or fnmatch.fnmatch(f,'.github/workflows/**/*.yaml')]
    jobs=steps=0
    triggers=[]
    actions=Counter()
    permissions_blocks=0
    deploy_indicators=0
    for rel in files:
        p=ROOT/rel
        txt=p.read_text(encoding='utf-8',errors='ignore')
        data=yaml.safe_load(txt) or {}
        onv=data.get('on')
        if isinstance(onv,dict): triggers.extend(onv.keys())
        elif isinstance(onv,list): triggers.extend([str(x) for x in onv])
        elif isinstance(onv,str): triggers.append(onv)
        if 'permissions' in data: permissions_blocks+=1
        j=data.get('jobs') or {}
        if isinstance(j,dict):
            jobs += len(j)
            for _,job in j.items():
                if isinstance(job,dict):
                    st=job.get('steps') or []
                    if isinstance(st,list):
                        steps += len(st)
                        for s in st:
                            if isinstance(s,dict):
                                u=s.get('uses')
                                if u: actions[u]+=1
                                blob=(str(s.get('name',''))+' '+str(s.get('run',''))+' '+str(u or '')).lower()
                                if any(k in blob for k in ['deploy','release','publish','ecr','gcr','docker/login-action','aws-actions/configure-aws-credentials','azure/login','google-github-actions/auth']):
                                    deploy_indicators+=1
                    if 'environment' in job:
                        deploy_indicators+=1

    out={
      'workflow_count':len(files),
      'workflow_files':sorted(files),
      'jobs_count':jobs,
      'steps_count':steps,
      'triggers':sorted(set(triggers)),
      'actions_used_top20':actions.most_common(20),
      'permissions_blocks':permissions_blocks,
      'deployment_indicators':deploy_indicators,
    }
    (OUT/'ci_cd_metrics.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    main()
