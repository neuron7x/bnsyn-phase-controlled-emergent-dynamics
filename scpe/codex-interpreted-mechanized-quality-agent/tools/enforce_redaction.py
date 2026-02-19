#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import yaml
ap=argparse.ArgumentParser(); ap.add_argument('--policy', required=True); ap.add_argument('--paths', nargs='+', required=True); args=ap.parse_args()
pol=yaml.safe_load(Path(args.policy).read_text())
rules=[(re.compile(r['pattern']), r['replace']) for r in pol.get('rules',[]) if r.get('type')=='regex']
for p in args.paths:
    path=Path(p)
    files=[f for f in path.rglob('*') if f.is_file()] if path.is_dir() else ([path] if path.exists() else [])
    for f in files:
        txt=f.read_text(encoding='utf-8', errors='ignore')
        for rx,rep in rules:
            txt=rx.sub(rep, txt)
        f.write_text(txt, encoding='utf-8')
print('ok')
