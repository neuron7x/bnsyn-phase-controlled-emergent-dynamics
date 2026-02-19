#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--evidence-root', required=True); ap.add_argument('--out', required=True); args=ap.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({'meta':True,'evidence_root':args.evidence_root}, indent=2)+'\n')
print('ok')
