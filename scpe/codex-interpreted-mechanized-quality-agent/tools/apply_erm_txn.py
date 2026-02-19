#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--erm', required=True); ap.add_argument('--txn', required=True); ap.add_argument('--out', required=True); args=ap.parse_args()
Path(args.out).write_text(json.dumps({'applied':False,'reason':'noop'}, indent=2)+'\n')
print('ok')
