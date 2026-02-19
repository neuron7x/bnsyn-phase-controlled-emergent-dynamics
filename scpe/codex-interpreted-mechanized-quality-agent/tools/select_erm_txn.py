#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--erm', required=True); ap.add_argument('--deadlock', required=True); ap.add_argument('--out', required=True); args=ap.parse_args()
Path(args.out).write_text('id: ERM.NOOP\n')
print('ok')
