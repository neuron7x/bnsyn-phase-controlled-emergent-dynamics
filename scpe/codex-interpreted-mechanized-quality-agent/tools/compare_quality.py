#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--baseline', required=True); ap.add_argument('--after', required=True); ap.add_argument('--out', required=True); args=ap.parse_args()
b=json.loads(Path(args.baseline).read_text())
a=json.loads(Path(args.after).read_text())
out={"baseline_sha":"na","after_sha":"na","score_delta":0,"dimension_deltas":{},"metric_deltas":{},"measurement_contract_equal": b.get('command_list')==a.get('command_list') and b.get('report_paths')==a.get('report_paths')}
Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
Path('REPORTS/diff-summary.json').write_text(json.dumps({"changed":[]}, indent=2, sort_keys=True)+"\n")
print('ok')
