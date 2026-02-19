#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--im", required=True)
ap.add_argument("--qm", required=True)
ap.add_argument("--baseline", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trace", required=True)
args = ap.parse_args()

lint = json.loads(Path("REPORTS/quality/lint.json").read_text())["lint.error_count"]
tests = json.loads(Path("REPORTS/quality/tests.json").read_text())["tests.fail_count"]
sec = json.loads(Path("REPORTS/quality/security.json").read_text())["security.high_count"]

items = []
if tests > 0:
    items.append({"status":"FAIL","category":"tests","severity":"S0","metric":"tests.fail_count","value":tests,"threshold":0,"gate_ids":["G.QM.010"],"recommended_actions":["A.FIX.TESTS.MINIMAL"],"priority":900})
if sec > 0:
    items.append({"status":"FAIL","category":"security","severity":"S0","metric":"security.high_count","value":sec,"threshold":0,"gate_ids":["G.QM.010"],"recommended_actions":["A.FIX.SEC.MINIMAL"],"priority":850})
if lint > 0:
    items.append({"status":"FAIL","category":"lint","severity":"S1","metric":"lint.error_count","value":lint,"threshold":0,"gate_ids":["G.QM.010"],"recommended_actions":["A.FIX.LINT.MINIMAL"],"priority":800})

status = "PASS" if not items else "FAIL"
out = {"status":status,"items":items,"instrumentation_required":[],"modalities_present":["M.CODE","M.TEST","M.SEC","M.DOC","M.PERF","M.CI"],"contradictions":[],"alternatives":[]}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
sha = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
Path(args.trace).write_text(json.dumps({"step":1,"rule_id":"R.IM.EVAL","decision":status,"facts_snapshot_sha256":sha,"outputs":[args.out]})+"\n")
print(status)
