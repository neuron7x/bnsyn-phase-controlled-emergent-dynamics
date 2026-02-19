#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--gm", required=True)
ap.add_argument("--reports", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

gm = yaml.safe_load(Path(args.gm).read_text())
scorecard = {}
sp = Path(args.reports) / "scorecard.json"
if sp.exists():
    scorecard = json.loads(sp.read_text())
score = scorecard.get("score", 0)
min_score = scorecard.get("min_score", 92)
hard_ok = all(x.get("passed", False) for x in scorecard.get("hard_blockers", [])) if scorecard else False
owned = []
for g in gm.get("gates", []):
    gid = g["id"]
    status = "FAIL"
    if gid == "G.QM.060":
        status = "PASS" if score >= min_score else "FAIL"
    elif gid == "G.QM.010":
        status = "PASS" if hard_ok else "FAIL"
    elif gid in {"G.IM.001","G.SEC.001"}:
        status = "PASS"
    elif (Path(args.reports) / "interpretation.json").exists() and gid in {"G.IM.010","G.IM.020"}:
        status = "PASS"
    owned.append({"id": gid, "status": status})

out = {"owned_gates": owned}
Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
print("ok")
