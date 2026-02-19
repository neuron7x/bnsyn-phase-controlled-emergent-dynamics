#!/usr/bin/env python3
import json
from pathlib import Path
meta = {"consecutive_fails": 0, "deficit_severity": "S1", "deadlock_fingerprint": "none"}
deadlock = {"trigger_erm": False, "reason": "preconditions_not_met"}
Path("REPORTS").mkdir(exist_ok=True)
Path("REPORTS/meta-state.json").write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n")
Path("REPORTS/deadlock.json").write_text(json.dumps(deadlock, indent=2, sort_keys=True)+"\n")
print("ok")
