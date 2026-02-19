from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "sse-sdo-max-gate.yml"
OUT = ROOT / "artifacts" / "sse_sdo" / "04_ci" / "DRIFT_REPORT.json"


def main() -> int:
    content = WORKFLOW.read_text(encoding="utf-8") if WORKFLOW.exists() else ""
    required = ["python scripts/sse_gate_runner.py", "artifacts/sse_sdo/**"]
    missing = [item for item in required if item not in content]
    payload = {"drift": missing, "status": "ok" if not missing else "drift_detected"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("OK" if not missing else "DRIFT")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
