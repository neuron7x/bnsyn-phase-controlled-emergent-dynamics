from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def run_case() -> dict[str, float | int]:
    from bnsyn.sim.network import run_simulation

    start = time.perf_counter()
    run_simulation(steps=120, dt_ms=0.1, seed=7, N=64)
    duration = time.perf_counter() - start
    return {
        "seed": 7,
        "N": 64,
        "steps": 120,
        "duration_seconds": duration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"profile": "perf-smoke-v1", "result": run_case()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
