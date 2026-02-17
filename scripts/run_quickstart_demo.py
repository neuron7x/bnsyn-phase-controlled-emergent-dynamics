#!/usr/bin/env python3
"""Generate and validate the canonical quickstart demo artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEMO_TIMEOUT_SECONDS = 30
CANONICAL_DEMO_CMD: tuple[str, ...] = (
    sys.executable,
    "-m",
    "bnsyn",
    "demo",
    "--steps",
    "120",
    "--dt-ms",
    "0.1",
    "--seed",
    "123",
    "--N",
    "32",
)
ARTIFACT_PATH = Path("artifacts/demo.json")


def _validate_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("demo output must be a JSON object")
    demo = payload.get("demo")
    if not isinstance(demo, dict) or not demo:
        raise ValueError("demo output must contain a non-empty 'demo' object")
    return payload


def main() -> int:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            CANONICAL_DEMO_CMD,
            check=True,
            capture_output=True,
            text=True,
            timeout=DEMO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"quickstart demo command timed out after {DEMO_TIMEOUT_SECONDS}s"
        ) from exc

    payload = _validate_payload(json.loads(proc.stdout))

    with ARTIFACT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Demo artifact written: artifacts/demo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
