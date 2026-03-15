#!/usr/bin/env python3
"""Generate and validate the canonical product demo artifact."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEMO_TIMEOUT_SECONDS = 120
CANONICAL_DEMO_CMD: tuple[str, ...] = (
    sys.executable,
    "-m",
    "bnsyn",
    "demo-product",
)
ARTIFACT_PATH = Path("artifacts/demo_product_stdout.txt")


def _validate_payload(text: str) -> str:
    required = [
        "STATUS: PASS",
        "ARTIFACT_DIR: artifacts/canonical_run",
        "REPORT: artifacts/canonical_run/index.html",
        "PRIMARY_VISUAL: artifacts/canonical_run/emergence_plot.png",
        "VALIDATE: bnsyn validate-bundle artifacts/canonical_run",
    ]
    for token in required:
        if token not in text:
            raise RuntimeError(f"missing expected token: {token}")
    return text


def _tail_40_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-40:])


def main() -> int:
    try:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create artifacts directory: {exc}") from exc

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
            f"timeout {DEMO_TIMEOUT_SECONDS}s: {' '.join(CANONICAL_DEMO_CMD)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        raise RuntimeError(
            f"command failed rc={exc.returncode}: {' '.join(CANONICAL_DEMO_CMD)} | {_tail_40_lines(combined).strip()}"
        ) from exc

    payload = _validate_payload(proc.stdout)

    try:
        with ARTIFACT_PATH.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            if not payload.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"cannot write {ARTIFACT_PATH}: {exc}") from exc

    print(f"Demo artifact written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"quickstart demo FAILED: {exc}")
        raise SystemExit(1)
