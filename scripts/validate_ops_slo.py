from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_FRONT_MATTER = {"service", "owner", "reviewed_on", "slo_window"}
REQUIRED_SECTIONS = [
    "# SLA/SLO",
    "## Targets",
    "## Core latency profile",
    "## Error budget policy",
    "## Evidence commands",
]
REQUIRED_TARGET_KEYS = {
    "determinism_pass_rate",
    "contract_health",
    "perf_smoke_p95_seconds",
    "security_scan_freshness_hours",
    "docs_build_health",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_ops_slo.py <path>")
        return 2
    path = Path(sys.argv[1])
    content = path.read_text(encoding="utf-8")

    match = re.match(r"^---\n(.*?)\n---\n", content, flags=re.DOTALL)
    if not match:
        print("ERROR: Missing YAML front matter")
        return 1

    fm_text = match.group(1)
    fm_keys = {line.split(":", 1)[0].strip() for line in fm_text.splitlines() if ":" in line}
    missing_fm = sorted(REQUIRED_FRONT_MATTER - fm_keys)
    if missing_fm:
        print(f"ERROR: Missing front matter keys: {', '.join(missing_fm)}")
        return 1

    for section in REQUIRED_SECTIONS:
        if section not in content:
            print(f"ERROR: Missing section: {section}")
            return 1

    for key in REQUIRED_TARGET_KEYS:
        if f"**{key}**" not in content:
            print(f"ERROR: Missing target key {key}")
            return 1

    if "```bash" not in content:
        print("ERROR: Evidence commands block missing")
        return 1

    print("SLA/SLO document validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
