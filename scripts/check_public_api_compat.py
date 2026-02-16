from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _major(version: str) -> int:
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"invalid semver: {version}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()

    from scripts.snapshot_public_api import collect_snapshot

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = collect_snapshot()
    breaking: list[str] = []

    for module_name, baseline_symbols in baseline["modules"].items():
        current_symbols = current["modules"].get(module_name)
        if current_symbols is None:
            breaking.append(f"removed module {module_name}")
            continue
        for symbol, old_sig in baseline_symbols.items():
            new_sig = current_symbols.get(symbol)
            if new_sig is None:
                breaking.append(f"removed symbol {module_name}.{symbol}")
            elif new_sig != old_sig:
                breaking.append(f"changed signature {module_name}.{symbol}: {old_sig} -> {new_sig}")

    if not breaking:
        print("public API compatibility check passed")
        return 0

    baseline_major = _major(str(baseline["version"]))
    current_major = _major(str(current["version"]))
    for item in breaking:
        print(f"BREAKING: {item}")

    if current_major > baseline_major:
        print("breaking changes allowed because major version bumped")
        return 0

    print("ERROR: breaking changes without major version bump")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
