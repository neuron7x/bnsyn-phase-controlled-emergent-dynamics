from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("quality/perf_results.json"))
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    budgets = _parse_simple_yaml(args.budgets)

    base = float(baseline["result"]["duration_seconds"])
    current = float(results["result"]["duration_seconds"])
    max_duration = float(budgets["max_duration_seconds"])
    max_regression = float(budgets["max_regression_percent"])
    regression_pct = ((current - base) / base) * 100.0 if base > 0 else 0.0

    if current > max_duration:
        print(f"ERROR: duration {current:.4f}s exceeds max {max_duration:.4f}s")
        return 1
    if regression_pct > max_regression:
        print(f"ERROR: regression {regression_pct:.2f}% exceeds {max_regression:.2f}%")
        return 1

    print(
        "perf budget check passed "
        f"(current={current:.4f}s baseline={base:.4f}s regression={regression_pct:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
