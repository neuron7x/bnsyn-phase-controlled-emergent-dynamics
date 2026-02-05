from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


METRIC_NAME = "coverage.xml line-rate"


def read_coverage_percent(xml_path: Path) -> float:
    if not xml_path.exists():
        raise FileNotFoundError(f"coverage file not found: {xml_path}")
    root = ET.parse(xml_path).getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise ValueError("coverage.xml missing line-rate attribute")
    return float(line_rate) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check coverage gate against baseline and floor")
    parser.add_argument("--coverage-xml", type=Path, default=Path("coverage.xml"))
    parser.add_argument("--baseline", type=Path, default=Path("quality/coverage_gate.json"))
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    try:
        current = read_coverage_percent(args.coverage_xml)
    except (FileNotFoundError, ValueError, ET.ParseError) as exc:
        print(f"FAIL: unable to read coverage metric {METRIC_NAME}: {exc}")
        return 1

    baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = float(baseline_data["baseline_percent"])
    minimum = float(baseline_data["minimum_percent"])
    metric = str(baseline_data.get("metric", METRIC_NAME))

    print(f"Metric: {metric}")
    print(f"Current coverage: {current:.2f}%")
    print(f"Baseline coverage: {baseline:.2f}%")
    print(f"Minimum coverage: {minimum:.2f}%")

    if baseline < minimum:
        print(
            "FAIL: invalid baseline configuration; "
            f"baseline_percent ({baseline:.2f}%) < minimum_percent ({minimum:.2f}%)"
        )
        return 1

    if current + args.tolerance < baseline:
        print(
            f"FAIL: coverage dropped below baseline by more than {args.tolerance:.2f}% "
            f"(current={current:.2f}%, baseline={baseline:.2f}%)"
        )
        return 1
    if current + args.tolerance < minimum:
        print(
            f"FAIL: coverage below minimum floor by more than {args.tolerance:.2f}% "
            f"(current={current:.2f}%, minimum={minimum:.2f}%)"
        )
        return 1

    print("PASS: coverage gate satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
