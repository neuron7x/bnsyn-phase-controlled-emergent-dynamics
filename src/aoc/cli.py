from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .contracts import task_contract_from_dict
from .controller import AOCController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aoc", description="Adaptive Orchestration Controller")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a deterministic AOC orchestration")
    run.add_argument("--config", required=True, help="YAML task contract path")
    run.add_argument("--output-dir", default=None, help="Output directory override")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        with Path(args.config).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        contract = task_contract_from_dict(payload)
        output_dir = Path(args.output_dir or payload.get("output_dir", "aoc_output"))
        controller = AOCController(contract, output_dir)
        verdict = controller.run()
        print(json.dumps(verdict, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
