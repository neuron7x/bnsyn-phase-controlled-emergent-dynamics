#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "experiment_results.schema.json"
DEFAULT_RESULTS_DIR = ROOT / "results" / "temp_ablation_v2"


def fail(message: str) -> None:
    print(f"[results-schema] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_schema(path: Path) -> dict:
    if not path.exists():
        fail(f"Schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_result_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        fail(f"Results path not found: {path}")
    for candidate in sorted(path.glob("*.json")):
        if candidate.name == "manifest.json":
            continue
        yield candidate


def validate_file(validator: Draft7Validator, path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if errors:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        )
        fail(f"{path}: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment results JSON files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_RESULTS_DIR],
        help="Result files or directories to validate.",
    )
    args = parser.parse_args()

    schema = load_schema(SCHEMA_PATH)
    validator = Draft7Validator(schema)

    files: list[Path] = []
    for target in args.paths:
        files.extend(iter_result_files(target))

    if not files:
        fail("No result files found for validation")

    for path in files:
        validate_file(validator, path)

    print(f"[results-schema] OK: validated {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
