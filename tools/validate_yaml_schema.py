"""Validate a YAML document against a JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_error(error: jsonschema.ValidationError) -> str:
    path = "/".join(str(part) for part in error.path)
    location = f" at '{path}'" if path else ""
    return f"{error.message}{location}"


def validate_yaml_schema(yaml_path: Path, schema_path: Path) -> list[str]:
    instance = _load_yaml(yaml_path)
    schema = _load_schema(schema_path)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda err: err.path)
    return [_format_error(error) for error in errors]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YAML against JSON schema")
    parser.add_argument("yaml_path", type=Path, help="Path to YAML document")
    parser.add_argument("schema_path", type=Path, help="Path to JSON schema")
    args = parser.parse_args()

    errors = validate_yaml_schema(args.yaml_path, args.schema_path)
    if errors:
        print(f"Schema validation failed for {args.yaml_path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Schema validation passed for {args.yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
