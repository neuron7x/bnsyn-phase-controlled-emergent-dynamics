from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, List

import yaml

ALLOWED_ACTIONS = {
    "actions/cache@v4",
    "actions/checkout@v4",
    "actions/dependency-review-action@v4",
    "actions/github-script@v8",
    "actions/setup-java@v4",
    "actions/setup-python@v5",
    "actions/upload-artifact@v4",
    "codecov/codecov-action@v5",
    "github/codeql-action/analyze@v3",
    "github/codeql-action/autobuild@v3",
    "github/codeql-action/init@v3",
    "gitleaks/gitleaks-action@v2",
    "ocaml/setup-ocaml@v2",
    "slackapi/slack-github-action@v2.1.1",
}

ALLOWED_PREFIXES = ("./", "docker://")


@dataclass(frozen=True)
class UsesLocation:
    workflow: Path
    value: str


def _collect_uses(node: Any, found: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                _collect_uses(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_uses(item, found)


def _load_yaml(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return yaml.safe_load(content) if content.strip() else None


def _iter_workflows() -> Iterable[Path]:
    workflows_dir = Path(".github/workflows")
    for suffix in ("*.yml", "*.yaml"):
        yield from sorted(workflows_dir.glob(suffix))


def _is_allowed(uses_value: str) -> bool:
    if uses_value.startswith(ALLOWED_PREFIXES):
        return True
    return uses_value in ALLOWED_ACTIONS


def main() -> int:
    violations: List[UsesLocation] = []
    for workflow in _iter_workflows():
        data = _load_yaml(workflow)
        if data is None:
            continue
        uses_entries: List[str] = []
        _collect_uses(data, uses_entries)
        for uses_value in uses_entries:
            if not _is_allowed(uses_value):
                violations.append(UsesLocation(workflow=workflow, value=uses_value))

    if violations:
        print("Disallowed GitHub Actions detected:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation.workflow}: {violation.value}", file=sys.stderr)
        print(
            "Update scripts/verify_actions_supply_chain.py to allow vetted actions.",
            file=sys.stderr,
        )
        return 1

    print("All GitHub Actions references are approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
