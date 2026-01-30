from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, List, Set, Tuple

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

ALLOWED_PREFIXES = ("docker://",)
REPO_ROOT = Path.cwd().resolve()


@dataclass(frozen=True)
class UsesViolation:
    source: Path
    value: str
    message: str


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


def _is_allowed_remote(uses_value: str) -> bool:
    if uses_value.startswith(ALLOWED_PREFIXES):
        return True
    return uses_value in ALLOWED_ACTIONS


def _resolve_local_reference(uses_value: str, base_dir: Path) -> Tuple[str, Path] | None:
    candidate = (base_dir / uses_value).resolve()
    if candidate.is_file() and candidate.suffix in {".yml", ".yaml"}:
        return ("workflow", candidate)
    action_yml = candidate / "action.yml"
    action_yaml = candidate / "action.yaml"
    if action_yml.exists():
        return ("action", action_yml)
    if action_yaml.exists():
        return ("action", action_yaml)
    return None


def _scan_action_file(
    action_path: Path,
    visited: Set[Path],
    violations: List[UsesViolation],
) -> None:
    if action_path in visited:
        return
    visited.add(action_path)

    data = _load_yaml(action_path)
    if data is None:
        return
    uses_entries: List[str] = []
    _collect_uses(data, uses_entries)
    base_dir = action_path.parent
    for uses_value in uses_entries:
        _validate_uses(
            uses_value=uses_value,
            base_dir=base_dir,
            source=action_path,
            visited=visited,
            violations=violations,
        )


def _validate_uses(
    uses_value: str,
    base_dir: Path,
    source: Path,
    visited: Set[Path],
    violations: List[UsesViolation],
) -> None:
    if uses_value.startswith("./"):
        resolved = _resolve_local_reference(uses_value, base_dir)
        if resolved is None:
            violations.append(
                UsesViolation(
                    source=source,
                    value=uses_value,
                    message="Local action not found (missing action.yml/action.yaml).",
                )
            )
            return
        ref_kind, ref_path = resolved
        if ref_kind == "workflow":
            if not ref_path.is_relative_to(REPO_ROOT / ".github" / "workflows"):
                violations.append(
                    UsesViolation(
                        source=source,
                        value=uses_value,
                        message="Reusable workflow must live under .github/workflows.",
                    )
                )
                return
            return
        _scan_action_file(ref_path, visited, violations)
        return
    if not _is_allowed_remote(uses_value):
        violations.append(
            UsesViolation(
                source=source,
                value=uses_value,
                message="Action reference is not in the approved allowlist.",
            )
        )


def main() -> int:
    violations: List[UsesViolation] = []
    visited_actions: Set[Path] = set()
    for workflow in _iter_workflows():
        data = _load_yaml(workflow)
        if data is None:
            continue
        uses_entries: List[str] = []
        _collect_uses(data, uses_entries)
        base_dir = REPO_ROOT
        for uses_value in uses_entries:
            _validate_uses(
                uses_value=uses_value,
                base_dir=base_dir,
                source=workflow,
                visited=visited_actions,
                violations=violations,
            )

    if violations:
        print("Disallowed GitHub Actions detected:", file=sys.stderr)
        for violation in violations:
            print(
                f"- {violation.source}: {violation.value} ({violation.message})",
                file=sys.stderr,
            )
        print(
            "Update scripts/verify_actions_supply_chain.py to allow vetted actions.",
            file=sys.stderr,
        )
        return 1

    print("All GitHub Actions references are approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
