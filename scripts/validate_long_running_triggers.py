from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import configparser
import re
import sys
import tomllib
from typing import Iterable

import yaml

ALLOWED_GATE_CLASSES = {"PR-gate", "long-running"}
ALLOWED_REUSABLE_VALUES = {"YES", "NO"}
FORBIDDEN_TRIGGERS = {"pull_request", "push"}
BEGIN_MARKER = "| Workflow file | Workflow name | Gate Class | Trigger set | Reusable? |"


class PolicyParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class InventoryRow:
    workflow_file: str
    gate_class: str
    reusable: str


@dataclass(frozen=True)
class PolicyResult:
    exit_code: int
    output_lines: list[str]


def parse_version(version_text: str) -> tuple[int, ...]:
    match = re.match(r"^\d+(?:\.\d+)*", version_text.strip())
    if not match:
        raise PolicyParseError(f"PYTHON_VERSION_SPEC_INVALID spec={version_text}")
    return tuple(int(part) for part in match.group(0).split("."))


def compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    max_len = max(len(left), len(right))
    padded_left = left + (0,) * (max_len - len(left))
    padded_right = right + (0,) * (max_len - len(right))
    if padded_left < padded_right:
        return -1
    if padded_left > padded_right:
        return 1
    return 0


def matches_prefix(
    runtime: tuple[int, ...],
    required: tuple[int, ...],
) -> bool:
    return runtime[: len(required)] == required


def specifier_allows(
    runtime: tuple[int, ...],
    spec: str,
) -> bool:
    spec = spec.strip()
    if not spec:
        raise PolicyParseError("PYTHON_VERSION_SPEC_EMPTY")
    operators = ("~=", "==", "!=", ">=", "<=", ">", "<")
    op = next((token for token in operators if spec.startswith(token)), "")
    version_text = spec[len(op) :].strip()
    if not op:
        op = "=="
    wildcard = version_text.endswith(".*")
    if wildcard:
        version_text = version_text[: -2]
    required = parse_version(version_text)
    runtime_cmp = compare_versions(runtime, required)
    if op == "==":
        if wildcard:
            return matches_prefix(runtime, required)
        return runtime_cmp == 0
    if op == "!=":
        if wildcard:
            return not matches_prefix(runtime, required)
        return runtime_cmp != 0
    if op == ">=":
        return runtime_cmp >= 0
    if op == "<=":
        return runtime_cmp <= 0
    if op == ">":
        return runtime_cmp > 0
    if op == "<":
        return runtime_cmp < 0
    if op == "~=":
        if len(required) == 1:
            upper = (required[0] + 1,)
        else:
            upper = (required[0], required[1] + 1)
        return runtime_cmp >= 0 and compare_versions(runtime, upper) < 0
    raise PolicyParseError(f"PYTHON_VERSION_SPEC_INVALID spec={spec}")


def requires_python_satisfied(runtime: tuple[int, ...], spec: str) -> bool:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise PolicyParseError("PYTHON_VERSION_SPEC_EMPTY")
    return all(specifier_allows(runtime, part) for part in parts)


def load_python_requires(project_root: Path) -> str:
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        requires = data.get("project", {}).get("requires-python")
        if requires:
            return str(requires)
    python_version_path = project_root / ".python-version"
    if python_version_path.exists():
        version_text = python_version_path.read_text(encoding="utf-8").strip()
        if version_text:
            return f"=={version_text}"
    setup_cfg_path = project_root / "setup.cfg"
    if setup_cfg_path.exists():
        config = configparser.ConfigParser()
        config.read(setup_cfg_path)
        if config.has_option("options", "python_requires"):
            requires = config.get("options", "python_requires").strip()
            if requires:
                return requires
    raise PolicyParseError("PYTHON_VERSION_POLICY_MISSING")


def validate_python_version(
    project_root: Path,
    runtime_version: tuple[int, int, int] | None,
) -> str:
    required = load_python_requires(project_root)
    runtime_info = runtime_version or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    runtime = (runtime_info[0], runtime_info[1], runtime_info[2])
    if not requires_python_satisfied(runtime, required):
        runtime_text = ".".join(str(part) for part in runtime)
        raise PolicyParseError(
            f"PYTHON_VERSION_UNSUPPORTED runtime={runtime_text} required={required}"
        )
    return required


def normalize_triggers(on_section: object) -> tuple[str, ...]:
    if on_section is None:
        raise PolicyParseError("WORKFLOW_ON_SECTION_MISSING")
    triggers: set[str] = set()
    if isinstance(on_section, str):
        if not on_section.strip():
            raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
        triggers.add(on_section)
    elif isinstance(on_section, list):
        if not on_section:
            raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
        for item in on_section:
            trigger = str(item)
            if trigger == "":
                raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
            triggers.add(trigger)
    elif isinstance(on_section, dict):
        if not on_section:
            raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
        for key in on_section.keys():
            trigger = str(key)
            if trigger == "":
                raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
            triggers.add(trigger)
    else:
        raise PolicyParseError("WORKFLOW_ON_SECTION_INVALID")
    if not triggers:
        raise PolicyParseError("WORKFLOW_ON_SECTION_EMPTY")
    return tuple(sorted(triggers))


def parse_inventory_table(text: str) -> tuple[dict[str, InventoryRow], list[str]]:
    lines = text.splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER)
    except StopIteration as exc:
        raise PolicyParseError("WORKFLOW_INVENTORY_TABLE_MISSING") from exc

    rows: dict[str, InventoryRow] = {}
    duplicates: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("EXCEPTION:") or stripped.startswith("> EXCEPTION:"):
            continue
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 5:
            raise PolicyParseError("WORKFLOW_INVENTORY_ROW_INVALID")
        if cells[0].startswith("---"):
            continue
        workflow_file = cells[0].strip("` ").strip()
        gate_class = cells[2].strip()
        reusable_raw = cells[4].strip()
        if not workflow_file or not workflow_file.endswith(".yml"):
            raise PolicyParseError(f"WORKFLOW_FILE_INVALID value={workflow_file}")
        if gate_class not in ALLOWED_GATE_CLASSES:
            raise PolicyParseError(f"GATE_CLASS_INVALID value={gate_class}")
        if reusable_raw not in ALLOWED_REUSABLE_VALUES:
            raise PolicyParseError(f"REUSABLE_VALUE_INVALID value={reusable_raw}")
        if workflow_file in rows:
            duplicates.append(workflow_file)
            continue
        rows[workflow_file] = InventoryRow(
            workflow_file=workflow_file,
            gate_class=gate_class,
            reusable=reusable_raw,
        )
    if not rows:
        raise PolicyParseError("WORKFLOW_INVENTORY_TABLE_EMPTY")
    return rows, duplicates


def load_workflows(workflows_dir: Path) -> dict[str, tuple[str, ...]]:
    workflows: dict[str, tuple[str, ...]] = {}
    for workflow_path in sorted(workflows_dir.glob("*.yml")):
        data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PolicyParseError(f"WORKFLOW_YAML_INVALID file={workflow_path.name}")
        if "on" in data:
            on_section = data["on"]
        elif True in data:
            on_section = data[True]
        else:
            raise PolicyParseError(f"WORKFLOW_ON_SECTION_MISSING file={workflow_path.name}")
        triggers = normalize_triggers(on_section)
        workflows[workflow_path.name] = triggers
    return workflows


def format_trigger_list(triggers: Iterable[str]) -> str:
    ordered = sorted(triggers)
    return f"[{','.join(ordered)}]"


def evaluate_policy(
    project_root: Path,
    runtime_version: tuple[int, int, int] | None,
) -> tuple[list[str], int]:
    validate_python_version(project_root, runtime_version)
    contracts_path = project_root / ".github/WORKFLOW_CONTRACTS.md"
    workflows_dir = project_root / ".github/workflows"
    rows, duplicates = parse_inventory_table(
        contracts_path.read_text(encoding="utf-8")
    )
    workflows = load_workflows(workflows_dir)

    violations: list[str] = []
    for workflow_file in sorted(set(duplicates)):
        violations.append(f"VIOLATION: DUPLICATE_WORKFLOW_ROW {workflow_file}")

    expected_files = set(rows.keys())
    actual_files = set(workflows.keys())
    for workflow_file in sorted(actual_files - expected_files):
        violations.append(f"VIOLATION: MISSING_GATE_CLASS_ROW {workflow_file}")
    for workflow_file in sorted(expected_files - actual_files):
        violations.append(f"VIOLATION: EXTRA_GATE_CLASS_ROW {workflow_file}")

    for workflow_file in sorted(actual_files & expected_files):
        row = rows[workflow_file]
        if row.gate_class != "long-running":
            continue
        triggers = workflows[workflow_file]
        forbidden = sorted(set(triggers) & FORBIDDEN_TRIGGERS)
        if forbidden:
            violations.append(
                "VIOLATION: LONG_RUNNING_FORBIDDEN_TRIGGER "
                f"{workflow_file} class=long-running "
                f"triggers={format_trigger_list(triggers)} "
                f"forbidden={format_trigger_list(forbidden)}"
            )

    return sorted(violations), len(workflows)


def run_policy(
    project_root: Path,
    dry_run: bool,
    runtime_version: tuple[int, int, int] | None,
) -> PolicyResult:
    try:
        violations, workflow_count = evaluate_policy(project_root, runtime_version)
    except PolicyParseError as exc:
        return PolicyResult(3, [f"PARSE_ERROR: {exc}"])

    if violations:
        exit_code = 0 if dry_run else 2
        return PolicyResult(exit_code, violations)

    return PolicyResult(
        0,
        [
            "OK: long_running_trigger_policy "
            f"workflows={workflow_count} violations=0"
        ],
    )


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    dry_run = False
    if len(args) == 2 and args[1] == "--dry-run":
        dry_run = True
    elif len(args) != 1:
        print("Usage: python scripts/validate_long_running_triggers.py [--dry-run]")
        return 3

    result = run_policy(Path("."), dry_run=dry_run, runtime_version=None)
    for line in result.output_lines:
        print(line)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
