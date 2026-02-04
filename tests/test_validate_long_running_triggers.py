from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate_long_running_triggers import run_policy


def write_pyproject(root: Path, requires: str = ">=3.11") -> None:
    root.joinpath("pyproject.toml").write_text(
        f"[project]\nrequires-python = \"{requires}\"\n",
        encoding="utf-8",
    )


def write_contracts(root: Path, rows: list[dict[str, str]]) -> None:
    contracts_dir = root / ".github"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CI/CD Workflow Contracts",
        "",
        "## Workflow Inventory Table (Authoritative)",
        "",
        "| Workflow file | Workflow name | Gate Class | Trigger set | Reusable? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `"
            + row["workflow_file"]
            + "` | `"
            + row.get("workflow_name", row["workflow_file"].replace(".yml", ""))
            + "` | "
            + row["gate_class"]
            + " | `"
            + row.get("trigger_set", "workflow_dispatch")
            + "` | "
            + row.get("reusable", "NO")
            + " |"
        )
    contracts_dir.joinpath("WORKFLOW_CONTRACTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_workflow(root: Path, name: str, on_section: object | None, include_on: bool = True) -> None:
    workflow_dir = root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    data: dict[object, object] = {"name": name.replace(".yml", "")}
    if include_on:
        data["on"] = on_section
    data["jobs"] = {"noop": {"runs-on": "ubuntu-latest", "steps": []}}
    workflow_dir.joinpath(name).write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def run_check(root: Path, dry_run: bool = False, runtime: tuple[int, int, int] = (3, 11, 0)):
    return run_policy(root, dry_run=dry_run, runtime_version=runtime)


def test_long_running_pull_request_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "pull_request")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_FORBIDDEN_TRIGGER long.yml class=long-running "
        "triggers=[pull_request] forbidden=[pull_request]",
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[pull_request] expected=[schedule,workflow_dispatch]",
    ]


def test_long_running_push_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "push")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_FORBIDDEN_TRIGGER long.yml class=long-running "
        "triggers=[push] forbidden=[push]",
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[push] expected=[schedule,workflow_dispatch]",
    ]


def test_long_running_combined_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", ["push", "pull_request"])
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_FORBIDDEN_TRIGGER long.yml class=long-running "
        "triggers=[pull_request,push] forbidden=[pull_request,push]",
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[pull_request,push] expected=[schedule,workflow_dispatch]",
    ]


def test_long_running_missing_schedule_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[workflow_dispatch] expected=[schedule,workflow_dispatch]"
    ]


def test_long_running_missing_workflow_dispatch_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "schedule")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[schedule] expected=[schedule,workflow_dispatch]"
    ]


def test_long_running_extra_trigger_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(
        tmp_path,
        "long.yml",
        {"schedule": None, "workflow_dispatch": None, "workflow_call": None},
    )
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[schedule,workflow_call,workflow_dispatch] "
        "expected=[schedule,workflow_dispatch]"
    ]


def test_reusable_long_running_schedule_violation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "reusable.yml",
                "gate_class": "long-running",
                "reusable": "YES",
            }
        ],
    )
    write_workflow(
        tmp_path,
        "reusable.yml",
        {"workflow_call": None, "schedule": None},
    )
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_TRIGGER_SET reusable.yml class=long-running "
        "reusable=YES triggers=[schedule,workflow_call] "
        "allowed=[workflow_call] or [workflow_call,workflow_dispatch]"
    ]


def test_reusable_long_running_allows_workflow_call(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "reusable.yml",
                "gate_class": "long-running",
                "reusable": "YES",
            }
        ],
    )
    write_workflow(tmp_path, "reusable.yml", "workflow_call")
    result = run_check(tmp_path)
    assert result.exit_code == 0
    assert result.output_lines == [
        "OK: long_running_trigger_policy workflows=1 violations=0"
    ]


def test_reusable_long_running_allows_dispatch(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "reusable.yml",
                "gate_class": "long-running",
                "reusable": "YES",
            }
        ],
    )
    write_workflow(
        tmp_path,
        "reusable.yml",
        {"workflow_call": None, "workflow_dispatch": None},
    )
    result = run_check(tmp_path)
    assert result.exit_code == 0
    assert result.output_lines == [
        "OK: long_running_trigger_policy workflows=1 violations=0"
    ]


def test_pr_gate_allows_push_and_pull_request(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "gate.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "gate.yml", ["pull_request", "push"])
    result = run_check(tmp_path)
    assert result.exit_code == 0
    assert result.output_lines == [
        "OK: long_running_trigger_policy workflows=1 violations=0"
    ]


def test_missing_inventory_row(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "present.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "present.yml", "workflow_dispatch")
    write_workflow(tmp_path, "missing.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: MISSING_GATE_CLASS_ROW missing.yml"
    ]


def test_extra_inventory_row(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "present.yml",
                "gate_class": "PR-gate",
            },
            {
                "workflow_file": "extra.yml",
                "gate_class": "long-running",
            },
        ],
    )
    write_workflow(tmp_path, "present.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: EXTRA_GATE_CLASS_ROW extra.yml"
    ]


def test_invalid_gate_class_value(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "invalid.yml",
                "gate_class": "invalid",
            }
        ],
    )
    write_workflow(tmp_path, "invalid.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: GATE_CLASS_INVALID value=invalid"
    ]


def test_on_section_forms(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {"workflow_file": "str.yml", "gate_class": "PR-gate"},
            {"workflow_file": "list.yml", "gate_class": "PR-gate"},
            {"workflow_file": "dict.yml", "gate_class": "PR-gate"},
        ],
    )
    write_workflow(tmp_path, "str.yml", "workflow_dispatch")
    write_workflow(tmp_path, "list.yml", ["workflow_dispatch", "push"])
    write_workflow(tmp_path, "dict.yml", {"workflow_dispatch": None})
    result = run_check(tmp_path)
    assert result.exit_code == 0
    assert result.output_lines == [
        "OK: long_running_trigger_policy workflows=3 violations=0"
    ]


def test_missing_on_section(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "missing.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "missing.yml", None, include_on=False)
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: WORKFLOW_ON_SECTION_MISSING file=missing.yml"
    ]


def test_empty_on_dict(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "empty.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "empty.yml", {})
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: WORKFLOW_ON_SECTION_EMPTY"
    ]


def test_empty_on_list(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "empty.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "empty.yml", [])
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: WORKFLOW_ON_SECTION_EMPTY"
    ]


def test_null_on_section(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "null.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "null.yml", None)
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: WORKFLOW_ON_SECTION_MISSING"
    ]


def test_duplicate_inventory_rows(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "dup.yml",
                "gate_class": "PR-gate",
            },
            {
                "workflow_file": "dup.yml",
                "gate_class": "PR-gate",
            },
        ],
    )
    write_workflow(tmp_path, "dup.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 2
    assert result.output_lines == [
        "VIOLATION: DUPLICATE_WORKFLOW_ROW dup.yml"
    ]


def test_invalid_reusable_value(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "invalid.yml",
                "gate_class": "PR-gate",
                "reusable": "MAYBE",
            }
        ],
    )
    write_workflow(tmp_path, "invalid.yml", "workflow_dispatch")
    result = run_check(tmp_path)
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: REUSABLE_VALUE_INVALID value=MAYBE"
    ]


def test_dry_run_masks_violations(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "long-running",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "push")
    result = run_check(tmp_path, dry_run=True)
    assert result.exit_code == 0
    assert result.output_lines == [
        "VIOLATION: LONG_RUNNING_FORBIDDEN_TRIGGER long.yml class=long-running "
        "triggers=[push] forbidden=[push]",
        "VIOLATION: LONG_RUNNING_TRIGGER_SET long.yml class=long-running "
        "reusable=NO triggers=[push] expected=[schedule,workflow_dispatch]",
    ]


def test_python_version_unsupported(tmp_path: Path) -> None:
    write_pyproject(tmp_path, requires=">=3.11")
    write_contracts(
        tmp_path,
        [
            {
                "workflow_file": "long.yml",
                "gate_class": "PR-gate",
            }
        ],
    )
    write_workflow(tmp_path, "long.yml", "workflow_dispatch")
    result = run_check(tmp_path, runtime=(3, 10, 0))
    assert result.exit_code == 3
    assert result.output_lines == [
        "PARSE_ERROR: PYTHON_VERSION_UNSUPPORTED runtime=3.10.0 required=>=3.11"
    ]
