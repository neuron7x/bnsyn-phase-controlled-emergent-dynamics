from __future__ import annotations

from pathlib import Path

import yaml

from tools.safety.check_safety_artifacts import (
    _parse_stpa_ids,
    main,
    validate_safety_artifacts,
)


REMOVE = object()


DEFAULT_STPA_TEXT = """
# STPA

- **L1**: loss
- **L2**: loss

| Hazard | Id |
| --- | --- |
| Input hazards | `H1` |

- Unsafe: **UCA1**
- Constraint: `SC-1`
""".strip()


def _apply_overrides(item: dict, overrides: dict[str, object]) -> None:
    for key, value in overrides.items():
        if value is REMOVE:
            item.pop(key, None)
        else:
            item[key] = value


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_artifacts(
    root: Path,
    safety_dir: str = "docs/safety",
    hazard_overrides: dict[str, object] | None = None,
    requirement_overrides: dict[str, object] | None = None,
    stpa_text: str | None = None,
) -> Path:
    safety_path = root / safety_dir
    safety_path.mkdir(parents=True, exist_ok=True)
    (safety_path / "followups.md").write_text(
        "# Safety Follow-ups\n\n## FUP-001: Example\n", encoding="utf-8"
    )
    (safety_path / "stpa.md").write_text(
        stpa_text or DEFAULT_STPA_TEXT,
        encoding="utf-8",
    )

    hazard = {
        "id": "H1",
        "title": "Example hazard",
        "loss_refs": ["L1"],
        "unsafe_control_actions": ["UCA1"],
        "safety_constraints": ["SC-1"],
        "severity": "medium",
        "likelihood": "low",
        "detectability": "high",
        "status": "unmitigated",
        "status_reason": "not yet enforced",
        "enforcement": [],
        "tests": [],
        "verification": [],
        "gates": [],
        "follow_up": "docs/safety/followups.md#FUP-001",
        "owner": "Safety",
        "last_reviewed": "2026-01-01",
    }
    if hazard_overrides:
        _apply_overrides(hazard, hazard_overrides)

    requirement = {
        "id": "REQ-EXAMPLE",
        "description": "Example requirement",
        "hazards": ["H1"],
        "safety_constraints": ["SC-1"],
        "status": "unmitigated",
        "status_reason": "not yet enforced",
        "enforcement": [],
        "tests": [],
        "verification": [],
        "gates": [],
        "follow_up": "docs/safety/followups.md#FUP-001",
        "owner": "Safety",
        "last_reviewed": "2026-01-01",
    }
    if requirement_overrides:
        _apply_overrides(requirement, requirement_overrides)

    _write_yaml(
        safety_path / "hazard_log.yml",
        {"schema_version": 1, "hazards": [hazard]},
    )
    _write_yaml(
        safety_path / "traceability.yml",
        {"schema_version": 1, "requirements": [requirement]},
    )
    return safety_path


def test_unmitigated_requires_follow_up_pass(tmp_path: Path) -> None:
    safety_dir = _write_artifacts(tmp_path)
    errors = validate_safety_artifacts(tmp_path, safety_dir)
    assert errors == []


def test_unmitigated_requires_follow_up_fail(tmp_path: Path) -> None:
    safety_dir = _write_artifacts(tmp_path, hazard_overrides={"follow_up": REMOVE})
    errors = validate_safety_artifacts(tmp_path, safety_dir)
    assert any("follow_up" in error for error in errors)


def test_additional_properties_fail(tmp_path: Path) -> None:
    safety_dir = _write_artifacts(tmp_path, hazard_overrides={"extra_field": "nope"})
    errors = validate_safety_artifacts(tmp_path, safety_dir)
    assert any("Additional properties are not allowed" in error for error in errors)


def test_parse_stpa_ids_formats() -> None:
    stpa_text = """
    - **L1** example
    - `H1` in code
    | Column | Value |
    | --- | --- |
    | UCA | **UCA1** |
    | SC | `SC-1` |
    """.strip()
    stpa_ids, errors = _parse_stpa_ids(stpa_text)
    assert errors == []
    assert stpa_ids.losses == {"L1"}
    assert stpa_ids.hazards == {"H1"}
    assert stpa_ids.ucas == {"UCA1"}
    assert stpa_ids.constraints == {"SC-1"}


def test_missing_paths_fail(tmp_path: Path) -> None:
    safety_dir = _write_artifacts(
        tmp_path,
        hazard_overrides={
            "enforcement": [{"code": "src/does_not_exist.py", "description": "missing"}]
        },
    )
    errors = validate_safety_artifacts(tmp_path, safety_dir)
    assert any("missing path" in error for error in errors)


def test_cli_args_smoke(tmp_path: Path) -> None:
    safety_dir = _write_artifacts(tmp_path)
    result = main(["--root", str(tmp_path), "--safety-dir", "docs/safety"])
    assert result == 0
