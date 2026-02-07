from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scripts import scan_placeholders

ALLOWED_STATUSES = {"OPEN", "IN_PROGRESS", "RESOLVED", "ACCEPTED_BY_DESIGN"}


def test_scan_placeholders_json_contract() -> None:
    findings = scan_placeholders.collect_findings()

    assert findings == sorted(
        findings, key=lambda item: (item.path, item.line, item.kind, item.signature)
    )
    assert all(item.path for item in findings)
    assert all(item.line > 0 for item in findings)
    assert all(item.kind in {"code", "docs", "script", "test"} for item in findings)


def test_registry_covers_all_scan_findings() -> None:
    registry_path = Path("docs/PLACEHOLDER_REGISTRY.md")
    registry_text = registry_path.read_text(encoding="utf-8")

    id_pattern = re.compile(r"^- ID: (PH-\d{4})$", re.MULTILINE)
    path_pattern = re.compile(r"^- Path: `([^`]+)`$", re.MULTILINE)
    status_pattern = re.compile(r"^- Status: ([A-Z_]+)$", re.MULTILINE)

    registry_ids = id_pattern.findall(registry_text)
    registry_paths = [
        entry.split(":", maxsplit=1)[0] for entry in path_pattern.findall(registry_text)
    ]
    registry_statuses = status_pattern.findall(registry_text)

    assert registry_ids
    assert len(set(registry_ids)) == len(registry_ids)
    assert len(registry_paths) == len(registry_ids)
    assert len(registry_statuses) == len(registry_ids)
    assert set(registry_statuses) <= ALLOWED_STATUSES

    findings = scan_placeholders.collect_findings()
    scan_paths = {item.path for item in findings}

    assert scan_paths <= set(registry_paths)

    open_like_paths = {
        path
        for path, status in zip(registry_paths, registry_statuses, strict=True)
        if status in {"OPEN", "IN_PROGRESS"}
    }
    assert open_like_paths <= scan_paths


def test_scan_placeholders_cli_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_placeholders", "--format", "json"])
    exit_code = scan_placeholders.main()
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
