from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from scripts import scan_placeholders

ALLOWED_STATUSES = {"OPEN", "IN_PROGRESS", "CLOSED", "ACCEPTED_BY_DESIGN"}


def _parse_registry_entries(registry_text: str) -> list[dict[str, str]]:
    entry_blocks = re.findall(r"(?ms)^- ID: PH-\d{4}$.*?(?=\n- ID: PH-\d{4}$|\Z)", registry_text)
    entries: list[dict[str, str]] = []
    field_pattern = re.compile(r"^- ([A-Za-z ]+):\s*(.*)$", re.MULTILINE)

    for block in entry_blocks:
        parsed: dict[str, str] = {}
        for key, value in field_pattern.findall(block):
            parsed[key.strip()] = value.strip()
        entries.append(parsed)
    return entries


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
    entries = _parse_registry_entries(registry_text)

    assert entries

    registry_ids = [entry["ID"] for entry in entries]
    registry_paths = [entry["Path"].strip("`").split(":", maxsplit=1)[0] for entry in entries]
    registry_statuses = [entry["Status"] for entry in entries]

    assert len(set(registry_ids)) == len(registry_ids)
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


def test_registry_schema_and_status_contract() -> None:
    registry_text = Path("docs/PLACEHOLDER_REGISTRY.md").read_text(encoding="utf-8")
    entries = _parse_registry_entries(registry_text)

    required_fields = {
        "ID",
        "Path",
        "Signature",
        "Risk",
        "Owner",
        "Fix Strategy",
        "Test Strategy",
        "Verification Test",
        "Status",
    }

    assert entries
    for entry in entries:
        assert required_fields <= set(entry)
        status = entry["Status"]
        assert status in ALLOWED_STATUSES

        if status == "OPEN":
            assert entry["Fix Strategy"].strip("`")
            assert entry["Test Strategy"].strip("`")

        if status == "CLOSED":
            assert "Evidence Ref" in entry
            assert entry["Evidence Ref"].strip("`")

    assert all(entry["Status"] != "OPEN" for entry in entries)


def test_scan_placeholders_cli_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["scan_placeholders", "--format", "json"])
    exit_code = scan_placeholders.main()
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
