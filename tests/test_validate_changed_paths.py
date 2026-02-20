from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import validate_changed_paths


def test_changed_paths_parses_status_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    porcelain = " M scripts/validate_changed_paths.py\n?? tests/test_validate_changed_paths.py\n"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=porcelain, stderr="")

    monkeypatch.setattr(validate_changed_paths.subprocess, "run", fake_run)

    assert validate_changed_paths._changed_paths() == [
        "scripts/validate_changed_paths.py",
        "tests/test_validate_changed_paths.py",
    ]


def test_changed_paths_uses_rename_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    porcelain = "R  docs/old.md -> docs/new.md\n"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=porcelain, stderr="")

    monkeypatch.setattr(validate_changed_paths.subprocess, "run", fake_run)

    assert validate_changed_paths._changed_paths() == ["docs/new.md"]


def test_main_fails_for_untracked_disallowed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    porcelain = "?? src/new_module.py\n"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=porcelain, stderr="")

    monkeypatch.setattr(validate_changed_paths.subprocess, "run", fake_run)

    with pytest.raises(validate_changed_paths.ValidationError, match="changed paths outside allowlist"):
        validate_changed_paths.main()
