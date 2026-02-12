from __future__ import annotations

import io
import json
import urllib.error

import pytest

from scripts.classify_changes import _api_get_json, classify_files


def test_docs_only_classification() -> None:
    flags = classify_files(["docs/ci.md", "README.md", ".github/PR_TEMPLATE.md"])
    assert flags.docs is True
    assert flags.docs_only is True
    assert flags.unknown_changed is False


def test_unknown_file_forces_fail_closed() -> None:
    flags = classify_files(["random.bin"])
    assert flags.docs_only is False
    assert flags.unknown_changed is True


def test_code_and_dependencies_classification() -> None:
    flags = classify_files(["src/main.py", "tests/test_main.py", "requirements-lock.txt"])
    assert flags.code_changed is True
    assert flags.tests_changed is True
    assert flags.deps_changed is True
    assert flags.dependency_manifest is True
    assert flags.docs_only is False


def test_workflow_and_actions_are_sensitive() -> None:
    flags = classify_files([".github/workflows/ci.yml", ".github/actions/x/action.yml"])
    assert flags.workflows_changed is True
    assert flags.docs_only is False


def test_api_get_json_retries_on_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        del request, timeout
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError("https://example", 503, "retry", {}, None)
        return io.BytesIO(json.dumps([{"filename": "docs/x.md"}]).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    payload = _api_get_json(url="https://example", token="token")
    assert payload == [{"filename": "docs/x.md"}]
    assert attempts["n"] == 3


def test_api_get_json_fails_closed_on_non_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        del request, timeout
        raise urllib.error.HTTPError("https://example", 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        _api_get_json(url="https://example", token="token")
