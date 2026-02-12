from __future__ import annotations
from pathlib import Path

import io
import json
import os
import subprocess
import urllib.error

import pytest
from hypothesis import given, strategies as st

from scripts import classify_changes
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


def test_read_non_pr_files_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_output(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise subprocess.CalledProcessError(1, ["git"], output="fatal")

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="failed to compute non-PR diff"):
        classify_changes._read_non_pr_files()


def test_main_pull_request_requires_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "gh.out"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["classify_changes", "--github-output", str(out)])
    with pytest.raises(SystemExit, match="BLOCKER"):
        classify_changes.main()


def test_main_non_pr_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "gh.out"

    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")

    def fake_read_non_pr() -> tuple[list[str], dict[str, str]]:
        return ["docs/a.md", "src/a.py"], {"non_pr_diff": "HEAD~1...HEAD"}

    monkeypatch.setattr(classify_changes, "_read_non_pr_files", fake_read_non_pr)
    monkeypatch.setattr("sys.argv", ["classify_changes", "--github-output", str(out)])

    assert classify_changes.main() == 0
    content = out.read_text(encoding="utf-8")
    assert "code_changed=true" in content
    assert "docs_only=false" in content


@given(st.lists(st.text(min_size=1, max_size=20), max_size=50))
def test_property_unknown_keeps_docs_only_false(file_list: list[str]) -> None:
    flagged = [f for f in file_list if not f.startswith("docs/")]
    if not flagged:
        flagged = ["unknown.file"]
    flags = classify_files(flagged)
    if flags.unknown_changed:
        assert flags.docs_only is False

def test_read_non_pr_files_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "a\nb\n")
    files, meta = classify_changes._read_non_pr_files()
    assert files == ["a", "b"]
    assert meta["non_pr_diff"] == "HEAD~1...HEAD"


def test_write_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    flags = classify_files(["src/a.py"])
    classify_changes._write_outputs(str(out), flags)
    content = out.read_text(encoding="utf-8")
    assert "code_changed=true" in content


def test_api_get_json_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        del request, timeout
        return io.BytesIO(json.dumps({"filename": "x"}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="unexpected GitHub API payload type"):
        _api_get_json(url="https://example", token="token")


def test_api_get_json_urlerror_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        del request, timeout
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="failed to read PR file list"):
        _api_get_json(url="https://example", token="token")


def test_read_pr_files_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    event = {
        "pull_request": {
            "number": 7,
            "base": {"sha": "base"},
            "head": {"sha": "head"},
        }
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    calls = {"n": 0}

    def fake_api_get_json(url: str, token: str) -> list[dict[str, object]]:
        del token
        calls["n"] += 1
        if calls["n"] == 1:
            assert "page=1" in url
            return [{"filename": "src/a.py"}]
        return []

    monkeypatch.setattr(classify_changes, "_api_get_json", fake_api_get_json)
    files, meta = classify_changes._read_pr_files(str(event_path), "o/r", "token")
    assert files == ["src/a.py"]
    assert meta == {"pr_number": "7", "base_sha": "base", "head_sha": "head"}


def test_read_pr_files_missing_payload(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected pull_request payload"):
        classify_changes._read_pr_files(str(event_path), "o/r", "token")


def test_main_pull_request_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "gh.out"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", "/tmp/event")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(classify_changes, "_read_pr_files", lambda event_path, repository, token: (["docs/a.md"], {"pr_number": "1", "base_sha": "b", "head_sha": "h"}))
    monkeypatch.setattr("sys.argv", ["classify_changes", "--github-output", str(out)])
    assert classify_changes.main() == 0
    assert "docs_only=true" in out.read_text(encoding="utf-8")

def test_classify_scripts_docker_spec_and_empty() -> None:
    flags = classify_files(["scripts/a.py", "docker/Dockerfile", "specs/model.tla"])
    assert flags.validation is True
    assert flags.docker_changed is True
    assert flags.spec_changed is True
    assert flags.docs_only is False

    empty = classify_files([])
    assert empty.docs_only is False


def test_api_get_json_retryable_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        del request, timeout
        raise urllib.error.HTTPError("https://example", 503, "unavailable", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        _api_get_json(url="https://example", token="token")


def test_module_entrypoint_executes_main(tmp_path: Path) -> None:
    out = tmp_path / "gh.out"
    env = os.environ.copy()
    env["GITHUB_EVENT_NAME"] = "push"
    cmd = [
        "python",
        "-m",
        "scripts.classify_changes",
        "--github-output",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    assert "EVIDENCE:" in proc.stdout
