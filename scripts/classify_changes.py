from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Callable, Iterable


API_TIMEOUT_SECONDS = 10
MAX_API_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
USER_AGENT = "bnsyn-ci-classifier/1.0"


@dataclass
class ChangeFlags:
    code: bool = False
    code_changed: bool = False
    tests_changed: bool = False
    validation: bool = False
    property: bool = False
    docs: bool = False
    dependency_manifest: bool = False
    workflows_changed: bool = False
    deps_changed: bool = False
    docker_changed: bool = False
    spec_changed: bool = False
    unknown_changed: bool = False
    docs_only: bool = True


def _mark_non_docs(flags: ChangeFlags) -> None:
    flags.docs_only = False


def _is_dependency_file(path: str) -> bool:
    return (
        path == "pyproject.toml"
        or path == "requirements-lock.txt"
        or (path.startswith("requirements") and path.endswith(".txt"))
        or path in {"setup.py", "setup.cfg", "poetry.lock", "uv.lock", "Pipfile", "Pipfile.lock"}
    )


def _is_docker_file(path: str) -> bool:
    return (
        path == "Dockerfile"
        or path.startswith("Dockerfile.")
        or path.startswith("docker/")
        or (path.startswith("compose") and (path.endswith(".yml") or path.endswith(".yaml")))
    )


def _is_spec_file(path: str) -> bool:
    return (
        path.startswith("specs/")
        or path.startswith("formal/")
        or path.endswith(".tla")
        or path.endswith(".cfg")
        or path.endswith(".coq")
        or path.endswith(".v")
    )


def _is_docs_file(path: str) -> bool:
    return (
        path.startswith("docs/")
        or path == "mkdocs.yml"
        or (path.startswith(".github/") and path.endswith(".md"))
        or path == "README.md"
        or path.startswith("README")
    )


def _apply_src(flags: ChangeFlags, path: str) -> bool:
    if not path.startswith("src/"):
        return False
    flags.code = True
    flags.code_changed = True
    flags.validation = True
    flags.property = True
    _mark_non_docs(flags)
    return True


def _apply_tests(flags: ChangeFlags, path: str) -> bool:
    if not path.startswith("tests/"):
        return False
    flags.code = True
    flags.code_changed = True
    flags.tests_changed = True
    flags.validation = True
    flags.property = True
    _mark_non_docs(flags)
    return True


def _apply_scripts(flags: ChangeFlags, path: str) -> bool:
    if not (path.startswith("scripts/") or path.startswith("tools/")):
        return False
    flags.validation = True
    _mark_non_docs(flags)
    return True


def _apply_workflows(flags: ChangeFlags, path: str) -> bool:
    if not (path.startswith(".github/workflows/") or path.startswith(".github/actions/")):
        return False
    flags.workflows_changed = True
    _mark_non_docs(flags)
    return True


def _apply_dependencies(flags: ChangeFlags, path: str) -> bool:
    if not _is_dependency_file(path):
        return False
    flags.dependency_manifest = True
    flags.deps_changed = True
    _mark_non_docs(flags)
    return True


def _apply_docker(flags: ChangeFlags, path: str) -> bool:
    if not _is_docker_file(path):
        return False
    flags.docker_changed = True
    _mark_non_docs(flags)
    return True


def _apply_specs(flags: ChangeFlags, path: str) -> bool:
    if not _is_spec_file(path):
        return False
    flags.spec_changed = True
    _mark_non_docs(flags)
    return True


def _apply_docs(flags: ChangeFlags, path: str) -> bool:
    if not _is_docs_file(path):
        return False
    flags.docs = True
    return True




def _classify_single_path(flags: ChangeFlags, path: str) -> bool:
    handlers: tuple[Callable[[ChangeFlags, str], bool], ...] = (
        _apply_src,
        _apply_tests,
        _apply_scripts,
        _apply_workflows,
        _apply_dependencies,
        _apply_docker,
        _apply_specs,
        _apply_docs,
    )
    for handler in handlers:
        if handler(flags, path):
            return True
    return False


def classify_files(files: Iterable[str]) -> ChangeFlags:
    flags = ChangeFlags()
    items = [f for f in files if f]

    handlers: tuple[Callable[[ChangeFlags, str], bool], ...] = (
        _apply_src,
        _apply_tests,
        _apply_scripts,
        _apply_workflows,
        _apply_dependencies,
        _apply_docker,
        _apply_specs,
        _apply_docs,
    )

    for path in items:
        matched = any(handler(flags, path) for handler in handlers)
        if not matched:
            flags.unknown_changed = True
            _mark_non_docs(flags)

    if not items:
        flags.docs_only = False

    return flags


def _api_get_json(url: str, token: str) -> list[dict[str, object]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise RuntimeError("unexpected GitHub API payload type")
                return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_API_RETRIES:
                raise RuntimeError(f"failed to read PR file list from GitHub API: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == MAX_API_RETRIES:
                raise RuntimeError(f"failed to read PR file list from GitHub API: {exc}") from exc

    if last_error is not None:
        raise RuntimeError(f"failed to read PR file list from GitHub API: {last_error}") from last_error
    raise RuntimeError("failed to read PR file list from GitHub API: unknown error")


def _read_pr_files(event_path: str, repository: str, token: str) -> tuple[list[str], dict[str, str]]:
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)

    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        raise RuntimeError("BLOCKER: expected pull_request payload in event")

    owner, repo = repository.split("/", 1)
    pr_number = pr["number"]
    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]

    files: list[str] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?{query}"
        payload = _api_get_json(url=url, token=token)

        if not payload:
            break

        files.extend(item["filename"] for item in payload)
        if len(payload) < 100:
            break
        page += 1

    return files, {
        "pr_number": str(pr_number),
        "base_sha": base_sha,
        "head_sha": head_sha,
    }


def _read_non_pr_files() -> tuple[list[str], dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1...HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to compute non-PR diff: {exc.output}") from exc

    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files, {"non_pr_diff": "HEAD~1...HEAD"}


def _write_outputs(path: str, flags: ChangeFlags) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in asdict(flags).items():
            handle.write(f"{key}={str(value).lower()}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-output", required=True)
    args = parser.parse_args()

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")

    if event_name == "pull_request":
        if not all([event_path, repository, token]):
            raise SystemExit("BLOCKER: missing required env for pull_request classification")
        files, meta = _read_pr_files(event_path=event_path, repository=repository, token=token)
        print(
            "EVIDENCE: "
            f"pr_number={meta['pr_number']} base_sha={meta['base_sha']} head_sha={meta['head_sha']}"
        )
    else:
        files, meta = _read_non_pr_files()
        print(f"EVIDENCE: non_pr_diff={meta['non_pr_diff']}")

    flags = classify_files(files)
    _write_outputs(args.github_output, flags)

    changed_hash = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
    print(f"EVIDENCE: changed_files_count={len(files)} changed_files_sha256={changed_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
