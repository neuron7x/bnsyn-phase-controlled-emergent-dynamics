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
from typing import Iterable


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


def classify_files(files: Iterable[str]) -> ChangeFlags:
    flags = ChangeFlags()
    items = [f for f in files if f]

    for file in items:
        if file.startswith("src/"):
            flags.code = True
            flags.code_changed = True
            flags.validation = True
            flags.property = True
            _mark_non_docs(flags)
            continue
        if file.startswith("tests/"):
            flags.code = True
            flags.code_changed = True
            flags.tests_changed = True
            flags.validation = True
            flags.property = True
            _mark_non_docs(flags)
            continue
        if file.startswith("scripts/") or file.startswith("tools/"):
            flags.validation = True
            _mark_non_docs(flags)
            continue
        if file.startswith(".github/workflows/") or file.startswith(".github/actions/"):
            flags.workflows_changed = True
            _mark_non_docs(flags)
            continue
        if (
            file == "pyproject.toml"
            or file == "requirements-lock.txt"
            or file.endswith(".txt") and file.startswith("requirements")
            or file in {"setup.py", "setup.cfg", "poetry.lock", "uv.lock", "Pipfile", "Pipfile.lock"}
        ):
            flags.dependency_manifest = True
            flags.deps_changed = True
            _mark_non_docs(flags)
            continue
        if (
            file == "Dockerfile"
            or file.startswith("Dockerfile.")
            or file.startswith("docker/")
            or file.startswith("compose") and (file.endswith(".yml") or file.endswith(".yaml"))
        ):
            flags.docker_changed = True
            _mark_non_docs(flags)
            continue
        if (
            file.startswith("specs/")
            or file.startswith("formal/")
            or file.endswith(".tla")
            or file.endswith(".cfg")
            or file.endswith(".coq")
            or file.endswith(".v")
        ):
            flags.spec_changed = True
            _mark_non_docs(flags)
            continue
        if (
            file.startswith("docs/")
            or file == "mkdocs.yml"
            or file.startswith(".github/") and file.endswith(".md")
            or file == "README.md"
            or file.startswith("README")
        ):
            flags.docs = True
            continue

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
