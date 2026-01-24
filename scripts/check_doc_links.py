from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
IGNORE_PREFIXES = ("http://", "https://", "mailto:")


@dataclass(frozen=True)
class LinkIssue:
    source: Path
    link: str
    resolved: Path


def _iter_markdown_files(root: Path) -> list[Path]:
    docs_root = root / "docs"
    files = sorted(docs_root.rglob("*.md"))
    return files


def _normalize_target(link: str) -> tuple[str, str]:
    cleaned = link.strip()
    if "#" in cleaned:
        path_part, anchor = cleaned.split("#", 1)
        return path_part, anchor
    return cleaned, ""


def _is_local_link(link: str) -> bool:
    return bool(link) and not link.startswith(IGNORE_PREFIXES) and not link.startswith("#")


def _resolve_target(source: Path, target: str, repo_root: Path) -> Path:
    if target.startswith("/"):
        return (repo_root / target.lstrip("/")).resolve()
    return (source.parent / target).resolve()


def _scan_links(source: Path, repo_root: Path) -> tuple[list[LinkIssue], list[Path]]:
    text = source.read_text(encoding="utf-8")
    issues: list[LinkIssue] = []
    referenced_readmes: list[Path] = []
    for match in LINK_PATTERN.finditer(text):
        raw_link = match.group(1).strip()
        if not _is_local_link(raw_link):
            continue
        target, _anchor = _normalize_target(raw_link)
        if not target:
            continue
        resolved = _resolve_target(source, target, repo_root)
        if resolved.name.startswith("README") and resolved.suffix == ".md":
            referenced_readmes.append(resolved)
        if not resolved.exists():
            issues.append(LinkIssue(source=source, link=raw_link, resolved=resolved))
    return issues, referenced_readmes


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    markdown_files = _iter_markdown_files(repo_root)
    extra_files: list[Path] = []
    issues: list[LinkIssue] = []
    for source in markdown_files:
        found_issues, readmes = _scan_links(source, repo_root)
        issues.extend(found_issues)
        extra_files.extend(readmes)
    extra_unique = sorted({path.resolve() for path in extra_files if path.exists()})
    for source in extra_unique:
        found_issues, _readmes = _scan_links(source, repo_root)
        issues.extend(found_issues)
    if issues:
        print("Doc link check failed. Missing targets:", file=sys.stderr)
        for issue in issues:
            print(
                f"- {issue.source.relative_to(repo_root)}: {issue.link} -> "
                f"{issue.resolved.relative_to(repo_root)}",
                file=sys.stderr,
            )
        return 1
    print("OK: doc link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
