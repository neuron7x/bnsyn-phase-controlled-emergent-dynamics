"""Validate internal markdown links and anchors for a docs tree."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HTML_HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"\s+#*\s*$", "", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _iter_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _extract_headings(path: Path) -> set[str]:
    anchors: set[str] = set()
    anchor_counts: dict[str, int] = {}
    in_code_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _HEADING_RE.match(line)
        if not match:
            continue
        heading = match.group(2).strip()
        base_slug = _slugify(heading)
        if not base_slug:
            continue
        count = anchor_counts.get(base_slug, 0)
        slug = base_slug if count == 0 else f"{base_slug}-{count}"
        anchor_counts[base_slug] = count + 1
        anchors.add(slug)
    return anchors


def _extract_links(text: str) -> Iterable[str]:
    for match in _LINK_RE.finditer(text):
        yield match.group(1)
    for match in _HTML_HREF_RE.finditer(text):
        yield match.group(1)


def _is_external(link: str) -> bool:
    return link.startswith(("http://", "https://", "mailto:", "data:"))


def _resolve_link(base_path: Path, link: str, repo_root: Path) -> tuple[Path, str | None]:
    path_part, anchor = link, None
    if "#" in link:
        path_part, anchor = link.split("#", 1)
    if not path_part:
        return base_path, anchor
    if path_part.startswith("/"):
        target = repo_root / path_part.lstrip("/")
    else:
        target = (base_path.parent / path_part).resolve()
    return target, anchor


def check_docs_links(root: Path, repo_root: Path | None = None) -> list[str]:
    repo_root = repo_root or Path.cwd()
    markdown_files = _iter_markdown_files(root)
    anchors = {path: _extract_headings(path) for path in markdown_files}

    errors: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for link in _extract_links(text):
            if _is_external(link):
                continue
            if link.startswith("#"):
                target_path, anchor = _resolve_link(path, link, repo_root)
            else:
                target_path, anchor = _resolve_link(path, link, repo_root)

            if target_path.suffix:
                exists = target_path.exists()
            else:
                exists = target_path.exists()
            if not exists:
                errors.append(f"{path}:{link}: missing target")
                continue

            if anchor:
                if target_path.suffix.lower() != ".md":
                    continue
                target_anchors = anchors.get(target_path, set())
                if anchor not in target_anchors:
                    errors.append(f"{path}:{link}: missing anchor")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate documentation links")
    parser.add_argument("root", type=Path, help="Docs root directory to scan")
    args = parser.parse_args()

    errors = check_docs_links(args.root)
    if errors:
        print("Broken documentation links detected:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Documentation links check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
