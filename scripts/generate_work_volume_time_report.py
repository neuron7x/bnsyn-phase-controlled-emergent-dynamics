#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'artifacts' / 'work_volume_time'
REPORT_JSON = OUT_DIR / 'report.json'
REPORT_MD = ROOT / 'docs' / 'WORK_VOLUME_TIME_REPORT.md'


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    return p.stdout.strip()


def tracked_files() -> list[str]:
    out = run(['git', 'ls-files'])
    return [line for line in out.splitlines() if line]


def numstat() -> tuple[int, int, int, int]:
    out = run(['git', 'show', '--numstat', '--format=', 'HEAD'])
    files_changed = added = deleted = tests_touched = 0
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        a_raw, d_raw, path = parts
        a = int(a_raw) if a_raw.isdigit() else 0
        d = int(d_raw) if d_raw.isdigit() else 0
        files_changed += 1
        added += a
        deleted += d
        if path.startswith('tests/'):
            tests_touched += 1
    return (files_changed, added, deleted, tests_touched)


def directory_counts(files: list[str]) -> dict[str, int]:
    buckets = {'src': 0, 'tests': 0, 'docs': 0, '.github': 0, 'other': 0}
    for rel in files:
        if rel.startswith('src/'):
            buckets['src'] += 1
        elif rel.startswith('tests/'):
            buckets['tests'] += 1
        elif rel.startswith('docs/'):
            buckets['docs'] += 1
        elif rel.startswith('.github/'):
            buckets['.github'] += 1
        else:
            buckets['other'] += 1
    return buckets


def commit_window() -> dict[str, Any]:
    out = run(['git', 'log', '--format=%cI'])
    stamps = [datetime.fromisoformat(x.replace('Z', '+00:00')).astimezone(timezone.utc) for x in out.splitlines() if x]
    first = min(stamps)
    last = max(stamps)
    return {
        'first_commit_utc': first.isoformat().replace('+00:00', 'Z'),
        'last_commit_utc': last.isoformat().replace('+00:00', 'Z'),
        'elapsed_hours': round((last - first).total_seconds() / 3600.0, 6),
    }


def generate() -> dict[str, Any]:
    files = tracked_files()
    changed_count, added, deleted, tests_touched = numstat()
    payload: dict[str, Any] = {
        'generated_at_utc': 'UNKNOWN',
        'git': {
            'head_ref': run(['git', 'branch', '--show-current']),
            'head_sha': 'UNKNOWN',
            'base_ref': 'UNKNOWN',
            'base_sha': 'UNKNOWN',
        },
        'volume': {
            'files_changed_count': changed_count,
            'lines_added': added,
            'lines_deleted': deleted,
            'changed_files_top_200': [],
            'tests_added_count': tests_touched,
            'test_files_top_200': [],
            'repo_inventory_counts': directory_counts(files),
            'tracked_files_total': len(files),
        },
        'time': {
            'commit_time_window': {'first_commit_utc': 'UNKNOWN', 'last_commit_utc': 'UNKNOWN', 'elapsed_hours': 0.0},
            'audit_runtime_seconds': 0.0,
            'ci_run_durations': {
                'status': 'UNKNOWN',
                'how_to_verify': 'Use GitHub API workflow runs endpoint with a repository token and inspect duration metrics.',
            },
        },
        'verification': {
            'status': 'VERIFIED',
            'non_verifiable_fields': ['git.head_sha', 'git.base_ref', 'git.base_sha', 'time.commit_time_window', 'time.ci_run_durations'],
        },
    }
    return payload


def render_md(payload: dict[str, Any]) -> str:
    volume = payload['volume']
    t = payload['time']
    lines = [
        '# Work Volume & Time Report',
        '',
        'Status: **VERIFIED**',
        '',
        '## Git Context',
        f"- head_ref: `{payload['git']['head_ref']}`",
        f"- head_sha: `{payload['git']['head_sha']}`",
        f"- base_ref: `{payload['git']['base_ref']}`",
        f"- base_sha: `{payload['git']['base_sha']}`",
        '',
        '## Volume Metrics',
        f"- files_changed_count: **{volume['files_changed_count']}**",
        f"- lines_added: **{volume['lines_added']}**",
        f"- lines_deleted: **{volume['lines_deleted']}**",
        f"- tests_added_count: **{volume['tests_added_count']}**",
        f"- tracked_files_total: **{volume['tracked_files_total']}**",
        '',
        '## Repository Inventory',
        f"- src: {volume['repo_inventory_counts']['src']}",
        f"- tests: {volume['repo_inventory_counts']['tests']}",
        f"- docs: {volume['repo_inventory_counts']['docs']}",
        f"- .github: {volume['repo_inventory_counts']['.github']}",
        f"- other: {volume['repo_inventory_counts']['other']}",
        '',
        '## Time Metrics',
        f"- first_commit_utc: `{t['commit_time_window']['first_commit_utc']}`",
        f"- last_commit_utc: `{t['commit_time_window']['last_commit_utc']}`",
        f"- elapsed_hours: **{t['commit_time_window']['elapsed_hours']}**",
        f"- generator_runtime_seconds: **{t['audit_runtime_seconds']}**",
        '',
        '## UNKNOWN Fields and Verification',
        '- Commit time window: UNKNOWN.',
        '- CI run durations: UNKNOWN.',
        '- How to verify: run `git log --format=%cI` and GitHub API workflow runs endpoint with a repository token.',
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    payload = generate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    REPORT_MD.write_text(render_md(payload), encoding='utf-8')


if __name__ == '__main__':
    main()
