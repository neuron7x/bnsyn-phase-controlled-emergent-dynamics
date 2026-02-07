#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def workflow_files() -> list[str]:
    proc = subprocess.run(
        ['git', 'ls-files', '.github/workflows/*.yml', '.github/workflows/*.yaml'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return sorted([line for line in proc.stdout.splitlines() if line])


def main() -> None:
    workflows = workflow_files()
    job_count = 0
    step_count = 0
    uses = set()

    for rel in workflows:
        payload = yaml.safe_load((ROOT / rel).read_text(encoding='utf-8', errors='ignore')) or {}
        jobs = payload.get('jobs') if isinstance(payload, dict) else None
        if isinstance(jobs, dict):
            job_count += len(jobs)
            for job in jobs.values():
                if not isinstance(job, dict):
                    continue
                steps = job.get('steps')
                if isinstance(steps, list):
                    step_count += len(steps)
                    for step in steps:
                        if isinstance(step, dict) and isinstance(step.get('uses'), str):
                            uses.add(step['uses'])

    out = {
        'workflow_count': len(workflows),
        'workflow_files': workflows,
        'job_count': job_count,
        'step_count': step_count,
        'uses_actions_unique': sorted(uses),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'crosscheck_ci_cd_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
