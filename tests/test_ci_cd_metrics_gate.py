from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_ci_cd_counts_nonzero_when_workflows_exist() -> None:
    subprocess.run(['python', 'scripts/compute_ci_cd_metrics.py'], check=True)
    payload = json.loads(Path('artifacts/math_audit/ci_cd_metrics.json').read_text(encoding='utf-8'))
    assert payload['workflow_count'] >= 0
    if payload['workflow_count'] > 0:
        assert payload['job_count'] > 0
        assert payload['step_count'] > 0
