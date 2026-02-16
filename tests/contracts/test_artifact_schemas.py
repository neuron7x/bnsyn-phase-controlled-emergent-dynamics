from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator

from bnsyn.cli import _cmd_sleep_stack


class Args:
    seed = 5
    N = 48
    backend = "reference"
    steps_wake = 30
    steps_sleep = 30
    out = ""


def test_sleep_stack_artifacts_validate_against_schemas() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "run"
        Args.out = str(out_dir)
        result = _cmd_sleep_stack(Args)
        assert result == 0

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

        manifest_schema = json.loads(Path("schemas/manifest.schema.json").read_text(encoding="utf-8"))
        metrics_schema = json.loads(
            Path("schemas/sleep_stack_metrics.schema.json").read_text(encoding="utf-8")
        )

        Draft202012Validator(manifest_schema).validate(manifest)
        Draft202012Validator(metrics_schema).validate(metrics)
