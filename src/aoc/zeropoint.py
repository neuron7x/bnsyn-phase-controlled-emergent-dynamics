from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import TaskContract


class ZeroPointManager:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def materialize(self, contract: TaskContract) -> dict[str, Any]:
        payload = {
            "task_prompt": contract.task_prompt,
            "normalized_constraints": contract.normalized_constraints,
            "acceptance_criteria": contract.acceptance_criteria,
            "innovation_band": asdict(contract.innovation_band),
            "evaluator_config": contract.evaluator_config,
            "invariants": contract.invariants,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        baseline_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload["canonical_baseline_hash"] = baseline_hash
        out_file = self.run_dir / "zeropoint.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
