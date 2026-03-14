from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any


def sha256_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EvidenceWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_trace(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.write_json(name, {"trace": rows})

    def emit_bundle(self) -> None:
        bundle = self.run_dir / "evidence_bundle"
        bundle.mkdir(exist_ok=True)
        for fname in [
            "zeropoint.json",
            "run_summary.json",
            "sigma_trace.json",
            "delta_trace.json",
            "audit_trace.json",
            "auditor_reliability_trace.json",
            "termination_verdict.json",
            "final_artifact.json",
        ]:
            src = self.run_dir / fname
            if src.exists():
                shutil.copy2(src, bundle / fname)


def as_dict_dataclass(instance: object) -> dict[str, Any]:
    return asdict(instance)
