"""Unified telemetry logger and analyzer for canonical/AOC runs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_TYPES: tuple[str, ...] = (
    "stage_started",
    "stage_finished",
    "stage_failed",
    "bundle_validated",
    "audit_verdict_emitted",
    "resume_used",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass(slots=True)
class TelemetryLogger:
    artifact_dir: Path
    run_id: str
    seed: int | None = None
    default_stage: str | None = None
    events_path: Path = field(init=False)
    summary_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_dir = self.artifact_dir.resolve()
        logs_dir = self.artifact_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = logs_dir / "events.jsonl"
        self.summary_path = self.artifact_dir / "run_health_summary.json"

    @classmethod
    def for_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        default_stage: str | None = None,
    ) -> TelemetryLogger:
        return cls(
            artifact_dir=_coerce_path(artifact_dir),
            run_id=run_id or new_run_id(),
            seed=seed,
            default_stage=default_stage,
        )

    def emit(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        seed: int | None = None,
        duration_ms: int | None = None,
        status: str | None = None,
        failure_reason: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported telemetry event_type: {event_type}")
        event: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "event_type": event_type,
            "run_id": self.run_id,
            "stage": stage or self.default_stage or "unknown",
            "seed": self.seed if seed is None else seed,
            "artifact_dir": self.artifact_dir.as_posix(),
            "duration_ms": duration_ms,
            "status": status,
            "failure_reason": failure_reason,
        }
        if extra:
            event.update(extra)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.refresh_summary(run_id=self.run_id)
        return event

    def time_stage(self, stage: str) -> float:
        self.emit("stage_started", stage=stage, status="running")
        return time.perf_counter()

    def finish_stage(self, stage: str, started_at: float, *, status: str = "completed", **extra: Any) -> dict[str, Any]:
        duration_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))
        return self.emit("stage_finished", stage=stage, duration_ms=duration_ms, status=status, **extra)

    def fail_stage(self, stage: str, started_at: float, *, failure_reason: str, status: str = "failed", **extra: Any) -> dict[str, Any]:
        duration_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))
        return self.emit(
            "stage_failed",
            stage=stage,
            duration_ms=duration_ms,
            status=status,
            failure_reason=failure_reason,
            **extra,
        )

    def refresh_summary(self, *, run_id: str | None = None) -> Path:
        return write_run_health_summary(self.events_path, output_path=self.summary_path, run_id=run_id)


def _load_events(events_path: str | Path) -> list[dict[str, Any]]:
    path = _coerce_path(events_path)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def latest_run_id(events_path: str | Path) -> str | None:
    events = _load_events(events_path)
    if not events:
        return None
    run_id = events[-1].get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _event_matches_run(event: dict[str, Any], run_id: str | None) -> bool:
    if run_id is None:
        return True
    return event.get("run_id") == run_id


def build_run_health_summary(events_path: str | Path, *, run_id: str | None = None) -> dict[str, Any]:
    events = [event for event in _load_events(events_path) if _event_matches_run(event, run_id)]
    resolved_run_id = run_id or (events[-1].get("run_id") if events else None)
    if resolved_run_id is not None:
        events = [event for event in events if event.get("run_id") == resolved_run_id]
    stage_details: dict[str, dict[str, Any]] = {}
    first_failure_stage: str | None = None
    first_failure_reason: str | None = None
    bundle_validation_status: str | None = None
    audit_verdict_status: str | None = None
    resume_events: list[dict[str, Any]] = []

    for event in events:
        stage_name = str(event.get("stage") or "unknown")
        detail = stage_details.setdefault(
            stage_name,
            {
                "stage": stage_name,
                "started_at": None,
                "finished_at": None,
                "duration_ms": None,
                "status": None,
                "failure_reason": None,
                "event_counts": {name: 0 for name in EVENT_TYPES},
            },
        )
        event_type = str(event.get("event_type") or "")
        if event_type in detail["event_counts"]:
            detail["event_counts"][event_type] += 1
        if event_type == "stage_started" and detail["started_at"] is None:
            detail["started_at"] = event.get("timestamp")
            detail["status"] = event.get("status") or "running"
        elif event_type == "stage_finished":
            detail["finished_at"] = event.get("timestamp")
            detail["duration_ms"] = event.get("duration_ms")
            detail["status"] = event.get("status") or "completed"
        elif event_type == "stage_failed":
            detail["finished_at"] = event.get("timestamp")
            detail["duration_ms"] = event.get("duration_ms")
            detail["status"] = event.get("status") or "failed"
            detail["failure_reason"] = event.get("failure_reason")
            if first_failure_stage is None:
                first_failure_stage = stage_name
                first_failure_reason = event.get("failure_reason") if isinstance(event.get("failure_reason"), str) else None
        elif event_type == "bundle_validated":
            bundle_validation_status = str(event.get("status") or "unknown")
        elif event_type == "audit_verdict_emitted":
            audit_verdict_status = str(event.get("status") or "unknown")
        elif event_type == "resume_used":
            resume_events.append(event)

    status_values = [detail["status"] for detail in stage_details.values() if isinstance(detail.get("status"), str)]
    failed = any(status == "failed" for status in status_values)
    completed = bool(stage_details) and not failed and all(status in {"completed", "skipped", "validated", "PASS", "FAIL"} or status == "PASS" for status in status_values if status is not None)

    return {
        "schema_version": "1.0.0",
        "run_id": resolved_run_id,
        "artifact_dir": events[-1].get("artifact_dir") if events else None,
        "seed": events[-1].get("seed") if events else None,
        "event_stream_path": _coerce_path(events_path).as_posix(),
        "event_count": len(events),
        "stage_count": len(stage_details),
        "first_event_at": events[0].get("timestamp") if events else None,
        "last_event_at": events[-1].get("timestamp") if events else None,
        "failed": failed,
        "completed": completed,
        "first_failure_stage": first_failure_stage,
        "first_failure_reason": first_failure_reason,
        "bundle_validation_status": bundle_validation_status,
        "audit_verdict_status": audit_verdict_status,
        "resume_count": len(resume_events),
        "resume_stages": [event.get("stage") for event in resume_events],
        "stages": [stage_details[name] for name in sorted(stage_details)],
    }


def write_run_health_summary(
    events_path: str | Path,
    *,
    output_path: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    events_file = _coerce_path(events_path)
    summary = build_run_health_summary(events_file, run_id=run_id)
    destination = _coerce_path(output_path) if output_path is not None else events_file.parent.parent / "run_health_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
