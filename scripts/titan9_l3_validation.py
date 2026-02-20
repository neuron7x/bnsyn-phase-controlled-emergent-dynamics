"""TITAN-9 L3 deterministic validation protocol runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PILLARS: tuple[str, ...] = (
    "GDM-Inference",
    "OAI-Alignment",
    "Meta-Architectural",
    "MS-Reliability",
    "Anthropic-Constitution",
    "xAI-Truth",
    "Mistral-Efficiency",
    "Apple-PCC",
    "Sovereign-Scale",
)

PROTOCOLS: dict[str, str] = {
    "GDM-Inference": "ALPHA-REASONING-RL-2026",
    "OAI-Alignment": "O3-INFERENCE-SCALING-LAW",
    "Meta-Architectural": "LLAMA-4.1-INFRA-STD",
    "MS-Reliability": "AZURE-GLOBAL-REACH-AGR",
    "Anthropic-Constitution": "CAI-2026-MONOSEMANTICITY",
    "xAI-Truth": "GROK-3-RT-SIGNAL",
    "Mistral-Efficiency": "MOE-2026-CID",
    "Apple-PCC": "PCC-2.0-ZERO-LEAKAGE",
    "Sovereign-Scale": "AIR-GAP-SOV-AI-2026",
}


@dataclass(frozen=True)
class PillarResult:
    pillar: str
    value: float
    benchmark: float
    weight: float
    delta: float
    protocol: str


@dataclass(frozen=True)
class ValidationResult:
    binary: str
    status: str
    weighted_score: float
    rows: tuple[PillarResult, ...]


def activation_command(target: str) -> str:
    return (
        "Activate TITAN-9 L3. Mode: Zero-Access Validation. "
        f"Target: {target}. Execute evaluation across the 9-Pillar Matrix. "
        "Provide Matrix Table and Delta-Summary. Awaiting Telemetry Input."
    )


def run_validation(
    metrics: dict[str, float], benchmarks: dict[str, float], weights: dict[str, float] | None = None
) -> ValidationResult:
    resolved_weights = {pillar: 1.0 for pillar in PILLARS}
    if weights is not None:
        resolved_weights.update(weights)

    rows: list[PillarResult] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for pillar in PILLARS:
        value = float(metrics.get(pillar, 0.0))
        benchmark = float(benchmarks.get(pillar, 1.0))
        if benchmark == 0.0:
            raise ValueError(f"Benchmark must be non-zero for pillar: {pillar}")
        weight = float(resolved_weights[pillar])
        delta = value - benchmark
        normalized = value / benchmark
        weighted_sum += normalized * weight
        total_weight += weight
        rows.append(
            PillarResult(
                pillar=pillar,
                value=value,
                benchmark=benchmark,
                weight=weight,
                delta=delta,
                protocol=PROTOCOLS[pillar],
            )
        )

    weighted_score = weighted_sum / total_weight
    if weighted_score >= 1.0:
        status = "STABLE"
        binary = "PASS"
    elif weighted_score >= 0.9:
        status = "DEGRADED"
        binary = "FAIL"
    else:
        status = "CRITICAL"
        binary = "FAIL"

    return ValidationResult(binary=binary, status=status, weighted_score=weighted_score, rows=tuple(rows))


def _format_text(result: ValidationResult) -> str:
    lines = [
        f"STATUS: {result.status}",
        f"BINARY: {result.binary}",
        f"METRIC: weighted_score={result.weighted_score:.6f}",
        "MATRIX:",
    ]
    for row in result.rows:
        lines.append(
            " | ".join(
                (
                    row.pillar,
                    f"value={row.value:.6f}",
                    f"benchmark={row.benchmark:.6f}",
                    f"delta={row.delta:.6f}",
                    f"weight={row.weight:.6f}",
                    f"reference={row.protocol}",
                )
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TITAN-9 L3 deterministic validation.")
    parser.add_argument("--telemetry", type=Path, help="Path to telemetry JSON file")
    parser.add_argument("--target", default="[Project Name/Data]", help="Target label")
    parser.add_argument("--init", action="store_true", help="Print protocol initialization command")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.init:
        print(activation_command(args.target))
        return 0
    if args.telemetry is None:
        raise SystemExit("--telemetry is required unless --init is provided")

    payload = json.loads(args.telemetry.read_text(encoding="utf-8"))
    result = run_validation(
        metrics=payload.get("metrics", {}),
        benchmarks=payload.get("benchmarks", {}),
        weights=payload.get("weights"),
    )

    if args.format == "json":
        print(
            json.dumps(
                {
                    "status": result.status,
                    "binary": result.binary,
                    "weighted_score": result.weighted_score,
                    "rows": [row.__dict__ for row in result.rows],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_format_text(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
