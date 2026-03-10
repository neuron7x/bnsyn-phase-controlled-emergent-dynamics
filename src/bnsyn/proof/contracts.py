"""Canonical proof artifact contracts and mode constants."""

from __future__ import annotations

CANONICAL_BASE_CONTRACT = "canonical-base"
CANONICAL_EXPORT_PROOF_CONTRACT = "canonical-export-proof"

CANONICAL_BASE_COMMAND = "bnsyn run --profile canonical --plot"
CANONICAL_EXPORT_PROOF_COMMAND = "bnsyn run --profile canonical --plot --export-proof"

BASE_ARTIFACTS: tuple[str, ...] = (
    "emergence_plot.png",
    "summary_metrics.json",
    "criticality_report.json",
    "avalanche_report.json",
    "phase_space_report.json",
    "run_manifest.json",
)

EXPORT_PROOF_ARTIFACTS: tuple[str, ...] = BASE_ARTIFACTS + ("proof_report.json",)


def command_for_export_proof(export_proof: bool) -> str:
    return CANONICAL_EXPORT_PROOF_COMMAND if export_proof else CANONICAL_BASE_COMMAND


def bundle_contract_for_export_proof(export_proof: bool) -> str:
    return CANONICAL_EXPORT_PROOF_CONTRACT if export_proof else CANONICAL_BASE_CONTRACT


def artifacts_for_export_proof(export_proof: bool) -> tuple[str, ...]:
    return EXPORT_PROOF_ARTIFACTS if export_proof else BASE_ARTIFACTS
