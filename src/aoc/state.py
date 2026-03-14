from __future__ import annotations

from dataclasses import dataclass

from .contracts import AuditResult, InnovationBand, SigmaIndex


@dataclass
class AOCState:
    iteration: int
    zeropoint_hash: str
    current_artifact_hash: str
    delta_from_zeropoint: float
    sigma: SigmaIndex
    audit: AuditResult
    band: InnovationBand
    status: str
