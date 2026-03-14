from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import AuditResult, InnovationBand, SigmaIndex

AOCStatus = Literal["INIT", "RUNNING", "STABILIZED", "FAILED", "MAX_ITER", "INCONCLUSIVE"]


@dataclass
class AOCState:
    iteration: int
    zeropoint_hash: str
    current_artifact_hash: str
    delta_from_zeropoint: float
    sigma: SigmaIndex
    audit: AuditResult
    band: InnovationBand
    status: AOCStatus

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
