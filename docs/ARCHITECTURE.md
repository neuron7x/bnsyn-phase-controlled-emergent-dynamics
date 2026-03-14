# AOC v1.0 Architecture

## Domains

- **Domain A — Contracts:** `aoc.contracts` defines TaskContract, InnovationBand, SigmaIndex, AuditResult, and reliability trace records.
- **Domain B — Runtime Kernel:** `aoc.controller`, `aoc.zeropoint`, `aoc.delta`, `aoc.sigma`, `aoc.termination`, `aoc.modulator`.
- **Domain C — Audit and Validation:** `aoc.audit` with independent functional, structural, and spec gates.
- **Domain D — Evidence and Traces:** `aoc.evidence` and the controller trace emission path.
- **Domain E — Delivery Surface:** `aoc.cli`, `examples/basic_task.yaml`, README run flow.

## Deterministic loop

1. Materialize `zeropoint.json`.
2. Generate deterministic candidate artifact.
3. Compute semantic/structural/functional deltas and weighted total.
4. Compute SigmaIndex.
5. Execute independent audit gates.
6. Record delta/sigma/audit/reliability traces.
7. Apply strict termination oracle.
8. Modulate constraints for next iteration if continuing.

## Fail-closed rules

- Missing/invalid invariants => FAIL.
- Critical structural audit failure => FAIL.
- Delta above innovation band max => INCONCLUSIVE (drift exceeded), never PASS.
- PASS only when all required conditions satisfy termination policy.
