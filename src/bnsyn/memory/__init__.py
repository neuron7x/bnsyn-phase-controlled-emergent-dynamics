"""Memory subpackage for trace storage and consolidation ledger.

Parameters
----------
None

Returns
-------
None

Notes
-----
Exports MemoryTrace for pattern storage/recall and ConsolidationLedger
for audit trail of consolidation events.

References
----------
docs/SPEC.md
"""

from .ledger import ConsolidationLedger as ConsolidationLedger
from .trace import MemoryTrace as MemoryTrace

__all__ = ["MemoryTrace", "ConsolidationLedger"]
