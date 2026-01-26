"""Smoke tests for memory trace and consolidation ledger.

Parameters
----------
None

Returns
-------
None

Notes
-----
Tests memory tagging, recall, consolidation, and ledger.

References
----------
docs/features/memory.md
"""

from __future__ import annotations

import numpy as np
import pytest

from bnsyn.memory import ConsolidationLedger, MemoryTrace


def test_memory_trace_basic() -> None:
    """Test basic MemoryTrace functionality."""
    trace = MemoryTrace(capacity=10)
    state = trace.get_state()
    assert state["count"] == 0

    # tag a pattern
    pattern = np.ones(5)
    trace.tag(pattern, importance=0.5)
    state = trace.get_state()
    assert state["count"] == 1
    assert len(trace.patterns) == 1


def test_memory_capacity() -> None:
    """Test capacity enforcement."""
    trace = MemoryTrace(capacity=3)

    for i in range(5):
        pattern = np.ones(5) * i
        trace.tag(pattern, importance=0.5)

    state = trace.get_state()
    assert state["count"] == 3  # capped at capacity


def test_memory_recall() -> None:
    """Test pattern recall."""
    trace = MemoryTrace(capacity=10)

    target = np.array([1.0, 0.0, 0.0])
    trace.tag(target, importance=0.8)

    # recall with exact match
    recalled = trace.recall(target, threshold=0.9)
    assert len(recalled) > 0


def test_memory_consolidation() -> None:
    """Test consolidation."""
    trace = MemoryTrace(capacity=10)

    pattern = np.ones(5)
    trace.tag(pattern, importance=0.5)

    # consolidate
    trace.consolidate(protein_level=0.9, temperature=1.0)

    state = trace.get_state()
    # importance should increase
    assert state["importance"][0] > 0.5


def test_ledger_basic() -> None:
    """Test basic ConsolidationLedger functionality."""
    ledger = ConsolidationLedger()
    assert len(ledger.get_history()) == 0

    # record event
    ledger.record_event(gate=0.8, temperature=1.0, step=100)

    history = ledger.get_history()
    assert len(history) == 1
    assert history[0]["gate"] == 0.8


def test_ledger_hash() -> None:
    """Test ledger hash stability."""
    ledger1 = ConsolidationLedger()
    ledger2 = ConsolidationLedger()

    ledger1.record_event(gate=0.5, temperature=1.0, step=100)
    ledger2.record_event(gate=0.5, temperature=1.0, step=100)

    # hashes should match for identical events
    assert ledger1.compute_hash() == ledger2.compute_hash()


def test_ledger_with_dualweights() -> None:
    """Test ledger with DualWeights info."""
    ledger = ConsolidationLedger()

    # record with tags
    ledger.record_event(
        gate=0.9,
        temperature=0.8,
        step=200,
        dw_tags=np.array([True, False, True]),
        dw_protein=0.75,
    )

    history = ledger.get_history()
    assert len(history) == 1
    assert history[0]["dw_protein"] == 0.75
