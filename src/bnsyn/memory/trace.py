"""Memory trace storage with pattern recall and consolidation.

Parameters
----------
None

Returns
-------
None

Notes
-----
Implements memory trace storage with capacity-limited FIFO forgetting,
similarity-based recall using cosine distance, and protein-dependent
consolidation dynamics.

References
----------
docs/SPEC.md
docs/SSOT.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


@dataclass
class MemoryTrace:
    """Store and recall memory patterns with consolidation dynamics.

    Parameters
    ----------
    capacity : int
        Maximum number of patterns to store (must be positive).
    patterns : list[Float64Array]
        Stored pattern vectors.
    importance : Float64Array
        Importance scores for each pattern.
    timestamps : Float64Array
        Storage timestamps for each pattern.
    recall_counters : Float64Array
        Number of recalls for each pattern.

    Notes
    -----
    - Uses FIFO forgetting when capacity is exceeded.
    - Recall uses cosine similarity with numpy.
    - Consolidation increases importance based on protein level.

    References
    ----------
    docs/SPEC.md
    """

    capacity: int
    patterns: list[Float64Array] = field(default_factory=list)
    importance: Float64Array = field(default_factory=lambda: np.array([], dtype=np.float64))
    timestamps: Float64Array = field(default_factory=lambda: np.array([], dtype=np.float64))
    recall_counters: Float64Array = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __post_init__(self) -> None:
        """Validate capacity parameter.

        Raises
        ------
        ValueError
            If capacity is non-positive.
        """
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def tag(self, pattern: Float64Array, importance: float) -> None:
        """Store a memory pattern with importance score.

        Parameters
        ----------
        pattern : Float64Array
            Pattern vector to store (must be 1D).
        importance : float
            Importance score for the pattern (must be non-negative).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If pattern is not 1D or importance is negative.

        Notes
        -----
        - If capacity is exceeded, removes oldest pattern (FIFO).
        - Stores pattern copy to prevent external modification.
        - Timestamp is the current pattern count.
        """
        if pattern.ndim != 1:
            raise ValueError("pattern must be 1D array")
        if importance < 0:
            raise ValueError("importance must be non-negative")

        # Remove oldest pattern if at capacity
        if len(self.patterns) >= self.capacity:
            self.patterns.pop(0)
            self.importance = self.importance[1:]
            self.timestamps = self.timestamps[1:]
            self.recall_counters = self.recall_counters[1:]

        # Store new pattern
        timestamp_value = float(len(self.patterns))
        self.patterns.append(pattern.copy())
        self.importance = np.append(self.importance, importance)
        self.timestamps = np.append(self.timestamps, timestamp_value)
        self.recall_counters = np.append(self.recall_counters, 0.0)

    def recall(self, cue: Float64Array, threshold: float) -> list[int]:
        """Retrieve patterns similar to the cue using cosine similarity.

        Parameters
        ----------
        cue : Float64Array
            Query pattern vector (must be 1D, non-zero norm).
        threshold : float
            Minimum cosine similarity for recall (must be in [-1, 1]).

        Returns
        -------
        list[int]
            Indices of patterns above threshold, sorted by similarity (descending).

        Raises
        ------
        ValueError
            If cue is not 1D, has zero norm, or threshold is out of range.

        Notes
        -----
        - Cosine similarity: dot(a, b) / (norm(a) * norm(b))
        - Increments recall counter for matched patterns.
        - Returns empty list if no patterns stored or no matches found.
        """
        if cue.ndim != 1:
            raise ValueError("cue must be 1D array")
        cue_norm = float(np.linalg.norm(cue))
        if cue_norm == 0:
            raise ValueError("cue must have non-zero norm")
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1, 1]")

        if len(self.patterns) == 0:
            return []

        # Compute cosine similarities
        similarities = []
        for pattern in self.patterns:
            if pattern.shape != cue.shape:
                similarities.append(-2.0)  # Invalid similarity for shape mismatch
                continue
            pattern_norm = float(np.linalg.norm(pattern))
            if pattern_norm == 0:
                similarities.append(-2.0)  # Invalid similarity for zero pattern
                continue
            sim = float(np.dot(cue, pattern) / (cue_norm * pattern_norm))
            similarities.append(sim)

        # Find matches above threshold
        matches = [(idx, sim) for idx, sim in enumerate(similarities) if sim >= threshold]
        matches.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in matches]

        # Increment recall counters
        for idx in indices:
            self.recall_counters[idx] += 1.0

        return indices

    def consolidate(self, protein_level: float, temperature: float) -> None:
        """Apply consolidation dynamics to stored patterns.

        Parameters
        ----------
        protein_level : float
            Global protein availability (must be in [0, 1]).
        temperature : float
            System temperature (must be non-negative).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If protein_level is out of range or temperature is negative.

        Notes
        -----
        - Increases importance for patterns based on protein level.
        - Consolidation strength scales with protein * (1 + temperature).
        - Uses multiplicative update: importance *= (1 + gain).
        """
        if not 0 <= protein_level <= 1:
            raise ValueError("protein_level must be in [0, 1]")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        if len(self.patterns) == 0:
            return

        # Consolidation gain: protein modulated by temperature
        gain = protein_level * (1.0 + temperature) * 0.1
        self.importance *= 1.0 + gain

    def get_state(self) -> dict[str, Any]:
        """Return memory trace state for inspection or serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'capacity', 'count', 'importance', 'timestamps',
            'recall_counters' keys.

        Notes
        -----
        Patterns are not included to avoid large serializations.
        """
        return {
            "capacity": self.capacity,
            "count": len(self.patterns),
            "importance": self.importance.copy(),
            "timestamps": self.timestamps.copy(),
            "recall_counters": self.recall_counters.copy(),
        }
