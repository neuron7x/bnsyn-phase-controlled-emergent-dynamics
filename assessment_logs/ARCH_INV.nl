     1	# Architecture & Invariants
     2	
     3	This page defines runtime invariants and safety boundaries for core modules.
     4	
     5	## Core modules
     6	
     7	- `bnsyn.validation.inputs`: strict boundary validators for array shape/dtype/finiteness.
     8	- `bnsyn.schemas.experiment`: declarative config schema and admissible simulation grid.
     9	- `bnsyn.sleep.cycle`: wake/sleep orchestration with deterministic stage transitions.
    10	- `bnsyn.provenance.manifest_builder`: deterministic manifest metadata capture.
    11	
    12	## Invariants
    13	
    14	### Input validation invariants
    15	
    16	- State vectors must be `np.ndarray`, `float64`, shape `(N,)`, finite (no NaN/Inf).
    17	- Spike vectors must be `np.ndarray`, `bool`, shape `(N,)`.
    18	- Connectivity matrices must be `np.ndarray`, `float64`, exact expected shape, finite.
    19	
    20	Failure mode:
    21	- Raises `TypeError` for non-array API boundary values.
    22	- Raises `ValueError` for dtype/shape/non-finite constraint violations.
    23	
    24	### Experiment schema invariants
    25	
    26	- `experiment.name` must match `^[a-z0-9_-]+$`.
    27	- `experiment.version` must match `^v[0-9]+$`.
    28	- `experiment.seeds` must be unique positive integers.
    29	- `simulation.dt_ms` must be one of: `0.01, 0.05, 0.1, 0.5, 1.0`.
    30	- `simulation.duration_ms / dt_ms` must be integral within tolerance.
    31	
    32	Failure mode:
    33	- Raises `pydantic.ValidationError` with explicit constraint message.
    34	
    35	### Sleep cycle invariants
    36	
    37	- `wake(duration_steps)` requires `duration_steps > 0`.
    38	- If memory recording is enabled, `record_interval` must be a positive integer.
    39	- Stage callbacks trigger exactly on stage transitions.
    40	
    41	Failure mode:
    42	- Raises `ValueError` for invalid step or interval parameters.
    43	
    44	### Provenance invariants
    45	
    46	- `git` SHA capture must use fixed non-shell command.
    47	- If git metadata is unavailable, fallback identifier is deterministic: `release-<version>`.
    48	- Manifest excludes `manifest.json` self-hash recursion.
    49	
    50	Failure mode:
    51	- Emits `UserWarning` and falls back without failing experiment generation.
    52	
    53	## Determinism controls
    54	
    55	- Hypothesis test generation is derandomized via project config.
    56	- Tests use fixed seeds for simulation reproducibility.
    57	- Fuzz-style validator test uses fixed RNG seed and bounded iterations.
