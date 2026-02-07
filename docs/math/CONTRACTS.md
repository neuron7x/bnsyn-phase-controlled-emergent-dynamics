# Mathematical Contracts

## Artifact Family: derived_data (`results/*.csv`, `benchmarks/*.json`)

- **Dimensions/Units**
  - scalar metrics are dimensionless unless explicitly documented in source benchmark docs.
- **Shape invariants**
  - JSON artifacts must parse to object or list structures.
  - CSV artifacts must parse with a stable header row.
- **Range constraints**
  - numeric values must be finite and satisfy `abs(x) <= 1e12`.
- **Conservation laws**
  - probability-like vectors (when present) must be represented as finite reals and are checked by numeric sanity gates.
- **Monotonicity**
  - not globally enforced across all benchmark outputs; category marked `MANUAL_REVIEW_REQUIRED`.
- **Domain assumptions**
  - all numeric fields are in real domain and cannot be NaN/Inf.
- **Tolerance**
  - anomaly scan threshold uses `|z| <= 4.0` with population statistics.

## Artifact Family: numeric_code (`src/**/*.py`)

- **Shape invariants**
  - source modules must be non-empty UTF-8 text files.
- **Range constraints**
  - static hazard scan identifies risky subtraction, division-by-zero literals, and exponential overflow patterns.
- **Domain assumptions**
  - hazard scan is informational and requires manual review for confirmed instability fixes.
- **Tolerance**
  - hazard scan has zero tolerance for parser errors; parse failures are `FAIL`.

## Artifact Family: reports/config (`docs/**/*.md`, root config)

- **Shape invariants**
  - text/config artifacts must be non-empty and hash-stable.
- **Domain assumptions**
  - deterministic provenance and checksum verification are mandatory.
