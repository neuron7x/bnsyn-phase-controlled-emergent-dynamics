     1	# CI Gates and Test Selection Strategy
     2	
     3	**Version:** 2.0  
     4	**Date:** 2026-01-27  
     5	**Repository:** neuron7x/bnsyn-phase-controlled-emergent-dynamics
     6	
     7	---
     8	
     9	## Overview
    10	
    11	This document defines the **3-tier test selection strategy** for BN-Syn, balancing fast PR feedback with comprehensive validation coverage.
    12	
    13	### Design Principles
    14	
    15	1. **BLOCKING gates** run on every PR (fast, <10 min total)
    16	2. **NON-BLOCKING validation** runs on schedule (thorough, 30+ min)
    17	3. **Isolation** ensures validation tests never block PRs
    18	4. **Observability** through artifacts, summaries, and logs
    19	
    20	---
    21	
    22	## Test Tiers
    23	
    24	### Tier 1: BLOCKING (PR Gates) ⚡
    25	
    26	**Trigger:** Every push, every PR  
    27	**Runtime:** <10 minutes total  
    28	**Purpose:** Fast feedback, prevent obviously broken code  
    29	**Policy:** Passing is mandatory for merge
    30	
    31	#### Included Tests
    32	
    33	1. **Smoke Suite** (`-m "not (validation or property)"`)
    34	   - Unit tests, fast integration tests, edge cases
    35	   - Target: ~85% code coverage
    36	   - Runtime: ~5 minutes
    37	
    38	2. **SSOT Validation**
    39	   - Bibliography, claims, **claims coverage** (NEW, CLM-0011)
    40	   - Normative tags, TierS misuse
    41	   - Runtime: <30 seconds
    42	
    43	3. **CLM-0011 Enforcement** - Ensures all normative claims have complete evidence
    44	4. **Build & Import** - Package build + import checks
    45	5. **Quality Gates** - Ruff, Pylint, Mypy
    46	6. **Security Scans** - Gitleaks, pip-audit, Bandit
    47	   - pip-audit runs with `--desc --format json` and stores `artifacts/pip-audit.json` for traceability
    48	
    49	**Total Runtime:** ~8-10 minutes
    50	
    51	---
    52	
    53	### Tier 2: NON-BLOCKING Validation 🔬
    54	
    55	**Trigger:** Daily 2 AM UTC + manual (mode: `elite`)  
    56	**Runtime:** ~30 minutes  
    57	**Policy:** Informational only
    58	
    59	- **Validation Tests** (`-m validation`): Tests validating empirical claims
    60	- **Property Tests** (`-m property`): Hypothesis tests with thorough profile (1000 examples)
    61	- **Chaos Tests** (`-m "validation and chaos"`): Fault injection resilience tests
    62	
    63	**Workflows:** `ci-validation.yml` (modes: `elite`, `property`, `chaos`)
    64	
    65	---
    66	
    67	### Tier 3: Performance Tracking 📊
    68	
    69	**Trigger:** Weekly Sunday 3 AM UTC + manual  
    70	**Runtime:** ~20 minutes  
    71	**Policy:** Non-blocking
    72	
    73	- 8 benchmark comparisons against golden baseline
    74	- **Workflow:** `benchmarks.yml` (tier=elite)
    75	
    76	---
    77	
    78	## Coverage Trend History
    79	
    80	Coverage trend observability is emitted directly by the reusable pytest workflow used by smoke/unit jobs.
    81	
    82	- Artifact name (stable): `coverage-trend-metrics`
    83	- Payload files: `coverage-trend.json`, `coverage-trend.csv`
    84	- Fields: `timestamp`, `sha`, `branch`, `total_coverage` (0..100 scale), `coverage_state` (critical/low/moderate/high/excellent)
    85	- Quantization thresholds: critical <50, low <70, moderate <85, high <95, excellent >=95
    86	- Retention policy: 90 days
    87	- Coverage XML remains mandatory and is uploaded as `coverage-xml-<sha>`
    88	
    89	Viewer entry point:
    90	1. Open GitHub Actions run for `ci-pr-atomic` or `ci-smoke`.
    91	2. Open job `tests-smoke`.
    92	3. Download artifact `coverage-trend-metrics`.
    93	
    94	---
    95	
    96	## Makefile Targets
    97	
    98	```bash
    99	make test                      # Smoke suite (excludes validation/property)
   100	make test-validation           # Run validation suite locally
   101	make validate-claims-coverage  # Check claims→evidence coverage
   102	make docs-evidence             # Regenerate EVIDENCE_COVERAGE.md
   103	```
   104	
   105	---
   106	
   107	## Summary Table
   108	
   109	| Tier | Marker | Trigger | Runtime | Blocks PR | Workflow |
   110	|------|--------|---------|---------|-----------|----------|
   111	| **BLOCKING** | smoke | Every PR | ~8 min | ✅ YES | `ci-pr.yml` |
   112	| **VALIDATION** | `@pytest.mark.validation` | Daily 2 AM | varies | ❌ NO | `ci-validation.yml` (mode: `elite`) |
   113	| **PROPERTY** | `@pytest.mark.property` | Daily 2:30 AM | ~10 min | ❌ NO | `ci-validation.yml` (mode: `property`) |
   114	| **CHAOS** | `@pytest.mark.chaos` | Daily 4 AM | ~20 min | ❌ NO | `ci-validation.yml` (mode: `chaos`) |
   115	| **BENCHMARKS** | N/A | Weekly Sun 3 AM | ~20 min | ❌ NO | `benchmarks.yml` (tier=elite) |
