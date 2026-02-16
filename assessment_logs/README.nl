     1	# BN-Syn Thermostated Bio-AI System
     2	
     3	BN-Syn is the deterministic reference implementation of the BN-Syn Thermostated Bio-AI System defined by the specification and governance artifacts in this repository.
     4	
     5	[![ci-pr](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml)
     6	[![ci-validation](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-validation.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-validation.yml)
     7	[![codeql](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/codeql.yml)
     8	[![codecov](https://codecov.io/gh/neuron7x/bnsyn-phase-controlled-emergent-dynamics/branch/main/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/gh/neuron7x/bnsyn-phase-controlled-emergent-dynamics)
     9	[![ci-pr-atomic](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml)
    10	
    11	BN-Syn is a deterministic, research-grade Bio-AI system that formalizes phase-controlled emergent dynamics with strict evidence and governance controls. This repository is the *single source of truth* for specifications, experiments, validation, and compliance artifacts.
    12	
    13	## Contents
    14	
    15	- [Quality Assurance](#quality-assurance)
    16	- [Project Status](#project-status)
    17	- [Validation & Testing Strategy](#validation--testing-strategy)
    18	- [Results: Temperature-Controlled Consolidation](#results-temperature-controlled-consolidation)
    19	- [Sleep–Emergence Stack](#sleepemergence-stack)
    20	- [Interactive Demo](#-interactive-demo)
    21	- [Start Here](#start-here)
    22	- [Repository Contract](#repository-contract)
    23	- [Quickstart](#quickstart)
    24	- [Codebase Readiness](#codebase-readiness)
    25	- [Demo Runbook](#demo-runbook)
    26	- [Release Notes](#release-notes)
    27	- [Development Workflow](#development-workflow)
    28	- [CI on Pull Requests](#ci-on-pull-requests)
    29	- [Architecture at a Glance](#architecture-at-a-glance)
    30	- [Evidence & Bibliography](#evidence--bibliography)
    31	- [How to Cite](#how-to-cite)
    32	- [License / Security / Contributing](#license--security--contributing)
    33	
    34	## Quality Assurance
    35	
    36	Quality gates are command-verifiable and documented in one place:
    37	
    38	- Canonical local test/coverage commands: [`docs/TESTING.md`](docs/TESTING.md)
    39	- CI gate definitions: [`docs/CI_GATES.md`](docs/CI_GATES.md)
    40	- Workflow contracts: [`.github/WORKFLOW_CONTRACTS.md`](.github/WORKFLOW_CONTRACTS.md)
    41	- Evidence mapping: [`docs/EVIDENCE_COVERAGE.md`](docs/EVIDENCE_COVERAGE.md)
    42	
    43	No aggregate quality percentages are published in README; only reproducible command outputs and artifacts are considered normative.
    44	
    45	---
    46	
    47	## Project Status
    48	
    49	Official status declaration: [`docs/STATUS.md`](docs/STATUS.md).
    50	
    51	## Validation & Testing Strategy
    52	
    53	Canonical commands for install, test, coverage, and coverage gate live in [`docs/TESTING.md`](docs/TESTING.md).
    54	
    55	BN-Syn implements a **3-tier test selection strategy** for optimal coverage without blocking development:
    56	
    57	### Tier 1: BLOCKING (PR Gates) ⚡
    58	**Every PR, ~8 min** — Fast smoke tests, SSOT validation, claims coverage enforcement (CLM-0011), security scans
    59	
    60	### Tier 2: NON-BLOCKING Validation 🔬
    61	**Daily 2 AM UTC** — 10 scientific validation tests + 8 property-based invariants (Hypothesis)
    62	
    63	### Tier 3: Performance Tracking 📊
    64	**Weekly Sunday 3 AM UTC** — Benchmark regression detection against golden baseline
    65	
    66	**Learn More:**
    67	- [CI Gates](docs/CI_GATES.md) — Test selection strategy
    68	- [Test Protocol](docs/ACTIONS_TEST_PROTOCOL.md) — GitHub Actions testing guide
    69	- [Evidence Coverage](docs/EVIDENCE_COVERAGE.md) — Claims→Evidence traceability
    70	
    71	---
    72	
    73	## Results: Temperature-Controlled Consolidation
    74	
    75	BN-Syn demonstrates **phase-controlled emergent dynamics** through temperature-gated synaptic consolidation. Our flagship experiment (v2) validates that piecewise cooling schedules improve consolidation stability while maintaining active protein synthesis and consolidation, demonstrating stability gains without trivially suppressing plasticity.
    76	
    77	### Key Findings (v2: Piecewise Cooling with Non-Trivial Consolidation)
    78	
    79	| Condition | w_cons Variance | w_total Variance | Protein Level | Reduction vs Fixed-High |
    80	|-----------|-----------------|------------------|---------------|-------------------------|
    81	| **cooling_piecewise** | 0.003039 | 0.010302 | 0.9999 | **18.77%** ✓ |
    82	| fixed_high | 0.003600 | 0.012683 | 0.9999 | baseline |
    83	| fixed_low | 0.000000 | 0.000000 | 0.0002 | — |
    84	| random_T | 0.004736 | 0.016460 | 0.9999 | worse |
    85	
    86	**Hypothesis H1 SUPPORTED**: Piecewise cooling reduces w_total stability variance by **18.77%** while maintaining active consolidation (protein=0.9999, |w_cons|=0.0012), exceeding the ≥10% target without trivially disabling plasticity.
    87	
    88	**v1 showed extreme variance reduction (99.996%) but achieved this by suppressing consolidation; v2 demonstrates stability gains with protein synthesis active.**
    89	
    90	### Visualizations
    91	
    92	![Stability Comparison](figures/temp_ablation_v2/hero.png)
    93	
    94	*Stability variance across temperature conditions (20 seeds). Lower variance indicates more reproducible consolidation.*
    95	
    96	![Comparison Grid](figures/temp_ablation_v2/comparison_grid.png)
    97	
    98	*Multi-panel view: temperature profiles, weight dynamics, protein synthesis, and stability metrics.*
    99	
   100	### Reproduce the Flagship Experiment
   101	
   102	```bash
   103	# Install with visualization dependencies
   104	pip install -e ".[dev,viz]"
   105	
   106	# Run full validation experiment v2 (20 seeds, ~2-3 minutes)
   107	python -m experiments.runner temp_ablation_v2
   108	
   109	# Generate visualizations
   110	python -m scripts.visualize_experiment --run-id temp_ablation_v2
   111	
   112	# Verify hypothesis
   113	python -m experiments.verify_hypothesis docs/HYPOTHESIS.md results/temp_ablation_v2
   114	```
   115	
   116	**Fast smoke test** (5 seeds):
   117	```bash
   118	python -m experiments.runner temp_ablation_v2 --seeds 5 --out results/_smoke
   119	```
   120	
   121	**Baseline v1 experiment** (extreme variance reduction but suppresses consolidation):
   122	```bash
   123	python -m experiments.runner temp_ablation_v1
   124	```
   125	
   126	See [`docs/HYPOTHESIS.md`](docs/HYPOTHESIS.md) for experimental design and acceptance criteria.
   127	
   128	---
   129	
   130	## Sleep–Emergence Stack
   131	
   132	BN-Syn now includes a **Sleep–Emergence Stack** that integrates sleep-wake cycles, memory consolidation, attractor crystallization, and phase transition tracking. This provides a cohesive framework for studying emergent dynamics with deterministic guarantees.
   133	
   134	### Quick Demo
   135	
   136	Run the end-to-end sleep-stack demo:
   137	
   138	```bash
   139	bnsyn sleep-stack --seed 123 --steps-wake 800 --steps-sleep 600 --out results/demo1
   140	```
   141	
   142	**Outputs:**
   143	- `results/demo1/manifest.json`: Reproducibility metadata (seed, params, git SHA)
   144	- `results/demo1/metrics.json`: Metrics (phase transitions, attractors, consolidation stats)
   145	- `figures/demo1/summary.png`: Summary figure (if matplotlib installed)
   146	
   147	**Expected runtime:** ~5-10 seconds
   148	
   149	**Scaled flagship run (N=2000, extended wake/sleep):**
   150	
   151	```bash
   152	python -m bnsyn.tools.run_scaled_sleep_stack \
   153	  --out artifacts/local_runs/scaled_sleep_stack_n2000 \
   154	  --seed 123 --n 2000 --steps-wake 2400 --steps-sleep 1800
   155	```
   156	
   157	**Optional scale benchmark:**
   158	
   159	```bash
   160	python -m bnsyn.tools.benchmark_sleep_stack_scale
   161	```
   162	
   163	For CI/quick smoke runs, use fast flags to reduce workload:
   164	
   165	```bash
   166	python -m bnsyn.tools.run_scaled_sleep_stack \
   167	  --out /tmp/bnsyn_scaled_smoke \
   168	  --seed 42 --n 80 --steps-wake 30 --steps-sleep 30 \
   169	  --baseline-steps-wake 20 --baseline-steps-sleep 10 \
   170	  --determinism-runs 1 --skip-backend-equivalence --skip-baseline \
   171	  --no-raster --no-plots
   172	```
   173	
   174	The generated `<out>/metrics.json` contains fields:
   175	- `seed`, `N_scaled`, `steps_wake_scaled`, `steps_sleep_scaled`
   176	- `determinism_hashes` (per-run manifest/metrics hashes)
   177	- `determinism_runs` (int) and `determinism_identical` (bool or null when runs < 2)
   178	- `backend_equivalence` (`atol`, `equivalent`, `max_abs_sigma_diff`, `skipped`; values may be null when skipped)
   179	- `baseline_skipped` (bool) and `baseline` (`wake_std_sigma`, `transitions`, `attractors` or null when baseline skipped)
   180	- `scaled` (`wake_std_sigma`, `transitions`, `attractors`, `crystallization_progress`)
   181	- `benchmark` (`elapsed_s`, `memory_current_bytes`, `memory_peak_bytes`)
   182	
   183	Benchmark output is written to `artifacts/local_runs/benchmarks_scale/metrics.json` and is machine-dependent.
   184	
   185	### Minimal Usage Example
   186	
   187	```python
   188	from bnsyn.rng import seed_all
   189	from bnsyn.sim.network import Network, NetworkParams
   190	from bnsyn.config import AdExParams, SynapseParams, CriticalityParams, TemperatureParams
   191	from bnsyn.temperature.schedule import TemperatureSchedule
   192	from bnsyn.sleep import SleepCycle, default_human_sleep_cycle
   193	from bnsyn.memory import MemoryConsolidator
   194	from bnsyn.criticality import PhaseTransitionDetector
   195	from bnsyn.emergence import AttractorCrystallizer
   196	
   197	# Initialize with deterministic seed
   198	pack = seed_all(42)
   199	net = Network(NetworkParams(N=64), AdExParams(), SynapseParams(), 
   200	              CriticalityParams(), dt_ms=0.5, rng=pack.np_rng)
   201	
   202	# Setup sleep-emergence stack
   203	temp_schedule = TemperatureSchedule(TemperatureParams())
   204	sleep_cycle = SleepCycle(net, temp_schedule, max_memories=100, rng=pack.np_rng)
   205	consolidator = MemoryConsolidator(capacity=100)
   206	phase_detector = PhaseTransitionDetector()
   207	crystallizer = AttractorCrystallizer(state_dim=64, snapshot_dim=50)
   208	
   209	# Wake phase: run network and record memories
   210	for _ in range(500):
   211	    m = net.step()
   212	    if _ % 20 == 0:
   213	        consolidator.tag(net.state.V_mV, importance=0.8)
   214	        sleep_cycle.record_memory(importance=0.8)
   215	    phase_detector.observe(m["sigma"], _)
   216	    crystallizer.observe(net.state.V_mV, temp_schedule.T or 1.0)
   217	
   218	# Sleep phase: consolidation + replay
   219	sleep_cycle.sleep(default_human_sleep_cycle())
   220	consolidator.consolidate(protein_level=0.9, temperature=0.8)
