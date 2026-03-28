  34: ## Quality Assurance
  35: 
  36: Quality gates are command-verifiable and documented in one place:
  37: 
  38: - Canonical local test/coverage commands: [`docs/TESTING.md`](docs/TESTING.md)
  39: - CI gate definitions: [`docs/CI_GATES.md`](docs/CI_GATES.md)
  40: - Workflow contracts: [`.github/WORKFLOW_CONTRACTS.md`](.github/WORKFLOW_CONTRACTS.md)
  41: - Evidence mapping: [`docs/EVIDENCE_COVERAGE.md`](docs/EVIDENCE_COVERAGE.md)
  42: 
  43: No aggregate quality percentages are published in README; only reproducible command outputs and artifacts are considered normative.
  44: 
  45: ---
  46: 
  47: ## Project Status
  48: 
  49: Official status declaration: [`docs/STATUS.md`](docs/STATUS.md).
  50: 
  51: ## Validation & Testing Strategy
  52: 
  53: Canonical commands for install, test, coverage, and coverage gate live in [`docs/TESTING.md`](docs/TESTING.md).
  54: 
  55: BN-Syn implements a **3-tier test selection strategy** for optimal coverage without blocking development:
  56: 
  57: ### Tier 1: BLOCKING (PR Gates) ⚡
  58: **Every PR, ~8 min** — Fast smoke tests, SSOT validation, claims coverage enforcement (CLM-0011), security scans
  59: 
  60: ### Tier 2: NON-BLOCKING Validation 🔬
  61: **Daily 2 AM UTC** — 10 scientific validation tests + 8 property-based invariants (Hypothesis)
  62: 
  63: ### Tier 3: Performance Tracking 📊
  64: **Weekly Sunday 3 AM UTC** — Benchmark regression detection against golden baseline
  65: 
  66: **Learn More:**
  67: - [CI Gates](docs/CI_GATES.md) — Test selection strategy
  68: - [Test Protocol](docs/ACTIONS_TEST_PROTOCOL.md) — GitHub Actions testing guide
  69: - [Evidence Coverage](docs/EVIDENCE_COVERAGE.md) — Claims→Evidence traceability
  70: 
