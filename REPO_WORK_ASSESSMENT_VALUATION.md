# Repository Work Assessment & Valuation Report

**ASSESSMENT BASIS:**
This valuation is based on existing committed artifacts only.
Authorship method (human vs AI) was NOT used as a discounting factor.
Tools are treated as neutral productivity multipliers.

## 1. Repository Overview
- URL: UNKNOWN (no canonical remote URL supplied in evidence set).
- Primary language(s): Python, JSON, Markdown, YAML/YML/TOML, RST.
- First commit: 0317fe1b36052d94c74c8140abec500a5ce3766d (2026-01-23 11:37:49 +0200, Yaroslav Vasylenko)
- Latest commit: a9c779c4610a9d7e8a3990235b964ca489c8ca4f (2026-02-07 17:50:27 +0000, Codex)
- Active days: 16
- Contributors: 4 total (1 human, 3 bot/AI)
- Total commits: 587

## 2. Artifact Inventory

| Category        | Files | LOC (code) | LOC (comments) | LOC (blank) |
|-----------------|------:|-----------:|---------------:|------------:|
| CORE_LOGIC      | 67    | 7,015      | 980            | 1,824       |
| TESTS           | 147   | 11,535     | 555            | 3,100       |
| CI_CD           | 0     | 0          | 0              | 0           |
| INFRASTRUCTURE  | 3     | 148        | 0              | 47          |
| DOCUMENTATION   | 105   | 7,093      | 2,247          | 2,873       |
| SCRIPTS_TOOLING | 121   | 11,970     | 476            | 2,255       |
| DATA_SCHEMAS    | 4     | 315        | 6              | 84          |
| CONFIGURATION   | 20    | 1,952      | 1,394          | 106         |
| STATIC_ASSETS   | 8     | 25,414     | 316            | 2,074       |
| GENERATED (excluded) | 96 | 109,993 | 89             | 305         |
|-----------------|------:|-----------:|---------------:|------------:|
| **TOTAL COUNTABLE** | **475** | **65,442** | **5,974** | **12,363** |

**Reconciliation check (required):**
- SUM category files (countable) = 67+147+0+3+105+121+4+20+8 = **475** ✅
- SUM category code LOC (countable) = 7,015+11,535+0+148+7,093+11,970+315+1,952+25,414 = **65,442** ✅
- Inventory file total = **571**; countable 475 + generated 96 = **571** ✅

### Core logic sublevel breakdown
- novel: 27 files
- known_model: 19 files
- glue: 9 files
- boilerplate: 16 files

### Documentation depth breakdown (Markdown)
- stubs (<20 lines): 16 files
- light (20–50 lines): 29 files
- substantive (>50 lines): 56 files
- total Markdown files: 101

## 3. Quality Assessment

| Dimension      | Level     | Key Evidence |
|----------------|-----------|--------------|
| Code Complexity| STANDARD→ADVANCED (A-average) | Radon average complexity A(3.4986), 1,464 analyzed blocks; grade distribution A:1295 B:119 C:44 D:4 E:2. |
| Test Maturity  | SOLID     | 141 unique test files, 617 test functions, 216 negative asserts, 84 property-test markers. |
| Documentation  | ADEQUATE→THOROUGH | 56 substantive markdown files (≥50 LOC), 16 stubs (<20 LOC). |
| Infrastructure | BASIC (under strict exclusion regex) | CI/CD sweep captured Dockerfile, Makefile, docker-compose only; workflow files dropped by provided exclusion regex. |

## 4. Effort Estimation

### Method: [Function Point / Line-Based / Commit-Based] (cross-validated)

| Work Category    | Volume | Complexity | Hours LOW | Hours MID | Hours HIGH |
|------------------|--------|------------|-----------|-----------|------------|
| Core Logic       | 67 files, 7,015 LOC | STANDARD→ADVANCED | 280 | 390 | 585 |
| Tests            | 147 files, 11,535 LOC | SOLID | 255 | 346 | 461 |
| CI_CD            | 0 files, 0 LOC | NONE | 0 | 0 | 0 |
| Infrastructure   | 3 files, 148 LOC | BASIC | 2 | 3 | 4 |
| Documentation    | 105 files, 7,093 LOC | MIXED | 101 | 142 | 203 |
| Scripts/Tooling  | 121 files, 11,970 LOC | STANDARD | 299 | 399 | 599 |
| Data/Schemas     | 4 files, 315 LOC | STANDARD | 11 | 16 | 21 |
| Configuration    | 20 files, 1,952 LOC | BASIC | 20 | 28 | 49 |
| Static Assets    | 8 files, 25,414 LOC-equivalent text | LOW | 64 | 95 | 169 |
| **TOTAL (line-based replacement)** | — | — | **1,032** | **1,419** | **2,091** |

### Cross-validation
- **Method 1 (Calendar-bound actual human investment):**
  - Active days = 16, human contributors = 1.
  - Hard calendar ceiling per protocol = 16 × 14 × 1 = **224 hours**.
  - Observed daily activity suggests practical realized range: **96–224 hours** (MID 160).
- **Method 2 (Line-based replacement):** **1,032–2,091 hours** (MID 1,419).
- **Method 3 (Function-point replacement):**
  - Counted items: core module interfaces 45, CLI parser commands 4, schema modules 4, integration surfaces 6 (numpy/scipy/yaml/matplotlib/streamlit/jax).
  - FP = (45×2) + (4×1) + (4×1) + (6×2) = **110 FP**.
  - 4–10 h/FP band (A-average mixed system) => **440–1,100 hours** (MID 770).

**Convergence binding check:**
- MID values diverge by >3x (160 vs 770 vs 1,419).
- Two closest replacement methods are FP MID 770 and line MID 1,419.
- Geometric mean = sqrt(770×1419) = **1,045 hours**.
- **Adopted MID replacement estimate = 1,045 hours** (not the largest method).

### Dual framing (required)
- **Framing A — Actual human investment (calendar-bounded):** 96 / 160 / 224 hours (LOW/MID/HIGH).
- **Framing B — Replacement cost effort (manual rebuild benchmark):** 770 / 1,045 / 1,419 hours (LOW/MID/HIGH).

Counted: committed artifacts currently present.
Excluded: GENERATED artifacts from effort/pricing.
UNKNOWN: off-repo development effort and private pre-commit work.

## 5. Monetary Valuation

### Rate justification
- Measured complexity evidence: radon average grade A with tail up to E; suitable tier is **Mid-to-Senior/Senior**, not automatic Expert.
- Applicable global rate band from protocol: **$40–$150/hr**.
- Selected rates:
  - Actual-human valuation: $50 / $80 / $120
  - Replacement-cost valuation: $60 / $90 / $140

### Valuation

| Framing | Hours | Rate ($/hr) | Subtotal |
|---|---:|---:|---:|
| **Actual LOW** | 96 | 50 | 4,800 |
| **Actual MID** | 160 | 80 | 12,800 |
| **Actual HIGH** | 224 | 120 | 26,880 |
| **Replacement LOW** | 770 | 60 | 46,200 |
| **Replacement MID** | 1,045 | 90 | 94,050 |
| **Replacement HIGH** | 1,419 | 140 | 198,660 |

### Adjustments applied
- No authorship-based discount applied to scope/value.
- Calendar limit applied only to actual-human framing.
- Replacement framing used for non-calendar rebuild economics.

### Sanity gates
- $/LOC (replacement high): 198,660 / 65,442 = **$3.04/LOC** (<$40) ✅
- $/active day (actual high): 26,880 / 16 = **$1,680/day** (<$5,000) ✅
- $/human hour max used: **$140/hr** (<$2,000/hr) ✅
- Buyer test: replacement MID $94,050 for 475 countable files with tests/docs/tooling is plausible in Senior global market band ✅

### Final Valuation Range

| | Amount (USD) |
|---|---:|
| **Actual human investment (LOW/MID/HIGH)** | **$4,800 / $12,800 / $26,880** |
| **Replacement cost (LOW/MID/HIGH)** | **$46,200 / $94,050 / $198,660** |

## 6. Caveats & Limitations
- [x] CI/CD inventory command in provided script uses exclusion regex `.git|...`, which also excludes `.github/**`; CI_CD appears as 0 under strict replay.
- [x] `cloc` unavailable; labeled fallback used.
- [x] `bc` unavailable in this environment; bot ratio percentage emitted as `N/A%` in contributor evidence.
- [x] Comment/code/blank counts are lexical, not parser-based.
- [x] Non-repository efforts (PM, stakeholder alignment, operations, legal/compliance) are excluded.

## 7. Evidence Appendix
- `audit_evidence/00_index.txt` — evidence collection completion index.
- `audit_evidence/01_metadata.txt` — commit/day metadata.
- `audit_evidence/02_contributors.txt` — human vs bot split.
- `audit_evidence/03_file_inventory.txt` — file manifest used for classification.
- `audit_evidence/04_loc.txt` — cloc fallback line counts.
- `audit_evidence/05_tests.txt` — deduplicated test inventory and quality markers.
- `audit_evidence/06_cicd.txt` — CI/CD and deployment config discovery.
- `audit_evidence/07_complexity.txt` — radon average and grade distribution.
- `audit_evidence/08_churn.txt` — git insertions/deletions and churn leaders.
- `audit_evidence/09_velocity.txt` — commits/day/author velocity sample.
- `audit_evidence/10_doc_sizes.txt` — documentation depth buckets.
- `audit_evidence/11_function_points.txt` — function-point supporting counts.
- `audit_report_data.json` — deterministic category totals + reconciliation inputs.
