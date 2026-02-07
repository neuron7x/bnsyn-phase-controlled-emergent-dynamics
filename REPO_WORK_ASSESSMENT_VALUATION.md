# Repository Work Assessment & Valuation Report

**ASSESSMENT BASIS:**
This valuation is based on existing committed artifacts only.
Authorship method (human vs AI) was NOT used as a discounting factor.
Tools are treated as neutral productivity multipliers.

## 1. Repository Overview
- URL: UNKNOWN (no canonical remote URL supplied in evidence set).
- Primary language(s): Python, JSON, Markdown, YAML/YML/TOML, RST.
- First commit: 0317fe1b36052d94c74c8140abec500a5ce3766d (2026-01-23 11:37:49 +0200, Yaroslav Vasylenko)
- Latest commit: 5391ffa2f88728123b30ccc501312662d463c1ee (2026-02-07 18:23:55 +0000, Codex)
- Active days: 16
- Contributors: 4 total (1 human, 3 bot/AI)
- Total commits: 587

## 2. Artifact Inventory

| Category        | Files | LOC (code) | LOC (comments) | LOC (blank) |
|-----------------|------:|-----------:|---------------:|------------:|
| CORE_LOGIC      | 68    | 7,159      | 985            | 1,852       |
| TESTS           | 148   | 11,625     | 555            | 3,140       |
| CI_CD           | 29    | 2,607      | 100            | 301         |
| INFRASTRUCTURE  | 3     | 148        | 0              | 47          |
| DOCUMENTATION   | 109   | 7,735      | 3,109          | 3,634       |
| SCRIPTS_TOOLING | 123   | 12,464     | 477            | 2,359       |
| DATA_SCHEMAS    | 4     | 315        | 6              | 84          |
| CONFIGURATION   | 26    | 2,058      | 1,425          | 120         |
| STATIC_ASSETS   | 8     | 25,414     | 316            | 2,074       |
| GENERATED (excluded) | 117 | 115,145 | 113            | 309         |
|-----------------|------:|-----------:|---------------:|------------:|
| **TOTAL COUNTABLE** | **518** | **69,525** | **6,973** | **13,611** |

**Reconciliation check (required):**
- SUM category files (countable) = 68+148+29+3+109+123+4+26+8 = **518** ✅
- SUM category code LOC (countable) = 7,159+11,625+2,607+148+7,735+12,464+315+2,058+25,414 = **69,525** ✅
- Inventory file total = **635**; countable 518 + generated 117 = **635** ✅

### Core logic sublevel breakdown
- novel: 28 files
- known_model: 19 files
- glue: 9 files
- boilerplate: 16 files

### Documentation depth breakdown (Markdown)
- stubs (<20 lines): 16 files
- light (20–50 lines): 30 files
- substantive (>50 lines): 59 files
- total Markdown files: 105

## 3. Quality Assessment

| Dimension      | Level     | Key Evidence |
|----------------|-----------|--------------|
| Code Complexity| STANDARD→ADVANCED (A-average) | Radon average complexity A(3.4986), 1,464 analyzed blocks; grade distribution A:1295 B:119 C:44 D:4 E:2. |
| Test Maturity  | SOLID     | 142 unique test files, 639 test functions, 216 negative asserts, 100 property-test markers. |
| Documentation  | THOROUGH  | 59 substantive markdown files (≥50 LOC), 16 stubs (<20 LOC), 30 light docs (20–50 LOC). |
| Infrastructure | PRODUCTION-LEAN | 29 workflow files plus Dockerfile, Makefile, docker-compose captured by CI/CD inventory command. |

## 4. Effort Estimation

### Method: [Function Point / Line-Based / Commit-Based] (cross-validated)

| Work Category    | Volume | Complexity | Hours LOW | Hours MID | Hours HIGH |
|------------------|--------|------------|-----------|-----------|------------|
| Core Logic       | 68 files, 7,159 LOC | STANDARD→ADVANCED | 286 | 398 | 597 |
| Tests            | 148 files, 11,625 LOC | SOLID | 258 | 349 | 465 |
| CI_CD            | 29 files, 2,607 LOC | STANDARD | 52 | 87 | 130 |
| Infrastructure   | 3 files, 148 LOC | BASIC | 2 | 3 | 4 |
| Documentation    | 109 files, 7,735 LOC | MIXED | 110 | 155 | 221 |
| Scripts/Tooling  | 123 files, 12,464 LOC | STANDARD | 312 | 416 | 624 |
| Data/Schemas     | 4 files, 315 LOC | STANDARD | 11 | 16 | 21 |
| Configuration    | 26 files, 2,058 LOC | BASIC | 21 | 29 | 51 |
| Static Assets    | 8 files, 25,414 LOC-equivalent text | LOW | 64 | 95 | 169 |
| **TOTAL (line-based replacement)** | — | — | **1,116** | **1,548** | **2,282** |

### Cross-validation
- **Method 1 (Calendar-bound actual human investment):**
  - Active days = 16, human contributors = 1.
  - Hard calendar ceiling per protocol = 16 × 14 × 1 = **224 hours**.
  - Realized bound used: **96–224 hours** (MID 160).
- **Method 2 (Line-based replacement):** **1,116–2,282 hours** (MID 1,548).
- **Method 3 (Function-point replacement):**
  - Counted items: core module interfaces 45, CLI parser commands 4, schema modules 4, integration surfaces 6.
  - FP = (45×2) + (4×1) + (4×1) + (6×2) = **110 FP**.
  - 4–10 h/FP band => **440–1,100 hours** (MID 770).

**Convergence binding check:**
- MID values diverge by >3x (160 vs 770 vs 1,548).
- Two closest replacement methods: FP MID 770 and line MID 1,548.
- Geometric mean = sqrt(770×1548) = **1,092 hours**.
- **Adopted MID replacement estimate = 1,092 hours** (not largest).

### Dual framing (required)
- **Framing A — Actual human investment (calendar-bounded):** 96 / 160 / 224 hours (LOW/MID/HIGH).
- **Framing B — Replacement cost effort (manual rebuild benchmark):** 770 / 1,092 / 1,548 hours (LOW/MID/HIGH).

Counted: committed artifacts currently present.
Excluded: GENERATED artifacts from effort/pricing.
UNKNOWN: off-repo development effort and private pre-commit work.

## 5. Monetary Valuation

### Rate justification
- Measured complexity evidence: radon A-average with tail up to E; justified tier is Mid-to-Senior/Senior.
- Global rate band used: $40–$150/hr.
- Selected rates:
  - Actual-human valuation: $50 / $80 / $120.
  - Replacement-cost valuation: $60 / $90 / $140.

### Valuation

| Framing | Hours | Rate ($/hr) | Subtotal |
|---|---:|---:|---:|
| **Actual LOW** | 96 | 50 | 4,800 |
| **Actual MID** | 160 | 80 | 12,800 |
| **Actual HIGH** | 224 | 120 | 26,880 |
| **Replacement LOW** | 770 | 60 | 46,200 |
| **Replacement MID** | 1,092 | 90 | 98,280 |
| **Replacement HIGH** | 1,548 | 140 | 216,720 |

### Adjustments applied
- No authorship-based discount applied to scope/value.
- Calendar limit applied only to actual-human framing.
- Replacement framing used for rebuild economics.

### Sanity gates
- $/LOC (replacement high): 216,720 / 69,525 = **$3.12/LOC** (<$40) ✅
- $/active day (actual high): 26,880 / 16 = **$1,680/day** (<$5,000) ✅
- $/human hour max used: **$140/hr** (<$2,000/hr) ✅
- Buyer test: replacement MID $98,280 for 518 countable files and full CI/test/doc footprint is market-plausible ✅

### Final Valuation Range

| | Amount (USD) |
|---|---:|
| **Actual human investment (LOW/MID/HIGH)** | **$4,800 / $12,800 / $26,880** |
| **Replacement cost (LOW/MID/HIGH)** | **$46,200 / $98,280 / $216,720** |

## 6. Caveats & Limitations
- [x] `cloc` unavailable in environment; labeled fallback used.
- [x] Comment/code/blank counts are lexical, not parser-based.
- [x] NON-code business costs (PM/legal/ops/support) excluded.
- [x] Inventory includes evidence artifacts; these are explicitly classified as GENERATED and excluded from effort.

## 7. Evidence Appendix
- `audit_evidence/00_index.txt` — evidence collection completion index.
- `audit_evidence/01_metadata.txt` — commit/day metadata.
- `audit_evidence/02_contributors.txt` — human vs bot split with calibrated ratio.
- `audit_evidence/03_file_inventory.txt` — file manifest used for classification.
- `audit_evidence/04_loc.txt` — cloc fallback line counts.
- `audit_evidence/05_tests.txt` — deduplicated test inventory and markers.
- `audit_evidence/06_cicd.txt` — CI/CD and deployment config discovery.
- `audit_evidence/07_complexity.txt` — radon average and grade distribution.
- `audit_evidence/08_churn.txt` — git insertions/deletions and churn leaders.
- `audit_evidence/09_velocity.txt` — commits/day/author velocity sample.
- `audit_evidence/10_doc_sizes.txt` — documentation depth buckets.
- `audit_evidence/11_function_points.txt` — function-point supporting counts.
- `audit_report_data.json` — deterministic category totals + reconciliation inputs.
