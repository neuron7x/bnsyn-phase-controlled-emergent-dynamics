     1	# Testing & Coverage (Canonical)
     2	
     3	This document is the **single source of truth** for running tests and coverage in this repository.
     4	
     5	## Install test dependencies
     6	
     7	```bash
     8	python -m pip install -e ".[test]"
     9	```
    10	
    11	Expected output pattern:
    12	- `Successfully installed bnsyn-...`
    13	- No import errors for `pytest`, `pytest-cov`, `hypothesis`.
    14	
    15	## Run default test suite
    16	
    17	```bash
    18	make test
    19	```
    20	
    21	Equivalent explicit command:
    22	
    23	```bash
    24	python -m pytest -m "not validation" -q
    25	```
    26	
    27	Expected output pattern:
    28	- Dots for passing tests.
    29	- Final summary with `passed` and optional `skipped`.
    30	
    31	## Run smoke marker tests
    32	
    33	```bash
    34	python -m pytest -m smoke -q
    35	```
    36	
    37	Expected output pattern:
    38	- Only smoke-marked tests.
    39	
    40	
    41	## Property test contour (Hypothesis)
    42	
    43	Hypothesis profiles are defined only in:
    44	- `tests/properties/conftest.py`
    45	
    46	Run property tests (requires `hypothesis` installed):
    47	
    48	```bash
    49	HYPOTHESIS_PROFILE=ci python -m pytest tests/properties -m property -q
    50	```
    51	
    52	Alternate profiles:
    53	
    54	```bash
    55	HYPOTHESIS_PROFILE=quick python -m pytest tests/properties -m property -q
    56	HYPOTHESIS_PROFILE=thorough python -m pytest tests/properties -m property -q
    57	```
    58	
    59	Run non-property tests without Hypothesis dependency:
    60	
    61	```bash
    62	python -m pytest -m "not property" -q
    63	```
    64	
    65	Behavior when `hypothesis` is missing:
    66	- `python -m pytest -m "not property" -q` succeeds.
    67	- `python -m pytest tests/properties -m property -q` fails with explicit `ModuleNotFoundError: No module named 'hypothesis'`.
    68	
    69	## Generate fast local coverage artifacts (canonical dev path)
    70	
    71	```bash
    72	make coverage-fast
    73	```
    74	
    75	Equivalent explicit command:
    76	
    77	```bash
    78	python -m pytest -m "not (validation or property)" --cov=bnsyn --cov-report=term-missing --cov-report=xml:coverage.xml -q
    79	```
    80	
    81	Artifacts:
    82	- Terminal report with missing lines by module (`term-missing`).
    83	- `coverage.xml` at repository root.
    84	
    85	## Generate coverage artifacts
    86	
    87	```bash
    88	make coverage
    89	```
    90	
    91	Equivalent explicit command:
    92	
    93	```bash
    94	python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml -q
    95	```
    96	
    97	Artifacts:
    98	- Terminal report with missing lines by module.
    99	- `coverage.xml` at repository root.
   100	
   101	## Generate / refresh coverage baseline
   102	
   103	```bash
   104	make coverage-baseline
   105	```
   106	
   107	Equivalent explicit command:
   108	
   109	```bash
   110	python -m scripts.generate_coverage_baseline --coverage-xml coverage.xml --output quality/coverage_gate.json --minimum-percent 99.0
   111	```
   112	
   113	This baseline uses the same metric enforced by the gate: `coverage.xml line-rate`.
   114	
   115	## Enforce coverage gate
   116	
   117	```bash
   118	make coverage-gate
   119	```
   120	
   121	Gate behavior:
   122	- Fails if current coverage drops below baseline in `quality/coverage_gate.json`.
   123	- Fails if current coverage drops below minimum floor in `quality/coverage_gate.json`.
   124	
   125	
   126	## API contract check (canonical)
   127	
   128	```bash
   129	make api-contract
   130	```
   131	
   132	Equivalent explicit command:
   133	
   134	```bash
   135	python -m scripts.check_api_contract --baseline quality/api_contract_baseline.json
   136	```
   137	
   138	Expected output pattern:
   139	- `API contract check passed`
   140	
   141	## CI parity checks (local)
   142	
   143	Use the same checks enforced in PR CI:
   144	
   145	```bash
   146	python -m pytest -q
   147	python -m pytest --cov=bnsyn --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml -q
   148	ruff check .
   149	```
   150	
   151	If a tool is unavailable locally, install via:
   152	
   153	```bash
   154	python -m pip install -e ".[test]"
   155	```
   156	
   157	Deferred gate note:
   158	- `mypy` is configured in CI quality workflow; run locally only after full `.[dev]` install.
   159	
   160	
   161	## Offline dependency workflow (Python 3.11)
   162	
   163	Notes:
   164	- `wheelhouse/` is platform-specific (implementation/ABI/platform tag). Build and validate for the same target.
   165	- `wheelhouse-build` requires internet access. `wheelhouse-validate` and `dev-env-offline` are offline.
   166	- `wheelhouse-build` uses `pip download --only-binary=:all: --no-deps`; lock file must stay fully pinned.
   167	- `wheelhouse-validate` writes `artifacts/wheelhouse_report.json` by default.
   168	
   169	Build the local wheelhouse from pinned lock dependencies:
   170	
   171	```bash
   172	make wheelhouse-build
   173	```
   174	
   175	Validate that every pinned dependency in `requirements-lock.txt` has a matching wheel in `wheelhouse/`:
   176	
   177	```bash
   178	make wheelhouse-validate
   179	make wheelhouse-report
   180	```
   181	
   182	Install the development environment fully offline from the local wheelhouse:
   183	
   184	```bash
   185	make dev-env-offline
   186	```
   187	
   188	Equivalent install commands:
   189	
   190	```bash
   191	python -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt
   192	python -m pip install --no-index --find-links wheelhouse --no-deps -e .
   193	```
   194	
   195	Failure modes:
   196	- Locked package has no wheel for the configured target tuple.
   197	- Marker applicability differs from the target environment.
   198	- Wheelhouse built for a different platform/ABI than the install target.
   199	
   200	Validation exit codes:
   201	- `0`: wheelhouse fully covers applicable locked requirements.
   202	- `1`: one or more applicable locked requirements are missing wheels.
   203	- `2`: lock contains unsupported or duplicate applicable requirement entries.
   204	
   205	Report contains additional diagnostics:
   206	- `duplicate_requirements`
   207	- `incompatible_wheels`
   208	- `malformed_wheels`
   209	
   210	## Updating lock and wheelhouse
   211	
   212	1. Refresh lock file after dependency changes:
   213	
   214	```bash
   215	pip-compile --extra=dev --generate-hashes --output-file=requirements-lock.txt pyproject.toml
   216	```
   217	
   218	2. Rebuild wheels for Python 3.11 and re-run validation:
   219	
   220	```bash
   221	make wheelhouse-build
   222	make wheelhouse-validate
   223	```
