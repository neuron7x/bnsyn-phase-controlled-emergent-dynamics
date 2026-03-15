     1	[project]
     2	name = "bnsyn"
     3	version = "0.2.0"
     4	description = "BN-Syn Thermostated Bio-AI System: AdEx + conductances + 3-factor plasticity + criticality control + temperature-gated consolidation"
     5	requires-python = ">=3.11"
     6	readme = "README.md"
     7	license = "MIT"
     8	license-files = ["LICENSE"]
     9	authors = [{name="BN-Syn Contributors"}]
    10	dependencies = [
    11	  "numpy==2.4.1",
    12	  "pydantic==2.12.5",
    13	  "scipy==1.17.0",
    14	  "jsonschema==4.26.0",
    15	  "joblib==1.4.2",
    16	]
    17	
    18	[project.optional-dependencies]
    19	dev = [
    20	  "hypothesis==6.151.5",
    21	  "pytest==9.0.2",
    22	  "pytest-cov==7.0.0",
    23	  "pyyaml==6.0.3",
    24	  "ruff==0.15.0",
    25	  "mypy==1.19.1",
    26	  "pylint==3.3.5",
    27	  "pydocstyle==6.3.0",
    28	  "bandit==1.9.3",
    29	  "validate-pyproject==0.25",
    30	  "pre-commit==4.5.1",
    31	  "pip-audit==2.10.0",
    32	  "psutil==7.2.2",
    33	  "sphinx==9.0.4",
    34	  "sphinx-autodoc-typehints==3.6.1",
    35	  "myst-parser==5.0.0",
    36	  "furo==2025.12.19",
    37	  "sphinx-copybutton==0.5.2",
    38	]
    39	
    40	test = [
    41	  "hypothesis==6.151.5",
    42	  "pytest==9.0.2",
    43	  "pytest-cov==7.0.0",
    44	  "pyyaml==6.0.3",
    45	  "psutil==7.2.2",
    46	]
    47	
    48	viz = [
    49	  "matplotlib==3.10.8",
    50	  "pillow==12.1.0",
    51	  "streamlit==1.42.1",
    52	  "plotly==6.5.2",
    53	]
    54	
    55	jax = [
    56	  "jax==0.6.0",
    57	  "jaxlib==0.9.0",
    58	]
    59	
    60	torch = [
    61	  "torch==2.10.0",
    62	]
    63	
    64	accelerators = [
    65	  "bnsyn[jax]",
    66	  "bnsyn[torch]",
    67	]
    68	
    69	[project.scripts]
    70	bnsyn = "bnsyn.cli:main"
    71	
    72	[build-system]
    73	requires = ["setuptools==79.0.1"]
    74	build-backend = "setuptools.build_meta"
    75	
    76	[tool.setuptools]
    77	package-dir = {"" = "src"}
    78	
    79	[tool.pytest.ini_options]
    80	addopts = "-q --strict-markers"
    81	testpaths = ["tests"]
    82	markers = [
    83	  "smoke: fast critical-path tests",
    84	  "validation: slow statistical/large-N validation tests (excluded from CI by default)",
    85	  "benchmark: benchmark regression tests (excluded from mutation runs)",
    86	  "performance: performance regression tests with timing assertions",
    87	  "integration: integration tests requiring multiple components",
    88	  "property: property-based tests using Hypothesis",
    89	  "chaos: chaos engineering tests with fault injection",
    90	]
    91	
    92	[tool.hypothesis]
    93	derandomize = true
    94	
    95	[tool.hypothesis.profiles.ci]
    96	max_examples = 200
    97	deadline = 10000
    98	print_blob = true
    99	
   100	[tool.hypothesis.profiles.quick]
   101	max_examples = 100
   102	deadline = 5000
   103	print_blob = true
   104	
   105	[tool.hypothesis.profiles.thorough]
   106	max_examples = 1000
   107	deadline = 20000
   108	print_blob = true
   109	
   110	[tool.ruff]
   111	line-length = 100
   112	target-version = "py311"
   113	
   114	[tool.mypy]
   115	python_version = "3.11"
   116	strict = true
   117	warn_return_any = true
   118	disallow_untyped_defs = true
   119	disallow_untyped_calls = true
   120	disallow_incomplete_defs = true
   121	check_untyped_defs = true
   122	no_implicit_optional = true
   123	warn_redundant_casts = true
   124	warn_unused_ignores = true
   125	warn_no_return = true
   126	warn_unused_configs = true
   127	plugins = ["pydantic.mypy"]
   128	
   129	# Optional visualization dependencies (no type stubs available)
   130	[[tool.mypy.overrides]]
   131	module = ["plotly.*", "streamlit.*", "matplotlib.*"]
   132	ignore_missing_imports = true
   133	
   134	[tool.pylint.main]
   135	recursive = true
   136	ignore-patterns = ["^test_.*\\.py$", "^conftest\\.py$"]
   137	fail-under = 7.5
   138	
   139	[tool.pylint.messages_control]
   140	disable = [
   141	  "import-error",
   142	  "invalid-name",
   143	  "broad-exception-caught",
   144	  "useless-import-alias",
   145	  "missing-function-docstring",
   146	]
   147	
   148	[tool.pylint.basic]
   149	good-names = ["i", "j", "k", "V", "N", "w", "dt", "dx", "dy", "R", "E"]
   150	
   151	[tool.pylint.design]
   152	max-args = 15
   153	
   154	[[tool.mypy.overrides]]
   155	module = "scipy.*"
   156	ignore_missing_imports = true
   157	
   158	[[tool.mypy.overrides]]
   159	module = "torch"
   160	ignore_missing_imports = true
   161	
   162	[[tool.mypy.overrides]]
   163	module = "jax.*"
   164	ignore_missing_imports = true
   165	
   166	[tool.setuptools.packages.find]
   167	where = ["src"]
