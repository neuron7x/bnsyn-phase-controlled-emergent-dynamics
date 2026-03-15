     1	
     2	==================================== ERRORS ====================================
     3	_____________ ERROR collecting tests/benchmarks/test_regression.py _____________
     4	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/benchmarks/test_regression.py'.
     5	Hint: make sure your test modules/packages have valid Python names.
     6	Traceback:
     7	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
     8	    return _bootstrap._gcd_import(name[level:], package, level)
     9	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    10	tests/benchmarks/test_regression.py:8: in <module>
    11	    from benchmarks.metrics import metrics_to_dict, run_benchmark
    12	benchmarks/metrics.py:12: in <module>
    13	    import psutil
    14	E   ModuleNotFoundError: No module named 'psutil'
    15	______________________ ERROR collecting tests/properties _______________________
    16	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    17	    return _bootstrap._gcd_import(name[level:], package, level)
    18	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    19	<frozen importlib._bootstrap>:1387: in _gcd_import
    20	    ???
    21	<frozen importlib._bootstrap>:1360: in _find_and_load
    22	    ???
    23	<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    24	    ???
    25	<frozen importlib._bootstrap>:935: in _load_unlocked
    26	    ???
    27	/root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/_pytest/assertion/rewrite.py:197: in exec_module
    28	    exec(co, module.__dict__)
    29	tests/properties/conftest.py:5: in <module>
    30	    from hypothesis import Verbosity, settings
    31	E   ModuleNotFoundError: No module named 'hypothesis'
    32	______________ ERROR collecting tests/test_claims_enforcement.py _______________
    33	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_claims_enforcement.py'.
    34	Hint: make sure your test modules/packages have valid Python names.
    35	Traceback:
    36	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    37	    return _bootstrap._gcd_import(name[level:], package, level)
    38	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    39	tests/test_claims_enforcement.py:21: in <module>
    40	    import yaml
    41	E   ModuleNotFoundError: No module named 'yaml'
    42	____________ ERROR collecting tests/test_experiments_declarative.py ____________
    43	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_experiments_declarative.py'.
    44	Hint: make sure your test modules/packages have valid Python names.
    45	Traceback:
    46	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    47	    return _bootstrap._gcd_import(name[level:], package, level)
    48	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    49	tests/test_experiments_declarative.py:10: in <module>
    50	    from bnsyn.experiments import declarative
    51	src/bnsyn/experiments/__init__.py:5: in <module>
    52	    from bnsyn.experiments.declarative import load_config, run_experiment, run_from_yaml
    53	src/bnsyn/experiments/declarative.py:17: in <module>
    54	    import yaml  # type: ignore[import-untyped]
    55	    ^^^^^^^^^^^
    56	E   ModuleNotFoundError: No module named 'yaml'
    57	__________ ERROR collecting tests/test_integration_experiment_flow.py __________
    58	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_integration_experiment_flow.py'.
    59	Hint: make sure your test modules/packages have valid Python names.
    60	Traceback:
    61	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    62	    return _bootstrap._gcd_import(name[level:], package, level)
    63	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    64	tests/test_integration_experiment_flow.py:8: in <module>
    65	    from bnsyn.experiments.declarative import run_from_yaml
    66	src/bnsyn/experiments/__init__.py:5: in <module>
    67	    from bnsyn.experiments.declarative import load_config, run_experiment, run_from_yaml
    68	src/bnsyn/experiments/declarative.py:17: in <module>
    69	    import yaml  # type: ignore[import-untyped]
    70	    ^^^^^^^^^^^
    71	E   ModuleNotFoundError: No module named 'yaml'
    72	_______________ ERROR collecting tests/test_manifest_tooling.py ________________
    73	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_manifest_tooling.py'.
    74	Hint: make sure your test modules/packages have valid Python names.
    75	Traceback:
    76	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    77	    return _bootstrap._gcd_import(name[level:], package, level)
    78	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    79	tests/test_manifest_tooling.py:5: in <module>
    80	    from tools.manifest import generate
    81	tools/manifest/generate.py:8: in <module>
    82	    import yaml
    83	E   ModuleNotFoundError: No module named 'yaml'
    84	__________ ERROR collecting tests/test_manifest_tooling_regression.py __________
    85	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_manifest_tooling_regression.py'.
    86	Hint: make sure your test modules/packages have valid Python names.
    87	Traceback:
    88	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
    89	    return _bootstrap._gcd_import(name[level:], package, level)
    90	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    91	tests/test_manifest_tooling_regression.py:9: in <module>
    92	    from tools.manifest import generate, validate
    93	tools/manifest/generate.py:8: in <module>
    94	    import yaml
    95	E   ModuleNotFoundError: No module named 'yaml'
    96	___________ ERROR collecting tests/test_scan_governed_docs_script.py ___________
    97	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_scan_governed_docs_script.py'.
    98	Hint: make sure your test modules/packages have valid Python names.
    99	Traceback:
   100	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   101	    return _bootstrap._gcd_import(name[level:], package, level)
   102	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   103	tests/test_scan_governed_docs_script.py:10: in <module>
   104	    from scripts import scan_governed_docs  # noqa: E402
   105	    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   106	scripts/scan_governed_docs.py:22: in <module>
   107	    import yaml
   108	E   ModuleNotFoundError: No module named 'yaml'
   109	_______________ ERROR collecting tests/test_schema_contracts.py ________________
   110	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_schema_contracts.py'.
   111	Hint: make sure your test modules/packages have valid Python names.
   112	Traceback:
   113	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   114	    return _bootstrap._gcd_import(name[level:], package, level)
   115	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   116	tests/test_schema_contracts.py:6: in <module>
   117	    import yaml
   118	E   ModuleNotFoundError: No module named 'yaml'
   119	_________ ERROR collecting tests/test_sync_required_status_contexts.py _________
   120	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_sync_required_status_contexts.py'.
   121	Hint: make sure your test modules/packages have valid Python names.
   122	Traceback:
   123	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   124	    return _bootstrap._gcd_import(name[level:], package, level)
   125	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   126	tests/test_sync_required_status_contexts.py:5: in <module>
   127	    from scripts.sync_required_status_contexts import build_payload, sync_required_status_contexts
   128	scripts/sync_required_status_contexts.py:7: in <module>
   129	    import yaml
   130	E   ModuleNotFoundError: No module named 'yaml'
   131	_________ ERROR collecting tests/test_validate_bibliography_script.py __________
   132	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_validate_bibliography_script.py'.
   133	Hint: make sure your test modules/packages have valid Python names.
   134	Traceback:
   135	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   136	    return _bootstrap._gcd_import(name[level:], package, level)
   137	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   138	tests/test_validate_bibliography_script.py:12: in <module>
   139	    from scripts import validate_bibliography as vb  # noqa: E402
   140	    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   141	scripts/validate_bibliography.py:17: in <module>
   142	    import yaml
   143	E   ModuleNotFoundError: No module named 'yaml'
   144	________ ERROR collecting tests/test_validate_long_running_triggers.py _________
   145	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_validate_long_running_triggers.py'.
   146	Hint: make sure your test modules/packages have valid Python names.
   147	Traceback:
   148	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   149	    return _bootstrap._gcd_import(name[level:], package, level)
   150	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   151	tests/test_validate_long_running_triggers.py:5: in <module>
   152	    import yaml
   153	E   ModuleNotFoundError: No module named 'yaml'
   154	_______________ ERROR collecting tests/test_validate_pr_gates.py _______________
   155	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_validate_pr_gates.py'.
   156	Hint: make sure your test modules/packages have valid Python names.
   157	Traceback:
   158	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   159	    return _bootstrap._gcd_import(name[level:], package, level)
   160	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   161	tests/test_validate_pr_gates.py:5: in <module>
   162	    from scripts.validate_pr_gates import validate_pr_gates
   163	scripts/validate_pr_gates.py:8: in <module>
   164	    import yaml
   165	E   ModuleNotFoundError: No module named 'yaml'
   166	__________ ERROR collecting tests/test_validate_workflow_contracts.py __________
   167	ImportError while importing test module '/workspace/bnsyn-phase-controlled-emergent-dynamics/tests/test_validate_workflow_contracts.py'.
   168	Hint: make sure your test modules/packages have valid Python names.
   169	Traceback:
   170	/root/.pyenv/versions/3.12.12/lib/python3.12/importlib/__init__.py:90: in import_module
   171	    return _bootstrap._gcd_import(name[level:], package, level)
   172	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   173	tests/test_validate_workflow_contracts.py:5: in <module>
   174	    from scripts.validate_workflow_contracts import validate_contracts
   175	scripts/validate_workflow_contracts.py:9: in <module>
   176	    import yaml
   177	E   ModuleNotFoundError: No module named 'yaml'
   178	=========================== short test summary info ============================
   179	ERROR tests/benchmarks/test_regression.py
   180	ERROR tests/properties - ModuleNotFoundError: No module named 'hypothesis'
   181	ERROR tests/test_claims_enforcement.py
   182	ERROR tests/test_experiments_declarative.py
   183	ERROR tests/test_integration_experiment_flow.py
   184	ERROR tests/test_manifest_tooling.py
   185	ERROR tests/test_manifest_tooling_regression.py
   186	ERROR tests/test_scan_governed_docs_script.py
   187	ERROR tests/test_schema_contracts.py
   188	ERROR tests/test_sync_required_status_contexts.py
   189	ERROR tests/test_validate_bibliography_script.py
   190	ERROR tests/test_validate_long_running_triggers.py
   191	ERROR tests/test_validate_pr_gates.py
   192	ERROR tests/test_validate_workflow_contracts.py
   193	!!!!!!!!!!!!!!!!!!! Interrupted: 14 errors during collection !!!!!!!!!!!!!!!!!!!
