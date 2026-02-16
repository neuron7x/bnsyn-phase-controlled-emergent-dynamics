# Build Documentation Locally

This guide documents deterministic local documentation builds for BN-Syn.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Build HTML documentation

```bash
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

## Run link integrity checks

```bash
python -m sphinx -b linkcheck docs docs/_build/linkcheck
```

If external links are unavailable due to network/transient failures, re-run to confirm.

## Build API docs subtree directly (optional)

`docs/api` is included in the main docs build via the root toctree. If you need to build the API tree alone:

```bash
python -m sphinx -b html docs/api docs/api/_build/html
```

## Generated artifacts

Generated script help text used by the scripts catalog is stored under:

- `docs/_generated/script_help/*.txt`

Regenerate by re-running each script with `--help` from repository root, for example:

```bash
python -m scripts.check_api_contract --help
```

The scripts registry page is `docs/SCRIPTS/index.md` and must cover every `scripts/*.py` file.
