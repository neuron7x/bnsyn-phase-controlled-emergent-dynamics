# API Documentation Build Instructions

## Quick start

```bash
python -m pip install -e ".[dev,docs]"
make docs-api
```

Expected output (abridged):

- HTML documentation is generated under `docs/sphinx/_build/html`.
- `sphinx-build -W` completes without warnings.

## Clean build

```bash
make docs-api-clean
```

## Docstring quality gate

```bash
make docstrings-check
```

Expected output includes a summary of scanned modules and a zero exit code.

## Troubleshooting

- **Import errors during autodoc**: ensure you installed editable dependencies
  with `python -m pip install -e ".[dev,docs]"` and that `src/` is on the
  Python path (handled in `docs/sphinx/conf.py`).
- **Sphinx warnings treated as errors**: resolve any missing docstrings or
  invalid references before rebuilding with `make docs-api`.
- **Autodoc missing modules**: verify that the module is listed in
  `docs/sphinx/api.rst` and that the package exports it in `src/bnsyn`.
