"""Sphinx configuration for BN-Syn API documentation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

project = "BN-Syn"
author = "BN-Syn Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "BaseModel,ConfigDict,Field,PositiveFloat",
}

autodoc_typehints = "none"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping: dict[str, tuple[str, str | None]] = {}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
