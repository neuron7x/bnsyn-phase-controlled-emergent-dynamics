from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "BN-Syn"
author = "BN-Syn Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "BN-Syn API Documentation"
html_last_updated_fmt = None
today = "1970-01-01"

os.environ.setdefault("PYTHONUTF8", "1")
