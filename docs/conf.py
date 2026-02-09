# Sphinx configuration for curate-ipsum
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone

# -- Project information -----------------------------------------------------

project = "curate-ipsum"
author = "Erik Goughnour"
copyright = f"{datetime.now(timezone.utc).year}, {author}"  # noqa: A001

try:
    release = importlib.metadata.version("curate-ipsum")
except importlib.metadata.PackageNotFoundError:
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Accept both .md and .rst
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
root_doc = "index"

# -- MyST parser options -----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",     # ::: directive syntax
    "deflist",         # definition lists
    "fieldlist",       # field lists
    "tasklist",        # - [x] checkboxes
    "attrs_inline",    # inline attributes
]
myst_heading_anchors = 3  # auto-generate anchors for h1-h3

# -- Autodoc options ---------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_mock_imports = [
    "mcp",
    "brs",
    "z3",
    "chromadb",
    "sentence_transformers",
    "kuzu",
    "scipy",
    "networkx",
    "angr",
    "httpx",
]

# Napoleon for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary
autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "curate-ipsum"
html_logo = "../icons/icon.png"
html_favicon = "../icons/icon.png"

html_theme_options = {
    "source_repository": "https://github.com/egoughnour/curate-ipsum",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#6738b7",
        "color-brand-content": "#533483",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9b72cf",
        "color-brand-content": "#b388ff",
    },
}

html_static_path = ["_static"]

# -- TODO extension ----------------------------------------------------------

todo_include_todos = True
