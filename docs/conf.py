import os
import sys

project   = "parkol"
copyright = "2026, Sarat Moka"
author    = "Sarat Moka"
release   = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = False

# Preserve source order
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members":        True,
    "undoc-members":  False,
    "show-inheritance": False,
}

autosummary_generate = True

html_theme = "sphinx_rtd_theme"
html_static_path = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
