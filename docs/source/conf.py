import pkg_resources

version = pkg_resources.get_distribution("bio-attention").version
release = version

import os
import sys

sys.path.insert(0, os.path.abspath("../bio-attention/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bio-attention"
copyright = "2023, Gaetan De Waele"
author = "Gaetan De Waele"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "nbsphinx",
]

autoclass_content = "class"

autodoc_default_options = {
    "member-order": "bysource",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
