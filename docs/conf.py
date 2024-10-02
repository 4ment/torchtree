# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date
from docutils import nodes
from docutils.parsers.rst import roles

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'torchtree'
copyright = f"{date.today().year}, Mathieu Fourment"
author = 'Mathieu Fourment'

# The full version, including alpha/beta/rc tags
with open(os.path.join("..", "torchtree", "_version.py")) as f:
    release = f.readlines()[-1].split()[-1].strip("\"'")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx.ext.intersphinx",
    'autoapi.extension',
    'sphinxcontrib.bibtex',
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/4ment/torchtree",
    "use_repository_button": True,
    "use_download_button": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


source_suffix = {".rst": "restructuredtext"}

# -- Extension configuration -------------------------------------------------

# AutoAPI configuration
autoapi_dirs = ["../torchtree"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_options = ["show-module-summary", "undoc-members", "show-inheritance"]
autodoc_typehints = "signature"

bibtex_bibfiles = ['bibliography/refs.bib']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

def colorcode_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Create a literal node with the class "keycode"
    node = nodes.literal(text, text, classes=["keycode"])
    return [node], []

# Register the new role
roles.register_local_role('keycode', colorcode_role)

def setup(app):
    app.add_css_file('custom.css')