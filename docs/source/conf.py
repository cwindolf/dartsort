# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from versioningit import get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dartsort'
copyright = '2026, Charlie Windolf'
author = 'Charlie Windolf'
release = get_version("../..")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
]
autosummary_generate = True
autodoc_typehints = 'description'

templates_path = ["_templates"]
exclude_patterns = []
myst_enable_extensions = ["substitution"]
myst_substitutions = {
    "release": release,
}
html_theme_options = {
    'sidebar_collapse': False
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
