import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Make your repo importable

import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# -- Project information -----------------------------------------------------
project = 'MyRepo'
author = 'Your Name'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',            # Pull docstrings
    'sphinx.ext.napoleon',           # NumPy / Google style docstrings
    'sphinx_autodoc_typehints',      # Include type hints
    'sphinx.ext.viewcode',           # Add links to source code
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # handles Google/Numpy
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'exclude-members': 'model_config,model_fields,__pydantic_fields__'
}
