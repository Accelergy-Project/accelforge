import os
import sys
sys.path.insert(0, os.path.abspath('_ext'))
sys.path.insert(0, os.path.abspath('../..'))  # Make your repo importable

import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# -- Project information -----------------------------------------------------
project = 'fastfusion'
author = 'Tanner Andrulis, Michael Gilbert'
release = '0.1.0'

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.autodoc',            # Pull docstrings
    'sphinx.ext.napoleon',           # NumPy / Google style docstrings
    'sphinx.ext.autosummary',        # Generate autodoc summaries
    'sphinx.ext.viewcode',           # Add links to source code
    'sphinx_autodoc_typehints',      # Include type hints
    'include_docstring',             # Include docstrings
    'include_notebook',              # Include notebooks
    'include_attrs',                 # Include attributes & their docstrings
    'include_functions',             # Include functions & their docstrings
    'inherited_attributes',          # Inherit docstrings from parent classes
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'exclude-members': 'model_config,model_fields,__pydantic_fields__,model_post_init',
}

# ---------- Autodoc settings ----------
# Show type hints inline in signatures
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

# Preserve default values
autodoc_preserve_defaults = True

# Force multi-line for long constructor signatures (Sphinx 7+)
autodoc_class_signature = "separated"

# ---------- HTML CSS to wrap signatures ----------
# Create docs/source/_static/custom.css with:
# .signature {
#     white-space: pre-wrap !important;
#     word-break: break-word;
# }
html_static_path = ["_static"]
# html_css_files = ["custom.css"]
# html_js_files = ["custom.js"]

# ---------- Optional: Napoleon settings ----------
# If using Google/NumPy style docstrings
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

nitpicky = True
