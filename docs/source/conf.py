import os
import sys
sys.path.insert(0, os.path.abspath('_ext'))
sys.path.insert(0, os.path.abspath('../..'))  # Make your repo importable

import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# -- Project information -----------------------------------------------------
project = 'accelforge'
author = 'Tanner Andrulis, Michael Gilbert'
release = '0.1.0'

# -- HTML output -------------------------------------------------------------
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'furo'
html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_book_theme'
# pip3 install sphinx-furo-theme

html_static_path = ['_static']
html_logo = '_static/logo.svg'

extensions = [
    'sphinx.ext.autodoc',            # Pull docstrings
    'sphinx.ext.napoleon',           # NumPy / Google style docstrings
    'sphinx.ext.autosummary',        # Generate autodoc summaries
    'sphinx.ext.viewcode',           # Add links to source code
    'sphinx_autodoc_typehints',      # Include type hints
    'sphinx.ext.intersphinx',        # Link to other projects' documentation
    'include_docstring',             # Include docstrings
    'include_notebook',              # Include notebooks
    'include_attrs',                 # Include attributes & their docstrings
    'include_functions',             # Include functions & their docstrings
    'inherited_attributes',          # Inherit docstrings from parent classes
    'include_yaml',                  # Include subsets of YAML files
    'sphinx_copybutton',             # Add copy button to code blocks
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
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]
# html_js_files = ["custom.js"]

# ---------- Optional: Napoleon settings ----------
# If using Google/NumPy style docstrings
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# Suppress warnings for things we can't control
suppress_warnings = [
    'ref.python',  # Multiple targets for same name
    # 'toc.not_included',  # Documents not in toctree (intentional)
    # 'misc.highlighting_failure',  # YAML lexing issues with Jinja2
]

nitpicky = True

# Suppress warnings for types that are intentionally private or external
nitpick_ignore = [
    # Private internal types (intentionally not documented)
    ('py:class', 'accelforge.util._basetypes.EvalableModel'),
    ('py:class', 'accelforge.util._basetypes.EvalableList'),
    ('py:class', 'accelforge.util._basetypes.EvalableDict'),
    ('py:class', 'accelforge.util._basetypes.ParsableDict'),
    ('py:class', 'accelforge.util._basetypes.EvalsTo'),
    ('py:class', 'accelforge.util._basetypes.TryEvalTo'),
    ('py:class', 'accelforge.util._basetypes.EvalExtras'),
    ('py:class', 'accelforge.util._basetypes.NoParse'),
    ('py:class', 'accelforge.util._basetypes._get_tag'),
    ('py:class', 'accelforge.util._setexpressions.InvertibleSet'),
    ('py:class', 'accelforge.util._setexpressions.InvertibleSet[str]'),
    ('py:class', 'accelforge.frontend.arch._ExtraAttrs'),
    ('py:class', 'accelforge.util.parallel._SVGJupyterRender'),
    # Pydantic internal types (short name from type hints)
    ('py:class', 'Tag'),
    ('py:class', 'pydantic.types.Tag'),
    ('py:class', 'pydantic.types.Discriminator'),
    ('py:class', 'Discriminator'),
    ('py:class', '_get_tag'),
    ('py:class', '_Parallel'),
    # Short names from type hints (EvalsTo, sympy, etc.)
    ('py:class', 'EvalsTo'),
    # Bogus refs from type hint / default rendering
    ('py:class', 'None = None'),
    ('py:class', 'TypeAlias MappingNodeTypes'),
    ('py:class', 'TypeAlias NodeList'),
    # Internal mapper types (not in api docs)
    ('py:class', 'accelforge.mapper.FFM._make_pmappings.pmapper_job.Job'),
    ('py:class', 'accelforge.mapper.FFM._join_pmappings.compatibility.Compatibility'),
    # Method on private base (referenced from guide)
    ('py:meth', 'accelforge.util._basetypes._FromYAMLAble.from_yaml'),
    # Python built-ins and standard library
    ('py:func', 'math.pi'),
    ('py:func', 'math.e'),
    ('py:func', 'math.tau'),
    ('py:func', 'math.inf'),
    ('py:func', 'math.nan'),
    ('py:func', 'range'),
    ('py:func', 'float'),
    ('py:func', 'float'),
    ('py:func', 'int'),
    ('py:func', 'str'),
    ('py:func', 'bool'),
    ('py:func', 'list'),
    ('py:func', 'tuple'),
    # External library internals
    # Collections.abc inherited methods (auto-generated, not controllable)
    ('py:class', "D[k] if k in D, else d.  d defaults to None."),
    ('py:class', "a set-like object providing a view on D's items"),
    ('py:class', "a set-like object providing a view on D's keys"),
    ('py:class', "an object providing a view on D's values"),
    # Short type names from type hints (sphinx-autodoc-typehints emits short names;
    # targets are accelforge.frontend.renames.* and hwcomponents.ComponentModel)
    ('py:class', 'IncEx'),
    ('py:class', 'optional'),

    # Ideally these should be fixed, but idk how
    ('py:class', 'EinsumName'),
    ('py:class', 'Rank'),
    ('py:class', 'RankVariable'),
    ('py:class', 'TensorName'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    # 'matplotlib': ('https://matplotlib.org/stable/contents.html', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'hwcomponents': ('https://accelergy-project.github.io/hwcomponents/', None),
}