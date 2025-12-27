YAML Parsing
============

YAML objects can include expressions that are parsed when they are loaded into Python.
To-be-parsed expressions can include Python code, and supported operations include many
standard library functions (*e.g.,* ``range``, ``min``) and functions from the ``math``
standard library (*e.g.,* ``log2``, ``ceil``).

The scope available for parsing includes the following in order of increasing
precedence:

- Variables defined in a top-level :py:class:`~fastfusion.frontend.variables.Variables`
  object.
- Variables defined in outer-level YAML objects. Dictionary keys can be referenced by
  names, and list entries by index. The dot syntax can be used to access dictionaries;
  for example, ``x.y.z`` is equivalent to ``outer_scope["x"]["y"]["z"]``.
- Variables defined in the current YAML object. Dictionary keys may reference each other
  as long as references are not cyclic.

The following is an example of valid parsed data:

.. code-block:: yaml

  variables:
    a: 123
    b: a + 5
    c: min(b, 3)
    d: sum(y for y in range(1, 10))

  # In some later scope
  ... outer_scope:
    x: 123
    y: a + x # Reference top-level variables
    inner_scope:
        a: 3 # Override outer scope
        b: outer_scope.x
        # Statements can be out-of-order if not cyclic referencing
        firt_item: second_item
        second_item: 3

Supported Arithmetic Operations
-------------------------------

The following are available expressions. In addition to the below, Python keywords that
are available witout import (*e.g.,* ``min``) are also available

- ``ceil``: :py:func:`math.ceil`
- ``comb``: `math.comb`
- ``copysign``: `math.copysign`
- ``fabs``: :py:func:`math.fabs`
- ``factorial``: :py:func:`math.factorial`
- ``floor``: :py:func:`math.floor`
- ``fmod``: :py:func:`math.fmod`
- ``frexp``: :py:func:`math.frexp`
- ``fsum``: :py:func:`math.fsum`
- ``gcd``: :py:func:`math.gcd`
- ``isclose``: `math.isclose`
- ``isfinite``: :py:func:`math.isfinite`
- ``isinf``: :py:func:`math.isinf`
- ``isnan``: :py:func:`math.isnan`
- ``isqrt``: :py:func:`math.isqrt`
- ``ldexp``: :py:func:`math.ldexp`
- ``modf``: :py:func:`math.modf`
- ``perm``: :py:func:`math.perm`
- ``prod``: :py:func:`math.prod`
- ``remainder``: :py:func:`math.remainder`
- ``trunc``: :py:func:`math.trunc`
- ``exp``: :py:func:`math.exp`
- ``expm1``: :py:func:`math.expm1`
- ``log``: :py:func:`math.log`
- ``log1p``: :py:func:`math.log1p`
- ``log2``: :py:func:`math.log2`
- ``log10``: :py:func:`math.log10`
- ``pow``: :py:func:`math.pow`
- ``sqrt``: :py:func:`math.sqrt`
- ``acos``: :py:func:`math.acos`
- ``asin``: :py:func:`math.asin`
- ``atan``: :py:func:`math.atan`
- ``atan2``: :py:func:`math.atan2`
- ``cos``: :py:func:`math.cos`
- ``dist``: :py:func:`math.dist`
- ``hypot``: :py:func:`math.hypot`
- ``sin``: :py:func:`math.sin`
- ``tan``: :py:func:`math.tan`
- ``degrees``: :py:func:`math.degrees`
- ``radians``: :py:func:`math.radians`
- ``acosh``: :py:func:`math.acosh`
- ``asinh``: :py:func:`math.asinh`
- ``atanh``: :py:func:`math.atanh`
- ``cosh``: :py:func:`math.cosh`
- ``sinh``: :py:func:`math.sinh`
- ``tanh``: :py:func:`math.tanh`
- ``erf``: :py:func:`math.erf`
- ``erfc``: :py:func:`math.erfc`
- ``gamma``: :py:func:`math.gamma`
- ``lgamma``: :py:func:`math.lgamma`
- ``pi``: :py:func:`math.pi`
- ``e``: :py:func:`math.e`
- ``tau``: :py:func:`math.tau`
- ``inf``: :py:func:`math.inf`
- ``nan``: :py:func:`math.nan`
- ``abs``: :py:func:`abs`
- ``round``: :py:func:`round`
- ``pow``: :py:func:`pow`
- ``sum``: :py:func:`sum`
- ``range``: :py:func:`range`
- ``len``: :py:func:`len`
- ``min``: :py:func:`min`
- ``max``: :py:func:`max`
- ``float``: :py:func:`float`
- ``int``: :py:func:`int`
- ``str``: :py:func:`str`
- ``bool``: :py:func:`bool`
- ``list``: :py:func:`list`
- ``tuple``: :py:func:`tuple`
- ``enumerate``: :py:func:`enumerate`
- ``getcwd``: :py:func:`os.getcwd`
- ``map``: :py:func:`map`

YAML Syntax and Extensions
--------------------------

We use an extended version of the standard YAML syntax, including the ``<<`` and ``<<<``
operators. ``<<``, when used as a dictionary key, will merge the contents of its value
with the current dictionary. ``<<<`` will merge the contents of its value and will merge
nested dictionaries. The ``!nomerge`` tag will block merging from occuring.

The following is a YAML parsing cheat sheet:

.. code-block:: yaml

  # YAML Nodes
  listNode:
  - element1
  - element2

  dict_node:
    key1: value1
    key2: value2

  # Styles
  list_block_style:
  - element1
  - element2
  list_flow_style: {element1, element2}

  dict_block_style:
    key1: value1
    key2: value2
  dict_flow_style: {key1: value1, key2: value2}

  # Anchors, Aliases, and Merge Keys

  # Anchors
  anchored_list_flow_style: &my_anchored_list
  - element1
  - element2
  anchored_list_block_style: &my_anchored_list [1, 2, 3, 4, 5]

  anchored_dict_flow_style: &my_anchored_dict
    key1: value1
    key2: value2
  anchored_dict_block_style: &my_anchored_dict {key1: value1, key2: value2}

  # Aliases
  my_list_alias: *my_anchored_list
  result_of_my_list_alias: [1, 2, 3, 4, 5]

  my_dict_alias: *my_anchored_dict
  result_of_my_dict_alias: {key1: value1, key2: value2}

  # Merge Keys
  anchored_dict_1: &my_anchored_dict
    key1: value1_dict1
    key2: value2_dict1

  anchored_dict_2: &my_anchored_dict2
    key2: value2_dict2
    key3: value3_dict2

  merged_dict:
    <<: [*my_anchored_dict, *my_anchored_dict2] # My_anchored_dict takes precedence

  result_of_merged_dict:
    key1: value1_dict1
    key2: value2_dict1 # Earlier anchors take precedence
    key3: value3_dict2

  merged_dict2:
    <<: *my_anchored_dict
    value2: override_value2 # Override value2

  result_of_merged_dict2:
    key1: value1_dict1
    key2: override_value2

  # Hierarchical Merge Keys
  anchored_dict_hierarchical_1: &my_anchored_dict
    key1: value1_dict1
    key2: {subkey1: subvalue1, subkey2: subvalue2}
    mylist: [d, e, f]
    mylist_nomerge: [4, 5, 6]

  merged_dict_hierarchical:
    <<<: *my_anchored_dict
    key2: {subkey1: override1} # subkey2: subvalue2 will come from the merge
    mylist: [a, b, c]
    mylist_nomerge: !nomerge [1, 2, 3]

  result_of_merged_dict_hierarchical:
    key1: value1_dict1
    key2: {subkey1: override1, subkey2: subvalue2}
    mylist: [a, b, c, d, e, f]
    mylist_nomerge: [1, 2, 3]

  merged_dict_non_hierarchical:
    <<: *my_anchored_dict
    key2: {subkey1: override1} # This will override all of key2
    mylist: [a, b, c]
    mylist_nomerge: !nomerge [1, 2, 3]

  result_of_merged_dict_non_hierarchical:
    key1: value1_dict1
    key2: {subkey1: override1}
    mylist: [a, b, c]
    mylist_nomerge: [1, 2, 3]



Jinja2 Templating
-----------------

We also support Jinja2 templating. To substitute Jinja2 variables, the
``jinja_parse_data`` argument can be passed to the
:py:meth:`~fastfusion.util.basetypes.FromYAMLAble.from_yaml` function. Additional Jinja2
functions are also supported, including:

- ``add_to_path(path)``: Add a path to the search path for the ``include`` function.

- ``cwd()``: Return the current working directory.

- ``find_path(path)``: Find a file in the search path and return the path to the file.

- ``include(path, key)``: Include a file and return the value of the key. For example,
  ``include(path/x.yaml, a)`` will open the file ``path/x.yaml``, look for a top-level
  dictionary, and return the ``a`` key from that dictionary. Multiple levels of indexing
  can be used, such as ``include(path/x.yaml, a.b.c)``.

- ``include_all(path, key)``: Include all files in a directory and return the value of the
  key. For example, ``include_all(path/dir, a)`` will open all files in the directory
  ``path/dir``, look for a top-level dictionary, and return the ``a`` key from that dictionary.

- ``include_text(path)``: Include a file and return the text of the file.

- ``path_exists(path)``: Check if a file exists in the search path.

The following is a Jinja2 template cheat sheet:

.. code-block:: yaml

  # Add files to be included in the environment
  {{add_to_path('path/to/some/dir')}}
  {{add_to_path('path/to/some/other/dir')}}

  variables:
    var1: 5
    var3: "{{cwd()}}/some_file.yaml" # {{cwd()}} is the directory of this file
    var4: "{{find_path('some_file.yaml')}}" # find_path searches all paths added by add_to_path
    var5: {{set_by_jinja}} # Sets the value to a "set_by_jinja" variable that must be defined

    {% if path_exists('some_file.yaml') %} # Check if a file exists
    var6: "some_file.yaml exists" # Include this line if the file exists
    {% else %}

  arch:
    # Include a subset of the file. Index into the structure with
    # dot-separated keys.
    nodes: {{include('other.arch.yaml', 'arch.nodes')}}

  # Include the entire file
  {{include_text('grab_text_from_file.yaml')}}

  compound_components:
    # Include the subsets of multiple files. They will be merged into one list.
    classes: {{include_all('compound_components/*.yaml', 'compound_components.classes')}}


  {% if enable_text_flag|default(False) %}
  text_included_if_enable_text_flag_is_true: |
    This text will be included if enable_text_flag is true. The |default(False) sets
    the default value of enable_text_flag to False if it is not set.
  {% endif %}
