YAML Parsing
============

AccelForge inputs can be evaluated from YAML files. YAML parsing occurs once when YAML
files are loaded into Python.

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
:py:meth:`~accelforge.util._basetypes._FromYAMLAble.from_yaml` function. Additional Jinja2
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

.. code-block:: jinja

  # Add files to be included in the environment
  {{add_to_path('path/to/some/dir')}}
  {{add_to_path('path/to/some/other/dir')}}

  # Set defaults for variables. If these are not overridden by jinja parse data, then
  # they will take on the values specified here.
  {% set var1 = var1 | default(5) %}
  {% set var2 = var2 | default(var1 + 1) %}

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
