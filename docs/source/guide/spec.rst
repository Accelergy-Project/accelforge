Input Specifications
====================

The :py:class:`~accelforge.frontend.spec.Spec` class is the main class that contains all
inputs to this framework. It includes the following:

.. include-attrs:: accelforge.frontend.spec.Spec

Some of the Spec's inputs are described in the following sections:

.. toctree::
   :maxdepth: 1

   spec/architecture
   spec/mapping
   spec/workload

Input Parsing
-------------

Input specifications can include arithmetic expressions and set expressions. The parsing
is described in the following:

.. toctree::
   :maxdepth: 1

   parsing/arithmetic_parsing

Additionally, inputs can be specified with YAML files using an extend YAML syntax, which
is described in the following:

.. toctree::
   :maxdepth: 1

   parsing/yaml_parsing
