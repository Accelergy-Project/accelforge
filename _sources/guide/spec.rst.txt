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

Additionally, the :py:class:`~accelforge.frontend.mapper.FFM` class is used to configure
the mapper through the :py:attr:`~accelforge.frontend.spec.Spec.mapper` attribute. Its
usage is described in :doc:`modeling/mapping`.

Input Parsing
-------------

Input specifications can include expressions that are evaluated. The evaluation
is described in the following:

.. toctree::
   :maxdepth: 1

   parsing/evaluation

Additionally, inputs can be specified with YAML files using an extend YAML syntax, which
is described in the following:

.. toctree::
   :maxdepth: 1

   parsing/yaml_parsing
