FastFusion
==========

FastFusion is a framework to model tensor algebra accelerators. It includes flexible,
user-defined specifications for components, architectures, and workloads, and, given
these specifications, quickly finds optimal fused mappings to program the workloads onto
the architectures.

FastFusion is based on multiple other projects. If you use FastFusion in your work,
please refer to :doc:`notes/citation` for how to cite the relevant projects.


This page includes the following:

.. contents::
   :depth: 1
   :local:
   :backlinks: none

Installation
------------

For native installation, install the package from PyPI:

.. code-block:: bash

   pip install fastfusion

Examples
--------

Example notebooks can be found by cloning the repository and navigating to the
``notebooks/examples`` directory.

.. code-block:: bash

   git clone https://github.com/Accelergy-Project/fastfusion.git
   cd fastfusion/notebooks/examples
   jupyter notebook

Additionally, example input files can be found in the ``examples`` directory.

.. code-block:: bash

   git clone https://github.com/Accelergy-Project/fastfusion.git
   cd fastfusion/examples
   ls


Documentation Overview
----------------------

Documentation is organized into the following sections:

- :doc:`Input Specifications <notes/spec>` - Overview of the inputs to fastfusion,
  including specifications of architectures, workloads, and mappings.
- :doc:`Modeling <notes/modeling>` - How FastFusion models the energy, area, and latency
  of an accelerator running a workload.
- :doc:`Citation <notes/citation>` - How to cite FastFusion in your work
- :doc:`Definitions <notes/definitions>` - Definitions of key concepts in FastFusion
- :doc:`Parsing <notes/parsing>` - Parsing of input specifications
- :doc:`Frequently Asked Questions <notes/faqs>` - Frequently asked questions about FastFusion

API Reference
-------------

The complete API reference is available in the :doc:`modules` section, which includes:

- :doc:`fastfusion.frontend <fastfusion.frontend>` - The input specifications for fastfusion
- :doc:`fastfusion.mapper <fastfusion.mapper>` - Algorithms that map workloads onto architectures
- :doc:`fastfusion.util <fastfusion.util>` - Utility functions and helpers

For detailed API documentation, see the :doc:`modules` section.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules

.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :glob:

   notes/*