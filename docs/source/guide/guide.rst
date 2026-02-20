User Guide
==========

Welcome to the AccelForge user guide. This guide will help you get started with using
AccelForge to design and model tensor algebra accelerators.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Installation
------------

PyPI Installation
~~~~~~~~~~~~~~~~~

For native installation, install the package from PyPI:

.. code-block:: bash

   pip install accelforge

Using Pre-Built Docker Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-built Docker images are available for AccelForge. TODO: Docker instructions

Getting Started
---------------

Quick Start
~~~~~~~~~~~

AccelForge models the energy, area, and latency of accelerator architectures running
tensor algebra workloads. The basic workflow is:

1. **Define your architecture** - Specify the hardware components, their organization,
   and their energy/area characteristics
2. **Define your workload** - Specify the tensor operations you want to execute
3. **Map the workload to the architecture** - Use AccelForge's mapper to find efficient
   ways to execute your workload on the architecture
4. **Analyze results** - Examine the energy, area, and latency of the resulting
   mapping(s)

Basic Example
~~~~~~~~~~~~~

Here's a minimal example of using AccelForge:

.. code-block:: python

   from accelforge import Spec

   # Load architecture and workload specifications
   spec = Spec.from_yaml_files(
       arch="examples/arches/simple.yaml",
       workload="examples/workloads/three_matmuls_annotated.yaml"
   )

   # Map the workload to the architecture
   results = spec.map_workload_to_arch()

   # Analyze the results
   print(f"Energy: {results.energy()} J")
   print(f"Latency: {results.latency()} seconds")


Examples
--------

Example Notebooks
~~~~~~~~~~~~~~~~~

Example Jupyter notebooks can be found by cloning the repository and navigating to the
``notebooks`` directory:

.. code-block:: bash

   git clone https://github.com/Accelergy-Project/accelforge.git
   cd accelforge/notebooks/tutorials
   jupyter notebook

Example Input Files
~~~~~~~~~~~~~~~~~~~

Example architecture, workload, and mapping YAML files can be found in the ``examples``
directory:

.. code-block:: bash

   git clone https://github.com/Accelergy-Project/accelforge.git
   cd accelforge/examples
   ls

The examples directory contains:

- **arches/** - Example architecture specifications, including various published and
  commercial accelerators
- **workloads/** - Example workload specifications including matrix multiplications,
  convolutions, and transformer models
- **mappings/** - Example mapping specifications

**Architecture:** An architecture specification defines the hardware structure,
including hardware components and how they are organized into a system. See
:doc:`spec/architecture` for information on architecture specifications.

**Workload:** A workload specification defines what is executed on the architecture.
Workloads are expressed as a cascade of Einsums. See :doc:`spec/workload` for
information on workload specifications.

**Mapping:** A mapping specifies how a workload is executed on an architecture,
including where and when each operation is performed, and how data is moved and stored
in the hardware's components. See :doc:`spec/mapping` for information on mapping
specifications. AccelForge's mapper automatically finds optimal mappings. See the
``mapper.ipynb`` notebook for examples.

Core Topics
-----------

The following sections provide detailed information about using AccelForge.

**Input Specifications:**: Learn how to specify architectures, workloads, and mappings:

- :doc:`spec` - Complete specification reference
- :doc:`spec/architecture` - Architecture specification details
- :doc:`spec/workload` - Workload specification details
- :doc:`spec/mapping` - Mapping specification details

**Modeling:** Understand how AccelForge models energy, area, and latency:

- :doc:`modeling` - Overview of the modeling process
- :doc:`modeling/component_energy_area` - Component-level modeling
- :doc:`modeling/accelerator_energy_latency` - System-level modeling of the accelerator
- :doc:`modeling/mapping` - How workloads are mapped to architectures

**Definitions:** Key terms and concepts:

- :doc:`definitions` - Definitions of key terms

Reference
---------

Additional Resources
~~~~~~~~~~~~~~~~~~~~

- :doc:`../modules` - Complete API reference
- :doc:`definitions` - Definitions of key terms
- :doc:`faqs` - Frequently asked questions
- :doc:`citation` - How to cite AccelForge
- :doc:`contributing` - Contributing guidelines

Support
-------

If you encounter issues or have questions:

- Check the :doc:`faqs` page
- Review `the AccelForge tutorials
  <https://github.com/Accelergy-Project/accelforge/tree/main/notebooks/tutorials>`_
- Review `the AccelForge examples
  <https://github.com/Accelergy-Project/accelforge/tree/main/examples>`_
- Browse the `source code on GitHub <https://github.com/Accelergy-Project/accelforge>`_
- Open an issue on `GitHub <https://github.com/Accelergy-Project/accelforge>`_

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   spec
   modeling
   definitions
   faqs
   timeloop_compare

Migrating from Timeloop
-----------------------

AccelForge is inspired by and designed to be a successor to the Timeloop [1]_ project,
and incorporates many of the same ideas, including:

- Analytical modeling of energy, area, and latency
- Component, architecture, workload, and mapping as separate objects
- Automated mapping of workloads to architectures

For users of Timeloop, AccelForge can be used as a faster Python-based alternative. Many
of the features of Timeloop are supported by AccelForge, and input specifications are
similar, though not identical.

While AccelForge is under active development and will in the future have a superset of
Timeloop's features, currently the two works have differing feature sets. For users who
are considering migrating, please review the :doc:`timeloop_compare` page for a
comparison of the features supported by each framework.

References
----------

.. [1] A. Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator
   Evaluation," 2019 IEEE International Symposium on Performance Analysis of Systems and
   Software (ISPASS), Madison, WI, USA, 2019, pp. 304-315, doi:
   10.1109/ISPASS.2019.00042. `Code Here <https://github.com/NVlabs/timeloop>`_
