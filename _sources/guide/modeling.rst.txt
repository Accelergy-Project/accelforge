How Modeling Works
==================

.. _accelerator-modeling:

Modeling calculates the energy, area, and latency of an architecture running a given
workload. This is done in three steps:

1. **Per-Component Energy, Area, and Leakage**: This step models the area and leak power
   power of each :py:class:`~accelforge.frontend.arch.Component` in the architecture. It
   then generates *per-action energy*, which is used by later steps in the model to find
   the energy of performing hardware
   :py:class:`~accelforge.frontend.arch.Action`\ s.

2. **Mapping the Workload onto the Accelerator**: This step generates
   :py:class:`~accelforge.frontend.mapping.mapping.Mapping` objects that map the workload onto
   the hardware.

3. **Modeling the Energy, Area, and Latency of the Mapping**: This step looks at the
   full mapping and calculates the number of hardware actions that occur, using it to
   total the energy and area of the accelerator.

In this package, the mapping and modeling steps are connected, letting the mapper
quickly find mappings that minimize the energy and latency of the accelerator.

These steps are detailed in the following sections:

.. toctree::
   :maxdepth: 1

   modeling/component_energy_area
   modeling/accelerator_energy_latency
   modeling/mapping