Accelerator Energy, Area, and Latency
=====================================

.. _accelerator-energy-latency:

To calculate energy and latency, we first need to look at the number of actions incurred
by each :py:class:`~accelforge.frontend.arch.Component` in the architecture.

.. _calculating-num-actions:

Calculating Number of Actions from A Mapping
--------------------------------------------

Except for :py:class:`~accelforge.frontend.arch.Compute`\ components (whose number of
compute actions, barring recomputation, depends only on workload), the number of actions
incurred by most :py:class:`~accelforge.frontend.arch.Component`\ s depends on the
component type, the workload, and the mapping.

For :py:class:`~accelforge.frontend.arch.Memory` and
:py:class:`~accelforge.frontend.arch.Toll` components, the number of actions
depends on the number of accesses to the component. They may be accessed in two ways:

- ``read``: The component is read from a lower-level component, or output values are read
  up to a higher-level component.
- ``write``: The component is written to a lower-level component, or input values are
  written from a higher-level component.

The number of actions incurred by accesses for each tensor are equal to the number of
values accessed times the bits per value of the tensor (determined by the workload),
divided by the :py:attr:`~accelforge.frontend.arch.TensorHolderAction.bits_per_action`
attribute. attribute. For example, if 1024 values are accessed with a bits per value of
16 bits and :py:attr:`~accelforge.frontend.arch.TensorHolderAction.bits_per_action` is
32, then 1024 * 16 / 32 = 512 actions are incurred.

Read+Modify+Writes (RMWs) to a component are counted as a read and a write. The first
read of output data is skipped because the value has not been written yet.

By default, the :py:attr:`~accelforge.frontend.arch.TensorHolderAction.bits_per_action`
attributes is set to 1, meaning that memory accesses are counted in terms of bits
accessed unless :py:attr:`~accelforge.frontend.arch.TensorHolderAction.bits_per_action`
is set to a different value.

Calculating Latency from a Pmapping
-----------------------------------

The total latency of a component, defined in the class's
:py:obj:`~accelforge.frontend.arch.Component.total_latency` field, is a Python
expression that is evaluated using the component's actions.

The :py:obj:`~accelforge.frontend.arch.Component.total_latency` field is
:docstring-lower:`accelforge.frontend.arch.Component.total_latency`


Calculating Area and Leak Power
-------------------------------

After :ref:`component-modeling` is completed, we can get area with the
:py:attr:`~accelforge.frontend.arch.Arch.per_component_total_area` and
:py:attr:`~accelforge.frontend.arch.Arch.total_area` attributes. Similarly, we can get
leak power with the
:py:attr:`~accelforge.frontend.arch.Arch.per_component_total_leak_power` and
:py:attr:`~accelforge.frontend.arch.Arch.total_leak_power` attributes.
