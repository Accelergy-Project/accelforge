.. _component-modeling:

Component Energy and Area
=========================

The energy and area of components in the architecture can either be specified directly
or by calls to the `HWComponents <https://github.com/Accelergy-Project/hwcomponents>`_
library.

Calculating Energy and Area
---------------------------

Component energy and area calculations will populate the following fields for each
component. If these fields are pre-specified, then they may be used as input to the
energy and area calculations.

- ``area``: :docstring:`accelforge.frontend.arch.Component.area`
- ``leak_power``: :docstring:`accelforge.frontend.arch.Component.leak_power`
- ``actions[<action name>].energy``: :docstring:`accelforge.frontend.arch.Action.energy`
- ``total_area``: :docstring:`accelforge.frontend.arch.Component.total_area`
- ``total_leak_power``: :docstring:`accelforge.frontend.arch.Component.total_leak_power`
- ``component_modeling_log``:
  :docstring:`accelforge.frontend.arch.Component.component_modeling_log`
- ``component_model``: :docstring:`accelforge.frontend.arch.Component.component_model`

Additionally, the following fields will affect the energy and area calculations:

- ``energy_scale``: :docstring:`accelforge.frontend.arch.Component.energy_scale`
- ``leak_power_scale``: :docstring:`accelforge.frontend.arch.Component.leak_power_scale`
- ``area_scale``: :docstring:`accelforge.frontend.arch.Component.area_scale`
- ``n_parallel_instances``:
  :docstring:`accelforge.frontend.arch.Component.n_parallel_instances`
- ``actions[<action name>].energy_scale``:
  :docstring:`accelforge.frontend.arch.Action.energy_scale`

The energy and area of a all components in the architecture can be calculated by calling
:py:meth:`~accelforge.frontend.spec.Spec.calculate_component_area_energy_latency_leak`.
Note that an Einsum name can be provided to populate symbols with the Einsum's symbols
from the workload; otherwise, if the architecture depends on something in the workload,
an error will be raised.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: spec_energy_area
   :language: python

We can also calculate the energy and area of individual components by calling
:py:meth:`~accelforge.frontend.arch.Component.calculate_area_energy_latency_leak` on
them.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: single_component_energy_area
   :language: python

There are additional :py:class:`~accelforge.frontend.config.Config` fields that affect
the energy and area calculations, which can be set in the
:py:class:`~accelforge.frontend.spec.Spec` object's ``config`` field:

.. include-attrs:: accelforge.frontend.config.Config

Specifying Area, Energy, Latency, and Leak Power
------------------------------------------------

One way to specify the area, energy, latency, and leak power of each component is to
directly set the ``area``, ``leak_power``, ``actions[<action name>].energy``, and
``actions[<action name>].latency`` fields. The following example from the TPU v4i
example architecture shows uses this approach:

.. include-yaml:: examples/arches/tpu_v4i.yaml
   :startfrom: GlobalBuffer
   :same-indent:

If any value is omitted, it will raise an appropriate error when
:py:meth:`~accelforge.frontend.spec.Spec.calculate_component_area_energy_latency_leak`
is called, so you may call this function to check whether you've missed anything.
``hwcomponents`` is invoked automatically if any of the fields are missing. If you don't
want it to be called, then you can do one of the following:

- If calling
  :py:meth:`~accelforge.frontend.spec.Spec.calculate_component_area_energy_latency_leak`,
  then you can set ``spec.config.component_models`` and
  ``spec.config.use_installed_component_models`` to an empty list and ``False``,
  respectively.
- If calling
  :py:meth:`~accelforge.frontend.arch.Component.calculate_area_energy_latency_leak`,
  then you can set ``models`` to an empty list.


Using the ``hwcomponents`` Library
-----------------------------------

``hwcomponents`` is invoked automatically when area and energy are not specified. The
following shows the fields used by ``hwcomponents``:

.. include:: ../../../../examples/misc/component_annotated.yaml
   :code: yaml

When ``hwcomponents`` has been used to calculate the energy and area of a component,
then the ``component_model`` field will be set to the ``hwcomponents`` model used to
calculate the energy and area.

In addition to looking at the ``component_modeling_log`` field, we can further inspect
the ``component_model`` field to see more information about the model.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: hwcomponents
   :language: python
