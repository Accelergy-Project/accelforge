Modeling Steps
==============

1. **Per-Component Energy, Area, and Leakage**: This step models the area and leakage
   energy of each :py:class:`~fastfusion.frontend.arch.Component` in the architecture.
   It then generates *per-action energy*, which is used by later steps in the model to
   find the energy of performing hardware
   :py:class:`~fastfusion.frontend.arch.Action`s.

2. **Per-Einsum Mapping**: This step generates candidate :ref:`Pmappings`, which each
   map a single Einsum to the hardware. Each pmapping will incur some number of actions,
   which can be translated into energy and latency using the per-action energy and
   latency values calculated in step 1.

3. **Joining Pmappings**: This step joins the :ref:`Pmappings` into a single
   :ref:`Mapping`, which maps the entire workload onto the hardware.


Per-Component Energy, Area, and Leakage
---------------------------------------

The energy, area, and leakage of each :py:class:`~fastfusion.frontend.arch.Component` in
the architecture can be calculated in two ways. First, it can be defined directly in the
architecture. This uses the following format:

```yaml
- name: main_memory
  attributes:
    _area: 123
    _leak_power: 456
  actions:
  - {name: read, arguments: {energy: 789}}
  - {name: write, arguments: {energy: 789}}
```

If all necessary energy, area, and leakage values are defined, then the architecture can
be used in step 2 without any additional information. If energy, area, and/or leakave
values are not defined, they can be calculated using the `hwcomponents` library:

.. _hwcomponents: https://github.com/Accelergy-Project/hwcomponents

Additional information can be added to the component to help with the calculation of the
energy, area, and leakage values. To call the `hwcomponents` library, `component_class`
must be provided, and additional non-underscored:ref:`underscore-discussion` attributes
can be provided to help the calculation.


```yaml
- name: main_memory
  component_class: DRAM
  attributes:
    _area: 123
    _leak_power: 456
  actions:
  - {name: read, arguments: {energy: 789}}
```

```python
spec = ff.Spec.from_yaml("arches/tpu_v4i_like.arch.yaml")
spec.calculate_component_energy_area(
    energy=True,
    area=True,
)

```

Total energy of a component is calculated by the sum, across all actions, of the energy
per action times the number of actions incurred as calculated in :ref:`calculating-num-actions`.

Per-Component Latency
---------------------

The latency of components can be specified in their attributes. Pythonic functions of
latency can be


Calculating Number of Actions
-----------------------------

Except for :py:class:`~fastfusion.frontend.arch.Compute`\ components (whose number of
compute actions, barring recomputation, depends only on workload), the number of actions
incurred by most :py:class:`~fastfusion.frontend.arch.Component`\ s depends on the
component type, the workload, and the mapping.

For :py:class:`~fastfusion.frontend.arch.Memory` and
:py:class:`~fastfusion.frontend.arch.ProcessingStage` components, the number of actions
depends on the number of accesses to the component. They may be accessed in two ways:
- `read`: The component is read from a lower-level component, or output values are read up
  to a higher-level component.
- `write`: The component is written to a lower-level component, or input values are
  written from a higher-level component.

The number of actions incurred by accesses for each tensor are equal to the number of
values accessed times the datawidth of the tensor (determined by that component's
:py:class:`~fastfusion.frontend.arch.TensorHolderAttributes`), divided by the
:py:class:`~fastfusion.frontend.arch.Action`\ s'
:py:class:`~fastfusion.frontend.arch.ActionArguments`' `bits_per_action` attribute. For
example, if 1024 values are accessed with a datawidth of 16 bits and `bits_per_action`
is 32, then 1024 * 16 / 32 = 512 actions are incurred.

Read+Modify+Writes (RMWs) to a component are counted as a read and a write. The first
read of output data is skipped because the value has not been written yet.
