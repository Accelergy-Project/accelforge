Modeling Steps
==============

1. **Per-Component Energy, Area, and Leakage**: This step models the area and leakage
   energy of each :ref:`Component` in the architecture. It then generates
   *per-action energy*, which is used by later steps in the model to find the energy of
   performing hardware :ref:`Actions`.

2. **Per-Einsum Mapping**: This step generates candidate :ref:`Pmappings`, which each
   map a single Einsum to the hardware. Each pmapping will incur some number of actions,
   which can be translated into energy and latency using the per-action energy and
   latency values calculated in step 1.

3. **Joining Pmappings**: This step joins the :ref:`Pmappings` into a single
   :ref:`Mapping`, which maps the entire workload onto the hardware.


Per-Component Energy, Area, and Leakage
---------------------------------------

The energy, area, and leakage of each :ref:`Component` in the architecture can be
calculated in two ways. First, it can be defined directly in the architecture. This uses
the following format:

```yaml
- name: main_memory
  attributes:
    _area: 123
    _leak_power: 456
  actions:
  - {name: read, arguments: {energy: 789}}
```

If all necessary energy, area, and leakage values are defined, then the architecture can
be used in step 2 without any additional information. Otherwise, the energy, area, and
leakage values are calculated using the `hwcomponents` library:

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
spec = ff.Spec.from_yaml("arches/tpu_v4i_like.yaml")
spec.calculate_component_energy_area(
    energy=True,
    area=True,
)

```
