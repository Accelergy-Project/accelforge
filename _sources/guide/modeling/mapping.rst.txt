Mapping with Fast & Fusiest
===========================

Mapping workloads onto accelerators uses the Fast and Fusiest Mapper (FFM), which
includes of two parts:

- The Turbo-Charged Pmapper: This part makes all Pareto-optimal pmappings for all
  Einsums.
- Fast and Fusiest Mapper (FFM): This part takes the Pareto-optimal pmappings and joins
  them into full mappings.

This document will walk you through how to use FFM to map a workload onto an
accelerator.

This document follows the ``notebooks/tutorials/mapper.ipynb`` notebook.

Creating a Spec
---------------

Before we dive into the mapper, we need to set up a
:py:class:`~accelforge.frontend.spec.Spec` object with the input spec. We can initialize
:py:class:`~accelforge.frontend.spec.Spec` objects from YAML files.

.. include-notebook:: notebooks/tutorials/mapper.ipynb
   :name: make_spec
   :language: python

We can set optimization metrics for the mapper by setting the `spec.mapper.metrics`
attribute to one of the :py:class:`~accelforge.frontend.mapper.metrics.Metrics` enum
values or a logical OR (|) of multiple values.

The following optimization metrics are available:

.. include-attrs:: accelforge.mapper.Metrics

Calling the Mapper
------------------

We call the Turbo-Charged Pmapper with the
:py:func:`~accelforge.frontend.spec.Spec.map_workload_to_arch` function.

.. include-notebook:: notebooks/tutorials/mapper.ipynb
   :name: map_workload_to_arch
   :language: python

In this code, there is a :py:attr:`~accelforge.frontend.mapper.FFM.max_fused_loops`
parameter that makes mapping faster by limiting the number of fused loops that can exist
in a single pmapping. The :py:class:`~accelforge.frontend.mapper.FFM` class has a
variety of knobs that can be used to speed up mapping:

.. include-attrs:: accelforge.frontend.mapper.FFM

Interpreting Output Results
---------------------------

The mapper outputs a :py:class:`~accelforge.mapper.FFM.mappings.Mappings` object, which
contains all Pareto-optimal mappings found for the given cascade of Einsums.

In general, if there is only one objective function (such as energy), this will include
only one mapping. If there are multiple objective functions, then many mappings may be
Pareto-optimal.

We can access the mappings in the :py:class:`~accelforge.mapper.FFM.mappings.Mappings`
object with the :py:func:`~accelforge.mapper.FFM.mappings.Mappings.energy` and
:py:func:`~accelforge.mapper.FFM.mappings.Mappings.latency` functions to get the energy
and latency of the mapping.

.. include-notebook:: notebooks/tutorials/mapper.ipynb
   :name: mapping_stats
   :language: python

The :py:class:`~accelforge.mapper.FFM.mappings.Mappings` object is a wrapper around a
dataframe that contains all of the resulting mapping stats. These stats are accessible
using a variety of functions:

.. include-functions:: accelforge.mapper.FFM.mappings.Mappings

How the Mapper Works, Debugging, and Advanced Usage
===================================================

The Fast and Fusiest Mapper (FFM) works using *partial mappings*, pmappings, which are
mappings for a subset of the workload.

The mapper works in two steps:

1. Make Partial Mappings: This part uses the Turbo-Charged Pmapper to all Pareto-optimal
   pmappings for all Einsums.
2. Join Partial Mappings: This part uses the Fast and Fusiest Mapper (FFM) to join the
   Pareto-optimal pmappings into full mappings.

To help with debugging, we can run each of these steps separately to inspect the
results.

Making Partial Mappings
-----------------------

Partial mappings are generated using the
:py:func:`~accelforge.mapper.FFM.main.make_pmappings` function. Pmappings are generated
in two stages.

1. Generate Pmapping Templates: This stage generates pmapping templates, which are
   pmappings without the loop bounds filled in.
2. Fill Pmapping Templates with Tile Shapes: This stage fills the loop bounds in the
   pmapping templates with tile shapes.

When :py:func:`~accelforge.mapper.FFM.main.make_pmappings` is called, it first generates
all pmapping templates for each Einsum. Each pmapping template starts as an ordering of
storage nodes (dataplacement) as well as a compute node choice (if there are multiple
compute nodes in the architecture). The mapper will then insert spatial and temporal
loops into the pmapping template.

Then, for each pmapping template, the mapper will generate all possible loop bounds for
the temporal and spatial loops. As it enumerates loop bound choices, it will prune
non-Pareto-optimal combinations to keep the search space tractable.

For debugging in this stage, you can use two main tools. First, you can directly inspect
the generated pmapping templates and check if all expected templates are included. These
will be logged as they are generated.

Next, you can inspect the outputted
:py:class:`~accelforge.mapper.FFM.pmappings.MultiEinsumPmappings` object. This object
includes a variety of functions to report the porportion of pmappings that were kept or
removed for various reasons, such as resource oversubscribtion or Pareto pruning.

.. include-functions:: accelforge.mapper.FFM.pmappings.MultiEinsumPmappings

Joining Partial Mappings
------------------------

After we have all Pareto-optimal pmappings for all Einsums, we can join them into full
mappings with the :py:func:`~accelforge.mapper.FFM.main.join_pmappings` function. This
function returns a :py:class:`~accelforge.mapper.FFM.mappings.Mappings` object, which
contains all Pareto-optimal mappings found for the given cascade of Einsums.

Joining will look at the *compatibility* for each pmapping, which is a representation of
how the pmapping exchanges data with other pmappings. Pairs of pmappings must respect
data dependencies to be compatible. Specifically, for every shared tensor, pmappings
must agree on the shared storage node and the loops above the storage node. This ensures
that they agree on the shape of shared tensor tiles and the order and memory level in
which these tiles are exchanged.

If there are no valid mappings to join, an error will be raised reporting the
compatibilities of the pmappings that were considered. It may be useful to
cross-reference these compatibilites with the pmapping templates that were generated by
the pmapping creation stage.
