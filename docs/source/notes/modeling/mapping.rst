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

This document follows the `notebooks/tutorials/FFM.ipynb` notebook.

Creating a Spec
------------------------

Before we dive into the mapper, we need to set up a
:py:class:`~fastfusion.frontend.spec.Spec` object with the input
spec. We can initialize
:py:class:`~fastfusion.frontend.spec.Spec` objects from YAML files.

.. include-notebook:: notebooks/tutorials/FFM.ipynb
   :name: make_spec
   :language: python

We can set optimization metrics for the mapper by setting the `spec.mapper.ffm.metrics`
attribute to one of the :py:class:`~fastfusion.mapper.FFM.Metrics` enum values or a
logical OR (|) of multiple values.

The following optimization metrics are available:

.. include-attrs:: fastfusion.mapper.FFM.Metrics

Making Partial Mappings
-----------------------

We call the Turbo-Charged Pmapper with the
:py:func:`~fastfusion.mapper.FFM.main.make_pmappings` function. This function returns a
:py:class:`~fastfusion.mapper.FFM.main.MultiEinsumPmappings` object, which contains all
Pareto-optimal pmappings for all Einsums.

.. include-notebook:: notebooks/tutorials/FFM.ipynb
   :name: make_pmappings
   :language: python

In this code, there is a ``max_fused_loops`` parameter that makes mapping faster by
limiting the number of fused loops that can exist in a single pmapping. The
``spec.mapper.ffm`` object has a variety of knobs that can be used to speed up mapping:

.. include-attrs:: fastfusion.frontend.mapper.FFM

To help with debugging, the :py:func:`~fastfusion.mapper.FFM.main.make_pmappings`
function will output all pmapping templates that it generates. A pmapping template is a
pmapping that has not been filled in with tile shapes; meaning that it is a stack of
loop nodes and storage nodes with loop bounds left unfilled.

If no valid pmappings are found for a given Einsum, it may be helpful to inspect the
pmapping templates outputted. The
:py:class:`~fastfusion.mapper.FFM.pmappings.MultiEinsumPmappings` object has additional
functions that can be used to help with debugging:

.. include-functions:: fastfusion.mapper.FFM.pmappings.MultiEinsumPmappings

Joining Partial Mappings
------------------------

After we have all Pareto-optimal pmappings for all Einsums, we can join them into full
mappings with the :py:func:`~fastfusion.mapper.FFM.main.join_pmappings` function. This
function returns a :py:class:`~fastfusion.mapper.FFM.mappings.Mappings` object, which
contains all Pareto-optimal mappings found for the given cascade of Einsums.

.. include-notebook:: notebooks/tutorials/FFM.ipynb
   :name: join_pmappings
   :language: python

Interpreting the Output
-----------------------

The :py:class:`~fastfusion.mapper.FFM.mappings.Mappings` object includes stats for the
mappings that were found, including, for each pmapping, resource usage and objective
metrics.

To access the stats, we can use the :py:meth:`~fastfusion.mapper.FFM.mappings.Mappings.access`
method, which will return a :py:class:`~fastfusion.mapper.FFM.mappings.Mappings` object
with only the columns that match the given key, and with the key removed from the column
names.

For example, if there are three columns ``Total<SEP>Energy``, ``Total<SEP>Area``, and
``EinsumA<SEP>Energy``, then ``mapping.access("Total")`` will return a Mappings object
with columns ``Energy`` and ``Area``, and ``mapping.access("Energy")`` will return a
Mappings object with columns ``Total`` and ``EinsumA``.

To render a mapping, we can use the
:py:meth:`~fastfusion.mapper.FFM.mappings.Mappings.render` method, which will return a
string representation of the mapping. In a Jupyter notebook, the mapping will render
automatically if it is the last object in the cell. Note that if there is more than one
Pareto-optimal mapping, you must index into a single mapping to render it.
