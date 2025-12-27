.. _specifying-workload:

Specifying the Workload
=======================

The :py:class:`~fastfusion.frontend.workload.Workload` object describes a cascade of
Einsums. An Einsum, described in ..., can represent a variety of tensor algebra kernels,
and a cascade of Einsums is a list of Einsums with data dependencies.

The following is an example workload for three back-to-back matrix multiplications:

The top-level Workload spec has the following attributes:

.. include-attrs:: fastfusion.frontend.workload.Workload

Each Einsum in the workload represents a single Einsum with the following attributes:

.. include-attrs:: fastfusion.frontend.workload.Einsum

Renaming Tensors and Rank Variables
-----------------------------------
:label:`renaming-tensors-rank-variables`

Renames allow us to write simple, generic names (*e.g.,* ``input``,
``reduced_rank_variable``) in our set expresssions and have them resolve to tensors or
rank variable in the Einsum.

Each Einsum object has a ``renames`` attribute. This attribute may be populated with one
of the following:

- A dictionary of ``{new_name: source_set_expression}`` expressions, where
  ``source_set_expression`` may resolve either to tensors or rank variables. This is the
  simplest method.
- A list of dictionaries, each one having the structure ``{name: new_name, source:
  source_set_expression, expected_count: 1}``. This method allows you to write an
  expected count, which is optional, and checks that your set expression returned the
  expected number of elements. For example, if your source set expression were
  ``Outputs()``, an expected count of 1 would pass if there were only one output tensor,
  but fail if there were two.


Additionally, you may define a separate top-level
:py:class:`~fastfusion.frontend.renames.Renames` object with structure mirroring the
workload. For example, one is in the bottom of the following workload:

.. include:: ../../../examples/workloads/gpt3_6.7B.workload.yaml
   :code: yaml

This renames format includes, for every Einsum, a ``tensor_accesses`` key and a
``rank_variables`` key. Both support the above dictionary or list-of-dictionary rename
formats.

If an Einsum in the renames is named ``default``, then its renames are applied to every
Einsum unless overridden.
