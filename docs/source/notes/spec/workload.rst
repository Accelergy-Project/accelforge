.. _specifying-workload:

Workload and Renames Specification
==================================

The :py:class:`~accelforge.frontend.workload` object describes a cascade of
Einsums. An Einsum, described in ..., can represent a variety of tensor algebra kernels,
and a cascade of Einsums is a list of Einsums with data dependencies.

The following is an example workload for three back-to-back matrix multiplications:

.. include:: ../../../../examples/workloads/three_matmuls.workload.yaml
   :code: yaml

The top-level Workload spec has the following attributes:

.. include-attrs:: accelforge.frontend.workload

Each Einsum in the workload represents a single Einsum with the following attributes:

.. include-attrs:: accelforge.frontend.workload.Einsum

And each tensor access has the following attributes:

.. include-attrs:: accelforge.frontend.workload.TensorAccess

Workloads include *ranks* and *rank variables*. Ranks are the dimensions of the tensors
in the Einsum, while rank variables are variables that index into these ranks. Generally
the rank names are uppercased versions of the rank variable names, but not always. In
more-complex workloads (such as the GPT example later in this doc), there may be cases
where we index into a rank with multiple different rank variables-- in this case, we may
use a projection dictionary instead of a list.

.. code-block:: yaml

  - name: Matmul0
    tensor_accesses:
    - {name: T0, projection: [m, n0]} # Implies projection: {M: m, N0: n0}
    - {name: W1, projection: [k, n0]} # Implies projection: {K: k, N0: n0}
    - {name: T1, projection: [n0, n1], output: True} # Implies projection: {N0: n0, N1: n1}

  - name: Matmul1
    tensor_accesses:
    # We can be explicit about the projection
    - {name: T1, projection: {M: m, N1: n1}}
    - {name: W1, projection: {N1: n1, N2: n2}}
    - {name: T2, projection: {M: m, N2: n2}, output: True}

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
:py:class:`~accelforge.frontend.renames.Renames` object with structure mirroring the
workload. For example, one is in the bottom of the following workload:

.. include:: ../../../../examples/workloads/gpt3_6.7B.workload.yaml
   :code: yaml

This renames format includes, for every Einsum, a ``tensor_accesses`` key and a
``rank_variables`` key. Both support the above dictionary or list-of-dictionary rename
formats.

If an Einsum in the renames is named ``default``, then its renames are applied to every
Einsum unless overridden.
