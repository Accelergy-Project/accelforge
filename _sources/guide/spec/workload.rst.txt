.. _specifying-workload:

Workload and Renames Specification
==================================

The :py:class:`~accelforge.frontend.workload` object describes a cascade of
Einsums. An Einsum, described in ..., can represent a variety of tensor algebra kernels,
and a cascade of Einsums is a list of Einsums with data dependencies.

The following is an example workload for three back-to-back matrix multiplications:

.. include:: ../../../../examples/workloads/three_matmuls_annotated.yaml
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

.. _renaming-tensors-rank-variables:

Renaming Tensors and Rank Variables
-----------------------------------

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
workload. For example, one is in the bottom of a GPT3 6.7B example workload:

.. include:: ../../../../examples/misc/gpt3_6.7B_verbose_annotated.yaml
   :code: yaml

This renames format includes, for every Einsum, a
:py:attr:`~accelforge.frontend.renames.EinsumRename.tensor_accesses` key and a
:py:attr:`~accelforge.frontend.renames.EinsumRename.rank_variables` key. Both support
the above dictionary or list-of-dictionary rename formats.

If an Einsum in the renames is named ``default``, then its renames are applied to every
Einsum unless overridden. Overriding is specific to a single name, so every rename in
the default must be overridden independently.

Concise Einsum Notation
-----------------------

Einsums can be written using concise string notation instead of verbose YAML
dictionaries. This allows you to concisely write the
:py:attr:`~accelforge.frontend.workload.Einsum.tensor_accesses`,
:py:attr:`~accelforge.frontend.workload.Einsum.renames` attributes in a single string.
This notation does not cover all attributes, though it may be mixed with other
attributes to fully specify the workload. Additionally, workload-level attributes such
as :py:attr:`~accelforge.frontend.workload.Workload.bits_per_value` and
:py:attr:`~accelforge.frontend.workload.Workload.persistent_tensors` can be used to set
these attribtues for all Einsums in the workload, letting you use a concise notation for
each Einsum.

The following is an example of two equivalent workloads, written in both verbose and
concise notation.

.. code-block:: yaml

  workload:
    einsums:
    # Concise version of the V Einsum.
    #
    # - The LHS of the equation is the output tensor name and projection. If not
    #   otherwise specified, the output tensor name is the name of the Einsum.
    # - The RHS of the equation are the input tensor names and projections.
    # Concise
    - V[b, m, h, e] = I[b, m, d] * WV[h, e, d]
    # Verbose
    - name: V
      tensor_accesses:
      - {name: I, projection: [b, m, d]}
      - {name: WV, projection: [h, e, d]}
      - {name: V, projection: [b, m, h, e], output: True}

    # Concise version of the QK Einsum. Notice that we used the M: p notation to support
    # a projection dictionary. Renames are specified separately in the renames section.
    # Concise
    - "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M: p, h, e]"
    # Verbose
    - name: QK
      tensor_accesses:
      - {name: Q, projection: [b, m, h, e]}
      - {name: K, projection: { B: b, M: p, H: h, E: e }}
      - {name: QK, projection: [b, m, p, h], output: True}
      renames: {output: QK, weight: K, input: Q}

    # To specify both the Einsum as a string as well as other attributes, we can use the
    # einsum keyword. Then we can specify additional attributes in the same entry. Note
    # that the einsum keyword will set the tensors and projections for the Einsum, and
    # an error will be raised if these are specified again in the same entry.
    # Concise
    - einsum: I[b, m, d] = I_in[b, m, d]
      is_copy_operation: True
      tensor_accesses: [{name: I_in, bits_per_value: 16}]
      renames: {input: I_in, output: I}
    # Verbose
    - name: I
      is_copy_operation: True
      tensor_accesses:
      - {name: I_in, projection: [b, m, d], bits_per_value: 16}
      - {name: I, projection: [b, m, d], output: True}
      renames: {input: I_in, output: I}

Below is the concise notation equivalent of the previously-shown verbose GPT3 6.7B
example workload:

.. include:: ../../../../examples/workloads/gpt3_6.7B.yaml
   :code: yaml