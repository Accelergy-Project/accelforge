.. _set-expressions:

Set Expressions
===============

Set expressions are used to describe sets of tensors and rank variables. Set expressions
are parsed for each pmapping template, meaning that they can reference specific tensors
for each Einsum.

As an example of a set expression, we can describe all tensors that are not intermediates
using the following:

.. code-block:: yaml

    ~Intermediates

Set expressions can use the full Python syntax, including the following:

- ``&``: Intersection
- ``|``: Union
- ``~``: Complement
- ``-``: Difference

You may also use Pythonic language with set expressions in some locations. For example,
we may want to use input tensors if and only if there are three or fewer total tensors:

.. code-block:: yaml

    Inputs if len(All) > 3 else All

Set expressions are parsed for every Einsum + Flattened-Architecture:ref:`flattening`
combination. The following set expressions are supported:

- ``All``: All tensors used in the current Einsum.
- ``Inputs``: Tensors input to the current Einsum.
- ``Intermediates``: Tensors produced by one Einsum and consumed by another.
- ``Nothing``: The empty set.
- ``Outputs``: Tensors output from the current Einsum.
- ``Persistent``: Tensors that must remain in backing storage for the full duration of
  the workload's execution. See:ref:`persistent-tensors`.
- ``Shared``: Tensors that are shared between multiple Einsums.
- ``Tensors``: Alias for ``All``.

Additionally, the following special variables are available:

- ``Above``: The set of all tensors that are stored in all memory objects above the
  current memory object in the hierarchy. Includes the same caveats as the previous
  ``MemoryObject``.
- ``<Any Tensor Name>``: Resolves to the tensor with the given name. If the tensor is
  not used in the current Einsum, then it resolves to the empty set.
- ``Einsum``: The name of the currently-processed Einsum. May be used in expressions
  such as ``Inputs if Einsum == "Conv" else All``.
- ``EinsumObject``: For complex logic using the Einsum object directly.
- ``MemoryObject.Tensors``: The set of all tensors that are stored in the memory object.
  Architectures are parsed from the top down, so this will only be available
  ``MemoryObject`` has been parsed. Lower-level memory objects may reference upper-level
  memory objects, but not vice versa. Additionally, this may not be used for energy and
  area calculations.

All tensor expressions can be converted into relevant rank variables by accessing
``.rank_variables``, which will return the set of all rank variables that index into the
tensor. If multiple tensors are referenced, then the union of all indexing rank
variables is returned. For example, `MemoryObject.Tensors.rank_variables` will return
the set of all rank variables that index into any of the tensors stored in
`MemoryObject`.

Additional keys can be defined following :ref:`renaming-tensors-rank-variables`.