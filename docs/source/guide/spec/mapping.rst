.. _specifying-mapping:

Mapping Specification
=====================
A *mapping* is the scheduling (in both time and space) of operations in every
computation step in a workload on an architecture. Mappings in AccelForge are written in
LoopTree notation (in Python and YAML equivalents). This documentation covers the
LoopTree notation, and how to write LoopTrees in YAML (and the equivalent Python class).


The LoopTree Notation
---------------------
To illustrate LoopTrees, consider the following cascade of two matrix-vector
multiplications:

.. math::

    A_{nA} = I_{nI} \times WA_{nI,nA} \\
    B_{nB} = A_{nA} \times WB_{nA,nB}

Moreover, we will process these Einsums using an architecture consisting of two memory
levels: OffChipBuffer and OnChipBuffer. Moreover, this architecture has compute units
named ComputeUnit.


The following LoopTree shows loops belonging to each Einsum and which tensors are
reused.

.. include:: ../../../../examples/mappings/simple_fused.yaml
   :code: yaml

There are five node types in LoopTree:
    - *Loop nodes*, rectangles in the LoopTree, represent nested for loops that iterate
      over rank variables in the workload. A loop may be shared between multiple fused
      Einsums, such as the outer :math:`nA` loop.
    - *Storage nodes*, cylinders in the LoopTree, represent the storage for tensor
      tiles.
    - *Compute nodes*, ovals in the LoopTree, represent the Einsum computation that is
      performed in that branch.
    - In this example, they represent the multiply-accumulate operations in the Einsums
      A and B using ComputeUnit.
    - *Splits* in the LoopTree represent points where two or more Einsums are processed
      sequentially.
    - *Binding nodes*, notes in the LoopTree, say how the physically-distributed
      memories of the storage nodes above them spread their tiles over their physical
      instances. See :ref:`binding-nodes` below.


.. _binding-nodes:

Binding Nodes
-------------
A memory that is *physically distributed* has one instance per coordinate of the array
it sits in, so a tile held there is spread over those instances: each spatial loop below
the storage node either partitions the tile along one rank of the tensor or hands every
instance a whole copy of it. Which of the two, and along which rank, is the tile's
*binding* to that memory.

A binding is per-Einsum, while the memory holding a shared tensor is not: two Einsums
that pass a tensor between them agree on where it lives but may want it laid out two
different ways, with an adapter shuffling the data in between. So a binding is a node of
its own rather than part of a storage node. The storage node of a shared tensor sits
above the split, where it speaks for every Einsum; a Binding node sits below the split
when the Einsums disagree and above it when they agree.

One Binding node speaks for every ``(memory, tensor)`` pair held by the storage nodes
above it, as a dictionary from component name to a dictionary from tensor name to that
tensor's binding in that component:

.. code-block:: yaml

    !Binding
    bindings:
      SharedBuffer:      # the memory
        A:               # the tensor it holds
        # One loop per spatial loop below the storage node, in the same order. `X` is
        # partitioned along rank `NA`; `Y` gets a whole copy of the tile (`DUPLICATE`).
        - {name: X, rank_variable: NA, tile_shape: null}
        - {name: Y, rank_variable: DUPLICATE, tile_shape: null}

Only the rank each loop binds is meaningful, and the pairing with the spatial loops
below is positional. Everything else about a binding loop -- how many iterations it
runs, in particular -- comes from the spatial loop it pairs with, and is filled in
before the mapping is modeled, so a binding written by hand gives each loop a
``rank_variable`` and a null ``tile_shape``. A mapping that names no binding is given
the locally-optimal one: partition along the rank each spatial loop's rank variable
indexes, and replicate where it indexes nothing.


Interpreting LoopTrees
----------------------
When interpreting LoopTrees, always keep the following rule in mind: *If we move storage
nodes lower in the LoopTree, then tile size, data reuse, and data lifetime stay the same
or decrease.*

The following can be seen in LoopTrees:

- *Tile size* for a tensor decreases if we move a storage node below a loop that indexes
  into a rank of the tensor. For example, moving :math:`WA_{kA,nA}` below the :math:`nA`
  loop decreases the tile size of :math:`WA_{kA,nA}` because it only needs to store the
  tile needed for one :math:`nA` iteration.
- *Reuse* for a tensor tile decreases if we move a storage node below a loop that does
  not index into a rank of the tensor. For example, moving :math:`WA_{kA,nA}` below the
  :math:`m` loop causes :math:`WA_{kA,nA}` to be re-fetched for each iteration of the
  :math:`m` loop, losing reuse.
- *Lifetime* of a tensor tile---how long a tensor tile lives in memory---decreases if we
  move a storage node below a loop that is above a branch. For example, moving
  :math:`WA` below the :math:`nA` loop but above the branch means :math:`WA` tiles would
  only be alive for the left branch, and could be freed before moving to the right
  branch.
- *Fusion* of two Einsums can be seen in the LoopTree by the absence of the shared
  tensor in off-chip storage node. For example, tensor :math:`A` is not stored off-chip,
  thus it is fused between Einsums A and B.


LoopTrees in YAML
-----------------
Here, we discuss how a LoopTree mapping can be written as YAML text, and used as
input/output to AccelForge.

As a YAML text, each node is a YAML dictionary with a tag representing its node types.
For example, a compute node has the following format::

.. code-block:: yaml
    !Compute  # this is a YAML tag
    einsum: name-of-einsum-to-compute  # a YAML dictionary element "key: value"
    component: name-of-compute-component-that-processes-the-einsum

Detailed documentation of each node types can be found in the API documentation.

In YAML, subsequent nodes in a LoopTree can be written as a list, but this requires an
additional node type: ``Nested``. To use a ``Nested`` node, simply place subsequent
nodes as the list elements of the ``nodes`` key::

.. code-block:: yaml
    !Nested
    nodes:
    - node_0
    - node_1
    - node_2
