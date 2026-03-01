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

There are four node types in LoopTree:
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
