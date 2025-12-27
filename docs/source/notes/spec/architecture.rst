Architecture
============

The architecture, defined by the :py:class:`~fastfusion.frontend.arch.Arch` class,
describes the hardware that is running the workload. An architecture is represented as a
tree, where branches in the tree represent different compute paths that may be taken.
For the rest of this section, we will assume that the architecture has been *flattened*,
meaning that there are no branches in the tree. The flattening procedure is described in
:ref:`flattening`.

A flattened architecture is a hierarchy of components with a
:py:class:`~fastfusion.frontend.arch.Compute` at the bottom. The following components
are supported:

- :py:class:`~fastfusion.frontend.arch.Memory` components store and reuse data.
- :py:class:`~fastfusion.frontend.arch.ProcessingStage` components perform some
  non-compute action (*e.g.,* quantizing or transferring data).
- :py:class:`~fastfusion.frontend.arch.Compute` components performs the Einsum's
  computation.

In the architecture file, each component is represented by a YAML dictionary. Component
types are preceded by the ``!`` character. An example architecture is shown below:

.. include:: ../../../examples/arches/tpu_v4i_like.arch.yaml
   :code: yaml


Flattening
----------

A given Einsum may be executed only on a single
:py:class:`~fastfusion.frontend.arch.Compute`, and it may use hardware objects between
the root of the tree and the leaf for that
:py:class:`~fastfusion.frontend.arch.Compute`. Flattening an architecture converts a
tree architecture into multiple parallel *Flattened-Architectures*, each one
representing one possible path from the root of the tree to the leaf for that
:py:class:`~fastfusion.frontend.arch.Compute`.

For example, in the architecture above, there are two compute units, the ``scalar_unit``
and the ``mac``. Flattening this architecture will produce two Flattened-Architectures;
one with a ``scalar_unit`` and one with a ``mac``. The partial mappings for each of
these architectures can be combined, and can share hardware that exists above both
compute units.

Inserting a :py:class:`~fastfusion.frontend.arch.Compute` directly into the top-level
architecture hierarchy will create an optional compute path that goes from the top node
to the compute. More complex topologies (*e.g.,* give an upper-level compute a private
cache) can be created by creating sub-branches following :ref:`sub-branches`.


Sub-Branches
------------

.. _sub-branches:

Sub-branches in the architecture can represent different execution paths. The following
branch types are supported:

- :py:class:`~fastfusion.frontend.arch.Parallel` represents multiple parallel branches,
  one of which is executed.
- :py:class:`~fastfusion.frontend.arch.Hierarchical` represents a single hierarchy,
  where each node is a parent of the following nodes.

Sub-branches are written with the following syntax:

.. code-block:: yaml

  - !Memory
    ...

  - !Memory
    ...

  - !Parallel
    nodes:
    - !Hierarchical
      nodes:
      - ... # First-branch nodes
    - !Hierarchical
      nodes:
      - ... # Second-branch nodes

  # If more nodes go down here, they are children of the outer-level node, not the
  !Parallel node.
  - !Memory
    ...

The top-level :py:class:`~fastfusion.frontend.arch.Arch` is a
:py:class:`~fastfusion.frontend.arch.Hierarchical`.


Spatial Fanouts
---------------

Spatial fanouts describe the spatial organization of components in the architecture. Any
component may have spatial fanouts, and fanouts are allowed in any dimension. For
example, in the architecture above, the ``LocalBuffer`` component has a size-4 spatial
fanout in the ``Z`` dimension, meaning that there are 4 instances of the component. All
child components are duplicated in the ``Z`` dimension as well.

The ``ArrayDummy`` component also has a spatial fanout in two dimensions, the
``reuse_input`` and ``reuse_output`` dimensions. Often, dummy components may be used to
instantiate a spatial fanout-- in this case, we use a dummy memory, constraining it to
keep nothing and have zero size, energy, area, and leakage power. This is equivalent to
writing the spatial fanout directly on the component beneath.

Reuse in spatial dimensions may be controlled with the ``reuse`` keyword, which takes in
a set expression that is parsed according to :ref:`set-expressions`. In the example,
nothing is reused spatially betweeen ``LocalBuffer`` instances, while inputs and outputs
are reused across registers in the ``reuse_input`` and ``reuse_output`` dimensions,
respectively.


Constraints
-----------

Constraints restrict the available actions that may be taken by hardware. Constraints
may be attached to any component in the architecture.

Tensors
~~~~~~~

Tensor constraints define how tensors may be stored and used by a particular component.
They are supported by :py:class:`~fastfusion.frontend.arch.Memory` components. They are
represented by the :py:class:`~fastfusion.frontend.constraints.Tensors` class, which
supports the following fields:

.. include-attrs:: fastfusion.frontend.constraints.Tensors

Spatial
~~~~~~~

Spatial constraints how data may be moved between components in a spatial fanout. They
are represented by the :py:class:`~fastfusion.frontend.constraints.Spatial` class, which
supports the following fields:

.. include-attrs:: fastfusion.frontend.constraints.Spatial

Spatial constraints are given as a dictionary, where keys match the name of a spatial
fanout. Extra spatial keys are ignored. The following example shows a spatial constraint
for the ``X`` and ``Y`` fanouts of the ``ArrayDummy`` component:

.. include-yaml:: examples/arches/tpu_v4i_like.arch.yaml
   :startfrom: ArrayDummy
   :same-indent:



Temporal