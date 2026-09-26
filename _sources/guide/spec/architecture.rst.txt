Arch Specification
==================

The architecture, defined by the :py:class:`~accelforge.frontend.arch.Arch` class,
describes the hardware that is running the workload. An architecture is represented as a
tree, where branches in the tree represent different compute paths that may be taken.
For the rest of this section, we will assume that the architecture has been *flattened*,
meaning that there are no branches in the tree. The flattening procedure is described in
the :ref:`flattening` section.

A flattened architecture is a hierarchy of components with a
:py:class:`~accelforge.frontend.arch.Compute` at the bottom. The following components
are supported:

- :py:class:`~accelforge.frontend.arch.Memory` components store and reuse data.
- :py:class:`~accelforge.frontend.arch.Toll` components perform some non-compute action
  (*e.g.,* quantizing or transferring data) and charge for data passing through them.
- :py:class:`~accelforge.frontend.arch.Compute` components performs the Einsum's
  computation.

In architecture YAML files, each component is represented by a YAML dictionary. Component
types are preceded by the ``!`` character. An example architecture is shown below:

.. include:: ../../../../examples/arches/tpu_v4i.yaml
   :code: yaml


.. _flattening:

Flattening
----------

A given Einsum may be executed only on a single
:py:class:`~accelforge.frontend.arch.Compute`, and it may use hardware objects between
the root of the tree and the leaf for that
:py:class:`~accelforge.frontend.arch.Compute`. Flattening an architecture converts a
tree architecture into multiple parallel *Flattened Architectures*, each one
representing one possible path from the root of the tree to the leaf for that
:py:class:`~accelforge.frontend.arch.Compute`.

For example, in the architecture above, there are two compute units, the ``scalar_unit``
and the ``mac``. Flattening this architecture will produce two Flattened Architectures;
one with a ``scalar_unit`` and one with a ``mac``. The partial mappings for each of
these architectures can be combined, and can share hardware that exists above both
compute units.

Inserting a :py:class:`~accelforge.frontend.arch.Compute` directly into the top-level
architecture hierarchy will create an optional compute path that goes from the top node
to the compute. More complex topologies (*e.g.,* give an upper-level compute a private
cache) can be created by creating sub-branches following :ref:`sub-branches`.


.. _sub-branches:

Sub-Branches
------------

Sub-branches in the architecture can represent different execution paths. The primary
`~accelforge.frontend.arch.Arch` class is a `~accelforge.frontend.arch.Hierarchical`
node, which represents a single hierarchy where each node is a parent of the following
nodes. Additionally, `~accelforge.frontend.arch.Fork` can branch off from the main
hierarchy. to represent alternate compute paths. They may be written with the following
syntax:

.. code-block:: yaml

  - !Memory
    ...

  - !Memory
    ...

  - !Fork
    nodes:
    - !Memory
      ...
    # This compute is the final node in the Fork. The Fork is terminated afterwards
    # (because we end the list), and the main hierarchy continues.
    - !Compute
      ...

  # Continuing the main hierarchy
  - !Memory
    ...

  - !Compute
    ...



Spatial Fanouts
---------------

Spatial fanouts describe the spatial organization of components in the architecture. Any
component may have spatial fanouts, and fanouts are allowed in any dimension. While any
:py:class:`~accelforge.frontend.arch.Leaf` node can instantiate spatial fanouts, it is
often convenient to use the dedicated :py:class:`~accelforge.frontend.arch.Container`
class.

When a fanout is instantiated, the given component, alongside all of its children, are
duplicated in the given dimension(s). For example, in the TPU v4i architecture above,
the ``LocalBuffer`` component has a size-4 spatial fanout in the ``Z`` dimension,
meaning
that there are 4 instances of the component. The register component has both the size-4
``Z`` fanout spatial fanout, as well as two size-128 spatial fanouts in the ``reuse_input``
and ``reuse_output`` dimensions, respectively.

Reuse in spatial dimensions may be controlled with the
:py:attr:`~accelforge.frontend.arch.Spatial.may_reuse` keyword, which takes in a set
expression that is evaluated according to the set expression section of the :ref:`Set
Expressions <set-expressions>` guide. In the example, nothing is reused spatially
betweeen ``LocalBuffer`` instances, while inputs and outputs are reused across registers
in the ``reuse_input`` and ``reuse_output`` dimensions, respectively. Additionally, the
``reuse`` keyword can be used to force reuse; for example, ``reuse: input`` means that
all spatial instances must use the same input values, otherwise the mapping will be
invalid.

Spatial fanouts support the following keywords:

.. include-attrs:: accelforge.frontend.arch.Spatial

Tensor Holders
--------------

Tensor holders, which include :py:class:`~accelforge.frontend.arch.Memory` and
:py:class:`~accelforge.frontend.arch.Toll` components, hold tensors.


:docstring:`accelforge.frontend.arch.Memory`.

:docstring:`accelforge.frontend.arch.Toll`.

:docstring:`accelforge.frontend.arch.Memory` and
:docstring:`accelforge.frontend.arch.Toll` support the following fields:

.. include-attrs:: accelforge.frontend.arch.TensorHolder

Additionally, :py:class:`~accelforge.frontend.arch.Memory` objects include:

.. include-attrs-except:: accelforge.frontend.arch.Memory accelforge.frontend.arch.TensorHolder

:py:class:`~accelforge.frontend.arch.Toll` objects also include:

.. include-attrs-except:: accelforge.frontend.arch.Toll accelforge.frontend.arch.TensorHolder


Additionally, they have an additional ``tensors`` field, which is used to define the
tensors that are held by the component. They are represented by the
:py:class:`~accelforge.frontend.arch.Tensors` class, which supports the following
fields:

.. include-attrs:: accelforge.frontend.arch.Tensors
