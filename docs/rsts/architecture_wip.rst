Architecture
============

The architecture describes the hardware that is running the workload. An architecture is
represented as a tree, where branches in the tree represent different compute paths that
may be taken. For the rest of this section, we will assume that the architecture has
been *flattened*, meaning that there are no branches in the tree. The flattening
procedure is described in :ref:`flattening`.

.. _flattening:

Flattening
==========

A given Einsum may be executed only on a single `!Compute`, and it may use hardware
objects between the root of the tree and the leaf for that `!Compute`. Flattening an
architecture converts a tree architecture into multiple parallel *Flattened-Architectures*,
each one representing one possible path from the root of the tree to the leaf for that
`!Compute`.

For example, consider the following architecture:

.. include:: ../../notebooks/examples/arches/tpu_v4i_like.yaml
   :code: yaml

There are two compute units in the architecture, the ``scalar_unit`` and the ``mac``.
Flattening this architecture will produce two Flattened-Architectures; one with a
``scalar_unit`` and one with a ``mac``. The partial mappings for each of these
architectures can be combined, and can share hardware that exists above both compute
units.