Architecture
============

The architecture describes the hardware that is running the workload. An architecture is
represented as a tree, where branches in the tree represent different compute paths that
may be taken. For the rest of this section, we will assume that the architecture has
been *flattened*, meaning that there are no branches in the tree. The flattening
procedure is described in :ref:`flattening`.

A flattened architecture is a hierarchy of components with a
:py:class:`~fastfusion.frontend.arch.Compute` at the bottom. The following components
are supported:

- :py:class:`~fastfusion.frontend.arch.Memory` components store and reuse data.
- :py:class:`~fastfusion.frontend.arch.ProcessingStage` components perform some
  non-compute action (*e.g.,* quantizing or transferring data).
- :py:class:`~fastfusion.frontend.arch.Compute` components performs the Einsum's
  computation.

An example architecture is shown below:

.. include:: ../../examples/arches/tpu_v4i_like.arch.yaml
   :code: yaml

Flattening
==========

A given Einsum may be executed only on a single
:py:class:`~fastfusion.frontend.arch.Compute`, and it may use hardware objects between
the root of the tree and the leaf for that
:py:class:`~fastfusion.frontend.arch.Compute`. Flattening an architecture converts a
tree architecture into multiple parallel *Flattened-Architectures*, each one
representing one possible path from the root of the tree to the leaf for that
:py:class:`~fastfusion.frontend.arch.Compute`.

For example, consider the following architecture:

.. include:: ../../examples/arches/tpu_v4i_like.arch.yaml
   :code: yaml

There are two compute units in the architecture, the ``scalar_unit`` and the ``mac``.
Flattening this architecture will produce two Flattened-Architectures; one with a
``scalar_unit`` and one with a ``mac``. The partial mappings for each of these
architectures can be combined, and can share hardware that exists above both compute
units.