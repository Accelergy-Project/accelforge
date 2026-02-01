Definitions
===========

.. glossary::

  Action
    An action is something performed by a hardware unit. For example, a read or a compute.

  Architecture
    An architecture is the hardware that is running the workload. It describes the
    components that are available and how they are connected.

  Component
    A component is a hardware unit in the architecture. For example, a memory or a compute
    unit.

  Dataflow
    The order in which a mappings iterates over tiles, noting that tiles may be abstract
    before the mapping is fully defined. :term:`Tile` for more information.

  Dataplacement
    Which tile(s) are stored in each memory level of the accelerator, and for what time
    period, noting that tiles and time periods may be abstract before the mapping is fully
    defined. See :term:`Tile`. for more information.

  Einsum
    An Einsum is a computation step that executes using tensors. It includes two
    operations; map, which which is performed on combinations of values from its input
    tensors, and reduce, which combines mapped values into outputs. For example, a
    matrix multiplication Einsum uses a multiply operation for the map and a sum
    operation for the reduce.

  Mapping
    A *mapping* is a schedule that maps operations and data movement onto the hardware.

  Pmapping
    A *partial mapping*, or *pmapping*, is a :term:`Mapping` of a subset of the workload
    to the hardware.

  Pmapping Template
    A *pmapping template* is a template for a pmapping. It includes all storage nodes
    (dataplacement) and loop nodes (dataflow), but does not have loop bounds defined
    (tile shapes).

  Rank
    A rank is a dimension of a tensor.

  Reuse
    Reuse occurs when a piece of data is used used in multiple computations, but fetched
    fewer times from some memory. For example, we may fetch a piece of data from DRAM to
    on-chip memory once, then use it in ten computations. This would incur nine reuses
    of the piece of data.

  Reuse Opportunity
    Reuse opportunity is when a piece of data is used multiple times by the workload. It
    may or may not be turned into reuse if the hardware successfully leverages it.

  Tensor
    A tensor is a multi-dimensional array of data. Tensors are produced and consumed by
    :term:`Einsum`\ s. A tensor's shape is parameterized by its :term:`Rank`\ s.

  Tile
    TODO

  Workload
    A workload is a cascade of :term:`Einsum`\ s that are executed by the architecture.
    :term:`Einsum`\ s produce and consume :term:`Tensor`\ s, which are exchanged between
    one another.
