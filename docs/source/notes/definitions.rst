Definitions
===========

Action
  An action is something performed by a hardware unit. For example, a read or a compute.

Mapping
  A *mapping* is a schedule that maps operations and data movement onto the hardware.

Component
  A component is a hardware unit in the architecture. For example, a memory or a compute
  unit.

Dataflow
  The order in which a mappings iterates over tiles, noting that tiles may be abstract
  before the mapping is fully defined. :ref:`Tile`.

Dataplacement
  Which tile(s) are stored in each memory level of the accelerator, and for what time
  period, noting that tiles and time periods may be abstract before the mapping is fully
  defined. :ref:`Tile`.

Pmapping
  A *partial mapping*, or *pmapping*, is a mapping of a subset of the workload to the
  hardware.

Pmapping Template
  A *pmapping template* is a template for a pmapping. It includes all storage nodes
  (dataplacement) and loop nodes (dataflow), but does not have loop bounds defined (tile
  shapes).

Reuse
  Reuse occurs when a piece of data is used used in multiple computations, but fetched
  fewer times from some memory. For example, we may fetch a piece of data from DRAM to
  on-chip memory once, then use it in ten computations. This would incur nine reuses of
  the piece of data.

Reuse Opportunity
  Reuse opportunity is when a piece of data is used multiple times by the workload. It
  may or may not be turned into reuse if the hardware successfully leverages it.

Tile
  TODO
