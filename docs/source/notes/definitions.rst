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

Tile
  TODO
