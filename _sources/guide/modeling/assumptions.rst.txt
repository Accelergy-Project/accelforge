Modeling Assumptions
====================

.. _modeling-assumptions:

AccelForge's analytical model relies on the following assumptions.

Energy and Latency Per Action
-----------------------------

- Energy per compute action is constant per Einsum. Each compute action consumes a fixed
  energy and latency.

- Energy per bit and latency per bit are constant for each memory and Einsum
  combination.

- The energy per action of a component is independent of its state (e.g., the number of
  elements stored, the addresses accessed, etc.).

Scheduling, Data Movement, and Memory Allocation
-------------------------------------------------

- All memory is explicitly allocated by the mapping. Allocations happen immediately when
  needed and are freed immediately when no longer needed.

- Memory hierarchy order is always followed per-tensor.

- Data arrives at its destination exactly when needed. There are no landing zones or
  multiple buffering.

- With the exception of sparse optimizations, there is no data-dependent scheduling, and
  all schedules are determined statically by the mapping.

- For each Einsum, all latencies (compute, memory accesses) can be perfectly overlapped.
  The total latency is the maximum of the per-component latencies.

Objective Optimization
----------------------

- All objectives (e.g., energy, latency) increase monotonically with the number of
  hardware actions. This property is used by the mapper for pruning during tile shape
  exploration.

- No hardware action count may be negative for a given Einsum.

Power Gating
------------

- Components with power gating enabled consume zero power if a given spatial unit is not
  used at all by a given Einsum.
