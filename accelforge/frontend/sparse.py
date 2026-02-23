"""Sparse optimization specification for AccelForge.

Parses the ``sparse_optimizations:`` YAML section. Supports both simplified
format names (csr/coo/bitmask/rle) with auto-expansion and explicit per-rank
format specification for expert use.

Example YAML (simplified)::

    sparse_optimizations:
      targets:
      - target: Buffer
        representation_format:
        - name: A
          format: bitmask
        - name: B
          format: csr
        action_optimization:
        - kind: gating
          target: A
          condition_on: [B]

Example YAML (expert, explicit per-rank)::

    sparse_optimizations:
      targets:
      - target: BackingStorage
        representation_format:
        - name: A
          ranks:
          - format: UOP
            payload_word_bits: 0
          - format: B
"""

from typing import Literal, Optional

from accelforge.frontend.renames import TensorName
from accelforge.util._basetypes import EvalableModel, EvalableList


class RankFormat(EvalableModel):
    """Per-rank format specification (expert mode).

    Ranks are ordered outer-to-inner. The outermost rank indexes the coarsest
    dimension, the innermost rank indexes the finest.
    """

    format: str
    """Format primitive name: UOP, CP, B, or RLE.

    - UOP (Uncompressed Offset Pair): offset array per fiber.
      payload = fibers * (fiber_shape + 1), metadata = 0.
      Empty fibers filtered when density model available.
      Trivial dimensions (fiber_shape <= 1) produce zero overhead.
    - CP (Coordinate Payload): explicit coordinates of nonzeros.
      metadata = fibers * ceil(ennz_per_fiber), payload = 0.
    - B (Bitmask): one bit per position, density-independent.
      metadata = fibers * fiber_shape, payload = 0.
    - RLE (Run-Length Encoding): run lengths for nonzeros.
      metadata = fibers * ennz_per_fiber (fractional), payload = 0.
    """

    metadata_word_bits: Optional[int] = None
    """Bits per metadata word for this rank. None = auto-derive from primitive:
    B → 1, CP → ceil(log2(dim_size)), RLE → ceil(log2(dim_size)).
    Overrides the parent RepresentationFormat.metadata_word_bits. """

    payload_word_bits: Optional[int] = None
    """Bits per payload word for this rank. None = auto-derive:
    UOP → ceil(log2(dim_size + 1)). Set to 0 to make UOP payload free. """

    flattened_rank_ids: Optional[list[list[str]]] = None
    """Dimension names flattened into this rank, e.g. [["C", "R"]].
    When set, fiber_shape = product of those dimension sizes.
    When None, auto-derived from tensor projection order (innermost format
    rank → innermost non-trivial loop dimension, proceeding outward). """


class RepresentationFormat(EvalableModel):
    """Per-tensor compressed format at a storage level.

    Either ``format`` (auto-expanded) or ``ranks`` (explicit) must be provided.
    If both are given, ``ranks`` takes precedence.

    Declaring a format on a tensor at a level has three effects:
    1. Data accesses reduced by floor(count * (1 - density)).
    2. Metadata access counts emitted as metadata_read/metadata_write actions.
    3. Operand marked as "has metadata" for compute classification (NE vs EZ).
    """

    name: str
    """Tensor name (must match a tensor in the workload). """

    format: Optional[str] = None
    """User-friendly format name, auto-expanded to per-rank primitives:

    - csr: (N-1) UOP + 1 CP. Metadata scales with nnz.
    - coo: N CP ranks. More metadata than CSR.
    - bitmask (or b): (N-1) UOP + 1 B. Metadata is density-independent.
    - rle: (N-1) UOP + 1 RLE. Metadata is fractional nnz.

    N = number of non-trivial dimensions (size > 1) at the storage level. """

    ranks: Optional[EvalableList[RankFormat]] = None
    """Explicit per-rank format specification (expert mode), outer-to-inner.
    Overrides ``format`` entirely. The cascade processes ranks outer-to-inner,
    re-conditioning the density model at each rank. """

    metadata_word_bits: Optional[int] = None
    """Default bits per metadata word for all auto-expanded ranks that don't
    specify their own. None = auto-derive per rank (B → 1, CP/RLE →
    ceil(log2(dim_size))). Per-rank metadata_word_bits override this.
    Affects format occupancy (metadata_bits = units * word_bits) and access
    count packing (floor(metadata_storage_width / word_bits) per access). """

    metadata_storage_width: Optional[int] = None
    """Physical SRAM width in bits for metadata packing. Determines how many
    metadata elements fit per SRAM access: floor(msw / word_bits) elements.
    None = fall back to the metadata_read action's bits_per_action in arch,
    then to the data read action's bits_per_action. """

    uop_payload_word_bits: Optional[int] = None
    """Override payload_word_bits for auto-expanded UOP ranks only. None =
    auto-derive from dimension size. 0 = free (zero storage/access cost).
    Does not affect non-UOP ranks or explicit ``ranks``. """

    def get_rank_formats(self, num_ranks: Optional[int] = None) -> list[RankFormat]:
        """Return per-rank formats, auto-expanding if needed.

        Parameters
        ----------
        num_ranks : int, optional
            Number of ranks for auto-expansion. Required if ``format`` is set
            and ``ranks`` is None.
        """
        if self.ranks is not None:
            return list(self.ranks)
        if self.format is None:
            return []
        if num_ranks is None:
            raise ValueError(
                f"num_ranks required to auto-expand format {self.format!r} "
                f"for tensor {self.name}"
            )
        from accelforge.model.sparse_formats import expand_format

        primitives = expand_format(self.format, num_ranks)
        result = []
        for p in primitives:
            if p.upper() == "UOP" and self.uop_payload_word_bits is not None:
                result.append(RankFormat(format=p, payload_word_bits=self.uop_payload_word_bits))
            else:
                result.append(RankFormat(format=p))
        return result


class ActionOptimization(EvalableModel):
    """Storage action filtering (SAF) optimization at a memory level.

    Reduces data reads by exploiting condition tensor sparsity. The optimization
    probability is: prob = 1 - product(P_nonempty_i) over condition_on tensors.
    Applied AFTER format compression. Fills are never reduced by SAF.
    """

    kind: Literal["gating", "skipping", "position_skipping"]
    """Optimization type: gating, skipping, or position_skipping.

    - gating: access initiated then discarded. Uses ceil rounding for
      read-write tensors, floor for read-only. Still consumes bandwidth.
      Does NOT reduce compute latency.
    - skipping: access never initiated. Uses floor rounding. Zero bandwidth.
      DOES reduce compute latency.
    - position_skipping: self-conditioned skipping using the target tensor's
      own format metadata. Requires condition_on: []. Enables position-space
      utilization model for PE load imbalance with spatial mapping.
    """

    target: str
    """Tensor whose read accesses are reduced. Fills are NOT reduced by SAF. """

    condition_on: list[str]
    """Tensors whose sparsity determines the optimization probability.
    P_nonempty = density for scalar access, 1 - prob_empty(tile) for tiled.
    Empty list [] for position_skipping (self-conditioned). """


class ComputeOptimization(EvalableModel):
    """Compute-level optimization (gating/skipping at the MAC).

    Uses a 9-state model: each operand is ENZ (nonzero), EZ (zero, dense format),
    or NE (absent, compressed format). The 9 joint states map to random, gated,
    skipped, or nonexistent compute. Whether an operand has metadata depends on
    whether a representation_format exists for it at any non-compute storage level.
    """

    kind: Literal["gating", "skipping"]
    """Optimization type: gating or skipping.

    - gating: non-effectual ops executed but output discarded. Energy at
      gated_compute rate. Does NOT reduce compute latency.
    - skipping: ops with a "not exist" operand are skipped entirely.
      Zero energy. DOES reduce compute latency.

    Floor rounding for all classifications (pessimistic).
    """

    target: str
    """Target tensor or operation name (e.g., Z, GEMM). """

    condition_on: list[str]
    """Operand tensors for compute classification. Should list 2 tensors for
    the full 9-state model (e.g., [A, B]). With <2, falls back to a simple
    product model. """


class SparseTarget(EvalableModel):
    """Sparse optimization configuration for one hardware component.

    Multiple entries may reference the same component (logically merged).
    """

    target: str
    """Component name from arch YAML (e.g., DRAM, Buffer, Reg, MAC). """

    representation_format: EvalableList[RepresentationFormat] = EvalableList()
    """Compressed formats for tensors at this level. """

    action_optimization: EvalableList[ActionOptimization] = EvalableList()
    """Storage action filtering optimizations at this level. Applied after
    format compression. Outer-level reductions propagate to inner levels. """

    compute_optimization: EvalableList[ComputeOptimization] = EvalableList()
    """Compute-level optimizations (only meaningful on Compute nodes). """


class SparseOptimizations(EvalableModel):
    """Top-level sparse optimizations specification.

    No-op when ``targets`` is empty. Each tensor referenced here must have a
    ``density`` set in the workload (defaults to 1.0 if absent).
    """

    targets: EvalableList[SparseTarget] = EvalableList()
    """Per-component sparse optimization configurations. """

    def get_targets_for(self, component_name: str) -> list[SparseTarget]:
        """Return all SparseTarget entries matching a component name."""
        return [t for t in self.targets if t.target == component_name]

    def get_formats_for(
        self, component_name: str, tensor_name: str
    ) -> list[RepresentationFormat]:
        """Return all RepresentationFormat entries for a (component, tensor) pair."""
        results = []
        for t in self.get_targets_for(component_name):
            for rf in t.representation_format:
                if rf.name == tensor_name:
                    results.append(rf)
        return results

    def get_action_optimizations_for(
        self, component_name: str
    ) -> list[ActionOptimization]:
        """Return all ActionOptimization entries for a component."""
        results = []
        for t in self.get_targets_for(component_name):
            results.extend(t.action_optimization)
        return results

    def get_compute_optimizations_for(
        self, component_name: str
    ) -> list[ComputeOptimization]:
        """Return all ComputeOptimization entries for a component."""
        results = []
        for t in self.get_targets_for(component_name):
            results.extend(t.compute_optimization)
        return results

    def has_format(self, component_name: str, tensor_name: str) -> bool:
        """Check if a tensor has a compressed format at a component.

        Returns True only if at least one RepresentationFormat entry has
        ``format`` or ``ranks`` set (entries with neither are ignored).
        """
        return any(
            rf.format is not None or rf.ranks is not None
            for rf in self.get_formats_for(component_name, tensor_name)
        )
