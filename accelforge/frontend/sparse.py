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

from typing import Optional

from accelforge.frontend.renames import TensorName
from accelforge.util._basetypes import EvalableModel, EvalableList


class RankFormat(EvalableModel):
    """Per-rank format specification (expert mode)."""

    format: str
    """Format primitive name: UOP, CP, B, or RLE."""

    metadata_word_bits: Optional[int] = None
    """Bits per metadata word. None uses default (data word size)."""

    payload_word_bits: Optional[int] = None
    """Bits per payload word. None uses default (data word size)."""


class RepresentationFormat(EvalableModel):
    """Per-tensor format specification at a storage level.

    Either ``format`` (auto-expanded) or ``ranks`` (explicit) must be provided.
    If both are given, ``ranks`` takes precedence.
    """

    name: str
    """Tensor name."""

    format: Optional[str] = None
    """User-friendly format name: csr, coo, bitmask, rle.
    Auto-expanded to per-rank primitives based on the number of
    non-trivial dimensions at the storage level."""

    ranks: Optional[EvalableList[RankFormat]] = None
    """Explicit per-rank format specification (expert mode).
    Ordered outer-to-inner."""

    metadata_word_bits: Optional[int] = None
    """Bits per metadata word. None = auto-derive from format:
    bitmask → 1, cp/csr/coo → ceil(log2(fiber_shape))."""

    metadata_storage_width: Optional[int] = None
    """Physical SRAM width in bits for metadata word packing (e.g. 28).
    None = skip physical packing (use logical counts)."""

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
        return [RankFormat(format=p) for p in primitives]


class ActionOptimization(EvalableModel):
    """Storage action filtering (SAF) optimization at a memory level."""

    kind: str
    """Optimization type: gating, skipping, or position_skipping."""

    target: str
    """Target tensor whose accesses are optimized."""

    condition_on: list[str]
    """Tensors whose sparsity is used to decide whether to skip/gate."""


class ComputeOptimization(EvalableModel):
    """Compute-level optimization (gating/skipping at the MAC)."""

    kind: str
    """Optimization type: gating or skipping."""

    target: str
    """Target tensor (typically the output accumulator)."""

    condition_on: list[str]
    """Tensors whose sparsity determines whether compute is effectual."""


class SparseTarget(EvalableModel):
    """Sparse optimization configuration for one hardware component."""

    target: str
    """Component name (e.g., Buffer, BackingStorage, Reg, MAC)."""

    representation_format: EvalableList[RepresentationFormat] = EvalableList()
    """Compressed representation formats for tensors at this level."""

    action_optimization: EvalableList[ActionOptimization] = EvalableList()
    """Storage action filtering optimizations at this level."""

    compute_optimization: EvalableList[ComputeOptimization] = EvalableList()
    """Compute-level optimizations at this level."""


class SparseOptimizations(EvalableModel):
    """Top-level sparse optimizations specification.

    Contains a list of per-target configurations. Multiple entries may
    reference the same target (they are logically merged by consumers).
    """

    targets: EvalableList[SparseTarget] = EvalableList()
    """Per-component sparse optimization configurations."""

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
        """Check if a tensor has a compressed format at a component."""
        return len(self.get_formats_for(component_name, tensor_name)) > 0
