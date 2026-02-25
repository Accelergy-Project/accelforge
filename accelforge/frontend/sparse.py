"""Sparse optimization specification for AccelForge."""

from typing import Literal, Optional

from pydantic import Field

from accelforge.util._basetypes import EvalableModel, EvalableList


class RankFormat(EvalableModel):
    """Per-rank format specification for explicit (expert) format definitions."""

    format: str
    """ Format primitive name: UOP, CP, B, or RLE. """

    metadata_word_bits: Optional[int] = None
    """ Bits per metadata word. None = auto-derived from format primitive. """

    payload_word_bits: Optional[int] = None
    """ Bits per payload word. None = auto-derived from dimension size. """

    flattened_rank_ids: Optional[list[list[str]]] = None
    """ Dimension names flattened into this rank, e.g. [["C", "R"]]. """

    def model_post_init(self, __context__=None) -> None:
        if self.format.upper() not in ("UOP", "CP", "B", "RLE"):
            raise ValueError(
                f"Unknown format primitive {self.format!r}. "
                f"Expected one of: UOP, CP, B, RLE"
            )


class RepresentationFormat(EvalableModel):
    """Per-tensor compressed format at a storage level.

    Specify ``format`` as one of: csr, coo, bitmask, rle.
    """

    _VALID_FORMATS = {"csr", "coo", "bitmask", "b", "rle"}

    name: str
    """ Tensor name (must match a tensor in the workload). """

    format: Optional[str] = None
    """ User-friendly format name (csr, coo, bitmask, rle), auto-expanded to per-rank primitives. """

    ranks: Optional[EvalableList[RankFormat]] = Field(None, exclude=True)
    """ Explicit per-rank format specification (internal), outer-to-inner. """

    metadata_word_bits: Optional[int] = None
    """ Default bits per metadata word for auto-expanded ranks. None = auto-derived per rank. """

    metadata_storage_width: Optional[int] = None
    """ Physical SRAM width in bits for metadata packing. None = fall back to arch. """

    uop_payload_word_bits: Optional[int] = None
    """ Override payload_word_bits for auto-expanded UOP ranks. None = auto-derived. """

    def has_explicit_ranks(self) -> bool:
        """True if explicit per-rank formats were provided (internal)."""
        return self.ranks is not None

    def model_post_init(self, __context__=None) -> None:
        if self.format is not None and self.format.lower() not in self._VALID_FORMATS:
            raise ValueError(
                f"Unknown format {self.format!r}. "
                f"Expected one of: csr, coo, bitmask, rle"
            )

    def get_rank_formats(self, num_ranks: Optional[int] = None) -> list[RankFormat]:
        """Return per-rank formats, auto-expanding if needed.

        Parameters
        ----------
        num_ranks : int, optional
            Number of ranks for auto-expansion. Required if ``format`` is set
            and ``ranks`` is None.

        Returns
        -------
        list[RankFormat]
            Per-rank format specifications, outer-to-inner.
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
    """Storage action optimization at a memory level."""

    kind: Literal["gating", "skipping"]
    """ Optimization type: gating (filter after access) or skipping (skip access). """

    target: str
    """ Tensor whose read accesses are reduced. """

    condition_on: list[str]
    """ Tensors whose sparsity determines the filtering probability.
    Include the target itself for self-conditioned skipping (e.g. condition_on: [A], target: A). """

    @property
    def is_self_conditioned(self) -> bool:
        """True if the target tensor is in its own condition_on list."""
        return self.target in self.condition_on


class ComputeOptimization(EvalableModel):
    """Compute-level optimization (gating or skipping at the MAC)."""

    kind: Literal["gating", "skipping"]
    """ Optimization type: gating (discard result) or skipping (skip entirely). """

    target: str
    """ Target tensor or operation name (e.g., Z, GEMM). """

    condition_on: list[str]
    """ Operand tensors for compute classification. """


class SparseTarget(EvalableModel):
    """Sparse optimization configuration for one hardware component."""

    target: str
    """ Component name from arch YAML (e.g., DRAM, Buffer, Reg, MAC). """

    representation_format: EvalableList[RepresentationFormat] = EvalableList()
    """ Compressed formats for tensors at this level. """

    action_optimization: EvalableList[ActionOptimization] = EvalableList()
    """ Storage action filtering optimizations at this level. """

    compute_optimization: EvalableList[ComputeOptimization] = EvalableList()
    """ Compute-level optimizations (only meaningful on Compute nodes). """


class SparseOptimizations(EvalableModel):
    """Top-level sparse optimizations specification."""

    targets: EvalableList[SparseTarget] = EvalableList()
    """ Per-component sparse optimization configurations. """

    def get_targets_for(self, component_name: str) -> list[SparseTarget]:
        """Return all SparseTarget entries matching a component name.

        Parameters
        ----------
        component_name : str
            The hardware component name to match (e.g., "DRAM", "Buffer").

        Returns
        -------
        list[SparseTarget]
            All SparseTarget entries whose ``target`` matches the component name.
        """
        return [t for t in self.targets if t.target == component_name]

    def get_formats_for(
        self, component_name: str, tensor_name: str
    ) -> list[RepresentationFormat]:
        """Return all RepresentationFormat entries for a (component, tensor) pair.

        Parameters
        ----------
        component_name : str
            The hardware component name to match.
        tensor_name : str
            The tensor name to match.

        Returns
        -------
        list[RepresentationFormat]
            All RepresentationFormat entries at the component for the tensor.
        """
        results = []
        for t in self.get_targets_for(component_name):
            for rf in t.representation_format:
                if rf.name == tensor_name:
                    results.append(rf)
        return results

    def get_action_optimizations_for(
        self, component_name: str
    ) -> list[ActionOptimization]:
        """Return all ActionOptimization entries for a component.

        Parameters
        ----------
        component_name : str
            The hardware component name to match.

        Returns
        -------
        list[ActionOptimization]
            All ActionOptimization entries at the component.
        """
        results = []
        for t in self.get_targets_for(component_name):
            results.extend(t.action_optimization)
        return results

    def get_compute_optimizations_for(
        self, component_name: str
    ) -> list[ComputeOptimization]:
        """Return all ComputeOptimization entries for a component.

        Parameters
        ----------
        component_name : str
            The hardware component name to match.

        Returns
        -------
        list[ComputeOptimization]
            All ComputeOptimization entries at the component.
        """
        results = []
        for t in self.get_targets_for(component_name):
            results.extend(t.compute_optimization)
        return results

    def has_format(self, component_name: str, tensor_name: str) -> bool:
        """Check if a tensor has a compressed format at a component.

        Parameters
        ----------
        component_name : str
            The hardware component name to check.
        tensor_name : str
            The tensor name to check.

        Returns
        -------
        bool
            True if at least one RepresentationFormat entry has ``format``
            or ``ranks`` set.
        """
        return any(
            rf.format is not None or rf.has_explicit_ranks()
            for rf in self.get_formats_for(component_name, tensor_name)
        )
