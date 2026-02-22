"""Sparse-adjusted occupancy and access count computations.

Computes the impact of sparse tensor formats on storage occupancy and
memory access counts. This is Sparseloop Phases 1-2:
  Phase 1: DefineCompressionFormatModels
  Phase 2: CalculateExpectedOccupancy

Functions here are pure math -- they take density/format parameters and
return adjusted counts. Integration with the AccelForge model pipeline
(symbolic.py, run_model.py) happens in later phases.
"""

import math
from dataclasses import dataclass
from typing import Optional

from accelforge.model.density_model import create_density_model
from accelforge.model.sparse_formats import (
    RankOccupancy,
    compute_format_occupancy,
    create_format_model,
)


@dataclass
class SparseOccupancy:
    """Sparse-adjusted occupancy for a (tensor, level) pair."""

    data_elements: int
    """Number of nonzero data elements (expected occupancy, ceil'd)."""

    data_bits: int
    """Data storage in bits = data_elements * bits_per_value."""

    format_units: float
    """Total format (metadata + payload) units across all ranks."""

    format_bits: float
    """Format storage in bits. Uses metadata/payload word bits if specified."""

    rank_occupancies: list[RankOccupancy]
    """Per-rank occupancy breakdown."""

    @property
    def total_bits(self) -> float:
        """Total storage = data + format."""
        return self.data_bits + self.format_bits


@dataclass
class FormatAccessCounts:
    """Format (metadata) access counts for a (tensor, level) pair."""

    rank_metadata_reads: list[float]
    """Per-rank metadata read counts."""

    rank_payload_reads: list[float]
    """Per-rank payload read counts."""

    rank_metadata_fills: list[float]
    """Per-rank metadata fill counts."""

    rank_payload_fills: list[float]
    """Per-rank payload fill counts."""

    @property
    def total_metadata_reads(self) -> float:
        return sum(self.rank_metadata_reads)

    @property
    def total_payload_reads(self) -> float:
        return sum(self.rank_payload_reads)

    @property
    def total_metadata_fills(self) -> float:
        return sum(self.rank_metadata_fills)

    @property
    def total_payload_fills(self) -> float:
        return sum(self.rank_payload_fills)

    @property
    def total_reads(self) -> float:
        return self.total_metadata_reads + self.total_payload_reads

    @property
    def total_fills(self) -> float:
        return self.total_metadata_fills + self.total_payload_fills


def compute_sparse_occupancy(
    density: float,
    tensor_size: int,
    tile_shape: int,
    bits_per_value: int,
    rank_formats: Optional[list[str]] = None,
    dimension_sizes: Optional[list[int]] = None,
    metadata_word_bits: Optional[list[Optional[int]]] = None,
    payload_word_bits: Optional[list[Optional[int]]] = None,
    distribution: str | None = None,
) -> SparseOccupancy:
    """Compute sparse-adjusted storage occupancy for a (tensor, level) pair.

    Parameters
    ----------
    density : float
        Tensor density (0.0 to 1.0).
    tensor_size : int
        Total tensor size (product of all dimensions).
    tile_shape : int
        Tile size at this level (product of tiled dimensions).
    bits_per_value : int
        Bits per data element.
    rank_formats : list[str], optional
        Format primitives per rank, outer to inner. None = no format (dense).
    dimension_sizes : list[int], optional
        Dimension sizes per rank, outer to inner. Required if rank_formats given.
    metadata_word_bits : list[int|None], optional
        Bits per metadata word per rank. None entries use bits_per_value.
    payload_word_bits : list[int|None], optional
        Bits per payload word per rank. None entries use bits_per_value.
    distribution : str or None
        Density distribution type. None = random (hypergeometric).
    """
    model = create_density_model(density, tensor_size, distribution)

    # Data occupancy
    data_elements = model.expected_occupancy_ceil(tile_shape)
    data_bits = data_elements * bits_per_value

    # Format occupancy
    if rank_formats and dimension_sizes:
        rank_occs, format_units = compute_format_occupancy(
            rank_formats, dimension_sizes, density, tensor_size,
            distribution=distribution,
        )

        # Convert units to bits using per-rank word sizes
        format_bits = 0.0
        if metadata_word_bits is None:
            metadata_word_bits = [None] * len(rank_formats)
        if payload_word_bits is None:
            payload_word_bits = [None] * len(rank_formats)

        for occ, mwb, pwb in zip(rank_occs, metadata_word_bits, payload_word_bits):
            md_bits = mwb if mwb is not None else bits_per_value
            pl_bits = pwb if pwb is not None else bits_per_value
            format_bits += occ.metadata_units * md_bits
            format_bits += occ.payload_units * pl_bits
    else:
        rank_occs = []
        format_units = 0.0
        format_bits = 0.0

    return SparseOccupancy(
        data_elements=data_elements,
        data_bits=data_bits,
        format_units=format_units,
        format_bits=format_bits,
        rank_occupancies=rank_occs,
    )


def compute_format_access_counts(
    rank_formats: list[str],
    dimension_sizes: list[int],
    density: float,
    tensor_size: int,
    tile_shape: int,
    algorithmic_reads: int,
    algorithmic_fills: int,
    distribution: str | None = None,
) -> FormatAccessCounts:
    """Compute format (metadata/payload) access counts for a (tensor, level).

    Format accesses scale with the ALGORITHMIC read/fill ratio (before
    compression). This is because metadata must always be read to enable
    decompression, regardless of how many data values are actually accessed.

    Parameters
    ----------
    rank_formats : list[str]
        Format primitives per rank, outer to inner.
    dimension_sizes : list[int]
        Dimension sizes per rank, outer to inner.
    density : float
        Tensor density.
    tensor_size : int
        Total tensor size.
    tile_shape : int
        Tile size at this level.
    algorithmic_reads : int
        Total algorithmic data reads (before any sparse reduction).
    algorithmic_fills : int
        Total algorithmic data fills (before any sparse reduction).
    distribution : str or None
        Density distribution type. None = random (hypergeometric).
    """
    # Per-tile format occupancy (single tile)
    model = create_density_model(density, tensor_size, distribution)

    occupancies = []
    fibers = 1
    for fmt_name, dim_size in zip(rank_formats, dimension_sizes):
        fmt = create_format_model(fmt_name)
        ennz = model.expected_occupancy(dim_size) if dim_size > 0 else 0.0
        occ = fmt.get_occupancy(fibers, dim_size, ennz)
        occupancies.append(occ)
        fibers = fmt.next_fibers(fibers, dim_size, ennz)

    # Scale by algorithmic tile access ratios
    read_ratio = algorithmic_reads / tile_shape if tile_shape > 0 else 0
    fill_ratio = algorithmic_fills / tile_shape if tile_shape > 0 else 0

    rank_md_reads = []
    rank_pl_reads = []
    rank_md_fills = []
    rank_pl_fills = []

    for occ in occupancies:
        rank_md_reads.append(math.ceil(occ.metadata_units * read_ratio))
        rank_pl_reads.append(math.ceil(occ.payload_units * read_ratio))
        rank_md_fills.append(math.ceil(occ.metadata_units * fill_ratio))
        rank_pl_fills.append(math.ceil(occ.payload_units * fill_ratio))

    return FormatAccessCounts(
        rank_metadata_reads=rank_md_reads,
        rank_payload_reads=rank_pl_reads,
        rank_metadata_fills=rank_md_fills,
        rank_payload_fills=rank_pl_fills,
    )
