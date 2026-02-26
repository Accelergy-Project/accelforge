"""Sparse-adjusted occupancy and access count computations.

Computes the impact of sparse tensor formats on storage occupancy and
memory access counts. Functions here are pure math — they take
density/format parameters and return adjusted counts.
"""

import math
from dataclasses import dataclass
from typing import Optional

from accelforge.model.density_model import create_density_model
from accelforge.model.sparse_formats import (
    RankOccupancy,
    _run_format_cascade,
    compute_format_occupancy,
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
    """Compute sparse-adjusted storage occupancy (data + format) for a (tensor, level) pair."""
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
    """Compute per-rank metadata/payload access counts, scaled by algorithmic read/fill ratios."""
    model = create_density_model(density, tensor_size, distribution)
    occupancies, _ = _run_format_cascade(rank_formats, dimension_sizes, model)

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
