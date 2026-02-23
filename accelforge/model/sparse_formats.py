"""Sparse format occupancy models and auto-expansion.

Implements the four format primitives (UOP, CP, B, RLE) and auto-expansion
from user-friendly names (csr/coo/bitmask/rle) to per-rank primitives.

Also re-exports ``RankFormat`` so internal code can import it from this
module (the per-rank format spec used by the sparse pipeline).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from accelforge.model.density_model import DensityModel

from accelforge.model.density_model import create_density_model
from accelforge.frontend.sparse import RankFormat  # canonical location

# Re-export so existing `from accelforge.model.sparse_formats import RankFormat` still works
__all__ = ["RankFormat", "expand_format", "RankOccupancy"]


@dataclass
class RankOccupancy:
    """Occupancy of a single rank in a sparse format (in units, not bits)."""

    metadata_units: float
    payload_units: float

    @property
    def total(self) -> float:
        return self.metadata_units + self.payload_units


class FormatModel(ABC):
    """Abstract base class for sparse format rank models."""

    @abstractmethod
    def get_occupancy(
        self,
        fibers: int,
        fiber_shape: int,
        expected_nnz_per_fiber: Optional[float] = None,
        density_model: "Optional[DensityModel]" = None,
    ) -> RankOccupancy:
        """Compute occupancy for this format rank.

        Parameters
        ----------
        fibers : int
            Number of fibers at this rank.
        fiber_shape : int
            Number of elements per fiber (dimension size).
        expected_nnz_per_fiber : float, optional
            Expected nonzeros per fiber from density model.
        density_model : DensityModel, optional
            Density model for prob_empty filtering (used by UOP).
        """
        ...

    @abstractmethod
    def next_fibers(
        self,
        fibers: int,
        fiber_shape: int,
        expected_nnz_per_fiber: Optional[float] = None,
        density_model: "Optional[DensityModel]" = None,
    ) -> int:
        """Number of fibers passed to the next inner rank.

        UOP is uncompressed so all sub-fibers exist.
        CP/B/RLE only keep non-empty fibers.
        """
        ...


class UOP(FormatModel):
    """Uncompressed Offset Pair -- stores offset array regardless of density.

    metadata = 0
    payload  = effective_fibers * (fiber_shape + 1)

    When a density_model is provided, empty fibers are filtered out:
    effective_fibers = fibers * (1 - prob_empty(fiber_shape)).
    """

    def get_occupancy(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                      density_model=None):
        # Trivial dimensions (fiber_shape <= 1) produce no payload.
        if fiber_shape <= 1:
            return RankOccupancy(metadata_units=0, payload_units=0)
        effective_fibers = fibers
        if density_model is not None:
            prob_empty = density_model.prob_empty(fiber_shape)
            effective_fibers = fibers * (1 - prob_empty)
        return RankOccupancy(
            metadata_units=0,
            payload_units=effective_fibers * (fiber_shape + 1),
        )

    def next_fibers(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                    density_model=None):
        # Trivial dimensions (fiber_shape <= 1): UOP is transparent.
        if fiber_shape <= 1:
            return fibers * fiber_shape
        effective_fibers = fibers
        if density_model is not None:
            prob_empty = density_model.prob_empty(fiber_shape)
            effective_fibers = fibers * (1 - prob_empty)
        return effective_fibers * fiber_shape


class CP(FormatModel):
    """Coordinate Payload -- stores coordinates of nonzero elements.

    metadata = fibers * ceil(expected_nnz_per_fiber)
    payload  = 0
    """

    def get_occupancy(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                      density_model=None):
        if (
            fibers == 0
            or expected_nnz_per_fiber is None
            or expected_nnz_per_fiber <= 0
        ):
            return RankOccupancy(metadata_units=0, payload_units=0)
        md = fibers * math.ceil(expected_nnz_per_fiber)
        return RankOccupancy(metadata_units=md, payload_units=0)

    def next_fibers(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                    density_model=None):
        if (
            fibers == 0
            or expected_nnz_per_fiber is None
            or expected_nnz_per_fiber <= 0
        ):
            return 0
        return fibers * math.ceil(expected_nnz_per_fiber)


class Bitmask(FormatModel):
    """Bitmask -- one bit per position, regardless of density.

    metadata = fibers * fiber_shape
    payload  = 0
    """

    def get_occupancy(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                      density_model=None):
        return RankOccupancy(
            metadata_units=fibers * fiber_shape,
            payload_units=0,
        )

    def next_fibers(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                    density_model=None):
        if (
            fibers == 0
            or expected_nnz_per_fiber is None
            or expected_nnz_per_fiber <= 0
        ):
            return 0
        return fibers * math.ceil(expected_nnz_per_fiber)


class RLE(FormatModel):
    """Run-Length Encoding -- stores run lengths for nonzero elements.

    metadata = fibers * expected_nnz_per_fiber  (NO ceil -- fractional)
    payload  = 0
    """

    def get_occupancy(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                      density_model=None):
        if (
            fibers == 0
            or expected_nnz_per_fiber is None
            or expected_nnz_per_fiber <= 0
        ):
            return RankOccupancy(metadata_units=0, payload_units=0)
        md = fibers * expected_nnz_per_fiber
        return RankOccupancy(metadata_units=md, payload_units=0)

    def next_fibers(self, fibers, fiber_shape, expected_nnz_per_fiber=None,
                    density_model=None):
        if (
            fibers == 0
            or expected_nnz_per_fiber is None
            or expected_nnz_per_fiber <= 0
        ):
            return 0
        return fibers * math.ceil(expected_nnz_per_fiber)


PRIMITIVES = {
    "UOP": UOP,
    "CP": CP,
    "B": Bitmask,
    "RLE": RLE,
}


def expand_format(user_format: str, num_ranks: int) -> list[str]:
    """Expand a user-friendly format name to per-rank primitives.

    Rules (outer to inner):
        CSR     -> UOP-...-UOP-CP   (num_ranks-1 UOPs + 1 CP)
        COO     -> CP-...-CP        (all CPs)
        bitmask -> UOP-...-UOP-B    (num_ranks-1 UOPs + 1 B)
        RLE     -> UOP-...-UOP-RLE  (num_ranks-1 UOPs + 1 RLE)

    Parameters
    ----------
    user_format : str
        User-friendly format name.
    num_ranks : int
        Number of format ranks (typically = number of non-trivial dimensions).

    Returns
    -------
    list[str]
        Per-rank primitive names, outer to inner.
    """
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1, got {num_ranks}")

    fmt = user_format.lower()
    if fmt == "csr":
        return ["UOP"] * (num_ranks - 1) + ["CP"]
    elif fmt == "coo":
        return ["CP"] * num_ranks
    elif fmt in ("bitmask", "b"):
        return ["UOP"] * (num_ranks - 1) + ["B"]
    elif fmt == "rle":
        return ["UOP"] * (num_ranks - 1) + ["RLE"]
    else:
        raise ValueError(
            f"Unknown format: {user_format!r}. "
            f"Expected one of: csr, coo, bitmask, rle"
        )


def create_format_model(primitive_name: str) -> FormatModel:
    """Create a FormatModel instance from a primitive name."""
    name = primitive_name.upper()
    if name not in PRIMITIVES:
        raise ValueError(
            f"Unknown format primitive: {primitive_name!r}. "
            f"Expected one of: {list(PRIMITIVES.keys())}"
        )
    return PRIMITIVES[name]()


def _run_format_cascade(
    rank_formats: list[str],
    dimension_sizes: list[int],
    model: "DensityModel",
) -> tuple[list[RankOccupancy], float]:
    """Run the format cascade, passing the same density model to all ranks.

    Traverses ranks outer-to-inner, propagating fiber counts.  The same
    density model is used for every rank (matching Sparseloop's approach
    of propagating the same data-tile constraint to all ranks; see
    tiling-tile-info.cpp:155).

    Parameters
    ----------
    rank_formats : list[str]
        Format primitive names, outer to inner.
    dimension_sizes : list[int]
        Dimension size for each rank, outer to inner.
    model : DensityModel
        Density model (shared across all ranks, not conditioned per-rank).

    Returns
    -------
    tuple[list[RankOccupancy], float]
        Per-rank occupancies and total format units (metadata + payload).
    """
    if len(rank_formats) != len(dimension_sizes):
        raise ValueError(
            f"rank_formats length ({len(rank_formats)}) != "
            f"dimension_sizes length ({len(dimension_sizes)})"
        )

    occupancies = []
    fibers = 1
    total = 0.0

    for fmt_name, dim_size in zip(rank_formats, dimension_sizes):
        fmt = create_format_model(fmt_name)
        ennz = model.expected_occupancy(dim_size) if dim_size > 0 else 0.0
        occ = fmt.get_occupancy(fibers, dim_size, ennz, density_model=model)
        occupancies.append(occ)
        total += occ.total
        fibers = fmt.next_fibers(fibers, dim_size, ennz, density_model=model)

    return occupancies, total


def compute_format_occupancy(
    rank_formats: list[str],
    dimension_sizes: list[int],
    density: float,
    tensor_size: int,
    distribution: str | None = None,
) -> tuple[list[RankOccupancy], float]:
    """Compute format occupancy across all ranks of a multi-rank format.

    Traverses ranks outer-to-inner, propagating fiber counts based on each
    rank's format type. Uses the density model for expected nonzero counts.

    Parameters
    ----------
    rank_formats : list[str]
        Format primitive names, outer to inner.
    dimension_sizes : list[int]
        Dimension size for each rank, outer to inner.
    density : float
        Overall tensor density.
    tensor_size : int
        Total tensor size (product of all dimensions).
    distribution : str or None
        Density distribution type. None = random (hypergeometric).

    Returns
    -------
    tuple[list[RankOccupancy], float]
        Per-rank occupancies and total format units (metadata + payload).
    """
    if len(rank_formats) != len(dimension_sizes):
        raise ValueError(
            f"rank_formats length ({len(rank_formats)}) must match "
            f"dimension_sizes length ({len(dimension_sizes)})"
        )

    model = create_density_model(density, tensor_size, distribution)
    return _run_format_cascade(rank_formats, dimension_sizes, model)
