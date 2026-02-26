"""Density models for sparse tensor analysis.

Provides pluggable density models that estimate how nonzero elements are
distributed across tiles of a sparse tensor:

- HypergeometricDensityModel: statistical model assuming random sparsity.
- StructuredDensityModel: deterministic model for structured sparsity
  (e.g., 2:4), where every tile has exactly density * tile nonzeros.
"""

import math
from abc import ABC, abstractmethod

from scipy.stats import hypergeom as _hypergeom


class DensityModel(ABC):
    """Abstract base class for density models.

    Subclasses must implement prob_empty, expected_occupancy, and
    expected_occupancy_ceil.
    """

    @abstractmethod
    def prob_empty(self, tile_shape: int) -> float:
        """P(tile is all zeros)."""
        ...

    @abstractmethod
    def expected_occupancy(self, tile_shape: int) -> float:
        """E[nnz in tile]."""
        ...

    @abstractmethod
    def expected_occupancy_ceil(self, tile_shape: int) -> int:
        """ceil(E[nnz in tile])."""
        ...

    @abstractmethod
    def conditioned(self, parent_shape: int, parent_occupancy: float) -> "DensityModel":
        """Return a new model conditioned on the parent rank's fiber statistics."""
        ...


class HypergeometricDensityModel(DensityModel):
    """Statistical model for random sparsity (hypergeometric distribution)."""

    def __init__(self, density: float, tensor_size: int):
        self.N = tensor_size
        self.density = density
        if density <= 0:
            self.r = 0
        elif density >= 1.0:
            self.r = tensor_size
        else:
            self.r = math.ceil(density * tensor_size)

    def prob(self, tile_shape: int, k: int) -> float:
        """P(tile has exactly k nonzeros) -- hypergeometric PMF."""
        if self.N == 0 or tile_shape == 0:
            return 1.0 if k == 0 else 0.0
        n = min(tile_shape, self.N)
        return float(_hypergeom.pmf(k, self.N, self.r, n))

    def prob_empty(self, tile_shape: int) -> float:
        """P(tile is all zeros) = prob(tile_shape, 0)."""
        return self.prob(tile_shape, 0)

    def expected_occupancy(self, tile_shape: int) -> float:
        """E[nnz in tile] = n * r / N (hypergeometric mean)."""
        if self.N == 0:
            return 0.0
        n = min(tile_shape, self.N)
        return n * self.r / self.N

    def expected_occupancy_ceil(self, tile_shape: int) -> int:
        """ceil(E[nnz in tile]) -- used for data capacity."""
        return math.ceil(self.expected_occupancy(tile_shape))

    def prob_at_least(self, tile_shape: int, k: int) -> float:
        """P(tile has >= k nonzeros) = 1 - CDF(k-1)."""
        if self.N == 0 or tile_shape == 0:
            return 1.0 if k <= 0 else 0.0
        n = min(tile_shape, self.N)
        return float(1.0 - _hypergeom.cdf(k - 1, self.N, self.r, n))

    def conditioned(self, parent_shape: int, parent_occupancy: float) -> "HypergeometricDensityModel":
        """Return a new model with N=parent_shape, r=ceil(parent_occupancy)."""
        if parent_shape <= 0 or parent_occupancy <= 0:
            new_r = 0
        else:
            new_r = min(math.ceil(parent_occupancy), parent_shape)
        # Use __new__ + direct assignment to avoid ceil(ceil(x)/N * N) drift
        m = HypergeometricDensityModel.__new__(HypergeometricDensityModel)
        m.N = parent_shape
        m.r = new_r
        m.density = new_r / parent_shape if parent_shape > 0 else 0.0
        return m

    def __repr__(self) -> str:
        return (
            f"HypergeometricDensityModel(density={self.density}, "
            f"N={self.N}, r={self.r})"
        )


class StructuredDensityModel(DensityModel):
    """Deterministic model for structured sparsity (e.g., 2:4)."""

    def __init__(self, density: float, tensor_size: int):
        self.density = density
        self.N = tensor_size

    def prob_empty(self, tile_shape: int) -> float:
        """Structured sparsity guarantees nonzeros in every tile."""
        if self.density <= 0.0 or tile_shape <= 0:
            return 1.0
        return 0.0

    def expected_occupancy(self, tile_shape: int) -> float:
        """Exact: density * min(tile_shape, N)."""
        if self.N == 0 or tile_shape <= 0:
            return 0.0
        return min(tile_shape, self.N) * self.density

    def expected_occupancy_ceil(self, tile_shape: int) -> int:
        """ceil of exact occupancy."""
        return math.ceil(self.expected_occupancy(tile_shape))

    def conditioned(self, parent_shape: int, parent_occupancy: float) -> "StructuredDensityModel":
        """Return a new structured model with narrowed N; density stays fixed."""
        return StructuredDensityModel(self.density, parent_shape)

    def __repr__(self) -> str:
        return (
            f"StructuredDensityModel(density={self.density}, N={self.N})"
        )


def create_density_model(
    density: float,
    tensor_size: int,
    distribution: str | None = None,
) -> DensityModel:
    """Create a density model: 'structured' for deterministic, None for hypergeometric."""
    if distribution == "structured":
        return StructuredDensityModel(density, tensor_size)
    if distribution is not None:
        raise ValueError(f"Unknown density distribution: {distribution!r}")
    return HypergeometricDensityModel(density, tensor_size)


def effectual_operations(total_ops: int, *densities: float) -> int:
    """Number of effectual (all-operands-nonzero) operations.

    Simple product model: effectual = round(total * d1 * d2 * ...).
    """
    result = float(total_ops)
    for d in densities:
        result *= d
    return round(result)
