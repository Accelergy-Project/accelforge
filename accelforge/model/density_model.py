"""Hypergeometric density model for sparse tensor analysis.

Uses the hypergeometric distribution to model how nonzero elements are
distributed across tiles of a sparse tensor. When drawing n elements from
a population of N containing r nonzeros, the number of nonzeros in the
sample follows Hypergeometric(N, r, n).
"""

import math

from scipy.stats import hypergeom as _hypergeom


class HypergeometricDensityModel:
    """Models the distribution of nonzero elements in tiles of a sparse tensor.

    Parameters
    ----------
    density : float
        Fraction of nonzero elements (0.0 to 1.0).
    tensor_size : int
        Total number of elements in the tensor.
    """

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
        """P(tile has exactly k nonzeros) -- hypergeometric PMF.

        Parameters
        ----------
        tile_shape : int
            Number of elements in the tile (sample size).
        k : int
            Number of nonzeros to query.
        """
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

    def __repr__(self) -> str:
        return (
            f"HypergeometricDensityModel(density={self.density}, "
            f"N={self.N}, r={self.r})"
        )


def effectual_operations(total_ops: int, *densities: float) -> int:
    """Number of effectual (all-operands-nonzero) operations.

    Simple product model: effectual = round(total * d1 * d2 * ...).
    The 3-state compute classification (Phase 5/6) may produce +/-1 differences.
    """
    result = float(total_ops)
    for d in densities:
        result *= d
    return round(result)
