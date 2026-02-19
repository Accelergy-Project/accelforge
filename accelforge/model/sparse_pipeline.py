"""Sparse pipeline: SAF probability, format compression, and compute classification.

Implements the sparse adjustment pipeline phases:
  Phase 4a: Format compression -- reduce data accesses by sparsity
  Phase 4b: Local SAF -- split random accesses into actual + gated/skipped
  Phase 4b: SAF propagation -- outer SAF reduces inner level counts
  Phase 5:  Compute classification -- 3-state ENZ/EZ/NE

Functions here are pure math -- they take counts and probabilities and return
adjusted counts. Integration with the model pipeline (run_model.py) happens
in later phases.
"""

import math
from dataclasses import dataclass

from accelforge.model.density_model import (
    HypergeometricDensityModel,
    effectual_operations,
)


# ---------------------------------------------------------------------------
# SAF probability
# ---------------------------------------------------------------------------


def compute_saf_probability(
    condition_on_densities: list[float],
    condition_on_tile_shapes: list[int] | None = None,
    condition_on_tensor_sizes: list[int] | None = None,
) -> float:
    """Compute optimization probability for one SAF.

    For each condition_on tensor, computes P(tile nonempty):
    - Scalar (tile=1) or tile >= tensor_size: P(nonempty) = density
    - Tiled (1 < tile < tensor_size): 1 - hypergeometric P(tile empty)

    optimization_prob = 1 - product(P_nonempty_i)

    For scalar with single condition: prob = 1 - d
    For scalar with multiple conditions: prob = 1 - product(d_i)

    Parameters
    ----------
    condition_on_densities : list[float]
        Densities of condition_on tensors.
    condition_on_tile_shapes : list[int], optional
        Tile shapes per condition_on tensor. None = all scalar.
    condition_on_tensor_sizes : list[int], optional
        Full tensor sizes per condition_on tensor. Required when tile > 1.
    """
    prob_all_nonempty = 1.0

    for i, density in enumerate(condition_on_densities):
        tile = 1 if condition_on_tile_shapes is None else condition_on_tile_shapes[i]
        tsize = None if condition_on_tensor_sizes is None else condition_on_tensor_sizes[i]

        if tile <= 1 or tsize is None or tile >= tsize:
            prob_nonempty = density
        else:
            model = HypergeometricDensityModel(density, tsize)
            prob_nonempty = 1.0 - model.prob_empty(tile)

        prob_all_nonempty *= prob_nonempty

    return 1.0 - prob_all_nonempty


# ---------------------------------------------------------------------------
# Phase 4a: Format compression
# ---------------------------------------------------------------------------


def apply_format_compression(
    algorithmic_accesses: int,
    density: float,
) -> int:
    """Reduce data accesses by compressed format sparsity.

    With a compressed format, zero-valued elements are not stored, so
    accesses to zero positions are eliminated.

    random = accesses - floor(accesses * (1 - density))

    Parameters
    ----------
    algorithmic_accesses : int
        Pre-compression access count.
    density : float
        Tensor density (0.0 to 1.0).

    Returns
    -------
    int
        Post-compression random accesses.
    """
    if density >= 1.0:
        return algorithmic_accesses
    if density <= 0.0:
        return 0
    sparsity = 1.0 - density
    removed = math.floor(algorithmic_accesses * sparsity)
    return algorithmic_accesses - removed


# ---------------------------------------------------------------------------
# Phase 4b: Local SAF
# ---------------------------------------------------------------------------


def apply_local_saf_reads(
    random_reads: int,
    optimization_prob: float,
    is_read_write: bool = False,
) -> tuple[int, int]:
    """Split random reads into actual + gated/skipped.

    Read-only tensor: gated = floor(random * prob)
    Read-write tensor: gated = ceil(random * prob)
    actual = random - gated

    Fills are NEVER reduced by local SAF (handled separately).

    Parameters
    ----------
    random_reads : int
        Post-compression random read count.
    optimization_prob : float
        SAF optimization probability.
    is_read_write : bool
        True if tensor is read-write (output accumulator).

    Returns
    -------
    tuple[int, int]
        (actual_reads, gated_or_skipped_reads)
    """
    if optimization_prob <= 0.0 or random_reads <= 0:
        return (random_reads, 0)
    if is_read_write:
        gated = math.ceil(random_reads * optimization_prob)
    else:
        gated = math.floor(random_reads * optimization_prob)
    actual = random_reads - gated
    return (actual, gated)


def apply_local_saf_updates(
    random_updates: int,
    optimization_prob: float,
) -> tuple[int, int]:
    """Split random updates into actual + gated/skipped.

    Updates always use floor rounding (same as read-only reads).

    Parameters
    ----------
    random_updates : int
        Post-compression random update count.
    optimization_prob : float
        SAF optimization probability.

    Returns
    -------
    tuple[int, int]
        (actual_updates, gated_or_skipped_updates)
    """
    if optimization_prob <= 0.0 or random_updates <= 0:
        return (random_updates, 0)
    gated = math.floor(random_updates * optimization_prob)
    actual = random_updates - gated
    return (actual, gated)


# ---------------------------------------------------------------------------
# Phase 4b: SAF propagation
# ---------------------------------------------------------------------------


def propagate_saf_reduction(
    count: int,
    optimization_prob: float,
) -> int:
    """Propagate SAF reduction from an outer level to inner levels.

    Outer SAF reduces the maximum counts seen by inner levels:
    remaining = count - floor(count * prob)

    Parameters
    ----------
    count : int
        Current count at the inner level.
    optimization_prob : float
        SAF probability from the outer level.

    Returns
    -------
    int
        Reduced count after propagation.
    """
    if optimization_prob <= 0.0 or count <= 0:
        return count
    removed = math.floor(count * optimization_prob)
    return count - removed


def compute_nested_saf_effective_prob(
    local_prob: float,
    outer_prob: float,
) -> float:
    """Compute effective probability for nested SAFs.

    When an outer level already filters with probability outer_prob,
    the inner level only handles what passed through. The effective
    local probability is adjusted:

    effective_p = 1 - (1 - local_p) / (1 - outer_p)

    Parameters
    ----------
    local_prob : float
        Local SAF probability at the inner level.
    outer_prob : float
        SAF probability from the outer level.

    Returns
    -------
    float
        Effective probability for the inner level.
    """
    if outer_prob >= 1.0:
        return 0.0
    return 1.0 - (1.0 - local_prob) / (1.0 - outer_prob)


# ---------------------------------------------------------------------------
# Phase 5: Compute classification
# ---------------------------------------------------------------------------


@dataclass
class ComputeClassification:
    """3-state compute classification result.

    ENZ (effectual nonzero) -> random_compute: always executed
    EZ  (effectual zero)    -> gated_compute:  executed but output gated
    NE  (not executed)      -> skipped_compute: not executed (skipping)
    """

    random_compute: int
    """Effectual computes (always executed, both operands nonzero)."""

    gated_compute: int
    """Ineffectual computes with gating (executed but output discarded)."""

    skipped_compute: int
    """Ineffectual computes with skipping (not executed, zero energy)."""

    @property
    def total(self) -> int:
        return self.random_compute + self.gated_compute + self.skipped_compute


def classify_compute(
    total_computes: int,
    operand_densities: list[float],
    compute_optimization_kind: str | None = None,
) -> ComputeClassification:
    """Classify computes into random/gated/skipped (3-state model).

    Without compute optimization: all computes are random (executed normally).
    With gating: ineffectual computes are gated (reduced energy).
    With skipping: ineffectual computes are skipped (zero energy).

    Parameters
    ----------
    total_computes : int
        Total algorithmic computes.
    operand_densities : list[float]
        Densities of operands involved in compute.
    compute_optimization_kind : str, optional
        "gating" or "skipping". None means no compute optimization.

    Returns
    -------
    ComputeClassification
        Classified compute counts.
    """
    if not compute_optimization_kind:
        return ComputeClassification(
            random_compute=total_computes,
            gated_compute=0,
            skipped_compute=0,
        )

    random = effectual_operations(total_computes, *operand_densities)
    ineffectual = total_computes - random

    if compute_optimization_kind == "gating":
        return ComputeClassification(
            random_compute=random,
            gated_compute=ineffectual,
            skipped_compute=0,
        )
    elif compute_optimization_kind == "skipping":
        return ComputeClassification(
            random_compute=random,
            gated_compute=0,
            skipped_compute=ineffectual,
        )
    else:
        raise ValueError(
            f"Unknown compute optimization kind: {compute_optimization_kind!r}. "
            f"Expected 'gating' or 'skipping'."
        )
