"""Sparse pipeline: SAF probability, format compression, and compute classification.

Pure-math functions for the sparse adjustment pipeline:
  - Format compression: reduce data accesses by sparsity
  - Local SAF: split random accesses into actual + gated/skipped
  - SAF propagation: outer SAF reduces inner level counts
  - Compute classification: 3-state ENZ/EZ/NE model

These functions take counts and probabilities and return adjusted counts.
Integration with the model pipeline (buffet_stats, compute_stats) happens
in sparse_adjustment.py.
"""

import math
from dataclasses import dataclass

from accelforge.model.density_model import (
    create_density_model,
    effectual_operations,
)


# ---------------------------------------------------------------------------
# SAF probability
# ---------------------------------------------------------------------------


def compute_saf_probability(
    condition_on_densities: list[float],
    condition_on_tile_shapes: list[int] | None = None,
    condition_on_tensor_sizes: list[int] | None = None,
    condition_on_distributions: list[str | None] | None = None,
) -> float:
    """Compute optimization probability for one SAF.

    optimization_prob = 1 - product(P_nonempty_i), where P_nonempty uses
    density for scalar tiles and 1 - prob_empty(tile) for tiled access.
    """
    prob_all_nonempty = 1.0

    for i, density in enumerate(condition_on_densities):
        tile = 1 if condition_on_tile_shapes is None else condition_on_tile_shapes[i]
        tsize = None if condition_on_tensor_sizes is None else condition_on_tensor_sizes[i]
        dist = None if condition_on_distributions is None else condition_on_distributions[i]

        if tile <= 1 or tsize is None or tile >= tsize:
            prob_nonempty = density
        else:
            model = create_density_model(density, tsize, dist)
            prob_nonempty = 1.0 - model.prob_empty(tile)

        prob_all_nonempty *= prob_nonempty

    return 1.0 - prob_all_nonempty


# ---------------------------------------------------------------------------
# Format compression
# ---------------------------------------------------------------------------


def apply_format_compression(
    algorithmic_accesses: int,
    density: float,
) -> int:
    """Reduce data accesses by density: accesses - floor(accesses * sparsity)."""
    if density >= 1.0:
        return algorithmic_accesses
    if density <= 0.0:
        return 0
    sparsity = 1.0 - density
    removed = math.floor(algorithmic_accesses * sparsity)
    return algorithmic_accesses - removed


# ---------------------------------------------------------------------------
# Local SAF
# ---------------------------------------------------------------------------


def apply_local_saf_reads(
    random_reads: int,
    optimization_prob: float,
    is_read_write: bool = False,
) -> tuple[int, int]:
    """Split random reads into (actual, gated/skipped).

    Uses ceil for read-write tensors, floor for read-only.
    """
    if optimization_prob <= 0.0 or random_reads <= 0:
        return (random_reads, 0)
    if is_read_write:
        gated = math.ceil(random_reads * optimization_prob)
    else:
        gated = math.floor(random_reads * optimization_prob)
    actual = random_reads - gated
    return (actual, gated)


# ---------------------------------------------------------------------------
# SAF propagation
# ---------------------------------------------------------------------------


def propagate_saf_reduction(
    count: int,
    optimization_prob: float,
) -> int:
    """Reduce count by SAF probability: count - floor(count * prob)."""
    if optimization_prob <= 0.0 or count <= 0:
        return count
    removed = math.floor(count * optimization_prob)
    return count - removed


def compute_nested_saf_effective_prob(
    local_prob: float,
    outer_prob: float,
) -> float:
    """Adjust local SAF prob for outer filtering: 1 - (1-local)/(1-outer)."""
    if outer_prob >= 1.0:
        return 0.0
    return 1.0 - (1.0 - local_prob) / (1.0 - outer_prob)


# ---------------------------------------------------------------------------
# Compute classification (9-state model)
# ---------------------------------------------------------------------------


def _round6(x: float) -> float:
    """Round to 6 decimal places for numerical stability."""
    return round(x * 1_000_000) / 1_000_000


@dataclass
class OperandStates:
    """Per-operand 3-state probabilities.

    ENZ: exist, nonzero (density)
    EZ:  exist, zero (only when no metadata — dense format)
    NE:  not exist (only when has metadata — compressed format)
    """

    p_enz: float
    p_ez: float
    p_ne: float


def compute_operand_states(density: float, has_metadata: bool) -> OperandStates:
    """Compute per-operand state probabilities.

    With metadata (compressed format): hardware can distinguish present/absent
    elements, so absent elements are NE (not exist).
    Without metadata: all elements exist (either nonzero ENZ or zero EZ).

    """
    d = _round6(density)
    if has_metadata:
        return OperandStates(p_enz=d, p_ez=0.0, p_ne=1.0 - d)
    else:
        return OperandStates(p_enz=d, p_ez=1.0 - d, p_ne=0.0)


@dataclass
class ComputeClassification:
    """Compute classification result from 9-state model.

    ENZ (effectual nonzero) -> random_compute: always executed
    EZ  (effectual zero)    -> gated_compute:  executed but output gated
    NE  (not executed)      -> skipped_compute: not executed (skipping)
    NE×NE                   -> nonexistent_compute: both operands absent
    """

    random_compute: int
    """Effectual computes (always executed, both operands nonzero)."""

    gated_compute: int
    """Ineffectual computes with gating (executed but output discarded)."""

    skipped_compute: int
    """Ineffectual computes with skipping (not executed, zero energy)."""

    nonexistent_compute: int = 0
    """Computes where both operands are absent (NE,NE) — never executed."""

    @property
    def total(self) -> int:
        return (self.random_compute + self.gated_compute
                + self.skipped_compute + self.nonexistent_compute)


def classify_compute(
    total_computes: int,
    operand_densities: list[float],
    compute_optimization_kind: str | None = None,
    operand_has_metadata: list[bool] | None = None,
) -> ComputeClassification:
    """Classify computes into random/gated/skipped/nonexistent using the
    9-state ENZ/EZ/NE joint probability model.
    """
    if not compute_optimization_kind:
        return ComputeClassification(
            random_compute=total_computes,
            gated_compute=0,
            skipped_compute=0,
            nonexistent_compute=0,
        )

    if len(operand_densities) < 2:
        # Single-operand: use simple product model (backward compat)
        random = effectual_operations(total_computes, *operand_densities)
        ineffectual = total_computes - random
        if compute_optimization_kind == "gating":
            return ComputeClassification(
                random_compute=random, gated_compute=ineffectual,
                skipped_compute=0, nonexistent_compute=0,
            )
        elif compute_optimization_kind == "skipping":
            return ComputeClassification(
                random_compute=random, gated_compute=0,
                skipped_compute=ineffectual, nonexistent_compute=0,
            )
        else:
            raise ValueError(
                f"Unknown compute optimization kind: {compute_optimization_kind!r}. "
                f"Expected 'gating' or 'skipping'."
            )

    if operand_has_metadata is None:
        operand_has_metadata = [False, False]

    # Per-operand state probabilities
    s0 = compute_operand_states(operand_densities[0], operand_has_metadata[0])
    s1 = compute_operand_states(operand_densities[1], operand_has_metadata[1])

    # 9 joint probabilities
    # (ENZ,ENZ), (ENZ,EZ), (ENZ,NE), (EZ,ENZ), (EZ,EZ), (EZ,NE),
    # (NE,ENZ), (NE,EZ), (NE,NE)
    p_enz_enz = s0.p_enz * s1.p_enz
    p_enz_ez = s0.p_enz * s1.p_ez
    p_enz_ne = s0.p_enz * s1.p_ne
    p_ez_enz = s0.p_ez * s1.p_enz
    p_ez_ez = s0.p_ez * s1.p_ez
    p_ez_ne = s0.p_ez * s1.p_ne
    p_ne_enz = s0.p_ne * s1.p_enz
    p_ne_ez = s0.p_ne * s1.p_ez
    p_ne_ne = s0.p_ne * s1.p_ne

    # Map to compute categories based on optimization kind:
    #   (ENZ,ENZ) → always random
    #   (ENZ,EZ)/(EZ,ENZ) → gated if gate, random if skip
    #   (ENZ,NE)/(NE,ENZ) → gated if gate, skipped if skip
    #   (EZ,EZ) → gated if gate, random if skip
    #   (EZ,NE)/(NE,EZ) → gated if gate, skipped if skip
    #   (NE,NE) → nonexistent always

    is_gating = compute_optimization_kind == "gating"
    is_skipping = compute_optimization_kind == "skipping"

    if not is_gating and not is_skipping:
        raise ValueError(
            f"Unknown compute optimization kind: {compute_optimization_kind!r}. "
            f"Expected 'gating' or 'skipping'."
        )

    p_random = p_enz_enz
    p_nonexistent = p_ne_ne

    if is_gating:
        # Gating: everything except ENZ×ENZ and NE×NE is gated
        p_gated = (p_enz_ez + p_ez_enz + p_enz_ne + p_ne_enz
                   + p_ez_ez + p_ez_ne + p_ne_ez)
        p_skipped = 0.0
    else:  # skipping
        # Skipping: NE combinations (except NE×NE) are skipped; EZ are random
        p_skipped = p_enz_ne + p_ne_enz + p_ez_ne + p_ne_ez
        p_random += p_enz_ez + p_ez_enz + p_ez_ez
        p_gated = 0.0

    # Pessimistic floor rounding
    skipped_float = total_computes * p_skipped
    gated_float = total_computes * p_gated
    nonexistent_float = total_computes * p_nonexistent

    skipped = math.floor(skipped_float)
    gated = math.floor(gated_float)
    nonexistent = math.floor(nonexistent_float)
    random = total_computes - skipped - gated - nonexistent

    return ComputeClassification(
        random_compute=random,
        gated_compute=gated,
        skipped_compute=skipped,
        nonexistent_compute=nonexistent,
    )
