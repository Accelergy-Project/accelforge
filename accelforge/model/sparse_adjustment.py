"""Post-processing sparse adjustments for the AccelForge model pipeline.

Applies sparse optimizations (format compression, SAF, compute classification)
to SymbolicAnalysisOutput after the dense analysis completes. This modifies
buffet_stats and compute_stats in-place before gather_actions/compute_energy.

Pipeline ordering (matches Sparseloop):
  Phase 2: Format compression → reduces fills (total_reads_to_parent)
  Phase 3: SAF probability computation
  Phase 4a: Apply SAF to element counts (child reads/writes)
  Phase 4b: Propagate SAF to compute
  Phase 5: Compute classification (3-state ENZ/EZ/NE)
  Final: Recompute action counts from modified element counts
"""

from accelforge.frontend import arch
from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.model._looptree.reuse.symbolic import SymbolicAnalysisOutput
from accelforge.model._looptree.types import Buffet
from accelforge.model.sparse_pipeline import (
    apply_format_compression,
    apply_local_saf_reads,
    apply_local_saf_updates,
    compute_saf_probability,
    classify_compute,
    propagate_saf_reduction,
)


def apply_sparse_adjustments(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
) -> None:
    """Apply sparse optimizations to reuse analysis results in-place.

    Modifies buffet_stats and compute_stats to reflect format compression,
    storage action filtering (SAF), and compute classification.

    No-op when spec.sparse_optimizations has no targets.

    Parameters
    ----------
    reuse : SymbolicAnalysisOutput
        Dense analysis results (modified in-place).
    spec : Spec
        Evaluated spec with sparse_optimizations and arch.
    job : Job
        Job context with einsum info and flattened arch.
    """
    sparse_opts = spec.sparse_optimizations
    if not sparse_opts.targets:
        return

    einsum_name = job.einsum_name
    workload = spec.workload
    einsum = workload.einsums[einsum_name]

    # Build tensor info lookup
    tensor_info = {}
    for ta in einsum.tensor_accesses:
        density = ta.density if ta.density is not None else 1.0
        tensor_info[ta.name] = {
            "density": density,
            "is_output": ta.output,
            "bits_per_value": ta.bits_per_value,
        }

    # Compute levels (skip these for buffet processing)
    compute_levels = set(c.level for c in reuse.compute_stats)

    # ========================================================================
    # Phase 2: Format compression — reduce fills, child reads, occupancy
    # ========================================================================
    # First, identify which (tensor, level) pairs have compressed formats.
    # Needed to avoid double-compressing child reads when the child level
    # also has a format (child's own Phase 2 processing handles it).
    formatted_buffets = set()
    for buffet in reuse.buffet_stats:
        if buffet.level in compute_levels:
            continue
        if buffet.tensor not in tensor_info:
            continue
        density = tensor_info[buffet.tensor]["density"]
        if density >= 1.0:
            continue
        if sparse_opts.get_formats_for(buffet.level, buffet.tensor):
            formatted_buffets.add((buffet.tensor, buffet.level))

    for buffet, stats in reuse.buffet_stats.items():
        if (buffet.tensor, buffet.level) not in formatted_buffets:
            continue

        density = tensor_info[buffet.tensor]["density"]

        # Apply compression to fills (element counts)
        stats.total_reads_to_parent = apply_format_compression(
            stats.total_reads_to_parent, density
        )
        stats.max_per_parent_reads_to_parent = apply_format_compression(
            stats.max_per_parent_reads_to_parent, density
        )
        stats.total_skipped_first_reads_to_parent = apply_format_compression(
            stats.total_skipped_first_reads_to_parent, density
        )
        stats.min_per_parent_skipped_first_reads_to_parent = apply_format_compression(
            stats.min_per_parent_skipped_first_reads_to_parent, density
        )

        # For output tensors, compress drains too
        if tensor_info[buffet.tensor]["is_output"]:
            stats.total_writes_to_parent = apply_format_compression(
                stats.total_writes_to_parent, density
            )
            stats.max_per_parent_writes_to_parent = apply_format_compression(
                stats.max_per_parent_writes_to_parent, density
            )

        # Compress occupancy
        stats.max_occupancy = apply_format_compression(
            stats.max_occupancy, density
        )

        # Also compress child reads (data served from this level).
        # Compressed format means only nonzero elements are stored and
        # served, so child's total_reads_to_parent is reduced.
        # Skip if the child has its own format (already handled above).
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
        if child_key is not None:
            child_has_format = (
                child_key.tensor, child_key.level
            ) in formatted_buffets
            if not child_has_format:
                child_s = reuse.buffet_stats[child_key]
                child_s.total_reads_to_parent = apply_format_compression(
                    child_s.total_reads_to_parent, density
                )
                child_s.max_per_parent_reads_to_parent = apply_format_compression(
                    child_s.max_per_parent_reads_to_parent, density
                )
                child_s.total_skipped_first_reads_to_parent = (
                    apply_format_compression(
                        child_s.total_skipped_first_reads_to_parent, density
                    )
                )
                child_s.min_per_parent_skipped_first_reads_to_parent = (
                    apply_format_compression(
                        child_s.min_per_parent_skipped_first_reads_to_parent,
                        density,
                    )
                )

    # ========================================================================
    # Phase 3-4a: SAF — reduce child reads/writes
    # ========================================================================
    # Collect SAF probabilities for propagation to compute
    saf_probs_for_compute = []  # list of (prob, kind) pairs

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        action_opts = sparse_opts.get_action_optimizations_for(buffet.level)
        for opt in action_opts:
            if opt.target != buffet.tensor:
                continue

            # Compute SAF probability from condition_on tensors.
            # Scalar compute: each access checks one element per condition
            # tensor, so we use the per-element model (tile_shape=1).
            # For tiled compute, this would need the product of shared-
            # dimension tile sizes at the compute level.
            cond_densities = []
            for cond_tensor in opt.condition_on:
                if cond_tensor not in tensor_info:
                    continue
                cond_densities.append(tensor_info[cond_tensor]["density"])

            if not cond_densities:
                continue

            prob = compute_saf_probability(cond_densities)

            if prob <= 0.0:
                continue

            # Record for compute propagation (input tensors only).
            # Output tensor SAFs (e.g., Z gating on [A,B]) reduce the
            # output's own reads/writes but do NOT independently reduce
            # compute — the input tensor SAFs already account for the
            # compute reduction.  Propagating output SAFs would double-
            # count the same condition.
            is_output_tensor = tensor_info[buffet.tensor]["is_output"]
            if not is_output_tensor:
                saf_probs_for_compute.append((prob, opt.kind))

            # Apply SAF to the TARGET tensor's child reads
            child_stats = reuse.get_child_buffet_stats(buffet)
            is_output = tensor_info[buffet.tensor]["is_output"]

            if child_stats is not None:
                # Reduce child's reads from this level
                actual, _ = apply_local_saf_reads(
                    child_stats.total_reads_to_parent,
                    prob,
                    is_read_write=is_output,
                )
                child_stats.total_reads_to_parent = actual

                actual_max, _ = apply_local_saf_reads(
                    child_stats.max_per_parent_reads_to_parent,
                    prob,
                    is_read_write=is_output,
                )
                child_stats.max_per_parent_reads_to_parent = actual_max

                # For output tensors, reduce child's writeback
                if is_output:
                    actual_w, _ = apply_local_saf_updates(
                        child_stats.total_writes_to_parent, prob
                    )
                    child_stats.total_writes_to_parent = actual_w

                    actual_w_max, _ = apply_local_saf_updates(
                        child_stats.max_per_parent_writes_to_parent, prob
                    )
                    child_stats.max_per_parent_writes_to_parent = actual_w_max

    # ========================================================================
    # Phase 4b-5: Propagate SAF to compute & classify
    # ========================================================================

    # Propagate SAF reductions to compute operations
    for prob, _kind in saf_probs_for_compute:
        for compute_key, compute_stats in reuse.compute_stats.items():
            compute_stats.total_ops = propagate_saf_reduction(
                compute_stats.total_ops, prob
            )
            compute_stats.max_per_unit_ops = propagate_saf_reduction(
                compute_stats.max_per_unit_ops, prob
            )

    # Apply compute classification
    for compute_key, compute_stats in reuse.compute_stats.items():
        compute_opts = sparse_opts.get_compute_optimizations_for(compute_key.level)
        if not compute_opts:
            continue

        for opt in compute_opts:
            operand_densities = [
                tensor_info[t]["density"]
                for t in opt.condition_on
                if t in tensor_info
            ]
            if not operand_densities:
                continue

            result = classify_compute(
                compute_stats.total_ops,
                operand_densities,
                opt.kind,
            )
            # Only effectual computes contribute to energy
            compute_stats.total_ops = result.random_compute
            compute_stats.max_per_unit_ops = min(
                compute_stats.max_per_unit_ops, result.random_compute
            )

    # ========================================================================
    # Recompute action counts from modified element counts
    # ========================================================================
    _recompute_action_counts(reuse, spec, job, compute_levels)


def _recompute_action_counts(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    compute_levels: set[str],
) -> None:
    """Zero out and recompute all action counts from modified element counts.

    Mirrors the action count computation in symbolic.py analyze_storage,
    using the same read_scale/write_scale derivation.
    """
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        # Find component object for read/write scale
        component_obj = spec.arch.find(buffet.level)
        if not isinstance(component_obj, arch.TensorHolder):
            continue

        tensor = buffet.tensor
        einsum = spec.workload.einsums[job.einsum_name]
        ta = None
        for t in einsum.tensor_accesses:
            if t.name == tensor:
                ta = t
                break
        if ta is None:
            continue

        # After evaluation, bits_per_value_scale is a dict[str, number]
        bpv_scale = component_obj.bits_per_value_scale  # type: ignore[assignment]
        bits_per_value_scale = bpv_scale[tensor] if tensor in bpv_scale else 1
        bits_per_value = bits_per_value_scale * ta.bits_per_value

        read_bpa = component_obj.actions["read"].bits_per_action
        read_scale = bits_per_value / read_bpa

        count_writes = not isinstance(component_obj, arch.Toll)
        if count_writes:
            write_bpa = component_obj.actions["write"].bits_per_action
            write_scale = bits_per_value / write_bpa
        else:
            write_scale = 0

        # Zero out action counts
        stats.total_write_actions = 0
        stats.max_per_unit_write_actions = 0
        stats.total_read_actions = 0
        stats.max_per_unit_read_actions = 0
        stats.total_skipped_first_write_actions = 0
        stats.min_per_unit_skipped_first_write_actions = 0
        stats.total_skipped_first_read_actions = 0
        stats.min_per_unit_skipped_first_read_actions = 0

        # Parent -> Me (downward fill): write actions on me
        stats.total_write_actions += stats.total_reads_to_parent * write_scale
        stats.max_per_unit_write_actions += (
            stats.total_reads_to_parent * write_scale
        )
        stats.total_skipped_first_write_actions += (
            stats.total_skipped_first_reads_to_parent * write_scale
        )
        stats.min_per_unit_skipped_first_write_actions += (
            stats.min_per_parent_skipped_first_reads_to_parent * write_scale
        )

        # Me -> Parent (upward writeback): read actions on me
        stats.total_read_actions += stats.total_writes_to_parent * read_scale
        stats.max_per_unit_read_actions += (
            stats.total_writes_to_parent * read_scale
        )

        # Peer exchanges (not modified by sparse, but include for completeness)
        stats.total_read_actions += stats.total_reads_to_peer * read_scale
        stats.total_write_actions += stats.total_reads_to_peer * write_scale

        # Child exchanges
        child = reuse.get_child_buffet_stats(buffet)
        if child is not None:
            # Me -> Child (downward fill to child): read actions on me
            stats.total_read_actions += (
                child.total_reads_to_parent * read_scale
            )
            stats.max_per_unit_read_actions += (
                child.max_per_parent_reads_to_parent * read_scale
            )
            stats.total_skipped_first_read_actions += (
                child.total_skipped_first_reads_to_parent * read_scale
            )
            stats.min_per_unit_skipped_first_read_actions += (
                child.min_per_parent_skipped_first_reads_to_parent * read_scale
            )

            # Child -> Me (upward writeback from child): write actions on me
            stats.total_write_actions += (
                child.total_writes_to_parent * write_scale
            )
            stats.max_per_unit_write_actions += (
                child.max_per_parent_writes_to_parent * write_scale
            )


def _get_child_buffet_key(
    reuse: SymbolicAnalysisOutput,
    buffet: Buffet,
    compute_levels: set[str],
) -> Buffet | None:
    """Find the child (inner-level) Buffet key for the same tensor.

    Mirrors get_child_buffet_stats but returns the Buffet key instead of
    stats, and skips compute-level buffets.
    """
    seen = False
    for b in reversed(list(reuse.buffet_stats.keys())):
        if not seen:
            seen = b == buffet
            continue
        if (
            b.tensor == buffet.tensor
            and b.einsum == buffet.einsum
            and b.level not in compute_levels
        ):
            return b
    return None


def _get_tensor_size(einsum, tensor_name: str) -> int:
    """Get the total tensor size (product of projected rank sizes) from the einsum.

    Falls back to 1 if the tensor or rank sizes cannot be determined.
    """
    for ta in einsum.tensor_accesses:
        if ta.name != tensor_name:
            continue
        if not hasattr(einsum, "rank_sizes") or not einsum.rank_sizes:
            return 1
        size = 1
        for rank in ta.projection:
            if rank in einsum.rank_sizes:
                size *= einsum.rank_sizes[rank]
        return max(size, 1)
    return 1
