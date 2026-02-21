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

Returns a tuple of:
  - dict of sparse-specific ActionKey → ActionCount for gated/skipped/
    metadata actions (only emitted when arch YAML declares the action name)
  - dict of per-rank format info keyed by (tensor, level)
  - dict of latency_info for sparse-adjusted latency computation
"""

import math
import re

from accelforge.frontend import arch
from accelforge.frontend.mapping import Temporal, Spatial, TensorHolder as MappingTensorHolder
from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.model._looptree.reuse.symbolic import SymbolicAnalysisOutput
from accelforge.model._looptree.types import Buffet
from accelforge.model.sparse import compute_format_access_counts
from accelforge.model.sparse_formats import compute_format_occupancy
from accelforge.model.sparse_pipeline import (
    apply_format_compression,
    apply_local_saf_reads,
    apply_local_saf_updates,
    compute_saf_probability,
    classify_compute,
    propagate_saf_reduction,
)
from accelforge.util._base_analysis_types import ActionCount, ActionKey


def _has_action(spec: Spec, component_name: str, action_name: str) -> bool:
    """Check if a component declares a specific action name in its arch."""
    component_obj = spec.arch.find(component_name)
    if component_obj is None:
        return False
    actions = component_obj.actions
    # Support both EvalableList[Action] (real) and dict (mock/test)
    if isinstance(actions, dict):
        return action_name in actions
    for a in actions:
        if hasattr(a, "name") and a.name == action_name:
            return True
    return False


def _emit(
    sparse_actions: dict[ActionKey, ActionCount],
    level: str,
    action: str,
    total: int | float,
) -> None:
    """Accumulate a sparse action count into the dict."""
    key = ActionKey(level, action)
    if key not in sparse_actions:
        sparse_actions[key] = ActionCount.default()
    sparse_actions[key].total += total
    sparse_actions[key].max_per_unit += total


def _ranks_have_flattened_ids(rank_format_objs: list) -> bool:
    """Check if any rank has explicit flattened_rank_ids."""
    return any(
        getattr(rf, "flattened_rank_ids", None)
        for rf in rank_format_objs
    )


def _compute_flattened_dimension_sizes(
    rank_format_objs: list,
    shape: dict[str, int],
) -> list[int]:
    """Compute per-rank fiber shapes from explicit flattened_rank_ids.

    For each rank with flattened_rank_ids, fiber_shape = product of the
    sizes of those dimensions in ``shape``. Dimension names are matched
    case-insensitively to rank variable names in ``shape``.

    Ranks without flattened_rank_ids get fiber_shape=1 (degenerate).
    """
    sizes = []
    for rf in rank_format_objs:
        fids = getattr(rf, "flattened_rank_ids", None)
        if fids and len(fids) > 0:
            # Use first flattening group (fids[0])
            dim_names = fids[0]
            size = 1
            for dname in dim_names:
                key = dname.lower()
                size *= shape.get(key, 1)
            sizes.append(max(size, 1))
        else:
            sizes.append(1)
    return sizes


def _get_tensor_rank_variables(einsum, tensor_name: str) -> set[str]:
    """Return the set of rank variables that project to a tensor.

    Inspects ``ta.projection`` for the tensor and extracts all rank
    variable names (lowercased).  For compound expressions like
    ``e + r``, both ``e`` and ``r`` are included.

    Returns empty set if tensor/projection not found.
    """
    ta = None
    for t in einsum.tensor_accesses:
        if t.name == tensor_name:
            ta = t
            break
    if ta is None:
        return set()

    projection = ta.projection
    if not isinstance(projection, dict):
        # List-style projection: each entry is a rank variable name
        if isinstance(projection, (list, tuple)):
            return {str(v).strip().lower() for v in projection}
        return set()

    rank_vars = set()
    for _rank_name, rank_var_expr in projection.items():
        expr_str = str(rank_var_expr).strip()
        # Extract all identifiers from the expression.
        # For simple "m" → {"m"}; for "e + r" → {"e", "r"};
        # for "2*p + r" → {"p", "r"}.
        for token in re.findall(r"[a-zA-Z_]\w*", expr_str):
            rank_vars.add(token.lower())
    return rank_vars


def _compute_flattened_tensor_size(
    rank_format_objs: list,
    full_shape: dict[str, int],
    einsum,
    tensor_name: str,
) -> int:
    """Compute tensor_size from flattened ranks, filtering to tensor dims.

    Only includes dimensions that actually project to ``tensor_name``.
    Ranks whose flattened_rank_ids contain no projecting dimensions
    contribute 1 (degenerate).  This avoids inflating tensor_size with
    dimensions from other tensors in the same loop nest.
    """
    projecting = _get_tensor_rank_variables(einsum, tensor_name)
    tensor_size = 1
    for rf in rank_format_objs:
        fids = getattr(rf, "flattened_rank_ids", None)
        if fids and len(fids) > 0:
            dim_names = fids[0]
            for dname in dim_names:
                key = dname.lower()
                if key in projecting:
                    tensor_size *= full_shape.get(key, 1)
        # Ranks without flattened_rank_ids: skip (don't multiply by 1)
    return max(tensor_size, 1)


def _get_tile_shape_at_level(
    reuse: SymbolicAnalysisOutput,
    job: Job,
    tensor_name: str,
    level_name: str,
) -> dict[str, int]:
    """Reconstruct per-dimension tile sizes at a storage level for a tensor.

    Walks the per-tensor mapping nodes top-to-bottom, starting from
    job.rank_variable_bounds, tracking current_shape at each Temporal/Spatial
    node, and stops at the TensorHolder matching level_name.

    Returns dict[rank_variable, tile_size]. Returns empty dict if the
    tensor has no mapping entry (graceful for mock tests).
    """
    if not hasattr(reuse, "tensor2mapping") or not reuse.tensor2mapping:
        return {}
    if tensor_name not in reuse.tensor2mapping:
        return {}
    if not hasattr(job, "rank_variable_bounds") or not job.rank_variable_bounds:
        return {}

    mapping = reuse.tensor2mapping[tensor_name]
    current_shape = dict(job.rank_variable_bounds)

    for node in mapping.nodes:
        if isinstance(node, (Temporal, Spatial)):
            rv = node.rank_variable
            if isinstance(rv, set):
                # Multi-rank-variable node — skip (shouldn't happen for
                # single-tensor mappings, but handle gracefully)
                continue
            ts = node.tile_shape
            if isinstance(ts, int):
                current_shape[rv] = ts
        elif isinstance(node, MappingTensorHolder):
            if node.component == level_name and tensor_name in node.tensors:
                return current_shape

    return current_shape


def _get_dimension_sizes_for_tensor(
    current_shape: dict[str, int],
    einsum,
    tensor_name: str,
) -> list[int]:
    """Map current_shape to per-tensor dimension sizes using ta.projection.

    Returns list of sizes for non-trivial dimensions (size > 1) in projection
    order (outer-to-inner). The length of this list = num_ranks for format
    expansion.

    Returns empty list if tensor or projection is not found.
    """
    ta = None
    for t in einsum.tensor_accesses:
        if t.name == tensor_name:
            ta = t
            break
    if ta is None:
        return []

    projection = ta.projection
    if not isinstance(projection, dict):
        return []

    sizes = []
    for rank_name, rank_var_expr in projection.items():
        # rank_var_expr is typically a simple variable name like "m"
        # For compound expressions like "m+n", use the full shape product
        rank_var = str(rank_var_expr).strip()
        if rank_var in current_shape:
            size = current_shape[rank_var]
        else:
            # Compound expression — skip this rank or use 1
            size = 1
        if size > 1:
            sizes.append(size)

    # If all dimensions are trivial (size 1), return [1] as minimum
    if not sizes:
        sizes = [1]

    return sizes


def _auto_derive_word_bits(
    primitive: str,
    dim_size: int,
) -> tuple[int | None, int | None]:
    """Auto-derive metadata/payload word bits for a rank primitive.

    Returns (metadata_word_bits, payload_word_bits).
    None means the field is not applicable for this primitive.
    """
    p = primitive.upper()
    if p == "UOP":
        # UOP: payload = ceil(log2(dim_size + 1)), no metadata
        pw = max(1, math.ceil(math.log2(dim_size + 1))) if dim_size > 0 else 1
        return None, pw
    elif p == "B":
        # Bitmask: 1 bit metadata, no payload
        return 1, None
    elif p == "CP":
        # Coordinate Payload: metadata = ceil(log2(dim_size))
        mw = max(1, math.ceil(math.log2(dim_size))) if dim_size > 1 else 1
        return mw, None
    elif p == "RLE":
        # Run-length: metadata = ceil(log2(dim_size))
        mw = max(1, math.ceil(math.log2(dim_size))) if dim_size > 1 else 1
        return mw, None
    return None, None


def apply_sparse_adjustments(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
) -> tuple[dict[ActionKey, ActionCount], dict, dict]:
    """Apply sparse optimizations to reuse analysis results in-place.

    Modifies buffet_stats and compute_stats to reflect format compression,
    storage action filtering (SAF), and compute classification.

    Returns a tuple of:
      - dict of sparse-specific action counts (gated_read, metadata_read,
        gated_compute, skipped_compute, etc.).  Only actions declared in the
        component's arch YAML are emitted.
      - dict of per-rank format info keyed by (tensor, level), containing
        rank_formats, rank_capacity, rank_access_counts, rank_word_bits.
      - dict of latency_info for sparse-adjusted latency computation,
        containing skipped action deltas, metadata actions per component,
        and compute latency ratio.

    No-op (returns empty dicts) when spec.sparse_optimizations has no targets.

    Parameters
    ----------
    reuse : SymbolicAnalysisOutput
        Dense analysis results (modified in-place).
    spec : Spec
        Evaluated spec with sparse_optimizations and arch.
    job : Job
        Job context with einsum info and flattened arch.
    """
    sparse_actions: dict[ActionKey, ActionCount] = {}
    per_rank_info: dict[tuple[str, str], dict] = {}
    latency_info: dict = {
        # Gated read/write deltas per (level, tensor): these are ADDED BACK
        # to post-sparse action counts for latency because gated reads still
        # consume port bandwidth.
        "gated_read_action_deltas": {},
        "gated_write_action_deltas": {},
        # Metadata actions per level (consume BW, added to latency)
        "metadata_read_actions": {},
        "metadata_write_actions": {},
        # Compute latency ratio: post-4b / pre-sparse
        "compute_latency_ratio": 1.0,
    }

    sparse_opts = spec.sparse_optimizations
    if not sparse_opts.targets:
        return sparse_actions, per_rank_info, latency_info

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
        if sparse_opts.get_formats_for(buffet.level, buffet.tensor):
            formatted_buffets.add((buffet.tensor, buffet.level))

    # Save pre-SAF, pre-compression algorithmic counts for per-rank access
    # count computation (needed by compute_format_access_counts).
    pre_saf_child_reads: dict[tuple[str, str], int] = {}
    pre_saf_fills: dict[tuple[str, str], int] = {}
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue
        if (buffet.tensor, buffet.level) not in formatted_buffets:
            continue
        # Save this level's fills (reads to parent)
        pre_saf_fills[(buffet.tensor, buffet.level)] = int(
            stats.total_reads_to_parent
        )
        # Save child's reads (data served from this level to child).
        # If no non-compute child exists, check for a compute-level child
        # (tensor goes directly to the compute unit).
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
        if child_key is None:
            child_key = _get_child_buffet_key(reuse, buffet, set())
        if child_key is not None:
            pre_saf_child_reads[(buffet.tensor, buffet.level)] = int(
                reuse.buffet_stats[child_key].total_reads_to_parent
            )
        else:
            pre_saf_child_reads[(buffet.tensor, buffet.level)] = 0

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
        # Note: compute-level children are NOT compressed here because
        # Phase 4b propagation may apply the same density factor via SAF,
        # leading to double-reduction.  Instead, levels with SAF + format
        # on the same tensor get a post-pipeline action correction (see
        # _apply_format_compression_to_saf_levels).
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
    # Track SAF deltas per (level, tensor) for gated/skipped action emission
    saf_deltas: dict[tuple[str, str], tuple[int, str, float]] = {}
    # Track write deltas for output tensor Z (for latency)
    saf_write_deltas: dict[tuple[str, str], tuple[int, str]] = {}

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        action_opts = sparse_opts.get_action_optimizations_for(buffet.level)
        for opt in action_opts:
            if opt.target != buffet.tensor:
                continue

            # Compute SAF probability from condition_on tensors.
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
            is_output_tensor = tensor_info[buffet.tensor]["is_output"]
            if not is_output_tensor:
                saf_probs_for_compute.append((prob, opt.kind))

            # Apply SAF to the TARGET tensor's child reads
            child_stats = reuse.get_child_buffet_stats(buffet)
            is_output = tensor_info[buffet.tensor]["is_output"]

            if child_stats is not None:
                # For output tensors, subtract first-k reads before SAF
                # (Sparseloop convention: SAF applied to M*N*(K-1), not M*N*K).
                effective_reads = child_stats.total_reads_to_parent
                effective_max = child_stats.max_per_parent_reads_to_parent
                if is_output:
                    effective_reads -= child_stats.total_skipped_first_reads_to_parent
                    effective_max -= child_stats.min_per_parent_skipped_first_reads_to_parent

                # Reduce child's reads from this level
                actual, delta = apply_local_saf_reads(
                    effective_reads,
                    prob,
                    is_read_write=is_output,
                )
                child_stats.total_reads_to_parent = actual

                # Track the delta for gated/skipped read emission
                saf_deltas[(buffet.level, buffet.tensor)] = (delta, opt.kind, prob)

                actual_max, _ = apply_local_saf_reads(
                    effective_max,
                    prob,
                    is_read_write=is_output,
                )
                child_stats.max_per_parent_reads_to_parent = actual_max

                # Clear child skipped_first — already applied to base
                if is_output:
                    child_stats.total_skipped_first_reads_to_parent = 0
                    child_stats.min_per_parent_skipped_first_reads_to_parent = 0

                # For output tensors, reduce child's writeback
                if is_output:
                    actual_w, write_delta = apply_local_saf_updates(
                        child_stats.total_writes_to_parent, prob
                    )
                    child_stats.total_writes_to_parent = actual_w

                    # Track write delta for latency
                    saf_write_deltas[(buffet.level, buffet.tensor)] = (
                        write_delta,
                        opt.kind,
                    )

                    actual_w_max, _ = apply_local_saf_updates(
                        child_stats.max_per_parent_writes_to_parent, prob
                    )
                    child_stats.max_per_parent_writes_to_parent = actual_w_max

    # ========================================================================
    # Emit gated/skipped read actions from SAF deltas
    # ========================================================================
    for (level, tensor), (delta, kind, _prob) in saf_deltas.items():
        if delta <= 0:
            continue
        if kind == "gating":
            action_name = "gated_read"
        elif kind in ("skipping", "position_skipping"):
            action_name = "skipped_read"
        else:
            continue
        if _has_action(spec, level, action_name):
            _emit(sparse_actions, level, action_name, delta)

    # ========================================================================
    # Build gated action deltas for latency (gating only, not skipping)
    # ========================================================================
    # For gating: SAF delta removes reads from action counts, but those gated
    # reads still consume port bandwidth. Track deltas to ADD BACK for latency.
    # For skipping: post-sparse action counts are already correct (no add-back).
    # Keyed by (level, tensor) for per-tensor bandwidth tracking.
    for (level, tensor), (delta, kind, _prob) in saf_deltas.items():
        if delta <= 0 or kind != "gating":
            continue
        component_obj = spec.arch.find(level)
        if component_obj is None or not isinstance(component_obj, arch.TensorHolder):
            continue
        read_bpa = component_obj.actions["read"].bits_per_action
        bpv = tensor_info[tensor]["bits_per_value"]
        bpv_scale = component_obj.bits_per_value_scale
        if hasattr(bpv_scale, '__getitem__') and tensor in bpv_scale:
            bpv = bpv * bpv_scale[tensor]
        read_scale = bpv / read_bpa
        action_delta = delta * read_scale
        lt_key = (level, tensor)
        latency_info["gated_read_action_deltas"].setdefault(lt_key, 0)
        latency_info["gated_read_action_deltas"][lt_key] += action_delta

    for (level, tensor), (write_delta, kind) in saf_write_deltas.items():
        if write_delta <= 0 or kind != "gating":
            continue
        component_obj = spec.arch.find(level)
        if component_obj is None or not isinstance(component_obj, arch.TensorHolder):
            continue
        if isinstance(component_obj, arch.Toll):
            continue
        write_bpa = component_obj.actions["write"].bits_per_action
        bpv = tensor_info[tensor]["bits_per_value"]
        bpv_scale = component_obj.bits_per_value_scale
        if hasattr(bpv_scale, '__getitem__') and tensor in bpv_scale:
            bpv = bpv * bpv_scale[tensor]
        write_scale = bpv / write_bpa
        action_delta = write_delta * write_scale
        lt_key = (level, tensor)
        latency_info["gated_write_action_deltas"].setdefault(lt_key, 0)
        latency_info["gated_write_action_deltas"][lt_key] += action_delta

    # ========================================================================
    # Phase 4b-5: Propagate SAF to compute & classify
    # ========================================================================

    # Save pre-SAF compute totals for gated/skipped compute emission
    pre_saf_compute: dict[str, int] = {}
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre_saf_compute[compute_key.level] = compute_stats.total_ops

    # Propagate SAF reductions to compute operations AND compute-level
    # buffet reads/writes.  When compute is skipped, the MAC-level reads
    # from Reg for ALL operands are reduced proportionally.  For gating,
    # compute-level reads are NOT reduced (gated operations still read).
    for prob, kind in saf_probs_for_compute:
        for compute_key, compute_stats in reuse.compute_stats.items():
            compute_stats.total_ops = propagate_saf_reduction(
                compute_stats.total_ops, prob
            )
            compute_stats.max_per_unit_ops = propagate_saf_reduction(
                compute_stats.max_per_unit_ops, prob
            )

    # For skipping: reduce compute-level buffet element counts using the
    # compound SAF probability.  Use effective_p to avoid double-reducing
    # tensors that already received their own Phase 4a SAF.
    #
    # compound_survival = product(1-p) over all skipping SAFs.  For each
    # compute-level buffet with local Phase 4a prob p_local:
    #   remaining_prob = 1 - compound_survival / (1 - p_local)
    # This correctly handles mutual skipping (A cond B, B cond A) where
    # each tensor's own SAF is a subset of the compound.
    skip_compound_survival = 1.0
    for prob, kind in saf_probs_for_compute:
        if kind in ("skipping", "position_skipping"):
            skip_compound_survival *= (1 - prob)

    if skip_compound_survival < 1.0 - 1e-12:
        for buffet, stats in reuse.buffet_stats.items():
            if buffet.level not in compute_levels:
                continue
            parent_level = None
            for b in reuse.buffet_stats:
                if (b.tensor == buffet.tensor
                        and b.level not in compute_levels):
                    child = reuse.get_child_buffet_stats(b)
                    if child is not None and child is stats:
                        parent_level = b.level
                        break
            # Get local SAF probability from Phase 4a (skipping only).
            local_prob = 0.0
            if parent_level and (parent_level, buffet.tensor) in saf_deltas:
                _, local_kind, p = saf_deltas[(parent_level, buffet.tensor)]
                if local_kind in ("skipping", "position_skipping"):
                    local_prob = p
            if local_prob >= 1.0 - 1e-12:
                continue
            remaining_survival = skip_compound_survival / (1 - local_prob)
            remaining_prob = 1.0 - remaining_survival
            if remaining_prob <= 1e-12:
                continue
            stats.total_reads_to_parent = propagate_saf_reduction(
                stats.total_reads_to_parent, remaining_prob
            )
            stats.max_per_parent_reads_to_parent = propagate_saf_reduction(
                stats.max_per_parent_reads_to_parent, remaining_prob
            )
            stats.total_writes_to_parent = propagate_saf_reduction(
                stats.total_writes_to_parent, remaining_prob
            )
            stats.max_per_parent_writes_to_parent = propagate_saf_reduction(
                stats.max_per_parent_writes_to_parent, remaining_prob
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

            # Check if storage-level SAF at parent levels already covers
            # the condition tensors. If so, the storage SAF has already
            # reduced compute_stats.total_ops in Phase 4b — those gated/
            # skipped iterations never reach the compute unit.
            storage_saf_covers = all(
                any(
                    (level, ct) in saf_deltas
                    for level in {b.level for b in reuse.buffet_stats if b.level not in compute_levels}
                )
                for ct in opt.condition_on
            )

            result = classify_compute(
                pre_saf_compute[compute_key.level],
                operand_densities,
                opt.kind,
            )
            # Only effectual computes contribute to energy
            compute_stats.total_ops = result.random_compute
            compute_stats.max_per_unit_ops = min(
                compute_stats.max_per_unit_ops, result.random_compute
            )

            # Only emit gated/skipped compute when there is NO storage-
            # level SAF covering the same condition.  When storage SAF
            # exists, gated iterations never reach the compute unit.
            if not storage_saf_covers:
                if result.gated_compute > 0 and _has_action(
                    spec, compute_key.level, "gated_compute"
                ):
                    _emit(
                        sparse_actions,
                        compute_key.level,
                        "gated_compute",
                        result.gated_compute,
                    )
                if result.skipped_compute > 0 and _has_action(
                    spec, compute_key.level, "skipped_compute"
                ):
                    _emit(
                        sparse_actions,
                        compute_key.level,
                        "skipped_compute",
                        result.skipped_compute,
                    )

    # Compute latency ratio: post-Phase-5 / pre-SAF
    # After compute classification, total_ops reflects ALL sparsity factors
    # (storage SAF propagation + compute skipping on all condition operands).
    # For skipping, only effectual computes fire; for gating, effectual + gated.
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre = pre_saf_compute.get(compute_key.level, 0)
        if pre > 0:
            latency_info["compute_latency_ratio"] = compute_stats.total_ops / pre
        break

    # ========================================================================
    # Emit metadata actions from format info
    # ========================================================================
    per_rank_info = _emit_metadata_actions(
        sparse_actions,
        latency_info,
        reuse,
        spec,
        job,
        compute_levels,
        formatted_buffets,
        saf_deltas,
        tensor_info,
        pre_saf_child_reads,
        pre_saf_fills,
    )

    # ========================================================================
    # Recompute action counts from modified element counts
    # ========================================================================
    _recompute_action_counts(reuse, spec, job, compute_levels)

    # ========================================================================
    # Post-pipeline: apply format compression to data read actions at levels
    # that have BOTH a compressed format AND an SAF on the same tensor, where
    # the child is at the compute level.
    #
    # Phase 2 doesn't compress compute-level children (to avoid double-
    # counting with Phase 4b propagation of the same tensor's density).
    # But when the level's SAF conditions on a DIFFERENT tensor than the
    # format, the format density is independent of the SAF density and both
    # should apply.  We correct this here by scaling the data read actions
    # by the format density.
    # ========================================================================
    _apply_format_compression_to_saf_levels(
        reuse, spec, compute_levels, formatted_buffets, tensor_info,
    )

    return sparse_actions, per_rank_info, latency_info


def _emit_metadata_actions(
    sparse_actions: dict[ActionKey, ActionCount],
    latency_info: dict,
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    compute_levels: set[str],
    formatted_buffets: set[tuple[str, str]],
    saf_deltas: dict[tuple[str, str], tuple[int, str, float]],
    tensor_info: dict,
    pre_saf_child_reads: dict[tuple[str, str], int],
    pre_saf_fills: dict[tuple[str, str], int],
) -> dict[tuple[str, str], dict]:
    """Emit metadata_read/metadata_write actions with per-rank computation.

    Uses per-rank format decomposition when tile shape info is available
    (real pipeline). Falls back to flat logic when tile info is missing
    (mock tests).

    Also populates latency_info["metadata_read_actions"] and
    latency_info["metadata_write_actions"] with data-word-equivalent
    bandwidth counts (for latency), which differ from the packed physical
    SRAM access counts used for energy.

    Returns per-rank info dict keyed by (tensor, level).
    """
    sparse_opts = spec.sparse_optimizations
    einsum_name = job.einsum_name
    workload = spec.workload
    einsum = workload.einsums[einsum_name]

    per_rank_info: dict[tuple[str, str], dict] = {}

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue
        if (buffet.tensor, buffet.level) not in formatted_buffets:
            continue

        level = buffet.level
        tensor = buffet.tensor

        formats = sparse_opts.get_formats_for(level, tensor)
        if not formats:
            continue
        fmt = formats[0]
        metadata_storage_width = fmt.metadata_storage_width

        # Get the component's read bits_per_action for scaling
        component_obj = spec.arch.find(level)
        if component_obj is None or not isinstance(component_obj, arch.TensorHolder):
            continue

        read_bpa = component_obj.actions["read"].bits_per_action

        # Fall back to the metadata_read action's bits_per_action when the
        # sparse YAML doesn't specify metadata_storage_width.  This captures
        # the physical SRAM width used for metadata packing (e.g. 4-bit for
        # iact_spad/reg, 8-bit for weight_spad in EyerissV2).
        if metadata_storage_width is None:
            try:
                md_action = component_obj.actions["metadata_read"]
                metadata_storage_width = int(md_action.bits_per_action)
            except (KeyError, IndexError):
                pass

        # Get the child buffet to determine post-SAF read counts.
        # If no non-compute child exists, fall back to a compute-level child
        # (tensor goes directly to compute).  Compute-level children are NOT
        # density-compressed by Phase 2, so the reads are raw iteration counts.
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
        child_is_compute = False
        if child_key is None:
            child_key = _get_child_buffet_key(reuse, buffet, set())
            child_is_compute = child_key is not None

        # Post-SAF data reads served from this level to child
        if child_key is not None:
            post_saf_data_reads = reuse.buffet_stats[child_key].total_reads_to_parent
        else:
            post_saf_data_reads = 0

        # Post-compression fills (current state)
        post_fills = stats.total_reads_to_parent

        # ---- Compute per-rank info (informational columns) ----
        current_shape = _get_tile_shape_at_level(reuse, job, tensor, level)

        # Check if explicit ranks with flattened_rank_ids are available
        if fmt.ranks is not None:
            rank_format_objs = fmt.get_rank_formats()
            if _ranks_have_flattened_ids(rank_format_objs):
                dimension_sizes = _compute_flattened_dimension_sizes(
                    rank_format_objs, current_shape
                ) if current_shape else []
            else:
                dimension_sizes = (
                    _get_dimension_sizes_for_tensor(current_shape, einsum, tensor)
                    if current_shape
                    else []
                )
        else:
            rank_format_objs = None  # Will be set below via auto-expansion
            dimension_sizes = (
                _get_dimension_sizes_for_tensor(current_shape, einsum, tensor)
                if current_shape
                else []
            )

        if dimension_sizes and any(d > 1 for d in dimension_sizes):
            density = tensor_info[tensor]["density"]

            # Compute tensor_size and tile_shape
            if fmt.ranks is not None and rank_format_objs is not None and _ranks_have_flattened_ids(rank_format_objs):
                tensor_size = _compute_flattened_tensor_size(
                    rank_format_objs, dict(job.rank_variable_bounds),
                    einsum, tensor,
                )
            else:
                full_shape = _get_dimension_sizes_for_tensor(
                    dict(job.rank_variable_bounds), einsum, tensor
                )
                tensor_size = 1
                for d in (full_shape if full_shape else dimension_sizes):
                    tensor_size *= d
            tile_shape = 1
            for d in dimension_sizes:
                tile_shape *= d

            # Get per-rank format primitives
            if rank_format_objs is None:
                num_ranks = len(dimension_sizes)
                rank_format_objs = fmt.get_rank_formats(num_ranks)
            rank_format_names = [rf.format for rf in rank_format_objs]

            # Compute per-rank occupancy (capacity)
            rank_occs, _ = compute_format_occupancy(
                rank_format_names, dimension_sizes, density, tensor_size
            )

            # Compute per-rank access counts using pre-SAF algorithmic counts
            alg_reads = pre_saf_child_reads.get((tensor, level), 0)
            alg_fills = pre_saf_fills.get((tensor, level), 0)

            rank_access = compute_format_access_counts(
                rank_format_names,
                dimension_sizes,
                density,
                tensor_size,
                tile_shape,
                alg_reads,
                alg_fills,
            )

            # Auto-derive per-rank word bits
            rank_word_bits = []
            for rf_obj, prim, dim_sz in zip(
                rank_format_objs, rank_format_names, dimension_sizes
            ):
                # YAML-specified word bits take precedence
                if rf_obj.metadata_word_bits is not None:
                    md_wb = rf_obj.metadata_word_bits
                elif fmt.metadata_word_bits is not None:
                    md_wb = fmt.metadata_word_bits
                else:
                    md_wb, _ = _auto_derive_word_bits(prim, dim_sz)

                if rf_obj.payload_word_bits is not None:
                    pl_wb = rf_obj.payload_word_bits
                else:
                    _, pl_wb = _auto_derive_word_bits(prim, dim_sz)

                rank_word_bits.append({"metadata": md_wb, "payload": pl_wb})

            # Store per-rank info (informational only)
            per_rank_info[(tensor, level)] = {
                "rank_formats": rank_format_names,
                "rank_capacity": [
                    (occ.metadata_units, occ.payload_units) for occ in rank_occs
                ],
                "rank_access_counts": rank_access,
                "rank_word_bits": rank_word_bits,
            }

        # ---- Emit metadata_read/metadata_write actions (per-rank model) ----
        # Uses compute_format_access_counts to determine per-rank metadata
        # and payload access counts, then sums across ranks in bits and
        # packs into SRAM words.  The per-rank model captures format-
        # specific density effects (bitmask is density-independent per tile,
        # CP scales with ennz), so we pass PRE-COMPRESSION algorithmic
        # counts to avoid double-counting density.
        #
        # For single-element stores (all dims are 1), the per-rank model
        # can't compute meaningful counts, but metadata is still accessed
        # once per data read/fill (1:1 companion).  Emit directly.
        if not (dimension_sizes and any(d > 1 for d in dimension_sizes)):
            # Single-element store: emit metadata as 1:1 with data accesses
            md_word_bits = 0
            if fmt.ranks is not None:
                for rf in fmt.get_rank_formats():
                    if rf.metadata_word_bits:
                        md_word_bits += rf.metadata_word_bits
            if not md_word_bits and fmt.metadata_word_bits:
                md_word_bits = fmt.metadata_word_bits
            if md_word_bits > 0:
                # Data reads/fills after Phase 2 + SAF
                data_reads = int(post_saf_data_reads)
                data_fills = int(stats.total_reads_to_parent)
                md_bpa = read_bpa  # default: pack using data bpa
                if metadata_storage_width and metadata_storage_width > 0:
                    md_bpa = metadata_storage_width
                md_read_actions = math.ceil(data_reads * md_word_bits / md_bpa)
                md_fill_actions = math.ceil(data_fills * md_word_bits / md_bpa)
                if md_read_actions > 0 and _has_action(spec, level, "metadata_read"):
                    _emit(sparse_actions, level, "metadata_read", md_read_actions)
                if md_fill_actions > 0 and _has_action(spec, level, "metadata_write"):
                    _emit(sparse_actions, level, "metadata_write", md_fill_actions)
                # Latency contribution
                bw_read = math.ceil(data_reads * md_word_bits / read_bpa)
                bw_fill = math.ceil(data_fills * md_word_bits / read_bpa)
                latency_info["metadata_read_actions"].setdefault(level, 0)
                latency_info["metadata_read_actions"][level] += bw_read
                latency_info["metadata_write_actions"].setdefault(level, 0)
                latency_info["metadata_write_actions"][level] += bw_fill
            continue

        # Effective algorithmic counts for emission (pre-compression).
        _saf_delta_val, saf_kind, _saf_prob = saf_deltas.get(
            (level, tensor), (0, "", 0.0)
        )
        gated_metadata_input_reads = 0
        if saf_kind == "gating":
            # Gating: actual metadata = post-SAF (effectual iterations only)
            # at full metadata_read rate. Gated metadata = the rest at
            # near-zero gated_metadata_read rate.
            if child_is_compute:
                effective_reads = int(post_saf_data_reads)
            else:
                effective_reads = int(post_saf_data_reads / density) if density > 0 else 0
            gated_metadata_input_reads = (
                pre_saf_child_reads.get((tensor, level), 0) - effective_reads
            )
            if gated_metadata_input_reads < 0:
                gated_metadata_input_reads = 0
        elif saf_kind in ("skipping", "position_skipping"):
            # Skipping: ALL format reads (both effectual and skipped) are
            # charged at the full metadata_read rate.  The format structure
            # must be traversed for all non-format-eliminated iterations.
            # Sparseloop does NOT split metadata energy for skipping.
            if child_is_compute:
                effective_reads = pre_saf_child_reads.get(
                    (tensor, level), int(post_saf_data_reads)
                )
            else:
                effective_reads = (
                    int(post_saf_data_reads / density)
                    if density > 0
                    else 0
                )
        else:
            # No SAF: use full pre-compression count
            effective_reads = pre_saf_child_reads.get((tensor, level), 0)

        effective_fills = pre_saf_fills.get((tensor, level), 0)

        # Metadata storage width for per-element packing
        msw = metadata_storage_width if (metadata_storage_width and metadata_storage_width > 0) else read_bpa

        # Helper: pack format access counts into SRAM words using per-element
        # packing.  Each metadata/payload element is an indivisible unit that
        # must fit within a single SRAM word (Sparseloop model).
        # Packing: floor(msw / word_bits) elements per SRAM access.
        def _pack_format(fac):
            reads, fills = 0, 0
            for i, wbits in enumerate(rank_word_bits):
                for units, wb in [
                    (fac.rank_metadata_reads[i], wbits["metadata"]),
                    (fac.rank_payload_reads[i], wbits["payload"]),
                ]:
                    if units > 0 and wb and wb > 0:
                        elems_per_word = max(1, msw // wb)
                        reads += math.ceil(units / elems_per_word)
                for units, wb in [
                    (fac.rank_metadata_fills[i], wbits["metadata"]),
                    (fac.rank_payload_fills[i], wbits["payload"]),
                ]:
                    if units > 0 and wb and wb > 0:
                        elems_per_word = max(1, msw // wb)
                        fills += math.ceil(units / elems_per_word)
            return reads, fills

        # Helper: compute total format bits (for bandwidth calculation)
        def _sum_format_bits(fac):
            rb, fb = 0, 0
            for i, wbits in enumerate(rank_word_bits):
                md_b = wbits["metadata"] or 0
                pl_b = wbits["payload"] or 0
                rb += fac.rank_metadata_reads[i] * md_b
                rb += fac.rank_payload_reads[i] * pl_b
                fb += fac.rank_metadata_fills[i] * md_b
                fb += fac.rank_payload_fills[i] * pl_b
            return rb, fb

        # Compute format access counts and pack into SRAM words
        emission_access = compute_format_access_counts(
            rank_format_names, dimension_sizes, density, tensor_size,
            tile_shape, effective_reads, effective_fills,
        )
        packed_reads, packed_fills = _pack_format(emission_access)
        total_read_bits, total_fill_bits = _sum_format_bits(emission_access)

        if packed_reads > 0 and _has_action(spec, level, "metadata_read"):
            _emit(sparse_actions, level, "metadata_read", packed_reads)
        if packed_fills > 0 and _has_action(spec, level, "metadata_write"):
            _emit(sparse_actions, level, "metadata_write", packed_fills)

        # Emit GATED metadata at gated_metadata_read rate (for gating SAF)
        if gated_metadata_input_reads > 0 and _has_action(spec, level, "gated_metadata_read"):
            gated_access = compute_format_access_counts(
                rank_format_names, dimension_sizes, density, tensor_size,
                tile_shape, gated_metadata_input_reads, 0,
            )
            gated_packed, _ = _pack_format(gated_access)
            if gated_packed > 0:
                _emit(sparse_actions, level, "gated_metadata_read", gated_packed)

        # Bandwidth-equivalent metadata counts for latency.
        # For gating: full count (actual + gated reads consume BW).
        if saf_kind == "gating":
            full_input_reads = pre_saf_child_reads.get((tensor, level), 0)
            full_access = compute_format_access_counts(
                rank_format_names,
                dimension_sizes,
                density,
                tensor_size,
                tile_shape,
                full_input_reads,
                effective_fills,
            )
            full_read_bits, _ = _sum_format_bits(full_access)
            bw_read = math.ceil(full_read_bits / read_bpa)
        else:
            bw_read = math.ceil(total_read_bits / read_bpa)
        bw_fill = math.ceil(total_fill_bits / read_bpa)
        latency_info["metadata_read_actions"].setdefault(level, 0)
        latency_info["metadata_read_actions"][level] += bw_read
        latency_info["metadata_write_actions"].setdefault(level, 0)
        latency_info["metadata_write_actions"][level] += bw_fill

    return per_rank_info


def _apply_format_compression_to_saf_levels(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    compute_levels: set[str],
    formatted_buffets: set[tuple[str, str]],
    tensor_info: dict[str, dict],
) -> None:
    """Scale data-read actions by format density at levels with SAF + format.

    When a level has a compressed format on tensor T AND an SAF targeting T
    (condition on a different tensor), the format density (d_T) and SAF
    condition density are independent.  Phase 2 doesn't compress compute-
    level children, so the data read actions only reflect the SAF reduction.
    This function applies the missing format density factor.

    Only applies when the child is at the compute level (no intermediate
    storage between this level and the compute unit for this tensor).
    """
    sparse_opts = spec.sparse_optimizations

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue
        if (buffet.tensor, buffet.level) not in formatted_buffets:
            continue

        # Check: does this level have an SAF on this tensor?
        level_has_saf_on_tensor = any(
            opt.target == buffet.tensor
            for opt in sparse_opts.get_action_optimizations_for(buffet.level)
        )
        if not level_has_saf_on_tensor:
            continue

        # Check: is the child at compute level (no non-compute child)?
        non_compute_child = _get_child_buffet_key(
            reuse, buffet, compute_levels
        )
        if non_compute_child is not None:
            continue  # non-compute child exists; Phase 2 already handled it

        # Apply format density to data read actions.
        density = tensor_info[buffet.tensor]["density"]
        stats.total_read_actions = apply_format_compression(
            stats.total_read_actions, density
        )
        stats.max_per_unit_read_actions = apply_format_compression(
            stats.max_per_unit_read_actions, density
        )


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

        # Me -> Parent (upward writeback): skip for output tensors
        is_output_tensor = tensor in einsum.output_tensor_names
        if not is_output_tensor:
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
