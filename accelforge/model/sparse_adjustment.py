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
"""

import math

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
) -> tuple[dict[ActionKey, ActionCount], dict]:
    """Apply sparse optimizations to reuse analysis results in-place.

    Modifies buffet_stats and compute_stats to reflect format compression,
    storage action filtering (SAF), and compute classification.

    Returns a tuple of:
      - dict of sparse-specific action counts (gated_read, metadata_read,
        gated_compute, skipped_compute, etc.).  Only actions declared in the
        component's arch YAML are emitted.
      - dict of per-rank format info keyed by (tensor, level), containing
        rank_formats, rank_capacity, rank_access_counts, rank_word_bits.

    No-op (returns empty dict, empty dict) when spec.sparse_optimizations
    has no targets.

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

    sparse_opts = spec.sparse_optimizations
    if not sparse_opts.targets:
        return sparse_actions, per_rank_info

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
        # Save child's reads (data served from this level to child)
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
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
    saf_deltas: dict[tuple[str, str], tuple[int, str]] = {}

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
                # Reduce child's reads from this level
                actual, delta = apply_local_saf_reads(
                    child_stats.total_reads_to_parent,
                    prob,
                    is_read_write=is_output,
                )
                child_stats.total_reads_to_parent = actual

                # Track the delta for gated/skipped read emission
                saf_deltas[(buffet.level, buffet.tensor)] = (delta, opt.kind)

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
    # Emit gated/skipped read actions from SAF deltas
    # ========================================================================
    for (level, tensor), (delta, kind) in saf_deltas.items():
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
    # Phase 4b-5: Propagate SAF to compute & classify
    # ========================================================================

    # Save pre-SAF compute totals for gated/skipped compute emission
    pre_saf_compute: dict[str, int] = {}
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre_saf_compute[compute_key.level] = compute_stats.total_ops

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

            # Emit gated/skipped compute actions
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

    # ========================================================================
    # Emit metadata actions from format info
    # ========================================================================
    per_rank_info = _emit_metadata_actions(
        sparse_actions,
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

    return sparse_actions, per_rank_info


def _emit_metadata_actions(
    sparse_actions: dict[ActionKey, ActionCount],
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    compute_levels: set[str],
    formatted_buffets: set[tuple[str, str]],
    saf_deltas: dict[tuple[str, str], tuple[int, str]],
    tensor_info: dict,
    pre_saf_child_reads: dict[tuple[str, str], int],
    pre_saf_fills: dict[tuple[str, str], int],
) -> dict[tuple[str, str], dict]:
    """Emit metadata_read/metadata_write actions with per-rank computation.

    Uses per-rank format decomposition when tile shape info is available
    (real pipeline). Falls back to flat logic when tile info is missing
    (mock tests).

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
        fmt_name = (fmt.format or "").lower()

        metadata_storage_width = fmt.metadata_storage_width

        # Get the component's read bits_per_action for scaling
        component_obj = spec.arch.find(level)
        if component_obj is None or not isinstance(component_obj, arch.TensorHolder):
            continue

        read_bpa = component_obj.actions["read"].bits_per_action

        # Get the child buffet to determine post-SAF read counts
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)

        # Post-SAF data reads served from this level to child
        if child_key is not None:
            post_saf_data_reads = reuse.buffet_stats[child_key].total_reads_to_parent
        else:
            post_saf_data_reads = 0

        # Post-compression fills (current state)
        post_fills = stats.total_reads_to_parent

        # SAF delta for this (level, tensor)
        saf_delta, _ = saf_deltas.get((level, tensor), (0, ""))

        # ---- Compute per-rank info (informational columns) ----
        current_shape = _get_tile_shape_at_level(reuse, job, tensor, level)
        dimension_sizes = (
            _get_dimension_sizes_for_tensor(current_shape, einsum, tensor)
            if current_shape
            else []
        )

        if dimension_sizes and any(d > 1 for d in dimension_sizes):
            num_ranks = len(dimension_sizes)
            density = tensor_info[tensor]["density"]
            # Compute tensor_size from full bounds (all tensor dimensions)
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

        # ---- Emit metadata_read/metadata_write actions (flat logic) ----
        # Uses post-SAF/post-compression data read tracking for action keys.
        # Per-rank columns are informational only.
        metadata_word_bits = fmt.metadata_word_bits
        if metadata_word_bits is None:
            if fmt_name == "bitmask":
                metadata_word_bits = 1
            elif fmt_name in ("cp", "csr", "coo"):
                metadata_word_bits = tensor_info[tensor]["bits_per_value"]
            else:
                continue

        if metadata_storage_width is not None and metadata_storage_width > 0:
            words_per_sram = metadata_storage_width // metadata_word_bits
            if words_per_sram < 1:
                words_per_sram = 1
            metadata_read_scale = 1.0 / words_per_sram
        else:
            metadata_read_scale = metadata_word_bits / read_bpa

        if fmt_name == "bitmask":
            actual_metadata_reads = post_saf_data_reads + saf_delta
            if metadata_storage_width is not None:
                actual_metadata_reads = math.ceil(
                    actual_metadata_reads * metadata_read_scale
                )
            if actual_metadata_reads > 0 and _has_action(
                spec, level, "metadata_read"
            ):
                _emit(
                    sparse_actions, level, "metadata_read", actual_metadata_reads
                )

            metadata_writes = post_fills
            if metadata_storage_width is not None:
                metadata_writes = math.ceil(
                    metadata_writes * metadata_read_scale
                )
            if metadata_writes > 0 and _has_action(
                spec, level, "metadata_write"
            ):
                _emit(sparse_actions, level, "metadata_write", metadata_writes)

        elif fmt_name in ("cp", "csr", "coo"):
            actual_metadata_reads = post_saf_data_reads
            if metadata_storage_width is not None:
                actual_metadata_reads = math.ceil(
                    actual_metadata_reads * metadata_read_scale
                )
            if actual_metadata_reads > 0 and _has_action(
                spec, level, "metadata_read"
            ):
                _emit(
                    sparse_actions, level, "metadata_read", actual_metadata_reads
                )

            metadata_writes = post_fills
            if metadata_storage_width is not None:
                metadata_writes = math.ceil(
                    metadata_writes * metadata_read_scale
                )
            if metadata_writes > 0 and _has_action(
                spec, level, "metadata_write"
            ):
                _emit(sparse_actions, level, "metadata_write", metadata_writes)

    return per_rank_info


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
