"""Post-processing sparse adjustments for the AccelForge model pipeline.

Applies sparse optimizations (format compression, SAF, compute classification)
to SymbolicAnalysisOutput after the dense analysis completes. This modifies
buffet_stats and compute_stats in-place before gather_actions/compute_energy.

Returns a SparseAnalysisOutput containing:
  - sparse_actions: gated/skipped/metadata ActionKey → ActionCount
    (only emitted when arch YAML declares the action name)
  - per_rank_info: per-rank format info keyed by (tensor, level)
  - latency_info: parameters for sparse-adjusted latency recomputation
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom as _binom

from accelforge.frontend import arch
from accelforge.frontend.mapping import (
    Spatial as SpatialNode,
    Temporal as TemporalNode,
    Storage as StorageNode,
    Toll as TollNode,
    Compute as ComputeNode,
)

from accelforge.frontend.spec import Spec
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.model._looptree.reuse.symbolic import (
    Compute,
    SymbolicAnalysisOutput,
)
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


@dataclass
class LatencyInfo:
    """Sparse-adjusted latency parameters produced by apply_sparse_adjustments.

    These are consumed by _compute_sparse_latency in run_model.py to recompute
    component latencies after sparsity reduces data transfers and compute.
    """

    # Gated read/write deltas per (level, tensor).  These are ADDED BACK to
    # post-sparse action counts for latency because gated reads still consume
    # port bandwidth.
    gated_read_action_deltas: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    gated_write_action_deltas: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    # Metadata actions per level (consume BW, added to latency).
    metadata_read_actions: dict[str, float] = field(default_factory=dict)
    metadata_write_actions: dict[str, float] = field(default_factory=dict)
    # Compute latency ratio: post-classification effectual ops / pre-sparse ops.
    compute_latency_ratio: float = 1.0
    # Position-space utilization: fraction of spatial instances effectively
    # utilized when position-skipping distributes work unevenly across PEs.
    # 1.0 = no overhead (dense or no position-skipping).
    position_space_utilization: float = 1.0


@dataclass
class BuffetActionDelta:
    """How sparsity changes one buffet's net action counts.

    These are additive deltas: sparse_actions = dense_actions + delta.
    Computed by diffing net action counts before and after
    _recompute_action_counts + _apply_format_compression_to_saf_levels.
    """

    total_read: float = 0
    max_per_unit_read: float = 0
    total_write: float = 0
    max_per_unit_write: float = 0


@dataclass
class ComputeActionDelta:
    """How sparsity changes one compute unit's action counts."""

    total_ops: float = 0
    max_per_unit_ops: float = 0


@dataclass
class SparseAnalysisOutput:
    """Structured output from apply_sparse_adjustments.

    Wraps the three categories of sparse analysis results:
    - sparse_actions: gated/skipped/metadata action counts for energy
    - per_rank_info: per-rank format metadata for reporting
    - latency_info: parameters for sparse-adjusted latency recomputation

    Plus action-level deltas for compositional gather_actions:
    - buffet_action_deltas: per-buffet read/write action deltas
    - compute_action_deltas: per-compute ops deltas
    """

    sparse_actions: dict[ActionKey, ActionCount] = field(default_factory=dict)
    per_rank_info: dict[tuple[str, str], dict] = field(default_factory=dict)
    latency_info: LatencyInfo = field(default_factory=LatencyInfo)
    buffet_action_deltas: dict[Buffet, BuffetActionDelta] = field(
        default_factory=dict
    )
    compute_action_deltas: dict[Compute, ComputeActionDelta] = field(
        default_factory=dict
    )


# Sparse action names.  These must match the action names declared in arch YAML.
# Data I/O actions (modifiers of base read/write):
GATED_READ = "gated_read"
SKIPPED_READ = "skipped_read"
# Compute actions (modifiers of base compute):
GATED_COMPUTE = "gated_compute"
SKIPPED_COMPUTE = "skipped_compute"
# Format metadata I/O actions:
METADATA_READ = "metadata_read"
METADATA_WRITE = "metadata_write"
GATED_METADATA_READ = "gated_metadata_read"

# Map SAF kind → sparse read action name
_SAF_KIND_TO_READ_ACTION = {
    "gating": GATED_READ,
    "skipping": SKIPPED_READ,
    "position_skipping": SKIPPED_READ,
}


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
    max_per_unit: int | float | None = None,
) -> None:
    """Accumulate a sparse action count into the dict.

    ``max_per_unit`` defaults to ``total`` (correct when spatial fanout = 1).
    Callers with spatial context should pass the per-unit value explicitly.
    """
    key = ActionKey(level, action)
    if key not in sparse_actions:
        sparse_actions[key] = ActionCount.default()
    sparse_actions[key].total += total
    sparse_actions[key].max_per_unit += max_per_unit if max_per_unit is not None else total


def _emit_if_declared(
    sparse_actions: dict[ActionKey, ActionCount],
    spec: Spec,
    level: str,
    action_name: str,
    total: int | float,
    max_per_unit: int | float | None = None,
) -> bool:
    """Emit a sparse action only if total > 0 and arch declares it.

    Returns True if the action was emitted.
    """
    if total <= 0:
        return False
    if not _has_action(spec, level, action_name):
        return False
    _emit(sparse_actions, level, action_name, total, max_per_unit=max_per_unit)
    return True


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
    ta = _find_tensor_access(einsum, tensor_name)
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


def _get_loops_below_level(
    mapping_nodes: list,
    buffet_level: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Walk mapping nodes from ``buffet_level`` down to Compute.

    Collects spatial and temporal tile sizes per rank variable.  When
    the same rank variable appears in multiple loops below the level,
    the innermost (last encountered) wins.

    Returns ``(spatial_tiles, temporal_tiles)`` dicts mapping lowercase
    rank variable name → tile size.
    """
    found = False
    spatial_tiles: dict[str, int] = {}
    temporal_tiles: dict[str, int] = {}
    for node in mapping_nodes:
        if not found:
            if isinstance(node, (StorageNode, TollNode)):
                if node.component == buffet_level:
                    found = True
            continue
        if isinstance(node, SpatialNode):
            rv = node.rank_variable
            if isinstance(rv, str):
                spatial_tiles[rv] = int(node.tile_shape)
        elif isinstance(node, TemporalNode):
            rv = node.rank_variable
            if isinstance(rv, str):
                temporal_tiles[rv] = int(node.tile_shape)
        elif isinstance(node, ComputeNode):
            break
    return spatial_tiles, temporal_tiles


def _compute_cond_temporal_tile(
    mapping_nodes: list,
    buffet_level: str,
    cond_tensor_name: str,
    einsum,
    stats_tile_shape: dict[str, int] | None,
) -> int:
    """Compute the temporal-only tile product for a condition tensor.

    For each rank variable projecting onto ``cond_tensor_name``:
    - If a temporal loop exists below ``buffet_level`` → use its tile size
    - Else if a spatial loop exists → use the spatial ``tile_shape``
      (per-PE tile, no temporal subdivision)
    - Else → use the level tile from ``stats_tile_shape``

    Returns the product of per-rank-variable temporal tiles (≥ 1).
    """
    if not stats_tile_shape:
        return 1
    cond_rank_vars = _get_tensor_rank_variables(einsum, cond_tensor_name)
    if not cond_rank_vars:
        return 1
    spatial_tiles, temporal_tiles = _get_loops_below_level(
        mapping_nodes, buffet_level,
    )
    tile = 1
    for rv in cond_rank_vars:
        if rv in temporal_tiles:
            tile *= temporal_tiles[rv]
        elif rv in spatial_tiles:
            tile *= spatial_tiles[rv]
        else:
            tile *= stats_tile_shape.get(rv, 1)
    return max(tile, 1)


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


def _compute_position_space_utilization(
    position_skip_tensors: list[tuple[str, float, dict]],
    mapping_nodes: list,
    level: str,
    einsum,
    rank_variable_bounds: dict[str, int],
    spec,
) -> float:
    """Compute average PE utilization under position-space tiling.

    When position-skipping distributes sparse work across spatial PEs,
    some PEs may get less work than others (load imbalance).  This models
    the position-space decomposition: for each possible
    occupancy of the tile, compute the fraction of spatial instances
    effectively utilized, then take the weighted average.

    For each tensor d with position-skipping:
      tile_d  = product of (spatial + temporal) sizes for d's rank vars
      spatial_d = product of spatial num_instances for d's rank vars
      E[util_d | occ > 0] = weighted average of occ/ceil(occ/spatial_d)/spatial_d

    Overall utilization = product across tensors.

    Returns 1.0 if no position-skipping or no spatial loops.
    """
    if not position_skip_tensors or not mapping_nodes:
        return 1.0

    # Build spatial fanout map: rv -> num_instances from arch + mapping.
    # SpatialNode gives rv -> (component, dimension_name).
    # Arch component's spatial gives dimension_name -> fanout.
    spatial_instances: dict[str, int] = {}
    temporal_tiles: dict[str, int] = {}
    found = False
    for node in mapping_nodes:
        if not found:
            if isinstance(node, (StorageNode, TollNode)):
                if node.component == level:
                    found = True
            continue
        if isinstance(node, SpatialNode):
            rv = node.rank_variable
            if isinstance(rv, str):
                comp_name = node.component
                dim_name = str(node.name)
                # Look up fanout from arch component's spatial definition
                for arch_node in (spec.arch.nodes or []):
                    if getattr(arch_node, 'name', None) == comp_name:
                        for s in (getattr(arch_node, 'spatial', None) or []):
                            if str(s.name) == dim_name:
                                spatial_instances[rv] = int(s.fanout)
                        break
        elif isinstance(node, TemporalNode):
            rv = node.rank_variable
            if isinstance(rv, str):
                temporal_tiles[rv] = int(node.tile_shape)
        elif isinstance(node, ComputeNode):
            break

    per_tensor_util = []
    for tensor_name, density, level_tile_shape in position_skip_tensors:
        # Get rank variables projecting to this tensor
        rvs = _get_tensor_rank_variables(einsum, tensor_name)
        if not rvs:
            continue

        # Compute tile size and spatial factor for this tensor.
        # tile_size = total tile at the buffet level (per-PE tile * spatial instances)
        # spatial_factor = product of spatial instances for tensor's rank vars
        tile_size = 1
        spatial_factor = 1
        for rv in rvs:
            per_pe = int(level_tile_shape.get(rv, 1))
            n_pe = spatial_instances.get(rv, 1)
            t = temporal_tiles.get(rv, 1)
            # Total iterations = per_pe_spatial * n_pe * temporal
            tile_size *= per_pe * n_pe * t
            spatial_factor *= n_pe

        if tile_size <= 0 or spatial_factor <= 1:
            # No spatial parallelism for this tensor
            continue
        if density >= 1.0:
            # Dense tensor — all spatial instances fully utilized
            per_tensor_util.append(1.0)
            continue

        # Compute E[util | occ > 0] using binomial distribution (vectorized)
        occs = np.arange(1, tile_size + 1)
        probs = _binom.pmf(occs, tile_size, density)
        weight_nonzero = probs.sum()
        if weight_nonzero > 0:
            utils = occs / np.ceil(occs / spatial_factor) / spatial_factor
            per_tensor_util.append(float(np.dot(probs, utils) / weight_nonzero))

    if not per_tensor_util:
        return 1.0
    result = 1.0
    for u in per_tensor_util:
        result *= u
    return result


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
    ta = _find_tensor_access(einsum, tensor_name)
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
        # Trivial dimensions (size 1) are excluded: UOP on a size-1 dim
        # produces zero overhead, and format auto-expansion uses
        # num_ranks = len(sizes) to match the count of non-trivial dims.
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


def _find_tensor_access(einsum, tensor_name: str):
    """Find a TensorAccess by name. Returns None if not found."""
    for t in einsum.tensor_accesses:
        if t.name == tensor_name:
            return t
    return None


def _effective_bits_per_value(
    component_obj, tensor: str, tensor_info: dict,
) -> float:
    """Return bits_per_value scaled by the component's bits_per_value_scale."""
    bpv = tensor_info[tensor]["bits_per_value"]
    bpv_scale = component_obj.bits_per_value_scale
    if hasattr(bpv_scale, '__getitem__') and tensor in bpv_scale:
        bpv = bpv * bpv_scale[tensor]
    return bpv


def _compress_buffet_stats(
    stats, density: float, is_output: bool, compress_occupancy: bool = False,
) -> None:
    """Apply format compression to a buffet's element counts in-place.

    Compresses reads-to-parent (fills), skipped-first, and optionally
    writes-to-parent (output tensors) and occupancy.
    """
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
    if is_output:
        stats.total_writes_to_parent = apply_format_compression(
            stats.total_writes_to_parent, density
        )
        stats.max_per_parent_writes_to_parent = apply_format_compression(
            stats.max_per_parent_writes_to_parent, density
        )
    if compress_occupancy:
        stats.max_occupancy = apply_format_compression(
            stats.max_occupancy, density
        )


def _get_child_key_with_fallback(
    reuse: SymbolicAnalysisOutput,
    buffet: Buffet,
    compute_levels: set[str],
) -> tuple[Buffet | None, bool]:
    """Find child buffet key, falling back to compute-level child if needed.

    First tries non-compute children. If none found, falls back to any child
    (including compute-level). Returns (child_key, is_compute_child).
    """
    child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
    if child_key is not None:
        return child_key, False
    child_key = _get_child_buffet_key(reuse, buffet, set())
    return child_key, child_key is not None


def _accumulate_gated_deltas(
    deltas: dict,
    direction: str,
    tensor_info: dict,
    spec: Spec,
    latency_info: LatencyInfo,
) -> None:
    """Accumulate gated action deltas for latency (reads or writes).

    ``direction`` is ``"read"`` or ``"write"``.  For writes, Toll components
    are skipped (no write actions).
    """
    target_dict = (
        latency_info.gated_read_action_deltas
        if direction == "read"
        else latency_info.gated_write_action_deltas
    )
    for (level, tensor), value in deltas.items():
        delta = value[0]
        kind = value[1]
        if delta <= 0 or kind != "gating":
            continue
        component_obj = spec.arch.find(level)
        if component_obj is None or not isinstance(component_obj, arch.TensorHolder):
            continue
        if direction == "write" and isinstance(component_obj, arch.Toll):
            continue
        bpv = _effective_bits_per_value(component_obj, tensor, tensor_info)
        bpa = component_obj.actions[direction].bits_per_action
        action_delta = delta * (bpv / bpa)
        lt_key = (level, tensor)
        target_dict.setdefault(lt_key, 0)
        target_dict[lt_key] += action_delta


def _pack_format(fac, rank_word_bits: list[dict], msw: int) -> tuple[int, int]:
    """Pack format access counts into SRAM words using per-element packing.

    Each metadata/payload element is an indivisible unit that must fit within
    a single SRAM word.  Packing: floor(msw / word_bits)
    elements per SRAM access.
    """
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


def _sum_format_bits(fac, rank_word_bits: list[dict]) -> tuple[int, int]:
    """Compute total format bits across all ranks (for bandwidth calculation)."""
    rb, fb = 0, 0
    for i, wbits in enumerate(rank_word_bits):
        md_b = wbits["metadata"] or 0
        pl_b = wbits["payload"] or 0
        rb += fac.rank_metadata_reads[i] * md_b
        rb += fac.rank_payload_reads[i] * pl_b
        fb += fac.rank_metadata_fills[i] * md_b
        fb += fac.rank_payload_fills[i] * pl_b
    return rb, fb


def apply_sparse_adjustments(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
) -> SparseAnalysisOutput:
    """Apply sparse optimizations to reuse analysis results in-place.

    Modifies buffet_stats and compute_stats to reflect format compression,
    storage action filtering (SAF), and compute classification.

    Returns a SparseAnalysisOutput containing:
      - sparse_actions: gated/skipped/metadata action counts.  Only actions
        declared in the component's arch YAML are emitted.
      - per_rank_info: per-rank format info keyed by (tensor, level).
      - latency_info: parameters for sparse-adjusted latency recomputation.

    No-op (returns empty output) when spec.sparse_optimizations has no targets.

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
    latency_info = LatencyInfo()

    sparse_opts = spec.sparse_optimizations
    if not sparse_opts.targets:
        return SparseAnalysisOutput(sparse_actions=sparse_actions)

    einsum_name = job.einsum_name
    workload = spec.workload
    einsum = workload.einsums[einsum_name]

    # Build tensor info lookup
    tensor_info = {}
    for ta in einsum.tensor_accesses:
        density = ta.density if ta.density is not None else 1.0
        tensor_info[ta.name] = {
            "density": density,
            "density_distribution": ta.density_distribution,
            "is_output": ta.output,
            "bits_per_value": ta.bits_per_value,
        }

    # Compute levels (skip these for buffet processing)
    compute_levels = set(c.level for c in reuse.compute_stats)

    # Snapshot dense compute ops before sparse adjustments modify them.
    dense_compute_ops: dict[Compute, tuple] = {}
    for ck, cs in reuse.compute_stats.items():
        dense_compute_ops[ck] = (cs.total_ops, cs.max_per_unit_ops)

    # Identify which (tensor, level) pairs have compressed formats.
    # Avoids double-compressing child reads when the child level also has
    # a format (child's own compression handles it).
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
        child_key, _ = _get_child_key_with_fallback(
            reuse, buffet, compute_levels
        )
        if child_key is not None:
            pre_saf_child_reads[(buffet.tensor, buffet.level)] = int(
                reuse.buffet_stats[child_key].total_reads_to_parent
            )
        else:
            pre_saf_child_reads[(buffet.tensor, buffet.level)] = 0

    for buffet, stats in reuse.buffet_stats.items():
        if (buffet.tensor, buffet.level) not in formatted_buffets:
            continue

        tensor = buffet.tensor
        density = tensor_info[tensor]["density"]
        is_output = tensor_info[tensor]["is_output"]

        # Compress this level's fills, skipped-first, drains, and occupancy
        _compress_buffet_stats(stats, density, is_output, compress_occupancy=True)

        # Compress child reads (data served from this level).
        # Skip if child has its own format. Compute-level children are
        # NOT compressed here — post-pipeline correction applies if both
        # format and SAF exist (see _apply_format_compression_to_saf_levels).
        child_key = _get_child_buffet_key(reuse, buffet, compute_levels)
        if child_key is not None:
            child_has_format = (
                child_key.tensor, child_key.level
            ) in formatted_buffets
            if not child_has_format:
                child_stats = reuse.buffet_stats[child_key]
                _compress_buffet_stats(child_stats, density, is_output)

    # Collect SAF probabilities for propagation to compute
    saf_probs_for_compute = []  # list of (prob, kind) pairs
    # Track SAF deltas per (level, tensor) for gated/skipped action emission
    saf_deltas: dict[tuple[str, str], tuple[int, str, float]] = {}
    # Track write deltas for output tensor Z (for latency)
    saf_write_deltas: dict[tuple[str, str], tuple[int, str]] = {}
    # Collect position-skipping tensors + level for position-space utilization
    # Each entry: (tensor_name, density, tile_shape_at_level)
    position_skip_info: list[tuple[str, float, dict]] = []
    position_skip_level: str | None = None

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue

        action_opts = sparse_opts.get_action_optimizations_for(buffet.level)
        for opt in action_opts:
            if opt.target != buffet.tensor:
                continue

            # Compute SAF probability from condition_on tensors using
            # temporal-only tile shapes (spatial factors divided out).
            cond_densities = []
            cond_distributions = []
            cond_tile_shapes = []
            cond_tensor_sizes = []
            for cond_tensor in opt.condition_on:
                if cond_tensor not in tensor_info:
                    continue
                cond_densities.append(tensor_info[cond_tensor]["density"])
                cond_distributions.append(
                    tensor_info[cond_tensor]["density_distribution"]
                )
                # Compute temporal-only tile shape for this cond tensor
                if job.mapping is not None:
                    tile = _compute_cond_temporal_tile(
                        job.mapping.nodes, buffet.level,
                        cond_tensor, einsum, stats.tile_shape,
                    )
                    # Compute full tensor size from rank_variable_bounds
                    cond_rvs = _get_tensor_rank_variables(
                        einsum, cond_tensor,
                    )
                    tsize = 1
                    for rv in cond_rvs:
                        tsize *= job.rank_variable_bounds.get(rv, 1)
                else:
                    tile = 1
                    tsize = 1
                cond_tile_shapes.append(tile)
                cond_tensor_sizes.append(max(tsize, 1))

            # Position-skipping with empty condition_on = self-conditioning.
            # The target tensor uses its own format metadata to skip empty
            # positions.  Treat as conditioning on itself.
            if not cond_densities and opt.kind == "position_skipping":
                target = buffet.tensor
                if target in tensor_info:
                    cond_densities = [tensor_info[target]["density"]]
                    cond_distributions = [
                        tensor_info[target]["density_distribution"]
                    ]
                    if job.mapping is not None:
                        tile = _compute_cond_temporal_tile(
                            job.mapping.nodes, buffet.level,
                            target, einsum, stats.tile_shape,
                        )
                        cond_rvs = _get_tensor_rank_variables(
                            einsum, target,
                        )
                        tsize = 1
                        for rv in cond_rvs:
                            tsize *= job.rank_variable_bounds.get(rv, 1)
                    else:
                        tile = 1
                        tsize = 1
                    cond_tile_shapes = [tile]
                    cond_tensor_sizes = [max(tsize, 1)]

                    # Collect for position-space utilization
                    d = tensor_info[target]["density"]
                    if d < 1.0:
                        if (position_skip_level is not None
                                and position_skip_level != buffet.level):
                            raise ValueError(
                                f"position_skipping declared at multiple levels: "
                                f"{position_skip_level!r} and {buffet.level!r}. "
                                f"Only one level may use position_skipping."
                            )
                        position_skip_info.append(
                            (target, d, stats.tile_shape or {})
                        )
                        position_skip_level = buffet.level

            if not cond_densities:
                continue

            prob = compute_saf_probability(
                cond_densities,
                condition_on_tile_shapes=cond_tile_shapes,
                condition_on_tensor_sizes=cond_tensor_sizes,
                condition_on_distributions=cond_distributions,
            )

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
                # For output tensors, subtract first-k reads before SAF.
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

    # Emit gated/skipped read actions from SAF deltas
    for (level, tensor), (delta, kind, _prob) in saf_deltas.items():
        action_name = _SAF_KIND_TO_READ_ACTION.get(kind)
        if action_name is not None:
            _emit_if_declared(sparse_actions, spec, level, action_name, delta)

    # Build gated action deltas for latency. Gated reads still consume
    # port bandwidth — track deltas to add back for latency calculation.
    _accumulate_gated_deltas(
        saf_deltas, "read", tensor_info, spec, latency_info
    )
    _accumulate_gated_deltas(
        saf_write_deltas, "write", tensor_info, spec, latency_info
    )

    # Save pre-SAF compute totals for gated/skipped compute emission
    pre_saf_compute: dict[str, int] = {}
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre_saf_compute[compute_key.level] = compute_stats.total_ops

    # Propagate SAF reductions to compute operations.
    for prob, kind in saf_probs_for_compute:
        for compute_key, compute_stats in reuse.compute_stats.items():
            compute_stats.total_ops = propagate_saf_reduction(
                compute_stats.total_ops, prob
            )
            compute_stats.max_per_unit_ops = propagate_saf_reduction(
                compute_stats.max_per_unit_ops, prob
            )

    # For skipping: reduce compute-level buffet element counts using the
    # compound SAF probability, adjusted per-tensor to avoid double-reducing
    # tensors that already received their own local SAF.
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
            # Get local SAF probability (skipping only).
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

    # Build set of all non-compute levels for has_metadata lookup
    all_non_compute_levels = {
        b.level for b in reuse.buffet_stats if b.level not in compute_levels
    }

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

            # Determine has_metadata for each condition tensor:
            # True if the tensor has a compressed format at any storage level.
            operand_has_metadata = [
                any(
                    (t, level) in formatted_buffets
                    for level in all_non_compute_levels
                )
                for t in opt.condition_on
                if t in tensor_info
            ]

            # Check if storage-level SAF already covers the condition
            # tensors (those iterations never reach the compute unit).
            storage_saf_covers = all(
                any(
                    (level, ct) in saf_deltas
                    for level in all_non_compute_levels
                )
                for ct in opt.condition_on
            )

            result = classify_compute(
                pre_saf_compute[compute_key.level],
                operand_densities,
                opt.kind,
                operand_has_metadata=operand_has_metadata,
            )
            # Only effectual computes contribute to energy
            compute_stats.total_ops = result.random_compute
            compute_stats.max_per_unit_ops = min(
                compute_stats.max_per_unit_ops, result.random_compute
            )
            # Only emit gated/skipped compute when no storage SAF covers
            # the same condition.
            if not storage_saf_covers:
                _emit_if_declared(
                    sparse_actions, spec, compute_key.level,
                    GATED_COMPUTE, result.gated_compute,
                )
                _emit_if_declared(
                    sparse_actions, spec, compute_key.level,
                    SKIPPED_COMPUTE, result.skipped_compute,
                )

    # Compute latency ratio: post-classification effectual ops / pre-SAF ops.
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre = pre_saf_compute.get(compute_key.level, 0)
        if pre > 0:
            latency_info.compute_latency_ratio = compute_stats.total_ops / pre
            break

    # Position-space utilization: load imbalance from position-skipping
    if position_skip_info and position_skip_level and job.mapping is not None:
        latency_info.position_space_utilization = (
            _compute_position_space_utilization(
                position_skip_info,
                job.mapping.nodes,
                position_skip_level,
                einsum,
                job.rank_variable_bounds,
                spec,
            )
        )

    # Emit metadata actions from format info
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

    # Snapshot dense net action counts before recompute overwrites them.
    dense_buffet_nets: dict[Buffet, tuple] = {}
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue
        dense_buffet_nets[buffet] = (
            stats.net_total_read_actions(),
            stats.net_max_per_unit_read_actions(),
            stats.net_total_write_actions(),
            stats.net_max_per_unit_write_actions(),
        )

    # Recompute action counts from modified element counts.
    _recompute_action_counts(reuse, spec, job, compute_levels, tensor_info)

    # Post-pipeline: apply format compression to data read actions at levels
    # with both a compressed format and an SAF on the same tensor where the
    # child is at compute level (format density wasn't applied earlier).
    _apply_format_compression_to_saf_levels(
        reuse, spec, compute_levels, formatted_buffets, tensor_info,
    )

    # ========================================================================
    # Compute action-level deltas (sparse - dense) for compositional path.
    # ========================================================================
    buffet_action_deltas: dict[Buffet, BuffetActionDelta] = {}
    for buffet, dense in dense_buffet_nets.items():
        stats = reuse.buffet_stats[buffet]
        buffet_action_deltas[buffet] = BuffetActionDelta(
            total_read=stats.net_total_read_actions() - dense[0],
            max_per_unit_read=stats.net_max_per_unit_read_actions() - dense[1],
            total_write=stats.net_total_write_actions() - dense[2],
            max_per_unit_write=stats.net_max_per_unit_write_actions() - dense[3],
        )

    compute_action_deltas: dict[Compute, ComputeActionDelta] = {}
    for ck, dense in dense_compute_ops.items():
        cs = reuse.compute_stats[ck]
        compute_action_deltas[ck] = ComputeActionDelta(
            total_ops=cs.total_ops - dense[0],
            max_per_unit_ops=cs.max_per_unit_ops - dense[1],
        )

    return SparseAnalysisOutput(
        sparse_actions=sparse_actions,
        per_rank_info=per_rank_info,
        latency_info=latency_info,
        buffet_action_deltas=buffet_action_deltas,
        compute_action_deltas=compute_action_deltas,
    )


def _emit_metadata_actions(
    sparse_actions: dict[ActionKey, ActionCount],
    latency_info: LatencyInfo,
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

    Also populates latency_info.metadata_read_actions and
    latency_info.metadata_write_actions with data-word-equivalent
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
                md_action = component_obj.actions[METADATA_READ]
                metadata_storage_width = int(md_action.bits_per_action)
            except (KeyError, IndexError):
                pass

        # Get the child buffet to determine post-SAF read counts.
        # Compute-level children are NOT density-compressed by format
        # compression, so the reads are raw iteration counts.
        child_key, child_is_compute = _get_child_key_with_fallback(
            reuse, buffet, compute_levels
        )

        # Post-SAF data reads served from this level to child
        if child_key is not None:
            post_saf_data_reads = reuse.buffet_stats[child_key].total_reads_to_parent
        else:
            post_saf_data_reads = 0

        # ---- Compute per-rank info (informational columns) ----
        current_shape = stats.tile_shape or {}

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
            dist = tensor_info[tensor]["density_distribution"]

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
                rank_format_names, dimension_sizes, density, tensor_size,
                distribution=dist,
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
                distribution=dist,
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
                # Data reads/fills after format compression + SAF
                data_reads = int(post_saf_data_reads)
                data_fills = int(stats.total_reads_to_parent)
                md_bpa = read_bpa  # default: pack using data bpa
                if metadata_storage_width and metadata_storage_width > 0:
                    md_bpa = metadata_storage_width
                md_read_actions = math.ceil(data_reads * md_word_bits / md_bpa)
                md_fill_actions = math.ceil(data_fills * md_word_bits / md_bpa)
                _emit_if_declared(sparse_actions, spec, level, METADATA_READ, md_read_actions)
                _emit_if_declared(sparse_actions, spec, level, METADATA_WRITE, md_fill_actions)
                # Latency contribution
                bw_read = math.ceil(data_reads * md_word_bits / read_bpa)
                bw_fill = math.ceil(data_fills * md_word_bits / read_bpa)
                latency_info.metadata_read_actions.setdefault(level, 0)
                latency_info.metadata_read_actions[level] += bw_read
                latency_info.metadata_write_actions.setdefault(level, 0)
                latency_info.metadata_write_actions[level] += bw_fill
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
                pre_reads = pre_saf_child_reads.get((tensor, level), 0)
                effective_reads = int(pre_reads * (1 - _saf_prob))
            gated_metadata_input_reads = (
                pre_saf_child_reads.get((tensor, level), 0) - effective_reads
            )
            if gated_metadata_input_reads < 0:
                gated_metadata_input_reads = 0
        elif saf_kind in ("skipping", "position_skipping"):
            # Skipping: ALL format reads (both effectual and skipped) are
            # charged at the full metadata_read rate.  The format structure
            # must be traversed for all non-format-eliminated iterations.
            # Metadata energy is NOT split for skipping.
            effective_reads = pre_saf_child_reads.get(
                (tensor, level), 0
            )
        else:
            # No SAF: use full pre-compression count
            effective_reads = pre_saf_child_reads.get((tensor, level), 0)

        effective_fills = pre_saf_fills.get((tensor, level), 0)

        # Metadata storage width for per-element packing
        msw = metadata_storage_width if (metadata_storage_width and metadata_storage_width > 0) else read_bpa

        # Compute format access counts and pack into SRAM words
        emission_access = compute_format_access_counts(
            rank_format_names, dimension_sizes, density, tensor_size,
            tile_shape, effective_reads, effective_fills,
            distribution=dist,
        )
        packed_reads, packed_fills = _pack_format(emission_access, rank_word_bits, msw)
        total_read_bits, total_fill_bits = _sum_format_bits(emission_access, rank_word_bits)

        _emit_if_declared(sparse_actions, spec, level, METADATA_READ, packed_reads)
        _emit_if_declared(sparse_actions, spec, level, METADATA_WRITE, packed_fills)

        # Emit GATED metadata at gated_metadata_read rate (for gating SAF)
        if gated_metadata_input_reads > 0 and _has_action(spec, level, GATED_METADATA_READ):
            gated_access = compute_format_access_counts(
                rank_format_names, dimension_sizes, density, tensor_size,
                tile_shape, gated_metadata_input_reads, 0,
                distribution=dist,
            )
            gated_packed, _ = _pack_format(gated_access, rank_word_bits, msw)
            _emit_if_declared(sparse_actions, spec, level, GATED_METADATA_READ, gated_packed)

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
                distribution=dist,
            )
            full_read_bits, _ = _sum_format_bits(full_access, rank_word_bits)
            bw_read = math.ceil(full_read_bits / read_bpa)
        elif saf_kind in ("skipping", "position_skipping") and not child_is_compute:
            # For latency BW, use post-SAF equivalent count to avoid
            # inflating cycle estimates.  Energy already uses full pre-SAF
            # count above (all iterations need metadata traversal).
            bw_eff = int(post_saf_data_reads / density) if density > 0 else 0
            bw_access = compute_format_access_counts(
                rank_format_names,
                dimension_sizes,
                density,
                tensor_size,
                tile_shape,
                bw_eff,
                effective_fills,
                distribution=dist,
            )
            bw_bits, _ = _sum_format_bits(bw_access, rank_word_bits)
            bw_read = math.ceil(bw_bits / read_bpa)
        else:
            bw_read = math.ceil(total_read_bits / read_bpa)
        bw_fill = math.ceil(total_fill_bits / read_bpa)
        latency_info.metadata_read_actions.setdefault(level, 0)
        latency_info.metadata_read_actions[level] += bw_read
        latency_info.metadata_write_actions.setdefault(level, 0)
        latency_info.metadata_write_actions[level] += bw_fill

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
    condition density are independent.  Format compression doesn't apply to compute-
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

        # Self-conditioned position-skipping: the SAF's local reduction
        # already captures the format density effect (both represent "only
        # nonzero elements are accessed").  Skip format correction to avoid
        # double-counting.
        saf_is_self_conditioned = any(
            opt.target == buffet.tensor
            and opt.kind == "position_skipping"
            and not opt.condition_on
            for opt in sparse_opts.get_action_optimizations_for(buffet.level)
        )
        if saf_is_self_conditioned:
            continue

        # Check: is the child at compute level (no non-compute child)?
        non_compute_child = _get_child_buffet_key(
            reuse, buffet, compute_levels
        )
        if non_compute_child is not None:
            continue  # non-compute child exists; format compression already handled it

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
    tensor_info: dict,
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
        ta = _find_tensor_access(einsum, tensor)
        if ta is None:
            continue

        bits_per_value = _effective_bits_per_value(
            component_obj, tensor, tensor_info,
        )

        read_bpa = component_obj.actions["read"].bits_per_action
        read_scale = bits_per_value / read_bpa

        count_writes = not isinstance(component_obj, arch.Toll)
        if count_writes:
            write_bpa = component_obj.actions["write"].bits_per_action
            write_scale = bits_per_value / write_bpa
        else:
            write_scale = 0

        # Save pre-sparse per-unit/total ratios.  After spatial accumulation,
        # max_per_unit_* stays per-instance while total_* is summed across all
        # instances.  Sparse adjustments scale all instances equally, so this
        # ratio is preserved.  We recompute totals below and then derive
        # per-unit from total * ratio, avoiding the bug where
        # child.max_per_parent_reads_to_parent (a spatial-accumulated total)
        # was incorrectly assigned to max_per_unit_read_actions.
        def _safe_ratio(per_unit, total):
            if total == 0:
                return 1 if per_unit == 0 else 0
            return per_unit / total

        read_pu_frac = _safe_ratio(
            stats.max_per_unit_read_actions, stats.total_read_actions
        )
        write_pu_frac = _safe_ratio(
            stats.max_per_unit_write_actions, stats.total_write_actions
        )
        skip_read_pu_frac = _safe_ratio(
            stats.min_per_unit_skipped_first_read_actions,
            stats.total_skipped_first_read_actions,
        )
        skip_write_pu_frac = _safe_ratio(
            stats.min_per_unit_skipped_first_write_actions,
            stats.total_skipped_first_write_actions,
        )

        # Zero out action counts
        stats.total_write_actions = 0
        stats.max_per_unit_write_actions = 0
        stats.total_read_actions = 0
        stats.max_per_unit_read_actions = 0
        stats.total_skipped_first_write_actions = 0
        stats.min_per_unit_skipped_first_write_actions = 0
        stats.total_skipped_first_read_actions = 0
        stats.min_per_unit_skipped_first_read_actions = 0
        # Also zero parent-derived action counts
        stats.total_parent_fill_write_actions = 0
        stats.max_per_parent_fill_write_actions = 0
        stats.total_skipped_first_parent_fill_write_actions = 0
        stats.min_per_parent_skipped_first_fill_write_actions = 0
        stats.total_parent_drain_read_actions = 0
        stats.max_per_parent_drain_read_actions = 0

        # Parent -> Me (downward fill): use parent-named attributes for
        # correct temporal reuse treatment
        stats.total_parent_fill_write_actions += (
            stats.total_reads_to_parent * write_scale
        )
        stats.max_per_parent_fill_write_actions += (
            stats.max_per_parent_reads_to_parent * write_scale
        )
        stats.total_skipped_first_parent_fill_write_actions += (
            stats.total_skipped_first_reads_to_parent * write_scale
        )
        stats.min_per_parent_skipped_first_fill_write_actions += (
            stats.min_per_parent_skipped_first_reads_to_parent * write_scale
        )

        # Me -> Parent (upward writeback): skip for output tensors
        is_output_tensor = tensor in einsum.output_tensor_names
        if not is_output_tensor:
            stats.total_parent_drain_read_actions += (
                stats.total_writes_to_parent * read_scale
            )
            stats.max_per_parent_drain_read_actions += (
                stats.max_per_parent_writes_to_parent * read_scale
            )

        # Peer exchanges (not modified by sparse, but include for completeness)
        stats.total_read_actions += stats.total_reads_to_peer * read_scale
        stats.total_write_actions += stats.total_reads_to_peer * write_scale

        # Child exchanges — compute total values only.
        # Per-unit values are derived from the saved ratio below.
        child = reuse.get_child_buffet_stats(buffet)
        if child is not None:
            # Me -> Child (downward fill to child): read actions on me
            stats.total_read_actions += (
                child.total_reads_to_parent * read_scale
            )
            stats.total_skipped_first_read_actions += (
                child.total_skipped_first_reads_to_parent * read_scale
            )

            # Child -> Me (upward writeback from child): write actions on me
            stats.total_write_actions += (
                child.total_writes_to_parent * write_scale
            )

        # Restore per-unit values from total using preserved spatial ratio
        stats.max_per_unit_read_actions = (
            stats.total_read_actions * read_pu_frac
        )
        stats.max_per_unit_write_actions = (
            stats.total_write_actions * write_pu_frac
        )
        stats.min_per_unit_skipped_first_read_actions = (
            stats.total_skipped_first_read_actions * skip_read_pu_frac
        )
        stats.min_per_unit_skipped_first_write_actions = (
            stats.total_skipped_first_write_actions * skip_write_pu_frac
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


