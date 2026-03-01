"""Sparse adjustments: format compression, SAF, and compute classification."""

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
    compute_saf_probability,
    classify_compute,
    propagate_saf_reduction,
)
from accelforge.util._base_analysis_types import ActionCount, ActionKey


@dataclass
class LatencyInfo:
    """Parameters for sparse-adjusted latency recomputation."""

    # Gated deltas added back to post-sparse actions (gated reads still consume BW).
    gated_read_action_deltas: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    gated_write_action_deltas: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    metadata_read_actions: dict[str, float] = field(default_factory=dict)
    metadata_write_actions: dict[str, float] = field(default_factory=dict)
    compute_latency_ratio: float = 1.0
    # PE utilization fraction under position-skipping load imbalance (1.0 = no overhead).
    position_space_utilization: float = 1.0


@dataclass
class BuffetActionDelta:
    """Additive delta: sparse_actions = dense_actions + delta."""

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
    """Output from apply_sparse_adjustments: sparse actions, per-rank info,
    latency info, and action deltas for compositional gather_actions.
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


@dataclass
class _PipelineState:
    """Shared state carried between sparse pipeline phases."""

    # Phase 1 outputs (read by all later phases)
    sparse_opts: object
    einsum: object
    tensor_info: dict
    compute_levels: set
    formatted_buffets: set
    dense_compute_ops: dict
    pre_saf_child_reads: dict
    pre_saf_fills: dict
    sparse_actions: dict
    latency_info: LatencyInfo

    # Tile shapes at each (tensor, level), computed from per-tensor mappings.
    tile_shapes: dict = field(default_factory=dict)

    # Phase 3 outputs (read by phases 4, 5)
    saf_probs_for_compute: list = field(default_factory=list)
    saf_deltas: dict = field(default_factory=dict)
    saf_write_deltas: dict = field(default_factory=dict)
    position_skip_info: list = field(default_factory=list)
    position_skip_level: str | None = None
    pre_saf_compute: dict = field(default_factory=dict)


# Action names (must match arch YAML declarations).
GATED_READ = "gated_read"
SKIPPED_READ = "skipped_read"
GATED_COMPUTE = "gated_compute"
SKIPPED_COMPUTE = "skipped_compute"
METADATA_READ = "metadata_read"
METADATA_WRITE = "metadata_write"
GATED_METADATA_READ = "gated_metadata_read"

_SAF_KIND_TO_READ_ACTION = {
    "gating": GATED_READ,
    "skipping": SKIPPED_READ,
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
    """Accumulate a sparse action count. max_per_unit defaults to total."""
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
    """Emit only if total > 0 and arch declares the action. Returns True if emitted."""
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
    """Per-rank fiber shapes from flattened_rank_ids (product of dim sizes, case-insensitive)."""
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
    """Return rank variables (lowercased) that project to this tensor."""
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


def _compute_buffet_tile_shapes(
    reuse: SymbolicAnalysisOutput,
    job: Job,
) -> dict[tuple[str, str], dict[str, int]]:
    """Compute tile shape at each (tensor, level) from per-tensor mappings.

    Walks each per-tensor mapping top-to-bottom, tracking the remaining
    iteration space shape. At each Storage/Toll node for the tensor,
    records the current shape (the tile dimensions the buffer sees).
    """
    tile_shapes: dict[tuple[str, str], dict[str, int]] = {}
    for tensor_name, mapping in reuse.tensor2mapping.items():
        shape = dict(job.rank_variable_bounds)
        for node in mapping.nodes:
            if isinstance(node, (TemporalNode, SpatialNode)):
                rv = str(node.rank_variable) if node.rank_variable else None
                if rv and rv in shape and node.tile_shape is not None:
                    try:
                        shape[rv] = int(node.tile_shape)
                    except (TypeError, ValueError):
                        pass
            elif isinstance(node, (StorageNode, TollNode)):
                tile_shapes[(tensor_name, node.component)] = dict(shape)
    return tile_shapes


def _get_loops_below_level(
    mapping_nodes: list,
    buffet_level: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Collect (spatial_tiles, temporal_tiles) per rank variable below buffet_level."""
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
    """Temporal-only tile product for a condition tensor (used for SAF probability)."""
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
    """Tensor size from flattened ranks, filtered to dims projecting to this tensor."""
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
    """Average PE utilization under position-skipping load imbalance.

    Returns 1.0 if no position-skipping or no spatial loops.
    """
    if not position_skip_tensors or not mapping_nodes:
        return 1.0

    # Build spatial fanout map: rv -> num_instances.
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

        # tile_size = per-PE tile * spatial instances; spatial_factor = product of instances.
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
    """Non-trivial dimension sizes (>1) for this tensor, in projection order."""
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
        # Trivial dims (size 1) excluded — UOP on size-1 produces zero overhead.
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
    """Auto-derive (metadata_word_bits, payload_word_bits) for a rank primitive."""
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
    """Apply format compression to a buffet's element counts in-place."""
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
    """Find child buffet key, falling back to compute-level. Returns (key, is_compute)."""
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
    """Accumulate gated action deltas for latency. Skips Toll for writes."""
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
    """Pack format access counts into SRAM words. Returns (reads, fills)."""
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
    """Apply sparse optimizations (format compression, SAF, compute classification)
    to reuse analysis results in-place. No-op when no sparse targets are configured.
    """
    state = _phase1_init(reuse, spec, job)
    if state is None:
        return SparseAnalysisOutput(sparse_actions={})
    _phase2_format_compression(reuse, state)
    _phase3_saf_application(reuse, spec, job, state)
    _phase4_compute_classification(reuse, spec, job, state)
    per_rank_info, dense_buffet_nets = _phase5_metadata_and_recompute(
        reuse, spec, job, state,
    )

    # Compute action-level deltas (sparse - dense) for compositional path.
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
    for ck, dense in state.dense_compute_ops.items():
        cs = reuse.compute_stats[ck]
        compute_action_deltas[ck] = ComputeActionDelta(
            total_ops=cs.total_ops - dense[0],
            max_per_unit_ops=cs.max_per_unit_ops - dense[1],
        )

    return SparseAnalysisOutput(
        sparse_actions=state.sparse_actions,
        per_rank_info=per_rank_info,
        latency_info=state.latency_info,
        buffet_action_deltas=buffet_action_deltas,
        compute_action_deltas=compute_action_deltas,
    )


def _phase1_init(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
) -> _PipelineState | None:
    """Phase 1: Build tensor info, identify formatted buffets, snapshot dense counts."""
    sparse_opts = spec.effective_sparse_optimizations
    if not sparse_opts.targets:
        return None

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

    # Identify formatted (tensor, level) pairs to avoid double-compression.
    formatted_buffets = set()
    for buffet in reuse.buffet_stats:
        if buffet.level in compute_levels:
            continue
        if buffet.tensor not in tensor_info:
            continue
        if sparse_opts.get_formats_for(buffet.level, buffet.tensor):
            formatted_buffets.add((buffet.tensor, buffet.level))

    # Save pre-SAF algorithmic counts for per-rank access computation.
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

    # Pre-compute tile shapes from per-tensor mappings (replaces stats.tile_shape).
    tile_shapes = _compute_buffet_tile_shapes(reuse, job)

    return _PipelineState(
        sparse_opts=sparse_opts,
        einsum=einsum,
        tensor_info=tensor_info,
        compute_levels=compute_levels,
        formatted_buffets=formatted_buffets,
        dense_compute_ops=dense_compute_ops,
        pre_saf_child_reads=pre_saf_child_reads,
        pre_saf_fills=pre_saf_fills,
        sparse_actions={},
        latency_info=LatencyInfo(),
        tile_shapes=tile_shapes,
    )


def _phase2_format_compression(
    reuse: SymbolicAnalysisOutput,
    state: _PipelineState,
) -> None:
    """Phase 2: Compress element counts at formatted levels by density."""
    for buffet, stats in reuse.buffet_stats.items():
        if (buffet.tensor, buffet.level) not in state.formatted_buffets:
            continue

        tensor = buffet.tensor
        density = state.tensor_info[tensor]["density"]
        is_output = state.tensor_info[tensor]["is_output"]

        # Compress this level's fills, skipped-first, drains, and occupancy
        _compress_buffet_stats(stats, density, is_output, compress_occupancy=True)

        # Compress child reads (data served from this level).
        # Skip if child has its own format. Compute-level children are
        # NOT compressed here — post-pipeline correction applies if both
        # format and SAF exist (see _apply_format_compression_to_saf_levels).
        child_key = _get_child_buffet_key(reuse, buffet, state.compute_levels)
        if child_key is not None:
            child_has_format = (
                child_key.tensor, child_key.level
            ) in state.formatted_buffets
            if not child_has_format:
                child_stats = reuse.buffet_stats[child_key]
                _compress_buffet_stats(child_stats, density, is_output)


def _phase3_saf_application(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    state: _PipelineState,
) -> None:
    """Phase 3: Compute SAF probabilities, apply to reads, emit gated/skipped actions."""
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in state.compute_levels:
            continue

        action_opts = state.sparse_opts.get_action_optimizations_for(buffet.level)
        for opt in action_opts:
            if opt.target != buffet.tensor:
                continue

            # SAF probability from condition_on tensors.
            cond_densities = []
            cond_distributions = []
            cond_tile_shapes = []
            cond_tensor_sizes = []
            for cond_tensor in opt.condition_on:
                if cond_tensor not in state.tensor_info:
                    continue
                cond_densities.append(state.tensor_info[cond_tensor]["density"])
                cond_distributions.append(
                    state.tensor_info[cond_tensor]["density_distribution"]
                )
                # Compute temporal-only tile shape for this cond tensor
                if job.mapping is not None:
                    tile = _compute_cond_temporal_tile(
                        job.mapping.nodes, buffet.level,
                        cond_tensor, state.einsum,
                        state.tile_shapes.get((buffet.tensor, buffet.level)),
                    )
                    # Compute full tensor size from rank_variable_bounds
                    cond_rvs = _get_tensor_rank_variables(
                        state.einsum, cond_tensor,
                    )
                    tsize = 1
                    for rv in cond_rvs:
                        tsize *= job.rank_variable_bounds.get(rv, 1)
                else:
                    tile = 1
                    tsize = 1
                cond_tile_shapes.append(tile)
                cond_tensor_sizes.append(max(tsize, 1))

            # Self-conditioned skipping: collect for position-space utilization.
            if opt.is_self_conditioned and cond_densities:
                target = buffet.tensor
                d = state.tensor_info.get(target, {}).get("density", 1.0)
                if d < 1.0:
                    if (state.position_skip_level is not None
                            and state.position_skip_level != buffet.level):
                        raise ValueError(
                            f"Self-conditioned skipping declared at multiple "
                            f"levels: {state.position_skip_level!r} and "
                            f"{buffet.level!r}. Only one level may use "
                            f"self-conditioned skipping."
                        )
                    state.position_skip_info.append(
                        (target, d, state.tile_shapes.get(
                            (buffet.tensor, buffet.level), {}
                        ))
                    )
                    state.position_skip_level = buffet.level

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
            is_output_tensor = state.tensor_info[buffet.tensor]["is_output"]
            if not is_output_tensor:
                state.saf_probs_for_compute.append((prob, opt.kind))

            # Apply SAF to the TARGET tensor's child reads
            child_stats = reuse.get_child_buffet_stats(buffet)
            is_output = state.tensor_info[buffet.tensor]["is_output"]

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
                state.saf_deltas[(buffet.level, buffet.tensor)] = (delta, opt.kind, prob)

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
                    actual_w, write_delta = apply_local_saf_reads(
                        child_stats.total_writes_to_parent, prob
                    )
                    child_stats.total_writes_to_parent = actual_w

                    # Track write delta for latency
                    state.saf_write_deltas[(buffet.level, buffet.tensor)] = (
                        write_delta,
                        opt.kind,
                    )

                    actual_w_max, _ = apply_local_saf_reads(
                        child_stats.max_per_parent_writes_to_parent, prob
                    )
                    child_stats.max_per_parent_writes_to_parent = actual_w_max

    # Emit gated/skipped read actions from SAF deltas
    for (level, tensor), (delta, kind, _prob) in state.saf_deltas.items():
        action_name = _SAF_KIND_TO_READ_ACTION.get(kind)
        if action_name is not None:
            _emit_if_declared(state.sparse_actions, spec, level, action_name, delta)

    # Build gated action deltas for latency (gated reads still consume BW).
    _accumulate_gated_deltas(
        state.saf_deltas, "read", state.tensor_info, spec, state.latency_info
    )
    _accumulate_gated_deltas(
        state.saf_write_deltas, "write", state.tensor_info, spec, state.latency_info
    )


def _phase4_compute_classification(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    state: _PipelineState,
) -> None:
    """Phase 4: Propagate SAF to compute, classify, compute latency ratio."""
    # Save pre-SAF compute totals for gated/skipped compute emission
    for compute_key, compute_stats in reuse.compute_stats.items():
        state.pre_saf_compute[compute_key.level] = compute_stats.total_ops

    # Propagate SAF reductions to compute operations.
    for prob, kind in state.saf_probs_for_compute:
        for compute_key, compute_stats in reuse.compute_stats.items():
            compute_stats.total_ops = propagate_saf_reduction(
                compute_stats.total_ops, prob
            )
            compute_stats.max_per_unit_ops = propagate_saf_reduction(
                compute_stats.max_per_unit_ops, prob
            )

    # Skipping: reduce compute-level element counts by compound SAF probability.
    skip_compound_survival = 1.0
    for prob, kind in state.saf_probs_for_compute:
        if kind == "skipping":
            skip_compound_survival *= (1 - prob)

    if skip_compound_survival < 1.0 - 1e-12:
        for buffet, stats in reuse.buffet_stats.items():
            if buffet.level not in state.compute_levels:
                continue
            parent_level = None
            for b in reuse.buffet_stats:
                if (b.tensor == buffet.tensor
                        and b.level not in state.compute_levels):
                    child = reuse.get_child_buffet_stats(b)
                    if child is not None and child is stats:
                        parent_level = b.level
                        break
            # Get local SAF probability (skipping only).
            local_prob = 0.0
            if parent_level and (parent_level, buffet.tensor) in state.saf_deltas:
                _, local_kind, p = state.saf_deltas[(parent_level, buffet.tensor)]
                if local_kind == "skipping":
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
        b.level for b in reuse.buffet_stats if b.level not in state.compute_levels
    }

    # Apply compute classification
    for compute_key, compute_stats in reuse.compute_stats.items():
        compute_opts = state.sparse_opts.get_compute_optimizations_for(compute_key.level)
        if not compute_opts:
            continue

        for opt in compute_opts:
            operand_densities = [
                state.tensor_info[t]["density"]
                for t in opt.condition_on
                if t in state.tensor_info
            ]
            if not operand_densities:
                continue

            # has_metadata: True if tensor has compressed format at any level.
            operand_has_metadata = [
                any(
                    (t, level) in state.formatted_buffets
                    for level in all_non_compute_levels
                )
                for t in opt.condition_on
                if t in state.tensor_info
            ]

            # Check if storage-level SAF already covers condition tensors.
            storage_saf_covers = all(
                any(
                    (level, ct) in state.saf_deltas
                    for level in all_non_compute_levels
                )
                for ct in opt.condition_on
            )

            result = classify_compute(
                state.pre_saf_compute[compute_key.level],
                operand_densities,
                opt.kind,
                operand_has_metadata=operand_has_metadata,
            )
            # Only effectual computes contribute to energy
            compute_stats.total_ops = result.random_compute
            compute_stats.max_per_unit_ops = min(
                compute_stats.max_per_unit_ops, result.random_compute
            )
            # Only emit when no storage SAF covers the same condition.
            if not storage_saf_covers:
                _emit_if_declared(
                    state.sparse_actions, spec, compute_key.level,
                    GATED_COMPUTE, result.gated_compute,
                )
                _emit_if_declared(
                    state.sparse_actions, spec, compute_key.level,
                    SKIPPED_COMPUTE, result.skipped_compute,
                )

    # Compute latency ratio: post-classification effectual ops / pre-SAF ops.
    for compute_key, compute_stats in reuse.compute_stats.items():
        pre = state.pre_saf_compute.get(compute_key.level, 0)
        if pre > 0:
            state.latency_info.compute_latency_ratio = compute_stats.total_ops / pre
            break

    # Position-space utilization: load imbalance from position-skipping
    if state.position_skip_info and state.position_skip_level and job.mapping is not None:
        state.latency_info.position_space_utilization = (
            _compute_position_space_utilization(
                state.position_skip_info,
                job.mapping.nodes,
                state.position_skip_level,
                state.einsum,
                job.rank_variable_bounds,
                spec,
            )
        )


def _phase5_metadata_and_recompute(
    reuse: SymbolicAnalysisOutput,
    spec: Spec,
    job: Job,
    state: _PipelineState,
) -> tuple:
    """Phase 5: Emit metadata actions, recompute action counts, post-pipeline correction.

    Returns (per_rank_info, dense_buffet_nets).
    """
    # Emit metadata actions from format info
    per_rank_info = _emit_metadata_actions(
        state.sparse_actions,
        state.latency_info,
        reuse,
        spec,
        job,
        state.compute_levels,
        state.formatted_buffets,
        state.saf_deltas,
        state.tensor_info,
        state.pre_saf_child_reads,
        state.pre_saf_fills,
        tile_shapes=state.tile_shapes,
    )

    # Snapshot dense net actions before recompute.
    dense_buffet_nets: dict[Buffet, tuple] = {}
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in state.compute_levels:
            continue
        dense_buffet_nets[buffet] = (
            stats.net_total_read_actions(),
            stats.net_max_per_unit_read_actions(),
            stats.net_total_write_actions(),
            stats.net_max_per_unit_write_actions(),
        )

    # Recompute action counts from modified element counts.
    _recompute_action_counts(reuse, spec, job, state.compute_levels, state.tensor_info)

    # Post-pipeline: format compression for levels with SAF + format at compute child.
    _apply_format_compression_to_saf_levels(
        reuse, spec, state.compute_levels, state.formatted_buffets, state.tensor_info,
    )

    return per_rank_info, dense_buffet_nets



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
    tile_shapes: dict[tuple[str, str], dict[str, int]] | None = None,
) -> dict[tuple[str, str], dict]:
    """Emit metadata_read/metadata_write actions and populate latency metadata counts.

    Returns per-rank info dict keyed by (tensor, level).
    """
    sparse_opts = spec.effective_sparse_optimizations
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

        # Fall back to metadata_read action's bits_per_action for packing width.
        if metadata_storage_width is None:
            try:
                md_action = component_obj.actions[METADATA_READ]
                metadata_storage_width = int(md_action.bits_per_action)
            except (KeyError, IndexError):
                pass

        # Get child buffet for post-SAF read counts.
        child_key, child_is_compute = _get_child_key_with_fallback(
            reuse, buffet, compute_levels
        )

        # Post-SAF data reads served from this level to child
        if child_key is not None:
            post_saf_data_reads = reuse.buffet_stats[child_key].total_reads_to_parent
        else:
            post_saf_data_reads = 0

        current_shape = (tile_shapes or {}).get(
            (buffet.tensor, buffet.level), {}
        )

        if fmt.has_explicit_ranks():
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
            if fmt.has_explicit_ranks() and rank_format_objs is not None and _ranks_have_flattened_ids(rank_format_objs):
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

        # Emit metadata_read/metadata_write actions.
        # Single-element stores (all dims are 1) emit 1:1 with data accesses.
        if not (dimension_sizes and any(d > 1 for d in dimension_sizes)):
            # Single-element store: emit metadata as 1:1 with data accesses
            md_word_bits = 0
            if fmt.has_explicit_ranks():
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

        _saf_delta_val, saf_kind, _saf_prob = saf_deltas.get(
            (level, tensor), (0, "", 0.0)
        )
        gated_metadata_input_reads = 0
        if saf_kind == "gating":
            # Gating: actual metadata at full rate, gated at reduced rate.
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
        elif saf_kind == "skipping":
            # Skipping: all iterations need metadata traversal (full rate).
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

        # BW-equivalent metadata counts for latency.
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
        elif saf_kind == "skipping" and not child_is_compute:
            # Use post-SAF equivalent for latency BW.
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
    """Apply format density to data-read actions at levels with both SAF and format.

    Only applies when the child is at compute level (format compression
    wasn't applied during the initial pass).
    """
    sparse_opts = spec.effective_sparse_optimizations

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

        # Self-conditioned skipping already captures format density; skip.
        saf_is_self_conditioned = any(
            opt.target == buffet.tensor
            and opt.is_self_conditioned
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
    """Recompute action counts from modified element counts (post-sparse)."""
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

        # Preserve per-unit/total ratio for spatial consistency.
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

        # Parent -> Me (downward fill): folded into regular write actions
        # (matches main's analyze_storage pattern)
        stats.total_write_actions += (
            stats.total_reads_to_parent * write_scale
        )
        stats.total_skipped_first_write_actions += (
            stats.total_skipped_first_reads_to_parent * write_scale
        )

        # Me -> Parent (upward writeback)
        stats.total_read_actions += (
            stats.total_writes_to_parent * read_scale
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
    """Find the child (inner-level) Buffet key for the same tensor, skipping compute."""
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


