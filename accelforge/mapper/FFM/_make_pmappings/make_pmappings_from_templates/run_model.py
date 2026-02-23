import logging
from collections import defaultdict

from sympy import Symbol
import accelforge.frontend.arch as arch
from accelforge.frontend.mapping import TensorHolder
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.model._looptree.reuse import symbolic
from accelforge.model._looptree.reuse.symbolic import (
    analyze_reuse_and_add_reservations_to_mapping,
)
from accelforge.model._looptree.energy import (
    compute_energy_from_actions,
    gather_actions,
    gather_actions_with_sparse,
)
from accelforge.model._looptree.latency.memory import component_latency
from accelforge.model.sparse_adjustment import apply_sparse_adjustments, LatencyInfo
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    nameloop2col,
    tensor2col,
    firstlatency2col,
    action2col,
    energy2col,
)
from accelforge.frontend.mapper.metrics import Metrics
import sympy
from numbers import Number
from accelforge.util._eval_expressions import MATH_FUNCS, eval_expression
from accelforge.util._sympy.broadcast_max import Max, MaxGeqZero


def run_model(
    job: Job,
    add_reservations: bool = True,
) -> tuple[list[Symbol], dict[str, float], dict[str, float], dict[str, float]]:
    pmapping = job.mapping
    spec = job.spec
    metrics = job.metrics
    is_copy_op = job.is_copy_operation
    workload = spec.workload

    df = {}

    reuse = analyze_reuse_and_add_reservations_to_mapping(
        job, add_reservations=add_reservations
    )

    # Phase 1: Dense latency (before sparse adjustments)
    latency = component_latency(reuse, job.flattened_arch, pmapping, spec)
    try:
        overall_latency = MaxGeqZero(*latency.values())
    except Exception as e:
        for k, v in latency.items():
            if not isinstance(v, (Number, sympy.Symbol, sympy.Expr)):
                raise ValueError(
                    f"Invalid type for latency: {k}: {type(v)} {str(v).strip()}"
                )

        raise ValueError(
            f"Error calculating latency for {job.einsum_name}. Could not calculate "
            f"a symbolic max of the following latencies:\n\t"
            + "\n\t".join(
                [f"{k}: {type(v)} {str(v).strip()}" for k, v in latency.items()]
            )
        )

    # Capture dense actions BEFORE sparse mutation.  Used by
    # gather_actions_with_sparse to compose sparse-adjusted actions
    # without reading from the mutated reuse.
    dense_actions = gather_actions(reuse, None, use_name=True)
    if metrics & Metrics.ACTIONS:
        dense_detailed_actions = gather_actions(
            reuse, None, verbose=True, use_name=True,
        )

    sparse_result = apply_sparse_adjustments(reuse, spec, job)
    per_rank_info = sparse_result.per_rank_info
    latency_info = sparse_result.latency_info

    # Recompute latency AFTER sparse adjustments using per-tensor
    # post-sparse action counts + gated deltas added back + metadata.
    # NOTE: _compute_sparse_latency still reads mutated buffet_stats
    # (max_per_unit_read_actions, max_per_unit_write_actions).
    has_sparse_latency = (
        latency_info.gated_read_action_deltas
        or latency_info.metadata_read_actions
        or latency_info.metadata_write_actions
        or latency_info.compute_latency_ratio != 1.0
        or latency_info.position_space_utilization < 1.0
    )
    if has_sparse_latency:
        latency = _compute_sparse_latency(
            reuse, latency_info, job.flattened_arch, spec
        )
        try:
            overall_latency = MaxGeqZero(*latency.values())
        except (TypeError, ValueError) as e:
            logging.warning(
                "Sparse latency calculation failed for %s, "
                "falling back to dense latency: %s",
                job.einsum_name, e,
            )

    used_fanout = {
        (component, dim): n
        for (component, einsum), dims in reuse.fanout.items()
        if einsum == job.einsum_name
        for dim, n in dims.items()
    }
    # If there's no loops that use this spatial fanout, the model won't
    # output any usage. We still reserve at least one spatial instance.
    for f in job.flattened_arch:
        if isinstance(f, arch.Spatialable):
            for s in f.spatial:
                used_fanout.setdefault((f.name, s.name), 1)

    # Scale used fanout to get actual usage
    spatial_usage = {}
    spatial_usage_df = {}
    memory_to_size = {}
    for node in job.flattened_arch:
        if isinstance(node, arch.Memory):
            memory_to_size[node.name] = node.size

        if isinstance(node, arch.Spatialable):
            for s in node.spatial:
                usage = used_fanout[node.name, s.name] / s.fanout
                scaled_usage = usage * s.usage_scale
                spatial_usage[node.name, s.name] = scaled_usage
                usage_key = f"usage<SEP>spatial<SEP>{node.name}<SEP>{s.name}"
                spatial_usage_df[usage_key] = scaled_usage

    # _power_gating expects raw fanout counts (it divides by s.fanout
    # internally), so pass used_fanout, not the pre-divided spatial_usage.
    component_to_non_power_gated_porp, _ = spec.arch._power_gating(
        compute_name=job.flattened_arch[-1].name,
        used_fanout=used_fanout,
    )

    if metrics & Metrics.ACTIONS:
        df.update(spatial_usage_df)

    # Compose sparse-adjusted actions from dense baseline + deltas.
    actions = gather_actions_with_sparse(
        dense_actions, sparse_result, use_name=True,
    )

    energy = compute_energy_from_actions(
        spec, actions, overall_latency, component_to_non_power_gated_porp
    )

    fusable_tensors = workload.tensor_names_used_in_multiple_einsums
    tensor_to_backing = {}
    for node in pmapping.nodes:
        if isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor not in tensor_to_backing and tensor in fusable_tensors:
                    tensor_to_backing[tensor] = node.component

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].component

    n_instances = workload.n_instances * workload.einsums[job.einsum_name].n_instances

    n_loop_options = set()
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.max_occupancy

        # Add metadata occupancy if this tensor has a sparse format at this level.
        # Metadata capacity (units) is already computed by the sparse pipeline in
        # per_rank_info.  Convert to bits and add to data occupancy so that
        # limit_capacity() rejects configs where data + metadata > buffer size.
        md_key = (buffet.tensor, buffet.level)
        if md_key in per_rank_info:
            info = per_rank_info[md_key]
            rank_cap = info.get("rank_capacity", [])
            rank_wb = info.get("rank_word_bits", [])
            for (md_units, pl_units), wb in zip(rank_cap, rank_wb):
                md_bits = wb.get("metadata") or 0
                pl_bits = wb.get("payload") or 0
                occupancy += md_units * md_bits + pl_units * pl_bits

        if occupancy == 0:
            continue
        if stats.persistent:
            occupancy *= n_instances

        for tensor, backing in tensor_to_backing.items():
            if (is_copy_op or buffet.tensor == tensor) and buffet.level == backing:
                df[tensor2col(tensor)] = occupancy / memory_to_size[buffet.level]

        total_occupancy.setdefault(buffet.level, {}).setdefault(stats.n_loops_above, 0)
        total_occupancy[buffet.level][stats.n_loops_above] += occupancy
        n_loop_options.add(stats.n_loops_above)

    for memory, occupancies in total_occupancy.items():
        if memory not in job.memories_track_all:
            continue
        running_total = 0
        for n_loop in sorted(n_loop_options):
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                df[nameloop2col(memory, n_loop)] = (
                    running_total / memory_to_size[memory]
                )

    if metrics & Metrics.ACTIONS:
        detailed_actions = gather_actions_with_sparse(
            dense_detailed_actions, sparse_result,
            verbose=True, use_name=True,
        )
        for key, count in detailed_actions.items():
            df[action2col(key)] = count.total * n_instances
        detailed_energy = compute_energy_from_actions(
            spec, detailed_actions, overall_latency, component_to_non_power_gated_porp
        )
        for key, energy_val in detailed_energy.items():
            df[energy2col(key)] = energy_val * n_instances
        for component, cur_latency in latency.items():
            df[f"latency<SEP>{component}"] = cur_latency * n_instances

        # Per-rank format columns (informational, pre-SAF logical counts)
        for (tensor, level), info in per_rank_info.items():
            rank_access = info.get("rank_access_counts")
            rank_cap = info.get("rank_capacity", [])
            for i, cap in enumerate(rank_cap):
                md_cap, pl_cap = cap
                df[f"format_capacity<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>metadata"] = md_cap
                df[f"format_capacity<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>payload"] = pl_cap
            if rank_access is not None:
                for i in range(len(rank_access.rank_metadata_reads)):
                    df[f"format_reads<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>metadata"] = rank_access.rank_metadata_reads[i]
                    df[f"format_reads<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>payload"] = rank_access.rank_payload_reads[i]
                    df[f"format_fills<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>metadata"] = rank_access.rank_metadata_fills[i]
                    df[f"format_fills<SEP>{level}<SEP>{tensor}<SEP>rank{i}<SEP>payload"] = rank_access.rank_payload_fills[i]

    if metrics & Metrics.LATENCY:
        df["Total<SEP>latency"] = overall_latency * n_instances
        # df[f"latency<SEP>compute"] = comp_latency * n_instances
        # For first latency, we'll follow the convention of treating compute
        # as a component, similarly to memory (see below).
        for compute_level, stats in reuse.compute_stats.items():  # FIRST LATENCY
            for idx, max_first_latency in stats.max_first_latency.items():
                df[firstlatency2col(compute_level.level, idx)] = (
                    max_first_latency * n_instances
                )

    if metrics.includes_dynamic_energy():
        dynamic_energy = [e for k, e in energy.items() if k.action != "leak"]
        df["Total<SEP>dynamic_energy"] = sum(dynamic_energy) * n_instances

    if metrics.includes_leak_energy():
        leak_energy = [e for k, e in energy.items() if k.action == "leak"]
        df["Total<SEP>leak_energy"] = sum(leak_energy) * n_instances

    per_memory_spatial_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        ignored = job.ignored_resources is not None and memory in job.ignored_resources
        key = f"usage<SEP>memory<SEP>{memory}"
        if not ignored:
            per_memory_spatial_usage_df[key] = (
                sum(occupancies.values()) / memory_to_size[memory]
            )
        if metrics & Metrics.ACTIONS:
            df[key] = sum(occupancies.values()) / memory_to_size[memory]

    if symbolic.PRINT_FORMULAS:
        for k, v in energy.items():
            print(f"{k}: {v}")
        for k, v in spatial_usage_df.items():
            print(f"{k}: {v}")
        for k, v in df.items():
            print(f"{k}: {v}")

    return (
        reuse.symbols,
        df,
        per_memory_spatial_usage_df,
        spatial_usage_df,
        reuse.tensor2mapping,
    )


def _compute_sparse_latency(reuse, latency_info: LatencyInfo, flattened_arch, spec):
    """Compute sparse-adjusted latency using post-sparse action counts.

    Uses post-sparse buffet_stats (after _recompute_action_counts) which
    already reflect SAF reductions. For gating, gated reads are added back
    because they still consume port bandwidth (cycles consumed, energy saved).

    Aggregates across all tensors per level (matching component_latency),
    then evaluates total_latency once per component.

    Metadata reads/writes are added at the level.

    Compute latency is scaled by compute_latency_ratio (post-4b / pre-sparse).
    """
    component_latency_result = {}

    symbol_table_base = {
        **dict(spec.variables),
        "variables": spec.variables,
        "max": Max,
        "min": sympy.Min,
        "sum": sympy.Add,
    }

    name2component = {node.name: node for node in flattened_arch}

    compute_obj = flattened_arch[-1]
    if not isinstance(compute_obj, arch.Compute):
        return {}

    compute_levels = set(c.level for c in reuse.compute_stats)

    # Aggregate post-sparse action counts per level (matching component_latency)
    component_to_actions: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    # Per-tensor tracking for max-based latency (e.g., Reg with dedicated ports)
    per_tensor_reads: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    per_tensor_writes: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level in compute_levels:
            continue
        component = buffet.level
        if component not in name2component:
            continue
        node = name2component[component]
        if not isinstance(node, arch.TensorHolder):
            continue

        # Ensure all declared actions have entries
        for action in node.actions:
            component_to_actions[component].setdefault(f"{action.name}_actions", 0)

        # Post-sparse action counts (SAF already applied)
        read_actions = (
            stats.max_per_unit_read_actions
            + stats.max_per_parent_drain_read_actions
        )
        write_actions = (
            stats.max_per_unit_write_actions
            + stats.max_per_parent_fill_write_actions
        )

        # For gating: add back gated deltas (gated reads consume BW)
        lt_key = (component, buffet.tensor)
        read_actions += latency_info.gated_read_action_deltas.get(lt_key, 0)
        write_actions += latency_info.gated_write_action_deltas.get(lt_key, 0)

        component_to_actions[component]["read_actions"] += read_actions
        per_tensor_reads[component][buffet.tensor] += read_actions
        # Per-unit computation-path reads only (no fill/drain)
        component_to_actions[component]["pu_read_actions"] += (
            stats.max_per_unit_read_actions
        )
        # Total actions across all spatial instances (for BW throttling
        # of shared levels above spatial, e.g. shared_glb)
        total_ra = (
            stats.total_read_actions
            + stats.total_parent_drain_read_actions
        )
        component_to_actions[component]["total_read_actions"] += total_ra
        if not isinstance(node, arch.Toll):
            component_to_actions[component]["write_actions"] += write_actions
            per_tensor_writes[component][buffet.tensor] += write_actions
            component_to_actions[component]["pu_write_actions"] += (
                stats.max_per_unit_write_actions
            )
            total_wa = (
                stats.total_write_actions
                + stats.total_parent_fill_write_actions
            )
            component_to_actions[component]["total_write_actions"] += total_wa

    # Add metadata actions per level (separate from data read/write —
    # the total_latency expression adds them: e.g. "read_actions + metadata_read_actions")
    for level, count in latency_info.metadata_read_actions.items():
        component_to_actions[level]["metadata_read_actions"] += count
    for level, count in latency_info.metadata_write_actions.items():
        component_to_actions[level]["metadata_write_actions"] += count

    # Compute latency: scale dense max_latency by compute_latency_ratio
    dense_compute_latency = Max(
        0, *[s.max_latency for s in reuse.compute_stats.values()]
    )
    ratio = latency_info.compute_latency_ratio
    compute_actions = dense_compute_latency * ratio
    component_to_actions[compute_obj.name]["compute_actions"] = compute_actions
    for action in compute_obj.actions:
        component_to_actions[compute_obj.name].setdefault(
            f"{action.name}_actions", 0
        )

    # Compute per-tensor max for levels with dedicated ports (e.g., Reg).
    # Use sympy Max to handle symbolic expressions correctly.
    for component in component_to_actions:
        if per_tensor_reads[component]:
            component_to_actions[component]["max_tensor_read_actions"] = Max(
                *per_tensor_reads[component].values()
            )
        if per_tensor_writes[component]:
            component_to_actions[component]["max_tensor_write_actions"] = Max(
                *per_tensor_writes[component].values()
            )

    # Synthetic variables (not real actions — skip in action-latency loop)
    _SYNTHETIC_ACTIONS = {
        "max_tensor_read_actions", "max_tensor_write_actions",
        "total_read_actions", "total_write_actions",
        "pu_read_actions", "pu_write_actions",
    }

    # Evaluate total_latency expression per component
    component_to_action_latency = defaultdict(dict)
    for component, actions in component_to_actions.items():
        node = name2component[component]
        for action_name, count in actions.items():
            if action_name in _SYNTHETIC_ACTIONS:
                continue
            action_key = action_name.rsplit("_", 1)[0]
            try:
                lat = node.actions[action_key].latency
            except (KeyError, TypeError):
                lat = 0
            component_to_action_latency[component][f"{action_key}_latency"] = (
                lat * count
            )

    for component, actions in component_to_actions.items():
        node = name2component[component]
        symbol_table = {
            "action2latency": component_to_action_latency[component],
            **symbol_table_base,
            **dict(node),
            **actions,
            **component_to_action_latency[component],
        }
        if node.total_latency is not None:
            component_latency_result[component] = eval_expression(
                node.total_latency,
                symbol_table,
                attr_name="latency",
                location=component,
            )
        elif isinstance(node, arch.Compute):
            component_latency_result[component] = sum(
                component_to_action_latency[component].values()
            )

        # Position-space utilization: divide by avg utilization fraction
        # when position-skipping causes PE load imbalance at compute.
        if (component in component_latency_result
                and isinstance(node, arch.Compute)
                and latency_info.position_space_utilization < 1.0):
            component_latency_result[component] /= (
                latency_info.position_space_utilization
            )

    return component_latency_result
