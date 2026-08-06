from numbers import Number
from sympy import Symbol
import accelforge.frontend.arch as arch
from accelforge.frontend.mapping import Loop, TensorHolder, Toll
from accelforge.mapper.FFM._make_pmappings.pmapper_job import Job
from accelforge.model._looptree.reuse import symbolic
from accelforge.util._frozenset import oset
from accelforge.model._looptree.reuse.symbolic import (
    analyze_reuse_and_add_reservations_to_mapping,
)
from accelforge.model._looptree.energy import (
    compute_energy_from_actions,
    gather_actions,
)
from accelforge.model._looptree.latency.memory import (
    communication_latency,
    component_latency,
)
from accelforge.mapper.FFM._join_pmappings.pmapping_dataframe import (
    memory_usage2col,
    reservation2col,
    complatency2col,
    commlatency2col,
    tensor2col,
    action2col,
    energy2col,
)
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.util._sympy.broadcast_max import max_nonzero
from accelforge.util.indent import print


def run_model(
    job: Job,
    add_reservations: bool = True,
) -> tuple[list[Symbol], dict[str, float], dict[str, float], dict[str, float]]:
    from accelforge.model.main import InvalidMappingError

    pmapping = job.mapping
    spec = job.spec_one_einsum
    metrics = job.metrics
    is_copy_op = job.is_copy_operation
    workload = spec.workload

    df = {}

    reuse = analyze_reuse_and_add_reservations_to_mapping(
        job, add_reservations=add_reservations
    )

    tensor_to_backing = {}
    n_shared_loops = 0
    n_loops = 0
    for node in pmapping.nodes:
        if isinstance(node, Loop):
            n_loops += 1
        elif isinstance(node, TensorHolder):
            for tensor in node.tensors:
                if tensor not in tensor_to_backing:
                    tensor_to_backing[tensor] = node.component
                    if tensor in job.fusable_tensors:
                        n_shared_loops = n_loops

    private_level = n_shared_loops

    latency, latency_per_level = component_latency(
        reuse,
        job.flattened_arch,
        pmapping,
        per_level_components=oset(job.components_track_latency),
        n_shared_loops=n_shared_loops,
    )
    overall_latency = max_nonzero(*latency.values())

    # try:
    #     overall_latency = max_nonzero(*latency.values())
    # except Exception as e:
    #     for k, v in latency.items():
    #         if not isinstance(v, (Number, sympy.Symbol, sympy.Expr)):
    #             raise ValueError(
    #                 f"Invalid type for latency: {k}: {type(v)} {str(v).strip()}"
    #             )

    #     raise ValueError(
    #         f"Error calculating latency for {job.einsum_name}. Could not calculate "
    #         f"a symbolic max of the following latencies:\n\t"
    #         + "\n\t".join(
    #             [f"{k}: {type(v)} {str(v).strip()}" for k, v in latency.items()]
    #         )
    #     )

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
                used = used_fanout[node.name, s.name]
                usage = used / s.fanout
                if isinstance(used, Number) and used > s.fanout:
                    raise InvalidMappingError(
                        f"The mapping uses {used} spatial instances of {node.name} "
                        f"dimension {s.name}, but its fanout is only {s.fanout}. "
                        f"Use spatial loops with a smaller fanout; i.e., reduce the "
                        f"loop bound, which can be done by increasing the spatial "
                        f"loop's tile shape or reducing the tile shape of an outer "
                        f"loop."
                    )
                scaled_usage = usage * s.usage_scale
                spatial_usage[node.name, s.name] = scaled_usage
                s = reservation2col(f"{node.name} {s.name}", private_level)
                spatial_usage_df[s] = scaled_usage

    component_to_non_power_gated_porp, _ = spec.arch._power_gating(
        compute_name=job.flattened_arch[-1].name,
        used_fanout=spatial_usage,
    )

    if metrics & Metrics.ACTIONS:
        df.update(spatial_usage_df)

    actions = gather_actions(reuse, None, use_name=True, spec=spec)
    energy = compute_energy_from_actions(
        spec, actions, overall_latency, component_to_non_power_gated_porp
    )

    # A Toll is a pass-through and must never be the outermost level backing a tensor —
    # that would leave no real Memory holding it.
    for node in pmapping.nodes:
        if isinstance(node, Toll):
            for tensor in node.tensors:
                if (
                    tensor in tensor_to_backing
                    and tensor_to_backing[tensor] == node.component
                ):
                    raise ValueError(
                        f"Toll '{node.component}' is the outermost level holding "
                        f"tensor '{tensor}' in einsum '{job.einsum_name}'. "
                        f"A Toll cannot be the outermost holder of a tensor — a "
                        f"Memory above the Toll must also keep it."
                    )

    total_occupancy = {}
    compute_unit = pmapping.nodes[-1].component

    n_instances = workload.n_instances * workload.einsums[job.einsum_name].n_instances

    # =================================================================================
    # Tensor sizes
    # =================================================================================
    n_loop_options = oset()
    for buffet, stats in reuse.buffet_stats.items():
        if buffet.level == compute_unit:
            continue

        occupancy = stats.max_occupancy

        if occupancy == 0:
            continue
        if stats.persistent:
            occupancy *= n_instances

        for tensor, backing in tensor_to_backing.items():
            if tensor not in job.fusable_tensors:
                continue
            if (is_copy_op or buffet.tensor == tensor) and buffet.level == backing:
                df[tensor2col(tensor)] = occupancy / memory_to_size[buffet.level]

        total_occupancy.setdefault(buffet.level, {}).setdefault(stats.n_loops_above, 0)
        total_occupancy[buffet.level][stats.n_loops_above] += occupancy
        n_loop_options.add(stats.n_loops_above)

        if metrics & Metrics.DETAILED_MEMORY_USAGE:
            key = memory_usage2col(buffet.level, buffet.tensor)
            df[key] = occupancy / memory_to_size[buffet.level]

    # =================================================================================
    # Reservations
    # =================================================================================
    for memory, occupancies in total_occupancy.items():
        if memory not in job.memories_track_all:
            continue
        size = memory_to_size[memory]
        running_total = 0
        for n_loop, occupancy in sorted(occupancies.items()):
            running_total += occupancy
            col = reservation2col(memory, int(min(n_loop, private_level)))
            df[col] = running_total / size
        if isinstance(running_total, Number) and running_total > size:
            raise InvalidMappingError(
                f"The mapping uses {running_total} bits of {memory} but its size is "
                f"only {size} bits. Use a smaller tile shape for loops below the "
                f"storage nodes of {memory}."
            )

    # =================================================================================
    # Actions
    # =================================================================================
    if metrics & Metrics.ACTIONS:
        detailed_actions = gather_actions(
            reuse, None, verbose=True, use_name=True, spec=spec
        )
        for key, count in detailed_actions.items():
            df[action2col(key)] = count.total * n_instances
        detailed_energy = compute_energy_from_actions(
            spec, detailed_actions, overall_latency, component_to_non_power_gated_porp
        )
        for key, energy_val in detailed_energy.items():
            df[energy2col(key)] = energy_val * n_instances

    actions_df = {}
    simple_actions = gather_actions(
        reuse, None, verbose=False, use_name=True, spec=spec
    )
    for key, count in simple_actions.items():
        actions_df[action2col(key)] = count.total * n_instances

    # =================================================================================
    # Latency
    # =================================================================================
    if metrics.includes_latency():
        for component, cur_latency in latency.items():
            df[f"component_latency<SEP>{component}"] = cur_latency * n_instances

        # Components shared across Einsums get per-level latency columns so joining can
        # sum their busy time across Einsums and let it overlap with the other Einsums'
        # latency. Their latency is folded into Total<SEP>latency at joining time
        # instead of here.
        for component, level_latency in latency_per_level.items():
            for level, cur_latency in level_latency.items():
                df[complatency2col(component, level)] = cur_latency * n_instances

        # The total latency is the sum of the Einsum's wind-up/down delay and the
        # slowest component's busy time. Fused-loop-crossing descent and ascent
        # latencies get their own columns because they may be overlapped with other
        # Einsums.
        einsum_delay, ascent_descent_latency = communication_latency(
            reuse,
            job.flattened_arch,
            tensor_to_backing,
            workload.einsums[job.einsum_name].output_tensor_names,
            n_fused=n_shared_loops,
        )
        for direction, per_level in ascent_descent_latency.items():
            for level, cur_latency in per_level.items():
                df[commlatency2col(direction, level)] = cur_latency * n_instances

        per_component_total = []
        for component, l in latency.items():
            if component not in latency_per_level:
                if not isinstance(l, Number) or l != 0:
                    per_component_total.append(l)

        slowest = max_nonzero(*per_component_total)
        df["Total<SEP>latency"] = (slowest + einsum_delay) * n_instances

    # =================================================================================
    # Energy
    # =================================================================================
    if metrics.includes_dynamic_energy():
        dynamic_energy = [e for k, e in energy.items() if k.action != "leak"]
        df["Total<SEP>dynamic_energy"] = sum(dynamic_energy) * n_instances

    if metrics.includes_leak_energy():
        leak_energy = [e for k, e in energy.items() if k.action == "leak"]
        df["Total<SEP>leak_energy"] = sum(leak_energy) * n_instances

    # =================================================================================
    # Memory usage
    # =================================================================================
    per_memory_spatial_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        ignored = memory in job.ignored_resources
        key = reservation2col(memory, private_level)
        if not ignored:
            per_memory_spatial_usage_df[key] = (
                sum(occupancies.values()) / memory_to_size[memory]
            )
        # if metrics & Metrics.ACTIONS:
        #     df[key] = sum(occupancies.values()) / memory_to_size[memory]

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
        actions_df,
    )
