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
)
from accelforge.model._looptree.latency.memory import component_latency
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
from accelforge.util._sympy.broadcast_max import MaxGeqZero


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
                s = f"usage<SEP>spatial<SEP>{node.name}<SEP>{s.name}"
                spatial_usage_df[s] = scaled_usage

    component_to_non_power_gated_porp, _ = spec.arch._power_gating(
        compute_name=job.flattened_arch[-1].name,
        used_fanout=spatial_usage,
    )

    if metrics & Metrics.ACTIONS:
        df.update(spatial_usage_df)

    actions = gather_actions(reuse, None, use_name=True)
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
        detailed_actions = gather_actions(reuse, None, verbose=True, use_name=True)
        for key, count in detailed_actions.items():
            df[action2col(key)] = count.total * n_instances
        detailed_energy = compute_energy_from_actions(
            spec, detailed_actions, overall_latency, component_to_non_power_gated_porp
        )
        for key, energy_val in detailed_energy.items():
            df[energy2col(key)] = energy_val * n_instances
        for component, cur_latency in latency.items():
            df[f"latency<SEP>{component}"] = cur_latency * n_instances

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
