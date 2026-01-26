from sympy import Symbol
import fastfusion.frontend.arch as arch
from fastfusion.frontend.mapping import TensorHolder
from fastfusion.mapper.FFM._make_pmappings.pmapper_job import Job
from fastfusion.model._looptree.reuse.symbolic import (
    analyze_reuse_and_add_reservations_to_mapping,
)
from fastfusion.model._looptree.energy import (
    compute_energy_from_actions,
    gather_actions,
)
from fastfusion.model._looptree.latency.memory import component_latency
from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import (
    nameloop2col,
    tensor2col,
    firstlatency2col,
    action2col,
    energy2col,
)
from fastfusion.frontend.mapper.metrics import Metrics
import sympy
from numbers import Number
from fastfusion.util._sympy.broadcast_max import Max


def run_model(
    job: Job,
) -> tuple[list[Symbol], dict[str, float], dict[str, float], dict[str, float]]:
    pmapping = job.mapping
    spec = job.spec
    metrics = job.metrics
    is_copy_op = job.is_copy_operation
    workload = spec.workload

    df = {}

    reuse = analyze_reuse_and_add_reservations_to_mapping(job)

    latency = component_latency(reuse, job.flattened_arch, pmapping, spec)
    try:
        overall_latency = Max(*latency.values()) if latency else 0
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

    component_to_max_fanout = {}
    component_to_usage_scale = {}
    memory_to_size = {}
    component_to_non_power_gated_porp = {}
    non_power_gated_instances = 1
    for node in job.flattened_arch:
        if isinstance(node, arch.TensorHolder):
            if isinstance(node, arch.Memory):
                memory_to_size[node.name] = node.size
        component_to_max_fanout[node.name] = {s.name: s.fanout for s in node.spatial}
        component_to_usage_scale[node.name] = {
            s.name: s.usage_scale for s in node.spatial
        }

    usage_df = {}
    unscaled_usage = {}
    for (component, einsum), per_dim_fanout in reuse.fanout.items():
        for dim, fanout in per_dim_fanout.items():
            usage = fanout / component_to_max_fanout[component][dim]
            usage_df[f"usage<SEP>spatial<SEP>{component}<SEP>{dim}"] = (
                usage * component_to_usage_scale[component][dim]
            )
            unscaled_usage.setdefault(component, {})[dim] = usage

    for node in job.flattened_arch:
        for s in node.spatial:
            if s.power_gateable:
                non_power_gated_instances *= unscaled_usage.get(node.name, {}).get(
                    s.name, 1
                )
        component_to_non_power_gated_porp[node.name] = non_power_gated_instances

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
                df[tensor2col(tensor)] = occupancy

        total_occupancy.setdefault(buffet.level, {}).setdefault(stats.n_loops_above, 0)
        total_occupancy[buffet.level][stats.n_loops_above] += occupancy
        n_loop_options.add(stats.n_loops_above)

    for memory, occupancies in total_occupancy.items():
        if memory not in job.memories_track_all:
            continue
        running_total = 0
        for n_loop in n_loop_options:
            if n_loop in occupancies:
                running_total += occupancies[n_loop]
                df[nameloop2col(memory, n_loop)] = running_total

    if metrics & Metrics.ACTIONS:
        detailed_actions = gather_actions(reuse, None, verbose=True, use_name=True)
        for key, count in detailed_actions.items():
            df[action2col(key)] = count.total * n_instances
        detailed_energy = compute_energy_from_actions(
            spec, detailed_actions, overall_latency, component_to_non_power_gated_porp
        )
        for key, energy_val in detailed_energy.items():
            df[energy2col(key)] = energy_val * n_instances

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
        for component, cur_latency in latency.items():
            df[f"latency<SEP>{component}"] = cur_latency * n_instances

    if metrics & Metrics.ENERGY:
        df["Total<SEP>energy"] = sum(energy.values()) * n_instances
        for key, energy in energy.items():
            df[f"energy<SEP>{key.level}<SEP>{key.action}"] = energy * n_instances

    per_memory_usage_df = {}
    for memory, occupancies in total_occupancy.items():
        key = f"usage<SEP>memory<SEP>{memory}"
        per_memory_usage_df[key] = sum(occupancies.values()) / memory_to_size[memory]

    return (
        reuse.symbols,
        df,
        per_memory_usage_df,
        usage_df,
        reuse.tensor2mapping,
    )
