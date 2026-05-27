from collections import defaultdict

from accelforge.frontend import arch
from accelforge.frontend.arch import Leaf, Memory, TensorHolder, Component
from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.frontend.mapping import Compute, Mapping
from accelforge.frontend.spec import Spec

from accelforge.model._looptree.accesses import isl_buffer_accesses_from_buffet_actions
from accelforge.model._looptree.mapping_utilities import get_leaves
from accelforge.model._looptree.reuse.isl import IslReuseAnalysisOutput
from accelforge.model._looptree.reuse import SymbolicAnalysisOutput
from accelforge.model._looptree.types import Buffet

from accelforge.model._looptree.reuse.symbolic import BuffetStats
from accelforge.util._eval_expressions import MATH_FUNCS, eval_expression
from accelforge.util._sympy.broadcast_max import Max, Min, MaxGeqZero
import symengine as se


def isl_to_summarized(
    looptree_results: IslReuseAnalysisOutput, mapping, workload
) -> SymbolicAnalysisOutput:
    accesses_stats = isl_buffer_accesses_from_buffet_actions(
        looptree_results, mapping, workload, is_path=False
    )
    buffet_stats = {
        Buffet(level=component, tensor=tensor, einsum=einsum): BuffetStats(
            total_writes_to_parent=accesses.max_per_unit_reads,
            total_reads_to_parent=accesses.max_per_unit_writes,
            read_scale=1,
            write_scale=1,
            count_upward=True,
            count_downward=True,
        )
        for (component, tensor, einsum), accesses in accesses_stats.items()
    }
    return SymbolicAnalysisOutput(buffet_stats=buffet_stats)


def component_latency(
    looptree_results: SymbolicAnalysisOutput,
    flattened_arch: FlattenedArch,
    mapping: Mapping,
    spec: Spec,
):
    component_to_actions: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    # Holds ``keywords" that do not map neatly to actions, e.g., max_hops for network
    component_to_keywords: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    name2component: dict[str, Component] = {node.name: node for node in flattened_arch}

    compute_obj = flattened_arch[-1]
    if not isinstance(compute_obj, arch.Compute):
        raise ValueError("Last node in flattened_arch must be a Compute")

    for buffet, buffet_stats in looptree_results.buffet_stats.items():
        component = buffet.level
        actions = component_to_actions[component]
        if component not in name2component:
            raise ValueError(f"Component {component} found in mapping but not arch")

        for action in name2component[component].actions:
            actions[f"{action.name}_actions"] += 0

        if isinstance(name2component[component], TensorHolder):
            actions["read_actions"] += (
                buffet_stats.max_per_unit_read_actions
                - buffet_stats.min_per_unit_skipped_first_read_actions
            )
            if not isinstance(name2component[component], arch.Toll):
                actions["write_actions"] += (
                    buffet_stats.max_per_unit_write_actions
                    - buffet_stats.min_per_unit_skipped_first_write_actions
                )
        elif isinstance(name2component[component], arch.Compute):
            pass
        else:
            raise NotImplementedError(
                f"Component {component} is not a TensorHolder or Compute"
            )

    network_to_max_link_traffic = defaultdict(lambda: defaultdict(lambda: 0))
    network_to_max_hops = defaultdict(lambda: [])
    for network, network_stats in looptree_results.network_stats.items():
        component = network.component
        if component not in name2component:
            raise ValueError(f"Component {component} found in mapping but not arch")

        dim_traffic = network_to_max_link_traffic[component]
        for dim, max_traffic_in_dim in network_stats.max_traffic.items():
            dim_traffic[dim] += max_traffic_in_dim

        network_to_max_hops[component].append(network_stats.max_hops)

    for network, network_stats in looptree_results.network_stats.items():
        component = network.component
        keywords = component_to_keywords[component]
        keywords["max_link_traffic"] = MaxGeqZero(
            *network_to_max_link_traffic[component].values()
        )
        keywords["max_hops"] = MaxGeqZero(
            *network_to_max_hops[component]
        )

    longest_compute_latency = Max(
        0, *[s.max_latency for s in looptree_results.compute_stats.values()]
    )
    component_to_actions[compute_obj.name]["compute_actions"] = longest_compute_latency

    for component, actions in component_to_actions.items():
        scale = getattr(name2component[component], "actions_scale", 1)
        if scale == 1:
            continue
        for action in actions:
            actions[action] = actions[action] * scale

    # TODO: Unhardcode "compute" name"
    component_to_action_latency = defaultdict(dict)
    for component, actions in component_to_actions.items():
        component_obj = name2component[component]
        for action, count in actions.items():
            action_name = action.rsplit("_", 1)[0]
            latency = component_obj.actions[action_name].latency
            component_to_action_latency[component][f"{action_name}_latency"] = (
                latency * count
            )

    component_latency = {}

    arch_vars = dict(spec.arch.variables) if spec.arch.variables else {}
    symbol_table_base = {  # TODO: Make a global symbol table initialization function
        **arch_vars,
        **dict(spec.variables),
        "variables": spec.variables,
        "arch_variables": spec.arch.variables,
        "max": Max,
        "min": Min,
        "sum": se.Add,
    }

    for component, component_obj in name2component.items():
        actions = component_to_actions[component]
        symbol_table = {
            "action2latency": component_to_action_latency[component],
            **symbol_table_base,
            **name2component[component].shallow_model_dump(include_None=True),
            **actions,
            **component_to_action_latency[component],
            **component_to_keywords[component],
        }
        if name2component[component].total_latency is not None:
            component_latency[component] = eval_expression(
                name2component[component].total_latency,
                symbol_table,
                attr_name="latency",
                location=component,
            )

    return component_latency
