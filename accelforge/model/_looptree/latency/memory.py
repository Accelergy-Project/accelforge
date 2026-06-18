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
from accelforge.util._basetypes import EvalableList
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


def _sum(*args):
    """Sum that accepts either a single iterable (e.g. a generator) or varargs, so
    total_latency expressions like ``sum(x for x in ...)`` and ``sum(*values)`` both
    evaluate to a symengine expression."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return se.Add(*args) if args else se.Integer(0)


def _max(*args):
    """Max that accepts either a single iterable (generator) or varargs."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return Max(*args)


def _min(*args):
    """Min that accepts either a single iterable (generator) or varargs."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        args = tuple(args[0])
    return Min(*args)


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
            actions[action.name] += 0

        if isinstance(name2component[component], TensorHolder):
            actions["read"] += (
                buffet_stats.max_per_unit_read_actions
                - buffet_stats.min_per_unit_skipped_first_read_actions
            )
            if not isinstance(name2component[component], arch.Toll):
                actions["write"] += (
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
    # Aggregates across tensors
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
    component_to_actions[compute_obj.name]["compute"] = longest_compute_latency

    new_component_to_actions: dict[str, list] = {}
    for component, action_counts in component_to_actions.items():
        component_obj = name2component[component]
        scale = getattr(component_obj, "actions_scale", 1)
        for action_name in action_counts:
            if action_name not in component_obj.actions:
                raise ValueError(
                    f"Action {action_name} not found in component {component}"
                )
        cur_actions = EvalableList()
        for a in component_obj.actions:
            a = a.model_copy()
            a._set_n_calls(action_counts.get(a.name, 0) * scale)
            cur_actions.append(a)
        new_component_to_actions[component] = cur_actions
    component_to_actions = new_component_to_actions

    component_latency = {}

    arch_vars = dict(spec.arch.variables) if spec.arch.variables else {}
    symbol_table_base = {  # TODO: Make a global symbol table initialization function
        **arch_vars,
        **dict(spec.variables),
        "variables": spec.variables,
        "arch_variables": spec.arch.variables,
        "max": _max,
        "min": _min,
        "sum": _sum,
    }

    for component in name2component:
        if component not in component_to_actions and component not in component_to_keywords:
            continue
        component_obj = name2component[component]
        dump = component_obj.shallow_model_dump(include_None=True)
        # Replace serialized `actions` dump with local Action copies that carry
        # the correct n_calls for this job, so formulas can access `a.n_calls`,
        # `a.throughput`, etc. without mutating the shared spec state.
        if component in component_to_actions:
            dump["actions"] = component_to_actions[component]
        if component in component_to_keywords:
            dump |= component_to_keywords[component]
        symbol_table = {**symbol_table_base, **dump}
        if component_obj.total_latency is not None:
            component_latency[component] = eval_expression(
                component_obj.total_latency,
                symbol_table,
                attr_name="latency",
                location=component,
            )

    return component_latency
