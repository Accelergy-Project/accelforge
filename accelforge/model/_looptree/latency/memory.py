from collections import defaultdict
from numbers import Number

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

from accelforge.model._looptree.reuse.symbolic import BuffetStats, NetworkStats
from accelforge.util._eval_expressions import MATH_FUNCS, eval_expression
from accelforge.util._frozenset import oset
from accelforge.util._sympy.broadcast_max import Max, Min, MaxGeqZero, max_nonzero
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


def communication_latency(
    reuse: SymbolicAnalysisOutput,
    flattened_arch: FlattenedArch,
    tensor_to_backing: dict[str, str],
    output_tensors,
) -> dict[str, object]:
    """
    Communication latency per component: the time to move one tile of each tensor to
    that level. Inputs travel from their backing storage down; outputs are produced
    after the worst input reaches compute, then travel from compute up. Each level on
    the path charges one call of each action it performs for the tensor; each network
    on the path charges its longest route. The whole path repeats once per temporal
    iteration above the backing storage. Returns, for each component, the worst
    communication latency over all tensors.
    """
    name2component = {n.name: n for n in flattened_arch}
    name2index = {n.name: i for i, n in enumerate(flattened_arch)}

    tensor_stats = {}
    for b, s in reuse.buffet_stats.items():
        tensor_stats.setdefault(b.tensor, []).append((b.level, s))
    for n, s in reuse.network_stats.items():
        tensor_stats.setdefault(n.tensor, []).append((n.component, s))

    communication = defaultdict(list)
    worst_input_to_compute = 0
    # Inputs first: outputs build on the worst input's arrival at compute.
    tensor_order = sorted(tensor_stats, key=lambda t: t in output_tensors)
    for tensor in tensor_order:
        stats = tensor_stats[tensor]
        is_output = tensor in output_tensors
        stats.sort(key=lambda c: name2index[c[0]], reverse=is_output)
        backing = tensor_to_backing[tensor]

        cur_latency = 0
        iterations = None
        tensor_latency = {}
        for level, stats in stats:
            if isinstance(stats, BuffetStats):
                if level == backing:
                    iterations = stats.iterations_above
                cur_latency += sum(
                    name2component[level].actions[action].latency
                    for action, count in stats.total_actions.items()
                    if not isinstance(count, Number) or count != 0
                )
                tensor_latency[level] = cur_latency
            elif isinstance(stats, NetworkStats):
                cur_latency += (
                    stats.max_hops * name2component[level].actions["hop"].latency
                )
                tensor_latency[level] = cur_latency
            else:
                raise ValueError(f"Unknown stats type: {type(stats)}")

        assert iterations is not None, f"Tensor {tensor} has no backing storage"
        start = worst_input_to_compute if is_output else 0
        for level in tensor_latency:
            communication[level].append(start + tensor_latency[level] * iterations)
        if not is_output:
            worst_input_to_compute = max_nonzero(
                worst_input_to_compute, cur_latency * iterations
            )

    return {level: max_nonzero(*vals) for level, vals in communication.items()}


def component_latency(
    looptree_results: SymbolicAnalysisOutput,
    flattened_arch: FlattenedArch,
    mapping: Mapping,
    spec: Spec,
    per_level_components: oset = oset(),
):
    """
    Returns (component latency, per-level component latency).

    The per-level dict covers components in per_level_components, mapping each to
    {n_loops_above: latency of the actions at that many loops above plus the latency of
    actions lower in the tree}. Each level includes the levels below it, so the
    shallowest level is the component's whole latency. Levels must be completed before
    moving to other branches of the LoopTree, but may be overlapped with the latencies
    of other branches below the current level.
    """
    component_to_actions: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    component_to_level_actions: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
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
            reads = buffet_stats.net_max_per_unit_actions("read")
            writes = buffet_stats.net_max_per_unit_actions("write")
            actions["read"] += reads
            include_writes = not isinstance(name2component[component], arch.Toll)
            n_loops_above = buffet_stats.n_loops_above
            if include_writes:
                actions["write"] += writes
            if component in per_level_components:
                level_actions = component_to_level_actions[component][n_loops_above]
                level_actions["read"] += reads
                if include_writes:
                    level_actions["write"] += writes
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
        keywords["max_hops"] = MaxGeqZero(*network_to_max_hops[component])
        actions = component_to_actions[component]
        for action in name2component[component].actions:
            actions[action.name] = 0

    longest_compute_latency = Max(
        0, *[s.max_latency for s in looptree_results.compute_stats.values()]
    )
    component_to_actions[compute_obj.name]["compute"] = longest_compute_latency

    for component, action_counts in component_to_actions.items():
        component_obj = name2component[component]
        for action_name in action_counts:
            if action_name not in component_obj.actions:
                raise ValueError(
                    f"Action {action_name} not found in component {component}"
                )

    component_latency = {}
    component_level_latency = {}

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
        if (
            component not in component_to_actions
            and component not in component_to_keywords
        ):
            continue
        component_obj = name2component[component]
        dump = component_obj.shallow_model_dump(include_None=True)
        if component in component_to_keywords:
            dump |= component_to_keywords[component]

        def eval_latency(action_counts):
            symbol_table = {**symbol_table_base, **dump}
            cur_actions = EvalableList()
            for action, count in action_counts.items():
                a = component_obj.actions[action].model_copy()
                a._set_n_calls(count * component_obj.actions_scale)
                cur_actions.append(a)
            symbol_table["actions"] = cur_actions

            return eval_expression(
                component_obj.total_latency,
                symbol_table,
                attr_name="latency",
                location=component,
            )

        counts = component_to_actions[component]
        if component in component_to_level_actions:
            actions = component_to_level_actions[component]
            # Evaluate on running action totals from the deepest level up so each
            # level's latency includes the levels below it and the shallowest level
            # is the whole latency.
            cumulative = defaultdict(lambda: 0)
            per_level = component_level_latency[component] = {}
            for level, cur_actions in sorted(actions.items(), reverse=True):
                for action, count in cur_actions.items():
                    cumulative[action] += count
                per_level[level] = eval_latency(cumulative)
            component_latency[component] = per_level[min(per_level)]
        else:
            component_latency[component] = eval_latency(counts)

    return component_latency, component_level_latency
