from collections import defaultdict
from numbers import Number

from accelforge.frontend import arch
from accelforge.frontend.arch import Memory, TensorHolder, Component
from accelforge.frontend.arch._flattened_arch import FlattenedArch
from accelforge.frontend.mapping import Compute, Mapping
from accelforge.frontend.mapping import TensorHolder as MappingTensorHolder

from accelforge.model._looptree.accesses import isl_buffer_accesses_from_buffet_actions
from accelforge.model._looptree.reuse.isl import IslReuseAnalysisOutput
from accelforge.model._looptree.reuse import SymbolicAnalysisOutput
from accelforge.model._looptree.types import Buffet

from accelforge.model._looptree.reuse.symbolic import BuffetStats, NetworkStats
from accelforge.util._frozenset import oset
from accelforge.util._sympy.broadcast_max import Max, MaxGeqZero, max_nonzero


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


def communication_latency(
    reuse: SymbolicAnalysisOutput,
    flattened_arch: FlattenedArch,
    tensor_to_backing: dict[str, str],
    output_tensors,
    n_fused: int = 0,
) -> tuple[object, dict[str, dict[int, object]]]:
    """
    Communication latency, split at the fused loops.

    Each tensor's path is a chain of connection latencies, one into each storage.
    Connections that cross loops are paid for each loop iteration, and are returned as
    descent (inputs moving down) or ascent (outputs moving up) latencies. These are
    multiplied by the number of iterations above the destination.

    The rest of each path is lumped into the Einsum delay: the slowest input winds down
    to compute, a computation runs, and the result winds up to the slowest output, all
    repeating once per iteration of the deepest backing storage.

    Returns (Einsum delay, {"descent" | "ascent": {level: latency}}).
    """
    name2component = {n.name: n for n in flattened_arch}
    name2index = {n.name: i for i, n in enumerate(flattened_arch)}
    compute = flattened_arch[-1]

    def connection_latencies(stats):
        """
        Pay latency for one action from every storage node. Once a buffet stats
        (reservation) is available, it can immediately receive data from above & start
        sending data to below. Per-level latency assumes that we can pre-send to a level
        below once this level is available, assuming that it arrives at the next level
        as soon as that level is available.
        """
        stats = sorted(stats, key=lambda c: name2index[c[0]])
        connections = []
        connection_latency = 0
        for i, (level, s) in enumerate(stats):
            if isinstance(s, NetworkStats):
                connection_latency += (
                    s.max_hops * name2component[level].actions["hop"].latency
                )
            elif isinstance(s, BuffetStats):
                actions = name2component[level].actions
                for action, count in s.net_total_actions().items():
                    if not isinstance(count, Number) or count != 0:
                        connection_latency += actions[action].latency
                connections.append((level, s, connection_latency))
                connection_latency = 0
            else:
                raise ValueError(f"Unknown stats type: {type(s)}")
        return connections, connection_latency

    tensor_stats = {}
    for b, s in reuse.buffet_stats.items():
        if b.level != compute.name:
            tensor_stats.setdefault(b.tensor, []).append((b.level, s))
    for n, s in reuse.network_stats.items():
        tensor_stats.setdefault(n.tensor, []).append((n.component, s))

    per_level_latencies = {"descent": defaultdict(list), "ascent": defaultdict(list)}
    slowest_input, slowest_output = 0, 0
    backing = []

    for tensor, stats in tensor_stats.items():
        is_output = tensor in output_tensors
        cur_latencies = per_level_latencies["ascent" if is_output else "descent"]

        connections, into_compute = connection_latencies(stats)
        cumulative_latency = into_compute
        for level, s, connection_latency in connections:
            if level == tensor_to_backing.get(tensor):
                backing.append(s)

            # < n_fused -> track each level independently
            if s.n_loops_above < n_fused:
                nloops = max(0, s.n_loops_above)
                cur_latencies[nloops].append(connection_latency * s.iterations_above)

            # >= n_fused -> lump all levels together. We can start once our
            # longest-latency input arrives at compute, does a computation, and then
            # goes back up to the longest-latency output.
            else:
                cumulative_latency += connection_latency

        if is_output:
            slowest_output = max_nonzero(slowest_output, cumulative_latency)
        else:
            slowest_input = max_nonzero(slowest_input, cumulative_latency)

    compute_latency = compute.actions["compute"].latency
    if backing:
        deepest_backing = max(backing, key=lambda s: s.n_loops_above)
        fused_iterations = deepest_backing.iterations_above
    else:
        fused_iterations = 1
    einsum_delay = slowest_input + slowest_output + compute_latency
    einsum_delay *= fused_iterations

    return (
        einsum_delay,
        {
            direction: {level: max_nonzero(*vals) for level, vals in per.items()}
            for direction, per in per_level_latencies.items()
        },
    )


def component_latency(
    looptree_results: SymbolicAnalysisOutput,
    flattened_arch: FlattenedArch,
    mapping: Mapping,
    per_level_components: oset,
    n_shared_loops: int,
):
    """
    Returns (component latency, per-level component latency).

    A component's latency is the sum, over its actions, of the action count (times the
    component's actions_scale) divided by the action's throughput. A Memory with
    separate_read_write_ports is two components for latency, "{name} (read)" and "{name}
    (write)". A network's `hop` count is the traffic over its most congested link.

    The per-level dict covers components in per_level_components, mapping each to
    {n_loops_above: latency of the actions at that many loops above plus the latency of
    actions lower in the tree}. Each level includes the levels below it, so the
    shallowest level is the component's whole latency.

    A transfer can only run while both ends' reservations are alive, so a storage's
    actions exchanged with its parent land at the storage's own loop level and actions
    exchanged with its child (or peers) at the child's level. Levels deeper than
    n_shared_loops cannot be shared with other Einsums and are collapsed into one.
    """
    component_to_actions: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )
    component_to_level_actions: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    name2component: dict[str, Component] = {node.name: node for node in flattened_arch}

    # May be different than actual name if reads and writes are separated
    latency_name2component: dict[str, Component] = {}

    compute_obj = flattened_arch[-1]
    if not isinstance(compute_obj, arch.Compute):
        raise ValueError("Last node in flattened_arch must be a Compute")

    def add(component_obj: Component, action: str, count, level=None):
        if action == "write" and isinstance(component_obj, arch.Toll):
            assert count == 0
            return

        name = component_obj.name
        if getattr(component_obj, "separate_read_write_ports", False):
            assert action in ("read", "write")
            name = f"{name} ({'read' if action == 'read' else 'write'})"

        latency_name2component[name] = component_obj
        count = 0 if count is None else count
        component_to_actions[name][action] += count
        if level is not None:
            level = min(level, n_shared_loops)
            component_to_level_actions[name][level][action] += count

    name2index = {node.name: i for i, node in enumerate(flattened_arch)}
    tensor2buffets: dict[str, list] = defaultdict(list)
    for buffet in looptree_results.buffet_stats:
        component_obj = name2component.get(buffet.level)
        if component_obj is None:
            raise ValueError(f"Component {buffet.level} found in mapping but not arch")
        if isinstance(component_obj, TensorHolder):
            tensor2buffets[buffet.tensor].append(buffet)
            for action in component_obj.actions:
                add(component_obj, action.name, 0)
        elif not isinstance(component_obj, arch.Compute):
            raise NotImplementedError(
                f"Component {buffet.level} is not a TensorHolder or Compute"
            )

    mapping_order: dict = {}
    for idx, node in enumerate(mapping.nodes):
        if isinstance(node, MappingTensorHolder):
            for tensor in node.tensors:
                mapping_order.setdefault((node.component, tensor), idx)
            mapping_order.setdefault(node.component, idx)

    def mapping_order_get(buffet):
        # Need this extra logic for copy Einsums, where the mapping has all storage
        # nodes changed to be only the copy source tensor
        return mapping_order.get(
            (buffet.level, buffet.tensor), mapping_order[buffet.level]
        )

    # =================================================================================
    # Buffet actions
    # =================================================================================
    bs = looptree_results.buffet_stats
    for bufs in tensor2buffets.values():
        bufs = sorted(bufs, key=mapping_order_get)
        for i, buffet in enumerate(bufs):
            stats = looptree_results.buffet_stats[buffet]
            component_obj = name2component[buffet.level]
            track = buffet.level in per_level_components

            # = above_loop_index
            n_above = min(stats.n_loops_above, n_shared_loops)
            n = i + 1
            child_above = bs[bufs[i + 1]].n_loops_above if n < len(bufs) else n_above

            for action, count in stats.net_max_per_unit_actions_to_parent().items():
                add(component_obj, action, count, n_above if track else None)
            for action, count in stats.net_max_per_unit_actions_to_child().items():
                add(component_obj, action, count, child_above if track else None)

    # =================================================================================
    # Compute actions
    # =================================================================================
    add(
        compute_obj,
        "compute",
        Max(0, *[s.max_latency for s in looptree_results.compute_stats.values()]),
        n_shared_loops if compute_obj.name in per_level_components else None,
    )

    # =================================================================================
    # Network actions
    # ==================================================================================
    network_to_level_dim_traffic = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    for network, network_stats in looptree_results.network_stats.items():
        component = network.component
        if component not in name2component:
            raise ValueError(f"Component {component} found in mapping but not arch")

        tensor_bufs = tensor2buffets[network.tensor]
        n2i = name2index
        bufs_after = [b for b in tensor_bufs if n2i[b.level] > n2i[component]]
        below = min([bs[b].n_loops_above for b in bufs_after], default=n_shared_loops)
        dim_traffic = network_to_level_dim_traffic[component][below]
        for dim, max_traffic_in_dim in network_stats.max_traffic.items():
            dim_traffic[dim] += max_traffic_in_dim

    for component, level_dim_traffic in network_to_level_dim_traffic.items():
        component_obj = name2component[component]
        for level, dim_traffic in level_dim_traffic.items():
            add(
                component_obj,
                "hop",
                MaxGeqZero(*dim_traffic.values()),
                level if component in per_level_components else None,
            )

    # =================================================================================
    # Actions to latency
    # =================================================================================
    component_latency = {}
    component_level_latency = {}

    for component, counts in component_to_actions.items():
        component_obj = latency_name2component[component]

        def counts2latency(action_counts):
            total = 0
            for action, count in action_counts.items():
                if action not in component_obj.actions:
                    raise ValueError(
                        f"Action {action} not found in component {component}"
                    )
                total += (
                    count
                    * component_obj.actions_scale
                    / component_obj.actions[action].throughput
                )
            return total

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
                per_level[level] = counts2latency(cumulative)
            component_latency[component] = per_level[min(per_level)]
        else:
            component_latency[component] = counts2latency(counts)

    return component_latency, component_level_latency
