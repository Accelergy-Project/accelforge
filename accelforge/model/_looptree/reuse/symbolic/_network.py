import copy
from accelforge.frontend import arch
from accelforge.frontend.arch import Network as NetworkSpec
from accelforge.frontend.mapping import (
    TensorHolder,
    TensorName
)
from accelforge.frontend._workload_isl._symbolic import (
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.util._sympy.broadcast_max import Min, Max, MaxGeqZero

from ._common import AnalysisInfo
from ._stats import NetworkStats


class NetworkAnalyzer:
    def __init__(self, network_stats):
        self.overall_max_hops = 0
        self.network_stats = network_stats

    def accumulate_child_result(
        self,
        child_result,
        info: AnalysisInfo,
        shape_repeats,
        einsum_name,
        child_shape,
        node,
    ):
        flattened_arch = info.job.flattened_arch

        for network, child_network_stats in child_result.network_stats.items():
            if network not in self.network_stats:
                self.network_stats[network] = NetworkStats()
            accumulated_network_stats = self.network_stats[network]

            accumulated_network_stats.total_hops += (
                child_network_stats.total_hops * shape_repeats
            )
            accumulated_network_stats.max_hops = MaxGeqZero(
                accumulated_network_stats.max_hops,
                child_network_stats.max_hops,
            )
            projection = info.einsum_tensor_to_projection[(einsum_name, network.tensor)]
            component_object = flattened_arch[network.component]
            workload_bpv = info.job.einsum.tensor_accesses[
                network.tensor
            ].bits_per_value
            bits_per_value = component_object.bits_per_value.get(
                network.tensor, workload_bpv
            )
            bits_per_action = component_object.bits_per_action
            if bits_per_action is not None:
                actions_per_value = bits_per_value / bits_per_action
            else:
                actions_per_value = bits_per_value
            volume = (
                compute_dense_tile_occupancy(projection, child_shape)
                * actions_per_value
            )

            if is_component_a_above_b(node.component, network.component, flattened_arch):
                continue

            relevancy = info.tensor_to_relevancy[network.tensor][node.rank_variable]

            last_fanout = child_result.fanout.get((node.component, einsum_name), {})
            last_fanout = last_fanout.get(node.name, 1)
            if isinstance(relevancy, Irrelevant):
                # Cost of multicasting is the cost of delivering along the dimension
                multicast_hops = shape_repeats * last_fanout
                multicast_cost = multicast_hops * volume
                self.overall_max_hops += multicast_hops

                accumulated_network_stats.total_hops += multicast_cost
                accumulated_network_stats.max_hops = MaxGeqZero(
                    accumulated_network_stats.max_hops,
                    self.overall_max_hops + child_network_stats.max_hops,
                )
            elif isinstance(relevancy, Relevant):
                # Cost of unicast is the cost of delivering to each point in
                # the dimension with shape as stride
                # TODO: we should use the actual stride
                total_unicast_cost = (
                    0.5 * (shape_repeats + 1) * shape_repeats * last_fanout * volume
                )
                max_unicast_hops = shape_repeats * last_fanout
                self.overall_max_hops += max_unicast_hops

                accumulated_network_stats.total_hops += total_unicast_cost
                accumulated_network_stats.max_hops = MaxGeqZero(
                    accumulated_network_stats.max_hops,
                    self.overall_max_hops + child_network_stats.max_hops,
                )
            elif isinstance(relevancy, PartiallyRelevant):
                raise NotImplementedError()
            else:
                raise RuntimeError(f"unhandled relevancy type {relevancy}")

        return self.overall_max_hops


def reduce_dicts(dict1: dict, dict2: dict, reduce_op):
    for key in dict1:
        if key not in dict2:
            dict2[key] = dict1[key]
        else:
            dict2[key] = reduce_op(dict1[key], dict2[key])


def get_total_to_per_unit(total, max_per_unit):
    if total == 0 and max_per_unit != 0:
        raise ValueError(f"total is 0 but max_per_unit is {max_per_unit}")
    if total == 0:
        return 1
    return max_per_unit / total


def has_parent_tensor_holder(
    tensor: TensorName, node_idx: int, info
) -> bool:
    for node in info.mapping[:node_idx]:
        if isinstance(node, TensorHolder) and tensor in node.tensors:
            return True
    return False


def find_component_object(
    component: str, flattened_arch: list[arch.Leaf]
) -> arch.TensorHolder:
    for node in flattened_arch:
        if node.name == component:
            return node
    raise ValueError(f"Component {component} not found in flattened arch")


def is_component_a_above_b(component_a: str, component_b: str, flattened_arch):
    a_found = False
    b_found = False
    for node in flattened_arch:
        if node.name == component_a:
            a_found = True
        if node.name == component_b:
            b_found = True

        if a_found and not b_found:
            return True
        elif b_found and not a_found:
            return False
    raise ValueError(f"Neither {component_a} nor {component_b} found in flattened arch")

