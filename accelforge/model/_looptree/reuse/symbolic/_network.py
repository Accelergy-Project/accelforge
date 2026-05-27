from accelforge.frontend.mapping import (
    Spatial
)
from accelforge.frontend._workload_isl._symbolic import (
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.util._sympy.broadcast_max import MaxGeqZero, MinGeqZero

from ._common import AnalysisInfo
from ._stats import NetworkStats, SymbolicAnalysisOutput


class NetworkAnalyzer:
    def __init__(self, network_stats):
        self.overall_max_hops: dict = {}
        self.network_stats = network_stats

    def accumulate_child_result(
        self,
        child_result: SymbolicAnalysisOutput,
        info: AnalysisInfo,
        shape_repeats,
        einsum_name,
        child_shape,
        node: Spatial,
    ):
        """This function is called for every repeated shape."""
        flattened_arch = info.job.flattened_arch

        for network, child_network_stats in child_result.network_stats.items():
            src_component = flattened_arch[network.source.level]
            if network not in self.network_stats:
                self.network_stats[network] = NetworkStats()
            accumulated_network_stats = self.network_stats[network]

            # We only need to update the summary if the spatial loop is for
            # a component higher than the network of interest
            if flattened_arch.is_above(node.component, network.component):
                accumulated_network_stats.total_hops += (
                    child_network_stats.total_hops * shape_repeats
                )
                accumulated_network_stats.max_hops = MaxGeqZero(
                    accumulated_network_stats.max_hops,
                    child_network_stats.max_hops,
                )
                for k, v in child_network_stats.max_traffic.items():
                    accumulated_network_stats.max_traffic[k] = MaxGeqZero(
                        accumulated_network_stats.max_traffic.get(k, 0),
                        v
                    )
                continue

            volume = self._get_data_volume(network, einsum_name, info, child_shape)

            relevancy = info.tensor_to_relevancy[network.tensor][node.rank_variable]

            # The fanout in this dimension in mapping nodes below, i.e., the stride
            last_fanout = child_result.fanout.get((node.component, einsum_name), {})
            last_fanout = last_fanout.get(node.name, 1)
            if isinstance(relevancy, Irrelevant):
                # The volume travels through link by link in one axis of the mesh
                # Distributed or not, the amount of total cost is the same.
                # However, the accesses now come from different physical memories
                total_cost = multicast_cost(shape_repeats, last_fanout)*volume
                max_hops = shape_repeats*last_fanout
                max_traffic = volume
            elif isinstance(relevancy, Relevant):
                # If distributed, then we bind data as locally as possible in the
                # physical buffers
                if src_component._get_physical_fanout_along(node.name) > 1:
                    physical_stride = src_component._get_physical_stride_along(node.name)

                    n_dsts_per_physical = MinGeqZero(
                        # if last_fanout > physical_stride, set n_dst to 1, which results in 0 hops
                        # later (which is correct because the set of destinations always overlap
                        # the set of sources).
                        MaxGeqZero(physical_stride / last_fanout, 1),
                        shape_repeats
                    )
                    n_activated_physical = MaxGeqZero(shape_repeats*last_fanout/physical_stride, 1)
                    total_cost = (
                        n_activated_physical
                        *
                        unicast_cost(n_dsts_per_physical, last_fanout)
                        *
                        volume
                    )
                    max_hops = MinGeqZero((n_dsts_per_physical-1)*last_fanout, physical_stride)
                    max_traffic = (n_dsts_per_physical-1)*volume
                else:
                    total_cost = unicast_cost(shape_repeats, last_fanout)*volume
                    max_hops = shape_repeats * last_fanout
                    max_traffic = (shape_repeats-1)*volume
            elif isinstance(relevancy, PartiallyRelevant):
                raise NotImplementedError()
            else:
                raise RuntimeError(f"unhandled relevancy type {relevancy}")

            # Each subsequent call to this function (i.e., over different iterations of a spatial loop)
            # adds more to the max hops
            self.overall_max_hops[network] = self.overall_max_hops.get(network, 0) + max_hops

            accumulated_network_stats.total_hops += (
                total_cost + child_network_stats.total_hops*shape_repeats
            )
            accumulated_network_stats.max_hops = MaxGeqZero(
                accumulated_network_stats.max_hops,
                self.overall_max_hops[network] + child_network_stats.max_hops,
            )
            accumulated_network_stats.max_traffic[node.name] = MaxGeqZero(
                accumulated_network_stats.max_traffic.get(node.name, 0),
                max_traffic + child_network_stats.max_traffic.get(node.name, 0)
            )

        return self.overall_max_hops

    def _get_data_volume(self, network, einsum_name, info, child_shape):
        flattened_arch = info.job.flattened_arch
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
        return volume


def multicast_cost(n_dsts, stride):
    """Returns total hops of multicast along a dimension."""
    return (n_dsts-1)*stride


def unicast_cost(n_dsts, stride):
    """Returns total hops of unicast along a dimension."""
    # Cost of unicast is the cost of delivering to each point in
    # the dimension with shape as stride
    return arithmetic_sum(n_dsts-1)*stride


def arithmetic_sum(n):
    return 0.5 * (n+1) * n