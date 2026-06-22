from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from accelforge.frontend.mapping import Spatial
from accelforge.frontend.arch.components import TopologySpec
from accelforge.frontend._workload_isl._symbolic import (
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.util._sympy.broadcast_max import max_nonzero, min_nonzero

from ._common import AnalysisInfo
from ._stats import NetworkStats, SymbolicAnalysisOutput


@dataclass
class PerLoopTransferCost:
    """The per-spatial-loop cost contributed by a single network, as computed
    by a :class:`TopologyModel`."""

    total_cost: Any
    """Total hops contributed by data movement over this spatial loop."""
    max_hops: Any
    """Hops added to the longest route by this spatial loop."""
    max_traffic: Any
    """Maximum traffic (in actions) on any single link along this dimension."""


class TopologyModel(ABC):
    """Computes the cost of moving data across a network of a given topology.

    Subclasses encapsulate everything topology-specific about how a tensor's
    data is delivered across a spatial fanout. :class:`NetworkAnalyzer` selects
    the model for each network from its component's
    :class:`~accelforge.frontend.arch.components.TopologySpec` and remains
    agnostic to the topology itself.

    Instances are stateful: they accumulate per-network max hops across the
    repeated spatial-loop iterations of a single :class:`NetworkAnalyzer`, so a
    fresh model is constructed for each analyzer (see :func:`get_topology_model`).
    """

    def __init__(self):
        # Running total of max hops per network, accumulated across the
        # repeated spatial-loop iterations handled by one NetworkAnalyzer.
        self.overall_max_hops: dict = {}

    def accumulate_max_hops(self, network, max_hops):
        """Add this loop's ``max_hops`` to ``network``'s running total and
        return the updated total.

        Each call to :meth:`NetworkAnalyzer.accumulate_child_result` (i.e., over
        a different iteration of a spatial loop) adds more to the max hops.
        """
        self.overall_max_hops[network] = (
            self.overall_max_hops.get(network, 0) + max_hops
        )
        return self.overall_max_hops[network]

    @abstractmethod
    def per_loop_transfer_cost(
        self,
        relevancy,
        *,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name: str,
    ) -> PerLoopTransferCost:
        """Return the :class:`PerLoopTransferCost` for moving ``volume`` of data across one
        spatial loop.

        Args:
            relevancy: The relevancy of the spatial loop's rank variable to the
                tensor (``Irrelevant``, ``Relevant``, or ``PartiallyRelevant``).
            shape_repeats: The number of iterations of this spatial loop.
            last_fanout: The fanout in this dimension among mapping nodes below
                (i.e., the stride).
            volume: The data volume (in actions) moved per destination.
            src_component: The flattened-arch component sourcing the data, used
                to query physical fanout/stride.
            dim_name: The name of the spatial dimension (e.g., ``X`` or ``Y``).
        """
        raise NotImplementedError


class MeshTopologyModel(TopologyModel):
    """Cost model for a mesh network.

    Data travels along one axis of the mesh. Multicast delivers a value to every
    point along the dimension; unicast delivers a distinct value to each point.
    When the source is physically distributed, data is bound as locally as
    possible across the physical buffers.
    """

    def per_loop_transfer_cost(
        self,
        relevancy,
        *,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name,
    ) -> PerLoopTransferCost:
        if isinstance(relevancy, Irrelevant):
            # The volume travels through link by link in one axis of the mesh
            # Distributed or not, the amount of total cost is the same.
            # However, the accesses now come from different physical memories
            total_cost = multicast_cost(shape_repeats, last_fanout) * volume
            max_hops = shape_repeats * last_fanout
            max_traffic = volume
        elif isinstance(relevancy, Relevant):
            # If distributed, then we bind data as locally as possible in the
            # physical buffers
            if src_component._get_physical_fanout_along(dim_name) > 1:
                physical_stride = src_component._get_physical_stride_along(dim_name)

                n_dsts_per_physical = min_nonzero(
                    # if last_fanout > physical_stride, set n_dst to 1, which results in 0 hops
                    # later (which is correct because the set of destinations always overlap
                    # the set of sources).
                    max_nonzero(physical_stride / last_fanout, 1),
                    shape_repeats,
                )
                n_activated_physical = max_nonzero(
                    shape_repeats * last_fanout / physical_stride, 1
                )
                total_cost = (
                    n_activated_physical
                    * unicast_cost(n_dsts_per_physical, last_fanout)
                    * volume
                )
                max_hops = min_nonzero(
                    (n_dsts_per_physical - 1) * last_fanout, physical_stride
                )
                max_traffic = (n_dsts_per_physical - 1) * volume
            else:
                total_cost = unicast_cost(shape_repeats, last_fanout) * volume
                max_hops = shape_repeats * last_fanout
                max_traffic = (shape_repeats - 1) * volume
        elif isinstance(relevancy, PartiallyRelevant):
            raise NotImplementedError()
        else:
            raise RuntimeError(f"unhandled relevancy type {relevancy}")

        return PerLoopTransferCost(
            total_cost=total_cost, max_hops=max_hops, max_traffic=max_traffic
        )


class AllToAllTopologyModel(TopologyModel):
    """Cost model for an all-to-all network using a switch (e.g. NVLink).

    Every node connects to every other node through a switch, so any
    source reaches any destination in one hop regardless of

    Physical stride is irrelevant, so ``last_fanout`` and physical distribution
    are not used.
    """

    HOPS_PER_TRANSFER = 1
    """Hops charged for one source-to-destination transfer across the switch.
    One switch traversal is treated as a single hop; the per-hop energy and
    latency come from the network component's ``hops`` action."""

    def per_loop_transfer_cost(
        self,
        relevancy,
        *,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name,
    ) -> PerLoopTransferCost:
        hops = self.HOPS_PER_TRANSFER

        # n - 1 other instances each receive the data across the switch. The
        # source already holds it (the set of destinations overlaps the set of
        # sources), so it needs no transfer to itself.
        n_dsts = shape_repeats - 1

        if isinstance(relevancy, (Irrelevant, Relevant)):
            # Same delivery count (and hence energy) whether the data is shared
            # (multicast) or distinct per instance (unicast): each of the n - 1
            # destinations is one switch traversal away.
            total_cost = n_dsts * hops * volume
            # Every route is a single switch traversal, independent of distance.
            max_hops = hops
            if isinstance(relevancy, Irrelevant):
                # Multicast: the switch replicates, so each link carries the
                # value at most once.
                max_traffic = volume
            else:
                # Unicast: the source's uplink to the switch carries all n - 1
                # distinct messages, making it the most congested link.
                max_traffic = n_dsts * volume
        elif isinstance(relevancy, PartiallyRelevant):
            raise NotImplementedError()
        else:
            raise RuntimeError(f"unhandled relevancy type {relevancy}")

        return PerLoopTransferCost(
            total_cost=total_cost, max_hops=max_hops, max_traffic=max_traffic
        )


# Registry of topology models
TOPOLOGY_MODELS: dict[TopologySpec, type[TopologyModel]] = {
    TopologySpec.MESH: MeshTopologyModel,
    TopologySpec.ALL_TO_ALL: AllToAllTopologyModel,
}


def get_topology_model(topology) -> TopologyModel:
    """Construct a fresh :class:`TopologyModel` for the given topology."""
    return TOPOLOGY_MODELS[topology]()


class NetworkAnalyzer:
    def __init__(self, network_stats, info: AnalysisInfo, einsum_name, node: Spatial):
        self.network_stats = network_stats
        # These don't change across calls to accumulate_child_result.
        self.info = info
        self.einsum_name = einsum_name
        self.node = node
        # Each network gets its own topology model, since different networks may
        # have different topologies. Models are constructed lazily, the first
        # time a network needs costing, and reused for the analyzer's lifetime so
        # their accumulated max hops persist.
        self.topology_models: dict = {}

    def _get_topology_model(self, network, topology) -> TopologyModel:
        if network not in self.topology_models:
            self.topology_models[network] = get_topology_model(topology)
        return self.topology_models[network]

    def accumulate_child_result(
        self,
        child_result: SymbolicAnalysisOutput,
        shape_repeats,
        child_shape,
    ):
        """This function is called for every repeated shape."""
        flattened_arch = self.info.job.flattened_arch

        for network, child_network_stats in child_result.network_stats.items():
            src_component = flattened_arch[network.source.level]
            if network not in self.network_stats:
                self.network_stats[network] = NetworkStats()
            accumulated_network_stats = self.network_stats[network]

            # We only need to update the summary if the spatial loop is for
            # a component higher than the network of interest
            if flattened_arch.is_above(self.node.component, network.component):
                accumulated_network_stats.total_hops += (
                    child_network_stats.total_hops * shape_repeats
                )
                accumulated_network_stats.max_hops = max_nonzero(
                    accumulated_network_stats.max_hops,
                    child_network_stats.max_hops,
                )
                for k, v in child_network_stats.max_traffic.items():
                    accumulated_network_stats.max_traffic[k] = max_nonzero(
                        accumulated_network_stats.max_traffic.get(k, 0), v
                    )
                continue

            volume = self._get_data_volume(network, child_shape)

            relevancy = self.info.tensor_to_relevancy[network.tensor][
                self.node.rank_variable
            ]

            # The fanout in this dimension in mapping nodes below, i.e., the stride
            last_fanout = child_result.fanout.get(
                (self.node.component, self.einsum_name), {}
            )
            last_fanout = last_fanout.get(self.node.name, 1)

            topology_model = self._get_topology_model(
                network, flattened_arch[network.component].topology
            )
            per_loop_transfer_cost = topology_model.per_loop_transfer_cost(
                relevancy,
                shape_repeats=shape_repeats,
                last_fanout=last_fanout,
                volume=volume,
                src_component=src_component,
                dim_name=self.node.name,
            )

            overall_max_hops = topology_model.accumulate_max_hops(
                network, per_loop_transfer_cost.max_hops
            )

            accumulated_network_stats.total_hops += (
                per_loop_transfer_cost.total_cost
                + child_network_stats.total_hops * shape_repeats
            )
            accumulated_network_stats.max_hops = max_nonzero(
                accumulated_network_stats.max_hops,
                overall_max_hops + child_network_stats.max_hops,
            )
            accumulated_network_stats.max_traffic[self.node.name] = max_nonzero(
                accumulated_network_stats.max_traffic.get(self.node.name, 0),
                per_loop_transfer_cost.max_traffic
                + child_network_stats.max_traffic.get(self.node.name, 0),
            )

        overall_max_hops = {}
        for model in self.topology_models.values():
            overall_max_hops.update(model.overall_max_hops)
        return overall_max_hops

    def _get_data_volume(self, network, child_shape):
        info = self.info
        einsum_name = self.einsum_name
        flattened_arch = info.job.flattened_arch
        projection = info.einsum_tensor_to_projection[(einsum_name, network.tensor)]
        component_object = flattened_arch[network.component]
        workload_bpv = info.job.einsum.tensor_accesses[network.tensor].bits_per_value
        bits_per_value = component_object.bits_per_value.get(
            network.tensor, workload_bpv
        )
        bits_per_action = component_object.bits_per_action
        if bits_per_action is not None:
            actions_per_value = bits_per_value / bits_per_action
        else:
            actions_per_value = bits_per_value
        volume = (
            compute_dense_tile_occupancy(projection, child_shape) * actions_per_value
        )
        return volume


def multicast_cost(n_dsts, stride):
    """Returns total hops of multicast along a dimension."""
    return (n_dsts - 1) * stride


def unicast_cost(n_dsts, stride):
    """Returns total hops of unicast along a dimension."""
    # Cost of unicast is the cost of delivering to each point in
    # the dimension with shape as stride
    return arithmetic_sum(n_dsts - 1) * stride


def arithmetic_sum(n):
    return 0.5 * (n + 1) * n
