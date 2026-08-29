from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from accelforge.frontend.mapping import Spatial
from accelforge.frontend.arch.components import TopologySpec
from accelforge.frontend.renames import RANK_DUPLICATE
from accelforge.frontend._workload_isl._symbolic import (
    compute_dense_tile_occupancy,
    Irrelevant,
    Relevant,
    PartiallyRelevant,
)

from accelforge.util._sympy.broadcast_max import (
    MaxGeqZero,
    max_nonzero,
    min_nonzero,
    min_take_zero,
)

from ._binding import binding_is_local
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

    reduction_tiles: Any = 0
    """Whole tiles that have to be read out of one memory and written into
    another to combine the partial sums this loop leaves scattered, summed over
    every instance. Zero unless the loop is a spatial reduction.

    The memories at either end are charged for this; ``total_cost`` is what the
    network charges for carrying it, and the two are not the same number -- a
    tile crossing a mesh pays a hop per link but is still read once and written
    once."""

    max_reduction_tiles_per_unit: Any = 0
    """The most of that work any one instance does. An all-reduce spreads it
    evenly; a reduction into a single owner lands all of it on that owner."""


@dataclass
class PhysicalLayout:
    n_dsts: Any
    """Destinations along this dimension, i.e. iterations of the spatial loop."""
    delta_between_dsts: Any
    """Array coordinates between neighbouring destinations."""

    @property
    def n_dsts_per_src(self):
        raise NotImplementedError()

    @property
    def is_fully_local(self):
        """Whether dst and src completely overlap."""
        return min_take_zero(self.n_dsts_per_src - 1, 1)


class SingularLayout(PhysicalLayout):
    @property
    def n_srcs(self):
        return 1

    @override
    @property
    def n_dsts_per_src(self):
        return self.n_dsts


@dataclass
class DistributedLayout(PhysicalLayout):
    """How the physical instances of a distributed source line up with the
    destinations of one spatial loop, along one dimension.

    Both are evenly spaced across the same span of array coordinates: the
    destinations ``last_fanout`` apart, the instances ``physical_stride`` apart.
    One destination per instance therefore sits with it and can read from it
    without going onto the network at all.
    """

    n_dsts: Any
    """Destinations along this dimension, i.e. iterations of the spatial loop."""
    delta_between_dsts: Any
    """Array coordinates between neighbouring destinations."""
    n_srcs: Any
    """Physical instances of the source memory taking part."""
    delta_between_srcs: Any
    """Array coordinates between neighbouring instances."""

    @property
    def n_dsts_per_src(self):
        if self.delta_between_srcs == 1:
            return 1
        else:
            return min_nonzero(
                # If delta_between_dsts > delta_between_srcs, then we set to 1.
                max_nonzero(self.delta_between_srcs / self.delta_between_dsts, 1),
                self.n_dsts,
            )

    @property
    def single_instance(self) -> bool:
        """Whether only one instance is active. It then holds everything there is
        to hold, however the mapping binds the tensor, so there is nothing to
        scatter across. False when the count is a symbol the model cannot resolve
        here, which leaves the general formulas to handle it."""
        return self.n_srcs == 1

    @property
    def destination_span(self):
        """Array coordinates from the first destination to the last."""
        return multicast_cost(self.n_dsts, self.delta_between_dsts)

    @property
    def instance_span(self):
        """Array coordinates from the first source instance to the last. Longer
        than the destinations' span when the source is distributed more finely
        than the spatial loop fans out."""
        return multicast_cost(self.n_srcs, self.delta_between_srcs)


type PhysicalLayout = SingularLayout | DistributedLayout


def physical_layout(
    src_component, dim_name, shape_repeats, last_fanout
) -> PhysicalLayout:
    """The :class:`PhysicalLayout` of ``src_component`` along ``dim_name``, or
    None if it has only one instance there and so holds everything wherever the
    mapping says its tensor is bound."""
    if src_component._get_physical_fanout_along(dim_name) <= 1:
        return SingularLayout(
            n_dsts=shape_repeats,
            delta_between_dsts=last_fanout,
        )

    physical_stride = src_component._get_physical_stride_along(dim_name)
    return DistributedLayout(
        n_dsts=shape_repeats,
        delta_between_dsts=last_fanout,
        n_srcs=ratio_geq_one(shape_repeats * last_fanout, physical_stride),
        delta_between_srcs=physical_stride,
    )


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
        src_bound_rank,
        dst_relevancy,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name: str,
        is_reduced_output: bool = False,
    ) -> PerLoopTransferCost:
        """Return the :class:`PerLoopTransferCost` for moving ``volume`` of data across one
        spatial loop.

        Args:
            src_bound_rank: The rank the mapping's binding partitions the source
                along for this loop, ``RANK_DUPLICATE`` where it replicates the
                tensor instead, or None where the source is not distributed (see
                :mod:`._binding`).
            dst_relevancy: The relevancy of the spatial loop's rank variable to the
                tensor (``Irrelevant``, ``Relevant``, or ``PartiallyRelevant``).
            shape_repeats: The number of iterations of this spatial loop.
            last_fanout: The fanout in this dimension among mapping nodes below
                (i.e., the stride).
            volume: The data volume (in actions) moved per destination.
            src_component: The flattened-arch component sourcing the data, used
                to query physical fanout/stride.
            dim_name: The name of the spatial dimension (e.g., ``X`` or ``Y``).
            is_reduced_output: Whether the tensor is an output that requires a
                reduction.
        """
        raise NotImplementedError


class MeshTopologyModel(TopologyModel):
    """Cost model for a mesh network.

    Data travels along one axis of the mesh. Multicast delivers a value to every
    point along the dimension; unicast delivers a distinct value to each point.

    When the source is physically distributed, the binding decides how far the
    data has to travel. A locally-bound source serves each destination from the
    instance it sits among, so the line breaks into independent groups; one bound
    along another rank has to serve every destination from every instance, and
    the mesh pays the distance to each.

    The closed forms are exact against an enumeration of the traffic whenever the
    destinations and the instances share a spacing. Where they do not, the
    scattered ones thin one evenly-spaced grid onto the other and under-count by
    a few percent at the boundary; see ``TestMeshAgainstBruteForce``.

    An output the einsum accumulates into runs the line the other way. A loop
    over a rank the output does not project onto leaves a partial copy of the
    tile at every destination, and they have to be summed -- which on a mesh is
    the multicast played backwards, at the same cost, because a node adds its own
    partial to what passes through before forwarding it. See
    :meth:`_spatial_reduction`.
    """

    def per_loop_transfer_cost(
        self,
        src_bound_rank,
        dst_relevancy,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name,
        is_reduced_output: bool = False,
    ) -> PerLoopTransferCost:
        layout = physical_layout(src_component, dim_name, shape_repeats, last_fanout)
        data_is_local = binding_is_local(src_bound_rank, dst_relevancy)

        if is_reduced_output and isinstance(dst_relevancy, Irrelevant):
            return self._spatial_reduction(src_bound_rank, layout, volume)

        match dst_relevancy, layout:
            case Irrelevant(), SingularLayout():
                # The volume travels through link by link in one axis of the mesh
                # Distributed or not, the amount of total cost is the same.
                # However, the accesses now come from different physical memories
                total_cost = multicast_cost(layout.n_dsts, layout.delta_between_dsts) * volume
                max_hops = (layout.n_dsts - 1) * layout.delta_between_dsts
                max_traffic = volume * layout.is_fully_local
            case Irrelevant(), DistributedLayout():
                if data_is_local:
                    # Every instance holds the whole tile, so none of them has to send
                    # past the destinations it sits among: the line breaks up into
                    # n_physical independent multicasts of n_dsts_per_physical points.
                    total_cost, max_hops, max_traffic = self._local_multicast(
                        layout, volume
                    )
                else:
                    total_cost, max_hops, max_traffic = self._scattered_multicast(
                        layout, volume
                    )
            case Relevant(), SingularLayout():
                total_cost = unicast_cost(layout.n_dsts, layout.delta_between_dsts) * volume
                max_hops = (layout.n_dsts-1) * layout.delta_between_dsts
                max_traffic = (layout.n_dsts - 1) * volume
            case Relevant(), DistributedLayout():
                if data_is_local or layout.single_instance:
                    total_cost, max_hops, max_traffic = self._local_unicast(
                        layout, volume
                    )
                else:
                    total_cost, max_hops, max_traffic = self._scattered_unicast(
                        layout, volume
                    )
            case PartiallyRelevant(), _:
                raise NotImplementedError(f"Partially relevant binding unimplemented")
            case _, _:
                raise RuntimeError(f"unhandled relevancy type {dst_relevancy}")

        return PerLoopTransferCost(
            total_cost=total_cost, max_hops=max_hops, max_traffic=max_traffic
        )

    @staticmethod
    def _spatial_reduction(src_bound_rank, layout: "PhysicalLayout", volume):
        """Sum the partial copies of the tile that a loop over a reduced rank
        leaves at every destination along the line.

        A reducing node adds its own partial to what arrives before passing it
        on, so one tile's worth crosses each link and the whole line costs the
        same as the multicast that runs the other way -- linear in the span, not
        quadratic as delivering each partial separately to a sink would be. It is
        also why the busiest link carries one tile however many destinations there
        are, where the all-to-all switch, which cannot add anything in flight,
        funnels every partial onto the owner's link.

        The binding decides whether the line is walked once or twice:

        - Replicated (``RANK_DUPLICATE``): every instance is meant to hold the
          finished tile, so the sum is carried back out and the reduction is an
          all-reduce -- the span twice over. Note this spans the *whole* line;
          :meth:`_local_multicast`'s break-up into independent groups is wrong
          here, because a partial in one group still has to reach the instances
          in all the others. For an input this same binding is the free case.
        - Partitioned along another rank, or a single instance: the sum is formed
          once where the finished tile lives and stays there. Each of the
          ``n_srcs`` owners reduces its own share along the line, and n_srcs
          reductions of a 1/n_srcs share cost the same wire as one reduction of
          the whole tile -- the mirror of the argument in
          :meth:`_scattered_multicast`.
        """
        span = multicast_cost(layout.n_dsts, layout.delta_between_dsts)
        # One partial per destination, all but one of which has to move.
        crossing = layout.n_dsts - 1
        trips = 2 if src_bound_rank == RANK_DUPLICATE else 1

        return PerLoopTransferCost(
            total_cost=trips * span * volume,
            max_hops=trips * span,
            max_traffic=trips * volume * min_take_zero(crossing, 1),
            # What the memories at either end owe is a count of tiles read out and
            # written in, and does not depend on how far they travelled.
            reduction_tiles=trips * crossing,
            max_reduction_tiles_per_unit=(
                2 * crossing / layout.n_dsts
                if src_bound_rank == RANK_DUPLICATE
                else crossing / layout.n_srcs
            ),
        )

    @staticmethod
    def _local_multicast(layout: "PhysicalLayout", volume):
        """Hand every destination the same tile, from the instance it sits among.

        Each instance holds a whole copy, so it multicasts to its own
        destinations and no further. Returns ``(total_cost, max_hops,
        max_traffic)``.
        """
        span = multicast_cost(layout.n_dsts_per_src, layout.delta_between_dsts)
        return (
            layout.n_srcs * span * volume,
            span,
            layout.is_fully_local * volume,
        )

    @staticmethod
    def _scattered_multicast(layout: "PhysicalLayout", volume):
        """Hand every destination the same tile, from a source partitioned along
        one of the tensor's ranks rather than replicated.

        Every instance then holds a ``1 / n_physical`` share that every
        destination needs, so each multicasts its share across all of them.
        n_physical multicasts of a 1/n_physical share is the same wire as one
        multicast of the whole tile -- analysis.ipynb makes the same point the
        other way round, that on a mesh an all-gather and a broadcast cost the
        same. Returns ``(total_cost, max_hops, max_traffic)``.
        """
        # An instance sitting outside the destinations has to reach across the
        # gap as well. Averaged over the instances that do, the gap contributes
        # its own length again in the same proportion as the instances that
        # overshoot -- hence the square.
        overshoot = MaxGeqZero(layout.instance_span - layout.destination_span)
        reach = layout.destination_span + overshoot**2 / (
            2 * max_nonzero(layout.instance_span, 1)
        )
        return (
            reach * volume,
            (layout.n_dsts - 1) * layout.delta_between_dsts,
            volume,
        )

    @staticmethod
    def _local_unicast(layout: "PhysicalLayout", volume):
        """Hand every destination its own tile, from the instance it sits among.

        Returns ``(total_cost, max_hops, max_traffic)``.
        """
        n_dsts = layout.n_dsts_per_src
        return (
            layout.n_srcs * unicast_cost(n_dsts, layout.delta_between_dsts) * volume,
            min_nonzero((n_dsts - 1) * layout.delta_between_dsts, layout.delta_between_srcs),
            (n_dsts - 1) * volume,
        )

    @staticmethod
    def _scattered_unicast(layout: "PhysicalLayout", volume):
        """Hand every destination its own tile, from a source partitioned along a
        rank other than the one the destinations are spread over.

        A destination's tile then straddles all ``n_physical`` instances in equal
        parts, so instead of reading ``volume`` from the instance it sits with it
        collects ``volume / n_physical`` from every one of them, and the mesh pays
        the distance to each. Returns ``(total_cost, max_hops, max_traffic)``.
        """
        # Destinations and instances are two evenly-spaced grids over the same
        # span, so the sum of the distances between every pair of them is the
        # pairwise sum over the finer of the two grids, thinned by how sparse each
        # set is on it. Exact when one spacing divides the other, which is how the
        # analyzer lays them out.
        n_grid = max_nonzero(layout.n_dsts, layout.n_srcs)
        spacing = min_nonzero(layout.delta_between_dsts, layout.delta_between_srcs)
        between_all_pairs = pairwise_cost(n_grid, spacing)

        return (
            # ...times a chunk each, of which every destination collects n_physical.
            between_all_pairs * layout.n_dsts * volume / n_grid**2,
            # The farthest chunk crosses the whole span.
            multicast_cost(n_grid, spacing),
            # The middle link separates about half the instances from about half
            # the destinations, and carries a chunk for each such pair in each
            # direction: (n_physical / 2)(shape_repeats / 2) chunks.
            layout.n_dsts * volume / 4,
        )


class AllToAllTopologyModel(TopologyModel):
    """Cost model for an all-to-all network using a switch (e.g. NVLink).

    Every node connects to every other node through a switch, so any
    source reaches any destination in one hop regardless of distance.
    """

    HOPS_PER_TRANSFER = 1
    """Hops charged for one source-to-destination transfer across the switch.
    One switch traversal is treated as a single hop; the per-hop energy and
    latency come from the network component's ``hops`` action."""

    def per_loop_transfer_cost(
        self,
        src_bound_rank,
        dst_relevancy,
        shape_repeats,
        last_fanout,
        volume,
        src_component,
        dim_name,
        is_reduced_output: bool = False,
    ) -> PerLoopTransferCost:
        hops = self.HOPS_PER_TRANSFER
        data_is_local = binding_is_local(src_bound_rank, dst_relevancy)

        layout = physical_layout(src_component, dim_name, shape_repeats, last_fanout)
        if layout is None:
            # A single instance holds the whole tile and serves every destination.
            n_physical = 1
            n_dsts_per_physical = shape_repeats
            data_is_local = True
        else:
            n_physical = layout.n_srcs
            n_dsts_per_physical = layout.n_dsts_per_src

        if is_reduced_output and isinstance(dst_relevancy, Irrelevant):
            return self._spatial_reduction(
                src_bound_rank, shape_repeats, n_physical, volume, hops
            )

        # One destination per instance sits with it, and reads whatever that
        # instance holds for it without going near the switch.
        n_co_located = min_nonzero(shape_repeats, n_physical)
        if data_is_local:
            # Such a destination finds its whole tile in the instance it sits with.
            delivered_locally = n_co_located
        else:
            # Its tile is spread over every instance instead, so all it finds next
            # door is the one share of it that instance holds.
            delivered_locally = n_co_located / n_physical

        # Both counted in units of `volume`: what the destinations ask for between
        # them, less what never has to cross.
        crossing = shape_repeats - delivered_locally
        # 1 while anything crosses at all, 0 once every read is local.
        crosses_at_all = min_take_zero(crossing, 1)

        if isinstance(dst_relevancy, (Irrelevant, Relevant)):
            # Same delivery count (and hence energy) whether the data is shared
            # (multicast) or distinct per instance (unicast): whatever a
            # destination cannot read locally is one switch traversal away,
            # wherever it comes from.
            total_cost = crossing * hops * volume
            # Every route is a single switch traversal, independent of distance.
            max_hops = crosses_at_all * hops
            if isinstance(dst_relevancy, Irrelevant):
                # Multicast: the switch replicates, so each link carries the
                # value at most once.
                max_traffic = crosses_at_all * volume
            elif data_is_local:
                # Unicast: an instance's uplink to the switch carries a distinct
                # message for each destination it serves but does not sit with,
                # making it the most congested link.
                max_traffic = (n_dsts_per_physical - 1) * volume
            else:
                # Unicast from a source partitioned along another rank: an
                # instance's uplink carries its one share of the tile for every
                # destination except the one it sits with.
                max_traffic = (shape_repeats - 1) * volume / n_physical
        elif isinstance(dst_relevancy, PartiallyRelevant):
            raise NotImplementedError()
        else:
            raise RuntimeError(f"unhandled relevancy type {dst_relevancy}")

        return PerLoopTransferCost(
            total_cost=total_cost, max_hops=max_hops, max_traffic=max_traffic
        )

    @staticmethod
    def _spatial_reduction(src_bound_rank, shape_repeats, n_physical, volume, hops):
        # One partial per destination, all but one of which has to move.
        crossing = shape_repeats - 1
        # 1 while anything crosses at all, 0 once the loop has a single iteration.
        crosses_at_all = min_take_zero(crossing, 1)

        # Implemented as reduce-scatter + all-gather
        if src_bound_rank == RANK_DUPLICATE:
            return PerLoopTransferCost(
                total_cost=2 * crossing * hops * volume,
                max_hops=2 * crosses_at_all * hops,
                max_traffic=2 * crosses_at_all * volume,
                # Both passes are read out of one memory and written into the
                # next, and the instances share that evenly between them.
                reduction_tiles=2 * crossing,
                max_reduction_tiles_per_unit=2 * crossing / shape_repeats,
            )

        # No replication, reduced at the destination
        return PerLoopTransferCost(
            total_cost=crossing * hops * volume,
            max_hops=crosses_at_all * hops,
            # n_physical is >= 1 by construction: SingularLayout reports one
            # instance and DistributedLayout's ratio_geq_one floors at one.
            max_traffic=crossing * volume / n_physical,
            reduction_tiles=crossing,
            # Only the owners combine anything, so a single owner does all of it.
            max_reduction_tiles_per_unit=crossing / n_physical,
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

            # We only charge a network for a spatial loop it actually has links
            # along: one above it replicates the whole network rather than
            # putting traffic on it, and one below it may fan out over wiring
            # private to an array unit. Either way, the child's stats still
            # carry up.
            if flattened_arch.is_above(
                self.node.component, network.component
            ) or not self._has_links_along_dim(network):
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
                accumulated_network_stats.total_reduction_values += (
                    child_network_stats.total_reduction_values * shape_repeats
                )
                accumulated_network_stats.max_per_unit_reduction_values = max_nonzero(
                    accumulated_network_stats.max_per_unit_reduction_values,
                    child_network_stats.max_per_unit_reduction_values,
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
            src_bound_rank = self.info.bindings.rank_by_loop.get(
                (id(self.node), network.source.level, network.tensor)
            )
            per_loop_transfer_cost = topology_model.per_loop_transfer_cost(
                src_bound_rank,
                relevancy,
                shape_repeats=shape_repeats,
                last_fanout=last_fanout,
                volume=volume,
                src_component=src_component,
                dim_name=self.node.name,
                is_reduced_output=self._is_reduced_output(network),
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
            # What the memories at either end owe for the combining, in values --
            # which is what a buffet counts its accesses in, not the network's
            # own actions.
            occupancy = self._get_tile_occupancy(network, child_shape)
            accumulated_network_stats.total_reduction_values += (
                per_loop_transfer_cost.reduction_tiles * occupancy
                + child_network_stats.total_reduction_values * shape_repeats
            )
            accumulated_network_stats.max_per_unit_reduction_values = max_nonzero(
                accumulated_network_stats.max_per_unit_reduction_values,
                per_loop_transfer_cost.max_reduction_tiles_per_unit * occupancy
                + child_network_stats.max_per_unit_reduction_values,
            )
            # Traffic the child puts on the links of *other* dimensions crosses
            # them once per instance of this loop, and the instances have links of
            # their own, so it carries through unchanged. Only this dimension's
            # own links also carry what this loop itself contributes.
            own_dim = self.node.name
            for dim, traffic in child_network_stats.max_traffic.items():
                if dim == own_dim:
                    continue
                accumulated_network_stats.max_traffic[dim] = max_nonzero(
                    accumulated_network_stats.max_traffic.get(dim, 0), traffic
                )
            accumulated_network_stats.max_traffic[own_dim] = max_nonzero(
                accumulated_network_stats.max_traffic.get(own_dim, 0),
                per_loop_transfer_cost.max_traffic
                + child_network_stats.max_traffic.get(own_dim, 0),
            )

        overall_max_hops = {}
        for model in self.topology_models.values():
            overall_max_hops.update(model.overall_max_hops)
        return overall_max_hops

    def _has_links_along_dim(self, network) -> bool:
        """Whether ``network`` has links along the dimension this analyzer's
        spatial loop fans out over.

        A ``!Network`` declared inside an ``!Array`` is that array's
        interconnect: its links run between array units, along the array's own
        spatial dimensions and no others. A fanout declared by a component
        *inside* a unit -- a PE grid under the array's buffer, say -- is wiring
        private to that unit, so its traffic never reaches the array's network
        even though the network sits above it in the flattened hierarchy.

        A network declared outside any array has no array to scope it to, so it
        carries every fanout below it.
        """
        network_component = self.info.job.flattened_arch[network.component]
        array = network_component._physical_spatial_component
        return array is None or array == self.node.component

    def _is_reduced_output(self, network) -> bool:
        """Whether this network carries an output that the einsum accumulates
        into. A spatial loop over a rank such a tensor does not project onto is
        then a reduction, not a broadcast. A copy operation writes its output
        without accumulating, so it never reduces."""
        einsum = self.info.workload.einsums[self.einsum_name]
        return (
            network.tensor in einsum.output_tensor_names
            and not self.info.is_copy_operation
        )

    def _get_tile_occupancy(self, network, child_shape):
        """The tile one destination asks for, in values.

        That is what a buffet counts its accesses in; :meth:`_get_data_volume`
        turns it into the network's own actions, which the network component's
        ``bits_per_action`` may make something else entirely.
        """
        projection = self.info.einsum_tensor_to_projection[
            (self.einsum_name, network.tensor)
        ]
        return compute_dense_tile_occupancy(projection, child_shape)

    def _get_data_volume(self, network, child_shape):
        info = self.info
        flattened_arch = info.job.flattened_arch
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
        return self._get_tile_occupancy(network, child_shape) * actions_per_value


def multicast_cost(n_dsts, stride):
    """Returns total hops of multicast along a dimension."""
    return (n_dsts - 1) * stride


def unicast_cost(n_dsts, stride):
    """Returns total hops of unicast along a dimension."""
    # Cost of unicast is the cost of delivering to each point in
    # the dimension with shape as stride
    return arithmetic_sum(n_dsts - 1) * stride


def pairwise_cost(n_dsts, stride):
    """Total hops of one unit of data travelling between every pair of
    `n_dsts` points along a dimension, i.e. `sum |i - j| * stride`, which equals
    `n(n^2 - 1) / 3 * stride`."""
    return n_dsts * (n_dsts**2 - 1) / 3 * stride


def arithmetic_sum(n):
    return 0.5 * (n + 1) * n


def ratio_geq_one(numerator, denominator):
    """
    Returns max(numerator/denominator, 1) with speedup if denominator == 1.
    Assumes numerator >= 1 and denominator >= 1.
    """
    if denominator == 1:
        return numerator
    else:
        return max_nonzero(numerator / denominator, 1)
