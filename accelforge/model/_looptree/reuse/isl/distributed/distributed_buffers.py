"""
Models for handling calculating the cost of a Workload on distributed buffer
architectures.
"""

import logging

import islpy as isl

from accelforge.frontend.mapping import MappingNode
from accelforge.model._looptree.reuse.isl.isl_functions import dim_projector_mask
from accelforge.model._looptree.reuse.isl.mapping_to_isl import DUMP_ISL_IR
from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import Fill, Occupancy
from accelforge.model._looptree.reuse.isl.spatial import (
    Reads,
    Transfers,
    TransferInfo,
    TransferModel,
)


def identify_mesh_casts(
    src_occupancy: isl.Map, dst_fill: isl.Map, dist_fn: isl.Map
) -> isl.Map:
    """
    Given srcs with data, fills to destinations, and a distance function, identify per data
    the srcs delivering that data to dsts.

    Parameters
    ----------
    src_occupancy:
        An isl.Map of the form { [src] -> [data] } corresponding to the data held
        at the buffer at space `src`.
    dst_fill:
        An isl.Map of the form { [dst] -> [data] } corresponding to the data requested
        at the element at space `dst`.
    dist_fn:
        A distance function { [src -> dst] -> [hops] } that accepts two points in
        space, corresponding to the `src` and `dst`, and returns the distance
        between the two points in terms of `hops`, a quantized atomic distance of
        data transmission cost.
    
    Returns
    -------
    { [data] -> [dst -> src] } where { [dst] -> [data] } and { [src] -> [data] } are in
    `src_occupancy` and `dst_fill` respectively, and where `[dst -> src]` is the infimum of
    `dst_fn(src, dst), ∀ src, dst s.t. { [src] -> [data] } ∈ `src_occupancy` and
    `{ [dst] -> [data] }` ∈ `dst_fill`.
    """
    # Makes { [dst -> data] -> [dst -> data] }
    fill_to_fill: isl.Map = dst_fill.wrap().identity()
    if DUMP_ISL_IR:
        logging.info(f"fill_to_fill: {fill_to_fill}")

    # Inverts src_occupancy s.t. data -> src.
    # i.e. { [xs, ys] -> [d0, d1] } to { [d0, d1] -> [xs, ys] }
    data_presence: isl.Map = src_occupancy.reverse()

    # { [dst -> data] -> [dst -> src] } where src contains data.
    fills_to_matches: isl.Map = fill_to_fill.uncurry( # { [[dst -> data] -> dst] -> data }
                                ).apply_range(data_presence # { [[dst -> data] -> dst] -> src }
                                ).curry() # { [[dst -> data] -> [dst -> src] }
    if DUMP_ISL_IR:
        logging.info(f"fills_to_matches: {fills_to_matches}")

    # Calculates the distance of a fill to the nearest src satisfying the fill.
    # { [dst -> data] -> [dist] }
    fill_min_dist: isl.Map = fills_to_matches.apply_range(dist_fn).lexmin()
    # Isolates the relevant minimal pairs.
    # { [dst -> data] -> [dst -> src] :.dst -> src is minimized distance }
    minimal_pairs: isl.Map = fill_min_dist.apply_range(
        # Note: Need to match fill -> min_dist with min_dist -> [fill -> match] as lexmin over
        # fill and match will minimize distance over the tuple (src, dst, data), but that
        # overconstrains the optimization as we want to minimize over distance (dst, data)
        # only for all src.
        fills_to_matches.range_map().apply_range(dist_fn).reverse()
    ).range().unwrap()
    if DUMP_ISL_IR:
        logging.info(f"minimal_pairs: {minimal_pairs}")

    # Isolates the multicast networks.
    # { [data] -> [dst -> src] : dst -> src is minimized distance }
    multicast_networks: isl.Map = minimal_pairs.curry().range().unwrap()
    # Devolves to a single source if multiple sources per domain point.
    multicast_networks = multicast_networks.uncurry().lexmin().curry()

    return multicast_networks


def calculate_extents_per_dim(mcns: isl.Map) -> list[isl.PwAff]:
    """
    Parameters
    ----------
    mcns:
        Mesh cast-networks, or networks in which all dsts per data are grouped with
        the closest src containing the data.

    Returns
    -------
    A list of `isl.PwAff` that gives the max extent (length) along dim_i per mcn,
    where i is the i-th `isl.PwAff`.

    Preconditions
    -------------
    `mcns` were generated with a Manhattan distance `dst_fn` by `identify_mesh_casts`
    s.t. all dimensions are orthogonal to each other in a metric space, where each
    unit movement in a dimension counts as 1 hop.

    We also assume `dst_fn` is translationally invariant (i.e., ∀src, dst,
    src', dst' ∈ space, if |src - dst| = |src' - dst'|,
    dst_fn(src, dst) = dst_fn(src', dst').
    """
    # Makes mcns from { [data] -> [dst -> src] } to { [data -> src] -> [dst] }
    potential_srcs: isl.Map = mcns.range_reverse().uncurry()
    # Sources are part of the extents, so we union it with the destinations.
    # { [data -> src] -> [src] }
    srcs: isl.Map = potential_srcs.domain().unwrap().range_map()
    # { [data -> src] -> [spacetime] }
    casting_extents: isl.Map = srcs.union(potential_srcs)

    # Projects away all dimensions but one to find their extent for hypercube.
    dims: int = potential_srcs.range_tuple_dim()
    # Creates a mask of what to project out.
    project_out_mask: list[bool] = [True] * dims
    dim_extents: list[isl.PwAff] = [None] * dims

    # Gets the extents of all dimensions
    for noc_dim in range(dims):
        # Project out all the dimensions of the output besides noc_dim.
        project_out_mask[noc_dim] = False
        # { [spacetime] -> [dimension] }
        extent_mapper: isl.Map = dim_projector_mask(
            casting_extents.range().get_space(), project_out_mask
        ).reverse()
        dim_extent_space: isl.Map = casting_extents.apply_range(extent_mapper)
        project_out_mask[noc_dim] = True

        # Finds max(noc_dim) - min(noc_dim) for each [data -> src]
        max_extent: isl.PwAff = dim_extent_space.dim_max(0)
        min_extent: isl.PwAff = dim_extent_space.dim_min(0)

        # Subtracts the max from the min to get the extent per [data -> src]
        dim_extents[noc_dim] = max_extent.sub(min_extent).coalesce()

    return dim_extents


class HypercubeMulticastModel(TransferModel):
    """
    Does distributed multicasting a mesh using worst-case multicasting
    behavior by assuming all multicasts are broadcasting to the convex
    hypercube that encapsulates all their destinations and sources.
    """

    def __init__(self, dist_fn: isl.Map):
        """
        Initializes the HypercubeMulticastModel with the distance function
        over the metric space.

        Because we are using calculate_extents_per_dim(mcns), we inherit the
        following requirements:
        `dst_fn` holds all dimensions are orthogonal to each other in a metric space,
        where each unit movement in a dimension counts as 1 hop.

        We also assume `dst_fn` is translationally invariant (i.e., ∀src, dst,
        src', dst' ∈ space, if |src - dst| = |src' - dst'|,
        dst_fn(src, dst) = dst_fn(src', dst').
        """
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the spatial transfers."

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Currently,
            we rely on dist_fn to deal with this rather than buffer.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        Fills that were fulfilled, Fills that were unfilled, and parent reads per
        position in spacetime. Then, gets hops per timestep.
        """
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_mesh_cast_hypercube(mcs)

        # TODO: Read once from all buffers, assert that
        # card(mcs) == tensor_size * duplication factor
        num_meshcasts: int = mcs.card()
        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(fills.map_)),
            hops=result,
            link_transfer=True,
        )

    def _cost_mesh_cast_hypercube(self, mcns: isl.Map) -> int:
        """
        Given a multicast_network, calculate the hypercube.

        Parameters
        ----------
        mcns:
            Multicast networks grouped together by [srcs -> data] fulfilling
            [dsts -> data], where there is at least 1 src and 1 dst in each mcn.
        
        Returns
        -------
        The upperbound of doing all the multicasts specified by the multicast
        networks, assuming they cast only to the convex space of the network.

        Preconditions
        -------------
        Because we are using calculate_extents_per_dim(mcns), we inherit the
        following requirements:
        `mcns` were generated with a Manhattan distance `dst_fn` by `identify_mesh_casts`
        s.t. all dimensions are orthogonal to each other in a metric space, where each
        unit movement in a dimension counts as 1 hop.

        We also assume `dst_fn` is translationally invariant (i.e., ∀src, dst,
        src', dst' ∈ space, if |src - dst| = |src' - dst'|,
        dst_fn(src, dst) = dst_fn(src', dst').
        """
        dim_extents: list[isl.PwAff] = calculate_extents_per_dim(mcns)
        # Tracks the total cost of the hypercube cast per [src -> data]
        one: isl.PwAff = isl.PwAff.val_on_domain(dim_extents[0].domain(), 1)
        hypercube_costs = isl.PwQPolynomial.from_pw_aff(one)

        # Calculates the cost of the hypercube, where the hypercube cost
        # = \sum_{i=0}^{D} ((extent_i - 1) * \prod_{j=0}^{i-1} extent_j)
        # = (\prod_{i=0}^{D} extent_i) - 1
        for dim_extent in dim_extents:
            # Adds the dim_extent times the casting volume to the hypercube
            # cost.
            dim_plus: isl.PwQPolynomial = isl.PwQPolynomial.from_pw_aff(
                dim_extent.add(one).coalesce()
            )
            hypercube_costs = hypercube_costs.mul(dim_plus).coalesce()
        hypercube_costs = hypercube_costs.sub(isl.PwQPolynomial.from_pw_aff(one))

        # Tracks the total cost of the hyppercube cast per data.
        hypercube_costs = hypercube_costs.sum()

        # Return the hypercube cost as a piecewise polynomial.
        return hypercube_costs.sum()
