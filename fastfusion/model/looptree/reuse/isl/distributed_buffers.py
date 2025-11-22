from fastfusion.model.looptree.reuse.isl.spatial import Transfers
import logging
from numbers import Real

import islpy as isl

from fastfusion.model.looptree.reuse.isl.isl_functions import dim_projector_mask
from fastfusion.model.looptree.reuse.isl.mapping_to_isl import (
    DUMP_ISL_IR
)
from fastfusion.model.looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy
)
from fastfusion.model.looptree.reuse.isl.spatial import (
    TransferInfo,
    TransferModel
)


def identify_mesh_casts(
    src_occupancy: isl.Map, dst_fill: isl.Map, dist_fn: isl.Map
) -> isl.Map:
    """

    Parameters
    ----------
    src_occupancy:
        An isl.Map of the form { [src] -> [data] } corresponding to the data held
        at the buffer at space `src`.
    dst_fill:
        An isl.Map of the form { [dst] -> [data] } corresponding to the data requested
        at the element at space `dst`.
    dist_fn:
        Example
        -------
        {
            [src -> dst] -> [hops]
        }
        
        A distance function that accepts two points in space, corresponding to
        the `src` and `dst`, and returns the distance between the two points in
        terms of `hops`, a quantized atomic distance of data transmission cost.
    """
    # Makes [[dst -> data] -> dst] -> data
    wrapped_dst_fill: isl.Set = dst_fill.wrap()
    wrapped_fill_identity: isl.Map = isl.Map.identity(
        wrapped_dst_fill.get_space().map_from_set()
    )
    wrapped_fill_identity = wrapped_fill_identity.intersect_domain(
        wrapped_dst_fill
    )
    if DUMP_ISL_IR:
        logging.info(f"wrapped_fill_identity: {wrapped_fill_identity}")

    # Makes { [dst -> data] -> [dst -> data] }
    uncurried_fill_identity: wrapped_fill_identity.uncurry()
    if DUMP_ISL_IR:
        logging.info(f"uncurried_fill_identity: {uncurried_fill_identity}")

    # Inverts src_occupancy s.t. data -> src.
    # i.e. { [xs, ys] -> [d0, d1] } to { [d0, d1] -> [xs, ys] }
    data_presence: isl.Map = src_occupancy.reverse()

    # { [[dst -> data] -> dst] -> [src] }
    fills_to_dst_TO_src: isl.Map = uncurried_fill_identity.apply(
        data_presence
    )
    # { [dst -> data] -> [dst -> src] }
    fills_to_matches: isl.Map = fills_to_dst_TO_src.curry()
    if DUMP_ISL_IR:
        logging.info(f"fills_to_matches: {filles_to_matches}")

    # Calculates the distance of all the dst-src pairs with matching data.
    # { [dst -> data] -> [dist] }
    distances_map: isl.Map = fills_to_matches.apply_range(dist_fn)
    # { [[dst -> data] -> [dst -> src]] -> [dst -> src] }
    fills_to_matches_TO_matches: isl.Map = fills_to_matches.range_map()
    # { [[dst -> data] -> [dst -> src]] -> [dist] }
    fills_to_matches_TO_dist: isl.Map = fills_to_matches_TO_matches.apply_range(
        dist_fn
    )

    # Gets the minimal distance pairs.
    # { [dst -> data] -> [dist] }
    lexmin_dists: isl.Map = distances_map.lexmin()
    # Associates each destination with its closest source containing the data.
    # { [dst -> data] -> [[dst -> data] -> [dst -> src]] }
    associate_dist_to_src: isl.Map = lexmin_dists.apply_range(
        dst_to_data_TO_dst_to_src_TO2_dist.reverse()
    )
    # Isolates the relevant minimal pairs.
    # { [dst -> data] -> [dst -> src] :.dst -> src is minimized distance }
    minimal_pairs: isl.Map = associate_dist_to_src.range().unwrap()
    if DUMP_ISL_IR:
        logging.log(f"minimal_pairs: {minimal_pairs}")

    # Isolates the multicast networks.
    # { [dst] -> [data -> [dst -> src]] : dst -> src is minimized distance } 
    multicast_networks: isl.Map = minimal_pairs.curry()
    # { [data] -> [dst -> src] }
    multicast_networks = multicast_networks.range().unwrap()
    print(multicast_networks)
    # { [data -> dst] -> [src] }
    multicast_networks = multicast_networks.uncurry()
    # Devolves to a single source if multiple sources per domain point.
    multicast_networks = multicast_networks.lexmin()
    # { [data] -> [dst -> src] }
    multicast_networks = multicast_networks.curry()

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

    The extents calculation will still work if this is not the case, but downstream
    users of the extents calculation will inherit this assumption which breaks
    cases like hypercube mesh calculations forfractal distances (e.g., you cannot
    access certain places in the NoC without an intermediate) and non-orthogonal 
    distances (e.g., x, y, and an xy axis between the orthogonal x and y axes). 
    )
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
    min_cost: Real = float('inf')
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
        dim_extent_space: isl.Map = casting_extents.apply(extent_mapper)
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

    def apply(fills: Fill, occ: Occupancy, dist_fn: isl.Map) -> TransferInfo:
        mcs: isl.Map = identify_mesh_casts(
            occ.map_, fills.map_, dist_fn
        )
        result: isl.PwQPolynomial = cost_mesh_cast_hypercube(mcs)

        # TODO: Read once from all buffers, assert that card(mcs) == tensor_size * D
        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_),
            parent_reads=Reads(occ.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(fills.map_))
        )
    
    def _cost_mesh_cast_hypercube(mcns: isl.Map) -> int:
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
