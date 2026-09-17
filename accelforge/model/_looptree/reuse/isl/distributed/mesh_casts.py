"""
Multicast-network construction for distributed transfer models: matching each
requested datum to its nearest source (`identify_mesh_casts`) and the
shared-shape helpers built on top of its result (fill-partition recovery,
per-dimension extents, tree re-keying, and fabric-crossing filtering).

Imports `islpy`, `isl_functions.dim_projector_mask`, and
`mapping_to_isl.DUMP_ISL_IR` only -- no dependency on `edge_pressure.py`,
`spatial.py`, or `distributed_buffers.py`, so this module cannot participate
in the import cycle those would otherwise risk.
"""

import logging

import islpy as isl

from accelforge.model._looptree.reuse.isl.isl_functions import dim_projector_mask
from accelforge.model._looptree.reuse.isl.mapping_to_isl import DUMP_ISL_IR

# [dst -> src] pairs whose distance is >= 1 hop, i.e. every fabric-crossing
# delivery (as opposed to a self-delivery, which never traverses a link).
# Parsed once at import rather than per `_fabric_crossing` call -- the string
# encodes `identify_mesh_casts`'s `dist_fn` contract (range tuple named
# `hops`), not anything instance- or call-specific.
_CROSSING_HOPS: isl.Set = isl.Set.read_from_str(
    isl.DEFAULT_CONTEXT, "{ hops[h] : h >= 1 }"
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
        A distance function { [dst -> src] -> [hops] } that accepts two points in
        space, corresponding to the `dst` and `src`, and returns the distance
        between the two points in terms of `hops`, a quantized atomic distance of
        data transmission cost.

        Caller contract: the tuple names of `dst_fill`'s and `src_occupancy`'s
        domains (the spacetime/node tuple, e.g. `noc[x, y]`) must match the
        corresponding tuple names in `dist_fn`'s domain -- ISL will raise on a
        name mismatch when `dist_fn` is applied. `dist_fn`'s range tuple must be
        named `hops` (consumers such as `FullyConnectedMulticastModel` and
        `StarMulticastModel` filter on `{ hops[h] : h >= 1 }` to distinguish
        self-deliveries from fabric-crossing ones).

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
    fills_to_matches: isl.Map = (
        fill_to_fill.uncurry()  # { [[dst -> data] -> dst] -> data }
        .apply_range(data_presence)  # { [[dst -> data] -> dst] -> src }
        .curry()
    )  # { [[dst -> data] -> [dst -> src] }
    if DUMP_ISL_IR:
        logging.info(f"fills_to_matches: {fills_to_matches}")

    # Calculates the distance of a fill to the nearest src satisfying the fill.
    # { [dst -> data] -> [dist] }
    fill_min_dist: isl.Map = fills_to_matches.apply_range(dist_fn).lexmin()
    # Isolates the relevant minimal pairs.
    # { [dst -> data] -> [dst -> src] :.dst -> src is minimized distance }
    minimal_pairs: isl.Map = (
        fill_min_dist.apply_range(
            # Note: Need to match fill -> min_dist with min_dist -> [fill -> match] as lexmin over
            # fill and match will minimize distance over the tuple (src, dst, data), but that
            # overconstrains the optimization as we want to minimize over distance (dst, data)
            # only for all src.
            fills_to_matches.range_map()
            .apply_range(dist_fn)
            .reverse()
        )
        .range()
        .unwrap()
    )
    if DUMP_ISL_IR:
        logging.info(f"minimal_pairs: {minimal_pairs}")

    # Isolates the multicast networks.
    # { [data] -> [dst -> src] : dst -> src is minimized distance }
    multicast_networks: isl.Map = minimal_pairs.curry().range().unwrap()
    # Devolves to a single source if multiple sources per domain point.
    multicast_networks = multicast_networks.uncurry().lexmin().curry()

    return multicast_networks


def _covered_fills(mcs: isl.Map) -> isl.Map:
    """
    Recover the { dst -> data } fills actually covered by a matched multicast
    source, from `identify_mesh_casts`'s result.

    Parameters
    ----------
    mcs:
        { [data] -> [dst -> src] }, the multicast-network map returned by
        `identify_mesh_casts`.

    Returns
    -------
    { dst -> data } -- every (destination, datum) pair that has a matched
    source in `mcs`. A model's fill partition is then
    `fulfilled = fills.map_.intersect(covered)` and
    `unfulfilled = fills.map_.subtract(covered)`.
    """
    return (
        mcs.range_reverse()  # { data -> [src -> dst] }
        .uncurry()  # { [data -> src] -> dst }
        .domain_factor_domain()  # Drops src, keeps -> dst. { data -> dst }
        .reverse()  # { dst -> data }
    )


def _mesh_node_tuple(mcns: isl.Map) -> tuple[str, int]:
    """
    Read the spacetime/node tuple's name and dimensionality off a multicast
    network map, instead of assuming a hardcoded name.

    Parameters
    ----------
    mcns:
        { [data] -> [dst -> src] }, the multicast-network map returned by
        `identify_mesh_casts`. `dst` and `src` share the same node tuple, so
        it suffices to read the name/dims off one side (`dst`, via
        `.range().unwrap()`'s domain).

    Returns
    -------
    `(name, dims)`: the node tuple's ISL tuple name (e.g. "noc" or "pe") and
    its dimensionality.
    """
    # Note: correct even when mcns is empty -- ISL preserves space/tuple-name
    # information on empty relations, so this does not require data to be
    # present.
    node_space: isl.Map = mcns.range().unwrap()
    name: str = node_space.get_tuple_name(isl.dim_type.in_)
    dims: int = node_space.dim(isl.dim_type.in_)
    return name, dims


def _per_src(mcns: isl.Map) -> tuple[isl.Map, isl.Map]:
    """
    Re-key multicast networks by tree `(data, src)`, plus a source lookup.

    Parameters
    ----------
    mcns:
        Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.

    Returns
    -------
    `(per_src, keymap)`:
    - `per_src`: { [data -> src] -> dst } -- every destination grouped under
      its tree's key.
    - `keymap`: { [data -> src] -> src } -- the tree's own source point,
      recovered from `per_src`'s (wrapped) domain. Used to inject/carry the
      source's own coordinates (e.g. its row/column) into computations keyed
      on the same tree.
    """
    per_src: isl.Map = mcns.range_reverse().uncurry()  # { [data -> src] -> dst }
    keymap: isl.Map = per_src.domain().unwrap().range_map()  # { [data -> src] -> src }
    return per_src, keymap


def _fabric_crossing(mcns: isl.Map, dist_fn: isl.Map) -> isl.Map:
    """
    Filter multicast networks down to fabric-crossing deliveries (>= 1 hop).

    Parameters
    ----------
    mcns:
        Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.
    dist_fn:
        The same distance function { [dst -> src] -> [hops] } passed to
        `identify_mesh_casts` -- see its caller contract on the `hops` range
        tuple name, which this filters on.

    Returns
    -------
    { [data] -> [dst -> src] }, restricted to entries whose `[dst -> src]`
    pair has `dist_fn(dst, src) >= 1`, i.e. every delivery that actually
    crosses the fabric. Self-deliveries (0 hops) are dropped, since they load
    no fabric link.
    """
    crossing_pairs: isl.Set = dist_fn.intersect_range(_CROSSING_HOPS).domain()
    return mcns.intersect_range(crossing_pairs)


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
    potential_srcs, srcs = _per_src(mcns)
    # Sources are part of the extents, so we union it with the destinations.
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
