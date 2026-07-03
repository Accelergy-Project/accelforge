"""
Models for handling calculating the cost of a Workload on distributed buffer
architectures.
"""

import logging

from dataclasses import dataclass

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

from typing import Optional


@dataclass(frozen=True)
class EdgePressure:
    """
    Per-edge memory pressure (link load) for a spatial transfer.

    Where a `TransferModel`'s `hops` collapses a whole routing to a single
    scalar, `EdgePressure` keeps the load broken out per physical edge: how many
    multicast trees cross each individual link -- the quantity a per-link
    bandwidth limit acts on, and what the symbolic network model calls
    `max_traffic`.

    "Edge" is a directed physical link, identified by the tuple name and
    coordinates of `load`'s domain, e.g. `yedge_u[x, t]` (the upward vertical
    link in column `x` between rows `t` and `t + 1`, XY mesh) or `spoke_in[n]` /
    `spoke_out[n]` (a node's ingress / egress link to the switch, star model).

    The load is keyed on the multicast tree `(data, src)`, not on individual
    destinations: within one tree a link is traversed once regardless of how
    many leaves hang off it, so `load` measures pressure (distinct flows over a
    link) rather than summed hops -- e.g. for XY routing, sum over edges of load
    == total hops.

    Preconditions
    -------------
    `bottleneck` and `eval_edge` assume the load is piecewise constant (the
    parameter-free regime the distributed models are validated in); there is no
    clean ISL "max of a quasi-polynomial over its domain" primitive, so the
    bottleneck is obtained by enumerating pieces.
    """

    # { edge -> number-of-trees } spanning every directed edge type the model
    # emits.
    load: isl.UnionPwQPolynomial

    def total(self) -> int:
        """
        Sum the load over every edge.

        For a single-direction (or single edge-type) pressure this is the total
        traffic; for the full mesh pressure it equals the model's total `hops`
        (sum over edges of load == total hops), which is the primary cross-check
        that the per-edge decomposition is correct.
        """
        pieces: list[isl.PwQPolynomial] = []
        self.load.foreach_pw_qpolynomial(pieces.append)
        total = 0
        for pwq in pieces:
            # `.sum()` collapses the edge-indexed domain away (summing over every
            # edge of this piece), leaving a 0-set-dim, parameter-free
            # polynomial -- exactly `_eval_const`'s precondition.
            total += _eval_const(pwq.sum())
        return total

    def bottleneck(self) -> int:
        """
        Return the load on the single most-pressured edge.

        This is the bandwidth-binding quantity: with a uniform per-link
        bandwidth, the most-congested edge saturates first, so its load sets the
        transfer's bandwidth-bound latency.

        The per-edge load is generally not constant across an edge type (e.g. a
        flooded column's upward link `yedge_u[x, t]` carries `t + 1` trees), so
        the maximum is found by enumerating the (finite, parameter-free) edge
        domain and evaluating the load at each edge -- sampling one point per
        piece would under-report a monotone load.
        """
        best = 0
        pieces: list[isl.PwQPolynomial] = []
        self.load.foreach_pw_qpolynomial(pieces.append)
        for pwq in pieces:
            edges: list[isl.Point] = []
            pwq.domain().foreach_point(edges.append)
            for edge in edges:
                best = max(best, int(str(pwq.eval(edge))))
        return best

    def eval_edge(self, name: str, coords: list[int]) -> int:
        """
        Look up the load on one named edge, e.g. `eval_edge("yedge_u", [0, 6])`.

        Returns 0 if no flow crosses that edge (it is outside the load's support).

        Parameters
        ----------
        name:
            The edge tuple name (`xedge_r`/`xedge_l`/`yedge_u`/`yedge_d`
            for the mesh model, `spoke_in`/`spoke_out` for the spokes model).
        coords:
            The integer edge coordinates within that tuple.
        """
        pieces: list[isl.PwQPolynomial] = []
        self.load.foreach_pw_qpolynomial(pieces.append)
        for pwq in pieces:
            if pwq.domain().get_space().get_tuple_name(isl.dim_type.set) != name:
                continue
            point: isl.Point = isl.Set.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ %s[%s] }" % (name, ", ".join(str(c) for c in coords)),
            ).sample_point()
            return int(str(pwq.eval(point)))
        return 0


def _eval_const(pwq: isl.PwQPolynomial) -> int:
    """
    Evaluate a parameter-free, already-reduced piecewise quasi-polynomial to its
    scalar value.

    Preconditions
    -------------
    `pwq` has no free set dimensions and no parameters -- e.g. the output of
    `.card()` on a parameter-free map/set, `.sum()` on a parameter-free
    polynomial, or a constant built by `_const_pwq`. Evaluated only at the
    space's zero point, so a `pwq` that still varies over domain points or
    parameters gives a meaningless result; callers must reduce to a true
    constant first (via `.sum()`/`.card()`).

    Returns
    -------
    The polynomial's constant value as a Python `int`.
    """
    # Note: isl.Val has no direct int() conversion in this islpy build (it
    # raises TypeError); round-tripping through str() is the working idiom
    # used throughout this module.
    return int(str(pwq.eval(isl.Point.zero(pwq.domain().get_space()))))


def _const_pwq(value: int) -> isl.PwQPolynomial:
    """
    Build a 0-dimensional, parameter-free constant `isl.PwQPolynomial` equal to
    `value`.

    Parameters
    ----------
    value:
        The integer the returned polynomial evaluates to everywhere (it has no
        domain dimensions or parameters to vary over).

    Returns
    -------
    An `isl.PwQPolynomial` over the empty (0-dim, 0-param) space, suitable
    wherever a parameter-free `hops` cost is expected. Round-trips through
    `_eval_const` back to `value`.
    """
    ctx = isl.DEFAULT_CONTEXT
    zero_dim: isl.Space = isl.Space.set_alloc(ctx, 0, 0)
    return isl.PwQPolynomial.from_qpolynomial(
        isl.QPolynomial.val_on_domain(zero_dim, isl.Val(value, ctx))
    )


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


def _edge_pressure_from_links(edge_maps: list[isl.Map]) -> EdgePressure:
    """
    Turn directed flow maps into an `EdgePressure`.

    Parameters
    ----------
    edge_maps:
        A list of { [data -> src] -> edge } maps, one per directed edge type,
        each associating a multicast tree with every edge its route traverses.

    Returns
    -------
    An `EdgePressure` whose `load` is { edge -> number-of-trees }: for each map
    we reverse it and take the cardinality (`reverse().card()` counts, per
    edge, how many distinct `(data, src)` trees cross it), then union the
    per-type results into one `UnionPwQPolynomial`.
    """
    acc: Optional[isl.UnionPwQPolynomial] = None
    for edge_map in edge_maps:
        # { edge -> #trees crossing it }
        per_edge: isl.PwQPolynomial = edge_map.reverse().card()
        contribution = isl.UnionPwQPolynomial.from_pw_qpolynomial(per_edge)
        acc = contribution if acc is None else acc.add(contribution)
    if acc is None:
        acc = isl.UnionPwQPolynomial.read_from_str(isl.DEFAULT_CONTEXT, "{ }")
    return EdgePressure(acc)


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
        calculate the spatial transfers.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Currently,
            we rely on `dist_fn` to deal with this rather than `buff`.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `fulfilled_fill`/`unfulfilled_fill` partition
        `fills` by whether `identify_mesh_casts` found a matched source for that
        (dst, data) pair -- a fill with no source holding its datum is
        `unfulfilled_fill`, not silently treated as fulfilled. `edge_pressure`
        is left at its default (`None`): the hypercube model costs a convex
        bounding box, which has no notion of individual links to report
        pressure on.
        """
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_mesh_cast_hypercube(mcs)
        # { dst -> data } fills actually covered by a matched source.
        covered: isl.Map = _covered_fills(mcs)

        # TODO: Read once from all buffers, assert that
        # card(mcs) == tensor_size * duplication factor
        n_meshcasts: int = mcs.card()
        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=result,
            link_transfer=True,
            # No per-link decomposition defined for the hypercube abstraction.
            edge_pressure=None,
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


class FullyConnectedMulticastModel(TransferModel):
    """
    Multicast cost model for a fully-connected fabric (e.g. an NVSwitch-style
    all-to-all interconnect).

    Every cross-node delivery costs exactly one hop regardless of distance, and
    a self-delivery (source node == destination node) costs zero:

        cost = | { (data, dst, src) in mcs : dist_fn(dst, src) >= 1 } |

    This is distance-independent in magnitude (one hop per crossing); `dist_fn`
    is used only to tell self-deliveries (0 hops) apart from fabric-crossing ones.
    """

    def __init__(self, dist_fn: isl.Map):
        """
        Initializes the model with the distance function over the metric space.

        Parameters
        ----------
        dist_fn:
            A distance function { [dst -> src] -> [hops] }. Only used to classify a
            delivery as self (0 hops) vs. fabric-crossing (>= 1 hop); the hop
            magnitude does not enter the cost.
        """
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the spatial transfers on a fully-connected fabric.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Unused; the
            topology is captured entirely by `dist_fn`.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `hops` is the number of fabric-crossing deliveries.
        `fulfilled_fill`/`unfulfilled_fill` partition `fills` by whether
        `identify_mesh_casts` matched a source. `edge_pressure` is left at its
        default (`None`): this model treats the fabric as a contention-free
        full mesh (one dedicated link per pair), so there is no per-link
        pressure to report.
        """
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_fully_connected(mcs)
        # { dst -> data } fills actually covered by a matched source.
        covered: isl.Map = _covered_fills(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=result,
            link_transfer=True,
            # No per-link decomposition defined for the full-mesh abstraction.
            edge_pressure=None,
        )

    def _cost_fully_connected(self, mcns: isl.Map) -> isl.PwQPolynomial:
        """
        Count the deliveries in `mcns` that traverse the fabric.

        Parameters
        ----------
        mcns:
            Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.

        Returns
        -------
        The number of (data, dst, src) deliveries with dist_fn(dst, src) >= 1, as a
        piecewise quasi-polynomial (constant when there are no parameters).
        """
        # [dst -> src] pairs that actually traverse the fabric (>= 1 hop).
        crossing_hops: isl.Set = isl.Set.read_from_str(
            isl.DEFAULT_CONTEXT, "{ hops[h] : h >= 1 }"
        )
        crossing_pairs: isl.Set = self.dist_fn.intersect_range(crossing_hops).domain()
        crossing: isl.Map = mcns.intersect_range(crossing_pairs)
        return crossing.wrap().card()


class XYRoutingMulticastModel(TransferModel):
    """
    Multicast cost model for XY (dimension-order) routing on a 2-D mesh.

    XY routing constrains every packet to travel along the X dimension first and
    only then along the Y dimension, so a multicast from one source forms a rigid
    tree:

    1. an X segment along the source's row, reaching every column that holds a
       destination, and
    2. an independent Y segment down each of those columns, starting from the
       source's row.

    The hop cost of one such tree (source `s = (xs, ys)` with destination set
    `D`) is therefore::

        x_extent({xs} u {xd : (xd, yd) in D})
          + sum over destination columns xd of
                y_extent({ys} u {yd : (xd, yd) in D})

    Because the Y segments restart from the source row in every column rather than
    sharing a trunk, this is an upper bound on free (any-monotone-path) routing and
    a lower bound on the hypercube model, giving the ordering::

        extent_DOR (floor)  <=  XY routing  <=  hypercube

    For example, source `(1, 0)` casting to `(0, 2)` and `(2, 2)` costs 4
    (floor), 6 (XY), and 8 (hypercube) respectively.

    Source selection is per destination: `identify_mesh_casts` pairs each
    destination with its nearest source (devolving ties), and destinations that
    share a source form one tree; the cost sums over all such trees and all data.

    Preconditions
    -------------
    The NoC is two-dimensional (no temporal dimensions in the spacetime), with
    the first coordinate routed before the second (X then Y). The node tuple's
    name is generic -- read off `fills`/`occs` at `apply` time, so any name
    works (e.g. `noc[x, y]` or `pe[x, y]`) as long as `fills`, `occs`, and
    `self.dist_fn` all agree on it (see the caller contract on
    `identify_mesh_casts`'s `dist_fn` parameter). The tuple must be exactly
    2-D; `apply` raises `ValueError` otherwise. N-dimensional dimension-order
    routing is a future extension. The returned cost is a parameter-free
    constant (the validated regime); parametric spacetimes are not yet
    supported.
    """

    def __init__(self, dist_fn: isl.Map):
        """
        Parameters
        ----------
        dist_fn:
            A distance function { [dst -> src] -> [hops] } used to pick each
            destination's nearest source (Manhattan, like the hypercube model).
        """
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the XY-routing spatial transfers.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Unused; the
            topology is captured by `dist_fn` and the node coordinates.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `hops` is the total XY-routing link count and whose
        `edge_pressure` is the per-directed-mesh-link decomposition backing it
        (`xedge_r`/`xedge_l`/`yedge_u`/`yedge_d` -- see `_directed_mesh_links`).
        `hops` is `pressure.total()`, so the two can never disagree.
        `fulfilled_fill`/`unfulfilled_fill` partition `fills` by whether
        `identify_mesh_casts` matched a source.

        Preconditions
        -------------
        The node tuple embedded in `fills`/`occs` must be exactly 2-D (XY
        routing is only defined for a 2-D mesh; see `_directed_mesh_links`),
        else raises `ValueError`.
        """
        # Note: `identify_mesh_casts` is called exactly once; both `hops` and
        # `edge_pressure` are derived from this single `mcs` so they cannot
        # fall out of sync.
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        links: list[isl.Map] = self._directed_mesh_links(mcs)
        pressure: EdgePressure = _edge_pressure_from_links(links)
        # `hops` is `EdgePressure.total()` (sum of per-edge loads == total link
        # count), wrapped as the constant `TransferInfo.hops` expects.
        hops: isl.PwQPolynomial = _const_pwq(pressure.total())
        # { dst -> data } fills actually covered by a matched source.
        covered: isl.Map = _covered_fills(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=hops,
            link_transfer=True,
            edge_pressure=pressure,
        )

    def _directed_mesh_links(self, mcns: isl.Map) -> list[isl.Map]:
        """
        Decompose every multicast tree in `mcns` onto the directed mesh links it
        traverses under XY routing.

        Each returned map is { [data -> src] -> edge } for one directed edge
        type, associating a tree with every link of that type on its route. The
        X phase runs along the source row out to every destination column; the Y
        phase runs down each destination column from the source row. Directions
        split at the source: rightward/leftward in X (at the source column `xs`)
        and upward/downward in Y (at the source row `ys`).

        Parameters
        ----------
        mcns:
            Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.

        Returns
        -------
        `[xedge_r, xedge_l, yedge_u, yedge_d]` maps. An edge `xedge_r[t, ys]`
        is the rightward link in row `ys` between columns `t` and `t + 1`;
        `yedge_u[x, t]` is the upward link in column `x` between rows `t` and
        `t + 1` (and `_l` / `_d` the opposite directions).

        Preconditions
        -------------
        The node tuple embedded in `mcns` must be exactly 2-D; every map string
        below assumes a 2-D `name[x, y]` node tuple. Raises `ValueError`
        otherwise.
        """
        ctx = isl.DEFAULT_CONTEXT
        # Reads the node tuple's name (and validates its dimensionality) off
        # `mcns` instead of assuming the literal 'noc', so this method works
        # for any 2-D node tuple.
        name, dims = _mesh_node_tuple(mcns)
        if dims != 2:
            raise ValueError(
                "XYRoutingMulticastModel requires a 2-D node tuple "
                f"(X then Y); got tuple '{name}' with {dims} dimensions. "
                "N-dimensional dimension-order routing is not yet supported."
            )

        # { [data -> src] -> dst name[x, y] } and a handle on the source per tree.
        per_src: isl.Map = mcns.range_reverse().uncurry()
        keymap: isl.Map = per_src.domain().unwrap().range_map()  # [data->src] -> src

        # --- Y phase: vertical links per (tree, destination column). ---
        # Key each destination's y by its column: { [data->src->col] -> yv[y] }.
        dst_y: isl.Map = per_src.apply_range(
            isl.Map.read_from_str(
                ctx,
                "{ %s[x, y] -> [col[x'] -> yv[y']] : x' = x and y' = y }" % name,
            )
        ).uncurry()
        # Inject the source row ys into every destination column so each Y segment
        # starts from the source.
        src_y: isl.Map = keymap.apply_range(
            isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> yv[ys] }" % name)
        )
        src_row: isl.Map = dst_y.domain().unwrap().range_product(src_y).uncurry()
        col_ys: isl.Map = dst_y.union(src_row)
        # Every link {ymin <= t < ymax} touched in a column, as a relation keyed by
        # the column (card of an explicit link set, robust where a min/max sum is
        # not). { [data->src->col] -> p[t] }.
        ylinks: isl.Map = col_ys.lexmin().apply_range(
            isl.Map.read_from_str(ctx, "{ yv[ymin] -> p[t] : t >= ymin }")
        ).intersect(
            col_ys.lexmax().apply_range(
                isl.Map.read_from_str(ctx, "{ yv[ymax] -> p[t] : t < ymax }")
            )
        )
        # Re-key links by the tree and carry ys so direction splits at the source:
        # { [data->src] -> [[col[x] -> p[t]] -> ysv[ys]] }.
        y_with_ys: isl.Map = ylinks.curry().range_product(
            keymap.apply_range(
                isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> ysv[ys] }" % name)
            )
        )
        yedge_u: isl.Map = y_with_ys.apply_range(
            isl.Map.read_from_str(
                ctx, "{ [[col[x] -> p[t]] -> ysv[ys]] -> yedge_u[x, t] : t >= ys }"
            )
        )
        yedge_d: isl.Map = y_with_ys.apply_range(
            isl.Map.read_from_str(
                ctx, "{ [[col[x] -> p[t]] -> ysv[ys]] -> yedge_d[x, t] : t < ys }"
            )
        )

        # --- X phase: horizontal links along the source row. ---
        # Columns spanned per tree = {src column} u {destination columns}. (Built
        # explicitly, not from calculate_extents_per_dim, which keeps only the
        # extent length and discards which links are used.)
        col_x: isl.Map = per_src.apply_range(
            isl.Map.read_from_str(ctx, "{ %s[x, y] -> cx[x] }" % name)
        ).union(
            keymap.apply_range(
                isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> cx[xs] }" % name)
            )
        )
        xlinks: isl.Map = col_x.lexmin().apply_range(
            isl.Map.read_from_str(ctx, "{ cx[xmin] -> ex[t] : t >= xmin }")
        ).intersect(
            col_x.lexmax().apply_range(
                isl.Map.read_from_str(ctx, "{ cx[xmax] -> ex[t] : t < xmax }")
            )
        )
        # Carry the source (xs, ys) so direction splits at xs and the row ys is
        # part of the edge identity: { [data->src] -> [ex[t] -> xy[xs, ys]] }.
        x_with_src: isl.Map = xlinks.range_product(
            keymap.apply_range(
                isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> xy[xs, ys] }" % name)
            )
        )
        xedge_r: isl.Map = x_with_src.apply_range(
            isl.Map.read_from_str(
                ctx, "{ [ex[t] -> xy[xs, ys]] -> xedge_r[t, ys] : t >= xs }"
            )
        )
        xedge_l: isl.Map = x_with_src.apply_range(
            isl.Map.read_from_str(
                ctx, "{ [ex[t] -> xy[xs, ys]] -> xedge_l[t, ys] : t < xs }"
            )
        )

        return [xedge_r, xedge_l, yedge_u, yedge_d]


class StarMulticastModel(TransferModel):
    """
    Multicast cost model for a star / central-switch fabric -- the spokes
    realization of a fully-connected interconnect (e.g. an NVSwitch, where every
    GPU connects to a shared switch rather than to a full mesh of peers).

    Every node has exactly one spoke (its bidirectional link to the switch). A
    delivery routes `src -> switch -> dst`: the source injects each datum once
    up its egress spoke (multicast fan-out happens at the switch, so one copy per
    datum regardless of how many destinations want it), and every destination
    receives its datum down its ingress spoke. Self-deliveries (a node already
    holding the datum) never cross the fabric and so load no spoke.
    """

    def __init__(self, dist_fn: isl.Map):
        """
        Parameters
        ----------
        dist_fn:
            A distance function { [dst -> src] -> [hops] }, used both to pick each
            destination's nearest source and to tell self-deliveries (0 hops, no
            spoke load) apart from fabric-crossing ones (>= 1 hop). The hop
            magnitude does not enter the spoke load -- on a star every crossing is
            one switch hop each way.
        """
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the spatial transfers on a star / central-switch fabric.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Unused; the
            topology is captured entirely by `dist_fn`.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `hops` is the total spoke traversals (injections plus
        deliveries) and whose `edge_pressure` is the per-spoke decomposition
        backing it (`spoke_in[n]`/`spoke_out[n]` -- see `_spoke_loads`). `hops`
        is `pressure.total()`, so the two can never disagree.
        `fulfilled_fill`/`unfulfilled_fill` partition `fills` by whether
        `identify_mesh_casts` matched a source.
        """
        # Note: `identify_mesh_casts` is called exactly once; both `hops` and
        # `edge_pressure` are derived from this single `mcs`.
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        ingress, egress = self._spoke_loads(mcs)
        acc: isl.UnionPwQPolynomial = isl.UnionPwQPolynomial.from_pw_qpolynomial(
            ingress
        ).add(isl.UnionPwQPolynomial.from_pw_qpolynomial(egress))
        pressure: EdgePressure = EdgePressure(acc)
        # `hops` is `EdgePressure.total()` (sum(ingress) + sum(egress)), wrapped
        # as the constant `TransferInfo.hops` expects.
        hops: isl.PwQPolynomial = _const_pwq(pressure.total())
        # { dst -> data } fills actually covered by a matched source.
        covered: isl.Map = _covered_fills(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=hops,
            link_transfer=True,
            edge_pressure=pressure,
        )

    def _spoke_loads(
        self, mcns: isl.Map
    ) -> tuple[isl.PwQPolynomial, isl.PwQPolynomial]:
        """
        Per-spoke ingress and egress load for the multicast networks `mcns`.

        Parameters
        ----------
        mcns:
            Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.

        Returns
        -------
        `(ingress, egress)` where `ingress` is { spoke_in[n] -> #data the node
        receives } and `egress` is { spoke_out[n] -> #data the node sources },
        counting only fabric-crossing (>= 1 hop) deliveries.
        """
        # Reads the node tuple's name off `mcns` instead of assuming the
        # literal 'noc'; unlike XY, the star model has no dimensionality
        # requirement.
        name, _dims = _mesh_node_tuple(mcns)

        # Keep only deliveries that actually cross the fabric (>= 1 hop); a node
        # already holding its datum loads no spoke.
        crossing_hops: isl.Set = isl.Set.read_from_str(
            isl.DEFAULT_CONTEXT, "{ hops[h] : h >= 1 }"
        )
        crossing_pairs: isl.Set = self.dist_fn.intersect_range(crossing_hops).domain()
        crossing: isl.Map = mcns.intersect_range(crossing_pairs)

        # { dst -> [src -> data] }: regroup so each delivery is keyed by destination.
        cur: isl.Map = crossing.reverse().curry()
        # Distinct (node, data) pairs per direction. A source injects each datum
        # once (multicast fans out at the switch); a destination receives each once.
        ingress_nodes: isl.Map = cur.range_factor_range()  # { name[dst] -> data }
        egress_nodes: isl.Map = cur.range().unwrap()       # { name[src] -> data }

        # Relabel the node tuple to the directed spoke edge, then count data per
        # spoke. { spoke_in[n] -> #data } and { spoke_out[n] -> #data }.
        dims: int = ingress_nodes.dim(isl.dim_type.in_)
        idx: str = ", ".join(f"i{k}" for k in range(dims))
        ingress: isl.PwQPolynomial = ingress_nodes.apply_domain(
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT, "{ %s[%s] -> spoke_in[%s] }" % (name, idx, idx)
            )
        ).card()
        egress: isl.PwQPolynomial = egress_nodes.apply_domain(
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT, "{ %s[%s] -> spoke_out[%s] }" % (name, idx, idx)
            )
        ).card()
        return ingress, egress
