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

from typing import Optional

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
        n_meshcasts: int = mcs.card()
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


class FullyConnectedMulticastModel(TransferModel):
    """
    Multicast cost model for a fully-connected fabric (e.g. an NVSwitch-style
    all-to-all interconnect).

    On a fully-connected fabric every cross-node delivery costs exactly one hop
    regardless of distance, and a self-delivery (source node == destination node)
    costs zero. The cost of a mapping is therefore the number of deliveries that
    actually traverse the fabric:

        cost = | { (data, dst, src) in mcs : dist_fn(dst, src) >= 1 } |

    This is distance-independent in magnitude (one hop per crossing); ``dist_fn``
    is used only to tell self-deliveries (0 hops) apart from fabric-crossing ones.

    See Also
    --------
    HypercubeMulticastModel :
        Worst-case convex-hypercube cost. On a one-hot fully-connected encoding it
        overestimates all-to-all traffic ~3x relative to this model (e.g. 168 vs
        56 hops for an 8-GPU all-to-all), because each unicast is costed as a
        (1 + 1)(1 + 1) - 1 = 3 hypercube instead of a single crossing.
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
            topology is captured entirely by ``dist_fn``.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `hops` is the number of fabric-crossing deliveries.
        """
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_fully_connected(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(fills.map_)),
            hops=result,
            link_transfer=True,
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

    The hop cost of one such tree (source ``s = (xs, ys)`` with destination set
    ``D``) is therefore::

        x_extent({xs} u {xd : (xd, yd) in D})
          + sum over destination columns xd of
                y_extent({ys} u {yd : (xd, yd) in D})

    Because the Y segments restart from the source row in every column rather than
    sharing a trunk, this is an upper bound on free (any-monotone-path) routing and
    a lower bound on the hypercube model, giving the ordering::

        extent_DOR (floor)  <=  XY routing  <=  hypercube

    For example, source ``(1, 0)`` casting to ``(0, 2)`` and ``(2, 2)`` costs 4
    (floor), 6 (XY), and 8 (hypercube) respectively.

    Source selection is per destination: ``identify_mesh_casts`` pairs each
    destination with its nearest source (devolving ties), and destinations that
    share a source form one tree; the cost sums over all such trees and all data.

    Preconditions
    -------------
    The NoC is two-dimensional, ``noc[x, y]`` (no temporal dimensions in the
    spacetime), with ``x`` routed before ``y``. N-dimensional dimension-order
    routing is a future extension. The returned cost is a parameter-free constant
    (the validated regime); parametric spacetimes are not yet supported.

    See Also
    --------
    HypercubeMulticastModel :
        Reaches every node in the bounding box -- an upper bound on XY routing.
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
            topology is captured by ``dist_fn`` and the ``noc[x, y]`` coordinates.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A TransferInfo whose `hops` is the total XY-routing link count.
        """
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_xy(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(fills.map_)),
            hops=result,
            link_transfer=True,
        )

    def _cost_xy(self, mcns: isl.Map) -> isl.PwQPolynomial:
        """
        Total XY-routing link count for the multicast networks `mcns`.

        Parameters
        ----------
        mcns:
            Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`,
            grouped per destination by nearest source.

        Returns
        -------
        The X-phase links (per source row) plus Y-phase links (per destination
        column), as a constant piecewise quasi-polynomial.
        """
        ctx = isl.DEFAULT_CONTEXT

        # X-phase: horizontal links along each source row. The x-extent of
        # {dsts} u {src} per (data, src) is exactly calculate_extents_per_dim()'s
        # first (x) dimension; summing it counts every X link.
        x_extent: isl.PwAff = calculate_extents_per_dim(mcns)[0]
        x_links: isl.Val = self._eval_const(
            isl.PwQPolynomial.from_pw_aff(x_extent).sum()
        )

        # Y-phase: vertical links per (data, src, destination column), each column
        # spanning from the source row ys to the destinations in that column.
        # { [data -> src] -> [dst noc[x, y]] }
        per_src: isl.Map = mcns.range_reverse().uncurry()
        # Split each destination into its column and its y-coordinate, keying y by
        # column: { [data -> src -> col[x]] -> [yv[y]] }.
        split_col: isl.Map = isl.Map.read_from_str(
            ctx, "{ noc[x, y] -> [col[x'] -> yv[y']] : x' = x and y' = y }"
        )
        dst_y: isl.Map = per_src.apply_range(split_col).uncurry()
        # Inject the source row ys into every destination column so each column's
        # Y segment starts from the source.
        src_y: isl.Map = per_src.domain().unwrap().range_map().apply_range(
            isl.Map.read_from_str(ctx, "{ noc[xs, ys] -> yv[ys] }")
        )
        src_row: isl.Map = (
            dst_y.domain().unwrap().range_product(src_y).uncurry()
        )
        # { [data -> src -> col] -> [yv[y]] }: all y-positions touched in a column.
        col_ys: isl.Map = dst_y.union(src_row)

        # Count the links {ymin <= p < ymax} in each column via cardinality (robust
        # where summing a min/max polynomial is not).
        ge_min: isl.Map = isl.Map.read_from_str(
            ctx, "{ yv[ymin] -> p[t] : t >= ymin }"
        )
        lt_max: isl.Map = isl.Map.read_from_str(
            ctx, "{ yv[ymax] -> p[t] : t < ymax }"
        )
        links: isl.Map = col_ys.lexmin().apply_range(ge_min).intersect(
            col_ys.lexmax().apply_range(lt_max)
        )
        y_links: isl.Val = self._eval_const(links.wrap().card())

        # Total links as a parameter-free constant.
        total: isl.Val = x_links.add(y_links)
        zero_dim: isl.Space = isl.Space.set_alloc(ctx, 0, 0)
        return isl.PwQPolynomial.from_qpolynomial(
            isl.QPolynomial.val_on_domain(zero_dim, total)
        )

    @staticmethod
    def _eval_const(pwq: isl.PwQPolynomial) -> isl.Val:
        """Evaluate a parameter-free piecewise quasi-polynomial to its value."""
        return pwq.eval(isl.Point.zero(pwq.domain().get_space()))


class StarMulticastModel(TransferModel):
    """
    Does distributed multicasting assuming all nodes are connected to a central node.
    """

    def __init__(self, reindexer: Optional[isl.Map] = None):
        """
        No distance function as hops for a star model are assumed to be 1 to and from center to any node, and
        all data must route through the center.

        Parameters
        ----------
        reindexer: 
            flattens an input so that 0 (or the lexmin across all dimensions) is the assumed center everything
            connects to.
        """
        self.reindexer: Optional[isl.Map] = reindexer
    
    
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
        if self.reindexer:
            occs_map = isl.apply_domain(self.relabeler)
            fills_map = isl.apply_domain(self.relabeler)
        else:
            occs_map = occs.map
            fills_map = fills_map

        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        result: isl.PwQPolynomial = self._cost_star_multicast(mcs)

        # TODO: Read once from all buffers, assert that
        # card(mcs) == tensor_size * duplication factor
        n_meshcasts: int = mcs.card()
        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(fills.map_)),
            hops=result,
            link_transfer=True,
        )

    def classify_src_dst(rel: isl.Map | isl.Set):
        """
        Build a quasi-affine classifier over a wrapped relation  [src -> dst].
    
            f([src -> dst]) = 0   if   src == dst
                            = 1   elif src is the lexmin  OR  dst is the lexmin
                            = 2   otherwise
    
        Works for any dimensionality: `src`/`dst` may be scalars (1-D) or
        tuples (n-D). The only requirement is that the src space and the dst
        space match.
    
        Parameters
        ----------
        rel:
            isl.Map  ({ src -> dst }, with src-space == dst-space)
            or isl.Set ({ [src -> dst] }, i.e. already wrapped).
            The lexmin of src-space is the center of the star.
    
        Returns
        -------
        isl.PwAff defined on the wrapped space [src -> dst].

        Preconditions
        -------------
        lexmin is unique.
        """
        # Accept either a relation or an already-wrapped set.
        m = rel.unwrap() if isinstance(rel, isl.Set) else rel
    
        assert m.dim(isl.dim_type.in_) == m.dim(isl.dim_type.out), \
            "src and dst must share the same space"
    
        space_set = m.domain()
    
        # Position-wise identity  src -> dst. Used to (a) select the diagonal
        # src == dst, and (b) carry the lexmin point from the domain (src)
        # space into the range (dst) space, so tuple names need not match.
        ident = isl.Map.identity(m.get_space())
    
        lex_src = space_set.lexmin()      # lexmin point in the src space
        lex_dst = lex_src.apply(ident)    # same point, in the dst space
    
        # The three regions, as relations  src -> dst  (all subsets of m):
        eq        = m.intersect(ident)                  # src == dst
        is_lexmin = m.intersect_domain(lex_src).union(  # src == lexmin
                    m.intersect_range(lex_dst))         #   or dst == lexmin
    
        # Move everything into the wrapped  [src -> dst]  set space:
        W  = m.wrap()
        R0 = eq.wrap()
        R1 = is_lexmin.wrap().subtract(R0)              # the "elif": drop src==dst
        R2 = W.subtract(R0).subtract(R1)                # everything else
    
        # A constant quasi-affine piece with value `c` on the given domain.
        def const_on(domain, c):
            ls  = isl.LocalSpace.from_space(domain.get_space())
            val = isl.Val.int_from_si(domain.get_ctx(), c)
            aff = isl.Aff.zero_on_domain(ls).set_constant_val(val)
            return isl.PwAff.from_aff(aff).intersect_domain(domain)
    
        # Disjoint domains, so union_add is just a disjoint union of pieces.
        return (const_on(R0, 0)
                .union_add(const_on(R1, 1))
                .union_add(const_on(R2, 2)))


    def _pairing(src_occupancy: isl.Map, dst_fill: isl.Map, dist_fn: isl.Map):
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

        Returns
        -------
        { [data] -> [dst -> src] } where { [dst] -> [data] } and { [src] -> [data] } are in
        `src_occupancy` and `dst_fill` respectively, and where `[dst -> src]` is the infimum of
        `dst_fn(src, dst), ∀ src, dst s.t. { [src] -> [data] } ∈ `src_occupancy` and
        `{ [dst] -> [data] }` ∈ `dst_fill`.

        Preconditions:
        No duplication of data.
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


    def _cost_star_multicast(self, mcs):
        """
        """
        raise NotImplementedError("WIP: star multicast cost not yet implemented")