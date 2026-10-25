"""
Models for handling calculating the cost of a Workload on distributed buffer
architectures.

The shared shape (one `dist_fn`, one `identify_mesh_casts` call per `apply`,
one `TransferInfo` assembly) lives on the `MulticastModel` base class; each
concrete model below contributes only its cost kernel via `_transfer_cost`.
The underlying primitives -- `EdgePressure` and its scalar-extraction helpers,
and the multicast-network construction (`identify_mesh_casts` and friends) --
live in `edge_pressure.py` and `mesh_casts.py` respectively.
"""

from abc import abstractmethod

from typing import Optional

import islpy as isl

from accelforge.frontend.mapping import MappingNode
from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import Fill, Occupancy
from accelforge.model._looptree.reuse.isl.spatial import (
    Reads,
    Transfers,
    TransferInfo,
    TransferModel,
)

from accelforge.model._looptree.reuse.isl.distributed.edge_pressure import (
    EdgePressure,
    _const_pwq,
    _edge_pressure_from_links,
    _union_pwqs,
)
from accelforge.model._looptree.reuse.isl.distributed.mesh_casts import (
    identify_mesh_casts,
    calculate_extents_per_dim,
    _covered_fills,
    _mesh_node_tuple,
    _per_src,
    _fabric_crossing,
)


class MulticastModel(TransferModel):
    """
    Common shape shared by every distance-driven multicast transfer model.

    Every concrete model below (`HypercubeMulticastModel`,
    `FullyConnectedMulticastModel`, `XYRoutingMulticastModel`,
    `StarMulticastModel`) is built from one `dist_fn` and computes `apply`
    identically up to its cost kernel: call `identify_mesh_casts` exactly
    once, derive the cost from that single result, partition the fills, and
    assemble a `TransferInfo`. This base class owns that shared shape; a
    subclass need only implement `_transfer_cost`.

    Because `hops` and `edge_pressure` are both derived here from the same
    `_transfer_cost` return value, `Σ_edges load == hops` (for models that
    report an `EdgePressure`) is structural rather than an invariant each
    subclass has to maintain by hand: whenever `_transfer_cost` returns an
    `EdgePressure` `p`, `hops` is *always* `_const_pwq(p.total())`.
    """

    def __init__(self, dist_fn: isl.Map):
        """
        Parameters
        ----------
        dist_fn:
            A distance function { [dst -> src] -> [hops] } that accepts two
            points in space, corresponding to `dst` and `src`, and returns
            the distance between them in `hops`, a quantized atomic distance
            of data transmission cost. Stored and passed straight through to
            `identify_mesh_casts` on every `apply` call.

            Caller contract (inherited from `identify_mesh_casts`): the
            tuple names of `fills`/`occs`'s domains (the spacetime/node
            tuple, e.g. `noc[x, y]`) must match the corresponding tuple
            names in `dist_fn`'s domain -- ISL raises on a name mismatch
            when `dist_fn` is applied. `dist_fn`'s range tuple must be named
            `hops` (`_fabric_crossing`, used by `FullyConnectedMulticastModel`
            and `StarMulticastModel`, filters on `{ hops[h] : h >= 1 }` to
            distinguish self-deliveries from fabric-crossing ones).
        """
        self.dist_fn = dist_fn

    def apply(self, buff: MappingNode, fills: Fill, occs: Occupancy) -> TransferInfo:
        """
        Given a buffer, its fills across time, and its occupancies across time,
        calculate the spatial transfers.

        Parameters
        ----------
        buff:
            The buffer whose spatial analysis is being considered. Not used
            by any current subclass (they all rely on `dist_fn` instead);
            kept for interface symmetry with `TransferModel.apply`.
        fills:
            The fill of `buffer` across time from parents.
        occs:
            The occupancy of `buffer` across time.

        Returns
        -------
        A `TransferInfo` whose `fulfilled_fill`/`unfulfilled_fill` partition
        `fills` by whether `identify_mesh_casts` found a matched source for
        that (dst, data) pair -- a fill with no source holding its datum is
        `unfulfilled_fill`, never silently treated as fulfilled. `hops` and
        `edge_pressure` both come from `_transfer_cost(mcs)`: if it returns
        an `EdgePressure` `p`, `edge_pressure=p` and
        `hops=_const_pwq(p.total())`; otherwise `edge_pressure=None` and
        `hops` is the returned polynomial directly (the topology has no
        per-link decomposition to report).
        """
        # `identify_mesh_casts` is called exactly once; every output below is
        # derived from this single `mcs`, so they cannot fall out of sync.
        mcs: isl.Map = identify_mesh_casts(occs.map_, fills.map_, self.dist_fn)
        cost: isl.PwQPolynomial | EdgePressure = self._transfer_cost(mcs)
        edge_pressure: Optional[EdgePressure]
        hops: isl.PwQPolynomial
        if isinstance(cost, EdgePressure):
            edge_pressure = cost
            hops = _const_pwq(cost.total())
        else:
            edge_pressure = None
            hops = cost
        # { dst -> data } fills actually covered by a matched source.
        covered: isl.Map = _covered_fills(mcs)

        return TransferInfo(
            fulfilled_fill=Transfers(fills.tags, fills.map_.intersect(covered)),
            parent_reads=Reads(occs.tags, mcs),
            unfulfilled_fill=Fill(fills.tags, fills.map_.subtract(covered)),
            hops=hops,
            link_transfer=True,
            edge_pressure=edge_pressure,
        )

    @abstractmethod
    def _transfer_cost(self, mcs: isl.Map) -> isl.PwQPolynomial | EdgePressure:
        """
        Compute this topology's transfer cost from a fixed multicast-network map.

        Parameters
        ----------
        mcs:
            Multicast networks { [data] -> [dst -> src] } from
            `identify_mesh_casts`, computed once by `apply` and shared with
            every other output `apply` derives.

        Returns
        -------
        Either a bare `isl.PwQPolynomial` (a topology with no per-link
        decomposition -- `apply` sets `hops` to exactly this polynomial and
        leaves `edge_pressure` at `None`), or an `EdgePressure` (a topology
        with one -- `apply` sets `hops` to `_const_pwq(edge_pressure.total())`,
        structurally keeping the scalar and the per-edge view consistent).
        """
        raise NotImplementedError


class HypercubeMulticastModel(MulticastModel):
    """
    Does distributed multicasting a mesh using worst-case multicasting
    behavior by assuming all multicasts are broadcasting to the convex
    hypercube that encapsulates all their destinations and sources.

    `edge_pressure` is always `None` on the resulting `TransferInfo`: the
    hypercube model costs a convex bounding box, which has no notion of
    individual links to report pressure on.

    Preconditions
    -------------
    Because the cost kernel uses `calculate_extents_per_dim(mcns)`, `dist_fn`
    must hold all dimensions orthogonal to each other in a metric space,
    where each unit movement in a dimension counts as 1 hop, and must be
    translationally invariant (i.e., ∀src, dst, src', dst' ∈ space, if
    |src - dst| = |src' - dst'|, dist_fn(src, dst) = dist_fn(src', dst')).
    """

    def _transfer_cost(self, mcs: isl.Map) -> isl.PwQPolynomial:
        # TODO: Read once from all buffers, assert that
        # card(mcs) == tensor_size * duplication factor
        return self._cost_mesh_cast_hypercube(mcs)

    def _cost_mesh_cast_hypercube(self, mcns: isl.Map) -> isl.PwQPolynomial:
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


class FullyConnectedMulticastModel(MulticastModel):
    """
    Multicast cost model for a fully-connected fabric (e.g. an NVSwitch-style
    all-to-all interconnect).

    Every cross-node delivery costs exactly one hop regardless of distance, and
    a self-delivery (source node == destination node) costs zero:

        cost = | { (data, dst, src) in mcs : dist_fn(dst, src) >= 1 } |

    This is distance-independent in magnitude (one hop per crossing); `dist_fn`
    is used only to tell self-deliveries (0 hops) apart from fabric-crossing ones.

    `edge_pressure` is always `None` on the resulting `TransferInfo`: this
    model treats the fabric as a contention-free full mesh (one dedicated
    link per pair), so there is no per-link pressure to report.
    """

    def _transfer_cost(self, mcs: isl.Map) -> isl.PwQPolynomial:
        return self._cost_fully_connected(mcs)

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
        crossing: isl.Map = _fabric_crossing(mcns, self.dist_fn)
        return crossing.wrap().card()


# Directed-edge vocabulary + the four name-independent link-emitting maps for
# XY routing (parsed once at import). `_XY_EDGE_TYPES` is the single source
# of truth for the edge names: it feeds both the maps below and
# `XYRoutingMulticastModel.EDGE_TYPES`, so the class constant and the parsed
# maps can never name the edges differently.
_XY_EDGE_TYPES: tuple[str, str, str, str] = (
    "xedge_r",
    "xedge_l",
    "yedge_u",
    "yedge_d",
)
_XEDGE_R, _XEDGE_L, _YEDGE_U, _YEDGE_D = _XY_EDGE_TYPES

# { [[col[x] -> span[t]] -> ysv[ys]] -> yedge_u/d[x, t] : ... } -- re-key a
# column's spanned vertical links (see `_span_links`) onto the up/down edge
# tuples, splitting direction at the source row `ys`.
_YEDGE_U_MAP: isl.Map = isl.Map.read_from_str(
    isl.DEFAULT_CONTEXT,
    "{ [[col[x] -> span[t]] -> ysv[ys]] -> %s[x, t] : t >= ys }" % _YEDGE_U,
)
_YEDGE_D_MAP: isl.Map = isl.Map.read_from_str(
    isl.DEFAULT_CONTEXT,
    "{ [[col[x] -> span[t]] -> ysv[ys]] -> %s[x, t] : t < ys }" % _YEDGE_D,
)
# { [span[t] -> xy[xs, ys]] -> xedge_r/l[t, ys] : ... } -- re-key the source
# row's spanned horizontal links onto the right/left edge tuples, splitting
# direction at the source column `xs`.
_XEDGE_R_MAP: isl.Map = isl.Map.read_from_str(
    isl.DEFAULT_CONTEXT,
    "{ [span[t] -> xy[xs, ys]] -> %s[t, ys] : t >= xs }" % _XEDGE_R,
)
_XEDGE_L_MAP: isl.Map = isl.Map.read_from_str(
    isl.DEFAULT_CONTEXT,
    "{ [span[t] -> xy[xs, ys]] -> %s[t, ys] : t < xs }" % _XEDGE_L,
)


def _span_links(coords: isl.Map) -> isl.Map:
    """
    Build the set of link indices spanned by a per-key coordinate map.

    Parameters
    ----------
    coords:
        { key -> name[c] }, one point per (key, coordinate) pair to span --
        e.g. every destination-plus-source y within one tree's column, or
        every destination-plus-source column along one tree's row. `name` is
        read directly off `coords`'s range tuple, so this works for any
        single-dimension range tuple.

    Returns
    -------
    { key -> span[t] : min(key) <= t < max(key) } -- the contiguous run of
    link indices between each key's minimum and maximum coordinate. Built as
    an explicit lexmin/lexmax intersection (a `card()`-friendly link set)
    rather than by summing a min/max polynomial, which trips a barvinok
    `summate` assertion at scale (e.g. the 8x8 case).
    """
    name: str = coords.get_tuple_name(isl.dim_type.out)
    lo: isl.Map = coords.lexmin().apply_range(
        isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ %s[lo] -> span[t] : t >= lo }" % name
        )
    )
    hi: isl.Map = coords.lexmax().apply_range(
        isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ %s[hi] -> span[t] : t < hi }" % name
        )
    )
    return lo.intersect(hi)


class XYRoutingMulticastModel(MulticastModel):
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
    `dist_fn` is expected to be Manhattan and translationally invariant, like the
    hypercube model (see `HypercubeMulticastModel`'s Preconditions), since nearest-
    source selection relies on the same distance shape.

    Its four directed link types are declared once as the class constant
    `EDGE_TYPES = ("xedge_r", "xedge_l", "yedge_u", "yedge_d")`.

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

    EDGE_TYPES = _XY_EDGE_TYPES

    def _transfer_cost(self, mcs: isl.Map) -> EdgePressure:
        return _edge_pressure_from_links(self._directed_mesh_links(mcs))

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
        `[xedge_r, xedge_l, yedge_u, yedge_d]` maps (see `EDGE_TYPES`). An
        edge `xedge_r[t, ys]` is the rightward link in row `ys` between
        columns `t` and `t + 1`; `yedge_u[x, t]` is the upward link in
        column `x` between rows `t` and `t + 1` (and `_l` / `_d` the
        opposite directions).

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
        per_src, keymap = _per_src(mcns)  # keymap: { [data->src] -> src }

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
        # Every link spanned in a column, keyed by the column (see
        # `_span_links`): { [data->src->col] -> span[t] }.
        ylinks: isl.Map = _span_links(col_ys)
        # Re-key links by the tree and carry ys so direction splits at the source:
        # { [data->src] -> [[col[x] -> span[t]] -> ysv[ys]] }.
        y_with_ys: isl.Map = ylinks.curry().range_product(
            keymap.apply_range(
                isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> ysv[ys] }" % name)
            )
        )
        yedge_u: isl.Map = y_with_ys.apply_range(_YEDGE_U_MAP)
        yedge_d: isl.Map = y_with_ys.apply_range(_YEDGE_D_MAP)

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
        # Every link spanned along the row, keyed by the tree (see
        # `_span_links`): { [data->src] -> span[t] }.
        xlinks: isl.Map = _span_links(col_x)
        # Carry the source (xs, ys) so direction splits at xs and the row ys is
        # part of the edge identity: { [data->src] -> [span[t] -> xy[xs, ys]] }.
        x_with_src: isl.Map = xlinks.range_product(
            keymap.apply_range(
                isl.Map.read_from_str(ctx, "{ %s[xs, ys] -> xy[xs, ys] }" % name)
            )
        )
        xedge_r: isl.Map = x_with_src.apply_range(_XEDGE_R_MAP)
        xedge_l: isl.Map = x_with_src.apply_range(_XEDGE_L_MAP)

        return [xedge_r, xedge_l, yedge_u, yedge_d]


# Spoke vocabulary for the star model, declared once and shared with
# `StarMulticastModel.EDGE_TYPES`.
_STAR_EDGE_TYPES: tuple[str, str] = ("spoke_in", "spoke_out")


class StarMulticastModel(MulticastModel):
    """
    Multicast cost model for a star / central-switch fabric -- the spokes
    realization of a fully-connected interconnect (e.g. an NVSwitch, where every
    GPU connects to a shared switch rather than to a full mesh of peers).

    Every node has exactly one spoke (its bidirectional link to the switch). A
    delivery routes `src -> switch -> dst`: the source injects each datum once
    up its egress spoke (multicast fan-out happens at the switch, so one copy per
    datum regardless of how many destinations want it), and every destination
    receives its datum down its ingress spoke. Self-deliveries (a node already
    holding the datum) never cross the fabric and so load no spoke; `dist_fn`'s
    hop magnitude otherwise never enters the spoke load -- on a star every
    crossing is one switch hop each way, so `dist_fn` is used only to pick each
    destination's nearest source and to classify self- vs. fabric-crossing
    deliveries (`_fabric_crossing`).

    Its two spoke directions are declared once as the class constant
    `EDGE_TYPES = ("spoke_in", "spoke_out")`.
    """

    EDGE_TYPES = _STAR_EDGE_TYPES

    def _transfer_cost(self, mcs: isl.Map) -> EdgePressure:
        return EdgePressure(_union_pwqs([*self._spoke_loads(mcs)]))

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
        crossing: isl.Map = _fabric_crossing(mcns, self.dist_fn)

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
        spoke_in, spoke_out = self.EDGE_TYPES
        ingress: isl.PwQPolynomial = ingress_nodes.apply_domain(
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ %s[%s] -> %s[%s] }" % (name, idx, spoke_in, idx),
            )
        ).card()
        egress: isl.PwQPolynomial = egress_nodes.apply_domain(
            isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ %s[%s] -> %s[%s] }" % (name, idx, spoke_out, idx),
            )
        ).card()
        return ingress, egress


# Directed ring-link vocabulary, declared once and shared with
# `RingMulticastModel.EDGE_TYPES`.
_RING_EDGE_TYPES: tuple[str, str] = ("ring_cw", "ring_ccw")


class RingMulticastModel(MulticastModel):
    """
    Multicast cost model for a bidirectional ring of `n` nodes with
    shortest-path routing.

    Node `g` has two outgoing links: clockwise `ring_cw[g]` (`g -> g+1 mod n`)
    and counter-clockwise `ring_ccw[g]` (`g -> g-1 mod n`). A destination at
    clockwise distance `d = (dst - src) mod n` is reached clockwise when
    `d <= n // 2` and counter-clockwise otherwise, so the antipodal tie of an
    even ring (`d = n / 2`) goes clockwise. A multicast tree `(data, src)`
    therefore covers two arcs that start at the source: clockwise out to its
    farthest clockwise destination and counter-clockwise out to its farthest
    counter-clockwise one. Each link on an arc is crossed once per tree,
    however many destinations lie beyond it, and a self-delivery crosses no
    link.

    The per-link load is reported as an `EdgePressure` over
    `EDGE_TYPES = ("ring_cw", "ring_ccw")`, so `hops` is the total number of
    links the trees cross. `ring_dist_fn` builds the matching shortest-path
    `dist_fn` for `identify_mesh_casts`' nearest-source matching.

    Preconditions
    -------------
    The node tuple is 1-D with coordinates in `[0, n)`; `apply` raises
    `ValueError` for any other dimensionality. As with the other distributed
    models, the cost is a parameter-free constant.
    """

    EDGE_TYPES = _RING_EDGE_TYPES

    def __init__(self, dist_fn: isl.Map, n: int):
        """
        Parameters
        ----------
        dist_fn:
            Distance function { [dst -> src] -> hops[h] }, as for
            `MulticastModel`; normally `RingMulticastModel.ring_dist_fn(n)`.
        n:
            Number of nodes on the ring. Routing depends on it (it is the
            wrap-around), and `mcs` only carries the nodes that take part in
            a transfer, so it cannot be read off the transfer.
        """
        super().__init__(dist_fn)
        if n < 2:
            raise ValueError(f"RingMulticastModel needs n >= 2, got {n}")
        self.n = n

    @staticmethod
    def ring_dist_fn(n: int, name: str = "noc") -> isl.Map:
        """
        Shortest-path hop count on an `n`-node bidirectional ring:
        `min((gd - gs) mod n, (gs - gd) mod n)`.

        Parameters
        ----------
        n:
            Number of nodes on the ring.
        name:
            The node tuple's name; must match `fills`/`occs`.

        Returns
        -------
        { [name[gd] -> name[gs]] -> hops[h] } over `0 <= gd, gs < n`.
        """
        return isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ [%(m)s[gd] -> %(m)s[gs]] -> hops[(gd - gs) mod %(n)d] : "
            "0 <= gd < %(n)d and 0 <= gs < %(n)d and 2 * ((gd - gs) mod %(n)d) <= %(n)d; "
            "[%(m)s[gd] -> %(m)s[gs]] -> hops[%(n)d - ((gd - gs) mod %(n)d)] : "
            "0 <= gd < %(n)d and 0 <= gs < %(n)d and 2 * ((gd - gs) mod %(n)d) > %(n)d }"
            % dict(m=name, n=n),
        )

    def _transfer_cost(self, mcs: isl.Map) -> EdgePressure:
        return _edge_pressure_from_links(self._directed_ring_links(mcs))

    def _directed_ring_links(self, mcns: isl.Map) -> list[isl.Map]:
        """
        Decompose every multicast tree in `mcns` onto the directed ring links
        it crosses.

        Parameters
        ----------
        mcns:
            Multicast networks { [data] -> [dst -> src] } from `identify_mesh_casts`.

        Returns
        -------
        `[ring_cw, ring_ccw]` maps, each { [data -> src] -> edge[e] }:
        `ring_cw[e]` is the link `e -> e+1 mod n` and `ring_ccw[e]` the link
        `e -> e-1 mod n`.
        """
        name, dims = _mesh_node_tuple(mcns)
        if dims != 1:
            raise ValueError(
                "RingMulticastModel requires a 1-D node tuple; got tuple "
                f"'{name}' with {dims} dimensions."
            )
        per_src, keymap = _per_src(mcns)  # { [data->src] -> dst }, { [data->src] -> src }
        # { [data -> src] -> [dst -> src] }: each delivery next to its tree's source.
        deliveries: isl.Map = per_src.range_product(keymap)
        # A delivery at clockwise distance d (1 <= d <= n/2) crosses the clockwise
        # links src+k for 0 <= k < d; one at d > n/2 crosses the n-d
        # counter-clockwise links src-k. The union over a tree's deliveries keeps
        # each link once, which is the arc out to the farthest destination.
        # d = 0 (self-delivery) matches neither.
        cw = isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ [%(m)s[t] -> %(m)s[s]] -> %(e)s[l] : 2 * ((t - s) mod %(n)d) <= %(n)d and "
            "exists (k : 0 <= k < (t - s) mod %(n)d and l = (s + k) mod %(n)d) }"
            % dict(m=name, e=self.EDGE_TYPES[0], n=self.n),
        )
        ccw = isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ [%(m)s[t] -> %(m)s[s]] -> %(e)s[l] : 2 * ((t - s) mod %(n)d) > %(n)d and "
            "exists (k : 0 <= k < %(n)d - ((t - s) mod %(n)d) and l = (s - k) mod %(n)d) }"
            % dict(m=name, e=self.EDGE_TYPES[1], n=self.n),
        )
        return [deliveries.apply_range(cw), deliveries.apply_range(ccw)]


class _FwdStar(StarMulticastModel):
    """
    `StarMulticastModel` with the spoke vocabulary renamed for the forward
    lanes of the donut. Only `EDGE_TYPES` differs: `_spoke_loads` builds its
    spoke tuples from `EDGE_TYPES` and one index per node dimension, so a 3-D
    projected hub space yields lane edges `bip_fwd[l, g, h]` with no change to
    the primitive.
    """

    EDGE_TYPES = ("bip_fwd", "hub_fwd")


class _BwdStar(StarMulticastModel):
    """
    `StarMulticastModel` with the spoke vocabulary renamed for the backward
    lanes of the donut (`bip_bwd[l, g, h]`); see `_FwdStar`.
    """

    EDGE_TYPES = ("bip_bwd", "hub_bwd")


# Node-tuple name of the per-hub star spaces `DonutMulticastModel` projects
# deliveries into. Internal only: it never appears in an edge name.
_HUB_TUPLE: str = "hub"


class DonutMulticastModel(MulticastModel):
    """
    Multicast cost model for the donut: `k` rings of `n` GPUs, joined by
    complete bipartite lanes between consecutive rings.

    GPU `(l, g)` is position `g` on ring `l`. Inside a ring, GPU `(l, g)` has
    a clockwise link to `(l, g+1 mod n)` and a counter-clockwise link to
    `(l, g-1 mod n)`. Between ring `l` and ring `l+1 mod k` every GPU of one
    ring has its own lane to every GPU of the other, a forward lane
    `(l, g) -> (l+1, h)` and a backward lane `(l, g) -> (l-1, h)` per pair.
    The node tuple is 2-D, `noc[l, g]`.

    The model is a composition of two existing models used as primitives, not a
    new routing kernel. One `RingMulticastModel` stands for the ring links of
    every ring: the deliveries inside ring `l` are projected onto the 1-D ring
    tuple with `l` carried in the tree key, the primitive's link decomposition
    is called on them, and the edge coordinates are re-embedded as
    `ring_cw[l, e]` and `ring_ccw[l, e]`. `StarMulticastModel` instances stand
    for the lanes: GPU `(l, g)` is the hub, the `n` GPUs of the next ring are
    its leaves, and the lane `(l, g) -> (l+1, h)` is that star's ingress spoke
    `h`. The backward lanes are a second star per GPU on the previous ring.
    The composite only projects deliveries into the primitives' node spaces,
    calls them and re-embeds the edge coordinates.

    Routing policy:

    - A delivery inside one ring never leaves it: it takes the shortest arc
      and the antipode goes clockwise (the `RingMulticastModel` rule).
    - A delivery to an adjacent ring takes one direct lane (forward if the
      destination ring is `l+1 mod k`, backward if it is `l-1 mod k`).
    - A delivery between non-adjacent rings is not supported:
      `_transfer_cost` raises `ValueError`. `donut_dist_fn` still prices it, so
      source selection works for any pair.

    The `EdgePressure` has `EDGE_TYPES = ("ring_cw", "ring_ccw", "bip_fwd",
    "bip_bwd")` with these coordinates: `ring_cw[l, e]` is the link
    `(l, e) -> (l, e+1 mod n)`, `ring_ccw[l, e]` is `(l, e) -> (l, e-1 mod n)`,
    `bip_fwd[l, g, h]` is the lane `(l, g) -> (l+1 mod k, h)` and
    `bip_bwd[l, g, h]` is the lane `(l, g) -> (l-1 mod k, h)`. `hops` is the
    total number of physical link crossings, `edge_pressure.total()`.

    The egress spoke of each hub star is dropped. In the star primitive the
    egress spoke is the source's shared uplink to the switch, but here every
    lane is a separate physical link and a GPU has no shared egress port, so
    keeping it would count each lane crossing twice.

    Preconditions
    -------------
    The node tuple is 2-D with `0 <= l < k` and `0 <= g < n`; `_transfer_cost`
    raises `ValueError` for any other dimensionality. `n` and `k` are
    constructor arguments because `mcs` only carries the nodes that take part
    in a transfer, so the wrap-arounds cannot be read off the transfer.
    Parameter-free constant regime only, like the other edge-pressure models.
    """

    EDGE_TYPES = (*_RING_EDGE_TYPES, "bip_fwd", "bip_bwd")

    # Design: one switch instead of a duplicated composite. The directed variant
    # (`DirectedDonutMulticastModel`) flips this flag and narrows `EDGE_TYPES`;
    # `__init__` and `_transfer_cost` read the flag, everything else is shared.
    _BACKWARD_LANES: bool = True

    def __init__(self, dist_fn: isl.Map, n: int, k: int):
        """
        Parameters
        ----------
        dist_fn:
            Distance function { [dst -> src] -> hops[h] } as for
            `MulticastModel`, normally `DonutMulticastModel.donut_dist_fn(n, k)`.
            Only used to select each destination's nearest source.
        n:
            GPUs per ring (`R`). Must be >= 2.
        k:
            Number of rings (`K`). Must be >= 3: at `k = 2` the forward and
            backward lane families coincide.

        Raises
        ------
        ValueError
            If `k < 3` or `n < 2`.
        """
        super().__init__(dist_fn)
        if k < 3:
            raise ValueError(f"DonutMulticastModel needs k >= 3, got {k}")
        if n < 2:
            raise ValueError(f"DonutMulticastModel needs n >= 2, got {n}")
        self.n = n
        self.k = k
        # Design: the hub stars' dist_fn is a star distance over the projected
        # tuple hub[l, g, x] (x = leaf index, n = the hub sentinel). It is only
        # read by `_fabric_crossing`, which needs "every leaf -> hub pair is a
        # crossing"; the hop magnitude never enters a spoke load.
        hub_dist = isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ [%(h)s[l, g, xd] -> %(h)s[l, g, xs]] -> hops[1] : "
            "0 <= l < %(k)d and 0 <= g < %(n)d and 0 <= xd <= %(n)d and "
            "0 <= xs <= %(n)d and xd != xs; "
            "[%(h)s[l, g, xd] -> %(h)s[l, g, xs]] -> hops[0] : "
            "0 <= l < %(k)d and 0 <= g < %(n)d and 0 <= xd <= %(n)d and "
            "xd = xs }" % dict(h=_HUB_TUPLE, n=n, k=k),
        )
        self.fwd = _FwdStar(hub_dist)
        # The backward star exists only on the bidirectional donut.
        self.bwd = _BwdStar(hub_dist) if self._BACKWARD_LANES else None

    @staticmethod
    def donut_dist_fn(
        n: int, k: int, name: str = "noc", directed: bool = False
    ) -> isl.Map:
        """
        Hop count between two GPUs of the donut under the routing policy.

        Inside one ring it is the ring distance (`RingMulticastModel` rule).
        Between different rings it is the ring-index distance along the
        shorter direction, `min((ld - ls) mod k, (ls - ld) mod k)`, i.e. the
        lane hops along the ring of rings; it is 1 for adjacent rings. With
        `directed=True` there are no backward lanes, so a cross-ring pair costs
        `(ld - ls) mod k` forward lane hops (the wrap-around is the only way
        round); same-ring pairs are unchanged.

        Parameters
        ----------
        n:
            GPUs per ring.
        k:
            Number of rings.
        name:
            The node tuple's name; must match `fills`/`occs`.
        directed:
            Price cross-ring pairs for forward-only lanes. Default `False`.

        Returns
        -------
        { [name[ld, gd] -> name[ls, gs]] -> hops[h] } over every pair of the
        `k * n` nodes; single-valued.
        """
        if directed:
            # Same-ring pieces as below; one forward-distance piece replaces
            # the two shorter-direction pieces.
            return isl.Map.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[(gd - gs) mod %(n)d] : "
                "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
                "0 <= gs < %(n)d and ld = ls and 2 * ((gd - gs) mod %(n)d) <= %(n)d; "
                "[%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[%(n)d - ((gd - gs) mod %(n)d)] : "
                "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
                "0 <= gs < %(n)d and ld = ls and 2 * ((gd - gs) mod %(n)d) > %(n)d; "
                "[%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[(ld - ls) mod %(k)d] : "
                "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
                "0 <= gs < %(n)d and ld != ls }" % dict(m=name, n=n, k=k),
            )
        return isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[(gd - gs) mod %(n)d] : "
            "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
            "0 <= gs < %(n)d and ld = ls and 2 * ((gd - gs) mod %(n)d) <= %(n)d; "
            "[%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[%(n)d - ((gd - gs) mod %(n)d)] : "
            "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
            "0 <= gs < %(n)d and ld = ls and 2 * ((gd - gs) mod %(n)d) > %(n)d; "
            "[%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[(ld - ls) mod %(k)d] : "
            "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
            "0 <= gs < %(n)d and ld != ls and 2 * ((ld - ls) mod %(k)d) <= %(k)d; "
            "[%(m)s[ld, gd] -> %(m)s[ls, gs]] -> hops[%(k)d - ((ld - ls) mod %(k)d)] : "
            "0 <= ld < %(k)d and 0 <= ls < %(k)d and 0 <= gd < %(n)d and "
            "0 <= gs < %(n)d and ld != ls and 2 * ((ld - ls) mod %(k)d) > %(k)d }"
            % dict(m=name, n=n, k=k),
        )

    def _transfer_cost(self, mcs: isl.Map) -> EdgePressure:
        """
        Project the deliveries into the ring and lane primitives and union
        their link loads.

        Parameters
        ----------
        mcs:
            Multicast networks { [data] -> [noc[ld, gd] -> noc[ls, gs]] }
            from `identify_mesh_casts`.

        Returns
        -------
        `EdgePressure` over `EDGE_TYPES`.

        Raises
        ------
        ValueError
            If the node tuple is not 2-D, or if a delivery goes between
            non-adjacent rings.
        """
        ctx = isl.DEFAULT_CONTEXT
        name, dims = _mesh_node_tuple(mcs)
        if dims != 2:
            raise ValueError(
                "DonutMulticastModel requires a 2-D node tuple noc[l, g]; got "
                f"tuple '{name}' with {dims} dimensions."
            )
        n, k = self.n, self.k
        subs = dict(m=name, h=_HUB_TUPLE, n=n, k=k)

        # Phase 1: unsupported deliveries. Every supported delivery is in the
        # same ring or in an adjacent one (the two families are disjoint at
        # k >= 3); anything else would need multi-hop lane routing.
        backward_ok = "or (ls - ld - 1) mod %(k)d = 0 " % subs if self._BACKWARD_LANES else ""
        supported = isl.Set.read_from_str(
            ctx,
            "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] : ld = ls or "
            "(ld - ls - 1) mod %(k)d = 0 " % subs + backward_ok + "}",
        )
        unsupported = mcs.subtract(mcs.intersect_range(supported))
        if not unsupported.is_empty():
            raise ValueError(
                f"{type(self).__name__}: deliveries between non-adjacent rings"
                + ("" if self._BACKWARD_LANES else " (or backward, ring l to l-1)")
                + f" are not supported: {unsupported}"
            )

        # Phase 2: same-ring family through the ring primitive. The ring index
        # rides in the tree key, so the primitive's 1-D tuple check passes.
        to_ring = isl.Map.read_from_str(
            ctx,
            "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] -> [ring[l] -> [%(m)s[gd] -> %(m)s[gs]]] "
            ": l = ld and ld = ls }" % subs,
        )
        # { [data -> ring[l]] -> [M[gd] -> M[gs]] }
        ring_mcs = mcs.apply_range(to_ring).uncurry()
        # Design: the primitive is built here (not in __init__) because its
        # tuple name comes from `mcs`; its dist_fn is unused by the direct
        # link decomposition call below.
        ring = RingMulticastModel(RingMulticastModel.ring_dist_fn(n, name), n)
        cw_edge, ccw_edge = RingMulticastModel.EDGE_TYPES
        ring_links: list[isl.Map] = []
        for links, edge in zip(ring._directed_ring_links(ring_mcs), (cw_edge, ccw_edge)):
            # { [[data -> ring[l]] -> M[gs]] -> edge[e] } -> { key -> edge[l, e] }
            key_to_ring: isl.Map = (
                links.domain().unwrap().domain_map().range_factor_range()
            )
            embed = isl.Map.read_from_str(
                ctx, "{ [%(e)s[e] -> ring[l]] -> %(e)s[l, e] }" % dict(e=edge)
            )
            ring_links.append(links.range_product(key_to_ring).apply_range(embed))
        ring_load = _edge_pressure_from_links(ring_links).load

        # Phase 3: lane families through the star primitives. Leaf x = h is the
        # destination position, the hub sentinel x = n the source GPU; only
        # the ingress spokes are kept (see the class docstring).
        fwd_proj = isl.Map.read_from_str(
            ctx,
            "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] -> [%(h)s[ls, gs, gd] -> %(h)s[ls, gs, %(n)d]] "
            ": (ld - ls - 1) mod %(k)d = 0 }" % subs,
        )
        fwd_ingress, _ = self.fwd._spoke_loads(mcs.apply_range(fwd_proj))
        lane_loads = [fwd_ingress]
        if self._BACKWARD_LANES:
            bwd_proj = isl.Map.read_from_str(
                ctx,
                "{ [%(m)s[ld, gd] -> %(m)s[ls, gs]] -> [%(h)s[ls, gs, gd] -> %(h)s[ls, gs, %(n)d]] "
                ": (ls - ld - 1) mod %(k)d = 0 }" % subs,
            )
            bwd_ingress, _ = self.bwd._spoke_loads(mcs.apply_range(bwd_proj))
            lane_loads.append(bwd_ingress)

        # Phase 4: union. `ring_load` is already a union; the lane loads join it.
        lanes = _union_pwqs(lane_loads)
        return EdgePressure(ring_load.add(lanes))


class DirectedDonutMulticastModel(DonutMulticastModel):
    """
    Forward-only donut: ring `l`'s GPUs send to ring `l+1 mod k` and there are
    no backward lanes. Each GPU has two ring links (cw, ccw) and `n` forward
    lanes.

    Same composition as `DonutMulticastModel` (it shares `__init__` and
    `_transfer_cost`; only the backward star is left out). The `EdgePressure`
    has `EDGE_TYPES = ("ring_cw", "ring_ccw", "bip_fwd")`, with no `bip_bwd`
    family at all, so `eval_edge("bip_bwd", ...)` is an unknown edge name.

    A delivery from ring `l` to ring `l-1 mod k`, or between non-adjacent
    rings, is not supported and raises `ValueError`. Use
    `DonutMulticastModel.donut_dist_fn(n, k, name, directed=True)` as `dist_fn`
    so source selection prices a cross-ring delivery as `(ld - ls) mod k`
    forward hops.

    Design: a subclass with its own `EDGE_TYPES` (rather than a constructor
    flag) keeps the edge vocabulary correct per class, which is what
    `EdgePressure` and `eval_edge` read.
    """

    EDGE_TYPES = (*_RING_EDGE_TYPES, "bip_fwd")
    _BACKWARD_LANES = False
