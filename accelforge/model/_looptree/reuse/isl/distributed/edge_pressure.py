"""
Per-edge memory pressure (link load) for distributed transfer models:
`EdgePressure` and the scalar-extraction / accumulation helpers that build and
consume it.

Leaf module -- depends only on `islpy`, `functools`, `dataclasses`, and
`typing`, so every other module in this package (`mesh_casts.py`,
`distributed_buffers.py`, and `../spatial.py`) can import from here without
risk of a cycle.
"""

import functools

from dataclasses import dataclass

from typing import Optional

import islpy as isl


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

    @functools.cached_property
    def _pieces(self) -> list[isl.PwQPolynomial]:
        """
        Cache `load`'s decomposition into per-edge-type pieces.

        Returns
        -------
        The list of `isl.PwQPolynomial` pieces `load` unions together, in
        `foreach_pw_qpolynomial` enumeration order.
        """
        # Note: `cached_property` works on this frozen dataclass only because
        # it is not `slots=True` -- the cache is written straight into the
        # instance `__dict__`, bypassing the frozen `__setattr__`.
        pieces: list[isl.PwQPolynomial] = []
        self.load.foreach_pw_qpolynomial(pieces.append)
        return pieces

    def total(self) -> int:
        """
        Sum the load over every edge.

        For a single-direction (or single edge-type) pressure this is the total
        traffic; for the full mesh pressure it equals the model's total `hops`
        (sum over edges of load == total hops), which is the primary cross-check
        that the per-edge decomposition is correct.
        """
        total = 0
        for pwq in self._pieces:
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
        for pwq in self._pieces:
            edges: list[isl.Point] = []
            pwq.domain().foreach_point(edges.append)
            for edge in edges:
                best = max(best, _eval_at(pwq, edge))
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
        for pwq in self._pieces:
            if pwq.domain().get_space().get_tuple_name(isl.dim_type.set) != name:
                continue
            point: isl.Point = isl.Set.read_from_str(
                isl.DEFAULT_CONTEXT,
                "{ %s[%s] }" % (name, ", ".join(str(c) for c in coords)),
            ).sample_point()
            return _eval_at(pwq, point)
        return 0


def _eval_at(pwq: isl.PwQPolynomial, point: isl.Point) -> int:
    """
    Evaluate a piecewise quasi-polynomial at one point and return a Python `int`.

    Parameters
    ----------
    pwq:
        The polynomial to evaluate.
    point:
        The point to evaluate at. Must lie in a space compatible with `pwq`'s
        domain (matching tuple name and dimensionality).

    Returns
    -------
    `pwq`'s value at `point`, as a Python `int`.

    Note: isl.Val has no direct int() conversion in this islpy build (it
    raises TypeError); round-tripping through str() is the working idiom
    used throughout this module.
    """
    return int(str(pwq.eval(point)))


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
    return _eval_at(pwq, isl.Point.zero(pwq.domain().get_space()))


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


def _union_pwqs(pwqs: list[isl.PwQPolynomial]) -> isl.UnionPwQPolynomial:
    """
    Accumulate a list of piecewise quasi-polynomials into one union.

    Parameters
    ----------
    pwqs:
        The polynomials to sum, e.g. one per directed edge type. May be
        empty.

    Returns
    -------
    The union-add of every `pwqs` entry (each promoted to an
    `isl.UnionPwQPolynomial` first). An empty `pwqs` yields the empty union
    `{ }`, so a caller with zero edge types (or a degenerate empty mesh) does
    not need to special-case the accumulation.
    """
    acc: Optional[isl.UnionPwQPolynomial] = None
    for pwq in pwqs:
        contribution = isl.UnionPwQPolynomial.from_pw_qpolynomial(pwq)
        acc = contribution if acc is None else acc.add(contribution)
    if acc is None:
        acc = isl.UnionPwQPolynomial.read_from_str(isl.DEFAULT_CONTEXT, "{ }")
    return acc


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
    per-type results into one `UnionPwQPolynomial` (via `_union_pwqs`).
    """
    # { edge -> #trees crossing it }, one per edge type.
    per_edge_loads: list[isl.PwQPolynomial] = [
        edge_map.reverse().card() for edge_map in edge_maps
    ]
    return EdgePressure(_union_pwqs(per_edge_loads))
