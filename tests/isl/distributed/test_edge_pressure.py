"""
Tests for per-edge memory pressure (link load) of the distributed transfer models.

Where a model's ``hops`` is a single scalar, ``EdgePressure`` breaks the load out
per *physical* directed edge: how many multicast trees cross each link. Two models
are covered:

- ``XYRoutingMulticastModel``'s ``edge_pressure`` (now reached via
  ``model.apply(0, fill, occ).edge_pressure``, not a standalone method -- see
  the API-change note on ``test_load_sums_to_hops`` below) -- directed mesh
  links (``xedge_r``/``xedge_l``/``yedge_u``/``yedge_d``). ``test_load_sums_to_hops``
  checks the per-edge loads sum back to the already-validated A-F hop totals
  (4/6/6/3/4/448). Bottlenecks and a couple of individual edge loads
  (hand-derived, geometry documented inline) pin the shape.

- ``StarMulticastModel``'s ``edge_pressure`` (same access-pattern change) --
  the spokes realization of a fully-connected fabric (``spoke_in[n]`` ingress,
  ``spoke_out[n]`` egress). For an N-way all-to-all each node receives N-1 and
  sources 1, so the ingress spokes are hottest at N-1. Tied to
  ``FullyConnectedMulticastModel`` by
  ``sum over nodes of ingress == FullyConnected crossing count``.

See accelforge/model/_looptree/reuse/isl/distributed/README.md.
"""

import unittest
from pathlib import Path

import islpy as isl

from accelforge.model._looptree.reuse.isl.mapping_to_isl.types import (
    Fill,
    Occupancy,
    SpatialTag,
)
from accelforge.model._looptree.reuse.isl.distributed.distributed_buffers import (
    XYRoutingMulticastModel,
    StarMulticastModel,
    FullyConnectedMulticastModel,
    # Design: this used to be a local copy (identical signature/behavior) kept
    # in sync by hand; now that the source module exposes one module-level,
    # `int`-returning `_eval_const` (D6), the test suite imports it directly
    # instead of re-deriving the "evaluate a parameter-free PwQPolynomial at
    # the space's zero point" idiom a second time.
    _eval_const,
)

# `construct_spacetime` and `load_solutions` are hoisted into `helpers.py`
# (single-sourced there, see that module's docstring for the run-from-repo-root
# import requirement) rather than redefined/re-imported locally here.
from .helpers import construct_spacetime, load_solutions


# 1-D Manhattan distance over a line of GPUs, for the star/all-to-all cases.
_MANHATTAN_1D: str = (
    "{ [noc[gd] -> noc[gs]] -> hops[gd - gs] : gd >= gs;"
    "  [noc[gd] -> noc[gs]] -> hops[gs - gd] : gd <  gs }"
)


class TestXYRoutingEdgePressure(unittest.TestCase):
    """Per-mesh-edge pressure for XY routing, reusing the XY hop test geometries."""

    TEST_CASES_FILE: str = Path(__file__).parent / "xy_routing" / "test_cases.yaml"
    testcases: dict = load_solutions(TEST_CASES_FILE)

    def test_load_sums_to_hops(self):
        """
        Summed per-edge load matches the trusted A-F hop totals.

        # NOTE (post-D5 API/semantics change): before D5, `XYRoutingMulticastModel`
        # computed `hops` via a standalone `_cost_xy` and `edge_pressure` via a
        # *second*, independent traversal of `mcns`; this test's "sum(load) ==
        # hops" check was therefore a genuine cross-implementation invariant --
        # a bug in either path's edge decomposition would surface here even if
        # each path's own totals looked locally correct. D5 replaced that with
        # one aggregation path: `apply()` now computes `hops = _const_pwq(
        # pressure.total())` *from* the same `EdgePressure`, so `pressure.total()
        # == info.hops` is true by construction, not by cross-checked
        # computation -- the invariant, as originally framed, is now
        # tautological. What is *not* tautological is comparing `pressure.total()`
        # against the yaml's hand-derived `xy_routing_hops` oracle (an
        # independent, human-computed number, unrelated to how the code is
        # wired) -- that is what this test still checks, so it keeps its
        # regression value as a per-case oracle check even though it can no
        # longer catch a hops/edge_pressure disagreement (there structurally
        # isn't one to catch anymore).
        """
        for test in self.testcases:
            expected = test["expected"]["xy_routing_hops"]
            if expected is None:
                continue
            dim_tags = construct_spacetime(test["dims"])
            fill = Fill(dim_tags, test["fill"])
            occ = Occupancy(dim_tags, test["occ"])
            model = XYRoutingMulticastModel(test["dist_fn"])
            pressure = model.apply(0, fill, occ).edge_pressure
            assert pressure.total() == expected, (
                f"Σ edge load {pressure.total()} != hops {expected}"
            )

    def test_case_F_bottleneck_and_edges(self):
        """
        Case F (8x8, datum (d0,d1) at node (d0,d1), requested by all of column
        x=d0): every datum of column c floods all of column c's 7 vertical links.
        The upward link yedge_u[c,t] carries data with d1<=t (load t+1), the
        downward yedge_d[c,t] carries data with d1>t (load 7-t); the busiest
        directed link is 7 (the physical link total is 8).
        """
        f = next(t for t in self.testcases if t["expected"]["xy_routing_hops"] == 448)
        dim_tags = construct_spacetime(f["dims"])
        model = XYRoutingMulticastModel(f["dist_fn"])
        pressure = model.apply(
            0, Fill(dim_tags, f["fill"]), Occupancy(dim_tags, f["occ"])
        ).edge_pressure
        assert pressure.bottleneck() == 7
        assert pressure.eval_edge("yedge_u", [0, 6]) == 7  # top link, all 7 below
        assert pressure.eval_edge("yedge_d", [0, 1]) == 6  # links below row 1
        assert pressure.eval_edge("xedge_r", [0, 0]) == 0  # no horizontal traffic

    def test_monotone_overlap_bottleneck(self):
        """
        Regression: the directed load is non-constant (a flooded up-link carries
        ``t + 1`` trees), so the bottleneck must enumerate the edge domain, not
        sample one point per piece. Three sources stacked up column 0 (rows 2, 1,
        0) all casting to (0, 3): the top up-link (rows 2->3) carries all three, so
        the bottleneck is 3 even though lower links carry 1 and 2.
        """
        tags = [SpatialTag(0, 0), SpatialTag(1, 0)]
        manhattan = isl.Map.read_from_str(isl.DEFAULT_CONTEXT, (
            "{ [noc[xd,yd]->noc[xs,ys]]->hops[(xd-xs)+(yd-ys)] : xd>=xs and yd>=ys;"
            "  [noc[xd,yd]->noc[xs,ys]]->hops[-(xd-xs)+-(yd-ys)] : xd<xs and yd<ys;"
            "  [noc[xd,yd]->noc[xs,ys]]->hops[-(xd-xs)+(yd-ys)] : xd<xs and yd>=ys;"
            "  [noc[xd,yd]->noc[xs,ys]]->hops[(xd-xs)+-(yd-ys)] : xd>=xs and yd<ys }"))
        occ = Occupancy(tags, isl.Map.read_from_str(isl.DEFAULT_CONTEXT, (
            "{ noc[x,y]->data[c] : (c=0 and x=0 and y=2) or (c=1 and x=0 and y=1)"
            "                       or (c=2 and x=0 and y=0) }")))
        fill = Fill(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ noc[x,y]->data[c] : x=0 and y=3 and 0<=c<3 }"))
        pressure = XYRoutingMulticastModel(manhattan).apply(0, fill, occ).edge_pressure
        assert pressure.eval_edge("yedge_u", [0, 2]) == 3  # top link, all three
        assert pressure.bottleneck() == 3                  # NOT 1 or 2

    def test_single_tree_cases_have_unit_bottleneck(self):
        """
        Cases A-E are single multicast trees, so no link is shared by two trees and
        every edge carries load exactly 1.
        """
        for test in self.testcases:
            exp = test["expected"]["xy_routing_hops"]
            if exp is None or exp == 448:  # F has overlapping trees
                continue
            dim_tags = construct_spacetime(test["dims"])
            model = XYRoutingMulticastModel(test["dist_fn"])
            pressure = model.apply(
                0, Fill(dim_tags, test["fill"]), Occupancy(dim_tags, test["occ"])
            ).edge_pressure
            assert pressure.bottleneck() == 1


class TestStarSpokePressure(unittest.TestCase):
    """Per-spoke pressure for the star (FC spokes) model on N-way all-to-all."""

    @staticmethod
    def _all_to_all(n: int):
        """Build (fill, occ, dist_fn) for an N-GPU all-to-all on a line."""
        tags = [SpatialTag(0, 0)]
        occ = Occupancy(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            f"{{ noc[g] -> data[d] : 0 <= g < {n} and d = g }}"))
        fill = Fill(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT,
            f"{{ noc[g] -> data[d] : 0 <= g < {n} and 0 <= d < {n} and d != g }}"))
        dist_fn = isl.Map.read_from_str(isl.DEFAULT_CONTEXT, _MANHATTAN_1D)
        return fill, occ, dist_fn

    def test_all_to_all_spoke_loads(self):
        """
        N-way all-to-all: every node receives N-1 (ingress) and sources 1 (egress);
        the bottleneck spoke is the ingress at N-1.
        """
        for n in (4, 8):
            fill, occ, dist_fn = self._all_to_all(n)
            pressure = StarMulticastModel(dist_fn).apply(0, fill, occ).edge_pressure
            for node in range(n):
                assert pressure.eval_edge("spoke_in", [node]) == n - 1
                assert pressure.eval_edge("spoke_out", [node]) == 1
            assert pressure.bottleneck() == n - 1

    def test_ingress_sum_equals_fully_connected_count(self):
        """
        Cross-model invariant: Σ ingress over spokes == FullyConnected crossing
        count (every crossing delivery is exactly one node's ingress).
        """
        for n in (4, 8):
            fill, occ, dist_fn = self._all_to_all(n)
            pressure = StarMulticastModel(dist_fn).apply(0, fill, occ).edge_pressure
            total_ingress = sum(pressure.eval_edge("spoke_in", [g]) for g in range(n))
            fc_hops = _eval_const(
                FullyConnectedMulticastModel(dist_fn).apply(0, fill, occ).hops
            )
            assert total_ingress == fc_hops == n * (n - 1)

    def test_star_hops_is_injections_plus_deliveries(self):
        """Star scalar hops == Σ egress + Σ ingress == N + N(N-1)."""
        for n in (4, 8):
            fill, occ, dist_fn = self._all_to_all(n)
            hops = _eval_const(StarMulticastModel(dist_fn).apply(0, fill, occ).hops)
            assert hops == n + n * (n - 1)


if __name__ == "__main__":
    unittest.main()
