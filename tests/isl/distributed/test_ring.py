"""
Tests for the RingMulticastModel (bidirectional ring, shortest-path routing,
antipodal tie clockwise).

- ``TestRingMulticastModel.test_gamut``: total links crossed (``hops``) for the
  hand-derived cases in ``ring/test_cases.yaml`` (n = 8).
- ``TestRingEdgePressure``: per-link loads on the nvlink_probe flow sets that
  separate a ring from a star (incast, perm_shift, personalised all-to-all),
  an odd ring with no antipodal tie, and ``hops == Σ load``.
- ``ring_dist_fn`` against ``min(d, n - d)``, and the 1-D precondition.

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
    RingMulticastModel,
)
from accelforge.model._looptree.reuse.isl.distributed.edge_pressure import _eval_const

from .helpers import run_hops_gamut


def Ring8(dist_fn: isl.Map) -> RingMulticastModel:
    """`run_hops_gamut` builds models as `model_cls(dist_fn)`; the yaml cases are all n = 8."""
    return RingMulticastModel(dist_fn, 8)


def _flows(n: int, pairs: list[tuple[int, int]]):
    """(fill, occ) with one datum data[s, t] per flow s -> t, held at s and wanted at t."""
    cons = " or ".join(f"(s = {s} and t = {t})" for s, t in pairs)
    tags = [SpatialTag(0, 0)]
    occ = Occupancy(tags, isl.Map.read_from_str(
        isl.DEFAULT_CONTEXT, f"{{ noc[g] -> data[s, t] : 0 <= g < {n} and s = g and ({cons}) }}"))
    fill = Fill(tags, isl.Map.read_from_str(
        isl.DEFAULT_CONTEXT, f"{{ noc[g] -> data[s, t] : 0 <= g < {n} and t = g and ({cons}) }}"))
    return fill, occ


def _apply(n: int, pairs: list[tuple[int, int]]):
    fill, occ = _flows(n, pairs)
    return RingMulticastModel(RingMulticastModel.ring_dist_fn(n), n).apply(0, fill, occ)


def _loads(n: int, pressure) -> tuple[list[int], list[int]]:
    cw, ccw = RingMulticastModel.EDGE_TYPES
    return ([pressure.eval_edge(cw, [g]) for g in range(n)],
            [pressure.eval_edge(ccw, [g]) for g in range(n)])


class TestRingMulticastModel(unittest.TestCase):
    """Total links crossed for the hand-derived yaml cases."""

    TEST_CASES_FILE: Path = Path(__file__).parent / "ring" / "test_cases.yaml"

    def test_gamut(self):
        run_hops_gamut(Ring8, self.TEST_CASES_FILE, "ring_hops")


class TestRingEdgePressure(unittest.TestCase):
    """Per-link ring loads, one datum per flow (the nvlink_probe encoding)."""

    def test_incast_7(self):
        """
        1..7 -> 0: sources 1-3 go counter-clockwise (ccw links 3,2,1 carry 1,2,3
        flows), 4-7 go clockwise (4 is the antipodal tie; cw links 4..7 carry
        1..4). The bottleneck is 4, not the star's 7.
        """
        info = _apply(8, [(i, 0) for i in range(1, 8)])
        cw, ccw = _loads(8, info.edge_pressure)
        assert cw == [0, 0, 0, 0, 1, 2, 3, 4]
        assert ccw == [0, 3, 2, 1, 0, 0, 0, 0]
        assert info.edge_pressure.bottleneck() == 4

    def test_perm_shift(self):
        """i -> i+k: every link of the shorter direction carries min(k, 8-k) flows; k = 4 goes clockwise."""
        for k in range(1, 8):
            info = _apply(8, [(i, (i + k) % 8) for i in range(8)])
            cw, ccw = _loads(8, info.edge_pressure)
            load = min(k, 8 - k)
            assert (cw, ccw) == (([load] * 8, [0] * 8) if k <= 4 else ([0] * 8, [load] * 8)), k
            assert info.edge_pressure.bottleneck() == load

    def test_personalised_all_to_all(self):
        """Every clockwise link carries 1+2+3+4 = 10 flows, every counter-clockwise one 1+2+3 = 6."""
        info = _apply(8, [(s, t) for s in range(8) for t in range(8) if s != t])
        cw, ccw = _loads(8, info.edge_pressure)
        assert cw == [10] * 8 and ccw == [6] * 8
        assert info.edge_pressure.bottleneck() == 10
        assert _eval_const(info.hops) == info.edge_pressure.total() == 128

    def test_odd_ring_has_no_tie(self):
        """n = 5: d = 1, 2 go clockwise and d = 3, 4 counter-clockwise, so both directions carry 3."""
        info = _apply(5, [(s, t) for s in range(5) for t in range(5) if s != t])
        cw, ccw = _loads(5, info.edge_pressure)
        assert cw == [3] * 5 and ccw == [3] * 5
        assert _eval_const(info.hops) == 30

    def test_multicast_shares_the_arc(self):
        """One datum at 0 wanted at 1, 2, 3: the clockwise links 0, 1, 2 each carry the tree once."""
        tags = [SpatialTag(0, 0)]
        occ = Occupancy(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ noc[g] -> data[c] : c = 0 and g = 0 }"))
        fill = Fill(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ noc[g] -> data[c] : c = 0 and 1 <= g <= 3 }"))
        info = RingMulticastModel(RingMulticastModel.ring_dist_fn(8), 8).apply(0, fill, occ)
        cw, ccw = _loads(8, info.edge_pressure)
        assert cw == [1, 1, 1, 0, 0, 0, 0, 0] and ccw == [0] * 8


class TestRingContract(unittest.TestCase):
    def test_ring_dist_fn_is_shortest_path(self):
        for n in (5, 8):
            dist = RingMulticastModel.ring_dist_fn(n)
            for gd in range(n):
                for gs in range(n):
                    pair = isl.Set.read_from_str(
                        isl.DEFAULT_CONTEXT, f"{{ [noc[{gd}] -> noc[{gs}]] }}")
                    hops = dist.intersect_domain(pair).range()
                    expected = isl.Set.read_from_str(
                        isl.DEFAULT_CONTEXT,
                        f"{{ hops[{min((gd - gs) % n, (gs - gd) % n)}] }}")
                    assert hops.is_equal(expected), (n, gd, gs, hops)

    def test_rejects_2d_nodes(self):
        tags = [SpatialTag(0, 0), SpatialTag(1, 0)]
        occ = Occupancy(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ noc[x, y] -> data[c] : c = 0 and x = 0 and y = 0 }"))
        fill = Fill(tags, isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ noc[x, y] -> data[c] : c = 0 and x = 1 and y = 0 }"))
        dist = isl.Map.read_from_str(
            isl.DEFAULT_CONTEXT, "{ [noc[xd, yd] -> noc[xs, ys]] -> hops[1] }")
        with self.assertRaises(ValueError):
            RingMulticastModel(dist, 8).apply(0, fill, occ)

    def test_rejects_tiny_ring(self):
        with self.assertRaises(ValueError):
            RingMulticastModel(RingMulticastModel.ring_dist_fn(2), 1)


class TestRingRelayed(unittest.TestCase):
    def test_relayed_hops(self):
        # 8-ring: 0 -> {1, 2, 3} is one cw arc 0->1->2->3 (hops 1->2, 2->3 relayed);
        # 0 -> {6} is the ccw arc 0->7->6 (hop 7->6 relayed); 5 -> {6} is one hop.
        n = 8
        tags = [SpatialTag(0, 0)]
        casts = [(0, [1, 2, 3, 6]), (5, [6])]
        occ = " or ".join(f"(i = {i} and g = {s})" for i, (s, _) in enumerate(casts))
        fill = " or ".join(f"(i = {i} and g = {t})" for i, (_, ts) in enumerate(casts) for t in ts)
        model = RingMulticastModel(RingMulticastModel.ring_dist_fn(n), n)
        info = model.apply(
            0,
            Fill(tags, isl.Map.read_from_str(isl.DEFAULT_CONTEXT, f"{{ noc[g] -> data[i] : 0 <= g < {n} and ({fill}) }}")),
            Occupancy(tags, isl.Map.read_from_str(isl.DEFAULT_CONTEXT, f"{{ noc[g] -> data[i] : 0 <= g < {n} and ({occ}) }}")),
        )
        rel = model.relayed_pressure(info.parent_reads.map_)
        want = {("ring_cw", 1): 1, ("ring_cw", 2): 1, ("ring_ccw", 7): 1}
        for edge in ("ring_cw", "ring_ccw"):
            for e in range(n):
                self.assertEqual(int(rel.eval_edge(edge, [e])), want.get((edge, e), 0), (edge, e))
        self.assertEqual(int(info.edge_pressure.total()), 6)

if __name__ == "__main__":
    unittest.main()
